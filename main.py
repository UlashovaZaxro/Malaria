import os
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import base64
# import numpy as np # Only needed if you do complex numpy operations
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# --- Application Setup ---
app = FastAPI(title="PyTorch Malaria Classifier")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- PyTorch Model Definition (MUST BE IDENTICAL TO YOUR TRAINED MODEL) ---
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # If BatchNorm was used here in training, UNCOMMENT it:
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # If BatchNorm was used here in training, UNCOMMENT it:
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            # Input features to nn.Linear depends on image size (128x128) after 3 MaxPool(2,2) layers:
            # 128 -> 64 -> 32 -> 16. So, 16x16.
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# --- Model Loading ---
MODEL_FILENAME = "malaria_model.pth" # Your PyTorch state_dict file
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

pytorch_model: CNNModel = None
model_loaded_successfully: bool = False
model_load_error_message: str = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
def load_model_on_startup():
    global pytorch_model, model_loaded_successfully, model_load_error_message
    
    pytorch_model = CNNModel().to(device) # Instantiate your model structure

    if not os.path.exists(MODEL_PATH):
        model_load_error_message = f"XATOLIK: Model fayli '{MODEL_PATH}' topilmadi."
        print(f"--- SERVER LOG: {model_load_error_message} ---")
        return

    try:
        pytorch_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        pytorch_model.eval()  # IMPORTANT: Set to evaluation mode
        model_loaded_successfully = True
        print(f"--- SERVER LOG: '{MODEL_FILENAME}' (PyTorch state_dict) modeli muvaffaqiyatli yuklandi '{device}' qurilmasiga. ---")
    except Exception as e:
        model_load_error_message = f"XATOLIK: PyTorch modelini '{MODEL_PATH}' yuklashda muammo yuz berdi: {e}"
        print(f"--- SERVER LOG: {model_load_error_message} ---")
        print(f"--- SERVER LOG: Iltimos, CNNModel klassi ta'rifi o'qitilgan model bilan bir xilligini va .pth fayli to'g'ri ekanligini tekshiring. ---")
        pytorch_model = None 

# --- Image Preprocessing for PyTorch ---
# !!! CRITICAL FOR ACCURACY !!!
# These transforms MUST EXACTLY match the non-augmentation transforms
# used during your PyTorch model's training/validation phase.

TARGET_IMAGE_SIZE = (128, 128) 

preprocess_transform_pytorch = transforms.Compose([
    transforms.Resize(TARGET_IMAGE_SIZE),
    transforms.ToTensor(),  # Converts PIL Image (H, W, C) [0-255] to PyTorch Tensor (C, H, W) [0.0-1.0]
    
    # >>> IMPORTANT NORMALIZATION STEP <<<
    # IF you used transforms.Normalize(mean, std) during training,
    # you MUST UNCOMMENT the line below and use the EXACT SAME mean and std values.
    # Otherwise, your model will get data in a different distribution than it expects,
    # leading to incorrect predictions.
    # Example (using ImageNet stats, replace with YOUR training values if different):
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image_pytorch(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = preprocess_transform_pytorch(image)
        image_tensor = image_tensor.unsqueeze(0) # Add batch dimension (1, C, H, W)
        return image_tensor
    except Exception as e:
        print(f"--- SERVER LOG: PyTorch rasmni qayta ishlashda xatolik: {e} ---")
        return None

# --- Class Names ---
CLASS_NAMES = ["Parazitlanmagan", "Parazitlangan"]

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "model_loaded": model_loaded_successfully,
        "model_load_error": model_load_error_message
    })

@app.post("/predict/", response_class=HTMLResponse)
async def predict_image_route(request: Request, file: UploadFile = File(...)):
    prediction_text = None
    error_message = None
    uploaded_image_data_url = None

    if not model_loaded_successfully or not pytorch_model:
        error_message = model_load_error_message or "Model yuklanmagan. Server loglarini tekshiring."
    elif not file.content_type.startswith("image/"):
        error_message = "Noto'g'ri fayl turi. Iltimos, rasm faylini yuklang."
    else:
        try:
            image_bytes = await file.read()
            img_str_base64 = base64.b64encode(image_bytes).decode("utf-8")
            uploaded_image_data_url = f"data:{file.content_type};base64,{img_str_base64}"

            processed_image_tensor = preprocess_image_pytorch(image_bytes)

            if processed_image_tensor is None:
                error_message = "Rasmni qayta ishlashda xatolik yuz berdi."
            else:
                processed_image_tensor = processed_image_tensor.to(device)
                
                with torch.no_grad(): # Disable gradient calculations
                    raw_prediction_tensor = pytorch_model(processed_image_tensor)
                
                # Assuming binary classification with a single sigmoid output from your PyTorch model
                probability_positive_class = raw_prediction_tensor.squeeze().item() 
                print(f"--- SERVER LOG: Model ehtimolligi (pozitiv klass uchun): {probability_positive_class:.4f} ---")

                threshold = 0.5 
                if probability_positive_class >= threshold:
                    predicted_class_name = CLASS_NAMES[1] # 'Parazitlangan'
                    confidence = probability_positive_class * 100
                else:
                    predicted_class_name = CLASS_NAMES[0] # 'Parazitlanmagan'
                    confidence = (1 - probability_positive_class) * 100
                prediction_text = f"Bashorat: {predicted_class_name} (Ishonch: {confidence:.2f}%)"
        except Exception as e_generic:
            error_message = "Bashorat qilishda kutilmagan xatolik."
            print(f"--- SERVER LOG: Generic exception during PyTorch prediction: {e_generic} ---")
            import traceback
            print(traceback.format_exc())

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction_text": prediction_text,
        "error": error_message,
        "image_data_url": uploaded_image_data_url,
        "model_loaded": model_loaded_successfully,
        "model_load_error": model_load_error_message
    })

if __name__ == "__main__":
    print("--- PyTorch Malaria Detection Server (FastAPI with Jinja2) ---")
    print("--- To run, execute in terminal: uvicorn main:app --reload --host 0.0.0.0 --port 8000 ---")
    # uvicorn.run(app, host="127.0.0.1", port=8000) # For direct run, but CLI is better for --reload