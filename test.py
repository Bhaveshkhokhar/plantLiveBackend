import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# ------------------------------------------
# 1. LOAD SAVED MODEL
# ------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant-disease-model-complete.pth")

# IMPORTANT FIX FOR PYTORCH 2.6+
model = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False   # <-- FIX FOR YOUR ERROR
)
model.eval()

# ------------------------------------------
# 2. CLASS LABELS
# ------------------------------------------
classes = [
    'Tomato___Late_blight',
    'Tomato___healthy',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Potato___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Tomato___Early_blight',
    'Tomato___Septoria_leaf_spot',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Strawberry___Leaf_scorch',
    'Peach___healthy',
    'Apple___Apple_scab',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Bacterial_spot',
    'Apple___Black_rot',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Peach___Bacterial_spot',
    'Apple___Cedar_apple_rust',
    'Tomato___Target_Spot',
    'Pepper,_bell___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Potato___Late_blight',
    'Tomato___Tomato_mosaic_virus',
    'Strawberry___healthy',
    'Apple___healthy',
    'Grape___Black_rot',
    'Potato___Early_blight',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Common_rust_',
    'Grape___Esca_(Black_Measles)',
    'Raspberry___healthy',
    'Tomato___Leaf_Mold',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Pepper,_bell___Bacterial_spot',
    'Corn_(maize)___healthy'
]

# ------------------------------------------
# 3. TRANSFORM (USE SAME AS TRAINING)
# ------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------
# 4. PREDICTION FUNCTIONS
# ------------------------------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

def predict_from_bytes(image_bytes):
    """Predict from image bytes (for API use)"""
    from io import BytesIO
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

# ------------------------------------------
# 5. COMMAND LINE USAGE
# ------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        exit()

    image_path = sys.argv[1]
    print("\nPredicting...\n")
    result = predict(image_path)
    print("Predicted:", result)
