import torch
import torchvision.transforms as transforms
from PIL import Image

# ------------------------------------------
# 1. LOAD FULL MODEL (NO ARCHITECTURE NEEDED)
# ------------------------------------------
MODEL_PATH = "backend\plant-disease-model-complete.pth"   # your saved file
model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

# ------------------------------------------
# 2. CLASS NAMES (UPDATE THESE)
# ------------------------------------------
classes = [
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4"
    # Add all your classes in correct order
]

# ------------------------------------------
# 3. IMAGE TRANSFORM (USE SAME AS TRAINING)
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
# 4. PREDICTION FUNCTION
# ------------------------------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

# ------------------------------------------
# 5. RUN FROM TERMINAL
# ------------------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        exit()

    image_path = sys.argv[1]
    prediction = predict(image_path)
    print("Predicted:", prediction)
