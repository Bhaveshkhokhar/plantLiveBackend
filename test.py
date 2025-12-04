import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

# ------------------------------------------
# 0. BASE CLASS USED DURING TRAINING
# ------------------------------------------
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# ------------------------------------------
# 1. ARCHITECTURE (ConvBlock + ResNet9)
# ------------------------------------------
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# ------------------------------------------
# 2. LOAD MODEL
# ------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant-disease-model-complete.pth")

model = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)
model.eval()


# ------------------------------------------
# 3. CLASSES
# ------------------------------------------
classes = [
    'Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Potato___healthy', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 
    'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 
    'Apple___Black_rot', 'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 
    'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 
    'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 
    'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 
    'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 
    'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy'
]


# ------------------------------------------
# 4. TRANSFORMS
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
# 5. PREDICT FUNCTIONS
# ------------------------------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]


def predict_from_bytes(image_bytes):
    from io import BytesIO
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]


# ------------------------------------------
# 6. COMMAND LINE USAGE
# ------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        exit()

    result = predict(sys.argv[1])
    print("Predicted:", result)
