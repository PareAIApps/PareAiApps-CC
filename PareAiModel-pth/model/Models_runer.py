import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import logging

from config import MODEL_PATH, IMAGE_SIZE, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arsitektur CNN
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * (IMAGE_SIZE[0] // 8) * (IMAGE_SIZE[1] // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Load model
try:
    model = CNNModel(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Model failed to load: {e}")

def get_model():
    return model

def preprocess_image(file):
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor
