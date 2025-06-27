# app/model_utils.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle  # You forgot to import this
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(device)
with open('Models/finalmodel1.pkl', 'rb') as f:
    model = pickle.load(f)
    

model.to(device)
model.eval()
    # Define the same transform used during training
transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

# Predict function)
def predict(image: Image.Image) -> dict:
    image_size = image.size
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs[0][1].item()
        prediction = torch.argmax(probs, dim=1).item()
        print(output)
        print(img_tensor)
    return {
        "vehicle_detected": prediction ==1,
        "confidence": round(confidence, 4),
        "image_size": image_size
    }
