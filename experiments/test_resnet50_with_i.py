import sys
import os
import torch

project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from models.resnet50_with_i import SolarResNet50WithI
from utils.dataset import SolarDataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SolarResNet50WithI(
    dropout=0.3,
    use_pretrained=True
).to(device)

print("Model device:", next(model.parameters()).device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

img_dir = r"E:\PV_RL_Project\data\raw\PanelImages"

dataset = SolarDataset(
    img_dir,
    transform=transform
)

sample = dataset[0]

image = sample[0].unsqueeze(0).to(device)
L = sample[1]
I = torch.tensor([sample[2]], dtype=torch.float32).to(device)

print("image shape:", image.shape)
print("I shape:", I.shape)

with torch.no_grad():
    pred = model(image, I)

print("pred shape:", pred.shape)
print("pred:", pred)
print("L:", L)