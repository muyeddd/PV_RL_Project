import sys
import torch
from torchvision import transforms

project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from models.efficientnet_v2_s_with_i import SolarEfficientNetV2SWithI
from utils.dataset import SolarDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = SolarEfficientNetV2SWithI(dropout=0.3, use_pretrained=True).to(device)
print("Model device:", next(model.parameters()).device)

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

img_dir = r"E:\PV_RL_Project\data\raw\PanelImages"

dataset = SolarDataset(img_dir, transform=transform)

sample = dataset[0]

# 兼容 dataset 返回 3 项或 4 项
if len(sample) == 3:
    image, L, I = sample
else:
    image, L, I = sample[:3]

image = image.unsqueeze(0).to(device)
I = torch.tensor([I], dtype=torch.float32).to(device)

print("image shape:", image.shape)
print("I shape:", I.shape)

with torch.no_grad():
    pred = model(image, I)

print("pred shape:", pred.shape)
print("pred:", pred)
print("L:", L)