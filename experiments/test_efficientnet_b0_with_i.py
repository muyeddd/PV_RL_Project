import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from utils.dataset import SolarDataset
from models.efficientnet_b0_with_i import SolarEfficientNetB0WithI

img_dir = r"E:\PV_RL_Project\data\raw\PanelImages"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = SolarDataset(
    img_dir=img_dir,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = SolarEfficientNetB0WithI().to(device)
print("Model device:", next(model.parameters()).device)

images, L, I, time_feat = next(iter(loader))

images = images.to(device)
I = I.to(device)

with torch.no_grad():
    preds = model(images, I)

print("images shape:", images.shape)
print("I shape:", I.shape)
print("preds shape:", preds.shape)
print("preds:", preds.squeeze())
print("L:", L)