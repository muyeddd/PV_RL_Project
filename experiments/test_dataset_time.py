import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from utils.dataset import SolarDataset

img_dir = r"E:\PV_RL_Project\data\raw\PanelImages"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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

images, L, I, time_feat = next(iter(loader))

print("images shape:", images.shape)
print("L:", L)
print("I:", I)
print("time_feat:", time_feat)