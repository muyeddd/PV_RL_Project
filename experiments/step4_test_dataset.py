import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.dataset import SolarDataset

project_root = r"E:\PV_RL_Project"
img_dir = os.path.join(project_root, "data", "raw", "PanelImages")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = SolarDataset(img_dir, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, L, I = next(iter(loader))

print("图片batch shape:", images.shape)
print("L:", L[:5])
print("I:", I[:5])