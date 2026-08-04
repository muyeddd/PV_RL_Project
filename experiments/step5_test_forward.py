import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.dataset import SolarDataset
from models.cnn_backbone import SolarResNet

project_root = r"E:\PV_RL_Project"
img_dir = os.path.join(project_root, "data", "raw", "PanelImages")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = SolarDataset(img_dir, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SolarResNet()

images, L, I = next(iter(loader))

pred = model(images).squeeze()

loss_fn = nn.MSELoss()
loss = loss_fn(pred, L)

print("预测值:", pred[:5].detach())
print("真实值:", L[:5])
print("loss:", loss.item())