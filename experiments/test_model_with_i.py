import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 把项目根目录加到路径里
project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from utils.dataset import SolarDataset
from models.cnn_with_i import SolarResNetWithI

# ======================
# 基本配置
# ======================
img_dir = r"E:\PV_RL_Project\data\raw\PanelImages"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# ======================
# 数据预处理
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# 数据集与DataLoader
# ======================
dataset = SolarDataset(img_dir, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ======================
# 模型
# ======================
model = SolarResNetWithI(dropout=0.3).to(device)
print("Model device:", next(model.parameters()).device)

# ======================
# 取一个batch测试前向传播
# ======================
images, L, I = next(iter(loader))

images = images.to(device)
I = I.to(device)

with torch.no_grad():
    preds = model(images, I)

print("images shape:", images.shape)
print("I shape:", I.shape)
print("preds shape:", preds.shape)
print("preds:", preds.squeeze()[:5])
print("L:", L[:5])