import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ====== 强制加入项目根目录 ======
sys.path.insert(0, r"E:\PV_RL_Project")

from utils.dataset import SolarDataset
from models.cnn_backbone import SolarResNet

# ====== 数据路径 ======
img_dir = r"E:\PV_RL_Project\data\raw\PanelImages"

# ====== 数据增强/预处理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ====== Dataset ======
dataset = SolarDataset(img_dir, transform=transform)

# 划分训练/验证（简单版）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# ====== 模型 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolarResNet().to(device)

# ====== 损失函数 & 优化器 ======
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ====== 训练参数 ======
epochs = 10

train_losses = []
val_losses = []

print("Start training...")

for epoch in range(epochs):

    # ================= TRAIN =================
    model.train()
    train_loss = 0

    for images, L, I in train_loader:
        images = images.to(device)
        L = L.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, L)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ================= VAL =================
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, L, I in val_loader:
            images = images.to(device)
            L = L.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, L)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# ====== 保存模型 ======
torch.save(model.state_dict(), "cnn_resnet18.pth")

# ====== 保存loss ======
import matplotlib.pyplot as plt

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Curve")
plt.savefig("training_curve.png")
plt.show()

print("Training finished. Model saved.")