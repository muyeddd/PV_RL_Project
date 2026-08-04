import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import matplotlib.pyplot as plt

# 项目根目录
project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from utils.dataset import SolarDataset
from models.cnn_with_i import SolarResNetWithI

# ======================
# 配置
# ======================
img_dir = os.path.join(project_root, "data", "raw", "PanelImages")
fig_dir = os.path.join(project_root, "outputs", "figures")
model_dir = os.path.join(project_root, "outputs", "models_ckpt")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ======================
# 数据
# ======================
def list_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f.lower())[1] in exts]
    return sorted(files)

def split_files(files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]
    return train_files, val_files, test_files

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

files = list_images(img_dir)
train_files, val_files, test_files = split_files(files, seed=SEED)

print("Total images:", len(files))
print("Train:", len(train_files), "Val:", len(val_files), "Test:", len(test_files))

train_ds = SolarDataset(img_dir, transform=transform_train)
train_ds.files = train_files

val_ds = SolarDataset(img_dir, transform=transform_eval)
val_ds.files = val_files

test_ds = SolarDataset(img_dir, transform=transform_eval)
test_ds.files = test_files

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ======================
# 模型
# ======================
model = SolarResNetWithI(dropout=0.3).to(device)
print("Model device:", next(model.parameters()).device)

# ======================
# 损失、优化器、调度器
# ======================
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

# ======================
# 工具函数
# ======================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for images, L, I in loader:
            images = images.to(device)
            L = L.to(device).unsqueeze(1)
            I = I.to(device)

            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    outputs = model(images, I)
                    loss = criterion(outputs, L)
            else:
                outputs = model(images, I)
                loss = criterion(outputs, L)

            total_loss += loss.item()
            preds.append(outputs.detach().cpu().numpy().reshape(-1))
            targets.append(L.detach().cpu().numpy().reshape(-1))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    return total_loss / len(loader), rmse, mae, r2, preds, targets

def plot_curve(train_losses, val_losses, path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve (Image + I)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_scatter(y_true, y_pred, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("True L")
    plt.ylabel("Predicted L")
    plt.title("Predicted vs True (Image + I)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ======================
# 训练
# ======================
best_val = float("inf")
best_epoch = -1
no_improve = 0
train_losses = []
val_losses = []

best_model_path = os.path.join(model_dir, "best_resnet18_with_i.pth")
final_model_path = os.path.join(model_dir, "final_resnet18_with_i.pth")

print("\nStart training...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train = 0.0

    for images, L, I in train_loader:
        images = images.to(device)
        L = L.to(device).unsqueeze(1)
        I = I.to(device)

        optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                outputs = model(images, I)
                loss = criterion(outputs, L)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, I)
            loss = criterion(outputs, L)
            loss.backward()
            optimizer.step()

        total_train += loss.item()

    train_loss = total_train / len(train_loader)

    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for images, L, I in val_loader:
            images = images.to(device)
            L = L.to(device).unsqueeze(1)
            I = I.to(device)

            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    outputs = model(images, I)
                    loss = criterion(outputs, L)
            else:
                outputs = model(images, I)
                loss = criterion(outputs, L)

            total_val += loss.item()

    val_loss = total_val / len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch [{epoch:02d}/{EPOCHS}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr_now:.2e}")

    if torch.cuda.is_available():
        print("GPU memory allocated:", f"{torch.cuda.memory_allocated()/1024**2:.1f} MB",
              "| reserved:", f"{torch.cuda.memory_reserved()/1024**2:.1f} MB")

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        print("Early stopping triggered at epoch", epoch)
        break

torch.save(model.state_dict(), final_model_path)

plot_curve(train_losses, val_losses, os.path.join(fig_dir, "training_curve_with_i.png"))

print("\nTraining finished.")
print("Best epoch:", best_epoch)
print("Best val loss:", best_val)
print("Best model saved to:", best_model_path)
print("Final model saved to:", final_model_path)

# ======================
# 测试集评估
# ======================
print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load(best_model_path, map_location=device))

test_loss, test_rmse, test_mae, test_r2, y_pred, y_true = evaluate(model, test_loader, criterion, device)

print("\n========== Test Results ==========")
print(f"Test Loss: {test_loss:.6f}")
print(f"Test RMSE:  {test_rmse:.6f}")
print(f"Test MAE:   {test_mae:.6f}")
print(f"Test R2:    {test_r2:.6f}")
print("==================================")

plot_scatter(y_true, y_pred, os.path.join(fig_dir, "test_scatter_with_i.png"))

print("GPU training confirmed:", torch.cuda.is_available())
print("Final device:", next(model.parameters()).device)