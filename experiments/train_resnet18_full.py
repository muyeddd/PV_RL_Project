import os
import re
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

# =========================
# 配置区
# =========================
PROJECT_ROOT = r"E:\PV_RL_Project"
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "PanelImages")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models_ckpt")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_WORKERS = 0  # Windows/PyCharm 最稳；如果你后面想提速可以改成 2
USE_AMP = True   # 混合精度，RTX3050可用

# ImageNet 标准归一化，配合预训练ResNet18
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================
# 固定随机种子
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


seed_everything(SEED)


# =========================
# 解析文件名
# =========================
def parse_filename(filename):

    filename = filename.replace(".jpg", "")

    parts = filename.split("_")

    l_idx = parts.index("L")
    i_idx = parts.index("I")

    L = float(parts[l_idx + 1])
    I = float(parts[i_idx + 1])

    return L, I


# =========================
# 数据集
# =========================
class SolarDataset(Dataset):
    def __init__(self, img_dir, file_list, transform=None):
        self.img_dir = img_dir
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.img_dir, filename)

        image = Image.open(path).convert("RGB")
        L, I = parse_filename(filename)

        if self.transform is not None:
            image = self.transform(image)

        L = torch.tensor(L, dtype=torch.float32)
        I = torch.tensor(I, dtype=torch.float32)
        return image, L, I


# =========================
# 模型
# =========================
class SolarResNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=True)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        # 标签是 [0,1] 范围，sigmoid 可以避免负预测值
        x = torch.sigmoid(x)
        return x


# 工具函数
# =========================
def list_images(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for f in os.listdir(img_dir):
        if os.path.splitext(f.lower())[1] in exts:
            files.append(f)
    return sorted(files)


def split_files(files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    preds = []
    targets = []
    total_loss = 0.0
    criterion = nn.MSELoss()

    for images, L, I in loader:
        images = images.to(device, non_blocking=True)
        L = L.to(device, non_blocking=True).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, L)
        total_loss += loss.item()

        preds.append(outputs.detach().cpu().numpy().reshape(-1))
        targets.append(L.detach().cpu().numpy().reshape(-1))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, rmse, mae, r2, preds, targets


def plot_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scatter(y_true, y_pred, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("True L")
    plt.ylabel("Predicted L")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 主函数
# =========================
def main():
    print("=" * 60)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    files = list_images(IMG_DIR)
    print("Total images:", len(files))
    print("First 3 files:", files[:3])

    train_files, val_files, test_files = split_files(
        files,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_ds = SolarDataset(IMG_DIR, train_files, transform=train_tf)
    val_ds = SolarDataset(IMG_DIR, val_files, transform=eval_tf)
    test_ds = SolarDataset(IMG_DIR, test_files, transform=eval_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    model = SolarResNet(dropout=0.3).to(device)
    print("Model device:", next(model.parameters()).device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and torch.cuda.is_available()))

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    train_losses = []
    val_losses = []

    best_model_path = os.path.join(MODEL_DIR, "best_resnet18.pth")
    final_model_path = os.path.join(MODEL_DIR, "final_resnet18.pth")

    print("\nStart training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for step, (images, L, I) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            L = L.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and torch.cuda.is_available())):
                outputs = model(images)
                loss = criterion(outputs, L)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        train_loss = total_train_loss / max(len(train_loader), 1)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, L, I in val_loader:
                images = images.to(device, non_blocking=True)
                L = L.to(device, non_blocking=True).unsqueeze(1)

                with torch.cuda.amp.autocast(enabled=(USE_AMP and torch.cuda.is_available())):
                    outputs = model(images)
                    loss = criterion(outputs, L)

                total_val_loss += loss.item()

        val_loss = total_val_loss / max(len(val_loader), 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}"
        )

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024 ** 2
            mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2
            print(f"GPU memory allocated: {mem_alloc:.1f} MB | reserved: {mem_reserved:.1f} MB")

        # 保存最好模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1

        # early stopping
        if no_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)

    # 训练曲线
    curve_path = os.path.join(FIG_DIR, "training_curve.png")
    plot_curve(train_losses, val_losses, curve_path)

    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print("Best val loss:", best_val_loss)
    print("Best model saved to:", best_model_path)
    print("Final model saved to:", final_model_path)
    print("Training curve saved to:", curve_path)

    # ====== 测试集评估 ======
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_rmse, test_mae, test_r2, y_pred, y_true = evaluate(model, test_loader, device)

    print("\n========== Test Results ==========")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")
    print(f"Test MAE:   {test_mae:.6f}")
    print(f"Test R2:    {test_r2:.6f}")
    print("==================================")

    scatter_path = os.path.join(FIG_DIR, "test_scatter.png")
    plot_scatter(y_true, y_pred, scatter_path)
    print("Test scatter saved to:", scatter_path)

    if torch.cuda.is_available():
        print("\nGPU training confirmed: YES")
        print("Final device:", next(model.parameters()).device)
    else:
        print("\nGPU training confirmed: NO, running on CPU")


if __name__ == "__main__":
    main()