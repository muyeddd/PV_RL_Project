import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# Project path
# ============================================================
project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from models.resnet50_with_i import SolarResNet50WithI
from utils.dataset import SolarDataset


# ============================================================
# Basic config
# ============================================================
SEED = 42
BATCH_SIZE = 32
MC_SAMPLES = 50

img_dir = os.path.join(project_root, "data", "raw", "PanelImages")
ckpt_path = os.path.join(project_root, "outputs", "models_ckpt", "best_resnet50_with_i.pth")

out_dir = os.path.join(project_root, "outputs", "mc_dropout_resnet50_with_i")
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# Dataset split functions: keep consistent with previous scripts
# ============================================================
def list_images(img_dir):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    files = [
        f for f in os.listdir(img_dir)
        if os.path.splitext(f.lower())[1] in exts
    ]
    files.sort()
    return files


def split_files(files, seed=42, train_ratio=0.7, val_ratio=0.15):
    rng = np.random.default_rng(seed)

    files = np.array(files)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train].tolist()
    val_files = files[n_train:n_train + n_val].tolist()
    test_files = files[n_train + n_val:].tolist()

    return train_files, val_files, test_files


def unpack_batch(batch):
    """
    Compatible with:
    1) image, L, I
    2) image, L, I, time_feat
    """
    if len(batch) == 3:
        images, L, I = batch
    else:
        images, L, I = batch[:3]
    return images, L, I


# ============================================================
# MC Dropout utilities
# ============================================================
def enable_dropout_only(model):
    """
    Keep the whole model in eval mode, but activate Dropout layers only.

    This is important because ResNet contains BatchNorm layers.
    We do not want BatchNorm running statistics to change during testing.
    """
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_predict_loader(model, loader, device, file_list=None, mc_samples=50, split_name="test"):
    """
    Perform MC Dropout prediction.

    Returns:
        DataFrame with:
        filename, y_true, pred_mean, pred_std, lower/upper intervals, abs_error, I
    """

    model.eval()

    y_true_all = []
    pred_mean_all = []
    pred_std_all = []
    I_all = []
    filenames_all = []

    start_idx = 0

    for batch in tqdm(loader, desc=f"MC Dropout predicting {split_name}"):
        images, L, I = unpack_batch(batch)

        images = images.to(device)
        L = L.to(device).float()
        I = I.to(device).float()

        batch_size = images.size(0)

        # keep BatchNorm eval, activate Dropout only
        enable_dropout_only(model)

        preds_mc = []

        for _ in range(mc_samples):
            pred = model(images, I)
            pred = pred.view(-1)
            preds_mc.append(pred.detach().cpu().numpy())

        preds_mc = np.stack(preds_mc, axis=0)  # [T, B]

        pred_mean = preds_mc.mean(axis=0)
        pred_std = preds_mc.std(axis=0, ddof=1)

        y_true = L.view(-1).detach().cpu().numpy()
        irr = I.view(-1).detach().cpu().numpy()

        y_true_all.append(y_true)
        pred_mean_all.append(pred_mean)
        pred_std_all.append(pred_std)
        I_all.append(irr)

        if file_list is not None:
            filenames_all.extend(file_list[start_idx:start_idx + batch_size])
        else:
            filenames_all.extend([f"{split_name}_{i}" for i in range(start_idx, start_idx + batch_size)])

        start_idx += batch_size

    y_true_all = np.concatenate(y_true_all)
    pred_mean_all = np.concatenate(pred_mean_all)
    pred_std_all = np.concatenate(pred_std_all)
    I_all = np.concatenate(I_all)

    df = pd.DataFrame({
        "filename": filenames_all,
        "y_true": y_true_all,
        "pred_mean": pred_mean_all,
        "pred_std": pred_std_all,
        "I": I_all,
    })

    df["abs_error"] = np.abs(df["y_true"] - df["pred_mean"])

    # Raw Gaussian-style MC intervals
    # 90%: z = 1.645
    # 95%: z = 1.96
    for conf, z in [(90, 1.645), (95, 1.96)]:
        df[f"lower_mc_{conf}"] = np.clip(df["pred_mean"] - z * df["pred_std"], 0.0, 1.0)
        df[f"upper_mc_{conf}"] = np.clip(df["pred_mean"] + z * df["pred_std"], 0.0, 1.0)
        df[f"width_mc_{conf}"] = df[f"upper_mc_{conf}"] - df[f"lower_mc_{conf}"]
        df[f"covered_mc_{conf}"] = (
            (df["y_true"] >= df[f"lower_mc_{conf}"]) &
            (df["y_true"] <= df[f"upper_mc_{conf}"])
        ).astype(int)

    return df


def evaluate_mc(df, split_name="test"):
    y_true = df["y_true"].values
    y_pred = df["pred_mean"].values
    pred_std = df["pred_std"].values
    abs_error = df["abs_error"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if SCIPY_AVAILABLE:
        pearson_std_error = pearsonr(pred_std, abs_error)[0]
        spearman_std_error = spearmanr(pred_std, abs_error)[0]
    else:
        pearson_std_error = np.corrcoef(pred_std, abs_error)[0, 1]
        spearman_std_error = np.nan

    rows = []

    for conf in [90, 95]:
        picp = df[f"covered_mc_{conf}"].mean()
        mpiw = df[f"width_mc_{conf}"].mean()
        pinaw = mpiw / (y_true.max() - y_true.min() + 1e-8)

        rows.append({
            "split": split_name,
            "confidence": conf,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "mean_pred_std": float(np.mean(pred_std)),
            "median_pred_std": float(np.median(pred_std)),
            "max_pred_std": float(np.max(pred_std)),
            "PICP": picp,
            "MPIW": mpiw,
            "PINAW": pinaw,
            "Pearson_std_error": pearson_std_error,
            "Spearman_std_error": spearman_std_error,
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Image dir:", img_dir)
    print("Checkpoint:", ckpt_path)
    print("MC samples:", MC_SAMPLES)

    # -----------------------------
    # Transforms
    # -----------------------------
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # -----------------------------
    # Data split
    # -----------------------------
    files = list_images(img_dir)
    train_files, val_files, test_files = split_files(files, seed=SEED)

    print("Total images:", len(files))
    print("Train:", len(train_files), "Val/Calibration:", len(val_files), "Test:", len(test_files))

    val_ds = SolarDataset(img_dir, transform=transform_eval)
    val_ds.files = val_files

    test_ds = SolarDataset(img_dir, transform=transform_eval)
    test_ds.files = test_files

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -----------------------------
    # Load model
    # -----------------------------
    model = SolarResNet50WithI(dropout=0.3, use_pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded.")

    # -----------------------------
    # MC prediction
    # -----------------------------
    cal_df = mc_predict_loader(
        model=model,
        loader=val_loader,
        device=device,
        file_list=val_files,
        mc_samples=MC_SAMPLES,
        split_name="calibration"
    )

    test_df = mc_predict_loader(
        model=model,
        loader=test_loader,
        device=device,
        file_list=test_files,
        mc_samples=MC_SAMPLES,
        split_name="test"
    )

    # -----------------------------
    # Evaluation
    # -----------------------------
    cal_summary = evaluate_mc(cal_df, split_name="calibration")
    test_summary = evaluate_mc(test_df, split_name="test")
    summary_df = pd.concat([cal_summary, test_summary], ignore_index=True)

    print("\nMC Dropout summary:")
    print(summary_df)

    # -----------------------------
    # Save
    # -----------------------------
    cal_path = os.path.join(out_dir, "mc_calibration_predictions.csv")
    test_path = os.path.join(out_dir, "mc_test_predictions.csv")
    summary_path = os.path.join(out_dir, "mc_summary.csv")

    cal_df.to_csv(cal_path, index=False)
    test_df.to_csv(test_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(cal_path)
    print(test_path)
    print(summary_path)


if __name__ == "__main__":
    main()