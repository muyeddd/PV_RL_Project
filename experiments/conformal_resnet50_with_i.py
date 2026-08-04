import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from models.resnet50_with_i import SolarResNet50WithI

# 这里沿用你原来 test_resnet50_with_i.py 里的 dataset / dataloader 导入方式
# 如果你的 test 脚本里不是这样写的，把原来的数据加载部分复制过来替换即可
from utils.dataset import SolarDataset


def unpack_batch(batch):
    """
    兼容两种 dataset 返回：
    1) image, L, I
    2) image, L, I, time_feat
    """
    if len(batch) == 3:
        images, L, I = batch
    else:
        images, L, I = batch[:3]
    return images, L, I


@torch.no_grad()
def predict_loader(model, loader, device, split_name="data"):
    model.eval()

    y_true_all = []
    y_pred_all = []
    I_all = []

    for batch in tqdm(loader, desc=f"Predicting {split_name}"):
        images, L, I = unpack_batch(batch)

        images = images.to(device)
        L = L.to(device).float()
        I = I.to(device).float()

        pred = model(images, I)

        pred = pred.view(-1).detach().cpu().numpy()
        true = L.view(-1).detach().cpu().numpy()
        irr = I.view(-1).detach().cpu().numpy()

        y_pred_all.append(pred)
        y_true_all.append(true)
        I_all.append(irr)

    y_pred_all = np.concatenate(y_pred_all)
    y_true_all = np.concatenate(y_true_all)
    I_all = np.concatenate(I_all)

    return y_true_all, y_pred_all, I_all


def conformal_quantile(abs_errors, alpha):
    """
    Split conformal regression quantile.
    使用有限样本修正：
    q = ceil((n + 1) * (1 - alpha)) / n 分位位置
    """
    n = len(abs_errors)
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(abs_errors, q_level, method="higher")


def interval_metrics(y_true, y_pred, lower, upper):
    covered = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    abs_error = np.abs(y_true - y_pred)

    picp = covered.mean()
    mpiw = width.mean()
    pinaw = mpiw / (y_true.max() - y_true.min() + 1e-8)

    try:
        pearson_corr = pearsonr(width, abs_error)[0]
    except Exception:
        pearson_corr = np.nan

    try:
        spearman_corr = spearmanr(width, abs_error)[0]
    except Exception:
        spearman_corr = np.nan

    return {
        "PICP": picp,
        "MPIW": mpiw,
        "PINAW": pinaw,
        "Pearson_width_error": pearson_corr,
        "Spearman_width_error": spearman_corr,
        "Mean_abs_error": abs_error.mean(),
    }


def coverage_by_bins(y_true, lower, upper, bins, name):
    covered = (y_true >= lower) & (y_true <= upper)
    rows = []

    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        mask = (y_true >= left) & (y_true < right)

        if mask.sum() == 0:
            rows.append({
                "bin_type": name,
                "bin": f"{left:.2f}-{right:.2f}",
                "count": 0,
                "coverage": np.nan
            })
        else:
            rows.append({
                "bin_type": name,
                "bin": f"{left:.2f}-{right:.2f}",
                "count": int(mask.sum()),
                "coverage": float(covered[mask].mean())
            })

    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_path = r"E:\PV_RL_Project\outputs\models_ckpt\best_resnet50_with_i.pth"

    model = SolarResNet50WithI(dropout=0.3, use_pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded:", ckpt_path)

    from torch.utils.data import DataLoader
    from torchvision import transforms

    # =========================
    # 和 train_resnet50_with_i.py 保持一致
    # =========================

    BATCH_SIZE = 32
    SEED = 42

    img_dir = os.path.join(project_root, "data", "raw", "PanelImages")  # 如果你训练脚本里不是这个路径，改成你原来的 img_dir

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

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

    # ============================================================
    # 1. calibration set 预测
    # ============================================================
    y_cal_true, y_cal_pred, I_cal = predict_loader(
        model, val_loader, device, split_name="calibration"
    )

    cal_abs_error = np.abs(y_cal_true - y_cal_pred)

    # ============================================================
    # 2. test set 预测
    # ============================================================
    y_test_true, y_test_pred, I_test = predict_loader(
        model, test_loader, device, split_name="test"
    )

    # 点预测指标
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae = mean_absolute_error(y_test_true, y_test_pred)
    r2 = r2_score(y_test_true, y_test_pred)

    print("\nPoint prediction:")
    print(f"RMSE = {rmse:.6f}")
    print(f"MAE  = {mae:.6f}")
    print(f"R2   = {r2:.6f}")

    # ============================================================
    # 3. conformal intervals
    # ============================================================
    alphas = [0.20, 0.10, 0.05]  # 80%, 90%, 95%
    summary_rows = []
    test_df = pd.DataFrame({
        "y_true": y_test_true,
        "y_pred": y_test_pred,
        "I": I_test,
        "abs_error": np.abs(y_test_true - y_test_pred)
    })

    for alpha in alphas:
        confidence = int((1 - alpha) * 100)

        q = conformal_quantile(cal_abs_error, alpha)

        lower = np.clip(y_test_pred - q, 0.0, 1.0)
        upper = np.clip(y_test_pred + q, 0.0, 1.0)

        metrics = interval_metrics(y_test_true, y_test_pred, lower, upper)

        row = {
            "confidence": confidence,
            "alpha": alpha,
            "q": q,
            **metrics
        }
        summary_rows.append(row)

        test_df[f"lower_{confidence}"] = lower
        test_df[f"upper_{confidence}"] = upper
        test_df[f"width_{confidence}"] = upper - lower
        test_df[f"covered_{confidence}"] = (
            (y_test_true >= lower) & (y_test_true <= upper)
        ).astype(int)

        print(f"\nConformal {confidence}% interval:")
        print(f"q = {q:.6f}")
        print(f"PICP = {metrics['PICP']:.6f}")
        print(f"MPIW = {metrics['MPIW']:.6f}")
        print(f"PINAW = {metrics['PINAW']:.6f}")
        print(f"Pearson(width, error) = {metrics['Pearson_width_error']:.6f}")
        print(f"Spearman(width, error) = {metrics['Spearman_width_error']:.6f}")

    # ============================================================
    # 4. coverage by L-bin and I-bin
    # ============================================================
    bin_rows = []

    # 先以 90% 区间为主做分箱覆盖率
    lower_90 = test_df["lower_90"].values
    upper_90 = test_df["upper_90"].values

    L_bins = np.linspace(0, 1, 11)
    bin_rows += coverage_by_bins(
        y_test_true, lower_90, upper_90, L_bins, name="L_true"
    )

    I_min, I_max = float(np.min(I_test)), float(np.max(I_test))
    I_bins = np.linspace(I_min, I_max, 6)
    bin_rows += coverage_by_bins(
        I_test, lower_90, upper_90, I_bins, name="I"
    )

    # ============================================================
    # 5. 保存结果
    # ============================================================
    out_dir = r"E:\PV_RL_Project\outputs\conformal_resnet50_with_i"
    os.makedirs(out_dir, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    bin_df = pd.DataFrame(bin_rows)

    test_df.to_csv(os.path.join(out_dir, "test_conformal_predictions.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "conformal_summary.csv"), index=False)
    bin_df.to_csv(os.path.join(out_dir, "coverage_by_bins_90.csv"), index=False)

    print("\nSaved to:")
    print(os.path.join(out_dir, "test_conformal_predictions.csv"))
    print(os.path.join(out_dir, "conformal_summary.csv"))
    print(os.path.join(out_dir, "coverage_by_bins_90.csv"))

def list_images(img_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
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
if __name__ == "__main__":
    main()