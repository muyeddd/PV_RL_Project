import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# 路径配置
# ============================================================
project_root = r"E:\PV_RL_Project"

mc_dir = os.path.join(project_root, "outputs", "mc_dropout_resnet50_with_i")
cal_path = os.path.join(mc_dir, "mc_calibration_predictions.csv")
test_path = os.path.join(mc_dir, "mc_test_predictions.csv")

out_dir = os.path.join(project_root, "outputs", "mc_conformal_resnet50_with_i")
os.makedirs(out_dir, exist_ok=True)

cal_df = pd.read_csv(cal_path)
test_df = pd.read_csv(test_path)

print("Loaded:")
print(cal_path)
print(test_path)
print("Calibration samples:", len(cal_df))
print("Test samples:", len(test_df))


# ============================================================
# 工具函数
# ============================================================
def conformal_quantile(scores, alpha):
    """
    Split conformal finite-sample corrected quantile.
    q_level = ceil((n + 1) * (1 - alpha)) / n
    """
    scores = np.asarray(scores)
    n = len(scores)
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(scores, q_level, method="higher")


def evaluate_intervals(df, lower_col, upper_col, width_col, confidence):
    y_true = df["y_true"].values
    y_pred = df["pred_mean"].values
    width = df[width_col].values
    abs_error = np.abs(y_true - y_pred)

    covered = ((y_true >= df[lower_col].values) &
               (y_true <= df[upper_col].values))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    picp = covered.mean()
    mpiw = width.mean()
    pinaw = mpiw / (y_true.max() - y_true.min() + 1e-8)

    if SCIPY_AVAILABLE:
        pearson_width_error = pearsonr(width, abs_error)[0]
        spearman_width_error = spearmanr(width, abs_error)[0]
    else:
        pearson_width_error = np.corrcoef(width, abs_error)[0, 1]
        spearman_width_error = np.nan

    return {
        "confidence": confidence,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "PICP": picp,
        "MPIW": mpiw,
        "PINAW": pinaw,
        "Pearson_width_error": pearson_width_error,
        "Spearman_width_error": spearman_width_error,
    }


def coverage_by_bins(bin_values, y_true, lower, upper, bins, name):
    covered = (y_true >= lower) & (y_true <= upper)
    rows = []

    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]

        if i == len(bins) - 2:
            mask = (bin_values >= left) & (bin_values <= right)
        else:
            mask = (bin_values >= left) & (bin_values < right)

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


# ============================================================
# 核心：MC + Conformal Calibration
# ============================================================
def main():
    eps = 1e-6

    # calibration nonconformity score:
    # score_i = |y_i - pred_mean_i| / (pred_std_i + eps)
    cal_abs_error = np.abs(cal_df["y_true"].values - cal_df["pred_mean"].values)
    cal_std = cal_df["pred_std"].values

    scores = cal_abs_error / (cal_std + eps)

    print("\nCalibration score statistics:")
    print(f"score mean   = {scores.mean():.6f}")
    print(f"score median = {np.median(scores):.6f}")
    print(f"score max    = {scores.max():.6f}")

    result_df = test_df.copy()
    summary_rows = []

    for alpha in [0.20, 0.10, 0.05]:
        confidence = int((1 - alpha) * 100)

        q = conformal_quantile(scores, alpha)

        half_width = q * (result_df["pred_std"].values + eps)

        lower = np.clip(result_df["pred_mean"].values - half_width, 0.0, 1.0)
        upper = np.clip(result_df["pred_mean"].values + half_width, 0.0, 1.0)

        result_df[f"lower_mc_conf_{confidence}"] = lower
        result_df[f"upper_mc_conf_{confidence}"] = upper
        result_df[f"width_mc_conf_{confidence}"] = upper - lower
        result_df[f"covered_mc_conf_{confidence}"] = (
            (result_df["y_true"].values >= lower) &
            (result_df["y_true"].values <= upper)
        ).astype(int)

        metrics = evaluate_intervals(
            result_df,
            lower_col=f"lower_mc_conf_{confidence}",
            upper_col=f"upper_mc_conf_{confidence}",
            width_col=f"width_mc_conf_{confidence}",
            confidence=confidence
        )

        metrics["alpha"] = alpha
        metrics["q"] = q
        summary_rows.append(metrics)

        print(f"\nMC + Conformal {confidence}% interval:")
        print(f"q = {q:.6f}")
        print(f"PICP = {metrics['PICP']:.6f}")
        print(f"MPIW = {metrics['MPIW']:.6f}")
        print(f"PINAW = {metrics['PINAW']:.6f}")
        print(f"Pearson(width, error) = {metrics['Pearson_width_error']:.6f}")
        print(f"Spearman(width, error) = {metrics['Spearman_width_error']:.6f}")

    summary_df = pd.DataFrame(summary_rows)

    # ============================================================
    # 分箱覆盖率：默认用 90% 区间
    # ============================================================
    lower_90 = result_df["lower_mc_conf_90"].values
    upper_90 = result_df["upper_mc_conf_90"].values
    y_true = result_df["y_true"].values

    bin_rows = []

    L_bins = np.linspace(0, 1, 11)
    bin_rows += coverage_by_bins(
        bin_values=y_true,
        y_true=y_true,
        lower=lower_90,
        upper=upper_90,
        bins=L_bins,
        name="L_true"
    )

    I_values = result_df["I"].values
    I_bins = np.linspace(I_values.min(), I_values.max(), 6)
    bin_rows += coverage_by_bins(
        bin_values=I_values,
        y_true=y_true,
        lower=lower_90,
        upper=upper_90,
        bins=I_bins,
        name="I"
    )

    bin_df = pd.DataFrame(bin_rows)

    # ============================================================
    # 同时生成对比表：Raw MC vs MC+Conformal
    # ============================================================
    raw_mc_summary_path = os.path.join(mc_dir, "mc_summary.csv")
    compare_rows = []

    if os.path.exists(raw_mc_summary_path):
        raw_df = pd.read_csv(raw_mc_summary_path)
        raw_test_df = raw_df[raw_df["split"] == "test"].copy()

        for _, row in raw_test_df.iterrows():
            compare_rows.append({
                "method": "Raw MC Dropout",
                "confidence": int(row["confidence"]),
                "PICP": row["PICP"],
                "MPIW": row["MPIW"],
                "PINAW": row["PINAW"],
                "Pearson_uncertainty_error": row["Pearson_std_error"],
                "Spearman_uncertainty_error": row["Spearman_std_error"],
            })

    for _, row in summary_df.iterrows():
        compare_rows.append({
            "method": "MC + Conformal",
            "confidence": int(row["confidence"]),
            "PICP": row["PICP"],
            "MPIW": row["MPIW"],
            "PINAW": row["PINAW"],
            "Pearson_uncertainty_error": row["Pearson_width_error"],
            "Spearman_uncertainty_error": row["Spearman_width_error"],
        })

    compare_df = pd.DataFrame(compare_rows)

    # ============================================================
    # 保存结果
    # ============================================================
    pred_out = os.path.join(out_dir, "mc_conformal_test_predictions.csv")
    summary_out = os.path.join(out_dir, "mc_conformal_summary.csv")
    bins_out = os.path.join(out_dir, "mc_conformal_coverage_by_bins_90.csv")
    compare_out = os.path.join(out_dir, "mc_vs_mc_conformal_compare.csv")

    result_df.to_csv(pred_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    bin_df.to_csv(bins_out, index=False)
    compare_df.to_csv(compare_out, index=False)

    print("\nSaved:")
    print(pred_out)
    print(summary_out)
    print(bins_out)
    print(compare_out)


if __name__ == "__main__":
    main()