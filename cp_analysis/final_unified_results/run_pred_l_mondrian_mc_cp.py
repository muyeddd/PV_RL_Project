import pandas as pd
import numpy as np
from scipy.stats import spearmanr


ALPHA = 0.10  # 90% prediction interval
EPS = 1e-8


def conformal_quantile(scores, alpha=0.10):
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]

    n = len(scores)
    if n == 0:
        raise ValueError("scores为空，无法计算分位数。")

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    return np.quantile(scores, q_level, method="higher")


def evaluate_interval(df, lower_col, upper_col, method_name):
    covered = (df["true_L"] >= df[lower_col]) & (df["true_L"] <= df[upper_col])
    width = df[upper_col] - df[lower_col]
    abs_error = np.abs(df["true_L"] - df["pred_L"])

    rho, p_value = spearmanr(width, abs_error)

    return {
        "method": method_name,
        "PICP": covered.mean(),
        "MPIW": width.mean(),
        "Median_width": width.median(),
        "Spearman_width_error": rho,
        "Spearman_p_value": p_value,
        "MAE": abs_error.mean(),
        "RMSE": np.sqrt(np.mean(abs_error ** 2)),
    }


def add_split_cp(calib, test, alpha=0.10):
    scores = np.abs(calib["true_L"] - calib["pred_L"])
    q = conformal_quantile(scores, alpha)

    test["split_lower"] = np.clip(test["pred_L"] - q, 0, 1)
    test["split_upper"] = np.clip(test["pred_L"] + q, 0, 1)
    test["split_width"] = test["split_upper"] - test["split_lower"]
    test["split_covered"] = (
        (test["true_L"] >= test["split_lower"])
        & (test["true_L"] <= test["split_upper"])
    )

    return test, q


def add_pred_l_mondrian_cp(calib, test, bins, alpha=0.10, min_calib_per_bin=30):
    calib = calib.copy()
    test = test.copy()

    calib["_bin"] = pd.cut(calib["pred_L"], bins=bins, include_lowest=True)
    test["_bin"] = pd.cut(test["pred_L"], bins=bins, include_lowest=True)

    global_scores = np.abs(calib["true_L"] - calib["pred_L"])
    global_q = conformal_quantile(global_scores, alpha)

    q_map = {}
    q_rows = []

    for b in calib["_bin"].dropna().unique():
        sub = calib[calib["_bin"] == b]
        scores = np.abs(sub["true_L"] - sub["pred_L"])

        if len(scores) < min_calib_per_bin:
            q = global_q
        else:
            q = conformal_quantile(scores, alpha)

        q_map[b] = q
        q_rows.append({
            "bin": str(b),
            "calib_count": len(scores),
            "q_abs_residual": q,
        })

    lower_list = []
    upper_list = []
    q_used_list = []
    bin_list = []

    for _, row in test.iterrows():
        b = row["_bin"]
        q = q_map.get(b, global_q)

        lower = max(0, row["pred_L"] - q)
        upper = min(1, row["pred_L"] + q)

        lower_list.append(lower)
        upper_list.append(upper)
        q_used_list.append(q)
        bin_list.append(str(b))

    test["pred_l_mondrian_lower"] = lower_list
    test["pred_l_mondrian_upper"] = upper_list
    test["pred_l_mondrian_width"] = np.array(upper_list) - np.array(lower_list)
    test["pred_l_mondrian_q"] = q_used_list
    test["pred_l_mondrian_bin"] = bin_list
    test["pred_l_mondrian_covered"] = (
        (test["true_L"] >= test["pred_l_mondrian_lower"])
        & (test["true_L"] <= test["pred_l_mondrian_upper"])
    )

    test = test.drop(columns=["_bin"])
    q_table = pd.DataFrame(q_rows)

    return test, q_table


def add_pred_l_mondrian_mc_interval_cp(calib, test, bins, alpha=0.10, min_calib_per_bin=30):
    """
    方法1：Raw MC interval + Pred-L-Mondrian conformal expansion

    校准分数：
    score = max(lower_mc_90 - y_true, y_true - upper_mc_90, 0)

    测试区间：
    [lower_mc_90 - q_b, upper_mc_90 + q_b]
    """
    calib = calib.copy()
    test = test.copy()

    calib["_bin"] = pd.cut(calib["pred_L"], bins=bins, include_lowest=True)
    test["_bin"] = pd.cut(test["pred_L"], bins=bins, include_lowest=True)

    calib["mc_interval_score"] = np.maximum.reduce([
        calib["lower_mc_90"] - calib["true_L"],
        calib["true_L"] - calib["upper_mc_90"],
        np.zeros(len(calib)),
    ])

    global_q = conformal_quantile(calib["mc_interval_score"], alpha)

    q_map = {}
    q_rows = []

    for b in calib["_bin"].dropna().unique():
        sub = calib[calib["_bin"] == b]
        scores = sub["mc_interval_score"]

        if len(scores) < min_calib_per_bin:
            q = global_q
        else:
            q = conformal_quantile(scores, alpha)

        q_map[b] = q
        q_rows.append({
            "bin": str(b),
            "calib_count": len(scores),
            "q_mc_expand": q,
        })

    lower_list = []
    upper_list = []
    q_used_list = []
    bin_list = []

    for _, row in test.iterrows():
        b = row["_bin"]
        q = q_map.get(b, global_q)

        lower = max(0, row["lower_mc_90"] - q)
        upper = min(1, row["upper_mc_90"] + q)

        lower_list.append(lower)
        upper_list.append(upper)
        q_used_list.append(q)
        bin_list.append(str(b))

    test["pred_l_mondrian_mc_lower"] = lower_list
    test["pred_l_mondrian_mc_upper"] = upper_list
    test["pred_l_mondrian_mc_width"] = np.array(upper_list) - np.array(lower_list)
    test["pred_l_mondrian_mc_q"] = q_used_list
    test["pred_l_mondrian_mc_bin"] = bin_list
    test["pred_l_mondrian_mc_covered"] = (
        (test["true_L"] >= test["pred_l_mondrian_mc_lower"])
        & (test["true_L"] <= test["pred_l_mondrian_mc_upper"])
    )

    test = test.drop(columns=["_bin"])
    q_table = pd.DataFrame(q_rows)

    return test, q_table


def add_pred_l_mondrian_std_mc_cp(calib, test, bins, alpha=0.10, min_calib_per_bin=30):
    """
    方法2：标准化 MC+CP

    校准分数：
    score = |y_true - pred_mean| / (pred_std + eps)

    测试区间：
    [pred_mean - q_b * pred_std, pred_mean + q_b * pred_std]
    """
    calib = calib.copy()
    test = test.copy()

    calib["_bin"] = pd.cut(calib["pred_L"], bins=bins, include_lowest=True)
    test["_bin"] = pd.cut(test["pred_L"], bins=bins, include_lowest=True)

    calib["std_mc_score"] = np.abs(calib["true_L"] - calib["pred_L"]) / (calib["pred_std"] + EPS)

    global_q = conformal_quantile(calib["std_mc_score"], alpha)

    q_map = {}
    q_rows = []

    for b in calib["_bin"].dropna().unique():
        sub = calib[calib["_bin"] == b]
        scores = sub["std_mc_score"]

        if len(scores) < min_calib_per_bin:
            q = global_q
        else:
            q = conformal_quantile(scores, alpha)

        q_map[b] = q
        q_rows.append({
            "bin": str(b),
            "calib_count": len(scores),
            "q_std_mc": q,
        })

    lower_list = []
    upper_list = []
    q_used_list = []
    bin_list = []

    for _, row in test.iterrows():
        b = row["_bin"]
        q = q_map.get(b, global_q)

        half_width = q * (row["pred_std"] + EPS)

        lower = max(0, row["pred_L"] - half_width)
        upper = min(1, row["pred_L"] + half_width)

        lower_list.append(lower)
        upper_list.append(upper)
        q_used_list.append(q)
        bin_list.append(str(b))

    test["pred_l_mondrian_std_mc_lower"] = lower_list
    test["pred_l_mondrian_std_mc_upper"] = upper_list
    test["pred_l_mondrian_std_mc_width"] = np.array(upper_list) - np.array(lower_list)
    test["pred_l_mondrian_std_mc_q"] = q_used_list
    test["pred_l_mondrian_std_mc_bin"] = bin_list
    test["pred_l_mondrian_std_mc_covered"] = (
        (test["true_L"] >= test["pred_l_mondrian_std_mc_lower"])
        & (test["true_L"] <= test["pred_l_mondrian_std_mc_upper"])
    )

    test = test.drop(columns=["_bin"])
    q_table = pd.DataFrame(q_rows)

    return test, q_table


def main():
    calib_raw = pd.read_csv("mc_calibration_predictions.csv")
    test_raw = pd.read_csv("mc_test_predictions.csv")

    # 统一列名
    calib = calib_raw.rename(columns={
        "y_true": "true_L",
        "pred_mean": "pred_L",
        "I": "irradiance",
    })

    test = test_raw.rename(columns={
        "y_true": "true_L",
        "pred_mean": "pred_L",
        "I": "irradiance",
    })

    required = [
        "filename",
        "true_L",
        "pred_L",
        "pred_std",
        "irradiance",
        "lower_mc_90",
        "upper_mc_90",
    ]

    for name, df in [("calib", calib), ("test", test)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} 缺少列: {missing}")

        for col in ["true_L", "pred_L", "pred_std", "irradiance", "lower_mc_90", "upper_mc_90"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[required].isna().any().any():
            print(df[required].isna().sum())
            raise ValueError(f"{name} 存在缺失值，请检查。")

    print("Calibration set:", calib.shape)
    print("Test set:", test.shape)

    # 分箱：按预测功率损失 pred_L
    pred_bins = np.linspace(0, 1, 6)

    # 1. Raw MC
    test["raw_mc_lower"] = np.clip(test["lower_mc_90"], 0, 1)
    test["raw_mc_upper"] = np.clip(test["upper_mc_90"], 0, 1)
    test["raw_mc_width"] = test["raw_mc_upper"] - test["raw_mc_lower"]

    # 2. Split CP
    test, split_q = add_split_cp(calib, test, ALPHA)

    # 3. Pred-L-Mondrian CP
    test, q_pred_l_cp = add_pred_l_mondrian_cp(calib, test, pred_bins, ALPHA)

    # 4. Pred-L-Mondrian MC Interval CP
    test, q_pred_l_mc_interval = add_pred_l_mondrian_mc_interval_cp(calib, test, pred_bins, ALPHA)

    # 5. Pred-L-Mondrian standardized MC+CP
    test, q_pred_l_std_mc = add_pred_l_mondrian_std_mc_cp(calib, test, pred_bins, ALPHA)

    # 总结指标
    summary = pd.DataFrame([
        evaluate_interval(test, "raw_mc_lower", "raw_mc_upper", "Raw MC"),
        evaluate_interval(test, "split_lower", "split_upper", "Split CP"),
        evaluate_interval(test, "pred_l_mondrian_lower", "pred_l_mondrian_upper", "Pred-L-Mondrian CP"),
        evaluate_interval(test, "pred_l_mondrian_mc_lower", "pred_l_mondrian_mc_upper", "Pred-L-Mondrian MC-Interval CP"),
        evaluate_interval(test, "pred_l_mondrian_std_mc_lower", "pred_l_mondrian_std_mc_upper", "Pred-L-Mondrian Std-MC CP"),
    ])

    # 保存
    test.to_csv("final_test_predictions_all_methods.csv", index=False, encoding="utf-8-sig")
    summary.to_csv("final_interval_summary.csv", index=False, encoding="utf-8-sig")
    q_pred_l_cp.to_csv("q_pred_l_mondrian_cp.csv", index=False, encoding="utf-8-sig")
    q_pred_l_mc_interval.to_csv("q_pred_l_mondrian_mc_interval_cp.csv", index=False, encoding="utf-8-sig")
    q_pred_l_std_mc.to_csv("q_pred_l_mondrian_std_mc_cp.csv", index=False, encoding="utf-8-sig")

    print("\n=== Pred-L-Mondrian MC+CP Summary ===")
    print(summary)

    print("\nSplit CP q =", split_q)

    print("\nSaved:")
    print("final_test_predictions_all_methods.csv")
    print("final_interval_summary.csv")
    print("q_pred_l_mondrian_cp.csv")
    print("q_pred_l_mondrian_mc_interval_cp.csv")
    print("q_pred_l_mondrian_std_mc_cp.csv")


if __name__ == "__main__":
    main()