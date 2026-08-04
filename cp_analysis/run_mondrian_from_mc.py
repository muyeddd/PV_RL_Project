import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ALPHA = 0.10  # 90% prediction interval


def conformal_quantile(scores, alpha=0.10):
    """
    Split conformal finite-sample quantile.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]

    n = len(scores)
    if n == 0:
        raise ValueError("scores 为空，无法计算 conformal quantile")

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    return np.quantile(scores, q_level, method="higher")


def evaluate_interval(df, lower_col, upper_col):
    covered = (df["true_L"] >= df[lower_col]) & (df["true_L"] <= df[upper_col])
    width = df[upper_col] - df[lower_col]
    abs_error = np.abs(df["true_L"] - df["pred_L"])

    return {
        "PICP": covered.mean(),
        "MPIW": width.mean(),
        "Median_width": width.median(),
        "MAE": abs_error.mean(),
        "RMSE": np.sqrt(np.mean(abs_error ** 2)),
    }


def add_split_cp(calib, test, alpha=0.10):
    scores = np.abs(calib["true_L"] - calib["pred_L"])
    q = conformal_quantile(scores, alpha)

    test = test.copy()
    test["split_lower"] = np.clip(test["pred_L"] - q, 0, 1)
    test["split_upper"] = np.clip(test["pred_L"] + q, 0, 1)
    test["split_width"] = test["split_upper"] - test["split_lower"]
    test["split_q"] = q
    test["split_covered"] = (
        (test["true_L"] >= test["split_lower"])
        & (test["true_L"] <= test["split_upper"])
    )

    return test, q


def add_mondrian_cp(calib, test, bin_col, bins, prefix, alpha=0.10, min_calib_per_bin=30):
    calib = calib.copy()
    test = test.copy()

    calib["_bin"] = pd.cut(calib[bin_col], bins=bins, include_lowest=True)
    test["_bin"] = pd.cut(test[bin_col], bins=bins, include_lowest=True)

    global_scores = np.abs(calib["true_L"] - calib["pred_L"])
    global_q = conformal_quantile(global_scores, alpha)

    q_map = {}
    count_map = {}

    for b in calib["_bin"].dropna().unique():
        sub = calib[calib["_bin"] == b]
        scores = np.abs(sub["true_L"] - sub["pred_L"])
        count_map[str(b)] = len(scores)

        if len(scores) < min_calib_per_bin:
            q_map[b] = global_q
        else:
            q_map[b] = conformal_quantile(scores, alpha)

    lowers = []
    uppers = []
    q_used = []
    bin_used = []

    for _, row in test.iterrows():
        b = row["_bin"]

        if pd.isna(b):
            q = global_q
        else:
            q = q_map.get(b, global_q)

        lower = max(0, row["pred_L"] - q)
        upper = min(1, row["pred_L"] + q)

        lowers.append(lower)
        uppers.append(upper)
        q_used.append(q)
        bin_used.append(str(b))

    test[f"{prefix}_lower"] = lowers
    test[f"{prefix}_upper"] = uppers
    test[f"{prefix}_width"] = np.array(uppers) - np.array(lowers)
    test[f"{prefix}_q"] = q_used
    test[f"{prefix}_bin"] = bin_used
    test[f"{prefix}_covered"] = (
        (test["true_L"] >= test[f"{prefix}_lower"])
        & (test["true_L"] <= test[f"{prefix}_upper"])
    )

    test = test.drop(columns=["_bin"])

    q_table = pd.DataFrame({
        "bin": [str(k) for k in q_map.keys()],
        "q": [float(v) for v in q_map.values()],
        "calib_count": [count_map.get(str(k), None) for k in q_map.keys()],
    })

    return test, q_table


def bin_coverage(df, lower_col, upper_col, group_col, bins, method_name):
    tmp = df.copy()
    tmp["bin"] = pd.cut(tmp[group_col], bins=bins, include_lowest=True)
    tmp["covered"] = (tmp["true_L"] >= tmp[lower_col]) & (tmp["true_L"] <= tmp[upper_col])
    tmp["width"] = tmp[upper_col] - tmp[lower_col]

    out = tmp.groupby("bin", observed=False).agg(
        n=("covered", "size"),
        coverage=("covered", "mean"),
        mpiw=("width", "mean"),
        true_mean=("true_L", "mean"),
        pred_mean=("pred_L", "mean"),
    ).reset_index()

    out["method"] = method_name
    out["group_by"] = group_col
    out["bin"] = out["bin"].astype(str)

    return out


def plot_group_coverage(bin_results, group_by, save_path):
    sub = bin_results[bin_results["group_by"] == group_by].copy()

    methods = list(sub["method"].unique())
    bins = list(sub["bin"].unique())

    x = np.arange(len(bins))
    width = 0.25

    plt.figure(figsize=(12, 5))

    for idx, method in enumerate(methods):
        m = sub[sub["method"] == method].set_index("bin").reindex(bins)
        positions = x + (idx - (len(methods) - 1) / 2) * width
        plt.bar(positions, m["coverage"], width=width, label=method)

    plt.axhline(0.90, linestyle="--", linewidth=2, label="Target 90%")
    plt.ylim(0, 1.05)
    plt.xticks(x, bins, rotation=30, ha="right")
    plt.ylabel("Empirical coverage")
    plt.xlabel(group_by)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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

    required = ["filename", "true_L", "pred_L", "irradiance", "pred_std"]

    for name, df in [("calib", calib), ("test", test)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} 缺少列: {missing}")

        for col in ["true_L", "pred_L", "irradiance", "pred_std"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[required].isna().any().any():
            print(df[required].isna().sum())
            raise ValueError(f"{name} 存在缺失值，请检查。")

    print("Calibration set:", calib.shape)
    print("Test set:", test.shape)

    # 1. Split CP
    test_out, split_q = add_split_cp(calib, test, ALPHA)

    # 2. Irradiance-Mondrian CP
    i_bins = np.linspace(0, 1, 6)
    test_out, i_q_table = add_mondrian_cp(
        calib=calib,
        test=test_out,
        bin_col="irradiance",
        bins=i_bins,
        prefix="i_mondrian",
        alpha=ALPHA,
    )

    # 3. Pred-L-Mondrian CP
    pred_bins = np.linspace(0, 1, 6)
    test_out, pred_q_table = add_mondrian_cp(
        calib=calib,
        test=test_out,
        bin_col="pred_L",
        bins=pred_bins,
        prefix="pred_mondrian",
        alpha=ALPHA,
    )

    # 总体指标
    summary = pd.DataFrame([
        {"method": "Split CP", **evaluate_interval(test_out, "split_lower", "split_upper")},
        {"method": "Irradiance-Mondrian CP", **evaluate_interval(test_out, "i_mondrian_lower", "i_mondrian_upper")},
        {"method": "Pred-L-Mondrian CP", **evaluate_interval(test_out, "pred_mondrian_lower", "pred_mondrian_upper")},
    ])

    # 分组覆盖率
    l_bins = np.linspace(0, 1, 11)
    i_bins = np.linspace(0, 1, 6)

    bin_results = pd.concat([
        bin_coverage(test_out, "split_lower", "split_upper", "true_L", l_bins, "Split CP"),
        bin_coverage(test_out, "i_mondrian_lower", "i_mondrian_upper", "true_L", l_bins, "Irradiance-Mondrian CP"),
        bin_coverage(test_out, "pred_mondrian_lower", "pred_mondrian_upper", "true_L", l_bins, "Pred-L-Mondrian CP"),

        bin_coverage(test_out, "split_lower", "split_upper", "irradiance", i_bins, "Split CP"),
        bin_coverage(test_out, "i_mondrian_lower", "i_mondrian_upper", "irradiance", i_bins, "Irradiance-Mondrian CP"),
        bin_coverage(test_out, "pred_mondrian_lower", "pred_mondrian_upper", "irradiance", i_bins, "Pred-L-Mondrian CP"),
    ], ignore_index=True)

    # 保存结果
    test_out.to_csv("mondrian_test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv("mondrian_summary.csv", index=False, encoding="utf-8-sig")
    bin_results.to_csv("mondrian_bin_coverage.csv", index=False, encoding="utf-8-sig")
    i_q_table.to_csv("i_mondrian_q_by_bin.csv", index=False, encoding="utf-8-sig")
    pred_q_table.to_csv("pred_mondrian_q_by_bin.csv", index=False, encoding="utf-8-sig")

    plot_group_coverage(bin_results, "true_L", "mondrian_coverage_by_true_L.png")
    plot_group_coverage(bin_results, "irradiance", "mondrian_coverage_by_irradiance.png")

    print("\n=== Mondrian CP Summary ===")
    print(summary)

    print("\nSplit CP q =", split_q)

    print("\nSaved files:")
    print("mondrian_test_predictions.csv")
    print("mondrian_summary.csv")
    print("mondrian_bin_coverage.csv")
    print("i_mondrian_q_by_bin.csv")
    print("pred_mondrian_q_by_bin.csv")
    print("mondrian_coverage_by_true_L.png")
    print("mondrian_coverage_by_irradiance.png")


if __name__ == "__main__":
    main()