import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def rmse(x):
    return np.sqrt(np.mean(np.asarray(x) ** 2))


def selective_curve(df, risk_col, error_col="abs_error", coverages=None):
    """
    按风险分数从低到高排序。
    coverage = 保留比例。
    每个coverage下，只保留低风险样本，计算保留样本误差。
    """
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 19)

    tmp = df.sort_values(risk_col, ascending=True).reset_index(drop=True)
    n = len(tmp)

    rows = []
    for c in coverages:
        k = max(1, int(np.floor(c * n)))
        retained = tmp.iloc[:k]

        rows.append({
            "coverage": c,
            "retained_n": k,
            "retained_mae": retained[error_col].mean(),
            "retained_rmse": rmse(retained[error_col]),
        })

    return pd.DataFrame(rows)


def oracle_curve(df, error_col="abs_error", coverages=None):
    """
    理想排序：直接按真实误差从小到大排序。
    这是理论最优筛选效果。
    """
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 19)

    tmp = df.sort_values(error_col, ascending=True).reset_index(drop=True)
    n = len(tmp)

    rows = []
    for c in coverages:
        k = max(1, int(np.floor(c * n)))
        retained = tmp.iloc[:k]

        rows.append({
            "coverage": c,
            "retained_n": k,
            "oracle_mae": retained[error_col].mean(),
            "oracle_rmse": rmse(retained[error_col]),
        })

    return pd.DataFrame(rows)


def ause(method_curve, oracle, metric="retained_mae"):
    """
    AUSE: method curve 与 oracle curve 的面积差。
    越小越好。
    """
    x = method_curve["coverage"].values

    if metric == "retained_mae":
        y_method = method_curve["retained_mae"].values
        y_oracle = oracle["oracle_mae"].values
    elif metric == "retained_rmse":
        y_method = method_curve["retained_rmse"].values
        y_oracle = oracle["oracle_rmse"].values
    else:
        raise ValueError("metric must be retained_mae or retained_rmse")

    diff = y_method - y_oracle

    # 手动梯形积分，避免不同 numpy 版本兼容问题
    area = np.sum((x[1:] - x[:-1]) * (diff[1:] + diff[:-1]) / 2)

    return area


def accepted_rejected_stats(df, risk_col, reject_rate=0.2):
    """
    拒绝风险最高的 reject_rate 样本。
    accepted = 低风险保留样本
    rejected = 高风险拒绝样本
    """
    tmp = df.sort_values(risk_col, ascending=True).reset_index(drop=True)
    n = len(tmp)
    n_reject = int(np.floor(reject_rate * n))
    n_accept = n - n_reject

    accepted = tmp.iloc[:n_accept]
    rejected = tmp.iloc[n_accept:]

    return {
        "accepted_ratio": n_accept / n,
        "rejected_ratio": n_reject / n,
        "accepted_n": n_accept,
        "rejected_n": n_reject,
        "accepted_mae": accepted["abs_error"].mean(),
        "rejected_mae": rejected["abs_error"].mean(),
        "accepted_rmse": rmse(accepted["abs_error"]),
        "rejected_rmse": rmse(rejected["abs_error"]),
        "mae_ratio_rejected_over_accepted": rejected["abs_error"].mean() / accepted["abs_error"].mean(),
        "rmse_ratio_rejected_over_accepted": rmse(rejected["abs_error"]) / rmse(accepted["abs_error"]),
    }


def main():
    df = pd.read_csv("final_test_predictions_all_methods.csv")

    required = [
        "true_L",
        "pred_L",
        "raw_mc_width",
        "split_width",
        "pred_l_mondrian_width",
        "pred_l_mondrian_mc_width",
        "pred_l_mondrian_std_mc_width",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    df["abs_error"] = (df["true_L"] - df["pred_L"]).abs()

    methods = [
        ("Raw MC", "raw_mc_width"),
        ("Split CP", "split_width"),
        ("Pred-L-Mondrian CP", "pred_l_mondrian_width"),
        ("Pred-L-Mondrian MC-Interval CP", "pred_l_mondrian_mc_width"),
        ("Pred-L-Mondrian Std-MC CP", "pred_l_mondrian_std_mc_width"),
    ]

    coverages = np.linspace(0.1, 1.0, 19)
    oracle = oracle_curve(df, coverages=coverages)

    all_curves = []
    summary_rows = []

    for method_name, risk_col in methods:
        curve = selective_curve(df, risk_col, coverages=coverages)
        curve["method"] = method_name
        curve["risk_col"] = risk_col

        all_curves.append(curve)

        stats = accepted_rejected_stats(df, risk_col, reject_rate=0.2)

        summary_rows.append({
            "method": method_name,
            "risk_col": risk_col,
            "AUSE_MAE": ause(curve, oracle, metric="retained_mae"),
            "AUSE_RMSE": ause(curve, oracle, metric="retained_rmse"),
            **stats
        })

    all_curves = pd.concat(all_curves, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    all_curves.to_csv("risk_coverage_curves.csv", index=False, encoding="utf-8-sig")
    oracle.to_csv("oracle_risk_coverage_curve.csv", index=False, encoding="utf-8-sig")
    summary.to_csv("final_risk_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Risk-Coverage Summary ===")
    print(summary)

    # 图1：Retained MAE
    plt.figure(figsize=(8, 5))
    for method_name, _ in methods:
        sub = all_curves[all_curves["method"] == method_name]
        plt.plot(sub["coverage"], sub["retained_mae"], marker="o", label=method_name)

    plt.plot(oracle["coverage"], oracle["oracle_mae"], linestyle="--", color="black", label="Oracle")
    plt.xlabel("Retained coverage")
    plt.ylabel("Retained MAE")
    plt.title("Risk-Coverage Curve (MAE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("risk_coverage_mae.png", dpi=300)
    plt.close()

    # 图2：Retained RMSE
    plt.figure(figsize=(8, 5))
    for method_name, _ in methods:
        sub = all_curves[all_curves["method"] == method_name]
        plt.plot(sub["coverage"], sub["retained_rmse"], marker="o", label=method_name)

    plt.plot(oracle["coverage"], oracle["oracle_rmse"], linestyle="--", color="black", label="Oracle")
    plt.xlabel("Retained coverage")
    plt.ylabel("Retained RMSE")
    plt.title("Risk-Coverage Curve (RMSE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("risk_coverage_rmse.png", dpi=300)
    plt.close()

    # 图3：AUSE-MAE柱状图
    plt.figure(figsize=(9, 5))
    plt.bar(summary["method"], summary["AUSE_MAE"])
    plt.ylabel("AUSE-MAE")
    plt.title("AUSE-MAE Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("ause_mae_comparison.png", dpi=300)
    plt.close()

    # 图4：Accepted vs Rejected MAE
    x = np.arange(len(summary))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, summary["accepted_mae"], width=width, label="Accepted 80%")
    plt.bar(x + width / 2, summary["rejected_mae"], width=width, label="Rejected 20%")
    plt.ylabel("MAE")
    plt.title("Accepted vs Rejected Error")
    plt.xticks(x, summary["method"], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accepted_rejected_mae.png", dpi=300)
    plt.close()

    print("\nSaved files:")
    print("final_risk_summary.csv")
    print("risk_coverage_curves.csv")
    print("risk_coverage_mae.png")
    print("risk_coverage_rmse.png")
    print("ause_mae_comparison.png")
    print("accepted_rejected_mae.png")


if __name__ == "__main__":
    main()