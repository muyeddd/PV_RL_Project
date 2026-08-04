import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# =========================
# 路径配置
# =========================
project_root = r"E:\PV_RL_Project"
result_dir = os.path.join(project_root, "outputs", "conformal_resnet50_with_i")
fig_dir = os.path.join(result_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

summary_path = os.path.join(result_dir, "conformal_summary.csv")
test_path = os.path.join(result_dir, "test_conformal_predictions.csv")
coverage_bin_path = os.path.join(result_dir, "coverage_by_bins_90.csv")

# =========================
# 读取数据
# =========================
summary_df = pd.read_csv(summary_path)
test_df = pd.read_csv(test_path)
bin_df = pd.read_csv(coverage_bin_path)

print("Loaded:")
print(summary_path)
print(test_path)
print(coverage_bin_path)

# =========================
# 全局绘图风格（论文风格）
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "lines.linewidth": 2.0,
})

# 颜色
C_MAIN = "#1f4e79"      # 深蓝
C_ACCENT = "#d62728"    # 红
C_SOFT = "#4c78a8"      # 柔蓝
C_GREEN = "#2ca02c"
C_ORANGE = "#ff7f0e"
C_GRAY = "#808080"

# =========================
# 图1：Coverage-Width Tradeoff
# =========================
def plot_tradeoff(summary_df):
    conf = summary_df["confidence"].values
    target = conf / 100.0
    picp = summary_df["PICP"].values
    mpiw = summary_df["MPIW"].values

    fig, ax1 = plt.subplots(figsize=(7.2, 5.2))

    ax1.plot(conf, picp, marker="o", markersize=7, color=C_MAIN, label="Empirical Coverage (PICP)")
    ax1.plot(conf, target, linestyle="--", color=C_ACCENT, label="Target Coverage")
    ax1.set_xlabel("Nominal Confidence Level (%)")
    ax1.set_ylabel("Coverage Probability")
    ax1.set_ylim(0.75, 1.0)
    ax1.grid(True, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(conf, mpiw, marker="s", markersize=6, color=C_ORANGE, label="Mean Interval Width (MPIW)")
    ax2.set_ylabel("Mean Prediction Interval Width")
    ax2.set_ylim(0, max(mpiw) * 1.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True)

    ax1.set_title("Coverage–Width Tradeoff of Conformal Prediction")
    fig.tight_layout()

    save_path = os.path.join(fig_dir, "fig1_tradeoff_picp_mpiw.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)

# =========================
# 图2：90% prediction interval scatter
# =========================
def plot_interval_scatter(test_df, sample_n=500, random_state=42):
    # 为了图更漂亮，只抽样一部分
    df = test_df.copy()
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=random_state)

    df = df.sort_values("y_true")

    x = df["y_true"].values
    y = df["y_pred"].values
    yerr_lower = y - df["lower_90"].values
    yerr_upper = df["upper_90"].values - y

    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    ax.errorbar(
        x, y,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        markersize=3.5,
        color=C_SOFT,
        ecolor=(0.2, 0.4, 0.7, 0.25),
        elinewidth=0.8,
        capsize=1.5,
        alpha=0.8,
        label="Prediction with 90% Interval"
    )

    lim_min = min(df["y_true"].min(), df["lower_90"].min(), df["y_pred"].min())
    lim_max = max(df["y_true"].max(), df["upper_90"].max(), df["y_pred"].max())

    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            linestyle="--", color=C_ACCENT, linewidth=1.8, label="Ideal Line")

    ax.set_xlabel("True Power Loss")
    ax.set_ylabel("Predicted Power Loss")
    ax.set_title("Prediction Intervals on Test Samples (90% Confidence)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    fig.tight_layout()
    save_path = os.path.join(fig_dir, "fig2_prediction_interval_scatter_90.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)

# =========================
# 图3：Coverage by L-bin
# =========================
def plot_coverage_by_lbin(bin_df):
    df = bin_df[bin_df["bin_type"] == "L_true"].copy()
    df = df.dropna(subset=["coverage"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    bars = ax.bar(df["bin"], df["coverage"], color=C_MAIN, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.axhline(0.90, linestyle="--", color=C_ACCENT, linewidth=1.8, label="Target 90% Coverage")

    ax.set_xlabel("True Power Loss Bin")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Coverage across Power Loss Bins (90% Interval)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(frameon=True)

    # 加样本数标注
    for rect, count in zip(bars, df["count"]):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.015,
                f"{count}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=30)
    fig.tight_layout()

    save_path = os.path.join(fig_dir, "fig3_coverage_by_lbin_90.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)

# =========================
# 图4：Coverage by I-bin
# =========================
def plot_coverage_by_ibin(bin_df):
    df = bin_df[bin_df["bin_type"] == "I"].copy()
    df = df.dropna(subset=["coverage"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    bars = ax.bar(df["bin"], df["coverage"], color=C_GREEN, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.axhline(0.90, linestyle="--", color=C_ACCENT, linewidth=1.8, label="Target 90% Coverage")

    ax.set_xlabel("Irradiance Bin")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Coverage across Irradiance Bins (90% Interval)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(frameon=True)

    for rect, count in zip(bars, df["count"]):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.015,
                f"{count}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=30)
    fig.tight_layout()

    save_path = os.path.join(fig_dir, "fig4_coverage_by_ibin_90.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)

# =========================
# 图5：Absolute Error vs Interval Width
# =========================
def plot_error_vs_width(test_df):
    x = test_df["width_90"].values
    y = test_df["abs_error"].values

    fig, ax = plt.subplots(figsize=(6.5, 5.6))

    ax.scatter(x, y, s=16, alpha=0.28, color=C_MAIN, edgecolors="none")

    # 简单线性趋势线
    coef = np.polyfit(x, y, 1)
    xp = np.linspace(x.min(), x.max(), 200)
    yp = coef[0] * xp + coef[1]
    ax.plot(xp, yp, color=C_ACCENT, linewidth=2.2, label="Trend Line")

    # 相关系数
    corr = np.corrcoef(x, y)[0, 1]

    ax.set_xlabel("Prediction Interval Width (90%)")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Absolute Error vs Prediction Interval Width")
    ax.grid(True, linestyle="--")
    ax.legend(frameon=True)
    ax.text(
        0.97, 0.95,
        f"Pearson r = {corr:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )

    fig.tight_layout()
    save_path = os.path.join(fig_dir, "fig5_error_vs_width_90.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)

# =========================
# 可选图6：区间宽度分布
# =========================
def plot_width_distribution(test_df):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    ax.hist(test_df["width_90"].values, bins=35, color=C_SOFT, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Prediction Interval Width (90%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of 90% Interval Widths")
    ax.grid(True, axis="y", linestyle="--")

    fig.tight_layout()
    save_path = os.path.join(fig_dir, "fig6_width_distribution_90.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    plot_tradeoff(summary_df)
    plot_interval_scatter(test_df)
    plot_coverage_by_lbin(bin_df)
    plot_coverage_by_ibin(bin_df)
    plot_error_vs_width(test_df)
    plot_width_distribution(test_df)

    print("\nAll figures have been saved to:")
    print(fig_dir)