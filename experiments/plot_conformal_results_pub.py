import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 路径配置
# =========================
project_root = r"E:\PV_RL_Project"
result_dir = os.path.join(project_root, "outputs", "conformal_resnet50_with_i")

raw_fig_dir = os.path.join(result_dir, "figures_raw")
pub_fig_dir = os.path.join(result_dir, "figures_pub")
os.makedirs(raw_fig_dir, exist_ok=True)
os.makedirs(pub_fig_dir, exist_ok=True)

summary_path = os.path.join(result_dir, "conformal_summary.csv")
test_path = os.path.join(result_dir, "test_conformal_predictions.csv")

summary_df = pd.read_csv(summary_path)
test_df = pd.read_csv(test_path)

print("Loaded:")
print(summary_path)
print(test_path)

# =========================
# 全局绘图风格（IEEE / Elsevier / Energy 风格）
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# 配色（克制、稳重）
# =========================
C_BLUE = "#1f4e79"
C_LIGHT_BLUE = "#4c78a8"
C_RED = "#d62728"
C_GREEN = "#2b8a3e"
C_ORANGE = "#d97706"
C_GRAY = "#666666"
C_LIGHT_GRAY = "#d9d9d9"
C_HIGHLIGHT = "#b22222"

# =========================
# 公共保存函数
# =========================
def save_fig(fig, filename):
    png_path = os.path.join(pub_fig_dir, filename + ".png")
    pdf_path = os.path.join(pub_fig_dir, filename + ".pdf")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", png_path)
    print("Saved:", pdf_path)

# =========================
# 图1：Coverage-Width Tradeoff
# =========================
def plot_tradeoff(summary_df):
    conf = summary_df["confidence"].values
    target = conf / 100.0
    picp = summary_df["PICP"].values
    mpiw = summary_df["MPIW"].values

    fig, ax1 = plt.subplots(figsize=(5.8, 4.2))

    ax1.plot(conf, picp, marker="o", markersize=5.5, color=C_BLUE, label="Empirical coverage")
    ax1.plot(conf, target, linestyle="--", color=C_RED, label="Target coverage")
    ax1.set_xlabel("Nominal confidence level (%)")
    ax1.set_ylabel("Coverage probability")
    ax1.set_xticks([80, 90, 95])
    ax1.set_ylim(0.78, 0.98)
    ax1.grid(True, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(conf, mpiw, marker="s", markersize=5, color=C_ORANGE, label="Mean interval width")
    ax2.set_ylabel("Mean prediction interval width")
    ax2.set_ylim(0.06, max(mpiw) * 1.15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True)

    fig.tight_layout()
    save_fig(fig, "fig1_tradeoff_pub")

# =========================
# 图2：90% Prediction Interval Scatter
# =========================
def plot_interval_scatter(test_df, sample_n=300, random_state=42):
    df = test_df.copy()

    # 分层抽样：低、中、高 L 区域都保留
    bins = pd.cut(df["y_true"], bins=[0, 0.2, 0.5, 1.0], include_lowest=True)
    sampled_list = []
    for _, group in df.groupby(bins):
        if len(group) > sample_n // 3:
            sampled_list.append(group.sample(sample_n // 3, random_state=random_state))
        else:
            sampled_list.append(group)
    df = pd.concat(sampled_list).drop_duplicates().sort_values("y_true")

    x = df["y_true"].values
    y = df["y_pred"].values
    yerr_lower = y - df["lower_90"].values
    yerr_upper = df["upper_90"].values - y

    fig, ax = plt.subplots(figsize=(5.6, 5.2))

    ax.errorbar(
        x, y,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        markersize=2.8,
        color=C_LIGHT_BLUE,
        ecolor=(76/255, 120/255, 168/255, 0.18),
        elinewidth=0.6,
        capsize=1,
        alpha=0.85,
        label="Predictions with 90% interval"
    )

    lim_min = 0.0
    lim_max = 1.0
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            linestyle="--", color=C_RED, linewidth=1.5, label="Ideal line")

    ax.set_xlabel("True power loss")
    ax.set_ylabel("Predicted power loss")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(True, linestyle="--")
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    save_fig(fig, "fig2_interval_scatter_90_pub")

# =========================
# 重新计算 coverage by bins（修正版）
# =========================
def coverage_by_bins(bin_values, y_true, lower, upper, bins):
    covered = (y_true >= lower) & (y_true <= upper)
    rows = []

    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]

        if i == len(bins) - 2:
            mask = (bin_values >= left) & (bin_values <= right)
        else:
            mask = (bin_values >= left) & (bin_values < right)

        if mask.sum() == 0:
            coverage = np.nan
            count = 0
        else:
            coverage = covered[mask].mean()
            count = int(mask.sum())

        rows.append({
            "bin": f"{left:.2f}–{right:.2f}",
            "coverage": coverage,
            "count": count
        })

    return pd.DataFrame(rows)

# =========================
# 图3：Coverage by L-bin
# =========================
def plot_coverage_by_lbin(test_df):
    bins = np.linspace(0, 1, 11)
    df = coverage_by_bins(
        bin_values=test_df["y_true"].values,
        y_true=test_df["y_true"].values,
        lower=test_df["lower_90"].values,
        upper=test_df["upper_90"].values,
        bins=bins
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    colors = []
    for cov in df["coverage"]:
        if np.isnan(cov):
            colors.append(C_LIGHT_GRAY)
        elif cov < 0.90:
            colors.append(C_HIGHLIGHT)
        else:
            colors.append(C_BLUE)

    bars = ax.bar(df["bin"], df["coverage"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.90, linestyle="--", color=C_RED, linewidth=1.5, label="Target 90% coverage")

    ax.set_xlabel("True power loss bin")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(loc="upper right", frameon=True)

    for rect, count in zip(bars, df["count"]):
        if count > 0:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.02,
                    f"{count}",
                    ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=25)
    fig.tight_layout()
    save_fig(fig, "fig3_coverage_by_lbin_pub")

# =========================
# 图4：Coverage by I-bin（修正版）
# =========================
def plot_coverage_by_ibin(test_df):
    I_values = test_df["I"].values
    bins = np.linspace(I_values.min(), I_values.max(), 6)

    df = coverage_by_bins(
        bin_values=I_values,
        y_true=test_df["y_true"].values,
        lower=test_df["lower_90"].values,
        upper=test_df["upper_90"].values,
        bins=bins
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    colors = []
    for cov in df["coverage"]:
        if np.isnan(cov):
            colors.append(C_LIGHT_GRAY)
        elif cov < 0.90:
            colors.append(C_GREEN)
        else:
            colors.append(C_BLUE)

    bars = ax.bar(df["bin"], df["coverage"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.90, linestyle="--", color=C_RED, linewidth=1.5, label="Target 90% coverage")

    ax.set_xlabel("Irradiance bin")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(loc="upper right", frameon=True)

    for rect, count in zip(bars, df["count"]):
        if count > 0:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.02,
                    f"{count}",
                    ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=25)
    fig.tight_layout()
    save_fig(fig, "fig4_coverage_by_ibin_pub")

# =========================
# 图5：Absolute Error vs Interval Width
# =========================
def plot_error_vs_width(test_df):
    x = test_df["width_90"].values
    y = test_df["abs_error"].values

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(x, y, s=8, alpha=0.22, color=C_LIGHT_BLUE, edgecolors="none")

    coef = np.polyfit(x, y, 1)
    xp = np.linspace(x.min(), x.max(), 200)
    yp = coef[0] * xp + coef[1]
    ax.plot(xp, yp, color=C_RED, linewidth=1.8, label="Trend")

    corr = np.corrcoef(x, y)[0, 1]

    ax.set_xlabel("Prediction interval width (90%)")
    ax.set_ylabel("Absolute error")
    ax.grid(True, linestyle="--")
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.97, 0.95,
        f"Pearson r = {corr:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9)
    )

    fig.tight_layout()
    save_fig(fig, "fig5_error_vs_width_pub")

# =========================
# 图6：Distribution of Interval Widths
# =========================
def plot_width_distribution(test_df):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))

    ax.hist(test_df["width_90"].values, bins=35,
            color=C_LIGHT_BLUE, alpha=0.85,
            edgecolor="black", linewidth=0.4)

    ax.set_xlabel("Prediction interval width (90%)")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", linestyle="--")

    fig.tight_layout()
    save_fig(fig, "fig6_width_distribution_pub")

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    plot_tradeoff(summary_df)
    plot_interval_scatter(test_df)
    plot_coverage_by_lbin(test_df)
    plot_coverage_by_ibin(test_df)
    plot_error_vs_width(test_df)
    plot_width_distribution(test_df)

    print("\nAll publication-style figures have been saved to:")
    print(pub_fig_dir)