import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib import transforms


# ============================================================
# 1. 路径设置
# ============================================================
CALIB_CSV = r"E:\PV_RL_Project\cp_analysis\final_unified_results\mc_calibration_predictions.csv"
TEST_CSV = r"E:\PV_RL_Project\cp_analysis\final_unified_results\mc_test_predictions.csv"

OUT_DIR = r"E:\PV_RL_Project\paper_figures\fig3_split_cp"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. 论文风格参数
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.3,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.linewidth": 1.15,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 低饱和论文配色
COLOR_BLUE = "#4C78A8"
COLOR_LIGHT_BLUE = "#A9CBE8"
COLOR_ORANGE = "#F28E2B"
COLOR_RED = "#E45756"
COLOR_GRAY = "#6B7280"
COLOR_LIGHT_GRAY = "#D1D5DB"
COLOR_BLACK = "#111111"

COLOR_GOOD = COLOR_BLUE
COLOR_BAD = COLOR_ORANGE


# ============================================================
# 3. 工具函数
# ============================================================
def conformal_quantile(scores, alpha):
    """
    Split conformal finite-sample quantile.
    """
    scores = np.asarray(scores)
    n = len(scores)

    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    return np.quantile(scores, q_level, method="higher")


def calc_picp_mpiw(y_true, y_pred, q):
    lower = y_pred - q
    upper = y_pred + q
    covered = (y_true >= lower) & (y_true <= upper)

    picp = covered.mean()
    mpiw = np.mean(upper - lower)

    return picp, mpiw, lower, upper, covered


def add_panel_label(ax, label):
    """
    面板标签放在子图左上方，但不压住 y 轴刻度。
    """
    ax.text(
        0.0,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False
    )


def make_bin_table(df, value_col, covered_col, bins):
    tmp = df.copy()
    tmp["bin"] = pd.cut(tmp[value_col], bins=bins, include_lowest=True, right=True)

    rows = []
    for interval in tmp["bin"].cat.categories:
        sub = tmp[tmp["bin"] == interval]
        n = len(sub)

        if n == 0:
            cov = np.nan
        else:
            cov = sub[covered_col].mean()

        left = max(interval.left, 0.0)
        right = interval.right

        rows.append({
            "bin": interval,
            "coverage": cov,
            "n": n,
            "label": f"{left:.1f}–{right:.1f}"
        })

    return pd.DataFrame(rows)


def style_axis(ax, grid_axis="y"):
    ax.tick_params(direction="in", length=5, width=1.0)
    ax.tick_params(which="minor", direction="in", length=3, width=0.8)
    ax.grid(True, axis=grid_axis, linewidth=0.45, alpha=0.26)
    ax.set_axisbelow(True)


def add_target_label_outside(ax, y=0.90, text="90% target"):
    """
    把 90% target 文字放在坐标轴右侧外部，
    避免和柱子、图例重叠。
    """
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        1.025,
        y,
        text,
        transform=trans,
        ha="left",
        va="center",
        fontsize=9.2,
        color=COLOR_BLACK,
        clip_on=False
    )


# ============================================================
# 4. 读取数据
# ============================================================
calib = pd.read_csv(CALIB_CSV)
test = pd.read_csv(TEST_CSV)

required = ["y_true", "pred_mean", "I"]

for name, df in [("calib", calib), ("test", test)]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}")

cal_y = calib["y_true"].to_numpy()
cal_pred = calib["pred_mean"].to_numpy()

test_y = test["y_true"].to_numpy()
test_pred = test["pred_mean"].to_numpy()
test_I = test["I"].to_numpy()

cal_scores = np.abs(cal_y - cal_pred)


# ============================================================
# 5. 计算 Split CP
# ============================================================
levels = [0.80, 0.90, 0.95]
rows = []
interval_results = {}

for conf in levels:
    alpha = 1 - conf
    q = conformal_quantile(cal_scores, alpha)

    picp, mpiw, lower, upper, covered = calc_picp_mpiw(
        test_y,
        test_pred,
        q
    )

    rows.append({
        "confidence": conf,
        "alpha": alpha,
        "q": q,
        "PICP": picp,
        "MPIW": mpiw,
    })

    interval_results[conf] = {
        "lower": lower,
        "upper": upper,
        "covered": covered,
        "width": upper - lower,
        "q": q
    }

summary = pd.DataFrame(rows)

summary.to_csv(
    os.path.join(OUT_DIR, "split_cp_summary_for_fig3_final_clean.csv"),
    index=False,
    encoding="utf-8-sig"
)

q90 = summary.loc[summary["confidence"] == 0.90, "q"].values[0]
picp90 = summary.loc[summary["confidence"] == 0.90, "PICP"].values[0]
mpiw90 = summary.loc[summary["confidence"] == 0.90, "MPIW"].values[0]

plot_df = pd.DataFrame({
    "y_true": test_y,
    "pred_mean": test_pred,
    "I": test_I,
    "lower_90": interval_results[0.90]["lower"],
    "upper_90": interval_results[0.90]["upper"],
    "covered_90": interval_results[0.90]["covered"],
})


# ============================================================
# 6. 创建四联图
# ============================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.4, 8.8),
    constrained_layout=False
)

ax_a, ax_b, ax_c, ax_d = axes.flatten()

# 手动控制间距，比 constrained_layout 更可控
plt.subplots_adjust(
    left=0.075,
    right=0.955,
    top=0.925,
    bottom=0.115,
    wspace=0.28,
    hspace=0.34
)


# ============================================================
# (a) Overall PICP and MPIW
# ============================================================
x = np.arange(len(summary))
labels = [f"{int(c * 100)}%" for c in summary["confidence"]]

ax_a.bar(
    x,
    summary["PICP"],
    width=0.56,
    color=COLOR_BLUE,
    edgecolor=COLOR_BLACK,
    linewidth=0.8,
    alpha=0.90,
    label="PICP"
)

ax_a.plot(
    x,
    summary["confidence"],
    linestyle="--",
    marker="o",
    color=COLOR_BLACK,
    linewidth=1.45,
    markersize=4.8,
    label="Nominal"
)

for i, row in summary.iterrows():
    ax_a.text(
        i,
        row["PICP"] + 0.006,
        f"{row['PICP']:.3f}",
        ha="center",
        va="bottom",
        fontsize=9.5
    )

ax_a.set_ylim(0.70, 1.00)
ax_a.set_ylabel("Coverage")
ax_a.set_xlabel("Nominal confidence level")
ax_a.set_xticks(x)
ax_a.set_xticklabels(labels)
ax_a.yaxis.set_major_formatter(PercentFormatter(1.0))
ax_a.yaxis.set_major_locator(MultipleLocator(0.05))
style_axis(ax_a)

ax_a2 = ax_a.twinx()
ax_a2.plot(
    x,
    summary["MPIW"],
    marker="s",
    color=COLOR_RED,
    linewidth=1.7,
    markersize=4.8,
    label="MPIW"
)
ax_a2.set_ylabel("Mean interval width")
ax_a2.set_ylim(0.00, max(summary["MPIW"]) * 1.28)
ax_a2.tick_params(direction="in", length=5, width=1.0)

h1, l1 = ax_a.get_legend_handles_labels()
h2, l2 = ax_a2.get_legend_handles_labels()

ax_a.legend(
    h1 + h2,
    l1 + l2,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.045, 0.84)
)

add_panel_label(ax_a, "(a)")


# ============================================================
# (b) 90% interval examples
# 解决点：
# 1. 按预测值排序，区间更平滑
# 2. (b) 标签放图外，不与蓝色区间重合
# ============================================================
n_show = 90

sorted_idx = np.argsort(plot_df["pred_mean"].to_numpy())
select_idx = sorted_idx[
    np.linspace(0, len(sorted_idx) - 1, n_show).astype(int)
]

sub = plot_df.iloc[select_idx].copy().reset_index(drop=True)
x_sub = np.arange(len(sub))

lower_vis = np.clip(sub["lower_90"].to_numpy(), 0, 1)
upper_vis = np.clip(sub["upper_90"].to_numpy(), 0, 1)

ax_b.fill_between(
    x_sub,
    lower_vis,
    upper_vis,
    color=COLOR_LIGHT_BLUE,
    alpha=0.55,
    linewidth=0,
    label="90% interval"
)

ax_b.plot(
    x_sub,
    sub["pred_mean"],
    color=COLOR_BLUE,
    linewidth=1.45,
    label="Prediction"
)

ax_b.scatter(
    x_sub,
    sub["y_true"],
    s=13,
    color=COLOR_BLACK,
    marker="x",
    linewidth=0.75,
    label="True value",
    zorder=4
)

ax_b.set_ylim(-0.03, 1.03)
ax_b.set_xlabel("Selected test samples sorted by predicted loss")
ax_b.set_ylabel("Power loss")
style_axis(ax_b, grid_axis="both")

# 图例稍微向右移，避免靠近面板标签
ax_b.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.055, 0.995),
    borderaxespad=0.0
)

ax_b.text(
    0.97,
    0.06,
    f"90% Split CP\nPICP = {picp90:.3f}\nq = {q90:.4f}",
    transform=ax_b.transAxes,
    ha="right",
    va="bottom",
    fontsize=9.5,
    bbox=dict(
        boxstyle="round,pad=0.32",
        facecolor="white",
        edgecolor=COLOR_BLACK,
        alpha=0.92,
        linewidth=0.8
    )
)

add_panel_label(ax_b, "(b)")


# ============================================================
# (c) Coverage by true power loss bins
# 解决点：
# 1. 不再显示柱顶竖排 n
# 2. 90% target 文字放在坐标轴外侧
# 3. 低于 90% 的柱子用橙色突出
# ============================================================
bins_L = np.linspace(0, 1, 11)
cov_L = make_bin_table(plot_df, "y_true", "covered_90", bins_L)

cov_L.to_csv(
    os.path.join(OUT_DIR, "split_cp_coverage_by_true_loss_bins.csv"),
    index=False,
    encoding="utf-8-sig"
)

xL = np.arange(len(cov_L))
heights_L = cov_L["coverage"].to_numpy()

colors_L = []
for cov in heights_L:
    if np.isnan(cov):
        colors_L.append(COLOR_LIGHT_GRAY)
    elif cov < 0.90:
        colors_L.append(COLOR_BAD)
    else:
        colors_L.append(COLOR_GOOD)

plot_heights_L = np.where(np.isnan(heights_L), 0.0, heights_L)

ax_c.bar(
    xL,
    plot_heights_L,
    color=colors_L,
    edgecolor=COLOR_BLACK,
    linewidth=0.75,
    alpha=0.92
)

# 空 bin 标 NA
for i, row in cov_L.iterrows():
    if row["n"] == 0:
        ax_c.text(
            i,
            0.615,
            "NA",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLOR_GRAY
        )

ax_c.axhline(
    0.90,
    linestyle="--",
    color=COLOR_BLACK,
    linewidth=1.25
)

# add_target_label_outside(ax_c, y=0.90, text="90% target")

ax_c.set_ylim(0.58, 1.05)
ax_c.set_ylabel("Empirical coverage")
ax_c.set_xlabel("True power loss bin")
ax_c.set_xticks(xL)
ax_c.set_xticklabels(cov_L["label"], rotation=35, ha="right")
ax_c.yaxis.set_major_formatter(PercentFormatter(1.0))
ax_c.yaxis.set_major_locator(MultipleLocator(0.10))
style_axis(ax_c)

add_panel_label(ax_c, "(c)")


# ============================================================
# (d) Coverage by irradiance bins
# 解决点：
# 1. 不再显示柱顶竖排 n
# 2. 90% target 文字放在坐标轴外侧
# ============================================================
bins_I = np.linspace(0, 1, 6)
cov_I = make_bin_table(plot_df, "I", "covered_90", bins_I)

cov_I.to_csv(
    os.path.join(OUT_DIR, "split_cp_coverage_by_irradiance_bins.csv"),
    index=False,
    encoding="utf-8-sig"
)

xI = np.arange(len(cov_I))
heights_I = cov_I["coverage"].to_numpy()

colors_I = [
    COLOR_BAD if cov < 0.90 else COLOR_GOOD
    for cov in heights_I
]

ax_d.bar(
    xI,
    heights_I,
    color=colors_I,
    edgecolor=COLOR_BLACK,
    linewidth=0.75,
    alpha=0.92
)

ax_d.axhline(
    0.90,
    linestyle="--",
    color=COLOR_BLACK,
    linewidth=1.25
)

# add_target_label_outside(ax_d, y=0.90, text="90% target")

ax_d.set_ylim(0.58, 1.05)
ax_d.set_ylabel("Empirical coverage")
ax_d.set_xlabel("Irradiance bin")
ax_d.set_xticks(xI)
ax_d.set_xticklabels(cov_I["label"], rotation=25, ha="right")
ax_d.yaxis.set_major_formatter(PercentFormatter(1.0))
ax_d.yaxis.set_major_locator(MultipleLocator(0.10))
style_axis(ax_d)

add_panel_label(ax_d, "(d)")


# ============================================================
# 7. 保存
# ============================================================
png_path = os.path.join(OUT_DIR, "fig3_split_cp_four_panel_final_clean.png")
pdf_path = os.path.join(OUT_DIR, "fig3_split_cp_four_panel_final_clean.pdf")
svg_path = os.path.join(OUT_DIR, "fig3_split_cp_four_panel_final_clean.svg")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")
plt.show()

print("\nSaved:")
print(png_path)
print(pdf_path)
print(svg_path)

print("\n90% Split CP:")
print(f"q90   = {q90:.6f}")
print(f"PICP  = {picp90:.6f}")
print(f"MPIW  = {mpiw90:.6f}")

print("\nCoverage tables saved:")
print(os.path.join(OUT_DIR, "split_cp_coverage_by_true_loss_bins.csv"))
print(os.path.join(OUT_DIR, "split_cp_coverage_by_irradiance_bins.csv"))