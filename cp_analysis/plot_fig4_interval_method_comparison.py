import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch


# ============================================================
# 1. 输出路径
# ============================================================
OUT_DIR = r"E:\PV_RL_Project\paper_figures\fig4_interval_comparison"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. 最终统一指标
# ============================================================
data = [
    {
        "method": "Raw MC",
        "short_name": "Raw\nMC",
        "PICP": 0.5590034965,
        "MPIW": 0.0480095643,
        "Median_width": 0.0488330230,
        "Spearman": 0.3083050828,
    },
    {
        "method": "Split CP",
        "short_name": "Split\nCP",
        "PICP": 0.9038461538,
        "MPIW": 0.1425909005,
        "Median_width": 0.1600073260,
        "Spearman": 0.1536770744,
    },
    {
        "method": "Pred-L-Mondrian CP",
        "short_name": "P-M\nCP",
        "PICP": 0.9087995338,
        "MPIW": 0.1429719598,
        "Median_width": 0.1442243825,
        "Spearman": 0.3152217353,
    },
    {
        "method": "Pred-L-Mondrian MC-Interval CP",
        "short_name": "P-MCInt\nCP",
        "PICP": 0.9095279720,
        "MPIW": 0.1384755069,
        "Median_width": 0.1385221412,
        "Spearman": 0.3500684125,
    },
    {
        "method": "Pred-L-Mondrian Std-MC CP",
        "short_name": "P-StdMC\nCP",
        "PICP": 0.9042832168,
        "MPIW": 0.1386212757,
        "Median_width": 0.1221266551,
        "Spearman": 0.3643061705,
    },
]

df = pd.DataFrame(data)
df.to_csv(os.path.join(OUT_DIR, "fig4_interval_method_comparison_data.csv"),
          index=False, encoding="utf-8-sig")


# ============================================================
# 3. 论文风格参数
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 12.5,
    "axes.titlesize": 13,
    "xtick.labelsize": 9.6,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 1.15,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 4. 配色
# ============================================================
COLOR_RAW = "#9CA3AF"       # 灰色：未校准
COLOR_BASE = "#4C78A8"      # 蓝色：基线
COLOR_IMPROVE = "#72B7B2"   # 青色：改进方法
COLOR_MAIN = "#E45756"      # 红色：最终主方法
COLOR_TARGET = "#111111"

colors = [
    COLOR_RAW,
    COLOR_BASE,
    COLOR_IMPROVE,
    COLOR_IMPROVE,
    COLOR_MAIN
]


# ============================================================
# 5. 工具函数
# ============================================================
def add_panel_label(ax, label):
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


def style_axis(ax):
    ax.tick_params(direction="in", length=5, width=1.0)
    ax.grid(True, axis="y", linewidth=0.45, alpha=0.28)
    ax.set_axisbelow(True)


def add_value_labels(ax, x, y, fmt="{:.3f}", y_offset=0.008, fontsize=8.8):
    for xi, yi in zip(x, y):
        ax.text(
            xi,
            yi + y_offset,
            fmt.format(yi),
            ha="center",
            va="bottom",
            fontsize=fontsize
        )


# ============================================================
# 6. 作图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

plt.subplots_adjust(
    left=0.065,
    right=0.985,
    top=0.88,
    bottom=0.24,
    wspace=0.30
)

x = np.arange(len(df))
xticklabels = df["short_name"].tolist()


# ------------------------------------------------------------
# (a) PICP@90
# ------------------------------------------------------------
ax = axes[0]

ax.bar(
    x,
    df["PICP"],
    color=colors,
    edgecolor="black",
    linewidth=0.75,
    alpha=0.92
)

ax.axhline(
    0.90,
    color=COLOR_TARGET,
    linestyle="--",
    linewidth=1.25
)

ax.set_ylabel("PICP@90")
ax.set_ylim(0.50, 0.96)
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style_axis(ax)
add_value_labels(ax, x, df["PICP"], fmt="{:.3f}", y_offset=0.008)
add_panel_label(ax, "(a)")

# 在虚线旁边标注目标，不放图例，避免拥挤
# ax.text(
#   4.55,
#   0.902,
#   "90% target",
#   ha="right",
#   va="bottom",
#   fontsize=9,
#   color=COLOR_TARGET
# )


# ------------------------------------------------------------
# (b) MPIW@90
# ------------------------------------------------------------
ax = axes[1]

ax.bar(
    x,
    df["MPIW"],
    color=colors,
    edgecolor="black",
    linewidth=0.75,
    alpha=0.92
)

ax.set_ylabel("MPIW@90")
ax.set_ylim(0.00, 0.165)
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
style_axis(ax)
add_value_labels(ax, x, df["MPIW"], fmt="{:.3f}", y_offset=0.004)
add_panel_label(ax, "(b)")

# 标注越低越好
# ax.text(
#    0.98,
#    0.94,
#    "lower is better",
#    transform=ax.transAxes,
#    ha="right",
#    va="top",
#    fontsize=9,
#    color="#374151"
# )


# ------------------------------------------------------------
# (c) Spearman(width, error)
# ------------------------------------------------------------
ax = axes[2]

ax.bar(
    x,
    df["Spearman"],
    color=colors,
    edgecolor="black",
    linewidth=0.75,
    alpha=0.92
)

ax.set_ylabel("Spearman correlation")
ax.set_ylim(0.00, 0.42)
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
style_axis(ax)
add_value_labels(ax, x, df["Spearman"], fmt="{:.3f}", y_offset=0.008)
add_panel_label(ax, "(c)")

# 标注越高越好
# ax.text(
#    0.98,
#    0.94,
#    "higher is better",
#   transform=ax.transAxes,
#    ha="right",
#    va="top",
#    fontsize=9,
#    color="#374151"
# )


# ============================================================
# 7. 总图例
# ============================================================
legend_handles = [
    Patch(facecolor=COLOR_RAW, edgecolor="black", label="Uncalibrated MC"),
    Patch(facecolor=COLOR_BASE, edgecolor="black", label="Split CP baseline"),
    Patch(facecolor=COLOR_IMPROVE, edgecolor="black", label="Prediction-stratified CP"),
    Patch(facecolor=COLOR_MAIN, edgecolor="black", label="Proposed method"),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.53, 1.03),
    ncol=4,
    frameon=False,
    fontsize=10
)


# ============================================================
# 8. 保存
# ============================================================
png_path = os.path.join(OUT_DIR, "fig4_interval_method_comparison.png")
pdf_path = os.path.join(OUT_DIR, "fig4_interval_method_comparison.pdf")
svg_path = os.path.join(OUT_DIR, "fig4_interval_method_comparison.svg")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")
plt.show()

print("\nSaved:")
print(png_path)
print(pdf_path)
print(svg_path)

print("\nData:")
print(df)