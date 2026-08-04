import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# =========================
# 1. 路径设置
# =========================
INPUT_CSV = r"E:\PV_RL_Project\cp_analysis\mc_test_predictions.csv"
OUT_DIR = r"E:\PV_RL_Project\paper_figures\fig2_pred_true"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(INPUT_CSV)

# 请确保列名正确
y_true = df["y_true"].to_numpy()
y_pred = df["pred_mean"].to_numpy()

# =========================
# 3. 计算指标
# =========================
n = len(df)
mae = np.mean(np.abs(y_pred - y_true))
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

# R^2
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot

# =========================
# 4. 期刊风格参数
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
})

# =========================
# 5. 作图
# =========================
fig, ax = plt.subplots(figsize=(8.6, 8.0))

# hexbin 密度图
hb = ax.hexbin(
    y_true,
    y_pred,
    gridsize=55,
    cmap="viridis",
    bins="log",
    mincnt=1,
    linewidths=0.0
)

# 理想对角线
ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    linewidth=1.8,
    color="black",
    label="Ideal line"
)

# 坐标范围
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)

# 轴标签
ax.set_xlabel(r"True power loss, $L$")
ax.set_ylabel(r"Predicted power loss, $\hat{L}$")

# 刻度
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# 网格
ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
ax.grid(False, which="minor")

# 刻度朝内
ax.tick_params(direction="in", length=6, width=1.2)
ax.tick_params(which="minor", direction="in", length=3, width=1.0)

# 长宽比接近1:1，更适合这种图
ax.set_aspect("equal", adjustable="box")

# 统计信息框
textstr = (
    f"N = {n}\n"
    f"MAE = {mae:.4f}\n"
    f"RMSE = {rmse:.4f}\n"
    f"$R^2$ = {r2:.4f}"
)

ax.text(
    0.05, 0.95, textstr,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="left",
    fontsize=16,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black",
        alpha=0.9,
        linewidth=1.2
    )
)

# 图例
# ax.legend(loc="lower right", frameon=True, edgecolor="black")

# 颜色条
cbar = fig.colorbar(hb, ax=ax, pad=0.02)
cbar.set_label("Sample density (log scale)", fontsize=16)
cbar.ax.tick_params(labelsize=13)

# 去标题（论文正文里通常靠图注解释，不一定需要标题）
# ax.set_title("Predicted vs True Power Loss")

plt.tight_layout()

# =========================
# 6. 保存
# =========================
png_path = os.path.join(OUT_DIR, "fig2_predicted_vs_true_hexbin.png")
pdf_path = os.path.join(OUT_DIR, "fig2_predicted_vs_true_hexbin.pdf")
svg_path = os.path.join(OUT_DIR, "fig2_predicted_vs_true_hexbin.svg")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")

plt.show()

print("图已保存到：")
print(png_path)
print(pdf_path)
print(svg_path)