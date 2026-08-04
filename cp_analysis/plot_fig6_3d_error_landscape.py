import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 1. 路径设置
# ============================================================
INPUT_CSV = r"E:\PV_RL_Project\cp_analysis\final_unified_results\mc_test_predictions.csv"
OUT_DIR = r"E:\PV_RL_Project\paper_figures\fig6_3d_error_landscape"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. 论文风格参数
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 1.1,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 3. 可调参数
# ============================================================
X_BINS = 18
Y_BINS = 12
MIN_COUNT = 10
SMOOTH_SIGMA = 1.05


# ============================================================
# 4. 工具函数
# ============================================================
def fill_nan_by_interpolation(Z):
    Z_df = pd.DataFrame(Z)
    Z_interp = Z_df.interpolate(axis=0, limit_direction="both")
    Z_interp = Z_interp.interpolate(axis=1, limit_direction="both")
    return Z_interp.values


def smooth_grid(Z, sigma=1.05):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(Z, sigma=sigma)
    except Exception:
        print("未检测到 scipy，跳过平滑。")
        return Z


def make_binned_surface(df, x_col, y_col, z_col, x_bins=18, y_bins=12, min_count=10):
    data = df[[x_col, y_col, z_col]].dropna().copy()

    x_edges = np.linspace(0, 1, x_bins + 1)
    y_edges = np.linspace(0, 1, y_bins + 1)

    data["x_bin"] = pd.cut(data[x_col], bins=x_edges, include_lowest=True)
    data["y_bin"] = pd.cut(data[y_col], bins=y_edges, include_lowest=True)

    grouped = data.groupby(["x_bin", "y_bin"], observed=False).agg(
        z_mean=(z_col, "mean"),
        count=(z_col, "count")
    ).reset_index()

    grouped["x_center"] = grouped["x_bin"].apply(lambda x: (x.left + x.right) / 2)
    grouped["y_center"] = grouped["y_bin"].apply(lambda x: (x.left + x.right) / 2)

    grouped.loc[grouped["count"] < min_count, "z_mean"] = np.nan

    pivot_z = grouped.pivot(index="y_center", columns="x_center", values="z_mean")
    pivot_n = grouped.pivot(index="y_center", columns="x_center", values="count")

    X, Y = np.meshgrid(
        pivot_z.columns.values.astype(float),
        pivot_z.index.values.astype(float)
    )

    Z = pivot_z.values.astype(float)
    N = pivot_n.values.astype(float)

    return X, Y, Z, N, grouped


def save_count_heatmap(X, Y, N, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    im = ax.imshow(
        N,
        origin="lower",
        aspect="auto",
        extent=[0, 1, 0, 1],
        cmap="Blues"
    )

    ax.set_xlabel(r"True power loss, $L$")
    ax.set_ylabel(r"Irradiance, $I$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sample count")

    ax.tick_params(direction="in", length=5, width=1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. 读取数据
# ============================================================
df = pd.read_csv(INPUT_CSV)

required_cols = ["y_true", "pred_mean", "I"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少列：{missing}")

df["abs_error"] = np.abs(df["pred_mean"] - df["y_true"])

print("测试集样本数:", len(df))
print("整体 MAE:", df["abs_error"].mean())


# ============================================================
# 6. 构造三维曲面数据
# ============================================================
X, Y, Z_raw, N, grouped = make_binned_surface(
    df,
    x_col="y_true",
    y_col="I",
    z_col="abs_error",
    x_bins=X_BINS,
    y_bins=Y_BINS,
    min_count=MIN_COUNT
)

grouped.to_csv(
    os.path.join(OUT_DIR, "fig6_error_landscape_binned_statistics_smooth.csv"),
    index=False,
    encoding="utf-8-sig"
)

save_count_heatmap(
    X,
    Y,
    N,
    os.path.join(OUT_DIR, "fig6_error_landscape_bin_counts_smooth.png")
)

Z_plot = fill_nan_by_interpolation(Z_raw)
Z_plot = smooth_grid(Z_plot, sigma=SMOOTH_SIGMA)
Z_plot = np.clip(Z_plot, 0, None)

print("有效网格数量:", np.sum(~np.isnan(Z_raw)))
print("最大分箱 MAE:", np.nanmax(Z_raw))
print("最小分箱 MAE:", np.nanmin(Z_raw))


# ============================================================
# 7. 绘制 3D error landscape
# ============================================================
fig = plt.figure(figsize=(8.2, 6.2))
ax = fig.add_subplot(111, projection="3d")

z_min = np.nanmin(Z_plot)
z_max = np.nanmax(Z_plot)

z_offset = z_min - 0.12 * (z_max - z_min + 1e-12)

surf = ax.plot_surface(
    X,
    Y,
    Z_plot,
    cmap="viridis",
    linewidth=0,
    edgecolor="none",
    antialiased=True,
    alpha=0.97,
    rstride=1,
    cstride=1
)

ax.contour(
    X,
    Y,
    Z_plot,
    zdir="z",
    offset=z_offset,
    levels=12,
    cmap="viridis",
    linewidths=0.85
)

ax.set_xlabel(r"True power loss, $L$", labelpad=10)
ax.set_ylabel(r"Irradiance, $I$", labelpad=10)
ax.set_zlabel("Mean absolute error", labelpad=10)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(z_offset, z_max * 1.10)

# 这个视角比上一版更紧凑
ax.view_init(elev=31, azim=-132)

ax.xaxis.pane.set_alpha(0.0)
ax.yaxis.pane.set_alpha(0.0)
ax.zaxis.pane.set_alpha(0.0)

ax.grid(True)

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis._axinfo["grid"]["linewidth"] = 0.45
    axis._axinfo["grid"]["linestyle"] = "-"
    axis._axinfo["grid"]["color"] = (0.65, 0.65, 0.65, 0.35)

cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.68,
    pad=0.075,
    aspect=18
)
cbar.set_label("Mean absolute error", labelpad=8)

plt.tight_layout()

png_path = os.path.join(OUT_DIR, "fig6_3d_error_landscape_smooth.png")
pdf_path = os.path.join(OUT_DIR, "fig6_3d_error_landscape_smooth.pdf")
svg_path = os.path.join(OUT_DIR, "fig6_3d_error_landscape_smooth.svg")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")

plt.show()

print("\nSaved:")
print(png_path)
print(pdf_path)
print(svg_path)
print(os.path.join(OUT_DIR, "fig6_error_landscape_bin_counts_smooth.png"))