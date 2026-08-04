import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 1. 路径设置
# ============================================================
OUT_DIR = r"E:\PV_RL_Project\paper_figures\fig7_3d_interval_width_landscape"
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATE_FILES = [
    r"E:\PV_RL_Project\cp_analysis\final_unified_results\final_test_predictions_all_methods.csv",
    r"E:\PV_RL_Project\cp_analysis\final_unified_results\pred_l_mondrian_mc_cp_test_predictions.csv",
    r"E:\PV_RL_Project\cp_analysis\final_test_predictions_all_methods.csv",
    r"E:\PV_RL_Project\cp_analysis\pred_l_mondrian_mc_cp_test_predictions.csv",
]

MC_TEST_CSV = r"E:\PV_RL_Project\cp_analysis\final_unified_results\mc_test_predictions.csv"


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
SMOOTH_SIGMA = 1.20

# 为了防止极少数异常网格主导色条，使用稳健显示上限
# 仅影响可视化，不影响保存的原始统计表
USE_ROBUST_COLOR_LIMIT = True
ROBUST_UPPER_Q = 0.98


# ============================================================
# 4. 工具函数
# ============================================================
def find_existing_file(paths):
    for p in paths:
        if os.path.exists(p):
            print(f"找到区间预测文件：{p}")
            return p
    return None


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fuzzy_find_column(df, must_have_tokens, optional_tokens=None):
    """
    模糊查找列名。
    must_have_tokens: 必须同时包含的关键词
    optional_tokens: 可选关键词，匹配越多优先级越高
    """
    optional_tokens = optional_tokens or []
    cols = list(df.columns)

    candidates = []
    for col in cols:
        low = col.lower()
        if all(t.lower() in low for t in must_have_tokens):
            score = sum(t.lower() in low for t in optional_tokens)
            candidates.append((score, col))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def infer_std_mc_width_column(df):
    """
    尝试寻找 Pred-L-Mondrian Std-MC CP 的区间宽度列。
    """
    width_candidates = [
        "pred_l_mondrian_std_mc_width",
        "pred_l_mondrian_std_width",
        "std_mc_width",
        "p_stdmc_width",
        "pstdmc_width",
        "width_std_mc",
        "width_pred_l_mondrian_std_mc",
    ]

    col = find_column(df, width_candidates)
    if col is not None:
        return col

    # 模糊查找 width 列
    fuzzy_col = fuzzy_find_column(
        df,
        must_have_tokens=["width"],
        optional_tokens=["std", "mc", "mondrian", "pred"]
    )
    if fuzzy_col is not None:
        print(f"通过模糊匹配找到宽度列：{fuzzy_col}")
        return fuzzy_col

    # 如果没有 width，尝试通过 lower / upper 自动计算
    lower_candidates = [
        "pred_l_mondrian_std_mc_lower",
        "lower_pred_l_mondrian_std_mc",
        "p_stdmc_lower",
        "pstdmc_lower",
        "std_mc_lower",
        "lower_std_mc",
    ]

    upper_candidates = [
        "pred_l_mondrian_std_mc_upper",
        "upper_pred_l_mondrian_std_mc",
        "p_stdmc_upper",
        "pstdmc_upper",
        "std_mc_upper",
        "upper_std_mc",
    ]

    lower_col = find_column(df, lower_candidates)
    upper_col = find_column(df, upper_candidates)

    if lower_col is None:
        lower_col = fuzzy_find_column(
            df,
            must_have_tokens=["lower"],
            optional_tokens=["std", "mc", "mondrian", "pred"]
        )

    if upper_col is None:
        upper_col = fuzzy_find_column(
            df,
            must_have_tokens=["upper"],
            optional_tokens=["std", "mc", "mondrian", "pred"]
        )

    if lower_col is not None and upper_col is not None:
        df["pred_l_mondrian_std_mc_width_auto"] = df[upper_col] - df[lower_col]
        print(f"未找到 width 列，已由 {upper_col} - {lower_col} 自动计算。")
        return "pred_l_mondrian_std_mc_width_auto"

    return None


def fill_nan_by_interpolation(Z):
    """
    对 NaN 网格做二维插值，避免曲面断裂。
    仅用于可视化。
    """
    Z_df = pd.DataFrame(Z)
    Z_interp = Z_df.interpolate(axis=0, limit_direction="both")
    Z_interp = Z_interp.interpolate(axis=1, limit_direction="both")
    return Z_interp.values


def smooth_grid(Z, sigma=1.20):
    """
    轻微平滑曲面。
    如果没有 scipy，则自动跳过。
    """
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(Z, sigma=sigma)
    except Exception:
        print("未检测到 scipy，跳过平滑。")
        return Z


def make_binned_surface(df, x_col, y_col, z_col, x_bins=18, y_bins=12, min_count=10):
    """
    构造二维分箱曲面。
    注意：正式绘图使用 median interval width，而不是 mean interval width。
    """
    data = df[[x_col, y_col, z_col]].dropna().copy()

    data = data[(data[x_col] >= 0) & (data[x_col] <= 1)]
    data = data[(data[y_col] >= 0) & (data[y_col] <= 1)]
    data = data[(data[z_col] >= 0)]

    x_edges = np.linspace(0, 1, x_bins + 1)
    y_edges = np.linspace(0, 1, y_bins + 1)

    data["x_bin"] = pd.cut(data[x_col], bins=x_edges, include_lowest=True)
    data["y_bin"] = pd.cut(data[y_col], bins=y_edges, include_lowest=True)

    grouped = data.groupby(["x_bin", "y_bin"], observed=False).agg(
        width_mean=(z_col, "mean"),
        width_median=(z_col, "median"),
        width_q25=(z_col, lambda x: np.quantile(x, 0.25)),
        width_q75=(z_col, lambda x: np.quantile(x, 0.75)),
        width_max=(z_col, "max"),
        count=(z_col, "count")
    ).reset_index()

    grouped["x_center"] = grouped["x_bin"].apply(lambda x: (x.left + x.right) / 2)
    grouped["y_center"] = grouped["y_bin"].apply(lambda x: (x.left + x.right) / 2)

    # 样本不足的网格设为 NaN
    grouped.loc[grouped["count"] < min_count, "width_median"] = np.nan

    # 关键：使用 median，而不是 mean
    pivot_z = grouped.pivot(
        index="y_center",
        columns="x_center",
        values="width_median"
    )

    pivot_n = grouped.pivot(
        index="y_center",
        columns="x_center",
        values="count"
    )

    X, Y = np.meshgrid(
        pivot_z.columns.values.astype(float),
        pivot_z.index.values.astype(float)
    )

    Z = pivot_z.values.astype(float)
    N = pivot_n.values.astype(float)

    return X, Y, Z, N, grouped


def save_count_heatmap(X, Y, N, out_path):
    """
    样本数热力图，仅用于自查或附录。
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    im = ax.imshow(
        N,
        origin="lower",
        aspect="auto",
        extent=[0, 1, 0, 1],
        cmap="Blues"
    )

    ax.set_xlabel(r"Predicted power loss, $\hat{L}$")
    ax.set_ylabel(r"Irradiance, $I$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sample count")

    ax.tick_params(direction="in", length=5, width=1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def save_2d_width_heatmap(X, Y, Z_plot, out_path):
    """
    额外保存二维 interval width 热力图。
    这张可作为附录，也方便检查 3D 图是否合理。
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    im = ax.imshow(
        Z_plot,
        origin="lower",
        aspect="auto",
        extent=[0, 1, 0, 1],
        cmap="viridis"
    )

    ax.set_xlabel(r"Predicted power loss, $\hat{L}$")
    ax.set_ylabel(r"Irradiance, $I$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median interval width")

    ax.tick_params(direction="in", length=5, width=1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. 读取区间预测文件
# ============================================================
interval_csv = find_existing_file(CANDIDATE_FILES)

if interval_csv is None:
    raise FileNotFoundError(
        "没有找到最终区间预测文件。请检查 final_test_predictions_all_methods.csv "
        "或 pred_l_mondrian_mc_cp_test_predictions.csv 是否存在。"
    )

df = pd.read_csv(interval_csv)

print("\n当前文件列名：")
print(list(df.columns))


# ============================================================
# 6. 统一列名
# ============================================================
pred_col = find_column(df, ["pred_L", "pred_mean", "y_pred", "prediction"])
if pred_col is None:
    pred_col = fuzzy_find_column(
        df,
        must_have_tokens=["pred"],
        optional_tokens=["mean", "loss", "l"]
    )

if pred_col is None:
    raise ValueError("找不到预测列，请确认文件中有 pred_L 或 pred_mean。")

irr_col = find_column(df, ["I", "irradiance", "Irradiance"])

if irr_col is None:
    if os.path.exists(MC_TEST_CSV):
        mc_df = pd.read_csv(MC_TEST_CSV)
        if "I" in mc_df.columns and len(mc_df) == len(df):
            df["I"] = mc_df["I"].values
            irr_col = "I"
            print("区间文件中没有 I，已从 mc_test_predictions.csv 补充。")
        else:
            raise ValueError("找不到 I 列，且 mc_test_predictions.csv 无法补充。")
    else:
        raise ValueError("找不到 I 列，请确认文件中有 I 或 irradiance。")

width_col = infer_std_mc_width_column(df)

if width_col is None:
    raise ValueError(
        "找不到 Pred-L-Mondrian Std-MC CP 的宽度列。\n"
        "请检查是否存在 pred_l_mondrian_std_mc_width，或 lower/upper 上下界列。\n"
        f"当前列名为：{list(df.columns)}"
    )

print("\n用于绘图的列：")
print("X:", pred_col)
print("Y:", irr_col)
print("Z:", width_col)


# ============================================================
# 7. 整理绘图数据
# ============================================================
plot_df = pd.DataFrame({
    "pred_L": df[pred_col],
    "I": df[irr_col],
    "interval_width": df[width_col],
}).dropna()

plot_df = plot_df[
    (plot_df["pred_L"] >= 0) &
    (plot_df["pred_L"] <= 1) &
    (plot_df["I"] >= 0) &
    (plot_df["I"] <= 1) &
    (plot_df["interval_width"] >= 0)
].copy()

print("\n区间宽度统计：")
print(plot_df["interval_width"].describe())

plot_df.to_csv(
    os.path.join(OUT_DIR, "fig7_interval_width_raw_data_for_plot.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 构造 median interval width 曲面
# ============================================================
X, Y, Z_raw, N, grouped = make_binned_surface(
    plot_df,
    x_col="pred_L",
    y_col="I",
    z_col="interval_width",
    x_bins=X_BINS,
    y_bins=Y_BINS,
    min_count=MIN_COUNT
)

grouped.to_csv(
    os.path.join(OUT_DIR, "fig7_interval_width_binned_statistics_median.csv"),
    index=False,
    encoding="utf-8-sig"
)

save_count_heatmap(
    X,
    Y,
    N,
    os.path.join(OUT_DIR, "fig7_interval_width_bin_counts_median.png")
)

Z_plot = fill_nan_by_interpolation(Z_raw)
Z_plot = smooth_grid(Z_plot, sigma=SMOOTH_SIGMA)
Z_plot = np.clip(Z_plot, 0, None)

# 稳健显示上限，防止极个别网格主导色条
if USE_ROBUST_COLOR_LIMIT:
    valid = Z_plot[np.isfinite(Z_plot)]
    z_upper = np.quantile(valid, ROBUST_UPPER_Q)
    Z_plot_display = np.clip(Z_plot, 0, z_upper)
else:
    Z_plot_display = Z_plot.copy()

save_2d_width_heatmap(
    X,
    Y,
    Z_plot_display,
    os.path.join(OUT_DIR, "fig7_interval_width_2d_heatmap_median.png")
)

print("\n分箱统计：")
print("有效网格数量:", np.sum(~np.isnan(Z_raw)))
print("最大分箱 median width:", np.nanmax(Z_raw))
print("最小分箱 median width:", np.nanmin(Z_raw))
print("显示用最大 width:", np.nanmax(Z_plot_display))


# ============================================================
# 9. 绘制 3D median interval width landscape
# ============================================================
fig = plt.figure(figsize=(8.0, 6.1))
ax = fig.add_subplot(111, projection="3d")

z_min = np.nanmin(Z_plot_display)
z_max = np.nanmax(Z_plot_display)

z_offset = z_min - 0.10 * (z_max - z_min + 1e-12)

surf = ax.plot_surface(
    X,
    Y,
    Z_plot_display,
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
    Z_plot_display,
    zdir="z",
    offset=z_offset,
    levels=12,
    cmap="viridis",
    linewidths=0.85
)

ax.set_xlabel(r"Predicted power loss, $\hat{L}$", labelpad=10)
ax.set_ylabel(r"Irradiance, $I$", labelpad=10)
ax.set_zlabel("Median interval width", labelpad=10)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(z_offset, z_max * 1.10)

ax.view_init(elev=29, azim=-130)

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
cbar.set_label("Median interval width", labelpad=8)

plt.tight_layout()

png_path = os.path.join(OUT_DIR, "fig7_3d_interval_width_landscape_median.png")
pdf_path = os.path.join(OUT_DIR, "fig7_3d_interval_width_landscape_median.pdf")
svg_path = os.path.join(OUT_DIR, "fig7_3d_interval_width_landscape_median.svg")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")

plt.show()

print("\nSaved:")
print(png_path)
print(pdf_path)
print(svg_path)
print(os.path.join(OUT_DIR, "fig7_interval_width_bin_counts_median.png"))
print(os.path.join(OUT_DIR, "fig7_interval_width_2d_heatmap_median.png"))