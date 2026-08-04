import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# =========================
# Global style
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.1
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.dpi"] = 150


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_metrics(df, y_col="y_true", pred_col="pred_mean"):
    y = df[y_col].values
    p = df[pred_col].values
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mae, rmse, r2


def bin_surface(df, x_col, y_col, z_col, x_bins=12, y_bins=12, min_count=5):
    """
    Create binned 2D surface:
    X axis: x_col bin centers
    Y axis: y_col bin centers
    Z axis: mean z_col in each bin
    """
    data = df[[x_col, y_col, z_col]].dropna().copy()

    x_edges = np.linspace(data[x_col].min(), data[x_col].max(), x_bins + 1)
    y_edges = np.linspace(data[y_col].min(), data[y_col].max(), y_bins + 1)

    data["x_bin"] = pd.cut(data[x_col], bins=x_edges, include_lowest=True)
    data["y_bin"] = pd.cut(data[y_col], bins=y_edges, include_lowest=True)

    grouped = data.groupby(["x_bin", "y_bin"], observed=True).agg(
        z_mean=(z_col, "mean"),
        count=(z_col, "count")
    ).reset_index()

    grouped = grouped[grouped["count"] >= min_count]

    x_centers = np.array([(iv.left + iv.right) / 2 for iv in grouped["x_bin"]])
    y_centers = np.array([(iv.left + iv.right) / 2 for iv in grouped["y_bin"]])

    grouped["x_center"] = x_centers
    grouped["y_center"] = y_centers

    pivot = grouped.pivot(index="y_center", columns="x_center", values="z_mean")

    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values

    return X, Y, Z


def plot_3d_surface(
    X,
    Y,
    Z,
    xlabel,
    ylabel,
    zlabel,
    out_prefix,
    cmap="viridis",
    elev=28,
    azim=-135
):
    """
    Paper-style 3D surface with contour projection.
    """
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    # Mask invalid values
    Z_masked = np.ma.masked_invalid(Z)

    surf = ax.plot_surface(
        X, Y, Z_masked,
        cmap=cmap,
        linewidth=0.25,
        edgecolor="none",
        antialiased=True,
        alpha=0.96
    )

    # bottom contour projection
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    z_offset = z_min - 0.08 * (z_max - z_min + 1e-12)

    ax.contour(
        X, Y, Z_masked,
        zdir="z",
        offset=z_offset,
        levels=10,
        cmap=cmap,
        linewidths=0.8
    )

    ax.set_zlim(z_offset, z_max)

    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_zlabel(zlabel, labelpad=8)

    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    ax.grid(True, linewidth=0.3, alpha=0.35)

    cb = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    cb.set_label(zlabel)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()


def plot_prediction_hexbin(df, out_prefix):
    """
    High-density predicted vs true plot.
    """
    mae, rmse, r2 = calc_metrics(df)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    hb = ax.hexbin(
        df["y_true"],
        df["pred_mean"],
        gridsize=65,
        mincnt=1,
        cmap="viridis",
        bins="log"
    )

    lim_min = min(df["y_true"].min(), df["pred_mean"].min())
    lim_max = max(df["y_true"].max(), df["pred_mean"].max())

    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            linestyle="--", linewidth=1.3, color="black")

    ax.set_xlabel("True power loss")
    ax.set_ylabel("Predicted power loss")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    text = (
        f"N = {len(df)}\n"
        f"MAE = {mae:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"$R^2$ = {r2:.4f}"
    )
    ax.text(
        0.04, 0.96, text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.6)
    )

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")

    ax.grid(True, linewidth=0.3, alpha=0.35)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()


def plot_error_by_bins(df, out_prefix):
    """
    Two-panel bar plot:
    (a) error by true power loss bins
    (b) error by irradiance bins
    """
    data = df.copy()
    data["abs_error"] = np.abs(data["pred_mean"] - data["y_true"])

    data["L_bin"] = pd.cut(data["y_true"], bins=np.linspace(0, 1, 11), include_lowest=True)
    data["I_bin"] = pd.cut(data["I"], bins=np.linspace(0, 1, 6), include_lowest=True)

    gL = data.groupby("L_bin", observed=True).agg(
        MAE=("abs_error", "mean"),
        n=("abs_error", "count")
    ).reset_index()

    gI = data.groupby("I_bin", observed=True).agg(
        MAE=("abs_error", "mean"),
        n=("abs_error", "count")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].bar(range(len(gL)), gL["MAE"])
    axes[0].set_xticks(range(len(gL)))
    axes[0].set_xticklabels([str(x) for x in gL["L_bin"]], rotation=45, ha="right")
    axes[0].set_xlabel("True power loss bin")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("(a) Error by true power loss")

    axes[1].bar(range(len(gI)), gI["MAE"])
    axes[1].set_xticks(range(len(gI)))
    axes[1].set_xticklabels([str(x) for x in gI["I_bin"]], rotation=45, ha="right")
    axes[1].set_xlabel("Irradiance bin")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("(b) Error by irradiance")

    for ax in axes:
        ax.grid(axis="y", linewidth=0.3, alpha=0.35)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()


def main():
    out_dir = "advanced_figures"
    ensure_dir(out_dir)

    # =========================
    # 1. Load unified point prediction CSV
    # =========================
    test_path = "mc_test_predictions.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError("找不到 mc_test_predictions.csv，请确认脚本放在 final_unified_results 文件夹中。")

    df = pd.read_csv(test_path)

    required = ["y_true", "pred_mean", "pred_std", "I"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"mc_test_predictions.csv 缺少列: {missing}")

    df["abs_error"] = np.abs(df["pred_mean"] - df["y_true"])

    # Fig. 1: Predicted vs true density
    plot_prediction_hexbin(
        df,
        os.path.join(out_dir, "fig2_predicted_vs_true_hexbin_unified")
    )

    # Fig. 2: Error by bins
    plot_error_by_bins(
        df,
        os.path.join(out_dir, "fig3_error_by_L_and_I_bins")
    )

    # Fig. 3D-1: Error landscape over true_L and I
    X, Y, Z = bin_surface(
        df,
        x_col="y_true",
        y_col="I",
        z_col="abs_error",
        x_bins=12,
        y_bins=10,
        min_count=8
    )
    plot_3d_surface(
        X, Y, Z,
        xlabel="True power loss",
        ylabel="Irradiance",
        zlabel="Mean absolute error",
        out_prefix=os.path.join(out_dir, "fig3D_error_landscape_trueL_I"),
        cmap="viridis",
        elev=30,
        azim=-135
    )

    # Fig. 3D-2: MC uncertainty and prediction error landscape
    X, Y, Z = bin_surface(
        df,
        x_col="pred_mean",
        y_col="pred_std",
        z_col="abs_error",
        x_bins=12,
        y_bins=10,
        min_count=8
    )
    plot_3d_surface(
        X, Y, Z,
        xlabel="Predicted power loss",
        ylabel="MC standard deviation",
        zlabel="Mean absolute error",
        out_prefix=os.path.join(out_dir, "fig3D_uncertainty_error_landscape"),
        cmap="plasma",
        elev=30,
        azim=-130
    )

    # =========================
    # 2. Load final interval predictions
    # =========================
    interval_candidates = [
        "final_test_predictions_all_methods.csv",
        "pred_l_mondrian_mc_cp_test_predictions.csv"
    ]

    interval_path = None
    for p in interval_candidates:
        if os.path.exists(p):
            interval_path = p
            break

    if interval_path is None:
        print("未找到 final_test_predictions_all_methods.csv 或 pred_l_mondrian_mc_cp_test_predictions.csv，跳过区间宽度三维图。")
    else:
        idf = pd.read_csv(interval_path)

        # Normalize possible column names
        if "y_true" in idf.columns and "true_L" not in idf.columns:
            idf = idf.rename(columns={"y_true": "true_L"})
        if "pred_mean" in idf.columns and "pred_L" not in idf.columns:
            idf = idf.rename(columns={"pred_mean": "pred_L"})

        # Add irradiance if not present
        if "I" not in idf.columns and "I" in df.columns and len(idf) == len(df):
            idf["I"] = df["I"].values

        width_col = "pred_l_mondrian_std_mc_width"
        needed = ["pred_L", "I", width_col]
        missing = [c for c in needed if c not in idf.columns]

        if missing:
            print(f"区间文件缺少列 {missing}，跳过区间宽度三维图。")
            print("当前列名为：", list(idf.columns))
        else:
            X, Y, Z = bin_surface(
                idf,
                x_col="pred_L",
                y_col="I",
                z_col=width_col,
                x_bins=12,
                y_bins=10,
                min_count=8
            )
            plot_3d_surface(
                X, Y, Z,
                xlabel="Predicted power loss",
                ylabel="Irradiance",
                zlabel="Interval width",
                out_prefix=os.path.join(out_dir, "fig3D_interval_width_landscape_std_mc_cp"),
                cmap="cividis",
                elev=30,
                azim=-135
            )

    print("\n高级图已生成到文件夹：advanced_figures")
    print("建议优先查看：")
    print("1. fig2_predicted_vs_true_hexbin_unified.png")
    print("2. fig3D_error_landscape_trueL_I.png")
    print("3. fig3D_uncertainty_error_landscape.png")
    print("4. fig3D_interval_width_landscape_std_mc_cp.png")


if __name__ == "__main__":
    main()