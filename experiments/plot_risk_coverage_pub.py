import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# Path configuration
# ============================================================
project_root = r"E:\PV_RL_Project"

mc_conf_pred_path = os.path.join(
    project_root, "outputs", "mc_conformal_resnet50_with_i", "mc_conformal_test_predictions.csv"
)

raw_mc_pred_path = os.path.join(
    project_root, "outputs", "mc_dropout_resnet50_with_i", "mc_test_predictions.csv"
)

split_conf_pred_path = os.path.join(
    project_root, "outputs", "conformal_resnet50_with_i", "test_conformal_predictions.csv"
)

out_dir = os.path.join(
    project_root, "outputs", "mc_conformal_resnet50_with_i", "risk_coverage_figures"
)
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# Load data
# ============================================================
mc_df = pd.read_csv(mc_conf_pred_path)
raw_df = pd.read_csv(raw_mc_pred_path)
split_df = pd.read_csv(split_conf_pred_path)

print("Loaded:")
print(mc_conf_pred_path)
print(raw_mc_pred_path)
print(split_conf_pred_path)

if len(mc_df) != len(raw_df) or len(mc_df) != len(split_df):
    raise ValueError(
        f"Length mismatch: MC+CP={len(mc_df)}, Raw MC={len(raw_df)}, Split CP={len(split_df)}"
    )


# ============================================================
# Build base dataframe
# ============================================================
df = pd.DataFrame()

df["y_true"] = mc_df["y_true"].values
df["pred_mean"] = mc_df["pred_mean"].values
df["abs_error"] = np.abs(df["y_true"] - df["pred_mean"])

# uncertainty scores
df["unc_raw_mc_std"] = raw_df["pred_std"].values
df["unc_split_cp_width"] = split_df["width_90"].values
df["unc_mc_cp_width"] = mc_df["width_mc_conf_90"].values

# sanity check
print("\nBasic statistics:")
print("N =", len(df))
print("Mean abs error =", df["abs_error"].mean())
print("Mean raw MC std =", df["unc_raw_mc_std"].mean())
print("Mean Split CP width =", df["unc_split_cp_width"].mean())
print("Mean MC+CP width =", df["unc_mc_cp_width"].mean())


# ============================================================
# Publication style
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.23,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Colors
# ============================================================
C_RAW = "#9e9e9e"
C_SPLIT = "#1f4e79"
C_MC_CP = "#2f855a"
C_ORACLE = "#b22222"
C_RANDOM = "#595959"
C_RED = "#b22222"


# ============================================================
# Save helper
# ============================================================
def save_fig(fig, filename):
    png_path = os.path.join(out_dir, filename + ".png")
    pdf_path = os.path.join(out_dir, filename + ".pdf")
    svg_path = os.path.join(out_dir, filename + ".svg")

    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", svg_path)


# ============================================================
# Risk-coverage calculation
# ============================================================
def compute_risk_curve(errors, uncertainty=None, rejection_grid=None,
                       mode="uncertainty", random_repeats=50, seed=42):
    """
    rejection ratio = fraction of most uncertain samples rejected.
    retained ratio = 1 - rejection ratio.

    mode:
        uncertainty: retain samples with lowest uncertainty
        oracle: retain samples with lowest true error
        random: randomly retain samples
    """
    rng = np.random.default_rng(seed)

    errors = np.asarray(errors)
    n = len(errors)

    if rejection_grid is None:
        rejection_grid = np.linspace(0.0, 0.80, 17)

    rows = []

    if mode == "uncertainty":
        uncertainty = np.asarray(uncertainty)
        order = np.argsort(uncertainty)  # low uncertainty first
    elif mode == "oracle":
        order = np.argsort(errors)       # low error first
    elif mode == "random":
        order = None
    else:
        raise ValueError(mode)

    for rejection in rejection_grid:
        retained_ratio = 1.0 - rejection
        k = max(1, int(round(n * retained_ratio)))

        if mode in ["uncertainty", "oracle"]:
            idx = order[:k]
            e = errors[idx]
            rmse = np.sqrt(np.mean(e ** 2))
            mae = np.mean(e)
        else:
            rmses = []
            maes = []
            for _ in range(random_repeats):
                idx = rng.choice(n, size=k, replace=False)
                e = errors[idx]
                rmses.append(np.sqrt(np.mean(e ** 2)))
                maes.append(np.mean(e))
            rmse = float(np.mean(rmses))
            mae = float(np.mean(maes))

        rows.append({
            "rejection_ratio": rejection,
            "retained_ratio": retained_ratio,
            "RMSE": rmse,
            "MAE": mae
        })

    return pd.DataFrame(rows)


def normalized_area_under_curve(curve_df, metric="RMSE"):
    x = curve_df["rejection_ratio"].values
    y = curve_df[metric].values

    # Compatible with NumPy 2.x and older NumPy versions
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(y, x)
    else:
        area = np.trapz(y, x)

    return area / (x.max() - x.min() + 1e-12)


def build_all_curves():
    rejection_grid = np.linspace(0.0, 0.80, 17)

    errors = df["abs_error"].values

    curves = {}

    curves["Oracle"] = compute_risk_curve(
        errors,
        rejection_grid=rejection_grid,
        mode="oracle"
    )

    curves["Random"] = compute_risk_curve(
        errors,
        rejection_grid=rejection_grid,
        mode="random",
        random_repeats=80
    )

    curves["Raw MC"] = compute_risk_curve(
        errors,
        uncertainty=df["unc_raw_mc_std"].values,
        rejection_grid=rejection_grid,
        mode="uncertainty"
    )

    curves["Split CP"] = compute_risk_curve(
        errors,
        uncertainty=df["unc_split_cp_width"].values,
        rejection_grid=rejection_grid,
        mode="uncertainty"
    )

    curves["MC+CP"] = compute_risk_curve(
        errors,
        uncertainty=df["unc_mc_cp_width"].values,
        rejection_grid=rejection_grid,
        mode="uncertainty"
    )

    return curves


# ============================================================
# Plot 1: RMSE risk-coverage curve
# ============================================================
def plot_rmse_curve(curves, ax=None, show_legend=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.7, 4.4))
    else:
        fig = ax.figure

    style = {
        "Oracle": dict(color=C_ORACLE, linestyle="--", marker="o"),
        "Random": dict(color=C_RANDOM, linestyle=":", marker="s"),
        "Raw MC": dict(color=C_RAW, linestyle="-", marker="^"),
        "Split CP": dict(color=C_SPLIT, linestyle="-", marker="D"),
        "MC+CP": dict(color=C_MC_CP, linestyle="-", marker="o"),
    }

    for name in ["Oracle", "Random", "Raw MC", "Split CP", "MC+CP"]:
        c = curves[name]
        ax.plot(
            c["rejection_ratio"],
            c["RMSE"],
            label=name,
            markersize=4.0,
            **style[name]
        )

    ax.set_xlabel("Rejected sample ratio")
    ax.set_ylabel("RMSE of retained samples")
    ax.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    if show_legend:
        ax.legend(frameon=True, borderpad=0.35, handlelength=1.8)

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig1_rmse_risk_coverage")


# ============================================================
# Plot 2: MAE risk-coverage curve
# ============================================================
def plot_mae_curve(curves, ax=None, show_legend=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.7, 4.4))
    else:
        fig = ax.figure

    style = {
        "Oracle": dict(color=C_ORACLE, linestyle="--", marker="o"),
        "Random": dict(color=C_RANDOM, linestyle=":", marker="s"),
        "Raw MC": dict(color=C_RAW, linestyle="-", marker="^"),
        "Split CP": dict(color=C_SPLIT, linestyle="-", marker="D"),
        "MC+CP": dict(color=C_MC_CP, linestyle="-", marker="o"),
    }

    for name in ["Oracle", "Random", "Raw MC", "Split CP", "MC+CP"]:
        c = curves[name]
        ax.plot(
            c["rejection_ratio"],
            c["MAE"],
            label=name,
            markersize=4.0,
            **style[name]
        )

    ax.set_xlabel("Rejected sample ratio")
    ax.set_ylabel("MAE of retained samples")
    ax.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    if show_legend:
        ax.legend(frameon=True, borderpad=0.35, handlelength=1.8)

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig2_mae_risk_coverage")


# ============================================================
# Plot 3: AURC / AUSE bar
# ============================================================
def compute_area_summary(curves):
    oracle_rmse_area = normalized_area_under_curve(curves["Oracle"], "RMSE")
    oracle_mae_area = normalized_area_under_curve(curves["Oracle"], "MAE")

    rows = []

    for name in ["Random", "Raw MC", "Split CP", "MC+CP"]:
        rmse_area = normalized_area_under_curve(curves[name], "RMSE")
        mae_area = normalized_area_under_curve(curves[name], "MAE")

        rows.append({
            "method": name,
            "AURC_RMSE": rmse_area,
            "AUSE_RMSE": rmse_area - oracle_rmse_area,
            "AURC_MAE": mae_area,
            "AUSE_MAE": mae_area - oracle_mae_area,
        })

    return pd.DataFrame(rows)


def plot_area_summary(area_df, ax=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.6, 4.1))
    else:
        fig = ax.figure

    methods = area_df["method"].tolist()
    vals = area_df["AUSE_RMSE"].values

    colors = []
    for m in methods:
        if m == "MC+CP":
            colors.append(C_MC_CP)
        elif m == "Split CP":
            colors.append(C_SPLIT)
        elif m == "Raw MC":
            colors.append(C_RAW)
        else:
            colors.append(C_RANDOM)

    bars = ax.bar(
        np.arange(len(methods)),
        vals,
        color=colors,
        edgecolor="black",
        linewidth=0.55,
        width=0.65
    )

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("AUSE based on RMSE")
    ax.grid(True, axis="y", linestyle="--")
    ax.set_axisbelow(True)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max(vals) * 0.025,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig3_ause_summary")


# ============================================================
# Plot 4: Accepted vs rejected error distribution
# ============================================================
def plot_accepted_rejected_box(ax=None, rejection=0.20):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
    else:
        fig = ax.figure

    uncertainty = df["unc_mc_cp_width"].values
    errors = df["abs_error"].values

    n = len(errors)
    k = int(round(n * (1.0 - rejection)))

    order = np.argsort(uncertainty)
    accepted_idx = order[:k]
    rejected_idx = order[k:]

    accepted_errors = errors[accepted_idx]
    rejected_errors = errors[rejected_idx]

    box = ax.boxplot(
        [accepted_errors, rejected_errors],
        tick_labels=[f"Accepted\n{1 - rejection:.0%}", f"Rejected\n{rejection:.0%}"],
        patch_artist=True,
        showfliers=False,
        widths=0.55
    )

    colors = [C_MC_CP, C_RED]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)

    ax.set_ylabel("Absolute error")
    ax.grid(True, axis="y", linestyle="--")
    ax.set_axisbelow(True)

    mean_acc = accepted_errors.mean()
    mean_rej = rejected_errors.mean()

    ax.text(
        0.02,
        0.96,
        f"Mean error: {mean_acc:.4f} vs {mean_rej:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2
    )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig4_accepted_rejected_error_box")


# ============================================================
# Main multipanel
# ============================================================
def plot_multipanel(curves, area_df):
    fig = plt.figure(figsize=(11.2, 8.4))
    gs = fig.add_gridspec(
        2, 2,
        wspace=0.27,
        hspace=0.32
    )

    ax1 = fig.add_subplot(gs[0, 0])
    plot_rmse_curve(curves, ax=ax1, show_legend=True)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_mae_curve(curves, ax=ax2, show_legend=False)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_area_summary(area_df, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_accepted_rejected_box(ax=ax4, rejection=0.20)

    labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label in zip([ax1, ax2, ax3, ax4], labels):
        ax.text(
            -0.15,
            1.07,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top"
        )

    png_path = os.path.join(out_dir, "fig5_risk_coverage_multipanel.png")
    pdf_path = os.path.join(out_dir, "fig5_risk_coverage_multipanel.pdf")
    svg_path = os.path.join(out_dir, "fig5_risk_coverage_multipanel.svg")

    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", svg_path)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    curves = build_all_curves()
    area_df = compute_area_summary(curves)

    # Save numerical results
    all_curve_rows = []
    for method, cdf in curves.items():
        temp = cdf.copy()
        temp["method"] = method
        all_curve_rows.append(temp)

    curves_df = pd.concat(all_curve_rows, ignore_index=True)

    curves_out = os.path.join(out_dir, "risk_coverage_curves.csv")
    area_out = os.path.join(out_dir, "risk_coverage_area_summary.csv")

    curves_df.to_csv(curves_out, index=False)
    area_df.to_csv(area_out, index=False)

    print("\nRisk-coverage area summary:")
    print(area_df)

    print("\nSaved CSV:")
    print(curves_out)
    print(area_out)

    # Save figures
    plot_rmse_curve(curves)
    plot_mae_curve(curves)
    plot_area_summary(area_df)
    plot_accepted_rejected_box(rejection=0.20)
    plot_multipanel(curves, area_df)

    print("\nAll risk-coverage figures saved to:")
    print(out_dir)