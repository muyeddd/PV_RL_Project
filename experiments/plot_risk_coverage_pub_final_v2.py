import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. Path configuration
# ============================================================
PROJECT_ROOT = r"E:\PV_RL_Project"

MC_CONF_PRED_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_conformal_resnet50_with_i",
    "mc_conformal_test_predictions.csv"
)

RAW_MC_PRED_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_dropout_resnet50_with_i",
    "mc_test_predictions.csv"
)

SPLIT_CONF_PRED_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "conformal_resnet50_with_i",
    "test_conformal_predictions.csv"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_conformal_resnet50_with_i",
    "risk_coverage_figures_final_v2"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Publication style
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 11,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.85,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.28,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 2. Colors
# ============================================================
C_ORACLE = "#b22222"
C_RANDOM = "#595959"
C_RAW = "#9e9e9e"
C_SPLIT = "#1f4e79"
C_MC_CP = "#2f855a"
C_ACCEPT = "#5ca37a"
C_REJECT = "#c85a5a"
C_GRID = "#d9d9d9"
C_BLACK = "#222222"


# ============================================================
# 3. Load data
# ============================================================
def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


check_file(MC_CONF_PRED_PATH)
check_file(RAW_MC_PRED_PATH)
check_file(SPLIT_CONF_PRED_PATH)

mc_df = pd.read_csv(MC_CONF_PRED_PATH)
raw_df = pd.read_csv(RAW_MC_PRED_PATH)
split_df = pd.read_csv(SPLIT_CONF_PRED_PATH)

print("Loaded:")
print(MC_CONF_PRED_PATH)
print(RAW_MC_PRED_PATH)
print(SPLIT_CONF_PRED_PATH)

if len(mc_df) != len(raw_df) or len(mc_df) != len(split_df):
    raise ValueError(
        f"Length mismatch: MC+CP={len(mc_df)}, Raw MC={len(raw_df)}, Split CP={len(split_df)}"
    )


# ============================================================
# 4. Column check
# ============================================================
required_mc_cols = ["y_true", "pred_mean", "width_mc_conf_90"]
required_raw_cols = ["pred_std"]
required_split_cols = ["width_90"]

for col in required_mc_cols:
    if col not in mc_df.columns:
        raise KeyError(f"Missing column in MC+CP file: {col}. Available: {list(mc_df.columns)}")

for col in required_raw_cols:
    if col not in raw_df.columns:
        raise KeyError(f"Missing column in Raw MC file: {col}. Available: {list(raw_df.columns)}")

for col in required_split_cols:
    if col not in split_df.columns:
        raise KeyError(f"Missing column in Split CP file: {col}. Available: {list(split_df.columns)}")


# ============================================================
# 5. Build base dataframe
# ============================================================
df = pd.DataFrame()
df["y_true"] = mc_df["y_true"].values
df["pred_mean"] = mc_df["pred_mean"].values
df["abs_error"] = np.abs(df["y_true"] - df["pred_mean"])

df["unc_raw_mc_std"] = raw_df["pred_std"].values
df["unc_split_cp_width"] = split_df["width_90"].values
df["unc_mc_cp_width"] = mc_df["width_mc_conf_90"].values

print("\nBasic statistics:")
print("N =", len(df))
print("Mean abs error =", df["abs_error"].mean())
print("Mean raw MC std =", df["unc_raw_mc_std"].mean())
print("Mean Split CP width =", df["unc_split_cp_width"].mean())
print("Mean MC+CP width =", df["unc_mc_cp_width"].mean())


# ============================================================
# 6. Save helper
# ============================================================
def save_fig(fig, name):
    png = os.path.join(OUT_DIR, name + ".png")
    pdf = os.path.join(OUT_DIR, name + ".pdf")
    svg = os.path.join(OUT_DIR, name + ".svg")

    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)
    print("Saved:", svg)


def style_axis(ax):
    ax.grid(True, linestyle="--", color=C_GRID)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(C_BLACK)

    ax.tick_params(axis="both", direction="out", length=3.5, width=0.8, colors=C_BLACK)


def trapz_compatible(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# ============================================================
# 7. Risk-coverage calculation
# ============================================================
def compute_risk_curve(
    errors,
    uncertainty=None,
    rejection_grid=None,
    mode="uncertainty",
    random_repeats=100,
    seed=42
):
    """
    rejection_ratio: rejected fraction of samples.
    retained_ratio: retained fraction of samples.

    mode:
        uncertainty: retain samples with the lowest uncertainty
        oracle: retain samples with the lowest true error
        random: retain random samples
    """
    rng = np.random.default_rng(seed)
    errors = np.asarray(errors)
    n = len(errors)

    if rejection_grid is None:
        rejection_grid = np.linspace(0.0, 0.80, 17)

    if mode == "uncertainty":
        uncertainty = np.asarray(uncertainty)
        order = np.argsort(uncertainty)
    elif mode == "oracle":
        order = np.argsort(errors)
    elif mode == "random":
        order = None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rows = []

    for rejection in rejection_grid:
        retained_ratio = 1.0 - rejection
        k = max(1, int(round(n * retained_ratio)))

        if mode in ["uncertainty", "oracle"]:
            idx = order[:k]
            e = errors[idx]
            rmse = float(np.sqrt(np.mean(e ** 2)))
            mae = float(np.mean(e))
        else:
            rmses, maes = [], []
            for _ in range(random_repeats):
                idx = rng.choice(n, size=k, replace=False)
                e = errors[idx]
                rmses.append(np.sqrt(np.mean(e ** 2)))
                maes.append(np.mean(e))
            rmse = float(np.mean(rmses))
            mae = float(np.mean(maes))

        rows.append({
            "rejection_ratio": float(rejection),
            "retained_ratio": float(retained_ratio),
            "RMSE": rmse,
            "MAE": mae
        })

    return pd.DataFrame(rows)


def build_all_curves():
    rejection_grid = np.linspace(0.0, 0.80, 17)
    errors = df["abs_error"].values

    curves = {
        "Oracle": compute_risk_curve(
            errors,
            rejection_grid=rejection_grid,
            mode="oracle"
        ),
        "Random": compute_risk_curve(
            errors,
            rejection_grid=rejection_grid,
            mode="random",
            random_repeats=120
        ),
        "Raw MC": compute_risk_curve(
            errors,
            uncertainty=df["unc_raw_mc_std"].values,
            rejection_grid=rejection_grid,
            mode="uncertainty"
        ),
        "Split CP": compute_risk_curve(
            errors,
            uncertainty=df["unc_split_cp_width"].values,
            rejection_grid=rejection_grid,
            mode="uncertainty"
        ),
        "MC+CP": compute_risk_curve(
            errors,
            uncertainty=df["unc_mc_cp_width"].values,
            rejection_grid=rejection_grid,
            mode="uncertainty"
        )
    }

    return curves


def normalized_area_under_curve(curve_df, metric="RMSE"):
    x = curve_df["rejection_ratio"].values
    y = curve_df[metric].values
    area = trapz_compatible(y, x)
    return area / (x.max() - x.min() + 1e-12)


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


# ============================================================
# 8. Plot settings
# ============================================================
METHOD_ORDER = ["Oracle", "Random", "Raw MC", "Split CP", "MC+CP"]

METHOD_STYLE = {
    "Oracle": {
        "color": C_ORACLE,
        "linestyle": "--",
        "marker": "o",
        "linewidth": 1.65,
        "markersize": 4.8,
        "alpha": 0.95
    },
    "Random": {
        "color": C_RANDOM,
        "linestyle": ":",
        "marker": "s",
        "linewidth": 1.75,
        "markersize": 4.4,
        "alpha": 0.95
    },
    "Raw MC": {
        "color": C_RAW,
        "linestyle": "-",
        "marker": "^",
        "linewidth": 1.75,
        "markersize": 4.8,
        "alpha": 0.95
    },
    "Split CP": {
        "color": C_SPLIT,
        "linestyle": "-",
        "marker": "D",
        "linewidth": 1.95,
        "markersize": 4.7,
        "alpha": 0.98
    },
    "MC+CP": {
        "color": C_MC_CP,
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.05,
        "markersize": 4.8,
        "alpha": 0.98
    },
}


# ============================================================
# 9. Figure 1: RMSE risk-coverage curve
# ============================================================
def plot_rmse_curve(curves, ax=None, show_legend=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.9, 4.45))
    else:
        fig = ax.figure

    for name in METHOD_ORDER:
        c = curves[name]
        ax.plot(
            c["rejection_ratio"],
            c["RMSE"],
            label=name,
            **METHOD_STYLE[name]
        )

    ax.set_xlabel("Rejection ratio")
    ax.set_ylabel("Retained RMSE")
    ax.set_xlim(-0.015, 0.815)

    ymax = max(curves["Random"]["RMSE"].max(), curves["Split CP"]["RMSE"].max())
    ax.set_ylim(0.0, ymax * 1.08)

    style_axis(ax)

    if show_legend:
        ax.legend(
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#bfbfbf",
            framealpha=0.90,
            borderpad=0.35,
            labelspacing=0.35,
            handlelength=2.1
        )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig1_rmse_risk_coverage_final_v2")


# ============================================================
# 10. Figure 2: MAE risk-coverage curve
# ============================================================
def plot_mae_curve(curves, ax=None, show_legend=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.9, 4.45))
    else:
        fig = ax.figure

    for name in METHOD_ORDER:
        c = curves[name]
        ax.plot(
            c["rejection_ratio"],
            c["MAE"],
            label=name,
            **METHOD_STYLE[name]
        )

    ax.set_xlabel("Rejection ratio")
    ax.set_ylabel("Retained MAE")
    ax.set_xlim(-0.015, 0.815)

    ymax = max(curves["Random"]["MAE"].max(), curves["Split CP"]["MAE"].max())
    ax.set_ylim(0.0, ymax * 1.08)

    style_axis(ax)

    if show_legend:
        ax.legend(
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#bfbfbf",
            framealpha=0.90,
            borderpad=0.35,
            labelspacing=0.35,
            handlelength=2.1
        )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig2_mae_risk_coverage_final_v2")


# ============================================================
# 11. Figure 3: AUSE-RMSE bar
# ============================================================
def plot_ause_bar(area_df, ax=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
    else:
        fig = ax.figure

    order = ["Random", "Split CP", "Raw MC", "MC+CP"]
    plot_df = area_df.set_index("method").loc[order].reset_index()

    methods = plot_df["method"].tolist()
    vals = plot_df["AUSE_RMSE"].values

    color_map = {
        "Random": C_RANDOM,
        "Split CP": C_SPLIT,
        "Raw MC": C_RAW,
        "MC+CP": C_MC_CP,
    }
    colors = [color_map[m] for m in methods]

    bars = ax.bar(
        np.arange(len(methods)),
        vals,
        color=colors,
        edgecolor=C_BLACK,
        linewidth=0.65,
        width=0.58
    )

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("AUSE-RMSE")
    ax.set_ylim(0, max(vals) * 1.22)

    style_axis(ax)
    ax.grid(True, axis="y", linestyle="--", color=C_GRID)
    ax.grid(False, axis="x")

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max(vals) * 0.035,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.6
        )

    ax.text(
        0.98,
        0.94,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color=C_BLACK
    )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig3_ause_rmse_bar_final_v2")


# ============================================================
# 12. Figure 4: Accepted vs rejected boxplot
# ============================================================
def get_accepted_rejected_errors(rejection=0.20):
    uncertainty = df["unc_mc_cp_width"].values
    errors = df["abs_error"].values

    n = len(errors)
    k = int(round(n * (1.0 - rejection)))

    order = np.argsort(uncertainty)
    accepted_idx = order[:k]
    rejected_idx = order[k:]

    accepted_errors = errors[accepted_idx]
    rejected_errors = errors[rejected_idx]

    return accepted_errors, rejected_errors


def safe_boxplot(ax, data, tick_labels, **kwargs):
    try:
        return ax.boxplot(data, tick_labels=tick_labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=tick_labels, **kwargs)


def plot_accepted_rejected_box(ax=None, rejection=0.20):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.35, 4.25))
    else:
        fig = ax.figure

    accepted_errors, rejected_errors = get_accepted_rejected_errors(rejection=rejection)

    data = [accepted_errors, rejected_errors]
    labels = [f"Accepted\n{1-rejection:.0%}", f"Rejected\n{rejection:.0%}"]

    box = safe_boxplot(
        ax,
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.50,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": C_BLACK,
            "markersize": 4.6,
            "linestyle": "none"
        },
        medianprops={
            "color": C_BLACK,
            "linewidth": 1.45
        },
        whiskerprops={
            "color": C_BLACK,
            "linewidth": 1.05
        },
        capprops={
            "color": C_BLACK,
            "linewidth": 1.05
        }
    )

    colors = [C_ACCEPT, C_REJECT]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor(C_BLACK)
        patch.set_linewidth(0.9)

    mean_acc = float(np.mean(accepted_errors))
    mean_rej = float(np.mean(rejected_errors))
    ratio = mean_rej / (mean_acc + 1e-12)

    ax.set_ylabel("Absolute error")
    style_axis(ax)
    ax.grid(True, axis="y", linestyle="--", color=C_GRID)
    ax.grid(False, axis="x")

    ax.set_ylim(0, 0.18)

    ax.text(
        0.04,
        0.92,
        f"Mean absolute error\nAccepted = {mean_acc:.4f}\nRejected = {mean_rej:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=C_BLACK
    )

    ax.text(
        0.96,
        0.92,
        f"{ratio:.2f}× higher",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=C_BLACK
    )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig4_accepted_rejected_box_final_v2")


# ============================================================
# 13. Final 3-panel main figure
# ============================================================
def plot_main_3panel(curves, area_df):
    fig = plt.figure(figsize=(11.4, 3.75))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.35, 1.0, 1.0],
        wspace=0.35
    )

    ax1 = fig.add_subplot(gs[0, 0])
    plot_rmse_curve(curves, ax=ax1, show_legend=True)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_ause_bar(area_df, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_accepted_rejected_box(ax=ax3, rejection=0.20)

    labels = ["(a)", "(b)", "(c)"]
    for ax, label in zip([ax1, ax2, ax3], labels):
        ax.text(
            -0.17,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top"
        )

    fig.tight_layout()
    save_fig(fig, "fig5_selective_risk_main_3panel_final_v2")


# ============================================================
# 14. Final 4-panel complete figure
# ============================================================
def plot_complete_4panel(curves, area_df):
    fig = plt.figure(figsize=(11.2, 8.2))
    gs = fig.add_gridspec(
        2,
        2,
        wspace=0.28,
        hspace=0.32
    )

    ax1 = fig.add_subplot(gs[0, 0])
    plot_rmse_curve(curves, ax=ax1, show_legend=True)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_mae_curve(curves, ax=ax2, show_legend=False)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_ause_bar(area_df, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_accepted_rejected_box(ax=ax4, rejection=0.20)

    labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label in zip([ax1, ax2, ax3, ax4], labels):
        ax.text(
            -0.15,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top"
        )

    fig.tight_layout()
    save_fig(fig, "fig6_selective_risk_complete_4panel_final_v2")


# ============================================================
# 15. Main
# ============================================================
if __name__ == "__main__":
    curves = build_all_curves()
    area_df = compute_area_summary(curves)

    # Save numerical results
    curve_rows = []
    for method_name, cdf in curves.items():
        temp = cdf.copy()
        temp["method"] = method_name
        curve_rows.append(temp)

    curves_df = pd.concat(curve_rows, ignore_index=True)

    curves_csv = os.path.join(OUT_DIR, "risk_coverage_curves_final_v2.csv")
    area_csv = os.path.join(OUT_DIR, "risk_coverage_area_summary_final_v2.csv")

    curves_df.to_csv(curves_csv, index=False)
    area_df.to_csv(area_csv, index=False)

    print("\nRisk-coverage area summary:")
    print(area_df)

    print("\nSaved CSV:")
    print(curves_csv)
    print(area_csv)

    # Individual figures
    plot_rmse_curve(curves)
    plot_mae_curve(curves)
    plot_ause_bar(area_df)
    plot_accepted_rejected_box(rejection=0.20)

    # Main journal figures
    plot_main_3panel(curves, area_df)
    plot_complete_4panel(curves, area_df)

    print("\nAll final figures saved to:")
    print(OUT_DIR)