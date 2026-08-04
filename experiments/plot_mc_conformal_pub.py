import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path config
# ============================================================
project_root = r"E:\PV_RL_Project"

# input files
raw_mc_summary_path = os.path.join(
    project_root, "outputs", "mc_dropout_resnet50_with_i", "mc_summary.csv"
)

split_conf_summary_path = os.path.join(
    project_root, "outputs", "conformal_resnet50_with_i", "conformal_summary.csv"
)

split_conf_pred_path = os.path.join(
    project_root, "outputs", "conformal_resnet50_with_i", "test_conformal_predictions.csv"
)

mc_conf_summary_path = os.path.join(
    project_root, "outputs", "mc_conformal_resnet50_with_i", "mc_conformal_summary.csv"
)

mc_conf_pred_path = os.path.join(
    project_root, "outputs", "mc_conformal_resnet50_with_i", "mc_conformal_test_predictions.csv"
)

# output dir
out_dir = os.path.join(
    project_root, "outputs", "mc_conformal_resnet50_with_i", "figures_pub"
)
os.makedirs(out_dir, exist_ok=True)

# load data
raw_mc_summary = pd.read_csv(raw_mc_summary_path)
split_conf_summary = pd.read_csv(split_conf_summary_path)
split_conf_pred = pd.read_csv(split_conf_pred_path)
mc_conf_summary = pd.read_csv(mc_conf_summary_path)
mc_conf_pred = pd.read_csv(mc_conf_pred_path)

print("Loaded:")
print(raw_mc_summary_path)
print(split_conf_summary_path)
print(split_conf_pred_path)
print(mc_conf_summary_path)
print(mc_conf_pred_path)


# ============================================================
# Publication style
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 0.45,
})


# ============================================================
# Colors
# ============================================================
C_BLUE = "#1f4e79"
C_LIGHT_BLUE = "#4c78a8"
C_RED = "#b22222"
C_ORANGE = "#d97706"
C_GREEN = "#2f855a"
C_GRAY = "#bdbdbd"
C_DARK = "#404040"


# ============================================================
# Helper
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


def get_raw_mc_90():
    df = raw_mc_summary.copy()
    df = df[(df["split"] == "test") & (df["confidence"] == 90)].iloc[0]
    return {
        "method": "Raw MC Dropout",
        "PICP": df["PICP"],
        "MPIW": df["MPIW"],
        "PINAW": df["PINAW"],
        "Pearson": df["Pearson_std_error"],
        "Spearman": df["Spearman_std_error"],
    }


def get_split_conf_90():
    df = split_conf_summary.copy()
    df = df[df["confidence"] == 90].iloc[0]

    # 兼容不同脚本保存的列名
    if "Pearson_width_error" in df.index:
        pearson_col = "Pearson_width_error"
    elif "Pearson(width, error)" in df.index:
        pearson_col = "Pearson(width, error)"
    else:
        raise KeyError(f"Cannot find Pearson column. Available columns: {list(df.index)}")

    if "Spearman_width_error" in df.index:
        spearman_col = "Spearman_width_error"
    elif "Spearman(width, error)" in df.index:
        spearman_col = "Spearman(width, error)"
    else:
        raise KeyError(f"Cannot find Spearman column. Available columns: {list(df.index)}")

    return {
        "method": "Split Conformal",
        "PICP": df["PICP"],
        "MPIW": df["MPIW"],
        "PINAW": df["PINAW"],
        "Pearson": df[pearson_col],
        "Spearman": df[spearman_col],
    }


def get_mc_conf_90():
    df = mc_conf_summary.copy()
    df = df[df["confidence"] == 90].iloc[0]
    return {
        "method": "MC + Conformal",
        "PICP": df["PICP"],
        "MPIW": df["MPIW"],
        "PINAW": df["PINAW"],
        "Pearson": df["Pearson_width_error"],
        "Spearman": df["Spearman_width_error"],
    }


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
            rows.append({
                "bin": f"{left:.1f}–{right:.1f}",
                "count": 0,
                "coverage": np.nan
            })
        else:
            rows.append({
                "bin": f"{left:.1f}–{right:.1f}",
                "count": int(mask.sum()),
                "coverage": float(covered[mask].mean())
            })

    return pd.DataFrame(rows)


# ============================================================
# Figure 1: method comparison
# ============================================================
def plot_method_comparison(ax=None):
    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    else:
        fig = ax.figure
        raise ValueError("This function creates 3 subplots; use standalone only.")

    raw = get_raw_mc_90()
    splitc = get_split_conf_90()
    mcc = get_mc_conf_90()

    methods = [raw["method"], splitc["method"], mcc["method"]]
    colors = [C_GRAY, C_BLUE, C_GREEN]

    # --- PICP
    vals = [raw["PICP"], splitc["PICP"], mcc["PICP"]]
    bars = axes[0].bar(methods, vals, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(0.90, linestyle="--", color=C_RED, linewidth=1.4)
    axes[0].set_ylim(0.45, 1.00)
    axes[0].set_ylabel("PICP")
    axes[0].set_title("Coverage probability")
    axes[0].grid(True, axis="y", linestyle="--")
    axes[0].tick_params(axis="x", rotation=15)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)

    # --- MPIW
    vals = [raw["MPIW"], splitc["MPIW"], mcc["MPIW"]]
    bars = axes[1].bar(methods, vals, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylim(0.00, 0.18)
    axes[1].set_ylabel("MPIW")
    axes[1].set_title("Mean interval width")
    axes[1].grid(True, axis="y", linestyle="--")
    axes[1].tick_params(axis="x", rotation=15)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)

    # --- Spearman
    vals = [raw["Spearman"], splitc["Spearman"], mcc["Spearman"]]
    bars = axes[2].bar(methods, vals, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylim(0.00, 0.40)
    axes[2].set_ylabel("Spearman(width, error)")
    axes[2].set_title("Adaptive uncertainty quality")
    axes[2].grid(True, axis="y", linestyle="--")
    axes[2].tick_params(axis="x", rotation=15)
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)

    fig.tight_layout(w_pad=2.0)

    if own_fig:
        save_fig(fig, "fig1_method_comparison")


# ============================================================
# Figure 2: MC+Conformal interval scatter
# ============================================================
def plot_mc_conformal_scatter(ax=None, sample_n=120, random_state=42):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
    else:
        fig = ax.figure

    df = mc_conf_pred.copy()

    # stratified sample
    bins = pd.cut(df["y_true"], bins=[0, 0.2, 0.5, 1.0], include_lowest=True)
    sampled = []
    n_each = max(sample_n // 3, 1)

    for _, g in df.groupby(bins, observed=True):
        if len(g) > n_each:
            sampled.append(g.sample(n_each, random_state=random_state))
        else:
            sampled.append(g)

    df = pd.concat(sampled).drop_duplicates().sort_values("y_true")

    x = df["y_true"].values
    y = df["pred_mean"].values
    yerr_lower = y - df["lower_mc_conf_90"].values
    yerr_upper = df["upper_mc_conf_90"].values - y

    ax.errorbar(
        x, y,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        markersize=2.5,
        color=C_GREEN,
        ecolor=(47/255, 133/255, 90/255, 0.18),
        elinewidth=0.55,
        capsize=0.8,
        alpha=0.82,
        label="MC + Conformal 90% interval"
    )

    ax.plot([0, 1], [0, 1], "--", color=C_RED, linewidth=1.5, label="Ideal line")

    ax.set_xlabel("True power loss")
    ax.set_ylabel("Predicted power loss")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--")
    ax.legend(loc="upper left", frameon=True, borderpad=0.35, handlelength=2.0)

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig2_mc_conformal_scatter_90")


# ============================================================
# Figure 3: width vs error comparison
# ============================================================
def plot_width_error_comparison(ax=None, sample_n=1600, random_state=42):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.1, 4.8))
    else:
        fig = ax.figure

    df1 = split_conf_pred.copy()
    df2 = mc_conf_pred.copy()

    if len(df1) > sample_n:
        df1 = df1.sample(sample_n, random_state=random_state)
    if len(df2) > sample_n:
        df2 = df2.sample(sample_n, random_state=random_state)

    x1 = df1["width_90"].values
    y1 = np.abs(df1["y_true"].values - df1["y_pred"].values)

    x2 = df2["width_mc_conf_90"].values
    y2 = np.abs(df2["y_true"].values - df2["pred_mean"].values)

    ax.scatter(
        x1, y1,
        s=10, alpha=0.22,
        color=C_BLUE,
        label="Split Conformal"
    )

    ax.scatter(
        x2, y2,
        s=10, alpha=0.22,
        color=C_GREEN,
        label="MC + Conformal"
    )

    # trend line
    coef1 = np.polyfit(x1, y1, 1)
    coef2 = np.polyfit(x2, y2, 1)

    xs1 = np.linspace(x1.min(), x1.max(), 100)
    xs2 = np.linspace(x2.min(), x2.max(), 100)

    ax.plot(xs1, coef1[0]*xs1 + coef1[1], color=C_BLUE, linewidth=2.0)
    ax.plot(xs2, coef2[0]*xs2 + coef2[1], color=C_GREEN, linewidth=2.0)

    ax.set_xlabel("Prediction interval width")
    ax.set_ylabel("Absolute error")
    ax.set_title("Interval width vs. prediction error")
    ax.grid(True, linestyle="--")
    ax.legend(loc="upper left", frameon=True, borderpad=0.35)

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig3_width_error_comparison")


# ============================================================
# Figure 4: coverage by L-bin
# ============================================================
def plot_coverage_by_lbin_compare(ax=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
    else:
        fig = ax.figure

    bins = np.linspace(0, 1, 11)

    split_df = coverage_by_bins(
        split_conf_pred["y_true"].values,
        split_conf_pred["y_true"].values,
        split_conf_pred["lower_90"].values,
        split_conf_pred["upper_90"].values,
        bins
    )

    mc_df = coverage_by_bins(
        mc_conf_pred["y_true"].values,
        mc_conf_pred["y_true"].values,
        mc_conf_pred["lower_mc_conf_90"].values,
        mc_conf_pred["upper_mc_conf_90"].values,
        bins
    )

    x = np.arange(len(split_df))
    width = 0.38

    bars1 = ax.bar(
        x - width/2,
        split_df["coverage"].values,
        width=width,
        color=C_BLUE,
        edgecolor="black",
        linewidth=0.5,
        label="Split Conformal"
    )

    bars2 = ax.bar(
        x + width/2,
        mc_df["coverage"].values,
        width=width,
        color=C_GREEN,
        edgecolor="black",
        linewidth=0.5,
        label="MC + Conformal"
    )

    ax.axhline(0.90, linestyle="--", color=C_RED, linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(split_df["bin"].tolist(), rotation=20)
    ax.set_xlabel("True power loss bin")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.0, 1.10)
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(loc="lower left", frameon=True, borderpad=0.35)

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig4_coverage_by_lbin_compare")


# ============================================================
# Figure 5: multi-panel main figure
# ============================================================
def plot_main_multipanel():
    fig = plt.figure(figsize=(11.2, 8.6))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.30)

    # (a) method comparison -> custom 3 subplots squeezed into one panel style
    axa = fig.add_subplot(gs[0, 0])
    axa.axis("off")

    subgs = gs[0, 0].subgridspec(1, 3, wspace=0.42)
    ax1 = fig.add_subplot(subgs[0, 0])
    ax2 = fig.add_subplot(subgs[0, 1])
    ax3 = fig.add_subplot(subgs[0, 2])

    raw = get_raw_mc_90()
    splitc = get_split_conf_90()
    mcc = get_mc_conf_90()

    methods = [raw["method"], splitc["method"], mcc["method"]]
    colors = [C_GRAY, C_BLUE, C_GREEN]

    vals = [raw["PICP"], splitc["PICP"], mcc["PICP"]]
    bars = ax1.bar(methods, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(0.90, linestyle="--", color=C_RED, linewidth=1.3)
    ax1.set_ylim(0.45, 1.00)
    ax1.set_ylabel("PICP")
    ax1.set_title("Coverage")
    ax1.grid(True, axis="y", linestyle="--")
    ax1.tick_params(axis="x", rotation=25, labelsize=7)

    vals = [raw["MPIW"], splitc["MPIW"], mcc["MPIW"]]
    bars = ax2.bar(methods, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylim(0.00, 0.18)
    ax2.set_ylabel("MPIW")
    ax2.set_title("Width")
    ax2.grid(True, axis="y", linestyle="--")
    ax2.tick_params(axis="x", rotation=25, labelsize=7)

    vals = [raw["Spearman"], splitc["Spearman"], mcc["Spearman"]]
    bars = ax3.bar(methods, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_ylim(0.00, 0.40)
    ax3.set_ylabel("Spearman")
    ax3.set_title("Adaptivity")
    ax3.grid(True, axis="y", linestyle="--")
    ax3.tick_params(axis="x", rotation=25, labelsize=7)

    # (b)
    axb = fig.add_subplot(gs[0, 1])
    plot_mc_conformal_scatter(ax=axb, sample_n=120)

    # (c)
    axc = fig.add_subplot(gs[1, 0])
    plot_width_error_comparison(ax=axc, sample_n=1600)

    # (d)
    axd = fig.add_subplot(gs[1, 1])
    plot_coverage_by_lbin_compare(ax=axd)

    labels = ["(a)", "(b)", "(c)", "(d)"]
    axes_for_label = [ax1, axb, axc, axd]

    for ax, lb in zip(axes_for_label, labels):
        ax.text(
            -0.18, 1.08, lb,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )

    png_path = os.path.join(out_dir, "fig5_mc_conformal_multipanel.png")
    pdf_path = os.path.join(out_dir, "fig5_mc_conformal_multipanel.pdf")
    svg_path = os.path.join(out_dir, "fig5_mc_conformal_multipanel.svg")

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
    plot_method_comparison()
    plot_mc_conformal_scatter()
    plot_width_error_comparison()
    plot_coverage_by_lbin_compare()
    plot_main_multipanel()

    print("\nAll publication-style figures saved to:")
    print(out_dir)