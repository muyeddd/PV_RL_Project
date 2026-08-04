import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path configuration
# ============================================================
project_root = r"E:\PV_RL_Project"
result_dir = os.path.join(project_root, "outputs", "conformal_resnet50_with_i")
fig_dir = os.path.join(result_dir, "figures_final_v4")
os.makedirs(fig_dir, exist_ok=True)

summary_path = os.path.join(result_dir, "conformal_summary.csv")
test_path = os.path.join(result_dir, "test_conformal_predictions.csv")

summary_df = pd.read_csv(summary_path)
test_df = pd.read_csv(test_path)

print("Loaded:")
print(summary_path)
print(test_path)


# ============================================================
# Publication style: IEEE / Elsevier / Energy / Applied Energy
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
    "grid.alpha": 0.23,
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
C_RED = "#d62728"
C_ORANGE = "#d97706"
C_LIGHT_GRAY = "#d9d9d9"
C_HIGHLIGHT = "#b22222"


# ============================================================
# Save figure: PNG + PDF + SVG
# ============================================================
def save_fig(fig, filename):
    png_path = os.path.join(fig_dir, filename + ".png")
    pdf_path = os.path.join(fig_dir, filename + ".pdf")
    svg_path = os.path.join(fig_dir, filename + ".svg")

    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", svg_path)


# ============================================================
# Bin-wise coverage calculation
# bin_values: variable used for binning
# y_true / lower / upper: used to determine whether true L is covered
# ============================================================
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
                "bin": f"{left:.2f}–{right:.2f}",
                "coverage": np.nan,
                "count": 0
            })
        else:
            rows.append({
                "bin": f"{left:.2f}–{right:.2f}",
                "coverage": float(covered[mask].mean()),
                "count": int(mask.sum())
            })

    return pd.DataFrame(rows)


# ============================================================
# Fig. 1: Coverage–width tradeoff
# ============================================================
def plot_tradeoff(summary_df, ax=None, show_legend=True):
    own_fig = ax is None

    if own_fig:
        fig, ax1 = plt.subplots(figsize=(5.8, 4.2))
    else:
        ax1 = ax
        fig = ax1.figure

    conf = summary_df["confidence"].values
    target = conf / 100.0
    picp = summary_df["PICP"].values
    mpiw = summary_df["MPIW"].values

    ax1.plot(
        conf, picp,
        marker="o",
        markersize=5.2,
        color=C_BLUE,
        label="Empirical coverage"
    )

    ax1.plot(
        conf, target,
        linestyle="--",
        color=C_RED,
        label="Target coverage"
    )

    ax1.set_xlabel("Nominal confidence level (%)")
    ax1.set_ylabel("Coverage probability")
    ax1.set_xticks([80, 90, 95])
    ax1.set_ylim(0.78, 0.98)
    ax1.grid(True, linestyle="--")

    ax2 = ax1.twinx()

    ax2.plot(
        conf, mpiw,
        marker="s",
        markersize=5.0,
        color=C_ORANGE,
        label="Mean interval width"
    )

    ax2.set_ylabel("Mean prediction interval width")
    ax2.set_ylim(0.06, max(mpiw) * 1.15)

    if show_legend:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="lower right",
            frameon=True,
            borderpad=0.35,
            handlelength=2.0
        )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig1_tradeoff_final_v4")


# ============================================================
# Fig. 2: 90% prediction interval scatter
# sample_n = 120 for cleaner publication visualization
# ============================================================
def plot_interval_scatter(test_df, sample_n=120, random_state=42, ax=None):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
    else:
        fig = ax.figure

    df = test_df.copy()

    # Stratified sampling over low / medium / high power-loss regions
    bins = pd.cut(
        df["y_true"],
        bins=[0, 0.2, 0.5, 1.0],
        include_lowest=True
    )

    sampled_list = []
    n_each = max(sample_n // 3, 1)

    for _, group in df.groupby(bins, observed=True):
        if len(group) > n_each:
            sampled_list.append(group.sample(n_each, random_state=random_state))
        else:
            sampled_list.append(group)

    df = pd.concat(sampled_list).drop_duplicates().sort_values("y_true")

    x = df["y_true"].values
    y = df["y_pred"].values
    yerr_lower = y - df["lower_90"].values
    yerr_upper = df["upper_90"].values - y

    ax.errorbar(
        x, y,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        markersize=2.3,
        color=C_LIGHT_BLUE,
        ecolor=(76 / 255, 120 / 255, 168 / 255, 0.15),
        elinewidth=0.5,
        capsize=0.7,
        alpha=0.82,
        label="Predictions with 90% interval"
    )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color=C_RED,
        linewidth=1.5,
        label="Ideal line"
    )

    ax.set_xlabel("True power loss")
    ax.set_ylabel("Predicted power loss")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--")

    ax.legend(
        loc="upper left",
        frameon=True,
        borderpad=0.35,
        handlelength=2.0
    )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig2_interval_scatter_90_final_v4")


# ============================================================
# Fig. 3: Coverage across power-loss bins
# Red hatched bars indicate under-coverage
# ============================================================
def plot_coverage_by_lbin(test_df, ax=None):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
    else:
        fig = ax.figure

    bins = np.linspace(0, 1, 11)

    df = coverage_by_bins(
        bin_values=test_df["y_true"].values,
        y_true=test_df["y_true"].values,
        lower=test_df["lower_90"].values,
        upper=test_df["upper_90"].values,
        bins=bins
    )

    colors = []
    for cov in df["coverage"]:
        if np.isnan(cov):
            colors.append(C_LIGHT_GRAY)
        elif cov < 0.90:
            colors.append(C_HIGHLIGHT)
        else:
            colors.append(C_BLUE)

    bars = ax.bar(
        df["bin"],
        df["coverage"],
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    # Hatch for under-coverage bins
    for bar, cov in zip(bars, df["coverage"]):
        if not np.isnan(cov) and cov < 0.90:
            bar.set_hatch("//")

    # Target 90% coverage line
    ax.axhline(
        0.90,
        linestyle="--",
        color=C_RED,
        linewidth=1.5
    )

    ax.set_xlabel("True power loss bin")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.0, 1.12)
    ax.grid(True, axis="y", linestyle="--")

    # Sample counts
    for rect, count in zip(bars, df["count"]):
        if count > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.025,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=7.8
            )

    ax.tick_params(axis="x", rotation=25)

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig3_coverage_by_lbin_final_v4")


# ============================================================
# Fig. 4: Coverage across irradiance bins
# Red hatched bars indicate under-coverage
# ============================================================
def plot_coverage_by_ibin(test_df, ax=None):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
    else:
        fig = ax.figure

    I_values = test_df["I"].values
    bins = np.linspace(I_values.min(), I_values.max(), 6)

    df = coverage_by_bins(
        bin_values=I_values,
        y_true=test_df["y_true"].values,
        lower=test_df["lower_90"].values,
        upper=test_df["upper_90"].values,
        bins=bins
    )

    colors = []
    for cov in df["coverage"]:
        if np.isnan(cov):
            colors.append(C_LIGHT_GRAY)
        elif cov < 0.90:
            colors.append(C_HIGHLIGHT)
        else:
            colors.append(C_BLUE)

    bars = ax.bar(
        df["bin"],
        df["coverage"],
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    # Hatch for under-coverage bins
    for bar, cov in zip(bars, df["coverage"]):
        if not np.isnan(cov) and cov < 0.90:
            bar.set_hatch("//")

    # Target 90% coverage line
    ax.axhline(
        0.90,
        linestyle="--",
        color=C_RED,
        linewidth=1.5
    )

    ax.set_xlabel("Irradiance bin")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.0, 1.12)
    ax.grid(True, axis="y", linestyle="--")

    # Sample counts
    for rect, count in zip(bars, df["count"]):
        if count > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.025,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=7.8
            )

    ax.tick_params(axis="x", rotation=25)

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig4_coverage_by_ibin_final_v4")


# ============================================================
# Fig. 7: 2×2 multi-panel figure for paper main text
# ============================================================
def plot_multipanel(summary_df, test_df):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))

    plot_tradeoff(summary_df, ax=axes[0, 0], show_legend=True)
    plot_interval_scatter(test_df, sample_n=120, ax=axes[0, 1])
    plot_coverage_by_lbin(test_df, ax=axes[1, 0])
    plot_coverage_by_ibin(test_df, ax=axes[1, 1])

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, label in zip(axes.flat, panel_labels):
        ax.text(
            -0.13,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )

    fig.tight_layout(w_pad=2.0, h_pad=2.2)

    png_path = os.path.join(fig_dir, "fig7_multipanel_conformal_final_v4.png")
    pdf_path = os.path.join(fig_dir, "fig7_multipanel_conformal_final_v4.pdf")
    svg_path = os.path.join(fig_dir, "fig7_multipanel_conformal_final_v4.svg")

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
    plot_tradeoff(summary_df)
    plot_interval_scatter(test_df, sample_n=120)
    plot_coverage_by_lbin(test_df)
    plot_coverage_by_ibin(test_df)
    plot_multipanel(summary_df, test_df)

    print("\nFinal v4 publication-style figures saved to:")
    print(fig_dir)