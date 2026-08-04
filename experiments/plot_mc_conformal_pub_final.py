import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path configuration
# ============================================================
project_root = r"E:\PV_RL_Project"

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

out_dir = os.path.join(
    project_root, "outputs", "mc_conformal_resnet50_with_i", "figures_final"
)
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# Load data
# ============================================================
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
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.7,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.22,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 0.45,
})


# ============================================================
# Colors
# ============================================================
C_RAW = "#bdbdbd"
C_SPLIT = "#1f4e79"
C_MC_CONF = "#2f855a"
C_RED = "#b22222"
C_DARK = "#333333"
C_GRID = "#d0d0d0"


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


def pick_col(row_or_df, candidates):
    available = row_or_df.index if hasattr(row_or_df, "index") else row_or_df.columns
    for c in candidates:
        if c in available:
            return c
    raise KeyError(f"Cannot find any of {candidates}. Available columns: {list(available)}")


# ============================================================
# Extract 90% summary
# ============================================================
def get_raw_mc_90():
    df = raw_mc_summary.copy()
    row = df[(df["split"] == "test") & (df["confidence"] == 90)].iloc[0]

    return {
        "method": "Raw MC",
        "PICP": float(row["PICP"]),
        "MPIW": float(row["MPIW"]),
        "PINAW": float(row["PINAW"]),
        "Pearson": float(row["Pearson_std_error"]),
        "Spearman": float(row["Spearman_std_error"]),
    }


def get_split_conf_90():
    df = split_conf_summary.copy()
    row = df[df["confidence"] == 90].iloc[0]

    pearson_col = pick_col(row, ["Pearson_width_error", "Pearson(width, error)"])
    spearman_col = pick_col(row, ["Spearman_width_error", "Spearman(width, error)"])

    return {
        "method": "Split CP",
        "PICP": float(row["PICP"]),
        "MPIW": float(row["MPIW"]),
        "PINAW": float(row["PINAW"]),
        "Pearson": float(row[pearson_col]),
        "Spearman": float(row[spearman_col]),
    }


def get_mc_conf_90():
    df = mc_conf_summary.copy()
    row = df[df["confidence"] == 90].iloc[0]

    pearson_col = pick_col(row, ["Pearson_width_error", "Pearson(width, error)"])
    spearman_col = pick_col(row, ["Spearman_width_error", "Spearman(width, error)"])

    return {
        "method": "MC+CP",
        "PICP": float(row["PICP"]),
        "MPIW": float(row["MPIW"]),
        "PINAW": float(row["PINAW"]),
        "Pearson": float(row[pearson_col]),
        "Spearman": float(row[spearman_col]),
    }


def get_metrics_df():
    rows = [get_raw_mc_90(), get_split_conf_90(), get_mc_conf_90()]
    return pd.DataFrame(rows)


# ============================================================
# Bin-wise coverage
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
# Figure A: compact metric comparison
# ============================================================
def draw_metric_comparison(axs):
    metrics_df = get_metrics_df()
    methods = metrics_df["method"].tolist()
    colors = [C_RAW, C_SPLIT, C_MC_CONF]

    panels = [
        ("PICP@90", "PICP", (0.50, 1.00), "{:.3f}"),
        ("MPIW@90", "MPIW", (0.00, 0.17), "{:.3f}"),
        ("Spearman", "Spearman", (0.00, 0.36), "{:.3f}"),
    ]

    for ax, (title, col, ylim, fmt) in zip(axs, panels):
        vals = metrics_df[col].values
        bars = ax.bar(
            np.arange(len(methods)),
            vals,
            color=colors,
            edgecolor="black",
            linewidth=0.55,
            width=0.68
        )

        if col == "PICP":
            ax.axhline(0.90, linestyle="--", color=C_RED, linewidth=1.35)

        ax.set_title(title, pad=4)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=18, ha="right")
        ax.grid(True, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        for b, v in zip(bars, vals):
            offset = 0.012 if col != "MPIW" else 0.004
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + offset,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=7.8
            )

    axs[0].set_ylabel("Value")


def plot_metric_comparison():
    fig, axs = plt.subplots(1, 3, figsize=(8.8, 3.0))
    draw_metric_comparison(axs)
    fig.tight_layout(w_pad=1.6)
    save_fig(fig, "fig1_metric_comparison_final")


# ============================================================
# Figure B: MC + Conformal scatter with intervals
# ============================================================
def plot_mc_conformal_scatter(ax=None, sample_n=110, random_state=42, show_legend=True):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(5.3, 5.0))
    else:
        fig = ax.figure

    df = mc_conf_pred.copy()

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
        markersize=2.3,
        color=C_MC_CONF,
        ecolor=(47 / 255, 133 / 255, 90 / 255, 0.17),
        elinewidth=0.52,
        capsize=0.7,
        alpha=0.82,
        label="MC+CP 90% interval"
    )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color=C_RED,
        linewidth=1.45,
        label="Ideal line"
    )

    ax.set_xlabel("True power loss")
    ax.set_ylabel("Predicted power loss")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    if show_legend:
        ax.legend(
            loc="upper left",
            frameon=True,
            borderpad=0.32,
            handlelength=1.8,
            labelspacing=0.3
        )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig2_mc_conformal_scatter_final")


# ============================================================
# Figure C: width-error relationship
# ============================================================
def prepare_width_error_data():
    s = split_conf_pred.copy()
    m = mc_conf_pred.copy()

    s["display_width"] = s["width_90"]
    s["display_abs_error"] = np.abs(s["y_true"] - s["y_pred"])

    m["display_width"] = m["width_mc_conf_90"]
    m["display_abs_error"] = np.abs(m["y_true"] - m["pred_mean"])

    return s, m


def binned_median_line(x, y, n_bins=9):
    x = np.asarray(x)
    y = np.asarray(y)

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges = np.unique(edges)

    xs, ys = [], []

    if len(edges) <= 2:
        return np.array(xs), np.array(ys)

    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)

        if mask.sum() >= 10:
            xs.append(np.median(x[mask]))
            ys.append(np.median(y[mask]))

    return np.array(xs), np.array(ys)


def plot_width_error_relationship(ax=None, sample_n=900, random_state=42):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
    else:
        fig = ax.figure

    s, m = prepare_width_error_data()

    metrics = get_metrics_df()
    split_spearman = metrics.loc[metrics["method"] == "Split CP", "Spearman"].iloc[0]
    mc_spearman = metrics.loc[metrics["method"] == "MC+CP", "Spearman"].iloc[0]

    if len(s) > sample_n:
        s_plot = s.sample(sample_n, random_state=random_state)
    else:
        s_plot = s

    if len(m) > sample_n:
        m_plot = m.sample(sample_n, random_state=random_state + 1)
    else:
        m_plot = m

    # Display y clipped for readability, but statistics are from full data
    y_max_display = 0.35

    ax.scatter(
        s_plot["display_width"],
        np.minimum(s_plot["display_abs_error"], y_max_display),
        s=9,
        alpha=0.18,
        color=C_SPLIT,
        label=f"Split CP, $\\rho$={split_spearman:.3f}"
    )

    ax.scatter(
        m_plot["display_width"],
        np.minimum(m_plot["display_abs_error"], y_max_display),
        s=9,
        alpha=0.20,
        color=C_MC_CONF,
        label=f"MC+CP, $\\rho$={mc_spearman:.3f}"
    )

    # Median trend lines
    xs_s, ys_s = binned_median_line(
        s["display_width"].values,
        np.minimum(s["display_abs_error"].values, y_max_display),
        n_bins=7
    )

    xs_m, ys_m = binned_median_line(
        m["display_width"].values,
        np.minimum(m["display_abs_error"].values, y_max_display),
        n_bins=9
    )

    if len(xs_s) > 1:
        ax.plot(xs_s, ys_s, color=C_SPLIT, marker="o", markersize=3.2, linewidth=2.0)

    if len(xs_m) > 1:
        ax.plot(xs_m, ys_m, color=C_MC_CONF, marker="o", markersize=3.2, linewidth=2.0)

    ax.set_xlabel("Prediction interval width")
    ax.set_ylabel("Absolute error")
    ax.set_ylim(0, y_max_display)
    ax.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        frameon=True,
        borderpad=0.32,
        handlelength=1.5,
        labelspacing=0.3
    )

    # Small note for clipped display
    ax.text(
        0.98,
        0.96,
        "errors clipped at 0.35 for display",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=C_DARK
    )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig3_width_error_relationship_final")


# ============================================================
# Figure D: L-bin coverage comparison
# ============================================================
def plot_lbin_coverage_compare(ax=None):
    own_fig = ax is None

    if own_fig:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
    else:
        fig = ax.figure

    bins = np.linspace(0, 1, 11)

    split_df = coverage_by_bins(
        bin_values=split_conf_pred["y_true"].values,
        y_true=split_conf_pred["y_true"].values,
        lower=split_conf_pred["lower_90"].values,
        upper=split_conf_pred["upper_90"].values,
        bins=bins
    )

    mc_df = coverage_by_bins(
        bin_values=mc_conf_pred["y_true"].values,
        y_true=mc_conf_pred["y_true"].values,
        lower=mc_conf_pred["lower_mc_conf_90"].values,
        upper=mc_conf_pred["upper_mc_conf_90"].values,
        bins=bins
    )

    x = np.arange(len(split_df))
    bar_w = 0.38

    bars1 = ax.bar(
        x - bar_w / 2,
        split_df["coverage"].values,
        width=bar_w,
        color=C_SPLIT,
        edgecolor="black",
        linewidth=0.45,
        label="Split CP"
    )

    bars2 = ax.bar(
        x + bar_w / 2,
        mc_df["coverage"].values,
        width=bar_w,
        color=C_MC_CONF,
        edgecolor="black",
        linewidth=0.45,
        label="MC+CP"
    )

    # Hatch under-coverage bars
    for bars, vals in [(bars1, split_df["coverage"].values), (bars2, mc_df["coverage"].values)]:
        for b, v in zip(bars, vals):
            if not np.isnan(v) and v < 0.90:
                b.set_hatch("//")

    ax.axhline(0.90, linestyle="--", color=C_RED, linewidth=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(split_df["bin"].tolist(), rotation=22, ha="right")
    ax.set_xlabel("True power loss bin")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.10)
    ax.grid(True, axis="y", linestyle="--")
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        frameon=True,
        borderpad=0.32,
        handlelength=1.8
    )

    if own_fig:
        fig.tight_layout()
        save_fig(fig, "fig4_lbin_coverage_compare_final")


# ============================================================
# Main multipanel figure
# ============================================================
def plot_final_multipanel():
    fig = plt.figure(figsize=(11.2, 8.5))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.34
    )

    # (a) compact metric comparison with 3 mini-axes
    subgs = gs[0, 0].subgridspec(1, 3, wspace=0.42)
    ax_a1 = fig.add_subplot(subgs[0, 0])
    ax_a2 = fig.add_subplot(subgs[0, 1])
    ax_a3 = fig.add_subplot(subgs[0, 2])
    draw_metric_comparison([ax_a1, ax_a2, ax_a3])

    # (b) MC+CP interval scatter
    ax_b = fig.add_subplot(gs[0, 1])
    plot_mc_conformal_scatter(ax=ax_b, sample_n=110, show_legend=True)

    # (c) width-error
    ax_c = fig.add_subplot(gs[1, 0])
    plot_width_error_relationship(ax=ax_c, sample_n=900)

    # (d) L-bin coverage
    ax_d = fig.add_subplot(gs[1, 1])
    plot_lbin_coverage_compare(ax=ax_d)

    # Panel labels
    label_specs = [
        (ax_a1, "(a)", -0.34, 1.10),
        (ax_b, "(b)", -0.16, 1.07),
        (ax_c, "(c)", -0.15, 1.07),
        (ax_d, "(d)", -0.15, 1.07),
    ]

    for ax, label, x, y in label_specs:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left"
        )

    png_path = os.path.join(out_dir, "fig5_mc_conformal_multipanel_final.png")
    pdf_path = os.path.join(out_dir, "fig5_mc_conformal_multipanel_final.pdf")
    svg_path = os.path.join(out_dir, "fig5_mc_conformal_multipanel_final.svg")

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
    plot_metric_comparison()
    plot_mc_conformal_scatter()
    plot_width_error_relationship()
    plot_lbin_coverage_compare()
    plot_final_multipanel()

    print("\nFinal publication-style figures saved to:")
    print(out_dir)