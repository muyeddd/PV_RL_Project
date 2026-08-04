import os
import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 0. Path configuration
# ============================================================
PROJECT_ROOT = r"E:\PV_RL_Project"

IMG_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "PanelImages"
)

MC_CONF_PRED_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_conformal_resnet50_with_i",
    "mc_conformal_test_predictions.csv"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_conformal_resnet50_with_i",
    "uncertain_sample_visualization"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Plot style
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 2. Helper functions
# ============================================================
def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def check_dir(path):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Directory not found: {path}")


def save_fig(fig, name):
    png_path = os.path.join(OUT_DIR, name + ".png")
    pdf_path = os.path.join(OUT_DIR, name + ".pdf")
    svg_path = os.path.join(OUT_DIR, name + ".svg")

    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", svg_path)


def find_image_path(filename):
    """
    Try to find image path robustly.

    The csv may store filename, path, image_path, file, img_name, etc.
    This function searches direct path first, then IMG_DIR/filename.
    """
    if pd.isna(filename):
        return None

    filename = str(filename)

    # If csv already stores a full path
    if os.path.exists(filename):
        return filename

    # Normal case
    candidate = os.path.join(IMG_DIR, filename)
    if os.path.exists(candidate):
        return candidate

    # Sometimes only basename is needed
    basename = os.path.basename(filename)
    candidate = os.path.join(IMG_DIR, basename)
    if os.path.exists(candidate):
        return candidate

    return None


def infer_filename_column(df):
    """
    Try common filename column names.
    """
    candidates = [
        "filename",
        "file_name",
        "image_name",
        "img_name",
        "path",
        "image_path",
        "filepath",
        "file",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        "Cannot find filename/path column. "
        f"Available columns: {list(df.columns)}\n"
        "Please check mc_conformal_test_predictions.csv."
    )


def crop_center_to_square(img):
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def load_image_for_plot(path, crop_square=True):
    img = Image.open(path).convert("RGB")
    if crop_square:
        img = crop_center_to_square(img)
    return img


def short_name(name, max_len=22):
    base = os.path.basename(str(name))
    if len(base) <= max_len:
        return base
    return base[:max_len - 3] + "..."


def make_caption(row):
    return (
        f"True={row['y_true']:.3f} | Pred={row['pred_mean']:.3f}\n"
        f"Err={row['abs_error']:.3f} | Width={row['width_mc_conf_90']:.3f}\n"
        f"PI90=[{row['lower_mc_conf_90']:.3f}, {row['upper_mc_conf_90']:.3f}]"
    )


# ============================================================
# 3. Load predictions
# ============================================================
check_dir(IMG_DIR)
check_file(MC_CONF_PRED_PATH)

df = pd.read_csv(MC_CONF_PRED_PATH)

print("Loaded:", MC_CONF_PRED_PATH)
print("Columns:", df.columns.tolist())
print("N =", len(df))

required_cols = [
    "y_true",
    "pred_mean",
    "lower_mc_conf_90",
    "upper_mc_conf_90",
    "width_mc_conf_90",
]

for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}. Available: {list(df.columns)}")

fname_col = infer_filename_column(df)
print("Filename column detected:", fname_col)

df["abs_error"] = np.abs(df["y_true"] - df["pred_mean"])
df["image_path"] = df[fname_col].apply(find_image_path)

missing = df["image_path"].isna().sum()
print("Missing image paths:", missing)

if missing > 0:
    miss_out = os.path.join(OUT_DIR, "missing_image_paths.csv")
    df[df["image_path"].isna()].to_csv(miss_out, index=False)
    print("Saved missing list:", miss_out)

df_valid = df.dropna(subset=["image_path"]).copy()

if len(df_valid) == 0:
    raise RuntimeError("No valid image paths found. Check IMG_DIR and filename column.")


# ============================================================
# 4. Select sample groups
# ============================================================
TOP_K = 12

# 1. Highest uncertainty
top_uncertainty = df_valid.sort_values(
    "width_mc_conf_90", ascending=False
).head(TOP_K).copy()
top_uncertainty["group"] = "Top uncertainty"

# 2. Highest absolute error
top_error = df_valid.sort_values(
    "abs_error", ascending=False
).head(TOP_K).copy()
top_error["group"] = "Top error"

# 3. Reliable samples: low uncertainty and low error
# Use combined rank to avoid selecting only one criterion
temp = df_valid.copy()
temp["rank_width"] = temp["width_mc_conf_90"].rank(ascending=True)
temp["rank_error"] = temp["abs_error"].rank(ascending=True)
temp["reliable_score"] = temp["rank_width"] + temp["rank_error"]

reliable = temp.sort_values(
    "reliable_score", ascending=True
).head(TOP_K).copy()
reliable["group"] = "Reliable"

# 4. Mixed representatives
mixed = pd.concat([
    top_uncertainty.head(4),
    top_error.head(4),
    reliable.head(4)
], ignore_index=True)
mixed["group"] = (
    ["High uncertainty"] * 4
    + ["High error"] * 4
    + ["Reliable"] * 4
)

selected = pd.concat(
    [top_uncertainty, top_error, reliable, mixed],
    ignore_index=True
)

selected_out = os.path.join(OUT_DIR, "selected_uncertain_cases.csv")
selected.to_csv(selected_out, index=False)
print("Saved selected cases:", selected_out)


# ============================================================
# 5. Plot grid function
# ============================================================
def plot_sample_grid(
    sample_df,
    title,
    filename,
    ncols=4,
    crop_square=True,
    title_color="#1f4e79"
):
    n = len(sample_df)
    nrows = math.ceil(n / ncols)

    fig_w = 12.0
    fig_h = nrows * 3.25 + 0.6

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for i, (_, row) in enumerate(sample_df.iterrows()):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        img = load_image_for_plot(row["image_path"], crop_square=crop_square)
        ax.imshow(img)
        ax.axis("off")

        # Add small category label if available
        group_label = row.get("group", "")
        if group_label:
            ax.text(
                0.02,
                0.98,
                str(group_label),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                color="white",
                bbox=dict(
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.55,
                    boxstyle="round,pad=0.22"
                )
            )

        caption = make_caption(row)

        ax.text(
            0.5,
            -0.10,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            color="#111111"
        )

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        color=title_color,
        y=0.995
    )

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.94,
        bottom=0.03,
        wspace=0.08,
        hspace=0.38
    )

    save_fig(fig, filename)


# ============================================================
# 6. Create figures
# ============================================================
plot_sample_grid(
    top_uncertainty,
    title="Top uncertainty samples selected by MC+CP interval width",
    filename="fig1_top_uncertainty_samples",
    ncols=4,
    title_color="#2f855a"
)

plot_sample_grid(
    top_error,
    title="Top error samples selected by absolute prediction error",
    filename="fig2_top_error_samples",
    ncols=4,
    title_color="#b22222"
)

plot_sample_grid(
    reliable,
    title="Reliable samples with low uncertainty and low error",
    filename="fig3_reliable_samples",
    ncols=4,
    title_color="#1f4e79"
)

plot_sample_grid(
    mixed,
    title="Representative comparison: high uncertainty, high error, and reliable cases",
    filename="fig4_mixed_representative_cases",
    ncols=4,
    title_color="#333333"
)


# ============================================================
# 7. Print summary statistics
# ============================================================
def summarize_group(name, data):
    print(f"\n{name}")
    print("-" * len(name))
    print("N:", len(data))
    print("Mean true L:", data["y_true"].mean())
    print("Mean pred L:", data["pred_mean"].mean())
    print("Mean abs error:", data["abs_error"].mean())
    print("Mean width:", data["width_mc_conf_90"].mean())
    print("Median width:", data["width_mc_conf_90"].median())


summarize_group("Top uncertainty", top_uncertainty)
summarize_group("Top error", top_error)
summarize_group("Reliable", reliable)

print("\nAll figures saved to:")
print(OUT_DIR)