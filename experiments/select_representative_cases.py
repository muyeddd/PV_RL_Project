import os
import math
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

PRED_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_conformal_resnet50_with_i",
    "mc_conformal_test_predictions.csv"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_conformal_resnet50_with_i",
    "representative_case_selection"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Settings
# ============================================================
N_CANDIDATES_PER_GROUP = 24

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.titlesize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 2. Helpers
# ============================================================
def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def check_dir(path):
    if not os.path.isdir(path):
        raise NotADirectoryError(path)


def infer_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise KeyError(
            f"Cannot find columns from {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def find_image_path(value):
    if pd.isna(value):
        return None

    value = str(value)

    if os.path.exists(value):
        return value

    basename = os.path.basename(value)

    p1 = os.path.join(IMG_DIR, value)
    if os.path.exists(p1):
        return p1

    p2 = os.path.join(IMG_DIR, basename)
    if os.path.exists(p2):
        return p2

    return None


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def crop_center_square(img):
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


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


def make_caption(row):
    return (
        f"ID: {row['case_id']}\n"
        f"True={row['y_true']:.3f}, Pred={row['pred_mean']:.3f}\n"
        f"Err={row['abs_error']:.3f}, Width={row['width_mc_conf_90']:.3f}"
    )


def plot_contact_sheet(df, title, name, ncols=4, title_color="#1f4e79"):
    df = df.copy().reset_index(drop=True)

    n = len(df)
    nrows = math.ceil(n / ncols)

    fig_w = 12.5
    fig_h = nrows * 3.15 + 0.7

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for i, row in df.iterrows():
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        img = load_image(row["image_path"])
        img = crop_center_square(img)

        ax.imshow(img)
        ax.axis("off")

        # category badge
        ax.text(
            0.02,
            0.98,
            row["candidate_group"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="white",
            bbox=dict(
                facecolor="#333333",
                edgecolor="none",
                alpha=0.65,
                boxstyle="round,pad=0.20"
            )
        )

        ax.text(
            0.5,
            -0.08,
            make_caption(row),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.8,
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
        bottom=0.035,
        wspace=0.08,
        hspace=0.38
    )

    save_fig(fig, name)


# ============================================================
# 3. Load prediction file
# ============================================================
check_dir(IMG_DIR)
check_file(PRED_PATH)

df = pd.read_csv(PRED_PATH)

print("Loaded:", PRED_PATH)
print("Columns:", df.columns.tolist())
print("N =", len(df))

fname_col = infer_col(
    df,
    ["filename", "file_name", "image_name", "img_name", "path", "image_path", "filepath", "file"]
)

pred_col = infer_col(df, ["pred_mean", "y_pred", "pred", "prediction"])
true_col = infer_col(df, ["y_true", "L", "label", "target"])
width_col = infer_col(df, ["width_mc_conf_90", "width_90", "interval_width"])
lower_col = infer_col(df, ["lower_mc_conf_90", "lower_90"], required=False)
upper_col = infer_col(df, ["upper_mc_conf_90", "upper_90"], required=False)

df = df.copy()
df["filename_for_use"] = df[fname_col].astype(str)
df["image_path"] = df[fname_col].apply(find_image_path)

df["y_true"] = df[true_col].astype(float)
df["pred_mean"] = df[pred_col].astype(float)
df["width_mc_conf_90"] = df[width_col].astype(float)
df["abs_error"] = np.abs(df["y_true"] - df["pred_mean"])

if lower_col is not None:
    df["lower_mc_conf_90"] = df[lower_col].astype(float)
else:
    df["lower_mc_conf_90"] = np.nan

if upper_col is not None:
    df["upper_mc_conf_90"] = df[upper_col].astype(float)
else:
    df["upper_mc_conf_90"] = np.nan

missing = df["image_path"].isna().sum()
print("Missing image paths:", missing)

if missing > 0:
    miss_path = os.path.join(OUT_DIR, "missing_image_paths.csv")
    df[df["image_path"].isna()].to_csv(miss_path, index=False)
    print("Saved missing list:", miss_path)

df = df.dropna(subset=["image_path"]).copy()

if len(df) == 0:
    raise RuntimeError("No valid image paths found.")


# ============================================================
# 4. Candidate pool construction
# ============================================================
def take_unique(data, n, sort_cols, ascending):
    data = data.copy()
    data = data.drop_duplicates("image_path")

    if len(data) == 0:
        return data

    data = data.sort_values(sort_cols, ascending=ascending)
    return data.head(n).copy()


# 1) Clean reliable candidates
clean_pool = df[
    (df["y_true"] <= 0.05)
    & (df["abs_error"] <= 0.02)
    & (df["width_mc_conf_90"] <= 0.06)
].copy()

# fallback if too few
if len(clean_pool) < 8:
    clean_pool = df[
        (df["y_true"] <= 0.08)
        & (df["abs_error"] <= 0.03)
    ].copy()

clean_pool = take_unique(
    clean_pool,
    N_CANDIDATES_PER_GROUP,
    sort_cols=["abs_error", "width_mc_conf_90", "y_true"],
    ascending=[True, True, True]
)
clean_pool["candidate_group"] = "Clean reliable"


# 2) Heavy soiling successful candidates
heavy_pool = df[
    (df["y_true"] >= 0.45)
    & (df["pred_mean"] >= 0.35)
    & (df["abs_error"] <= 0.18)
].copy()

# fallback if too few
if len(heavy_pool) < 8:
    heavy_pool = df[
        (df["y_true"] >= 0.35)
        & (df["pred_mean"] >= 0.30)
        & (df["abs_error"] <= 0.25)
    ].copy()

heavy_pool = take_unique(
    heavy_pool,
    N_CANDIDATES_PER_GROUP,
    sort_cols=["y_true", "pred_mean", "abs_error"],
    ascending=[False, False, True]
)
heavy_pool["candidate_group"] = "Heavy success"


# 3) High uncertainty / ambiguous candidates
unc_pool = take_unique(
    df,
    N_CANDIDATES_PER_GROUP,
    sort_cols=["width_mc_conf_90", "abs_error"],
    ascending=[False, False]
)
unc_pool["candidate_group"] = "High uncertainty"


# 4) Failure candidates
failure_pool = df[
    ((df["y_true"] >= 0.50) & (df["pred_mean"] <= 0.25))
    | (df["abs_error"] >= df["abs_error"].quantile(0.98))
].copy()

failure_pool = take_unique(
    failure_pool,
    N_CANDIDATES_PER_GROUP,
    sort_cols=["abs_error", "y_true"],
    ascending=[False, False]
)
failure_pool["candidate_group"] = "Failure case"


# Add case IDs
groups = [
    ("clean", clean_pool),
    ("heavy", heavy_pool),
    ("unc", unc_pool),
    ("fail", failure_pool),
]

all_candidates = []

for prefix, data in groups:
    data = data.copy().reset_index(drop=True)
    data["case_id"] = [f"{prefix}_{i:03d}" for i in range(len(data))]
    all_candidates.append(data)

all_candidates = pd.concat(all_candidates, ignore_index=True)

all_csv = os.path.join(OUT_DIR, "all_representative_candidate_cases.csv")
all_candidates.to_csv(all_csv, index=False)
print("Saved all candidates:", all_csv)


# ============================================================
# 5. Generate candidate sheets
# ============================================================
plot_contact_sheet(
    clean_pool.assign(case_id=[f"clean_{i:03d}" for i in range(len(clean_pool))]),
    title="Candidate clean reliable samples",
    name="candidate_01_clean_reliable",
    title_color="#1f4e79"
)

plot_contact_sheet(
    heavy_pool.assign(case_id=[f"heavy_{i:03d}" for i in range(len(heavy_pool))]),
    title="Candidate heavy-soiling successful samples",
    name="candidate_02_heavy_success",
    title_color="#2f855a"
)

plot_contact_sheet(
    unc_pool.assign(case_id=[f"unc_{i:03d}" for i in range(len(unc_pool))]),
    title="Candidate high-uncertainty / ambiguous samples",
    name="candidate_03_high_uncertainty",
    title_color="#b8860b"
)

plot_contact_sheet(
    failure_pool.assign(case_id=[f"fail_{i:03d}" for i in range(len(failure_pool))]),
    title="Candidate failure cases",
    name="candidate_04_failure_cases",
    title_color="#b22222"
)


# ============================================================
# 6. Manual selection file
# ============================================================
manual_csv = os.path.join(OUT_DIR, "manual_selected_cases.csv")

if not os.path.exists(manual_csv):
    template = all_candidates.copy()
    template["selected"] = 0
    template["final_group"] = template["candidate_group"]
    template.to_csv(manual_csv, index=False)

    print("\nManual selection template created:")
    print(manual_csv)
    print("\nNext operation:")
    print("1. Open the four candidate PNG files.")
    print("2. Open manual_selected_cases.csv.")
    print("3. Set selected=1 for the samples you want to use.")
    print("4. Adjust final_group if needed.")
    print("5. Run this script again to generate final selected figures.")

else:
    manual_df = pd.read_csv(manual_csv)
    manual_df["selected"] = pd.to_numeric(manual_df["selected"], errors="coerce").fillna(0).astype(int)

    selected = manual_df[manual_df["selected"] == 1].copy()

    if len(selected) == 0:
        print("\nmanual_selected_cases.csv exists, but no row has selected=1.")
        print("Please set selected=1 for chosen samples, then rerun.")
    else:
        if "final_group" not in selected.columns:
            selected["final_group"] = selected["candidate_group"]

        selected["candidate_group"] = selected["final_group"]

        selected_csv = os.path.join(OUT_DIR, "representative_selected_cases.csv")
        selected.to_csv(selected_csv, index=False)

        print("\nSaved final selected cases:")
        print(selected_csv)
        print(selected[["case_id", "final_group", "y_true", "pred_mean", "abs_error", "width_mc_conf_90"]])

        plot_contact_sheet(
            selected,
            title="Final representative cases selected for qualitative analysis",
            name="final_representative_selected_cases",
            ncols=4,
            title_color="#333333"
        )

        print("\nFinal representative figure and CSV have been generated.")
        print("Use representative_selected_cases.csv for Grad-CAM.")