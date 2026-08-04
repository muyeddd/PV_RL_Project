import os
import sys
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50


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

CKPT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "models_ckpt",
    "best_resnet50_with_i.pth"
)

SELECTED_CASES_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_conformal_resnet50_with_i",
    "representative_case_selection",
    "representative_selected_cases.csv"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "mc_conformal_resnet50_with_i",
    "gradcam_resnet50_with_i"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Basic settings
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
DROPOUT = 0.3

# 每类选几个样本
N_HIGH_UNC = 2
N_HIGH_ERR = 2
N_RELIABLE = 2

# 如果你想手动指定样本，把文件名填到这里；否则保持 None
MANUAL_FILENAMES = None
# MANUAL_FILENAMES = [
#     "xxx.jpg",
#     "yyy.jpg",
# ]


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
# 2. Model definition
# ============================================================
class SolarResNet50WithI(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        backbone = resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.i_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True)
        )

        self.regressor = nn.Sequential(
            nn.Linear(in_features + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_i):
        img_feat = self.backbone(x_img)
        i_feat = self.i_branch(x_i.unsqueeze(1))
        feat = torch.cat([img_feat, i_feat], dim=1)
        out = self.regressor(feat)
        return torch.sigmoid(out)


# ============================================================
# 3. Helpers
# ============================================================
def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def check_dir(path):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Directory not found: {path}")


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


def infer_filename_column(df):
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
        f"Available columns: {list(df.columns)}"
    )


def find_image_path(value):
    if pd.isna(value):
        return None

    value = str(value)

    if os.path.exists(value):
        return value

    basename = os.path.basename(value)

    candidate = os.path.join(IMG_DIR, value)
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.join(IMG_DIR, basename)
    if os.path.exists(candidate):
        return candidate

    return None


def infer_i_column(df):
    candidates = [
        "I",
        "i",
        "irradiance",
        "Irradiance",
        "x_i",
        "input_i",
        "solar_irradiance",
        "radiation",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def parse_i_from_filename(filename):
    """
    优先使用项目里的 utils.parser.py。
    如果失败，再尝试简单正则。
    """
    basename = os.path.basename(str(filename))

    # Try project parser
    try:
        if PROJECT_ROOT not in sys.path:
            sys.path.append(PROJECT_ROOT)

        from utils.parser import parse_filename

        parsed = parse_filename(basename)

        if isinstance(parsed, dict):
            for key in ["I", "i", "irradiance", "Irradiance"]:
                if key in parsed:
                    return float(parsed[key])

        if isinstance(parsed, (tuple, list)) and len(parsed) >= 3:
            return float(parsed[2])

    except Exception:
        pass

    # Fallback regex
    patterns = [
        r"[Ii][_=:-]?([0-9]+(?:\.[0-9]+)?)",
        r"irr(?:adiance)?[_=:-]?([0-9]+(?:\.[0-9]+)?)",
    ]

    for p in patterns:
        m = re.search(p, basename)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass

    return None


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    print("\nCheckpoint loaded:", ckpt_path)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])

    return model


def pil_to_rgb_array(img):
    return np.asarray(img).astype(np.float32) / 255.0


def make_overlay(rgb, cam, alpha=0.42):
    """
    rgb: H x W x 3, 0-1
    cam: H x W, 0-1
    """
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]
    overlay = (1 - alpha) * rgb + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return heatmap, overlay


def short_caption(row):
    return (
        f"{row['group']}\n"
        f"True={row['y_true']:.3f}, Pred={row['pred_for_display']:.3f}\n"
        f"Err={row['abs_error']:.3f}, Width={row['width_mc_conf_90']:.3f}"
    )


# ============================================================
# 4. Grad-CAM implementation for regression
# ============================================================
class RegressionGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, image_tensor, i_tensor):
        self.model.zero_grad(set_to_none=True)

        image_tensor = image_tensor.to(DEVICE)
        i_tensor = i_tensor.to(DEVICE)

        output = self.model(image_tensor, i_tensor)
        target = output.squeeze()

        target.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        # activations: [B, C, H, W]
        # gradients:   [B, C, H, W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        pred = float(output.detach().cpu().item())

        return cam, pred


# ============================================================
# 5. Preprocess
# ============================================================
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def prepare_image(path):
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    tensor = eval_transform(img).unsqueeze(0)
    rgb = pil_to_rgb_array(img_resized)
    return img_resized, tensor, rgb


# ============================================================
# 6. Load selected cases
# ============================================================
check_dir(IMG_DIR)
check_file(CKPT_PATH)
check_file(SELECTED_CASES_PATH)

cases_df = pd.read_csv(SELECTED_CASES_PATH)

print("Loaded selected cases:", SELECTED_CASES_PATH)
print("Columns:", cases_df.columns.tolist())
print("N =", len(cases_df))

fname_col = infer_filename_column(cases_df)
i_col = infer_i_column(cases_df)

cases_df["image_path"] = cases_df[fname_col].apply(find_image_path)

if i_col is not None:
    print("I column detected:", i_col)
    cases_df["I_value"] = cases_df[i_col].astype(float)
else:
    print("No I column detected. Trying to parse I from filename...")
    cases_df["I_value"] = cases_df[fname_col].apply(parse_i_from_filename)

missing_img = cases_df["image_path"].isna().sum()
missing_i = cases_df["I_value"].isna().sum()

print("Missing image paths:", missing_img)
print("Missing I values:", missing_i)

if missing_img > 0:
    miss_path = os.path.join(OUT_DIR, "missing_image_paths.csv")
    cases_df[cases_df["image_path"].isna()].to_csv(miss_path, index=False)
    print("Saved missing image paths:", miss_path)

if missing_i > 0:
    miss_i_path = os.path.join(OUT_DIR, "missing_i_values.csv")
    cases_df[cases_df["I_value"].isna()].to_csv(miss_i_path, index=False)
    print("Saved missing I values:", miss_i_path)

cases_df = cases_df.dropna(subset=["image_path", "I_value"]).copy()

if len(cases_df) == 0:
    raise RuntimeError("No valid cases left after checking image_path and I_value.")


# Ensure required numerical columns
required_cols = [
    "y_true",
    "width_mc_conf_90",
]

for c in required_cols:
    if c not in cases_df.columns:
        raise KeyError(f"Missing required column: {c}. Available: {list(cases_df.columns)}")

if "pred_mean" in cases_df.columns:
    cases_df["pred_for_display"] = cases_df["pred_mean"]
elif "y_pred" in cases_df.columns:
    cases_df["pred_for_display"] = cases_df["y_pred"]
else:
    raise KeyError("Cannot find prediction column: pred_mean or y_pred.")

if "abs_error" not in cases_df.columns:
    cases_df["abs_error"] = np.abs(cases_df["y_true"] - cases_df["pred_for_display"])


# ============================================================
# 7. Select representative cases
# ============================================================
def select_cases(df):
    df = df.copy()

    # ========================================================
    # Case 1:
    # If representative_selected_cases.csv is used,
    # directly use all manually selected representative rows.
    # ========================================================
    if "selected" in df.columns:
        df["selected"] = pd.to_numeric(
            df["selected"],
            errors="coerce"
        ).fillna(0).astype(int)

        df = df[df["selected"] == 1].copy()

    if "final_group" in df.columns:
        df["group"] = df["final_group"].astype(str)

    if len(df) > 0 and "case_id" in df.columns:
        return df.drop_duplicates("image_path").copy()

    # ========================================================
    # Case 2:
    # Original automatic selection logic.
    # This part is used only when no manual selected file exists.
    # ========================================================
    if MANUAL_FILENAMES is not None:
        basenames = set([os.path.basename(x) for x in MANUAL_FILENAMES])
        selected = df[
            df[fname_col].apply(
                lambda x: os.path.basename(str(x)) in basenames
            )
        ].copy()

        if "group" not in selected.columns:
            selected["group"] = "Manual"

        return selected

    if "group" not in df.columns:
        df["group"] = ""

    group_text = df["group"].astype(str).str.lower()

    high_unc = df[group_text.str.contains("high uncertainty")].copy()

    if len(high_unc) == 0:
        high_unc = df[group_text.str.contains("top uncertainty")].copy()

    if len(high_unc) == 0:
        high_unc = df.sort_values("width_mc_conf_90", ascending=False).copy()

    high_unc = (
        high_unc
        .sort_values("width_mc_conf_90", ascending=False)
        .drop_duplicates("image_path")
        .head(N_HIGH_UNC)
    )

    high_unc["group"] = "High uncertainty"

    high_err = df[group_text.str.contains("high error")].copy()

    if len(high_err) == 0:
        high_err = df[group_text.str.contains("top error")].copy()

    if len(high_err) == 0:
        high_err = df.sort_values("abs_error", ascending=False).copy()

    high_err = (
        high_err
        .sort_values("abs_error", ascending=False)
        .drop_duplicates("image_path")
        .head(N_HIGH_ERR)
    )

    high_err["group"] = "High error"

    reliable = df[group_text.str.contains("reliable")].copy()

    if len(reliable) == 0:
        temp = df.copy()
        temp["rank_width"] = temp["width_mc_conf_90"].rank(ascending=True)
        temp["rank_error"] = temp["abs_error"].rank(ascending=True)
        temp["reliable_score"] = temp["rank_width"] + temp["rank_error"]

        reliable = temp.sort_values("reliable_score", ascending=True).copy()

    reliable = (
        reliable
        .sort_values("abs_error", ascending=True)
        .drop_duplicates("image_path")
        .head(N_RELIABLE)
    )

    reliable["group"] = "Reliable"

    selected = pd.concat(
        [high_unc, high_err, reliable],
        ignore_index=True
    )

    return selected

# ============================================================
# 8. Load model
# ============================================================
model = SolarResNet50WithI(dropout=DROPOUT).to(DEVICE)
model = load_checkpoint(model, CKPT_PATH)
model.eval()

target_layer = model.backbone.layer4[-1]
cam_generator = RegressionGradCAM(model, target_layer)

print("\nDevice:", DEVICE)
print("Target layer: model.backbone.layer4[-1]")


# ============================================================
# 9. Generate Grad-CAM for each selected case
# ============================================================
results = []

case_images = []

for idx, row in selected.iterrows():
    img_path = row["image_path"]
    i_value = float(row["I_value"])

    img_resized, image_tensor, rgb = prepare_image(img_path)
    i_tensor = torch.tensor([i_value], dtype=torch.float32)

    cam, pred_runtime = cam_generator(image_tensor, i_tensor)
    heatmap, overlay = make_overlay(rgb, cam, alpha=0.42)

    case_id = f"{idx:02d}_{row['group'].replace(' ', '_')}"
    base = os.path.splitext(os.path.basename(str(row[fname_col])))[0]
    safe_base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", base)[:80]

    # Save individual images
    original_path = os.path.join(OUT_DIR, f"{case_id}_{safe_base}_original.png")
    heatmap_path = os.path.join(OUT_DIR, f"{case_id}_{safe_base}_heatmap.png")
    overlay_path = os.path.join(OUT_DIR, f"{case_id}_{safe_base}_overlay.png")

    plt.imsave(original_path, rgb)
    plt.imsave(heatmap_path, heatmap)
    plt.imsave(overlay_path, overlay)

    results.append({
        "filename": row[fname_col],
        "group": row["group"],
        "image_path": img_path,
        "I_value": i_value,
        "y_true": float(row["y_true"]),
        "pred_in_csv": float(row["pred_for_display"]),
        "pred_runtime": pred_runtime,
        "abs_error_csv": float(row["abs_error"]),
        "width_mc_conf_90": float(row["width_mc_conf_90"]),
        "original_png": original_path,
        "heatmap_png": heatmap_path,
        "overlay_png": overlay_path,
    })

    case_images.append({
        "row": row,
        "rgb": rgb,
        "heatmap": heatmap,
        "overlay": overlay,
        "pred_runtime": pred_runtime,
    })

results_df = pd.DataFrame(results)
results_csv = os.path.join(OUT_DIR, "gradcam_results.csv")
results_df.to_csv(results_csv, index=False)
print("\nSaved Grad-CAM results:", results_csv)

cam_generator.remove_hooks()


# ============================================================
# 10. Plot publication-style Grad-CAM panel
# ============================================================
def plot_gradcam_panel(case_images, name="fig1_gradcam_representative_cases"):
    n_cases = len(case_images)
    ncols = 3
    nrows = n_cases

    fig_w = 9.6
    fig_h = max(2.35 * nrows, 5.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    col_titles = ["Original image", "Grad-CAM heatmap", "Overlay"]

    for c in range(ncols):
        axes[0, c].set_title(col_titles[c], fontsize=12, fontweight="bold", pad=8)

    group_colors = {
        "High uncertainty": "#2f855a",
        "High error": "#b22222",
        "Reliable": "#1f4e79",
    }

    for r, item in enumerate(case_images):
        row = item["row"]
        rgb = item["rgb"]
        heatmap = item["heatmap"]
        overlay = item["overlay"]

        imgs = [rgb, heatmap, overlay]

        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(imgs[c])
            ax.axis("off")

        group = str(row["group"])
        color = group_colors.get(group, "#333333")

        label = (
            f"{group}\n"
            f"True={row['y_true']:.3f}, Pred={row['pred_for_display']:.3f}, "
            f"Err={row['abs_error']:.3f}, Width={row['width_mc_conf_90']:.3f}"
        )

        axes[r, 0].text(
            -0.06,
            0.5,
            label,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9.0,
            color=color,
            fontweight="bold"
        )

    fig.suptitle(
        "Grad-CAM explanation for representative PV power-loss predictions",
        fontsize=15,
        fontweight="bold",
        y=0.995
    )

    fig.subplots_adjust(
        left=0.20,
        right=0.99,
        top=0.955,
        bottom=0.02,
        wspace=0.04,
        hspace=0.16
    )

    save_fig(fig, name)


plot_gradcam_panel(case_images, "fig1_gradcam_representative_cases")


# ============================================================
# 11. Also create compact 2x3 overlay-only figure
# ============================================================
def plot_overlay_only(case_images, name="fig2_gradcam_overlay_compact"):
    n = len(case_images)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 3.4 * nrows))

    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    group_colors = {
        "High uncertainty": "#2f855a",
        "High error": "#b22222",
        "Reliable": "#1f4e79",
    }

    for i, item in enumerate(case_images):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        row = item["row"]
        overlay = item["overlay"]
        group = str(row["group"])
        color = group_colors.get(group, "#333333")

        ax.imshow(overlay)
        ax.axis("off")

        ax.text(
            0.02,
            0.98,
            group,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox=dict(
                facecolor=color,
                edgecolor="none",
                alpha=0.78,
                boxstyle="round,pad=0.22"
            )
        )

        caption = (
            f"True={row['y_true']:.3f}, Pred={row['pred_for_display']:.3f}\n"
            f"Err={row['abs_error']:.3f}, Width={row['width_mc_conf_90']:.3f}"
        )

        ax.text(
            0.5,
            -0.08,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.8,
            color="#111111"
        )

    fig.suptitle(
        "Compact Grad-CAM overlays for representative cases",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.93,
        bottom=0.06,
        wspace=0.08,
        hspace=0.32
    )

    save_fig(fig, name)


plot_overlay_only(case_images, "fig2_gradcam_overlay_compact")


print("\nAll Grad-CAM outputs saved to:")
print(OUT_DIR)