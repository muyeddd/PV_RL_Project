import os
import sys
import re
import math
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

# 注意：这里读取的是人工选择后的代表性样本
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
    "gradcam_representative_cases"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Basic settings
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
DROPOUT = 0.3

# 可选 target layer:
# "layer4[-1]"：语义强，定位粗
# "layer4[-2]"：略早一层，可能稍细
# "layer3[-1]"：定位更细，但语义可能弱
TARGET_LAYER_NAME = "layer4[-1]"

# 热力图颜色。建议论文用 "turbo" 或 "magma"，不要用太老派的 jet
CAM_CMAP = "turbo"

# overlay 透明度
OVERLAY_ALPHA = 0.42

# 是否额外保存每个样本的单独图像
SAVE_INDIVIDUAL_IMAGES = True


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
# 3. Utility functions
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


def infer_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise KeyError(
            f"Cannot find any column from {candidates}. "
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


def parse_i_from_filename(filename):
    """
    优先调用项目 utils.parser.parse_filename。
    如果失败，再用简单正则尝试解析 I。
    """
    basename = os.path.basename(str(filename))

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


def get_target_layer(model, name):
    if name == "layer4[-1]":
        return model.backbone.layer4[-1]
    if name == "layer4[-2]":
        return model.backbone.layer4[-2]
    if name == "layer3[-1]":
        return model.backbone.layer3[-1]

    raise ValueError(
        f"Unknown TARGET_LAYER_NAME: {name}. "
        "Use 'layer4[-1]', 'layer4[-2]', or 'layer3[-1]'."
    )


def normalize_cam(cam):
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam


def make_overlay(rgb, cam, alpha=0.42, cmap_name="turbo"):
    """
    rgb: H x W x 3, range 0-1
    cam: H x W, range 0-1
    """
    cmap = plt.get_cmap(cmap_name)
    heatmap = cmap(cam)[..., :3]
    overlay = (1 - alpha) * rgb + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return heatmap, overlay


def safe_group_name(group):
    group = str(group)

    mapping = {
        "Clean reliable": "Clean reliable",
        "Heavy success": "Heavy success",
        "High uncertainty": "High uncertainty",
        "Failure case": "Failure case",
        "Reliable": "Clean reliable",
        "High error": "Failure case",
        "Top uncertainty": "High uncertainty",
        "Top error": "Failure case",
    }

    return mapping.get(group, group)


def group_color(group):
    group = safe_group_name(group)

    if "Clean" in group or "Reliable" in group:
        return "#1f4e79"
    if "Heavy" in group:
        return "#2f855a"
    if "uncertainty" in group.lower():
        return "#b8860b"
    if "Failure" in group or "error" in group.lower():
        return "#b22222"

    return "#333333"


def file_safe_name(text, max_len=80):
    text = str(text)
    text = os.path.splitext(os.path.basename(text))[0]
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    return text[:max_len]


# ============================================================
# 4. Grad-CAM implementation for regression
# ============================================================
class RegressionGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor, i_tensor):
        self.model.zero_grad(set_to_none=True)

        image_tensor = image_tensor.to(DEVICE)
        i_tensor = i_tensor.to(DEVICE)

        output = self.model(image_tensor, i_tensor)
        target = output.squeeze()

        target.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM did not capture activations or gradients. "
                "Check target layer."
            )

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
        cam = normalize_cam(cam)

        pred_runtime = float(output.detach().cpu().item())

        return cam, pred_runtime


# ============================================================
# 5. Image preprocessing
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
    rgb = np.asarray(img_resized).astype(np.float32) / 255.0

    tensor = eval_transform(img).unsqueeze(0)

    return img_resized, tensor, rgb


# ============================================================
# 6. Load manually selected representative cases
# ============================================================
check_dir(IMG_DIR)
check_file(CKPT_PATH)
check_file(SELECTED_CASES_PATH)

cases_df = pd.read_csv(SELECTED_CASES_PATH)

print("Loaded selected representative cases:", SELECTED_CASES_PATH)
print("Columns:", cases_df.columns.tolist())
print("N =", len(cases_df))

fname_col = infer_col(
    cases_df,
    ["filename_for_use", "filename", "file_name", "image_name", "img_name",
     "path", "image_path", "filepath", "file"]
)

true_col = infer_col(
    cases_df,
    ["y_true", "L", "label", "target"]
)

pred_col = infer_col(
    cases_df,
    ["pred_mean", "y_pred", "pred", "prediction", "pred_for_display"]
)

width_col = infer_col(
    cases_df,
    ["width_mc_conf_90", "width_90", "interval_width"]
)

lower_col = infer_col(
    cases_df,
    ["lower_mc_conf_90", "lower_90"],
    required=False
)

upper_col = infer_col(
    cases_df,
    ["upper_mc_conf_90", "upper_90"],
    required=False
)

i_col = infer_col(
    cases_df,
    ["I_value", "I", "i", "irradiance", "Irradiance", "x_i", "input_i"],
    required=False
)

cases_df = cases_df.copy()

# 如果文件里有 selected 列，只保留 selected=1 的样本
if "selected" in cases_df.columns:
    cases_df["selected"] = pd.to_numeric(
        cases_df["selected"],
        errors="coerce"
    ).fillna(0).astype(int)

    cases_df = cases_df[cases_df["selected"] == 1].copy()

# group 使用 final_group
if "final_group" in cases_df.columns:
    cases_df["group"] = cases_df["final_group"].astype(str)
elif "candidate_group" in cases_df.columns:
    cases_df["group"] = cases_df["candidate_group"].astype(str)
elif "group" in cases_df.columns:
    cases_df["group"] = cases_df["group"].astype(str)
else:
    cases_df["group"] = "Representative"

cases_df["group"] = cases_df["group"].apply(safe_group_name)

# image path
if fname_col == "image_path":
    cases_df["image_path_for_use"] = cases_df[fname_col].apply(find_image_path)
else:
    cases_df["image_path_for_use"] = cases_df[fname_col].apply(find_image_path)

# numerical columns
cases_df["y_true_for_use"] = cases_df[true_col].astype(float)
cases_df["pred_for_use"] = cases_df[pred_col].astype(float)
cases_df["width_for_use"] = cases_df[width_col].astype(float)

if lower_col is not None:
    cases_df["lower_for_use"] = cases_df[lower_col].astype(float)
else:
    cases_df["lower_for_use"] = np.nan

if upper_col is not None:
    cases_df["upper_for_use"] = cases_df[upper_col].astype(float)
else:
    cases_df["upper_for_use"] = np.nan

if "abs_error" in cases_df.columns:
    cases_df["abs_error_for_use"] = cases_df["abs_error"].astype(float)
else:
    cases_df["abs_error_for_use"] = np.abs(
        cases_df["y_true_for_use"] - cases_df["pred_for_use"]
    )

# irradiance
if i_col is not None:
    cases_df["I_for_use"] = cases_df[i_col].astype(float)
else:
    print("No I column detected. Trying to parse I from filename...")
    cases_df["I_for_use"] = cases_df[fname_col].apply(parse_i_from_filename)

missing_img = cases_df["image_path_for_use"].isna().sum()
missing_i = cases_df["I_for_use"].isna().sum()

print("After selected filtering N =", len(cases_df))
print("Missing image paths:", missing_img)
print("Missing I values:", missing_i)

if missing_img > 0:
    p = os.path.join(OUT_DIR, "missing_image_paths.csv")
    cases_df[cases_df["image_path_for_use"].isna()].to_csv(p, index=False)
    print("Saved missing image path list:", p)

if missing_i > 0:
    p = os.path.join(OUT_DIR, "missing_i_values.csv")
    cases_df[cases_df["I_for_use"].isna()].to_csv(p, index=False)
    print("Saved missing I list:", p)

cases_df = cases_df.dropna(
    subset=["image_path_for_use", "I_for_use"]
).copy()

if len(cases_df) == 0:
    raise RuntimeError(
        "No valid selected cases left. "
        "Check representative_selected_cases.csv and image/I columns."
    )

# 去重，保持人工选择顺序
cases_df = cases_df.drop_duplicates("image_path_for_use").reset_index(drop=True)

selected_out = os.path.join(OUT_DIR, "gradcam_input_cases_used.csv")
cases_df.to_csv(selected_out, index=False)

print("\nCases used for Grad-CAM:")
print(
    cases_df[
        ["group", fname_col, "y_true_for_use", "pred_for_use",
         "abs_error_for_use", "width_for_use", "I_for_use"]
    ]
)

print("Saved cases used:", selected_out)


# ============================================================
# 7. Load model
# ============================================================
model = SolarResNet50WithI(dropout=DROPOUT).to(DEVICE)
model = load_checkpoint(model, CKPT_PATH)
model.eval()

target_layer = get_target_layer(model, TARGET_LAYER_NAME)
cam_generator = RegressionGradCAM(model, target_layer)

print("\nDevice:", DEVICE)
print("Target layer:", TARGET_LAYER_NAME)
print("Colormap:", CAM_CMAP)


# ============================================================
# 8. Generate Grad-CAM results
# ============================================================
case_images = []
results = []

for idx, row in cases_df.iterrows():
    img_path = row["image_path_for_use"]
    i_value = float(row["I_for_use"])

    img_resized, image_tensor, rgb = prepare_image(img_path)
    i_tensor = torch.tensor([i_value], dtype=torch.float32)

    cam, pred_runtime = cam_generator(image_tensor, i_tensor)
    heatmap, overlay = make_overlay(
        rgb,
        cam,
        alpha=OVERLAY_ALPHA,
        cmap_name=CAM_CMAP
    )

    group = str(row["group"])
    case_id = row["case_id"] if "case_id" in row.index else f"case_{idx:02d}"
    safe_base = file_safe_name(row[fname_col])

    if SAVE_INDIVIDUAL_IMAGES:
        original_path = os.path.join(
            OUT_DIR,
            f"{idx:02d}_{case_id}_{safe_base}_original.png"
        )
        heatmap_path = os.path.join(
            OUT_DIR,
            f"{idx:02d}_{case_id}_{safe_base}_heatmap.png"
        )
        overlay_path = os.path.join(
            OUT_DIR,
            f"{idx:02d}_{case_id}_{safe_base}_overlay.png"
        )

        plt.imsave(original_path, rgb)
        plt.imsave(heatmap_path, heatmap)
        plt.imsave(overlay_path, overlay)
    else:
        original_path = ""
        heatmap_path = ""
        overlay_path = ""

    results.append({
        "case_index": idx,
        "case_id": case_id,
        "filename": row[fname_col],
        "group": group,
        "image_path": img_path,
        "I_value": i_value,
        "y_true": float(row["y_true_for_use"]),
        "pred_in_csv": float(row["pred_for_use"]),
        "pred_runtime": pred_runtime,
        "abs_error": float(row["abs_error_for_use"]),
        "width_mc_conf_90": float(row["width_for_use"]),
        "lower_mc_conf_90": float(row["lower_for_use"]) if not pd.isna(row["lower_for_use"]) else np.nan,
        "upper_mc_conf_90": float(row["upper_for_use"]) if not pd.isna(row["upper_for_use"]) else np.nan,
        "target_layer": TARGET_LAYER_NAME,
        "original_png": original_path,
        "heatmap_png": heatmap_path,
        "overlay_png": overlay_path,
    })

    case_images.append({
        "row": row,
        "case_id": case_id,
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
# 9. Plot 3-column panel: Original | Heatmap | Overlay
# ============================================================
def plot_gradcam_panel(case_images, name="fig1_gradcam_representative_panel"):
    n_cases = len(case_images)
    ncols = 3
    nrows = n_cases

    fig_w = 9.8
    fig_h = max(2.35 * nrows, 5.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if nrows == 1:
        axes = np.array([axes])

    axes = axes.reshape(nrows, ncols)

    col_titles = ["Original", "Grad-CAM", "Overlay"]

    for c in range(ncols):
        axes[0, c].set_title(
            col_titles[c],
            fontsize=12,
            fontweight="bold",
            pad=8
        )

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
        color = group_color(group)

        label = (
            f"{group}\n"
            f"True={row['y_true_for_use']:.3f}, Pred={row['pred_for_use']:.3f}\n"
            f"Err={row['abs_error_for_use']:.3f}, Width={row['width_for_use']:.3f}"
        )

        axes[r, 0].text(
            -0.06,
            0.50,
            label,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.8,
            color=color,
            fontweight="bold"
        )

    fig.suptitle(
        "Representative qualitative examples with Grad-CAM explanations",
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


# ============================================================
# 10. Plot compact overlay-only figure
# ============================================================
def plot_overlay_only(case_images, name="fig2_gradcam_overlay_compact"):
    n = len(case_images)
    ncols = 4 if n >= 8 else 3
    nrows = math.ceil(n / ncols)

    fig_w = 12.0 if ncols == 4 else 10.0
    fig_h = nrows * 3.25 + 0.6

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if nrows == 1:
        axes = np.array([axes])

    axes = axes.reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for i, item in enumerate(case_images):
        r = i // ncols
        c = i % ncols

        ax = axes[r, c]
        row = item["row"]
        overlay = item["overlay"]

        group = str(row["group"])
        color = group_color(group)

        ax.imshow(overlay)
        ax.axis("off")

        ax.text(
            0.02,
            0.98,
            group,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            color="white",
            bbox=dict(
                facecolor=color,
                edgecolor="none",
                alpha=0.80,
                boxstyle="round,pad=0.22"
            )
        )

        caption = (
            f"True={row['y_true_for_use']:.3f}, Pred={row['pred_for_use']:.3f}\n"
            f"Err={row['abs_error_for_use']:.3f}, Width={row['width_for_use']:.3f}"
        )

        ax.text(
            0.5,
            -0.075,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.4,
            color="#111111"
        )

    fig.suptitle(
        "Grad-CAM overlays for manually selected representative cases",
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
        hspace=0.34
    )

    save_fig(fig, name)


# ============================================================
# 11. Optional: grouped overlay figure by sample type
# ============================================================
def plot_grouped_overlay(case_images, name="fig3_gradcam_grouped_overlay"):
    groups = []
    for item in case_images:
        g = str(item["row"]["group"])
        if g not in groups:
            groups.append(g)

    max_per_group = max(
        sum(str(item["row"]["group"]) == g for item in case_images)
        for g in groups
    )

    nrows = len(groups)
    ncols = max_per_group

    fig_w = max(10.5, ncols * 3.0)
    fig_h = nrows * 3.0 + 0.6

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    axes = axes.reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for r, g in enumerate(groups):
        group_items = [
            item for item in case_images
            if str(item["row"]["group"]) == g
        ]

        for c, item in enumerate(group_items):
            ax = axes[r, c]
            row = item["row"]
            overlay = item["overlay"]

            ax.imshow(overlay)
            ax.axis("off")

            caption = (
                f"True={row['y_true_for_use']:.3f}, Pred={row['pred_for_use']:.3f}\n"
                f"Err={row['abs_error_for_use']:.3f}, Width={row['width_for_use']:.3f}"
            )

            ax.text(
                0.5,
                -0.08,
                caption,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.0,
                color="#111111"
            )

        axes[r, 0].text(
            -0.08,
            0.5,
            g,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
            color=group_color(g),
            fontweight="bold"
        )

    fig.suptitle(
        "Grouped Grad-CAM overlays by representative condition",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.92,
        bottom=0.06,
        wspace=0.08,
        hspace=0.36
    )

    save_fig(fig, name)


# ============================================================
# 12. Run plotting
# ============================================================
plot_gradcam_panel(
    case_images,
    name=f"fig1_gradcam_representative_panel_{TARGET_LAYER_NAME.replace('[','').replace(']','')}"
)

plot_overlay_only(
    case_images,
    name=f"fig2_gradcam_overlay_compact_{TARGET_LAYER_NAME.replace('[','').replace(']','')}"
)

plot_grouped_overlay(
    case_images,
    name=f"fig3_gradcam_grouped_overlay_{TARGET_LAYER_NAME.replace('[','').replace(']','')}"
)

print("\nAll Grad-CAM outputs saved to:")
print(OUT_DIR)