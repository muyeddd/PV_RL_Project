
"""
Paper1 full figure/table library generator (v2)
==============================================
Purpose
-------
Generate the full Paper1 figure/table library for:
1) progress / group-meeting reporting
2) later screening into submission-quality Paper1 figures

This script is REPORT-ONLY:
- reads frozen CSV outputs
- does NOT retrain, recalibrate, or alter formal results
- skips gracefully when a required source file is absent

This is a candidate reporting library. A generated figure/table is not
automatically paper-ready. Each item must pass scientific-source audit and
visual review before use.

Run from repository root:
    python paper1_make_all_figures_tables_v2.py

Main output:
    outputs/paper1_clean_random_v1/paper_figures_tables_v2/
        figures/
        tables/

Expected runtime:
- usually 20–120 s
- depends mainly on loading predictions.csv and saving many high-resolution files
- no GPU / no network / no Codex needed
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path.cwd()
PROTOCOL = "paper1_clean_random_v1"
FINAL_RT = ROOT / "outputs" / PROTOCOL / "final_stage6_v1" / "random_test"
FINAL_SD = ROOT / "outputs" / PROTOCOL / "final_stage6_v1" / "sealed_dates"
DATE_GROUPED_METRICS_PATH = ROOT / "outputs" / "date_grouped_v1" / "resnet50_with_i" / "formal_cv" / "cv_summary.csv"
MC_STAGE1A_DIR = ROOT / "outputs" / PROTOCOL / "uq_stage1a_inference_v1"
MC_STAGE1B_METRICS_PATH = ROOT / "outputs" / PROTOCOL / "uq_stage1b_intervals_v1" / "all_interval_metrics.csv"
OUT_ROOT = ROOT / "outputs" / PROTOCOL / "paper_figures_tables_v2"
FIG_DIR = OUT_ROOT / "figures"
TAB_DIR = OUT_ROOT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_ROOTS = [ROOT / "outputs" / PROTOCOL, ROOT / "experiments", ROOT]

DATA_SPLIT = [
    ("TRAIN", 25830, "Model fitting"),
    ("MODEL_VALIDATION", 3692, "Model selection"),
    ("CP_CALIBRATION", 2951, "Conformal calibration"),
    ("DECISION_DEVELOPMENT", 1844, "Decision development"),
    ("RANDOM_TEST", 2582, "Untouched same-domain final test"),
    ("SEALED_DATES", 8855, "Unseen-date stress test"),
]
VAL_POINT = dict(R2=0.9319799542, RMSE=0.0750622129, MAE=0.0372662693, Bias=-0.0027940)
VAL_Q50   = dict(R2=0.93023282, RMSE=0.0760201, MAE=0.0352178, Bias=np.nan)
METHOD_COMPARISON = pd.DataFrame([
    ["Raw MC",                   0.54826, 0.04853, 0.46323],
    ["Split CP",                 0.90727, 0.15594, 0.33542],
    ["Irradiance Mondrian",      0.90998, 0.15987, 0.32586],
    ["Pred-loss Mondrian",       0.90510, 0.15885, 0.32201],
    ["Pred-loss Mondrian + MC",  0.90618, 0.15311, 0.31642],
    ["Std-MC conformal",         0.89859, 0.15136, 0.30173],
    ["CQR",                      0.9105206074, 0.1336699080, 0.2745067344],
], columns=["Method", "PICP", "MPIW", "IntervalScore"])
CQR_DEV = dict(PICP=0.9105206074, MPIW=0.1336699080, MedianWidth=0.1156324091, CoverageError=0.0105206074, IntervalScore=0.2745067344)
QHAT = 0.004862844288256299
TARGET_COVERAGE = 0.90

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.labelsize": 9.2,
    "axes.titlesize": 10.0,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.32,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FFBF79", "#9D755D"]
DARK = "#222222"
GRID = "#DADADA"

# Fig01--Fig04 use an isolated Chinese academic-paper style so that Fig05 and
# later figures retain their original appearance.
FIG01_04_RC = {
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ],
    "font.size": 9.0,
    "axes.titlesize": 12.0,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.5,
    "axes.edgecolor": "#4A4A4A",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.color": "#D9DEE3",
    "grid.linewidth": 0.45,
    "grid.alpha": 0.55,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

def style_ax(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)

def savefig(fig, stem: str):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)

def save_table(df: pd.DataFrame, stem: str, caption: str, formats: Optional[dict]=None):
    out = df.copy()
    if formats:
        for c, fmt in formats.items():
            if c in out.columns:
                out[c] = out[c].map(lambda x: "" if pd.isna(x) else fmt.format(x))
    out.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    try:
        out.to_latex(TAB_DIR / f"{stem}.tex", index=False, escape=False, caption=caption, label=f"tab:{stem}")
    except Exception:
        pass
    nrows, ncols = out.shape
    width = max(6.0, min(11.5, 1.25*ncols + 1.1))
    height = max(1.9, 0.36*(nrows+2))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    table = ax.table(cellText=out.values, colLabels=out.columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.28)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.0)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F2F2F2")
        else:
            cell.set_facecolor("white")
    ax.set_title(caption, fontsize=9.5, pad=10)
    ax.plot([0.01, 0.99], [0.92, 0.92], transform=ax.transAxes, color=DARK, linewidth=0.8)
    ax.plot([0.01, 0.99], [0.10, 0.10], transform=ax.transAxes, color=DARK, linewidth=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(TAB_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)

def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None

def load_date_grouped_fold_metrics() -> Optional[pd.DataFrame]:
    """Load only the formal date-grouped CV fold rows; never infer metrics."""
    df = read_csv_if_exists(DATE_GROUPED_METRICS_PATH)
    required = {"row_type", "fold", "validation_r2", "validation_rmse", "validation_mae"}
    if df is None or df.empty or not required.issubset(df.columns):
        return None
    folds = df.loc[df["row_type"].astype(str).str.lower() == "fold"].copy()
    for column in ["fold", "validation_r2", "validation_rmse", "validation_mae"]:
        folds[column] = pd.to_numeric(folds[column], errors="coerce")
    if (
        len(folds) != 4
        or set(folds["fold"].dropna().astype(int)) != {1, 2, 3, 4}
        or folds[["fold", "validation_r2", "validation_rmse", "validation_mae"]].isna().any().any()
    ):
        return None
    out = folds[["fold", "validation_r2", "validation_rmse", "validation_mae"]].rename(
        columns={"fold": "Fold", "validation_r2": "R2", "validation_rmse": "RMSE", "validation_mae": "MAE"}
    )
    bias_column = next((c for c in ("validation_bias", "bias") if c in folds.columns), None)
    if bias_column is not None:
        bias = pd.to_numeric(folds[bias_column], errors="coerce")
        if bias.notna().all():
            out["Bias"] = bias.to_numpy()
    out["Fold"] = out["Fold"].astype(int)
    return out.sort_values("Fold").reset_index(drop=True)

MC_ROLE_FILES = {
    "CP_CALIBRATION": "cp_calibration_predictions.csv",
    "DECISION_DEVELOPMENT": "decision_development_predictions.csv",
}

def load_mc_role_predictions(role: str) -> Optional[pd.DataFrame]:
    """Load a Stage1A sample-level MC file only when its role and schema agree."""
    file_name = MC_ROLE_FILES.get(role)
    if file_name is None:
        return None
    df = read_csv_if_exists(MC_STAGE1A_DIR / file_name)
    required = ["role", "true_L", "point_pred", "mc_mean", "mc_std"]
    if df is None or df.empty or any(c not in df.columns for c in required):
        return None
    if set(df["role"].astype(str).unique()) != {role}:
        return None
    numeric = df[["true_L", "point_pred", "mc_mean", "mc_std"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        return None
    out = df.copy()
    out[["true_L", "point_pred", "mc_mean", "mc_std"]] = numeric
    return out

def load_available_mc_roles():
    return [(role, df) for role in MC_ROLE_FILES if (df := load_mc_role_predictions(role)) is not None]

def load_raw_mc_picp_by_role() -> dict:
    """Load role-labelled raw-MC PICP from Stage1B when explicitly available."""
    df = read_csv_if_exists(MC_STAGE1B_METRICS_PATH)
    required = {"method", "evaluation_role", "PICP"}
    if df is None or df.empty or not required.issubset(df.columns):
        return {}
    rows = df.loc[df["method"].astype(str).str.lower() == "raw_mc", ["evaluation_role", "PICP"]].copy()
    rows["PICP"] = pd.to_numeric(rows["PICP"], errors="coerce")
    rows = rows.dropna(subset=["PICP"])
    if rows["evaluation_role"].duplicated().any():
        return {}
    return dict(zip(rows["evaluation_role"].astype(str), rows["PICP"].astype(float)))

def pick_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    for c in df.columns:
        lc = c.lower()
        for n in names:
            if n.lower() in lc:
                return c
    return None

def finite_xy(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def metric_r2(y, p):
    y, p = finite_xy(y, p)
    if len(y) < 2: return np.nan
    denom = np.sum((y-y.mean())**2)
    return 1 - np.sum((y-p)**2)/denom if denom > 0 else np.nan

def metric_rmse(y, p):
    y, p = finite_xy(y, p)
    return float(np.sqrt(np.mean((p-y)**2))) if len(y) else np.nan

def metric_mae(y, p):
    y, p = finite_xy(y, p)
    return float(np.mean(np.abs(p-y))) if len(y) else np.nan

def prediction_cols(df: pd.DataFrame):
    true_c = pick_col(df, ["true_L", "true", "y_true", "target", "label"])
    point_c = pick_col(df, ["point_pred", "prediction", "pred", "y_pred"])
    q05_c = pick_col(df, ["q05", "q_05", "q0.05", "lower_raw", "raw_lower"])
    q50_c = pick_col(df, ["q50", "q_50", "median_pred", "median"])
    q95_c = pick_col(df, ["q95", "q_95", "q0.95", "upper_raw", "raw_upper"])
    low_c = pick_col(df, ["cqr_lower", "lower", "final_lower", "interval_lower"])
    up_c = pick_col(df, ["cqr_upper", "upper", "final_upper", "interval_upper"])
    irr_c = pick_col(df, ["irradiance", "irradiance_normalized", "I"])
    date_c = pick_col(df, ["date"])
    return true_c, point_c, q05_c, q50_c, q95_c, low_c, up_c, irr_c, date_c

def load_random_predictions(): return read_csv_if_exists(FINAL_RT / "predictions.csv")
def load_random_prediction_metrics(): return read_csv_if_exists(FINAL_RT / "prediction_metrics.csv")
def load_random_interval_metrics(): return read_csv_if_exists(FINAL_RT / "interval_metrics.csv")
def load_decision_metrics(): return read_csv_if_exists(FINAL_RT / "decision_metrics.csv")
def load_economic_metrics(): return read_csv_if_exists(FINAL_RT / "economic_metrics.csv")
def load_break_even(): return read_csv_if_exists(FINAL_RT / "break_even_review_cost.csv")

def discover_validation_predictions():
    for root in SEARCH_ROOTS:
        if not root.exists(): continue
        for p in root.rglob("*.csv"):
            s = str(p).lower()
            if "prediction" in p.name.lower() and ("validation" in s or "model_validation" in s or "valid" in p.name.lower()):
                df = read_csv_if_exists(p)
                if df is None or df.empty: continue
                t, pred, *_ = prediction_cols(df)
                if t and pred: return df
    return None

def discover_history():
    for root in SEARCH_ROOTS:
        if not root.exists(): continue
        for p in root.rglob("*.csv"):
            name = p.name.lower()
            if any(k in name for k in ("history", "epoch", "training", "log", "metrics")):
                df = read_csv_if_exists(p)
                if df is None or df.empty: continue
                ep = pick_col(df, ["epoch"])
                tr = pick_col(df, ["train_loss", "training_loss", "loss_train"])
                va = pick_col(df, ["val_loss", "validation_loss", "loss_val"])
                if ep and tr and va:
                    return df[[ep, tr, va]].rename(columns={ep:"epoch", tr:"train_loss", va:"val_loss"})
    return None

def get_cqr_arrays(df):
    t, p, q05, q50, q95, lo, up, irr, date = prediction_cols(df)
    if not t or not q50: return None
    true = df[t].to_numpy(float)
    med = df[q50].to_numpy(float)
    raw_lo = df[q05].to_numpy(float) if q05 else None
    raw_up = df[q95].to_numpy(float) if q95 else None
    fin_lo = df[lo].to_numpy(float) if lo else (np.clip(raw_lo - QHAT, 0, 1) if raw_lo is not None else None)
    fin_up = df[up].to_numpy(float) if up else (np.clip(raw_up + QHAT, 0, 1) if raw_up is not None else None)
    irr_v = df[irr].to_numpy(float) if irr else None
    date_v = df[date].astype(str).to_numpy() if date else None
    return true, med, raw_lo, raw_up, fin_lo, fin_up, irr_v, date_v

def draw_box(ax, x, y, w, h, text, fc="#F7F9FC"):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor="#6B7280", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8.2)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", linewidth=1.0, color="#4B5563"))

# --- figures helpers ---
def simple_scatter_plot(df, stem, title, metric_text=None):
    t, pred, *_ = prediction_cols(df)
    if not t or not pred:
        print(f"[SKIP] {stem}: true/pred columns not found."); return
    y, pr = finite_xy(df[t], df[pred])
    if len(y) == 0:
        print(f"[SKIP] {stem}: no finite data."); return
    err = np.abs(pr-y)
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    sc = ax.scatter(y, pr, c=err, s=10, alpha=0.55, cmap="viridis", linewidths=0)
    lo = min(float(y.min()), float(pr.min()), 0.0); hi = max(float(y.max()), float(pr.max()), 1.0)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.25)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual relative power loss"); ax.set_ylabel("Predicted relative power loss")
    ax.set_title(title); style_ax(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Absolute error")
    if metric_text is None:
        metric_text = f"$R^2$={metric_r2(y,pr):.3f}\nRMSE={metric_rmse(y,pr):.3f}\nMAE={metric_mae(y,pr):.3f}"
    ax.text(0.03, 0.97, metric_text, transform=ax.transAxes, ha="left", va="top", bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BBBBBB", alpha=0.92), fontsize=8.3)
    savefig(fig, stem)

def _plot_conditional(file_name, stem, title):
    f = read_csv_if_exists(FINAL_RT / file_name)
    if f is None:
        print(f"[SKIP] {stem}: {file_name} not found."); return
    bl = pick_col(f, ["bin_label"]); picp = pick_col(f, ["PICP"])
    if not bl or not picp:
        print(f"[SKIP] {stem}: required columns not found."); return
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(np.arange(len(f)), f[picp].to_numpy(float), marker="o", linewidth=1.8)
    ax.axhline(TARGET_COVERAGE, linestyle="--", linewidth=1.1, label="Target = 0.90")
    ax.set_xticks(np.arange(len(f)), f[bl].astype(str), rotation=20, ha="right")
    ax.set_ylabel("PICP"); ax.set_ylim(0.8, 0.97); ax.set_title(title); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, stem)

# --- Figures 1-42 ---
def fig01_framework():
    with plt.rc_context(FIG01_04_RC):
        fig, ax = plt.subplots(figsize=(10.4, 2.75))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        xs = [0.015, 0.215, 0.415, 0.615, 0.815]
        box_w, box_h, box_y = 0.17, 0.50, 0.22
        labels = [
            "组件图像\n＋ 辐照度",
            "点预测模型\n（ResNet50＋辐照度）",
            "不确定性表征\n蒙特卡洛法（MC）\n保形预测（CP）\n保形分位数回归（CQR）",
            "清洗—等待—复核\n三态决策",
            "经济性评估\n（后悔值／成本）",
        ]
        fills = ["#F3F6F8", "#EEF4F7", "#EEF3F5", "#F2F5F2", "#F7F3ED"]
        for x, label, fill in zip(xs, labels, fills):
            ax.add_patch(Rectangle(
                (x, box_y), box_w, box_h,
                facecolor=fill, edgecolor="#64727D", linewidth=0.9,
            ))
            ax.text(
                x + box_w / 2, box_y + box_h / 2, label,
                ha="center", va="center", color="#263238",
                fontsize=8.8, linespacing=1.38,
            )
        for i in range(len(xs) - 1):
            ax.annotate(
                "", xy=(xs[i + 1] - 0.008, 0.47),
                xytext=(xs[i] + box_w + 0.008, 0.47),
                arrowprops=dict(
                    arrowstyle="-|>", color="#64727D",
                    linewidth=1.0, mutation_scale=10,
                    shrinkA=0, shrinkB=0,
                ),
            )
        ax.set_title("图1  Paper1总体研究框架", pad=10)
        fig.subplots_adjust(left=0.025, right=0.985, bottom=0.08, top=0.82)
        savefig(fig, "Fig01_overall_framework")

def fig02_problem_to_redesign():
    with plt.rc_context(FIG01_04_RC):
        fig, ax = plt.subplots(figsize=(9.0, 4.05))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        box_y, box_h, box_w = 0.14, 0.68, 0.40
        panels = [
            (
                0.035, "#FAF2F1", "#A67570", "前期方案存在的问题",
                [
                    "随机帧划分不够严格",
                    "仅有高决定系数 R² 不足以支撑可信决策",
                    "缺少显式不确定性量化",
                    "测试集不应参与方法选择",
                ],
            ),
            (
                0.565, "#EFF5F7", "#6A8793", "重新设计后的正式协议",
                [
                    "按角色严格划分数据集",
                    "单独设置保形校准集",
                    "冻结随机测试集用于最终评估",
                    "构建“预测—决策—经济性”完整链条",
                ],
            ),
        ]
        for x, fill, edge, heading, items in panels:
            ax.add_patch(Rectangle(
                (x, box_y), box_w, box_h,
                facecolor=fill, edgecolor=edge, linewidth=0.95,
            ))
            ax.text(
                x + box_w / 2, box_y + box_h - 0.105, heading,
                ha="center", va="center", fontsize=10.2,
                fontweight="semibold", color="#263238",
            )
            body = "\n".join(f"•  {item}" for item in items)
            ax.text(
                x + 0.038, box_y + box_h - 0.205, body,
                ha="left", va="top", fontsize=9.0,
                color="#30383D", linespacing=1.85,
            )
        ax.annotate(
            "", xy=(0.545, 0.48), xytext=(0.455, 0.48),
            arrowprops=dict(
                arrowstyle="-|>", color="#65737C",
                linewidth=1.2, mutation_scale=11,
                shrinkA=0, shrinkB=0,
            ),
        )
        ax.text(0.50, 0.525, "重构", ha="center", va="bottom", fontsize=8.2, color="#65737C")
        ax.set_title("图2  研究方案重构的原因与改进思路", pad=10)
        fig.subplots_adjust(left=0.025, right=0.985, bottom=0.06, top=0.85)
        savefig(fig, "Fig02_earlier_limitations_and_redesign")

def fig03_data_split():
    fig03_rc = {
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "font.size": 8.7,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.8,
        "axes.edgecolor": "#767676",
        "axes.linewidth": 0.55,
        "xtick.color": "#404040",
        "ytick.color": "#303030",
        "xtick.major.width": 0.55,
        "xtick.major.size": 2.8,
        "grid.color": "#DCE1E5",
        "grid.linewidth": 0.42,
        "grid.alpha": 0.58,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(fig03_rc):
        labels = ["训练集", "模型验证集", "保形校准集", "决策开发集", "随机测试集", "未见日期测试集"]
        counts = np.array([r[1] for r in DATA_SPLIT], dtype=int)
        expected_counts = np.array([25830, 3692, 2951, 1844, 2582, 8855], dtype=int)
        protocol_shares = np.array([70, 10, 8, 5, 7], dtype=int)
        if not np.array_equal(counts, expected_counts):
            raise ValueError("Fig03 data do not match the frozen Paper1 partition.")
        if counts[:5].sum() != 36899:
            raise ValueError("Fig03 main-protocol total must be N=36,899.")
        if protocol_shares.sum() != 100:
            raise ValueError("Fig03 main-protocol shares must sum to 100%.")

        role_blue = "#758C9C"
        test_blue = "#3F647E"
        stress_purple = "#A8A4AE"
        neutral_text = "#30343A"
        secondary_text = "#666B70"

        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(10.6, 3.55),
            gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.30},
        )

        # (a) Frozen sample counts for the five protocol roles and the separate
        # unseen-date distribution-shift stress test.
        y_positions = np.array([5.0, 4.0, 3.0, 2.0, 1.0, -0.35])
        count_colors = [role_blue] * 4 + [test_blue, stress_purple]
        bars = ax_a.barh(
            y_positions, counts, height=0.46,
            color=count_colors, edgecolor="white", linewidth=0.45, zorder=3,
        )
        ax_a.set_yticks(y_positions, labels)
        ax_a.set_xlabel("样本数量", labelpad=7)
        ax_a.set_xticks([0, 10000, 20000, 30000])
        ax_a.xaxis.set_major_formatter(lambda value, _pos: f"{int(value):,}")
        ax_a.grid(axis="x")
        ax_a.set_axisbelow(True)
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        ax_a.spines["left"].set_visible(False)
        ax_a.spines["bottom"].set_color("#888888")
        ax_a.tick_params(axis="y", length=0, pad=7)
        ax_a.tick_params(axis="x", pad=3)
        ax_a.axhline(
            0.30, color="#BFC2C7", linewidth=0.65,
            linestyle=(0, (3, 3)), zorder=1,
        )
        for bar, n in zip(bars, counts):
            ax_a.text(
                n + 520,
                bar.get_y() + bar.get_height() / 2,
                f"{n:,}", va="center", ha="left",
                fontsize=8.1, color=neutral_text, clip_on=False,
            )
        ax_a.set_xlim(0, 30200)
        ax_a.set_ylim(-0.90, 5.65)
        ax_a.text(
            -0.22, 1.025, "(a)", transform=ax_a.transAxes,
            ha="left", va="bottom", fontsize=10.0,
            fontweight="semibold", color=neutral_text,
        )

        # (b) The 100% main protocol is kept separate from the auxiliary
        # unseen-date test; repeated fills follow the role grouping in panel (a).
        protocol_roles = ["训练", "验证", "校准", "开发", "测试"]
        protocol_colors = [role_blue] * 4 + [test_blue]
        starts = np.r_[0, np.cumsum(protocol_shares[:-1])]
        centers = starts + protocol_shares / 2
        main_y = 2.75
        main_height = 0.62
        for start, share, color in zip(starts, protocol_shares, protocol_colors):
            ax_b.barh(
                main_y, share, left=start, height=main_height,
                color=color, edgecolor="white", linewidth=0.7, zorder=3,
            )
        for center, share in zip(centers, protocol_shares):
            ax_b.text(
                center, main_y, f"{share}%",
                ha="center", va="center", fontsize=8.0,
                fontweight="semibold", color="white", zorder=4,
            )

        role_label_y = np.array([3.55, 3.55, 3.88, 3.55, 3.88])
        for center, role, text_y in zip(centers, protocol_roles, role_label_y):
            bar_edge_y = main_y + main_height / 2
            ax_b.plot(
                [center, center], [bar_edge_y, text_y - 0.12],
                color="#A8ADB2", linewidth=0.5, zorder=2,
            )
            ax_b.text(
                center, text_y, role, ha="center", va="center",
                fontsize=8.0, color=neutral_text,
            )

        ax_b.text(
            0, 4.18, "主实验协议  N=36,899",
            ha="left", va="center", fontsize=8.8,
            fontweight="medium", color=neutral_text,
        )
        ax_b.barh(
            1.10, 8.0, left=0, height=0.22,
            color=stress_purple, edgecolor="none", zorder=3,
        )
        ax_b.text(
            10.5, 1.10, "未见日期压力测试  N=8,855",
            ha="left", va="center", fontsize=8.4,
            fontweight="medium", color=neutral_text,
        )
        ax_b.text(
            10.5, 0.65, "额外压力测试，不参与模型开发与校准",
            ha="left", va="center", fontsize=7.7, color=secondary_text,
        )
        ax_b.set_xlim(-1.0, 101.0)
        ax_b.set_ylim(0.25, 4.50)
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        for spine in ax_b.spines.values():
            spine.set_visible(False)
        ax_b.text(
            0.0, 1.025, "(b)", transform=ax_b.transAxes,
            ha="left", va="bottom", fontsize=10.0,
            fontweight="semibold", color=neutral_text,
        )

        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.91)
        savefig(fig, "Fig03_data_partition_protocol")

def fig04_date_distribution():
    with plt.rc_context(FIG01_04_RC):
        counts = {}
        sources = [
            load_random_predictions(),
            read_csv_if_exists(FINAL_SD / "predictions.csv"),
            discover_validation_predictions(),
        ]
        for source in sources:
            if source is None:
                continue
            *_, date_col = prediction_cols(source)
            if not date_col:
                continue
            date_values = source[date_col].astype(str)
            parsed_dates = pd.to_datetime(date_values, errors="coerce")
            display_dates = date_values.copy()
            valid = parsed_dates.notna()
            display_dates.loc[valid] = parsed_dates.loc[valid].dt.strftime("%Y-%m-%d")
            for date_text, n in display_dates.value_counts().items():
                counts[date_text] = counts.get(date_text, 0) + int(n)
        if not counts:
            print("[SKIP] Fig04: no discoverable date column found in available CSVs.")
            return

        sample_counts = pd.Series(counts).sort_index()
        unseen_dates = {"2017-06-15", "2017-06-24", "2017-06-30"}
        development_color = "#6F8FA8"
        unseen_color = "#C17B5D"
        bar_colors = [unseen_color if date in unseen_dates else development_color for date in sample_counts.index]
        x = np.arange(len(sample_counts))
        fig_width = max(8.4, min(12.2, 0.58 * len(sample_counts) + 2.8))
        fig, ax = plt.subplots(figsize=(fig_width, 4.25))
        bars = ax.bar(
            x, sample_counts.values, width=0.72,
            color=bar_colors, edgecolor="white", linewidth=0.65,
        )
        ax.set_xticks(x, sample_counts.index, rotation=38, ha="right", rotation_mode="anchor")
        ax.set_xlabel("采集日期", labelpad=8)
        ax.set_ylabel("样本数量", labelpad=7)
        ax.set_title("图4  不同采集日期的样本数量分布", pad=11)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_formatter(lambda value, _pos: f"{int(value):,}")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#707070")
        ax.spines["bottom"].set_color("#707070")
        ax.tick_params(axis="x", labelsize=8.0, length=3, pad=4)
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=development_color, edgecolor="none", label="开发阶段涉及日期"),
            Rectangle((0, 0), 1, 1, facecolor=unseen_color, edgecolor="none", label="未见日期测试"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, ncol=2)

        label_all_bars = len(sample_counts) <= 18
        offset = max(float(sample_counts.max()) * 0.016, 1.0)
        for bar, date_text, n in zip(bars, sample_counts.index, sample_counts.values):
            if label_all_bars or date_text in unseen_dates:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{int(n):,}",
                    ha="center", va="bottom", fontsize=7.3,
                    color="#4A4A4A",
                )
        ax.margins(x=0.015, y=0.13)
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.28, top=0.86)
        savefig(fig, "Fig04_date_sample_distribution")

def fig05_date_role_heatmap():
    print("[SKIP] Fig05: role-by-date heatmap requires explicit per-role date manifests.")

def fig06_model_structure():
    fig, ax = plt.subplots(figsize=(9.0, 2.8)); ax.axis("off")
    draw_box(ax, 0.04, 0.28, 0.16, 0.42, "Panel image")
    draw_box(ax, 0.28, 0.28, 0.18, 0.42, "ResNet50\nimage encoder", fc="#EFF6FF")
    draw_box(ax, 0.54, 0.52, 0.13, 0.18, "Irradiance", fc="#FEF3C7")
    draw_box(ax, 0.54, 0.24, 0.13, 0.18, "Feature fusion", fc="#F3E8FF")
    draw_box(ax, 0.77, 0.28, 0.16, 0.42, "Regression head\n→ relative loss")
    draw_arrow(ax, 0.20, 0.49, 0.28, 0.49); draw_arrow(ax, 0.46, 0.49, 0.54, 0.33); draw_arrow(ax, 0.605, 0.52, 0.605, 0.42); draw_arrow(ax, 0.67, 0.33, 0.77, 0.49)
    ax.set_title("ResNet50 + irradiance regression model", fontsize=10.1, pad=10)
    savefig(fig, "Fig06_model_structure")

def fig07_training_curve():
    df = discover_history()
    if df is None:
        print("[SKIP] Fig07: no training-history CSV with epoch/train_loss/val_loss found."); return
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.plot(df["epoch"], df["train_loss"], marker="o", markersize=2.8, label="Training")
    ax.plot(df["epoch"], df["val_loss"], marker="s", markersize=2.6, label="Validation")
    best_i = int(np.nanargmin(df["val_loss"].to_numpy(float))); best_ep = df.iloc[best_i]["epoch"]
    ax.axvline(best_ep, linestyle="--", linewidth=1.1, label=f"Best epoch = {int(best_ep)}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training and validation convergence"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig07_training_validation_curve")

def fig08_validation_scatter():
    df = discover_validation_predictions()
    if df is None:
        print("[SKIP] Fig08: validation prediction CSV not found."); return
    simple_scatter_plot(df, "Fig08_validation_actual_vs_predicted", "Validation: actual vs. predicted power loss", "$R^2$=0.932\nRMSE=0.075\nMAE=0.037")

def fig09_random_scatter():
    df = load_random_predictions()
    if df is None:
        print("[SKIP] Fig09: RANDOM_TEST predictions.csv not found."); return
    simple_scatter_plot(df, "Fig09_random_test_actual_vs_predicted", "RANDOM_TEST: actual vs. predicted power loss", "$R^2$=0.925\nRMSE=0.078\nMAE=0.039")

def fig10_error_distribution():
    df = load_random_predictions()
    if df is None:
        print("[SKIP] Fig10: RANDOM_TEST predictions.csv not found."); return
    t, pred, *_ = prediction_cols(df)
    if not t or not pred:
        print("[SKIP] Fig10: true/pred columns not found."); return
    y, pr = finite_xy(df[t], df[pred]); e = pr-y; ae = np.abs(e)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(e, bins=45, density=True, alpha=0.72, edgecolor="white", linewidth=0.35)
    counts, edges = np.histogram(e, bins=60, density=True); centers = (edges[:-1] + edges[1:]) / 2
    smooth = np.convolve(counts, np.ones(5)/5, mode="same")
    ax.plot(centers, smooth, linewidth=1.8, label="Smoothed density")
    ax.axvline(np.mean(e), linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(e):.3f}")
    ax.axvline(np.median(e), linestyle=":", linewidth=1.4, label=f"Median = {np.median(e):.3f}")
    ax.set_xlabel("Prediction residual (predicted − actual)"); ax.set_ylabel("Density"); ax.set_title("RANDOM_TEST residual distribution"); ax.legend(frameon=False); style_ax(ax)
    ax.text(0.98, 0.96, f"MAE = {np.mean(ae):.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.3)
    savefig(fig, "Fig10_random_test_residual_distribution")

def fig11_residual_vs_true():
    df = load_random_predictions()
    if df is None:
        print("[SKIP] Fig11: RANDOM_TEST predictions.csv not found."); return
    t, pred, *_ = prediction_cols(df)
    if not t or not pred:
        print("[SKIP] Fig11: true/pred columns not found."); return
    y, pr = finite_xy(df[t], df[pred]); r = pr-y
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(y, r, s=9, alpha=0.42, linewidths=0); ax.axhline(0, linestyle="--", linewidth=1.1)
    bins = np.linspace(0, 1, 11); idx = np.digitize(y, bins, right=True); centers=[]; med=[]
    for k in range(1, len(bins)):
        m = idx == k
        if m.any(): centers.append((bins[k-1]+bins[k])/2); med.append(np.median(r[m]))
    if centers: ax.plot(centers, med, marker="o", markersize=3.5, linewidth=1.8, label="Binned median residual"); ax.legend(frameon=False)
    ax.set_xlabel("Actual relative power loss"); ax.set_ylabel("Residual (predicted − actual)"); ax.set_title("Residual structure across power-loss levels"); style_ax(ax)
    savefig(fig, "Fig11_residual_vs_true_loss")

def fig12_mc_flow():
    fig, ax = plt.subplots(figsize=(8.3, 2.8)); ax.axis("off")
    xs = [0.03, 0.23, 0.46, 0.70]; labs = ["Input sample", "MC Dropout\nrepeated forward passes", "Predictive mean\nand standard deviation", "Raw interval /\nuncertainty signal"]
    for x, lab in zip(xs, labs): draw_box(ax, x, 0.30, 0.16, 0.38, lab)
    for i in range(len(xs)-1): draw_arrow(ax, xs[i]+0.16, 0.49, xs[i+1], 0.49)
    ax.set_title("MC Dropout uncertainty-estimation workflow", fontsize=10.1, pad=10)
    savefig(fig, "Fig12_MC_dropout_workflow")

def fig13_mc_example(): print("[SKIP] Fig13: sample-level MC repeated-pass outputs not found in frozen CSVs.")

def fig14_point_vs_mcmean():
    role_data = load_available_mc_roles()
    if not role_data:
        print("[SKIP] Fig14: sample-level paired point and MC-mean outputs not found.")
        return
    fig, axes = plt.subplots(1, len(role_data), figsize=(5.0 * len(role_data), 3.8), squeeze=False)
    for ax, (role, df) in zip(axes[0], role_data):
        point, mc_mean = finite_xy(df["point_pred"], df["mc_mean"])
        low = min(float(point.min()), float(mc_mean.min()))
        high = max(float(point.max()), float(mc_mean.max()))
        ax.scatter(point, mc_mean, s=9, alpha=0.35, linewidths=0)
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.1, color=DARK, label="$y=x$")
        mean_abs_diff = float(np.mean(np.abs(point - mc_mean)))
        ax.text(
            0.04, 0.94, f"N={len(point):,}\nMean |difference|={mean_abs_diff:.6f}",
            transform=ax.transAxes, va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#BBBBBB"),
        )
        ax.set_xlabel("Point prediction")
        ax.set_ylabel("MC predictive mean")
        ax.set_title(role)
        style_ax(ax)
    fig.suptitle("Sample-level point prediction vs. MC predictive mean", fontsize=10.1)
    savefig(fig, "Fig14_point_prediction_vs_MC_mean")

def fig15_mc_std_distribution():
    role_data = load_available_mc_roles()
    if not role_data:
        print("[SKIP] Fig15: sample-level MC predictive std values not found.")
        return
    all_std = np.concatenate([df["mc_std"].to_numpy(float) for _, df in role_data])
    bins = np.linspace(float(all_std.min()), float(all_std.max()), 31)
    if np.unique(bins).size < 2:
        print("[SKIP] Fig15: sample-level MC predictive std values are not variable.")
        return
    fig, axes = plt.subplots(1, len(role_data), figsize=(5.0 * len(role_data), 3.7), squeeze=False, sharex=True)
    for ax, (role, df), color in zip(axes[0], role_data, PALETTE):
        values = df["mc_std"].to_numpy(float)
        mean_std = float(values.mean())
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.35)
        ax.axvline(mean_std, linestyle="--", linewidth=1.2, color=DARK, label=f"Mean = {mean_std:.6f}")
        ax.set_xlabel("MC predictive std")
        ax.set_ylabel("Sample count")
        ax.set_title(role)
        ax.legend(frameon=False)
        style_ax(ax)
    fig.suptitle("Sample-level MC predictive std distributions", fontsize=10.1)
    savefig(fig, "Fig15_MC_predictive_std_summary")

def fig16_picp():
    d = METHOD_COMPARISON; fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars = ax.bar(np.arange(len(d)), d.PICP, color=PALETTE[:len(d)], edgecolor="white")
    ax.axhline(TARGET_COVERAGE, linestyle="--", linewidth=1.2, label="Target coverage = 0.90")
    ax.set_xticks(np.arange(len(d)), d.Method, rotation=25, ha="right"); ax.set_ylabel("PICP"); ax.set_ylim(0.5, 0.94); ax.set_title("Coverage comparison of uncertainty methods"); ax.legend(frameon=False, loc="lower right"); style_ax(ax)
    for b, v in zip(bars, d.PICP): ax.text(b.get_x()+b.get_width()/2, v+0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2)
    savefig(fig, "Fig16_method_comparison_PICP")

def fig17_mpiw():
    d = METHOD_COMPARISON; fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars = ax.bar(np.arange(len(d)), d.MPIW, color=PALETTE[:len(d)], edgecolor="white")
    ax.set_xticks(np.arange(len(d)), d.Method, rotation=25, ha="right"); ax.set_ylabel("MPIW"); ax.set_title("Sharpness comparison of uncertainty methods"); style_ax(ax)
    for b, v in zip(bars, d.MPIW): ax.text(b.get_x()+b.get_width()/2, v+0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2)
    savefig(fig, "Fig17_method_comparison_MPIW")

def fig18_interval_score():
    d = METHOD_COMPARISON; fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars = ax.bar(np.arange(len(d)), d.IntervalScore, color=PALETTE[:len(d)], edgecolor="white")
    ax.set_xticks(np.arange(len(d)), d.Method, rotation=25, ha="right"); ax.set_ylabel("Interval score (α = 0.10)"); ax.set_title("Overall interval-quality comparison"); style_ax(ax)
    for b, v in zip(bars, d.IntervalScore): ax.text(b.get_x()+b.get_width()/2, v+0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2)
    savefig(fig, "Fig18_method_comparison_interval_score")

def fig19_picp_mpiw():
    d = METHOD_COMPARISON; fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for _, row in d.iterrows(): ax.scatter(row.MPIW, row.PICP, s=58 if row.Method=="CQR" else 36, zorder=3); ax.annotate(row.Method, (row.MPIW, row.PICP), xytext=(5,4), textcoords="offset points", fontsize=7.3)
    ax.axhline(TARGET_COVERAGE, linestyle="--", linewidth=1.1); ax.set_xlabel("MPIW (lower is sharper)"); ax.set_ylabel("PICP"); ax.set_title("Coverage–sharpness trade-off"); style_ax(ax)
    savefig(fig, "Fig19_PICP_vs_MPIW_tradeoff")

def fig20_picp_intervalscore():
    d = METHOD_COMPARISON; fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for _, row in d.iterrows(): ax.scatter(row.IntervalScore, row.PICP, s=58 if row.Method=="CQR" else 36, zorder=3); ax.annotate(row.Method, (row.IntervalScore, row.PICP), xytext=(5,4), textcoords="offset points", fontsize=7.3)
    ax.axhline(TARGET_COVERAGE, linestyle="--", linewidth=1.1); ax.set_xlabel("Interval score (lower is better)"); ax.set_ylabel("PICP"); ax.set_title("Coverage–interval-score relationship"); style_ax(ax)
    savefig(fig, "Fig20_PICP_vs_interval_score")

def fig21_cqr_flow():
    fig, ax = plt.subplots(figsize=(9.0, 2.9)); ax.axis("off")
    xs = [0.03, 0.25, 0.47, 0.70]; labs = ["Train q0.05 / q0.50 /\nq0.95 on TRAIN", "Compute conformity scores\non CP_CALIBRATION", "Estimate $\\hat q$\n(calibration offset)", "Final CQR interval on\nRANDOM_TEST"]
    for x, lab in zip(xs, labs): draw_box(ax, x, 0.28, 0.18, 0.40, lab)
    for i in range(len(xs)-1): draw_arrow(ax, xs[i]+0.18, 0.48, xs[i+1], 0.48)
    ax.set_title("Conformalized quantile regression (CQR) workflow", fontsize=10.1, pad=10); savefig(fig, "Fig21_CQR_workflow")

def fig22_raw_vs_conformal():
    df = load_random_predictions();
    if df is None: print("[SKIP] Fig22: predictions.csv not found."); return
    arr = get_cqr_arrays(df)
    if arr is None: print("[SKIP] Fig22: CQR columns not found."); return
    true, med, raw_lo, raw_up, lo, up, irr, date = arr
    if raw_lo is None or raw_up is None or lo is None or up is None: print("[SKIP] Fig22: raw/final interval columns unavailable."); return
    order = np.argsort(med); sel = order[np.linspace(0, len(order)-1, min(80, len(order))).astype(int)]; x = np.arange(len(sel))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.fill_between(x, raw_lo[sel], raw_up[sel], alpha=0.24, label="Raw quantile interval")
    ax.fill_between(x, lo[sel], up[sel], alpha=0.22, label="Conformalized interval")
    ax.plot(x, med[sel], linewidth=1.3, label="q50"); ax.scatter(x, true[sel], s=11, label="Actual", zorder=3)
    ax.set_xlabel("Representative samples ordered by q50"); ax.set_ylabel("Relative power loss"); ax.set_title("Raw quantile interval vs. conformalized CQR interval"); ax.legend(frameon=False, ncol=2); style_ax(ax)
    savefig(fig, "Fig22_raw_vs_conformal_interval")

def fig23_cqr_interval_examples():
    df = load_random_predictions();
    if df is None: print("[SKIP] Fig23: predictions.csv not found."); return
    arr = get_cqr_arrays(df)
    if arr is None: print("[SKIP] Fig23: CQR columns not found."); return
    true, med, raw_lo, raw_up, lo, up, irr, date = arr
    if lo is None or up is None: print("[SKIP] Fig23: final interval unavailable."); return
    order = np.argsort(med); nshow = min(120, len(order)); sel = order[np.linspace(0, len(order)-1, nshow).astype(int)]; x = np.arange(nshow); covered = (true[sel] >= lo[sel]) & (true[sel] <= up[sel])
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.fill_between(x, lo[sel], up[sel], alpha=0.22, label="90% CQR interval")
    ax.plot(x, med[sel], linewidth=1.25, label="q50")
    ax.scatter(x[covered], true[sel][covered], s=10, label="Covered actual", zorder=3)
    if (~covered).any(): ax.scatter(x[~covered], true[sel][~covered], s=20, marker="x", label="Uncovered actual", zorder=4)
    ax.set_xlabel("Representative samples ordered by q50"); ax.set_ylabel("Relative power loss"); ax.set_title("CQR intervals on RANDOM_TEST"); ax.legend(frameon=False, ncol=2); style_ax(ax)
    savefig(fig, "Fig23_CQR_interval_examples")

def fig24_width_distribution():
    df = load_random_predictions();
    if df is None: print("[SKIP] Fig24: predictions.csv not found."); return
    arr = get_cqr_arrays(df)
    if arr is None: print("[SKIP] Fig24: CQR columns not found."); return
    width = arr[5] - arr[4]; width = width[np.isfinite(width)]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(width, bins=45, density=True, alpha=0.74, edgecolor="white", linewidth=0.35)
    ax.axvline(np.mean(width), linestyle="--", linewidth=1.2, label=f"Mean = {np.mean(width):.3f}")
    ax.axvline(np.median(width), linestyle=":", linewidth=1.4, label=f"Median = {np.median(width):.3f}")
    ax.set_xlabel("CQR interval width"); ax.set_ylabel("Density"); ax.set_title("Distribution of CQR interval widths"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig24_CQR_interval_width_distribution")

def fig25_width_vs_q50():
    df = load_random_predictions();
    if df is None: print("[SKIP] Fig25: predictions.csv not found."); return
    arr = get_cqr_arrays(df)
    if arr is None: print("[SKIP] Fig25: CQR columns not found."); return
    true, med, raw_lo, raw_up, lo, up, irr, date = arr; width = up-lo; mask = np.isfinite(med) & np.isfinite(width)
    fig, ax = plt.subplots(figsize=(5.4, 4.0)); ax.scatter(med[mask], width[mask], s=9, alpha=0.35, linewidths=0)
    bins = np.linspace(0,1,11); ids = np.digitize(med[mask], bins, right=True); centers=[]; medw=[]
    for k in range(1,len(bins)):
        m = ids == k
        if m.any(): centers.append((bins[k-1]+bins[k])/2); medw.append(np.median(width[mask][m]))
    if centers: ax.plot(centers, medw, marker="o", markersize=3.8, linewidth=1.8, label="Binned median width"); ax.legend(frameon=False)
    ax.set_xlabel("q50 predicted relative power loss"); ax.set_ylabel("CQR interval width"); ax.set_title("Uncertainty width across predicted loss levels"); style_ax(ax)
    savefig(fig, "Fig25_interval_width_vs_q50")

def fig26_dev_vs_final():
    f = load_random_interval_metrics()
    if f is None or f.empty:
        print("[SKIP] Fig26: final interval_metrics.csv not found."); return
    row = f.iloc[0]
    final = {"PICP": float(row[pick_col(f, ["PICP"])]), "MPIW": float(row[pick_col(f, ["MPIW"])]), "MedianWidth": float(row[pick_col(f, ["median_width"])]), "IntervalScore": float(row[pick_col(f, ["mean_interval_score_alpha_0p10", "interval_score"])])}
    metrics = ["PICP", "MPIW", "MedianWidth", "IntervalScore"]; dev_vals = [CQR_DEV[m] for m in metrics]; fin_vals = [final[m] for m in metrics]
    x = np.arange(len(metrics)); w = 0.36; fig, ax = plt.subplots(figsize=(6.5, 3.9))
    ax.bar(x-w/2, dev_vals, width=w, label="Decision development"); ax.bar(x+w/2, fin_vals, width=w, label="RANDOM_TEST")
    ax.set_xticks(x, ["PICP","MPIW","Median width","Interval score"]); ax.set_ylabel("Metric value"); ax.set_title("CQR calibration stability: development vs. final"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig26_CQR_development_vs_final")

def fig27_q50_conditional(): _plot_conditional("conditional_coverage_q50.csv", "Fig27_conditional_coverage_by_q50", "Conditional coverage across q50 bins")
def fig28_irradiance_conditional(): _plot_conditional("conditional_coverage_irradiance.csv", "Fig28_conditional_coverage_by_irradiance", "Conditional coverage across irradiance bins")

def fig29_decision_rule():
    fig, ax = plt.subplots(figsize=(8.0, 2.9)); ax.axis("off")
    draw_box(ax, 0.05, 0.22, 0.25, 0.52, "If lower bound > τ\n→ CLEAN", fc="#ECFDF5")
    draw_box(ax, 0.375, 0.22, 0.25, 0.52, "If upper bound < τ\n→ WAIT", fc="#EFF6FF")
    draw_box(ax, 0.70, 0.22, 0.25, 0.52, "Otherwise\n→ REVIEW", fc="#FEF3C7")
    draw_arrow(ax, 0.30, 0.48, 0.375, 0.48); draw_arrow(ax, 0.625, 0.48, 0.70, 0.48)
    ax.set_title("Tri-state decision rule based on the CQR interval", fontsize=10.1, pad=10); savefig(fig, "Fig29_tristate_decision_rule")

def fig30_ader_main():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Fig30: decision_metrics.csv not found."); return
    taucol = pick_col(d, ["tau"]); mcol = pick_col(d, ["method"]); ader = pick_col(d, ["auto_decision_error_rate"])
    if not taucol or not mcol or not ader: print("[SKIP] Fig30: required columns not found."); return
    sub = d[np.isclose(d[taucol].astype(float), 0.15)].copy(); fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(np.arange(len(sub)), sub[ader].astype(float), color=PALETTE[:len(sub)], edgecolor="white")
    ax.set_xticks(np.arange(len(sub)), sub[mcol].astype(str), rotation=18, ha="right"); ax.set_ylabel("ADER"); ax.set_title("Automatic decision error rate at τ = 0.15"); style_ax(ax)
    savefig(fig, "Fig30_automatic_decision_error_rate_tau015")

def fig31_decision_main_grouped():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Fig31: decision_metrics.csv not found."); return
    taucol = pick_col(d, ["tau"]); mcol = pick_col(d, ["method"])
    cols = {"FCR": pick_col(d,["false_clean_rate_oracle_wait"]), "MCR": pick_col(d,["missed_clean_rate_oracle_clean"]), "RR": pick_col(d,["review_rate"]), "ADC": pick_col(d,["auto_decision_coverage"]), "ADER": pick_col(d,["auto_decision_error_rate"])}
    if not taucol or not mcol or any(v is None for v in cols.values()): print("[SKIP] Fig31: required columns not found."); return
    sub = d[np.isclose(d[taucol].astype(float), 0.15)].copy(); metrics = list(cols.keys()); x = np.arange(len(metrics)); w = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [float(row[cols[k]]) for k in metrics]
        ax.bar(x + (i-1)*w, vals, width=w, label=str(row[mcol]), edgecolor="white")
    ax.set_xticks(x, metrics); ax.set_ylabel("Rate"); ax.set_title("Safety–automation metrics at τ = 0.15"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig31_decision_metrics_tau015")

def fig32_review_rate_vs_tau():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Fig32: decision_metrics.csv not found."); return
    taucol = pick_col(d,["tau"]); mcol = pick_col(d,["method"]); rr = pick_col(d,["review_rate"])
    if not taucol or not mcol or not rr: print("[SKIP] Fig32: required columns not found."); return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for meth, grp in d.groupby(mcol): grp = grp.sort_values(taucol); ax.plot(grp[taucol].astype(float), grp[rr].astype(float), marker="o", label=str(meth))
    ax.set_xlabel("Threshold τ"); ax.set_ylabel("Review rate"); ax.set_title("Review rate under different thresholds"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig32_review_rate_vs_tau")

def fig33_ader_vs_tau():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Fig33: decision_metrics.csv not found."); return
    taucol = pick_col(d,["tau"]); mcol = pick_col(d,["method"]); ader = pick_col(d,["auto_decision_error_rate"])
    if not taucol or not mcol or not ader: print("[SKIP] Fig33: required columns not found."); return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for meth, grp in d.groupby(mcol): grp = grp.sort_values(taucol); ax.plot(grp[taucol].astype(float), grp[ader].astype(float), marker="o", label=str(meth))
    ax.set_xlabel("Threshold τ"); ax.set_ylabel("ADER"); ax.set_title("Automatic decision error rate under different thresholds"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig33_ADER_vs_tau")

def fig34_rr_ader_tradeoff():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Fig34: decision_metrics.csv not found."); return
    taucol = pick_col(d,["tau"]); mcol = pick_col(d,["method"]); rr = pick_col(d,["review_rate"]); ader = pick_col(d,["auto_decision_error_rate"])
    if not taucol or not mcol or not rr or not ader: print("[SKIP] Fig34: required columns not found."); return
    sub = d[d[mcol].astype(str) == "cqr_interval_tristate"].copy()
    if sub.empty: print("[SKIP] Fig34: cqr_interval_tristate rows not found."); return
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    for _, row in sub.iterrows(): ax.scatter(float(row[rr]), float(row[ader]), s=46); ax.annotate(f"τ={float(row[taucol]):.2f}", (float(row[rr]), float(row[ader])), xytext=(5,4), textcoords="offset points", fontsize=7.4)
    ax.set_xlabel("Review rate"); ax.set_ylabel("ADER"); ax.set_title("Automation–safety trade-off of CQR tri-state"); style_ax(ax)
    savefig(fig, "Fig34_review_rate_vs_ADER_tradeoff")

def fig35_decision_flow():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Fig35: decision_metrics.csv not found."); return
    taucol = pick_col(d,["tau"]); mcol = pick_col(d,["method"]); cols = {"pred_clean_n": pick_col(d,["pred_clean_n"]), "pred_wait_n": pick_col(d,["pred_wait_n"]), "pred_review_n": pick_col(d,["pred_review_n"])}
    if not taucol or not mcol or any(v is None for v in cols.values()): print("[SKIP] Fig35: required columns not found."); return
    sub = d[(d[mcol].astype(str)=="cqr_interval_tristate") & (np.isclose(d[taucol].astype(float),0.15))]
    if sub.empty: print("[SKIP] Fig35: target row not found."); return
    row = sub.iloc[0]; vals = [int(row[cols["pred_clean_n"]]), int(row[cols["pred_wait_n"]]), int(row[cols["pred_review_n"]])]; labs = ["CLEAN", "WAIT", "REVIEW"]
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.bar(labs, vals, color=[PALETTE[2], PALETTE[0], PALETTE[1]], edgecolor="white")
    ax.set_ylabel("Number of samples"); ax.set_title("Decision action distribution (CQR tri-state, τ = 0.15)"); style_ax(ax)
    savefig(fig, "Fig35_decision_action_distribution_tau015")

def fig36_mean_regret():
    e = load_economic_metrics()
    if e is None: print("[SKIP] Fig36: economic_metrics.csv not found."); return
    taucol = pick_col(e,["tau"]); mcol = pick_col(e,["method"]); reg = pick_col(e,["mean_regret_r0"])
    if not taucol or not mcol or not reg: print("[SKIP] Fig36: required columns not found."); return
    sub = e[np.isclose(e[taucol].astype(float), 0.15)].copy(); fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(np.arange(len(sub)), sub[reg].astype(float), color=PALETTE[:len(sub)], edgecolor="white")
    ax.set_xticks(np.arange(len(sub)), sub[mcol].astype(str), rotation=18, ha="right"); ax.set_ylabel("Mean regret"); ax.set_title("Mean regret at τ = 0.15"); style_ax(ax)
    savefig(fig, "Fig36_mean_regret_tau015")

def fig37_total_cost():
    e = load_economic_metrics()
    if e is None: print("[SKIP] Fig37: economic_metrics.csv not found."); return
    taucol = pick_col(e,["tau"]); mcol = pick_col(e,["method"]); total = pick_col(e,["mean_total_cost_r0"])
    if not taucol or not mcol or not total: print("[SKIP] Fig37: required columns not found."); return
    sub = e[np.isclose(e[taucol].astype(float), 0.15)].copy(); fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(np.arange(len(sub)), sub[total].astype(float), color=PALETTE[:len(sub)], edgecolor="white")
    ax.set_xticks(np.arange(len(sub)), sub[mcol].astype(str), rotation=18, ha="right"); ax.set_ylabel("Mean total cost"); ax.set_title("Mean total cost at τ = 0.15"); style_ax(ax)
    savefig(fig, "Fig37_mean_total_cost_tau015")

def fig38_regret_components():
    e = load_economic_metrics()
    if e is None: print("[SKIP] Fig38: economic_metrics.csv not found."); return
    taucol = pick_col(e,["tau"]); mcol = pick_col(e,["method"]); fcr = pick_col(e,["false_clean_regret_mean_per_N"]); mcr = pick_col(e,["missed_clean_regret_mean_per_N"])
    if not taucol or not mcol or not fcr or not mcr: print("[SKIP] Fig38: required columns not found."); return
    sub = e[np.isclose(e[taucol].astype(float), 0.15)].copy(); x = np.arange(len(sub)); a = sub[fcr].astype(float).to_numpy(); b = sub[mcr].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.bar(x, a, label="False-clean regret", edgecolor="white"); ax.bar(x, b, bottom=a, label="Missed-clean regret", edgecolor="white")
    ax.set_xticks(x, sub[mcol].astype(str), rotation=18, ha="right"); ax.set_ylabel("Mean regret per sample"); ax.set_title("Regret decomposition at τ = 0.15"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig38_regret_decomposition_tau015")

def fig39_break_even():
    b = load_break_even()
    if b is None: print("[SKIP] Fig39: break_even_review_cost.csv not found."); return
    taucol = pick_col(b,["tau"]); p = pick_col(b,["break_even_vs_point"]); q = pick_col(b,["break_even_vs_q50"])
    if not taucol or not p or not q: print("[SKIP] Fig39: required columns not found."); return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(b[taucol].astype(float), b[p].astype(float), marker="o", label="vs point threshold")
    ax.plot(b[taucol].astype(float), b[q].astype(float), marker="s", label="vs q50 threshold")
    ax.set_xlabel("Threshold τ"); ax.set_ylabel("Break-even review cost"); ax.set_title("Break-even review cost of CQR tri-state"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig39_break_even_review_cost")

def fig40_random_vs_sealed():
    rt_p = load_random_prediction_metrics(); sd_p = read_csv_if_exists(FINAL_SD / "prediction_metrics.csv")
    rt_i = load_random_interval_metrics(); sd_i = read_csv_if_exists(FINAL_SD / "interval_metrics.csv")
    if rt_p is None or sd_p is None or rt_i is None or sd_i is None:
        print("[SKIP] Fig40: random/sealed metrics not fully available."); return
    r2_rt = float(rt_p.loc[rt_p["model"].astype(str)=="point_pred", "R2"].iloc[0]); r2_sd = float(sd_p.loc[sd_p["model"].astype(str)=="point_pred", "R2"].iloc[0])
    picp_rt = float(rt_i["PICP"].iloc[0]); picp_sd = float(sd_i["PICP"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.8, 3.8)); x = np.arange(2)
    ax.bar(x-0.18, [r2_rt, picp_rt], width=0.36, label="RANDOM_TEST"); ax.bar(x+0.18, [r2_sd, picp_sd], width=0.36, label="SEALED_DATES")
    ax.set_xticks(x, ["Point $R^2$", "CQR PICP"]); ax.set_title("Same-domain vs. unseen-date stress test"); ax.legend(frameon=False); style_ax(ax)
    savefig(fig, "Fig40_random_test_vs_sealed_dates")

def fig41_date_grouped_folds():
    d = load_date_grouped_fold_metrics()
    if d is None:
        print("[SKIP] Fig41: authoritative date-grouped fold metrics source not found.")
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.8)); bars = ax.bar(d["Fold"].astype(str), d["R2"], color=PALETTE[:len(d)], edgecolor="white")
    ax.set_ylabel("$R^2$"); ax.set_title("Historical date-grouped 4-fold variation"); style_ax(ax)
    for bar, v in zip(bars, d["R2"]): ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.4)
    savefig(fig, "Fig41_date_grouped_fold_R2_variation")

def fig42_roadmap():
    fig, ax = plt.subplots(figsize=(9.0, 2.8)); ax.axis("off")
    xs = [0.03, 0.26, 0.50, 0.75]; labs = ["Paper1\nstrict prediction protocol", "Trustworthy prediction\n(CQR)", "Risk-aware cleaning\ndecision", "Paper2\nlong-horizon RL scheduling"]
    for x, lab in zip(xs, labs): draw_box(ax, x, 0.29, 0.18, 0.40, lab, fc="#F8FAFC")
    for i in range(len(xs)-1): draw_arrow(ax, xs[i]+0.18, 0.49, xs[i+1], 0.49)
    ax.set_title("Research roadmap from trustworthy prediction to scheduling", fontsize=10.1, pad=10); savefig(fig, "Fig42_research_roadmap")

# --- Tables 1-16 ---
def table01(): save_table(pd.DataFrame(DATA_SPLIT, columns=["Role","N","Purpose"]), "Table01_data_partition", "Table 1. Data partition and purpose.")

def table02():
    rows = [["MODEL_VALIDATION", "point_pred", VAL_POINT["R2"], VAL_POINT["RMSE"], VAL_POINT["MAE"], VAL_POINT["Bias"]], ["MODEL_VALIDATION", "q50", VAL_Q50["R2"], VAL_Q50["RMSE"], VAL_Q50["MAE"], VAL_Q50["Bias"]]]
    f = load_random_prediction_metrics()
    if f is not None:
        for _, r in f.iterrows(): rows.append(["RANDOM_TEST", r["model"], r["R2"], r["RMSE"], r["MAE"], r["bias"]])
    df = pd.DataFrame(rows, columns=["Dataset","Model","R²","RMSE","MAE","Bias"])
    save_table(df, "Table02_prediction_performance", "Table 2. Point and q50 prediction performance.", {"R²":"{:.4f}", "RMSE":"{:.4f}", "MAE":"{:.4f}", "Bias":"{:.4f}"})

def table03():
    df = load_date_grouped_fold_metrics()
    if df is None:
        print("[SKIP] Table03: authoritative date-grouped fold metrics source not found.")
        return
    formats = {c: "{:.4f}" for c in ["R2", "RMSE", "MAE", "Bias"] if c in df.columns}
    save_table(df, "Table03_date_grouped_folds", "Table 3. Formal date-grouped 4-fold performance.", formats)

def table04():
    role_data = load_available_mc_roles()
    if not role_data:
        print("[SKIP] Table04: authoritative role-labelled MC Dropout outputs not found.")
        return
    raw_picp = load_raw_mc_picp_by_role()
    rows = []
    for role, samples in role_data:
        rows.append({
            "DatasetRole": role,
            "N": len(samples),
            "Point_MAE": metric_mae(samples["true_L"], samples["point_pred"]),
            "MCMean_MAE": metric_mae(samples["true_L"], samples["mc_mean"]),
            "MeanAbsDiff_PointVsMCMean": float(np.mean(np.abs(samples["point_pred"] - samples["mc_mean"]))),
            "MeanPredictiveStd": float(samples["mc_std"].mean()),
        })
    df = pd.DataFrame(rows)
    if raw_picp:
        df["RawMCPICP"] = df["DatasetRole"].map(raw_picp)
    formats = {c: "{:.6f}" for c in df.columns if c not in ("DatasetRole", "N")}
    save_table(df, "Table04_MC_summary", "Table 4. Role-aware summary of MC Dropout statistics.", formats)

def table05(): save_table(METHOD_COMPARISON.copy(), "Table05_uncertainty_methods", "Table 5. Comparison of uncertainty interval methods.", {"PICP":"{:.4f}", "MPIW":"{:.4f}", "IntervalScore":"{:.4f}"})

def table06():
    f = load_random_interval_metrics(); rows = [["DECISION_DEVELOPMENT", CQR_DEV["PICP"], CQR_DEV["MPIW"], CQR_DEV["MedianWidth"], CQR_DEV["CoverageError"], CQR_DEV["IntervalScore"]]]
    if f is not None and not f.empty:
        r = f.iloc[0]
        rows.append(["RANDOM_TEST", float(r[pick_col(f,["PICP"])]), float(r[pick_col(f,["MPIW"])]), float(r[pick_col(f,["median_width"])]), float(r[pick_col(f,["coverage_error"])]), float(r[pick_col(f,["mean_interval_score_alpha_0p10", "interval_score"])])])
    df = pd.DataFrame(rows, columns=["Dataset","PICP","MPIW","MedianWidth","CoverageError","IntervalScore"])
    save_table(df, "Table06_CQR_dev_vs_final", "Table 6. CQR calibration on development and final test.", {c:"{:.4f}" for c in ["PICP","MPIW","MedianWidth","CoverageError","IntervalScore"]})

def _cond_table(file_name, stem, caption):
    f = read_csv_if_exists(FINAL_RT / file_name)
    if f is None: print(f"[SKIP] {stem}: {file_name} not found."); return
    cols = [c for c in ["binning_variable", "bin_label", "N", "PICP", "MPIW"] if c in f.columns]
    df = f[cols].copy(); save_table(df, stem, caption, {"PICP":"{:.4f}", "MPIW":"{:.4f}"})

def table07(): _cond_table("conditional_coverage_q50.csv", "Table07_q50_conditional_coverage", "Table 7. Conditional coverage across q50 bins.")
def table08(): _cond_table("conditional_coverage_irradiance.csv", "Table08_irradiance_conditional_coverage", "Table 8. Conditional coverage across irradiance bins.")

def table09():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Table09: decision_metrics.csv not found."); return
    tau = pick_col(d,["tau"]); sub = d[np.isclose(d[tau].astype(float), 0.15)].copy() if tau else d.copy()
    cols_map = {"Method": pick_col(sub,["method"]), "FCR": pick_col(sub,["false_clean_rate_oracle_wait"]), "MCR": pick_col(sub,["missed_clean_rate_oracle_clean"]), "RR": pick_col(sub,["review_rate"]), "ADC": pick_col(sub,["auto_decision_coverage"]), "ADER": pick_col(sub,["auto_decision_error_rate"])}
    if any(v is None for v in cols_map.values()): print("[SKIP] Table09: required columns not found."); return
    df = pd.DataFrame({k: sub[v].values for k, v in cols_map.items()})
    save_table(df, "Table09_decision_main_tau015", "Table 9. Main decision results at τ = 0.15.", {"FCR":"{:.4f}", "MCR":"{:.4f}", "RR":"{:.4f}", "ADC":"{:.4f}", "ADER":"{:.4f}"})

def table10():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Table10: decision_metrics.csv not found."); return
    meth = pick_col(d,["method"]); tau = pick_col(d,["tau"])
    if not meth or not tau: print("[SKIP] Table10: required columns not found."); return
    sub = d[d[meth].astype(str)=="cqr_interval_tristate"].copy()
    cols_map = {"tau": tau, "RR": pick_col(sub,["review_rate"]), "ADC": pick_col(sub,["auto_decision_coverage"]), "ADER": pick_col(sub,["auto_decision_error_rate"])}
    if any(v is None for v in cols_map.values()): print("[SKIP] Table10: required columns not found."); return
    df = pd.DataFrame({k: sub[v].values for k, v in cols_map.items()})
    save_table(df, "Table10_CQR_decision_sensitivity", "Table 10. Decision sensitivity of the CQR tri-state rule.", {"tau":"{:.2f}", "RR":"{:.4f}", "ADC":"{:.4f}", "ADER":"{:.4f}"})

def table11():
    d = load_decision_metrics()
    if d is None: print("[SKIP] Table11: decision_metrics.csv not found."); return
    cols_keep = [c for c in ["method", "tau", "review_rate", "auto_decision_coverage", "auto_decision_error_rate"] if c in d.columns]
    if len(cols_keep) < 5: print("[SKIP] Table11: required columns not found."); return
    df = d[cols_keep].copy()
    save_table(df, "Table11_decision_all_methods_all_tau", "Table 11. Decision comparison across all methods and thresholds.", {"tau":"{:.2f}", "review_rate":"{:.4f}", "auto_decision_coverage":"{:.4f}", "auto_decision_error_rate":"{:.4f}"})

def table12():
    e = load_economic_metrics()
    if e is None: print("[SKIP] Table12: economic_metrics.csv not found."); return
    tau = pick_col(e,["tau"]); sub = e[np.isclose(e[tau].astype(float),0.15)].copy() if tau else e.copy()
    cols_map = {"Method": pick_col(sub,["method"]), "MeanRegret": pick_col(sub,["mean_regret_r0"]), "MeanTotalCost": pick_col(sub,["mean_total_cost_r0"]), "ReviewRate": pick_col(sub,["review_rate"])}
    if any(v is None for v in cols_map.values()): print("[SKIP] Table12: required columns not found."); return
    df = pd.DataFrame({k: sub[v].values for k, v in cols_map.items()})
    save_table(df, "Table12_economic_main_tau015", "Table 12. Main economic results at τ = 0.15.", {"MeanRegret":"{:.4f}", "MeanTotalCost":"{:.4f}", "ReviewRate":"{:.4f}"})

def table13():
    e = load_economic_metrics()
    if e is None: print("[SKIP] Table13: economic_metrics.csv not found."); return
    meth = pick_col(e,["method"]); tau = pick_col(e,["tau"]); sub = e[e[meth].astype(str)=="cqr_interval_tristate"].copy() if meth else e.copy()
    cols_map = {"tau": tau, "MeanRegret": pick_col(sub,["mean_regret_r0"]), "MeanTotalCost": pick_col(sub,["mean_total_cost_r0"]), "ReviewRate": pick_col(sub,["review_rate"])}
    if any(v is None for v in cols_map.values()): print("[SKIP] Table13: required columns not found."); return
    df = pd.DataFrame({k: sub[v].values for k, v in cols_map.items()})
    save_table(df, "Table13_CQR_economic_sensitivity", "Table 13. Economic sensitivity of the CQR tri-state rule.", {"tau":"{:.2f}", "MeanRegret":"{:.4f}", "MeanTotalCost":"{:.4f}", "ReviewRate":"{:.4f}"})

def table14():
    b = load_break_even()
    if b is None: print("[SKIP] Table14: break_even_review_cost.csv not found."); return
    cols_map = {"tau": pick_col(b,["tau"]), "BreakEven_vs_Point": pick_col(b,["break_even_vs_point"]), "BreakEven_vs_q50": pick_col(b,["break_even_vs_q50"])}
    if any(v is None for v in cols_map.values()): print("[SKIP] Table14: required columns not found."); return
    df = pd.DataFrame({k: b[v].values for k, v in cols_map.items()})
    save_table(df, "Table14_break_even_review_cost", "Table 14. Break-even review cost of the CQR tri-state rule.", {"tau":"{:.2f}", "BreakEven_vs_Point":"{:.4f}", "BreakEven_vs_q50":"{:.4f}"})

def table15():
    rt_pred = load_random_prediction_metrics(); rt_int = load_random_interval_metrics(); d = load_decision_metrics(); e = load_economic_metrics(); b = load_break_even(); key_rows = []
    if rt_pred is not None:
        r = rt_pred.loc[rt_pred["model"].astype(str)=="point_pred"].iloc[0]
        key_rows.append(["Point prediction", f"R²={float(r['R2']):.4f}, RMSE={float(r['RMSE']):.4f}, MAE={float(r['MAE']):.4f}"])
    key_rows.append(["Uncertainty method selection", "CQR achieved the best overall interval score under near-target coverage"])
    if rt_int is not None:
        r = rt_int.iloc[0]; key_rows.append(["Final CQR interval", f"PICP={float(r['PICP']):.4f}, MPIW={float(r['MPIW']):.4f}"])
    if d is not None:
        tau = pick_col(d,["tau"]); meth = pick_col(d,["method"]); ader = pick_col(d,["auto_decision_error_rate"]); sub = d[np.isclose(d[tau].astype(float),0.15)]
        point = sub[sub[meth].astype(str)=="point_threshold"]; cqr = sub[sub[meth].astype(str)=="cqr_interval_tristate"]
        if (not point.empty) and (not cqr.empty): key_rows.append(["Decision gain at τ=0.15", f"ADER {float(point[ader].iloc[0]):.4f} → {float(cqr[ader].iloc[0]):.4f}"])
    if e is not None:
        tau = pick_col(e,["tau"]); meth = pick_col(e,["method"]); reg = pick_col(e,["mean_regret_r0"]); sub = e[np.isclose(e[tau].astype(float),0.15)]
        point = sub[sub[meth].astype(str)=="point_threshold"]; cqr = sub[sub[meth].astype(str)=="cqr_interval_tristate"]
        if (not point.empty) and (not cqr.empty): key_rows.append(["Economic gain at τ=0.15", f"Mean regret {float(point[reg].iloc[0]):.4f} → {float(cqr[reg].iloc[0]):.4f}"])
    if b is not None:
        tau = pick_col(b,["tau"]); bep = pick_col(b,["break_even_vs_point"]); row = b[np.isclose(b[tau].astype(float),0.15)]
        if not row.empty: key_rows.append(["Break-even review cost", f"vs point at τ=0.15: {float(row[bep].iloc[0]):.4f}"])
    df = pd.DataFrame(key_rows, columns=["Milestone", "KeyResult"])
    save_table(df, "Table15_project_milestones", "Table 15. Key milestones and headline results.")

def table16():
    df = pd.DataFrame([
        ["Conditional coverage mismatch", "Tables 7–8", "Some bins remain under-covered", "Conditional/adaptive calibration"],
        ["Cross-date robustness", "Historical date-grouped variation and sealed stress test", "Generalization remains limited", "Distribution shift handling"],
        ["Current claim scope", "RANDOM_TEST succeeds, unseen dates are harder", "Main conclusion should stay same-domain", "Paper2 / robust scheduling"],
    ], columns=["Issue", "Evidence", "CurrentImpact", "NextStep"])
    save_table(df, "Table16_limitations_and_next_steps", "Table 16. Current limitations and next steps.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Paper1 figures and tables.")
    parser.add_argument(
        "--only",
        type=str.lower,
        metavar="FIGURE",
        help="Generate one supported figure only (currently: fig03).",
    )
    args = parser.parse_args()
    if args.only not in (None, "fig03"):
        parser.error(f"unknown figure '{args.only}'; supported value: fig03")

    print("Paper1 full figure/table generation (v2)")
    print(f"Repository root: {ROOT}")
    print(f"Output directory: {OUT_ROOT}")
    print("This is report-only. No training, no recalibration, no formal-result overwrite.\n")
    if args.only == "fig03":
        fig03_data_split()
    else:
        for fn in [fig01_framework, fig02_problem_to_redesign, fig03_data_split, fig04_date_distribution, fig05_date_role_heatmap, fig06_model_structure, fig07_training_curve, fig08_validation_scatter, fig09_random_scatter, fig10_error_distribution, fig11_residual_vs_true, fig12_mc_flow, fig13_mc_example, fig14_point_vs_mcmean, fig15_mc_std_distribution, fig16_picp, fig17_mpiw, fig18_interval_score, fig19_picp_mpiw, fig20_picp_intervalscore, fig21_cqr_flow, fig22_raw_vs_conformal, fig23_cqr_interval_examples, fig24_width_distribution, fig25_width_vs_q50, fig26_dev_vs_final, fig27_q50_conditional, fig28_irradiance_conditional, fig29_decision_rule, fig30_ader_main, fig31_decision_main_grouped, fig32_review_rate_vs_tau, fig33_ader_vs_tau, fig34_rr_ader_tradeoff, fig35_decision_flow, fig36_mean_regret, fig37_total_cost, fig38_regret_components, fig39_break_even, fig40_random_vs_sealed, fig41_date_grouped_folds, fig42_roadmap]:
            fn()
        for fn in [table01, table02, table03, table04, table05, table06, table07, table08, table09, table10, table11, table12, table13, table14, table15, table16]:
            fn()
    print("\nDone.")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables : {TAB_DIR}")
    print("If any item was [SKIP], it means the required frozen source file was absent or had incompatible columns.")
    print("Do NOT commit generated PNG/PDF/SVG yet. First inspect visual quality, then decide what to keep for group meeting / paper.")
