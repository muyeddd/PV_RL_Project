import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 1. 路径设置
# ============================================================
OUT_DIR = r"E:\PV_RL_Project\paper_figures\fig5_risk_coverage"
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATE_DIRS = [
    r"E:\PV_RL_Project\cp_analysis\final_unified_results",
    r"E:\PV_RL_Project\cp_analysis",
]


def find_file(filename):
    for d in CANDIDATE_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            print(f"找到文件: {path}")
            return path
    return None


CURVE_CSV = find_file("risk_coverage_curves.csv")
ORACLE_CSV = find_file("oracle_risk_coverage_curve.csv")
SUMMARY_CSV = find_file("risk_coverage_summary.csv")

if SUMMARY_CSV is None:
    SUMMARY_CSV = find_file("final_risk_summary.csv")

if CURVE_CSV is None:
    raise FileNotFoundError("找不到 risk_coverage_curves.csv")

if ORACLE_CSV is None:
    raise FileNotFoundError("找不到 oracle_risk_coverage_curve.csv")


# ============================================================
# 2. 读取数据
# ============================================================
curves = pd.read_csv(CURVE_CSV)
oracle = pd.read_csv(ORACLE_CSV)

if SUMMARY_CSV is not None:
    summary = pd.read_csv(SUMMARY_CSV)
else:
    print("未找到 risk_coverage_summary.csv，使用已确认的最终指标重建 summary。")
    summary = pd.DataFrame([
        {
            "method": "Raw MC",
            "AUSE_MAE": 0.0120882071,
            "AUSE_RMSE": 0.0298426859,
            "accepted_mae": 0.0277856932,
            "rejected_mae": 0.0586652709,
            "accepted_rmse": 0.0531172660,
            "rejected_rmse": 0.0958788627,
            "mae_ratio_rejected_over_accepted": 2.1113481113,
            "rmse_ratio_rejected_over_accepted": 1.8050413716,
        },
        {
            "method": "Split CP",
            "AUSE_MAE": 0.0149165304,
            "AUSE_RMSE": 0.0351626638,
            "accepted_mae": 0.0338650599,
            "rejected_mae": 0.0343008011,
            "accepted_rmse": 0.0640874287,
            "rejected_rmse": 0.0636080928,
            "mae_ratio_rejected_over_accepted": 1.0137315628,
            "rmse_ratio_rejected_over_accepted": 0.9925205935,
        },
        {
            "method": "Pred-L-Mondrian CP",
            "AUSE_MAE": 0.0113714413,
            "AUSE_RMSE": 0.0267702158,
            "accepted_mae": 0.0310724755,
            "rejected_mae": 0.0455085594,
            "accepted_rmse": 0.0594556876,
            "rejected_rmse": 0.0796023659,
            "mae_ratio_rejected_over_accepted": 1.4645939405,
            "rmse_ratio_rejected_over_accepted": 1.3388519947,
        },
        {
            "method": "Pred-L-Mondrian MC-Interval CP",
            "AUSE_MAE": 0.0104010202,
            "AUSE_RMSE": 0.0248828463,
            "accepted_mae": 0.0292149714,
            "rejected_mae": 0.0529439912,
            "accepted_rmse": 0.0558425661,
            "rejected_rmse": 0.0894656226,
            "mae_ratio_rejected_over_accepted": 1.8122122239,
            "rmse_ratio_rejected_over_accepted": 1.6021044319,
        },
        {
            "method": "Pred-L-Mondrian Std-MC CP",
            "AUSE_MAE": 0.0102947868,
            "AUSE_RMSE": 0.0258791108,
            "accepted_mae": 0.0285537828,
            "rejected_mae": 0.0555906735,
            "accepted_rmse": 0.0548324711,
            "rejected_rmse": 0.0919325447,
            "mae_ratio_rejected_over_accepted": 1.9468759680,
            "rmse_ratio_rejected_over_accepted": 1.6766077268,
        },
    ])

# 保存本图数据
curves.to_csv(
    os.path.join(OUT_DIR, "fig5_risk_coverage_curves_data.csv"),
    index=False,
    encoding="utf-8-sig"
)
summary.to_csv(
    os.path.join(OUT_DIR, "fig5_risk_coverage_summary_data.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 3. 方法顺序与名称
# ============================================================
method_order = [
    "Raw MC",
    "Split CP",
    "Pred-L-Mondrian CP",
    "Pred-L-Mondrian MC-Interval CP",
    "Pred-L-Mondrian Std-MC CP",
]

short_names = {
    "Raw MC": "Raw\nMC",
    "Split CP": "Split\nCP",
    "Pred-L-Mondrian CP": "P-M\nCP",
    "Pred-L-Mondrian MC-Interval CP": "P-MCInt\nCP",
    "Pred-L-Mondrian Std-MC CP": "P-StdMC\nCP",
}

legend_names = {
    "Raw MC": "Raw MC",
    "Split CP": "Split CP",
    "Pred-L-Mondrian CP": "P-M CP",
    "Pred-L-Mondrian MC-Interval CP": "P-MCInt CP",
    "Pred-L-Mondrian Std-MC CP": "P-StdMC CP",
}

colors = {
    "Raw MC": "#9CA3AF",
    "Split CP": "#4C78A8",
    "Pred-L-Mondrian CP": "#72B7B2",
    "Pred-L-Mondrian MC-Interval CP": "#3A9D9A",
    "Pred-L-Mondrian Std-MC CP": "#E45756",
    "Oracle": "#111111",
}

line_styles = {
    "Raw MC": "-.",
    "Split CP": "-",
    "Pred-L-Mondrian CP": "-",
    "Pred-L-Mondrian MC-Interval CP": "-",
    "Pred-L-Mondrian Std-MC CP": "-",
}


# ============================================================
# 4. 论文风格参数
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 12.5,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9.6,
    "axes.linewidth": 1.15,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 5. 工具函数
# ============================================================
def add_panel_label(ax, label):
    ax.text(
        0.0,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False
    )


def style_axis(ax):
    ax.tick_params(direction="in", length=5, width=1.0)
    ax.grid(True, axis="both", linewidth=0.45, alpha=0.25)
    ax.set_axisbelow(True)


# ============================================================
# 6. 整理 summary 顺序
# ============================================================
summary = summary.copy()
summary["method"] = pd.Categorical(
    summary["method"],
    categories=method_order,
    ordered=True
)
summary = summary.sort_values("method").reset_index(drop=True)


# ============================================================
# 7. 作图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

plt.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.80,
    bottom=0.23,
    wspace=0.26
)

ax_a, ax_b = axes


# ============================================================
# (a) Risk–Coverage curve
# ============================================================
for method in method_order:
    sub = curves[curves["method"] == method].copy()
    if sub.empty:
        continue

    is_main = method == "Pred-L-Mondrian Std-MC CP"

    ax_a.plot(
        sub["coverage"],
        sub["retained_mae"],
        linestyle=line_styles[method],
        marker="o",
        markersize=3.5 if not is_main else 4.0,
        linewidth=1.35 if not is_main else 2.25,
        color=colors[method],
        label=legend_names[method],
        alpha=0.72 if not is_main else 0.98,
        zorder=5 if is_main else 3
    )

# Oracle 曲线
if "oracle_mae" in oracle.columns:
    ax_a.plot(
        oracle["coverage"],
        oracle["oracle_mae"],
        linestyle="--",
        linewidth=1.7,
        color=colors["Oracle"],
        label="Oracle",
        zorder=2
    )

ax_a.set_xlabel("Retained coverage")
ax_a.set_ylabel("Retained MAE")
ax_a.set_xlim(0.08, 1.02)
ax_a.set_ylim(0.00, 0.038)

style_axis(ax_a)
add_panel_label(ax_a, "(a)")

ax_a.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.52, 1.27),
    ncol=3,
    columnspacing=1.1,
    handlelength=2.5
)


# ============================================================
# (b) Accepted vs rejected MAE
# ============================================================
x = np.arange(len(summary))
bar_width = 0.36

accepted = summary["accepted_mae"].to_numpy()
rejected = summary["rejected_mae"].to_numpy()

ax_b.bar(
    x - bar_width / 2,
    accepted,
    width=bar_width,
    color="#A9CBE8",
    edgecolor="black",
    linewidth=0.75,
    alpha=0.95,
    label="Accepted 80%"
)

ax_b.bar(
    x + bar_width / 2,
    rejected,
    width=bar_width,
    color="#F28E2B",
    edgecolor="black",
    linewidth=0.75,
    alpha=0.95,
    label="Rejected 20%"
)

# 数值标注
for i, v in enumerate(accepted):
    ax_b.text(
        i - bar_width / 2,
        v + 0.0012,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5
    )

for i, v in enumerate(rejected):
    ax_b.text(
        i + bar_width / 2,
        v + 0.0012,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5
    )

# 主方法倍数标注：不用箭头，直接放在最后一组上方
main_row = summary[summary["method"] == "Pred-L-Mondrian Std-MC CP"]
if not main_row.empty:
    idx = int(main_row.index[0])
    ratio = float(main_row["mae_ratio_rejected_over_accepted"].values[0])
    y_top = max(
        float(main_row["accepted_mae"].values[0]),
        float(main_row["rejected_mae"].values[0])
    )

    ax_b.text(
        idx,
        y_top + 0.0075,
        f"{ratio:.2f}×",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#111111"
    )

ax_b.set_ylabel("MAE")
ax_b.set_xticks(x)
ax_b.set_xticklabels([short_names[m] for m in summary["method"]])
ax_b.set_ylim(0.00, 0.070)

style_axis(ax_b)
add_panel_label(ax_b, "(b)")

# 图例放到子图外上方，不压柱子
ax_b.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.50, 1.20),
    ncol=2,
    columnspacing=1.2,
    handlelength=1.8
)


# ============================================================
# 8. 保存
# ============================================================
png_path = os.path.join(OUT_DIR, "fig5_risk_coverage_final_clean.png")
pdf_path = os.path.join(OUT_DIR, "fig5_risk_coverage_final_clean.pdf")
svg_path = os.path.join(OUT_DIR, "fig5_risk_coverage_final_clean.svg")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")
plt.show()

print("\nSaved:")
print(png_path)
print(pdf_path)
print(svg_path)

print("\nSummary:")
print(summary[[
    "method",
    "AUSE_MAE",
    "accepted_mae",
    "rejected_mae",
    "mae_ratio_rejected_over_accepted"
]])