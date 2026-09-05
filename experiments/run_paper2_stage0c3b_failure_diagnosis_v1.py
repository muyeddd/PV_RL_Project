#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0C-3B-FD
Failure diagnosis for the rejected P2-0C-3B emulator-v1.

AUDIT ONLY.
This script does NOT change any gate, radius, support threshold, source
weighting, conformal qhat, or emulator output. It does NOT fit/generate a new
emulator. Its purpose is to localize why 3B failed before any v2 design.

Questions
---------
1) Are LODO support failures concentrated outside the actual WAPP deployment
   domain, or do actual WAPP query values also lose support when one Paper1
   date is removed?
2) Which held-out dates / true-L bands drive unsupported targets?
3) Is the WAPP-relevant distributional failure mainly a lower-bound /
   coverage-calibration problem rather than q50 bias/MAE/width magnitude?
4) Which Paper1 dates and low-L bands have the strongest lower clipping?

Inputs
------
- Paper1 DECISION_DEVELOPMENT cqr_predictions.csv
- P2-0C-1B daily_common_temp_power_bridge.csv
- P2-0C-3B lodo_target_support.csv
- P2-0C-3B lodo_date_metrics.csv

Outputs
-------
support_by_heldout_date.csv
support_by_loss_band.csv
wapp_query_support_by_removed_date.csv
boundary_by_date.csv
boundary_by_loss_band.csv
failure_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_P1_N = 1844
EXPECTED_P1_DATES = 12
EXPECTED_WAPP_VALID = 729
EXPECTED_ROLE = "DECISION_DEVELOPMENT"

LOCAL_RADIUS = 0.01
LOCAL_MIN_SAMPLES = 20
LOCAL_MIN_DATES = 3
QHAT = 0.004862844288256299

LOSS_BANDS = [
    (-np.inf, 0.02, "<0.02"),
    (0.02, 0.05, "0.02-0.05"),
    (0.05, 0.10, "0.05-0.10"),
    (0.10, 0.14, "0.10-0.14"),
    (0.14, 0.16, "0.14-0.16"),
    (0.16, 0.17, "0.16-0.17"),
    (0.17, np.inf, ">=0.17"),
]


def qstats(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "max": float(x.max()),
    }


def assign_band(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    labels = np.empty(len(arr), dtype=object)
    for lo, hi, label in LOSS_BANDS:
        mask = (arr >= lo) & (arr < hi)
        labels[mask] = label
    return pd.Series(labels, index=values.index, dtype="object")


def load_p1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "sample_id", "date", "role", "true_L",
        "q05", "q50", "q95", "lower", "upper", "width",
        "covered", "lower_clipped",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing Paper1 columns: {sorted(missing)}")

    if len(df) != EXPECTED_P1_N:
        raise RuntimeError(f"Expected {EXPECTED_P1_N} Paper1 rows, found {len(df)}")

    roles = set(df["role"].astype(str))
    if roles != {EXPECTED_ROLE}:
        raise PermissionError(f"Unexpected Paper1 roles: {sorted(roles)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if df["date"].nunique() != EXPECTED_P1_DATES:
        raise RuntimeError(
            f"Expected {EXPECTED_P1_DATES} Paper1 dates, found {df['date'].nunique()}"
        )

    for col in ["true_L", "q05", "q50", "q95", "lower", "upper", "width"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(
        df[["true_L", "q05", "q50", "q95", "lower", "upper", "width"]]
        .to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite Paper1 values.")

    expected_clip = df["q05"].to_numpy(dtype=float) <= (QHAT + 1e-12)
    actual_clip = df["lower_clipped"].astype(bool).to_numpy()
    if not np.array_equal(expected_clip, actual_clip):
        raise RuntimeError(
            "Paper1 lower_clipped does not match q05 <= frozen qhat."
        )

    return df


def load_wapp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "L_power_proxy", "bridge_valid", "power_bridge_model"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing WAPP columns: {sorted(missing)}")

    valid = df["bridge_valid"].fillna(False).astype(bool)
    out = df.loc[valid].copy()
    if len(out) != EXPECTED_WAPP_VALID:
        raise RuntimeError(
            f"Expected {EXPECTED_WAPP_VALID} valid WAPP rows, found {len(out)}"
        )

    models = set(out["power_bridge_model"].astype(str))
    if models != {"COMMON_TEMPERATURE_PVWATTS_RATIO"}:
        raise RuntimeError(f"Unexpected bridge model(s): {sorted(models)}")

    out["L_power_proxy"] = pd.to_numeric(
        out["L_power_proxy"], errors="raise"
    )
    return out


def load_support(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "target_sample_id", "target_date", "target_true_L",
        "candidate_samples", "candidate_dates", "support_ok", "wapp_relevant",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing LODO support columns: {sorted(missing)}")
    if len(df) != EXPECTED_P1_N:
        raise RuntimeError(
            f"Expected {EXPECTED_P1_N} LODO support rows, found {len(df)}"
        )
    df["target_date"] = pd.to_datetime(
        df["target_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    df["target_true_L"] = pd.to_numeric(
        df["target_true_L"], errors="raise"
    )
    return df


def load_date_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "heldout_date",
        "support_fraction",
        "wapp_relevant_support_fraction",
        "actual_bias", "generated_bias",
        "actual_mae", "generated_mae",
        "actual_width_median", "generated_width_median",
        "actual_coverage", "generated_coverage",
        "actual_lower_clipped_fraction",
        "generated_lower_clipped_fraction",
        "actual_rho_width_abs_error",
        "generated_rho_width_abs_error",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing date-metric columns: {sorted(missing)}")
    if len(df) != EXPECTED_P1_DATES:
        raise RuntimeError(
            f"Expected {EXPECTED_P1_DATES} held-out date rows, found {len(df)}"
        )
    df["heldout_date"] = pd.to_datetime(
        df["heldout_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    return df


def support_by_heldout_date(support: pd.DataFrame, wapp_max: float) -> pd.DataFrame:
    rows = []
    for date, g in support.groupby("target_date", sort=True):
        actual_domain = g["target_true_L"] <= wapp_max
        margin_domain = g["target_true_L"] <= (wapp_max + 0.01)

        def frac(mask) -> float:
            if int(mask.sum()) == 0:
                return np.nan
            return float(g.loc[mask, "support_ok"].astype(bool).mean())

        unsupported = g[~g["support_ok"].astype(bool)]
        rows.append({
            "heldout_date": date,
            "N": int(len(g)),
            "supported": int(g["support_ok"].astype(bool).sum()),
            "unsupported": int((~g["support_ok"].astype(bool)).sum()),
            "support_fraction": float(g["support_ok"].astype(bool).mean()),
            "N_actual_wapp_domain": int(actual_domain.sum()),
            "actual_wapp_domain_support_fraction": frac(actual_domain),
            "N_margin_domain": int(margin_domain.sum()),
            "margin_domain_support_fraction": frac(margin_domain),
            "unsupported_true_L_min": (
                float(unsupported["target_true_L"].min())
                if len(unsupported) else np.nan
            ),
            "unsupported_true_L_max": (
                float(unsupported["target_true_L"].max())
                if len(unsupported) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def support_by_loss_band(support: pd.DataFrame) -> pd.DataFrame:
    work = support.copy()
    work["loss_band"] = assign_band(work["target_true_L"])

    rows = []
    for _, _, label in LOSS_BANDS:
        g = work[work["loss_band"].eq(label)]
        if len(g) == 0:
            continue
        rows.append({
            "loss_band": label,
            "targets": int(len(g)),
            "supported": int(g["support_ok"].astype(bool).sum()),
            "unsupported": int((~g["support_ok"].astype(bool)).sum()),
            "support_fraction": float(g["support_ok"].astype(bool).mean()),
            "candidate_samples_median": float(g["candidate_samples"].median()),
            "candidate_dates_median": float(g["candidate_dates"].median()),
            "dates_represented": int(g["target_date"].nunique()),
        })
    return pd.DataFrame(rows)


def audit_actual_wapp_queries_under_removed_dates(
    p1: pd.DataFrame,
    wapp: pd.DataFrame,
) -> pd.DataFrame:
    q = wapp["L_power_proxy"].to_numpy(dtype=float)
    rows = []

    for removed_date in sorted(p1["date"].unique()):
        source = p1[~p1["date"].eq(removed_date)].copy()
        src_l = source["true_L"].to_numpy(dtype=float)
        src_dates = source["date"].to_numpy(dtype=str)

        counts = np.zeros(len(q), dtype=int)
        date_counts = np.zeros(len(q), dtype=int)

        for i, val in enumerate(q):
            m = np.abs(src_l - val) <= LOCAL_RADIUS
            counts[i] = int(m.sum())
            date_counts[i] = int(np.unique(src_dates[m]).size)

        ok = (
            (counts >= LOCAL_MIN_SAMPLES)
            & (date_counts >= LOCAL_MIN_DATES)
        )

        rows.append({
            "removed_paper1_date": removed_date,
            "wapp_queries": int(len(q)),
            "supported_queries": int(ok.sum()),
            "unsupported_queries": int((~ok).sum()),
            "supported_fraction": float(ok.mean()),
            "candidate_samples_min": int(counts.min()),
            "candidate_samples_q05": float(np.quantile(counts, 0.05)),
            "candidate_samples_median": float(np.median(counts)),
            "candidate_dates_min": int(date_counts.min()),
            "candidate_dates_q05": float(np.quantile(date_counts, 0.05)),
            "candidate_dates_median": float(np.median(date_counts)),
            "unsupported_L_min": float(q[~ok].min()) if (~ok).any() else np.nan,
            "unsupported_L_max": float(q[~ok].max()) if (~ok).any() else np.nan,
        })

    return pd.DataFrame(rows)


def boundary_by_date(p1: pd.DataFrame, wapp_max: float) -> pd.DataFrame:
    rows = []
    for date, g in p1.groupby("date", sort=True):
        actual_domain = g[g["true_L"] <= wapp_max]
        margin_domain = g[g["true_L"] <= (wapp_max + 0.01)]

        def safe_frac(sub: pd.DataFrame, col: str) -> float:
            return float(sub[col].astype(bool).mean()) if len(sub) else np.nan

        rows.append({
            "date": date,
            "N_all": int(len(g)),
            "lower_clip_all": safe_frac(g, "lower_clipped"),
            "coverage_all": float(g["covered"].mean()),
            "N_actual_wapp_domain": int(len(actual_domain)),
            "lower_clip_actual_wapp_domain": safe_frac(
                actual_domain, "lower_clipped"
            ),
            "coverage_actual_wapp_domain": (
                float(actual_domain["covered"].mean())
                if len(actual_domain) else np.nan
            ),
            "N_margin_domain": int(len(margin_domain)),
            "lower_clip_margin_domain": safe_frac(
                margin_domain, "lower_clipped"
            ),
            "coverage_margin_domain": (
                float(margin_domain["covered"].mean())
                if len(margin_domain) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def boundary_by_loss_band(p1: pd.DataFrame) -> pd.DataFrame:
    work = p1.copy()
    work["loss_band"] = assign_band(work["true_L"])
    rows = []

    for _, _, label in LOSS_BANDS:
        g = work[work["loss_band"].eq(label)]
        if len(g) == 0:
            continue
        rows.append({
            "loss_band": label,
            "N": int(len(g)),
            "dates": int(g["date"].nunique()),
            "true_L_median": float(g["true_L"].median()),
            "q05_median": float(g["q05"].median()),
            "lower_clipped_fraction": float(
                g["lower_clipped"].astype(bool).mean()
            ),
            "coverage": float(g["covered"].astype(bool).mean()),
            "q50_mae": float(
                np.mean(np.abs(g["q50"].to_numpy() - g["true_L"].to_numpy()))
            ),
            "width_median": float(g["width"].median()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-0C-3B failure diagnosis without changing emulator gates."
    )
    p.add_argument("--paper1-dev-cqr", required=True, type=Path)
    p.add_argument("--wapp-power-bridge", required=True, type=Path)
    p.add_argument("--lodo-support", required=True, type=Path)
    p.add_argument("--lodo-date-metrics", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0c_3b_failure_diagnosis_v1"
        ),
    )
    args = p.parse_args()

    paths = [
        args.paper1_dev_cqr.expanduser().resolve(),
        args.wapp_power_bridge.expanduser().resolve(),
        args.lodo_support.expanduser().resolve(),
        args.lodo_date_metrics.expanduser().resolve(),
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/7] Load Paper1 DEV and WAPP deployment queries")
    p1 = load_p1(paths[0])
    wapp = load_wapp(paths[1])
    wapp_max = float(wapp["L_power_proxy"].max())

    print("[2/7] Load rejected 3B support and date metrics")
    support = load_support(paths[2])
    date_metrics = load_date_metrics(paths[3])

    print("[3/7] Localize LODO support failures")
    by_date = support_by_heldout_date(support, wapp_max)
    by_band = support_by_loss_band(support)

    print("[4/7] Audit actual 729 WAPP queries after removing each Paper1 date")
    wapp_removed = audit_actual_wapp_queries_under_removed_dates(p1, wapp)

    print("[5/7] Audit boundary clipping by source date and loss band")
    boundary_date = boundary_by_date(p1, wapp_max)
    boundary_band = boundary_by_loss_band(p1)

    print("[6/7] Decompose 3B distributional mismatch")
    date_metrics = date_metrics.copy()
    date_metrics["coverage_gap_generated_minus_actual"] = (
        date_metrics["generated_coverage"] - date_metrics["actual_coverage"]
    )
    date_metrics["lower_clip_gap_generated_minus_actual"] = (
        date_metrics["generated_lower_clipped_fraction"]
        - date_metrics["actual_lower_clipped_fraction"]
    )
    date_metrics["mae_gap_generated_minus_actual"] = (
        date_metrics["generated_mae"] - date_metrics["actual_mae"]
    )

    worst_coverage = date_metrics.loc[
        date_metrics["coverage_gap_generated_minus_actual"].abs().idxmax()
    ]
    worst_clip = date_metrics.loc[
        date_metrics["lower_clip_gap_generated_minus_actual"].abs().idxmax()
    ]

    actual_domain = support["target_true_L"] <= wapp_max
    margin_domain = support["target_true_L"] <= (wapp_max + 0.01)

    summary = {
        "stage": "P2-0C-3B-FD",
        "diagnostic_only": True,
        "emulator_v1_status": "REJECTED_NOT_FROZEN",
        "no_gate_changed": True,
        "no_fallback_added": True,
        "deployment_domain": {
            "wapp_valid_queries": int(len(wapp)),
            "wapp_L_max": wapp_max,
            "margin_upper_used_in_3b": wapp_max + 0.01,
        },
        "lodo_target_support": {
            "global_fraction": float(
                support["support_ok"].astype(bool).mean()
            ),
            "actual_wapp_domain_rows": int(actual_domain.sum()),
            "actual_wapp_domain_supported_fraction": float(
                support.loc[actual_domain, "support_ok"].astype(bool).mean()
            ),
            "margin_domain_rows": int(margin_domain.sum()),
            "margin_domain_supported_fraction": float(
                support.loc[margin_domain, "support_ok"].astype(bool).mean()
            ),
            "unsupported_distribution": qstats(
                support.loc[
                    ~support["support_ok"].astype(bool),
                    "target_true_L",
                ]
            ),
        },
        "actual_wapp_query_support_after_each_date_removal": {
            "supported_fraction_across_removed_dates": qstats(
                wapp_removed["supported_fraction"]
            ),
            "minimum_supported_fraction": float(
                wapp_removed["supported_fraction"].min()
            ),
            "worst_removed_date": str(
                wapp_removed.loc[
                    wapp_removed["supported_fraction"].idxmin(),
                    "removed_paper1_date",
                ]
            ),
            "minimum_candidate_samples_over_all_removed_dates": int(
                wapp_removed["candidate_samples_min"].min()
            ),
            "minimum_candidate_dates_over_all_removed_dates": int(
                wapp_removed["candidate_dates_min"].min()
            ),
        },
        "distribution_failure_signature": {
            "global_generated_coverage_mean": float(
                date_metrics["generated_coverage"].mean()
            ),
            "global_actual_coverage_mean": float(
                date_metrics["actual_coverage"].mean()
            ),
            "global_generated_lower_clip_mean": float(
                date_metrics["generated_lower_clipped_fraction"].mean()
            ),
            "global_actual_lower_clip_mean": float(
                date_metrics["actual_lower_clipped_fraction"].mean()
            ),
            "worst_coverage_gap_date": str(worst_coverage["heldout_date"]),
            "worst_coverage_gap": float(
                worst_coverage["coverage_gap_generated_minus_actual"]
            ),
            "worst_lower_clip_gap_date": str(worst_clip["heldout_date"]),
            "worst_lower_clip_gap": float(
                worst_clip["lower_clip_gap_generated_minus_actual"]
            ),
        },
        "decision_rule": (
            "Do not design emulator-v2 until this diagnosis is reviewed. "
            "If deployment-query support remains high but clipping/coverage "
            "mismatch dominates, revise interval/boundary transport rather "
            "than widening the L radius."
        ),
    }

    print("[7/7] Write failure-diagnosis outputs")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_date.to_csv(
        out_dir / "support_by_heldout_date.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_band.to_csv(
        out_dir / "support_by_loss_band.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wapp_removed.to_csv(
        out_dir / "wapp_query_support_by_removed_date.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boundary_date.to_csv(
        out_dir / "boundary_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boundary_band.to_csv(
        out_dir / "boundary_by_loss_band.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out_dir / "failure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "support_by_heldout_date.csv")
    print(out_dir / "support_by_loss_band.csv")
    print(out_dir / "wapp_query_support_by_removed_date.csv")
    print(out_dir / "boundary_by_date.csv")
    print(out_dir / "boundary_by_loss_band.csv")
    print(out_dir / "failure_summary.json")
    print(
        "IMPORTANT: diagnosis only. Do not widen support radius or generate "
        "WAPP perception trajectories from emulator-v1."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
