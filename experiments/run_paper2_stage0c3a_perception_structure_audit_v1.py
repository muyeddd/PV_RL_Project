#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0C-3A
Paper1 perception residual / joint-structure audit before emulator construction.

AUDIT ONLY:
- no emulator is fitted;
- no WAPP q50/interval/width trajectory is generated;
- RANDOM_TEST and SEALED_DATES are forbidden;
- DECISION_DEVELOPMENT only.

Questions answered
------------------
1) What is the joint Paper1 perception-error structure conditional on true_L?
2) Does CQR width still relate to |q50-true_L| after controlling coarsely for q50?
3) How much lower-bound clipping exists in the WAPP-relevant low-loss region?
4) Are source frames strongly clustered by date/time, such that pooled frame
   resampling would over-weight correlated frames?
5) What constraints should the later emulator preserve?

The script uses timestamp only to diagnose within-day frame dependence.
It does NOT assume that second/minute-scale source autocorrelation should be
transferred to daily WAPP trajectories.

Outputs
-------
per_date_residual_audit.csv
trueL_bin_audit.csv
q50_bin_confounding_audit.csv
temporal_dependence_by_date.csv
audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 1844
EXPECTED_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_DATES = 12

Q50_BINS = 10
TRUE_L_BINS = 10
TEMPORAL_MAX_GAP_SECONDS = 300.0

SHUFFLE_REPS = 2000
SHUFFLE_SEED = 20260905

# WAPP P2-0C-1B observed support from the frozen audit.
EXPECTED_WAPP_VALID = 729
WAPP_RELEVANCE_MARGIN = 0.01


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


def spearman(a, b) -> float:
    aa = pd.to_numeric(pd.Series(a), errors="coerce")
    bb = pd.to_numeric(pd.Series(b), errors="coerce")
    ok = np.isfinite(aa) & np.isfinite(bb)
    if int(ok.sum()) < 3:
        return float("nan")
    if aa[ok].nunique() < 2 or bb[ok].nunique() < 2:
        return float("nan")
    return float(aa[ok].corr(bb[ok], method="spearman"))


def load_paper1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "sample_id", "date", "timestamp", "image_path", "role",
        "true_L", "q05", "q50", "q95", "lower", "upper", "width",
        "raw_width", "covered", "lower_clipped", "upper_clipped",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing Paper1 columns: {sorted(missing)}")

    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} rows, found {len(df)}")

    roles = set(df["role"].astype(str))
    if roles != {EXPECTED_ROLE}:
        raise PermissionError(
            f"Only {EXPECTED_ROLE} is authorized; found {sorted(roles)}"
        )

    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise RuntimeError("sample_id must be unique and non-null.")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    if df["date"].nunique() != EXPECTED_DATES:
        raise RuntimeError(
            f"Expected {EXPECTED_DATES} dates, found {df['date'].nunique()}"
        )

    if not np.array_equal(
        df["timestamp"].dt.strftime("%Y-%m-%d").to_numpy(),
        df["date"].to_numpy(),
    ):
        raise RuntimeError("timestamp calendar date does not match date column.")

    numeric_cols = [
        "true_L", "q05", "q50", "q95", "lower", "upper", "width", "raw_width"
    ]
    num = df[numeric_cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(num.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite numeric values in Paper1 DEV.")

    if (num["true_L"] < 0).any() or (num["true_L"] > 1).any():
        raise RuntimeError("true_L outside [0,1].")
    if (num["lower"] < 0).any() or (num["upper"] > 1).any():
        raise RuntimeError("interval outside [0,1].")
    if (num["lower"] > num["upper"]).any():
        raise RuntimeError("lower > upper.")
    if not np.allclose(
        num["upper"].to_numpy() - num["lower"].to_numpy(),
        num["width"].to_numpy(),
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("width != upper-lower.")

    covered_expected = (
        (num["true_L"] >= num["lower"]) & (num["true_L"] <= num["upper"])
    )
    if not np.array_equal(
        covered_expected.to_numpy(dtype=bool),
        df["covered"].astype(bool).to_numpy(),
    ):
        raise RuntimeError("covered flag inconsistent with interval.")

    return df


def load_wapp_bridge(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "L_power_proxy", "bridge_valid", "power_bridge_model"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing WAPP bridge columns: {sorted(missing)}")

    valid = df["bridge_valid"].fillna(False).astype(bool)
    if int(valid.sum()) != EXPECTED_WAPP_VALID:
        raise RuntimeError(
            f"Expected {EXPECTED_WAPP_VALID} valid WAPP bridge days, "
            f"found {int(valid.sum())}"
        )
    models = set(df.loc[valid, "power_bridge_model"].astype(str))
    if models != {"COMMON_TEMPERATURE_PVWATTS_RATIO"}:
        raise RuntimeError(f"Unexpected bridge model(s): {sorted(models)}")

    x = pd.to_numeric(df.loc[valid, "L_power_proxy"], errors="raise")
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite WAPP L_power_proxy.")
    return df.loc[valid].copy()


def add_residual_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["q50_error"] = out["q50"] - out["true_L"]
    out["abs_q50_error"] = out["q50_error"].abs()
    out["lower_offset_from_true"] = out["lower"] - out["true_L"]
    out["upper_offset_from_true"] = out["upper"] - out["true_L"]
    out["q50_minus_lower"] = out["q50"] - out["lower"]
    out["upper_minus_q50"] = out["upper"] - out["q50"]
    return out


def per_date_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, g in df.groupby("date", sort=True):
        rows.append({
            "date": date,
            "N": int(len(g)),
            "true_L_median": float(g["true_L"].median()),
            "q50_bias": float(g["q50_error"].mean()),
            "q50_mae": float(g["abs_q50_error"].mean()),
            "q50_error_std": float(g["q50_error"].std(ddof=1)),
            "width_mean": float(g["width"].mean()),
            "width_median": float(g["width"].median()),
            "coverage": float(g["covered"].mean()),
            "lower_clipped_fraction": float(g["lower_clipped"].mean()),
            "rho_width_abs_error": spearman(g["width"], g["abs_q50_error"]),
            "start_timestamp": str(g["timestamp"].min()),
            "end_timestamp": str(g["timestamp"].max()),
        })
    return pd.DataFrame(rows)


def make_quantile_bins(series: pd.Series, q: int) -> pd.Series:
    # Use rank(method='first') only to produce near-equal-size audit groups;
    # raw numeric values are retained and reported in every bin.
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=q, labels=False, duplicates="drop")


def trueL_bin_audit(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["trueL_bin"] = make_quantile_bins(work["true_L"], TRUE_L_BINS)
    rows = []
    for b, g in work.groupby("trueL_bin", sort=True):
        rows.append({
            "trueL_bin": int(b),
            "N": int(len(g)),
            "dates": int(g["date"].nunique()),
            "true_L_min": float(g["true_L"].min()),
            "true_L_median": float(g["true_L"].median()),
            "true_L_max": float(g["true_L"].max()),
            "q50_bias": float(g["q50_error"].mean()),
            "q50_mae": float(g["abs_q50_error"].mean()),
            "width_median": float(g["width"].median()),
            "width_iqr": float(
                g["width"].quantile(0.75) - g["width"].quantile(0.25)
            ),
            "coverage": float(g["covered"].mean()),
            "lower_clipped_fraction": float(g["lower_clipped"].mean()),
            "rho_width_abs_error": spearman(g["width"], g["abs_q50_error"]),
        })
    return pd.DataFrame(rows)


def q50_bin_confounding_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    work = df.copy()
    work["q50_bin"] = make_quantile_bins(work["q50"], Q50_BINS)

    rows = []
    within_width_rank = np.full(len(work), np.nan)
    within_error_rank = np.full(len(work), np.nan)

    for b, idx in work.groupby("q50_bin", sort=True).groups.items():
        g = work.loc[idx]
        rows.append({
            "q50_bin": int(b),
            "N": int(len(g)),
            "dates": int(g["date"].nunique()),
            "q50_min": float(g["q50"].min()),
            "q50_median": float(g["q50"].median()),
            "q50_max": float(g["q50"].max()),
            "true_L_median": float(g["true_L"].median()),
            "abs_error_median": float(g["abs_q50_error"].median()),
            "width_median": float(g["width"].median()),
            "lower_clipped_fraction": float(g["lower_clipped"].mean()),
            "rho_width_abs_error": spearman(g["width"], g["abs_q50_error"]),
        })
        within_width_rank[idx] = g["width"].rank(pct=True).to_numpy()
        within_error_rank[idx] = g["abs_q50_error"].rank(pct=True).to_numpy()

    pooled_within = spearman(within_width_rank, within_error_rank)
    return pd.DataFrame(rows), pooled_within


def within_q50_shuffle_null(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["q50_bin"] = make_quantile_bins(work["q50"], Q50_BINS)

    observed = spearman(work["width"], work["abs_q50_error"])
    rng = np.random.default_rng(SHUFFLE_SEED)
    widths = work["width"].to_numpy(dtype=float)
    errors = work["abs_q50_error"].to_numpy(dtype=float)
    bins = work["q50_bin"].to_numpy()

    null = np.empty(SHUFFLE_REPS, dtype=float)
    unique_bins = np.unique(bins)

    for r in range(SHUFFLE_REPS):
        shuffled = widths.copy()
        for b in unique_bins:
            idx = np.flatnonzero(bins == b)
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[r] = spearman(shuffled, errors)

    p_one_sided = float((1 + np.sum(null >= observed)) / (SHUFFLE_REPS + 1))
    return {
        "q50_bins": Q50_BINS,
        "repetitions": SHUFFLE_REPS,
        "seed": SHUFFLE_SEED,
        "observed_global_rho": observed,
        "shuffle_null_distribution": qstats(null),
        "observed_minus_null_mean": float(observed - np.mean(null)),
        "one_sided_empirical_p": p_one_sided,
        "interpretation": (
            "Shuffling width within q50 bins approximately preserves the "
            "width-vs-q50 relationship while destroying within-bin pairing "
            "between width and absolute q50 error."
        ),
    }


def temporal_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, g0 in df.groupby("date", sort=True):
        g = g0.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 3:
            continue

        gap = g["timestamp"].diff().dt.total_seconds().to_numpy(dtype=float)
        pair_ok = np.zeros(len(g), dtype=bool)
        pair_ok[1:] = (
            np.isfinite(gap[1:])
            & (gap[1:] > 0)
            & (gap[1:] <= TEMPORAL_MAX_GAP_SECONDS)
        )

        current = np.flatnonzero(pair_ok)
        previous = current - 1

        def pair_rho(col: str) -> float:
            if len(current) < 3:
                return float("nan")
            return spearman(
                g.loc[previous, col].to_numpy(),
                g.loc[current, col].to_numpy(),
            )

        rows.append({
            "date": date,
            "N": int(len(g)),
            "eligible_adjacent_pairs": int(len(current)),
            "pair_fraction_of_possible": float(
                len(current) / max(1, len(g) - 1)
            ),
            "gap_seconds_median_eligible": (
                float(np.median(gap[current])) if len(current) else np.nan
            ),
            "lag1_spearman_q50_error": pair_rho("q50_error"),
            "lag1_spearman_abs_q50_error": pair_rho("abs_q50_error"),
            "lag1_spearman_width": pair_rho("width"),
            "lag1_spearman_q50": pair_rho("q50"),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-0C-3A Paper1 perception structure audit."
    )
    p.add_argument("--paper1-dev-cqr", required=True, type=Path)
    p.add_argument("--wapp-power-bridge", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0c_3a_perception_structure_audit_v1"
        ),
    )
    args = p.parse_args()

    paper1_path = args.paper1_dev_cqr.expanduser().resolve()
    wapp_path = args.wapp_power_bridge.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for path in (paper1_path, wapp_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/7] Load authorized Paper1 DECISION_DEVELOPMENT CQR asset")
    p1 = add_residual_columns(load_paper1(paper1_path))

    print("[2/7] Load frozen WAPP semantic-bridge queries")
    wapp = load_wapp_bridge(wapp_path)
    wapp_l = pd.to_numeric(wapp["L_power_proxy"], errors="raise")
    wapp_max = float(wapp_l.max())
    relevance_hi = min(1.0, wapp_max + WAPP_RELEVANCE_MARGIN)
    relevant = p1[p1["true_L"] <= relevance_hi].copy()

    print("[3/7] Audit residuals by date and true-L bins")
    by_date = per_date_audit(p1)
    by_trueL = trueL_bin_audit(p1)

    print("[4/7] Audit q50-width confounding + within-bin shuffle null")
    by_q50, pooled_within_q50 = q50_bin_confounding_audit(p1)
    shuffle = within_q50_shuffle_null(p1)

    print("[5/7] Audit within-day temporal clustering")
    temporal = temporal_audit(p1)

    print("[6/7] Build scientific interpretation diagnostics")
    global_rho = spearman(p1["width"], p1["abs_q50_error"])
    relevant_rho = spearman(relevant["width"], relevant["abs_q50_error"])

    timestamp_monotonic_within_date = True
    for _, g in p1.groupby("date", sort=True):
        timestamp_monotonic_within_date &= bool(
            g.sort_values("timestamp")["timestamp"].is_monotonic_increasing
        )

    temporal_resid = pd.to_numeric(
        temporal["lag1_spearman_q50_error"], errors="coerce"
    )
    temporal_width = pd.to_numeric(
        temporal["lag1_spearman_width"], errors="coerce"
    )

    summary = {
        "stage": "P2-0C-3A",
        "audit_only": True,
        "emulator_fitted": False,
        "wapp_perception_trajectory_generated": False,
        "paper1_asset": {
            "role": EXPECTED_ROLE,
            "rows": int(len(p1)),
            "dates": int(p1["date"].nunique()),
            "random_test_used": False,
            "sealed_dates_used": False,
            "timestamp_available": True,
        },
        "wapp_relevance": {
            "valid_query_days": int(len(wapp)),
            "L_query_distribution": qstats(wapp_l),
            "paper1_relevant_true_L_upper": relevance_hi,
            "paper1_relevant_rows": int(len(relevant)),
            "paper1_relevant_dates": int(relevant["date"].nunique()),
        },
        "global_perception_structure": {
            "q50_error_distribution": qstats(p1["q50_error"]),
            "abs_q50_error_distribution": qstats(p1["abs_q50_error"]),
            "width_distribution": qstats(p1["width"]),
            "coverage": float(p1["covered"].mean()),
            "lower_clipped_fraction": float(p1["lower_clipped"].mean()),
            "upper_clipped_fraction": float(p1["upper_clipped"].mean()),
            "rho_width_abs_q50_error": global_rho,
            "rho_q50_width": spearman(p1["q50"], p1["width"]),
            "rho_trueL_width": spearman(p1["true_L"], p1["width"]),
        },
        "wapp_relevant_source_structure": {
            "true_L_distribution": qstats(relevant["true_L"]),
            "q50_error_distribution": qstats(relevant["q50_error"]),
            "abs_q50_error_distribution": qstats(relevant["abs_q50_error"]),
            "width_distribution": qstats(relevant["width"]),
            "coverage": float(relevant["covered"].mean()),
            "lower_clipped_fraction": float(relevant["lower_clipped"].mean()),
            "upper_clipped_fraction": float(relevant["upper_clipped"].mean()),
            "rho_width_abs_q50_error": relevant_rho,
        },
        "q50_confounding_audit": {
            "q50_bins": Q50_BINS,
            "pooled_within_q50_bin_rank_rho_width_abs_error": pooled_within_q50,
            "within_q50_shuffle_test": shuffle,
            "mandatory_future_ablation": (
                "shuffle width within q50 bins in RL evaluation, preserving "
                "the q50 distribution while breaking sample-level width/error pairing"
            ),
        },
        "date_cluster_audit": {
            "date_sample_count_distribution": qstats(by_date["N"]),
            "date_q50_bias_distribution": qstats(by_date["q50_bias"]),
            "date_q50_mae_distribution": qstats(by_date["q50_mae"]),
            "date_width_median_distribution": qstats(by_date["width_median"]),
            "date_coverage_distribution": qstats(by_date["coverage"]),
            "date_max_sample_share": float(
                by_date["N"].max() / by_date["N"].sum()
            ),
        },
        "temporal_diagnostic": {
            "max_gap_seconds_for_adjacent_pair": TEMPORAL_MAX_GAP_SECONDS,
            "dates_audited": int(len(temporal)),
            "eligible_pair_count_total": int(
                temporal["eligible_adjacent_pairs"].sum()
            ),
            "lag1_q50_error_rho_across_dates": qstats(temporal_resid),
            "lag1_width_rho_across_dates": qstats(temporal_width),
            "seconds_scale_dependence_will_not_be_transferred_to_daily_wapp": True,
        },
        "emulator_design_constraints": [
            "Condition on true_L only in v1; do not use WAPP irradiance.",
            "Resample the joint perception tuple/residual structure, not q50 and width independently.",
            "Use date-balanced source selection so dense source dates cannot dominate.",
            "Preserve lower-bound clipping semantics explicitly when transporting intervals to low-L WAPP queries.",
            "Validate the emulator by leave-one-date-out Paper1 reconstruction before generating WAPP trajectories.",
            "Do not transfer intraday frame autocorrelation directly to daily WAPP perception; test daily persistence separately as a sensitivity analysis.",
            "Keep a within-q50 width-shuffle RL ablation mandatory before claiming unique decision value from uncertainty width.",
        ],
        "notes": [
            "This audit diagnoses source structure; it does not declare width to be causally informative.",
            "A strong global width-error correlation can partly arise from q50/true-L regime effects and boundary clipping.",
            "Leave-one-date-out emulator validation is the next scientific gate.",
        ],
    }

    print("[7/7] Write audit outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_date.to_csv(
        out_dir / "per_date_residual_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_trueL.to_csv(
        out_dir / "trueL_bin_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_q50.to_csv(
        out_dir / "q50_bin_confounding_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    temporal.to_csv(
        out_dir / "temporal_dependence_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "per_date_residual_audit.csv")
    print(out_dir / "trueL_bin_audit.csv")
    print(out_dir / "q50_bin_confounding_audit.csv")
    print(out_dir / "temporal_dependence_by_date.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: source-structure audit only. Do NOT generate WAPP "
        "perception trajectories until leave-one-date-out emulator validation passes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
