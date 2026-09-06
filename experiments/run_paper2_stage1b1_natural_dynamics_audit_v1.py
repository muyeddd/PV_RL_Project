#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1B-1
Dry-accumulation and rain-response natural-dynamics audit.

PURPOSE
-------
Before fitting any counterfactual transition model, quantify what the frozen
WAPP field data actually support.

Primary questions:
1) What is the empirical distribution of uncontaminated DRY_NATURAL daily
   increments in the physical hidden-state / power-loss-proxy domain?
2) How much negative day-to-day motion remains on dry days (measurement /
   reconstruction variation, weak natural cleaning, wind effects, etc.)?
3) Are dry increments stable across Year1/Year2 and across calendar months?
4) What is the empirical rain-induced change in soiling for resolved,
   non-cleaning-confounded rain events?
5) Is rain response better described by an additive change, a fractional
   removal, or neither?

THIS STAGE DOES NOT
-------------------
- fit a simulator;
- clip negative dry increments away;
- force dry accumulation to be positive;
- assume rain resets soiling to zero;
- impose a monotonic rain-dose law;
- freeze Year1/Year2 as the RL split;
- train Gym/PPO.

FROZEN INPUTS
-------------
- P2-1A environment_master_ledger.csv
- P2-1A transition_audit.csv
- P2-0B-5.5c rain_response_audit.csv

PRIMARY DRY STATE
-----------------
Use delta_L_power_proxy / delta_S_physical as the physical-domain increment.
Observational delta_S_observed is retained as a sensitivity diagnostic.

RAIN METRICS
------------
For resolved, non-cleaning-confounded events with finite S_pre/S_post:
    delta_S = S_post - S_pre
    removal = S_pre - S_post  (positive means natural cleaning)

Fractional removal is only computed when S_pre >= 0.01 to avoid unstable
division near zero:
    removal_fraction = (S_pre - S_post) / S_pre

A 0.005 threshold is reported as sensitivity only.

PRIMARY CONSISTENCY GATES
-------------------------
- exactly 494 DRY_NATURAL transitions, consistent with frozen P2-1A;
- all primary dry physical increments finite;
- dry transitions are uncontaminated by destination manual cleaning,
  endpoint rain, maintenance, or invalid states;
- both audit years contain >= 150 dry transitions;
- exactly 69 rain-event rows;
- >= 30 resolved, non-cleaning-confounded, finite rain events for additive
  response diagnostics;
- >= 20 primary fractional-response events with S_pre >= 0.01.

These gates test DATA READINESS, not whether dry increments are positive or
whether rain response is monotonic.

OUTPUTS
-------
dry_transition_samples.csv
dry_overall_summary.csv
dry_year_summary.csv
dry_month_summary.csv
dry_source_clean_sensitivity.csv
rain_event_samples.csv
rain_overall_summary.csv
rain_bin_summary.csv
rain_pre_soiling_bin_summary.csv
audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DRY = 494
EXPECTED_RAIN_EVENTS = 69

MIN_DRY_PER_YEAR = 150
MIN_RAIN_ADDITIVE = 30
MIN_RAIN_FRACTIONAL = 20

PRIMARY_RATIO_PRE_SOIL_THRESHOLD = 0.01
SENSITIVITY_RATIO_PRE_SOIL_THRESHOLD = 0.005


def parse_bool_series(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.isna().any():
            raise RuntimeError(f"{name}: NaN in numeric boolean column.")
        if (~x.isin([0, 1])).any():
            raise RuntimeError(f"{name}: values outside 0/1.")
        return x.astype(int).astype(bool)

    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "y": True, "n": False,
    }
    x = s.astype(str).str.strip().str.lower()
    mapped = x.map(mapping)
    if mapped.isna().any():
        vals = sorted(x[mapped.isna()].unique().tolist())[:10]
        raise RuntimeError(f"{name}: unparseable boolean values {vals}")
    return mapped.astype(bool)


def qstats(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q = x.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q10": float(q.loc[0.10]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]),
        "q90": float(q.loc[0.90]),
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


def mad(values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)].to_numpy(dtype=float)
    if len(x) == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def sign_summary(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {}
    return {
        "fraction_negative": float((x < 0).mean()),
        "fraction_zero": float(np.isclose(x, 0.0, atol=1e-12, rtol=0.0).mean()),
        "fraction_positive": float((x > 0).mean()),
        "fraction_lt_minus_0p005": float((x < -0.005).mean()),
        "fraction_gt_plus_0p005": float((x > 0.005).mean()),
        "fraction_lt_minus_0p01": float((x < -0.01).mean()),
        "fraction_gt_plus_0p01": float((x > 0.01).mean()),
    }


def robust_outlier_fraction(values, k: float = 5.0) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)].to_numpy(dtype=float)
    if len(x) == 0:
        return {}
    med = float(np.median(x))
    m = float(np.median(np.abs(x - med)))
    if m <= 0:
        return {
            "median": med,
            "mad": m,
            "threshold_low": med,
            "threshold_high": med,
            "fraction_outside_median_plusminus_5MAD": float("nan"),
        }
    lo = med - k * m
    hi = med + k * m
    return {
        "median": med,
        "mad": m,
        "threshold_low": float(lo),
        "threshold_high": float(hi),
        "fraction_outside_median_plusminus_5MAD": float(
            ((x < lo) | (x > hi)).mean()
        ),
    }


def load_ledger(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "audit_period", "state_valid",
        "L_power_proxy", "S_soil_observed", "S_soil_physical",
        "rain_day", "rain_mm_day",
        "modb_manual_cleaning_day", "scheduled_maintenance",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Ledger missing: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    for col in [
        "state_valid", "rain_day",
        "modb_manual_cleaning_day", "scheduled_maintenance",
    ]:
        df[col] = parse_bool_series(df[col], col)

    for col in [
        "L_power_proxy", "S_soil_observed", "S_soil_physical", "rain_mm_day"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["date"].duplicated().any():
        raise RuntimeError("Ledger has duplicate dates.")
    return df


def load_transitions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "transition_index", "source_date", "dest_date",
        "source_state_valid", "dest_state_valid",
        "source_manual_clean_day", "dest_manual_clean_day",
        "source_rain_day", "dest_rain_day",
        "source_scheduled_maintenance", "dest_scheduled_maintenance",
        "rain_adjacent", "maintenance_adjacent",
        "transition_class", "natural_transition_candidate",
        "delta_S_observed", "delta_S_physical", "delta_L_power_proxy",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Transition audit missing: {sorted(missing)}")

    for col in ["source_date", "dest_date"]:
        df[col] = pd.to_datetime(df[col], errors="raise").dt.normalize()

    for col in [
        "source_state_valid", "dest_state_valid",
        "source_manual_clean_day", "dest_manual_clean_day",
        "source_rain_day", "dest_rain_day",
        "source_scheduled_maintenance", "dest_scheduled_maintenance",
        "rain_adjacent", "maintenance_adjacent",
        "natural_transition_candidate",
    ]:
        df[col] = parse_bool_series(df[col], col)

    for col in [
        "delta_S_observed", "delta_S_physical", "delta_L_power_proxy"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_rain(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "rain_event_id", "start_date", "end_date", "rain_days",
        "rain_mm_total", "rain_bin", "pre_date", "post_date",
        "S_pre", "S_post", "delta_S_post_minus_pre",
        "soiling_decreased_after_rain", "resolved", "cleaning_confounded",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Rain audit missing: {sorted(missing)}")

    for col in ["start_date", "end_date", "pre_date", "post_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()

    for col in [
        "soiling_decreased_after_rain", "resolved", "cleaning_confounded"
    ]:
        df[col] = parse_bool_series(df[col], col)

    for col in [
        "rain_days", "rain_mm_total", "S_pre", "S_post",
        "delta_S_post_minus_pre"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def dry_group_summary(g: pd.DataFrame, label_cols: dict) -> dict:
    x = g["delta_L_power_proxy"]
    row = {
        **label_cols,
        "n": int(len(g)),
        "delta_L_mean": float(x.mean()),
        "delta_L_median": float(x.median()),
        "delta_L_mad": mad(x),
        "delta_L_q05": float(x.quantile(0.05)),
        "delta_L_q25": float(x.quantile(0.25)),
        "delta_L_q75": float(x.quantile(0.75)),
        "delta_L_q95": float(x.quantile(0.95)),
        "fraction_negative": float((x < 0).mean()),
        "fraction_positive": float((x > 0).mean()),
        "fraction_abs_gt_0p005": float((x.abs() > 0.005).mean()),
        "source_L_median": float(g["source_L"].median()),
        "rho_source_L_vs_delta": spearman(g["source_L"], x),
        "source_manual_clean_fraction": float(
            g["source_manual_clean_day"].mean()
        ),
    }
    return row


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-1B-1 dry/rain natural-dynamics audit."
    )
    p.add_argument("--master-ledger", required=True, type=Path)
    p.add_argument("--transition-audit", required=True, type=Path)
    p.add_argument("--rain-audit", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1b1_natural_dynamics_audit_v1"
        ),
    )
    args = p.parse_args()

    paths = {
        "ledger": args.master_ledger.expanduser().resolve(),
        "transitions": args.transition_audit.expanduser().resolve(),
        "rain": args.rain_audit.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    print("[1/8] Load frozen P2-1A ledger/transitions and rain-event audit")
    ledger = load_ledger(paths["ledger"])
    transitions = load_transitions(paths["transitions"])
    rain = load_rain(paths["rain"])

    print("[2/8] Build uncontaminated dry-transition sample")
    dry = transitions[
        transitions["transition_class"].eq("DRY_NATURAL")
    ].copy()

    source_cols = ledger[
        [
            "date", "audit_period", "L_power_proxy",
            "S_soil_observed", "S_soil_physical",
        ]
    ].rename(
        columns={
            "date": "source_date",
            "audit_period": "source_period",
            "L_power_proxy": "source_L",
            "S_soil_observed": "source_S_observed",
            "S_soil_physical": "source_S_physical",
        }
    )

    dest_cols = ledger[
        ["date", "audit_period", "L_power_proxy"]
    ].rename(
        columns={
            "date": "dest_date",
            "audit_period": "dest_period",
            "L_power_proxy": "dest_L",
        }
    )

    dry = dry.merge(
        source_cols, on="source_date", how="left", validate="many_to_one"
    ).merge(
        dest_cols, on="dest_date", how="left", validate="many_to_one"
    )

    dry["dest_month"] = dry["dest_date"].dt.month.astype(int)

    dry_count_gate = bool(len(dry) == EXPECTED_DRY)

    physical_finite_gate = bool(
        np.isfinite(
            dry[
                ["delta_S_physical", "delta_L_power_proxy", "source_L", "dest_L"]
            ].to_numpy(dtype=float)
        ).all()
    )

    dry_integrity_gate = bool(
        dry["source_state_valid"].all()
        and dry["dest_state_valid"].all()
        and (~dry["dest_manual_clean_day"]).all()
        and (~dry["rain_adjacent"]).all()
        and (~dry["maintenance_adjacent"]).all()
    )

    print("[3/8] Audit dry increment distribution and state dependence")
    dry_overall_row = {
        "n": int(len(dry)),
        **{
            f"delta_L_{k}": v
            for k, v in qstats(dry["delta_L_power_proxy"]).items()
        },
        **{
            f"delta_L_sign_{k}": v
            for k, v in sign_summary(dry["delta_L_power_proxy"]).items()
        },
        "delta_L_mad": mad(dry["delta_L_power_proxy"]),
        "rho_source_L_vs_delta_L": spearman(
            dry["source_L"], dry["delta_L_power_proxy"]
        ),
        "rho_source_L_vs_abs_delta_L": spearman(
            dry["source_L"], dry["delta_L_power_proxy"].abs()
        ),
        "source_manual_clean_transition_count": int(
            dry["source_manual_clean_day"].sum()
        ),
        "observational_delta_distribution": qstats(
            dry["delta_S_observed"]
        ),
        "physical_delta_distribution": qstats(
            dry["delta_S_physical"]
        ),
        "robust_outlier_audit": robust_outlier_fraction(
            dry["delta_L_power_proxy"]
        ),
    }
    dry_overall = pd.DataFrame([{
        k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
        for k, v in dry_overall_row.items()
    }])

    print("[4/8] Audit Year1/Year2, monthly, and post-clean dry sensitivity")
    year_rows = []
    for period, g in dry.groupby("source_period", sort=True):
        year_rows.append(
            dry_group_summary(g, {"source_period": str(period)})
        )
    dry_year_summary = pd.DataFrame(year_rows)

    dry_year_counts_gate = bool(
        len(dry_year_summary) == 2
        and (dry_year_summary["n"] >= MIN_DRY_PER_YEAR).all()
    )

    month_rows = []
    for month, g in dry.groupby("dest_month", sort=True):
        month_rows.append(
            dry_group_summary(g, {"dest_month": int(month)})
        )
    dry_month_summary = pd.DataFrame(month_rows)

    sens_rows = []
    for label, mask in [
        ("ALL_DRY", np.ones(len(dry), dtype=bool)),
        ("EXCLUDE_SOURCE_MANUAL_CLEAN_DAY", ~dry["source_manual_clean_day"].to_numpy()),
        ("SOURCE_MANUAL_CLEAN_DAY_ONLY", dry["source_manual_clean_day"].to_numpy()),
    ]:
        g = dry.loc[mask]
        if len(g) == 0:
            continue
        sens_rows.append(
            dry_group_summary(g, {"subset": label})
        )
    dry_source_clean_sensitivity = pd.DataFrame(sens_rows)

    print("[5/8] Build resolved, unconfounded rain-response sample")
    rain_primary = rain[
        rain["resolved"].astype(bool)
        & (~rain["cleaning_confounded"].astype(bool))
    ].copy()

    finite_mask = np.isfinite(
        rain_primary[
            ["S_pre", "S_post", "rain_mm_total", "rain_days"]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    rain_primary = rain_primary.loc[finite_mask].copy()

    rain_primary["delta_S"] = (
        rain_primary["S_post"] - rain_primary["S_pre"]
    )
    rain_primary["removal"] = (
        rain_primary["S_pre"] - rain_primary["S_post"]
    )
    rain_primary["soiling_decreased_recomputed"] = (
        rain_primary["removal"] > 0
    )

    rain_primary["fractional_primary_eligible"] = (
        rain_primary["S_pre"] >= PRIMARY_RATIO_PRE_SOIL_THRESHOLD
    )
    rain_primary["fractional_sensitivity_eligible"] = (
        rain_primary["S_pre"] >= SENSITIVITY_RATIO_PRE_SOIL_THRESHOLD
    )

    rain_primary["removal_fraction_primary"] = np.where(
        rain_primary["fractional_primary_eligible"],
        rain_primary["removal"] / rain_primary["S_pre"],
        np.nan,
    )
    rain_primary["removal_fraction_sensitivity"] = np.where(
        rain_primary["fractional_sensitivity_eligible"],
        rain_primary["removal"] / rain_primary["S_pre"],
        np.nan,
    )

    rain_count_gate = bool(len(rain) == EXPECTED_RAIN_EVENTS)
    rain_additive_gate = bool(len(rain_primary) >= MIN_RAIN_ADDITIVE)
    rain_fractional_n = int(
        rain_primary["fractional_primary_eligible"].sum()
    )
    rain_fractional_gate = bool(
        rain_fractional_n >= MIN_RAIN_FRACTIONAL
    )

    print("[6/8] Audit rain additive/fractional response and dose diagnostics")
    frac_primary = rain_primary.loc[
        rain_primary["fractional_primary_eligible"],
        "removal_fraction_primary",
    ]
    frac_sens = rain_primary.loc[
        rain_primary["fractional_sensitivity_eligible"],
        "removal_fraction_sensitivity",
    ]

    rain_overall_row = {
        "rain_event_rows_total": int(len(rain)),
        "resolved_unconfounded_finite_events": int(len(rain_primary)),
        "fractional_primary_eligible_events": rain_fractional_n,
        "fractional_sensitivity_eligible_events": int(
            rain_primary["fractional_sensitivity_eligible"].sum()
        ),
        "soiling_decrease_fraction": float(
            rain_primary["soiling_decreased_recomputed"].mean()
        ) if len(rain_primary) else np.nan,
        "delta_S_distribution": qstats(rain_primary["delta_S"]),
        "removal_distribution": qstats(rain_primary["removal"]),
        "removal_fraction_primary_distribution": qstats(frac_primary),
        "removal_fraction_sensitivity_distribution": qstats(frac_sens),
        "rho_rain_mm_vs_removal": spearman(
            rain_primary["rain_mm_total"], rain_primary["removal"]
        ),
        "rho_rain_mm_vs_removal_fraction_primary": spearman(
            rain_primary.loc[
                rain_primary["fractional_primary_eligible"],
                "rain_mm_total",
            ],
            frac_primary,
        ),
        "rho_S_pre_vs_removal": spearman(
            rain_primary["S_pre"], rain_primary["removal"]
        ),
        "rho_S_pre_vs_removal_fraction_primary": spearman(
            rain_primary.loc[
                rain_primary["fractional_primary_eligible"],
                "S_pre",
            ],
            frac_primary,
        ),
    }
    rain_overall = pd.DataFrame([{
        k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
        for k, v in rain_overall_row.items()
    }])

    rain_bin_rows = []
    for rain_bin, g in rain_primary.groupby("rain_bin", dropna=False, sort=True):
        primary = g[g["fractional_primary_eligible"]]
        rain_bin_rows.append({
            "rain_bin": str(rain_bin),
            "n": int(len(g)),
            "rain_mm_median": float(g["rain_mm_total"].median()),
            "S_pre_median": float(g["S_pre"].median()),
            "delta_S_median": float(g["delta_S"].median()),
            "removal_median": float(g["removal"].median()),
            "soiling_decrease_fraction": float(
                g["soiling_decreased_recomputed"].mean()
            ),
            "fractional_eligible_n": int(len(primary)),
            "removal_fraction_median": (
                float(primary["removal_fraction_primary"].median())
                if len(primary) else np.nan
            ),
        })
    rain_bin_summary = pd.DataFrame(rain_bin_rows)

    # Pre-soiling bins are fixed, physically interpretable bands.
    bins = [-np.inf, 0.01, 0.03, 0.06, np.inf]
    labels = ["<0.01", "0.01-0.03", "0.03-0.06", ">=0.06"]
    rain_primary["pre_soiling_bin"] = pd.cut(
        rain_primary["S_pre"],
        bins=bins,
        labels=labels,
        right=False,
    )

    pre_bin_rows = []
    for label in labels:
        g = rain_primary[
            rain_primary["pre_soiling_bin"].astype(str).eq(label)
        ]
        if len(g) == 0:
            continue
        eligible = g[g["fractional_primary_eligible"]]
        pre_bin_rows.append({
            "pre_soiling_bin": label,
            "n": int(len(g)),
            "S_pre_median": float(g["S_pre"].median()),
            "rain_mm_median": float(g["rain_mm_total"].median()),
            "removal_median": float(g["removal"].median()),
            "soiling_decrease_fraction": float(
                g["soiling_decreased_recomputed"].mean()
            ),
            "fractional_eligible_n": int(len(eligible)),
            "removal_fraction_median": (
                float(eligible["removal_fraction_primary"].median())
                if len(eligible) else np.nan
            ),
        })
    rain_pre_soiling_bin_summary = pd.DataFrame(pre_bin_rows)

    print("[7/8] Evaluate P2-1B-1 data-readiness gates")
    all_primary = bool(
        dry_count_gate
        and physical_finite_gate
        and dry_integrity_gate
        and dry_year_counts_gate
        and rain_count_gate
        and rain_additive_gate
        and rain_fractional_gate
    )

    summary = {
        "stage": "P2-1B-1",
        "audit_only": True,
        "transition_model_fitted": False,
        "counterfactual_environment_built": False,
        "rl_started": False,
        "dry_dynamics": {
            "dry_transitions": int(len(dry)),
            "dry_count_expected": EXPECTED_DRY,
            "delta_L_distribution": qstats(dry["delta_L_power_proxy"]),
            "delta_L_sign": sign_summary(dry["delta_L_power_proxy"]),
            "delta_L_mad": mad(dry["delta_L_power_proxy"]),
            "rho_source_L_vs_delta_L": spearman(
                dry["source_L"], dry["delta_L_power_proxy"]
            ),
            "rho_source_L_vs_abs_delta_L": spearman(
                dry["source_L"], dry["delta_L_power_proxy"].abs()
            ),
            "source_manual_clean_transition_count": int(
                dry["source_manual_clean_day"].sum()
            ),
            "robust_outlier_audit": robust_outlier_fraction(
                dry["delta_L_power_proxy"]
            ),
        },
        "dry_cross_period": {
            "year_rows": dry_year_summary.to_dict(orient="records"),
            "months_represented": int(dry["dest_month"].nunique()),
            "dry_per_year_gate_minimum": MIN_DRY_PER_YEAR,
            "dry_per_year_gate_pass": dry_year_counts_gate,
        },
        "rain_response": {
            "rain_event_rows_total": int(len(rain)),
            "resolved_unconfounded_finite_events": int(len(rain_primary)),
            "additive_minimum": MIN_RAIN_ADDITIVE,
            "additive_gate_pass": rain_additive_gate,
            "fractional_primary_threshold_S_pre": (
                PRIMARY_RATIO_PRE_SOIL_THRESHOLD
            ),
            "fractional_primary_eligible_events": rain_fractional_n,
            "fractional_minimum": MIN_RAIN_FRACTIONAL,
            "fractional_gate_pass": rain_fractional_gate,
            "soiling_decrease_fraction": float(
                rain_primary["soiling_decreased_recomputed"].mean()
            ) if len(rain_primary) else np.nan,
            "delta_S_distribution": qstats(rain_primary["delta_S"]),
            "removal_distribution": qstats(rain_primary["removal"]),
            "removal_fraction_primary_distribution": qstats(frac_primary),
            "rho_rain_mm_vs_removal": spearman(
                rain_primary["rain_mm_total"], rain_primary["removal"]
            ),
            "rho_rain_mm_vs_removal_fraction_primary": spearman(
                rain_primary.loc[
                    rain_primary["fractional_primary_eligible"],
                    "rain_mm_total",
                ],
                frac_primary,
            ),
            "rho_S_pre_vs_removal": spearman(
                rain_primary["S_pre"], rain_primary["removal"]
            ),
            "important_caution": (
                "No monotonic rain-dose relationship is assumed or claimed. "
                "Fractional response is not evaluated near zero pre-soiling."
            ),
        },
        "primary_gates": {
            "dry_count_pass": dry_count_gate,
            "dry_physical_finite_pass": physical_finite_gate,
            "dry_integrity_pass": dry_integrity_gate,
            "dry_year_support_pass": dry_year_counts_gate,
            "rain_event_count_pass": rain_count_gate,
            "rain_additive_support_pass": rain_additive_gate,
            "rain_fractional_support_pass": rain_fractional_gate,
            "all_primary_gates_pass": all_primary,
        },
        "decision_after_review": (
            "If PASS, do not automatically fit a deterministic positive drift. "
            "Use the observed sign/spread/year/month/rain diagnostics to choose "
            "a minimally sufficient stochastic counterfactual dynamics model."
        ),
        "next_step_if_pass": (
            "Review dry/rain structure, then P2-1B-2 should compare a small "
            "number of scientifically justified transition-model candidates "
            "before freezing P2-1C environment dynamics."
        ),
        "next_step_if_fail": (
            "Do not build a simulator. Resolve insufficient or inconsistent "
            "dry/rain dynamics support first."
        ),
    }

    print("[8/8] Write P2-1B-1 audit outputs")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dry.to_csv(
        out_dir / "dry_transition_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dry_overall.to_csv(
        out_dir / "dry_overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dry_year_summary.to_csv(
        out_dir / "dry_year_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dry_month_summary.to_csv(
        out_dir / "dry_month_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dry_source_clean_sensitivity.to_csv(
        out_dir / "dry_source_clean_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rain_primary.to_csv(
        out_dir / "rain_event_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rain_overall.to_csv(
        out_dir / "rain_overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rain_bin_summary.to_csv(
        out_dir / "rain_bin_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rain_pre_soiling_bin_summary.to_csv(
        out_dir / "rain_pre_soiling_bin_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "dry_transition_samples.csv")
    print(out_dir / "dry_overall_summary.csv")
    print(out_dir / "dry_year_summary.csv")
    print(out_dir / "dry_month_summary.csv")
    print(out_dir / "dry_source_clean_sensitivity.csv")
    print(out_dir / "rain_event_samples.csv")
    print(out_dir / "rain_overall_summary.csv")
    print(out_dir / "rain_bin_summary.csv")
    print(out_dir / "rain_pre_soiling_bin_summary.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: P2-1B-1 is an audit only. Do NOT build Gym/PPO or force "
        "negative dry increments to zero before scientific review."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
