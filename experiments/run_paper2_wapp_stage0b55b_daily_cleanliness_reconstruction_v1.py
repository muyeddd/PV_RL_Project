#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0B-5.5b
WAPP Malanville daily cleanliness reconstruction v1.

This stage consumes the FROZEN 5.5a primary baseline rule (min_support=30)
and constructs the observational daily cleanliness state:

    C_d^WAPP = R_d^corr / B_j
    S_d^soil = 1 - C_d^WAPP

Important semantics
-------------------
- B_j is the cycle-specific operational clean reference after authoritative
  Table-12 ModB manual cleaning.
- Rain never creates a new cycle and never resets C to 1.
- In-period ModB cleaning days use the SAME-DAY POST-CLEAN baseline
  observation as that day's state. The mixed all-day ratio is retained only
  as a diagnostic and is not used as the physical daily state.
- Cycle 0 is anchored by 2021-08-08 pre-period provenance; its baseline is
  the first valid report-period day (2021-08-09 under the frozen 5.5a audit).
- Observational reconstruction is NEVER clipped:
      C_d^WAPP > 1 is retained;
      S_d^soil < 0 is retained.
  Physical simulator projection, if needed, is a later stage and must report
  any boundary projection explicitly.
- This stage does NOT create the Paper1 power-loss bridge and does NOT build
  an RL state.

Expected upstream result from 5.5a
----------------------------------
- 27 cycles
- 1 pre-period anchor cycle
- 26 observed Table-12 ModB cleaning cycles
- min_support=30 resolves all 27 cycles
- all 26 in-period cycles use SAME_DAY_POST_CLEAN
- cycle 0 uses PREPERIOD_ANCHOR_FIRST_VALID_DAY
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_SUPPORT = 30
EXPECTED_DAYS = 730
EXPECTED_CYCLES = 27
EXPECTED_INPERIOD_CLEANINGS = 26
MIN_REQUIRED_COVERAGE = 0.95


def finite_quantiles(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    qs = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q01": float(qs.loc[0.01]),
        "q05": float(qs.loc[0.05]),
        "q25": float(qs.loc[0.25]),
        "q50": float(qs.loc[0.50]),
        "q75": float(qs.loc[0.75]),
        "q95": float(qs.loc[0.95]),
        "q99": float(qs.loc[0.99]),
        "max": float(x.max()),
    }


def load_inputs(
    daily_path: Path,
    cycles_path: Path,
    candidates_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(daily_path, encoding="utf-8-sig")
    cycles = pd.read_csv(cycles_path, encoding="utf-8-sig")
    cand = pd.read_csv(candidates_path, encoding="utf-8-sig")

    daily_required = {
        "date",
        "n_valid",
        "ratio_corr_median",
        "ratio_corr_mad",
        "ratio_corr_iqr",
        "rain_mm_day",
        "has_cleaning_pulse",
        "scheduled_maintenance",
    }
    cycle_required = {
        "cycle_id",
        "cycle_start_boundary",
        "cycle_report_start",
        "cycle_report_end",
        "anchor_type",
        "observed_cleaning_event",
    }
    cand_required = {
        "cycle_id",
        "min_support",
        "cycle_start_boundary",
        "baseline_resolved",
        "baseline_source",
        "baseline_date",
        "baseline_n",
        "baseline_value",
        "fallback_days_after_cleaning",
    }

    for name, df, req in [
        ("daily", daily, daily_required),
        ("cycles", cycles, cycle_required),
        ("baseline_candidates", cand, cand_required),
    ]:
        missing = req.difference(df.columns)
        if missing:
            raise RuntimeError(f"{name} missing columns: {sorted(missing)}")

    daily["date_ts"] = pd.to_datetime(daily["date"], errors="raise")
    cycles["cycle_report_start_ts"] = pd.to_datetime(
        cycles["cycle_report_start"], errors="raise"
    )
    cycles["cycle_report_end_ts"] = pd.to_datetime(
        cycles["cycle_report_end"], errors="raise"
    )
    cycles["cycle_start_boundary_ts"] = pd.to_datetime(
        cycles["cycle_start_boundary"], errors="raise"
    )

    return daily, cycles, cand


def validate_upstream(
    daily: pd.DataFrame,
    cycles: pd.DataFrame,
    cand: pd.DataFrame,
) -> pd.DataFrame:
    if len(daily) != EXPECTED_DAYS:
        raise RuntimeError(f"Expected {EXPECTED_DAYS} daily rows, found {len(daily)}.")
    if daily["date_ts"].duplicated().any():
        raise RuntimeError("Daily table has duplicate dates.")

    expected_dates = pd.date_range("2021-08-09", "2023-08-08", freq="D")
    if not np.array_equal(
        daily["date_ts"].sort_values().to_numpy(dtype="datetime64[ns]"),
        expected_dates.to_numpy(dtype="datetime64[ns]"),
    ):
        raise RuntimeError("Daily table does not exactly cover the 730 report days.")

    if len(cycles) != EXPECTED_CYCLES:
        raise RuntimeError(f"Expected {EXPECTED_CYCLES} cycles, found {len(cycles)}.")
    if cycles["cycle_id"].duplicated().any():
        raise RuntimeError("Cycle inventory has duplicate cycle_id values.")

    observed_cleanings = int(
        pd.to_numeric(cycles["observed_cleaning_event"], errors="coerce")
        .fillna(0)
        .astype(bool)
        .sum()
    )
    if observed_cleanings != EXPECTED_INPERIOD_CLEANINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_INPERIOD_CLEANINGS} observed ModB cleanings, "
            f"found {observed_cleanings}."
        )

    primary = cand[pd.to_numeric(cand["min_support"], errors="coerce").eq(
        PRIMARY_SUPPORT
    )].copy()

    if len(primary) != EXPECTED_CYCLES:
        raise RuntimeError(
            f"Expected {EXPECTED_CYCLES} primary baseline rows, found {len(primary)}."
        )
    if primary["cycle_id"].duplicated().any():
        raise RuntimeError("Primary baseline candidates have duplicate cycle_id.")
    if not primary["baseline_resolved"].fillna(False).astype(bool).all():
        raise RuntimeError("At least one primary baseline is unresolved.")

    if primary["baseline_value"].isna().any():
        raise RuntimeError("At least one primary baseline_value is NaN.")
    if not np.isfinite(
        pd.to_numeric(primary["baseline_value"], errors="coerce")
    ).all():
        raise RuntimeError("At least one primary baseline_value is non-finite.")
    if (pd.to_numeric(primary["baseline_value"], errors="coerce") <= 0).any():
        raise RuntimeError("At least one primary baseline_value is non-positive.")

    cycle0 = primary[primary["cycle_id"].eq(0)]
    if len(cycle0) != 1:
        raise RuntimeError("Expected exactly one cycle-0 primary baseline.")
    if cycle0.iloc[0]["baseline_source"] != "PREPERIOD_ANCHOR_FIRST_VALID_DAY":
        raise RuntimeError("Cycle 0 baseline source does not match frozen 5.5a result.")

    inperiod = primary[~primary["cycle_id"].eq(0)]
    source_counts = inperiod["baseline_source"].value_counts().to_dict()
    if source_counts != {"SAME_DAY_POST_CLEAN": EXPECTED_INPERIOD_CLEANINGS}:
        raise RuntimeError(
            "In-period primary baseline source pattern differs from frozen 5.5a result: "
            f"{source_counts}"
        )

    fallback = pd.to_numeric(
        primary["fallback_days_after_cleaning"], errors="coerce"
    ).fillna(0.0)
    if not np.allclose(fallback.to_numpy(dtype=float), 0.0, atol=0.0, rtol=0.0):
        raise RuntimeError("Primary baseline unexpectedly contains fallback days.")

    return primary.sort_values("cycle_id").reset_index(drop=True)


def assign_cycles(
    daily: pd.DataFrame,
    cycles: pd.DataFrame,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    baseline_cols = [
        "cycle_id",
        "baseline_source",
        "baseline_date",
        "baseline_n",
        "baseline_value",
    ]
    cycle_aug = cycles.merge(
        primary[baseline_cols],
        on="cycle_id",
        how="left",
        validate="one_to_one",
    )

    rows = []
    for d in daily.itertuples(index=False):
        matches = cycle_aug[
            (cycle_aug["cycle_report_start_ts"] <= d.date_ts)
            & (cycle_aug["cycle_report_end_ts"] >= d.date_ts)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Date {d.date} maps to {len(matches)} cycles; expected exactly one."
            )
        cyc = matches.iloc[0]

        is_modb_cleaning_day = (
            int(cyc["cycle_id"]) > 0
            and d.date_ts == cyc["cycle_start_boundary_ts"]
        )

        ratio_daily = float(d.ratio_corr_median) if pd.notna(
            d.ratio_corr_median
        ) else np.nan
        baseline = float(cyc["baseline_value"])

        if is_modb_cleaning_day:
            ratio_used = baseline
            obs_source = "SAME_DAY_POST_CLEAN_BASELINE"
        else:
            ratio_used = ratio_daily
            obs_source = (
                "DAILY_CORRECTED_RATIO"
                if np.isfinite(ratio_daily)
                else "MISSING_DAILY_CORRECTED_RATIO"
            )

        if np.isfinite(ratio_used):
            cleanliness = ratio_used / baseline
            soiling = 1.0 - cleanliness
        else:
            cleanliness = np.nan
            soiling = np.nan

        rows.append(
            {
                "date": d.date,
                "cycle_id": int(cyc["cycle_id"]),
                "cycle_start_boundary": str(cyc["cycle_start_boundary"]),
                "cycle_report_start": str(cyc["cycle_report_start"]),
                "cycle_report_end": str(cyc["cycle_report_end"]),
                "baseline_source": str(cyc["baseline_source"]),
                "baseline_date": str(cyc["baseline_date"]),
                "baseline_n": int(cyc["baseline_n"]),
                "B_j": baseline,
                "ratio_corr_daily_mixed_diagnostic": ratio_daily,
                "ratio_used_for_cleanliness": ratio_used,
                "cleanliness_observation_source": obs_source,
                "C_WAPP": cleanliness,
                "S_soil": soiling,
                "n_valid_daily_ratio": int(d.n_valid),
                "ratio_corr_mad": d.ratio_corr_mad,
                "ratio_corr_iqr": d.ratio_corr_iqr,
                "rain_mm_day": float(d.rain_mm_day),
                "rain_day": bool(float(d.rain_mm_day) > 0.0),
                "has_any_cleaning_pulse": bool(d.has_cleaning_pulse),
                "modb_manual_cleaning_day": bool(is_modb_cleaning_day),
                "scheduled_maintenance": bool(d.scheduled_maintenance),
                "observational_clipping_applied": False,
            }
        )

    out = pd.DataFrame(rows)

    if len(out) != EXPECTED_DAYS:
        raise RuntimeError("Reconstruction row count changed unexpectedly.")
    if out["date"].duplicated().any():
        raise RuntimeError("Reconstruction contains duplicate dates.")

    return out


def build_cycle_summary(recon: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle_id, g in recon.groupby("cycle_id", sort=True):
        c = pd.to_numeric(g["C_WAPP"], errors="coerce")
        s = pd.to_numeric(g["S_soil"], errors="coerce")
        valid = np.isfinite(c)

        rows.append(
            {
                "cycle_id": int(cycle_id),
                "cycle_start_boundary": g["cycle_start_boundary"].iloc[0],
                "B_j": float(g["B_j"].iloc[0]),
                "baseline_source": g["baseline_source"].iloc[0],
                "baseline_date": g["baseline_date"].iloc[0],
                "days_total": int(len(g)),
                "days_valid": int(valid.sum()),
                "rain_days": int(g["rain_day"].sum()),
                "modb_cleaning_days": int(g["modb_manual_cleaning_day"].sum()),
                "C_min": float(c[valid].min()) if valid.any() else np.nan,
                "C_median": float(c[valid].median()) if valid.any() else np.nan,
                "C_max": float(c[valid].max()) if valid.any() else np.nan,
                "S_min": float(s[valid].min()) if valid.any() else np.nan,
                "S_median": float(s[valid].median()) if valid.any() else np.nan,
                "S_max": float(s[valid].max()) if valid.any() else np.nan,
                "days_C_gt_1": int((c[valid] > 1.0).sum()) if valid.any() else 0,
                "days_S_lt_0": int((s[valid] < 0.0).sum()) if valid.any() else 0,
            }
        )
    return pd.DataFrame(rows)


def make_summary(
    daily_path: Path,
    cycles_path: Path,
    candidates_path: Path,
    recon: pd.DataFrame,
    cycle_summary: pd.DataFrame,
) -> dict:
    valid = np.isfinite(pd.to_numeric(recon["C_WAPP"], errors="coerce"))
    c = pd.to_numeric(recon.loc[valid, "C_WAPP"], errors="coerce")
    s = pd.to_numeric(recon.loc[valid, "S_soil"], errors="coerce")

    coverage = float(valid.mean())
    if coverage < MIN_REQUIRED_COVERAGE:
        raise RuntimeError(
            f"Observed cleanliness coverage {coverage:.6f} < "
            f"frozen gate {MIN_REQUIRED_COVERAGE:.2f}."
        )

    clean_days = recon[recon["modb_manual_cleaning_day"]].copy()
    if len(clean_days) != EXPECTED_INPERIOD_CLEANINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_INPERIOD_CLEANINGS} ModB cleaning days, "
            f"found {len(clean_days)}."
        )
    if not np.allclose(
        pd.to_numeric(clean_days["C_WAPP"], errors="coerce").to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("A ModB cleaning day is not exactly C_WAPP=1 by baseline definition.")

    # Mixed-day diagnostic: shows why the all-day daily ratio was not used.
    mixed = pd.to_numeric(
        clean_days["ratio_corr_daily_mixed_diagnostic"], errors="coerce"
    )
    base = pd.to_numeric(clean_days["B_j"], errors="coerce")
    mixed_rel_diff = (mixed / base) - 1.0

    rain_days = recon[recon["rain_day"]]
    maintenance_days = recon[recon["scheduled_maintenance"]]

    return {
        "stage": "P2-0B-5.5b",
        "cleanliness_reconstruction_generated": True,
        "cleanliness_frozen": False,
        "soiling_state_generated": True,
        "soiling_state_frozen": False,
        "power_loss_bridge_generated": False,
        "rl_state_generated": False,
        "input": {
            "daily_corrected_ratio_audit": str(daily_path),
            "cycle_inventory": str(cycles_path),
            "baseline_candidates": str(candidates_path),
        },
        "frozen_5_5a_rule_used": {
            "primary_min_support": PRIMARY_SUPPORT,
            "cycles": EXPECTED_CYCLES,
            "preperiod_anchor_cycles": 1,
            "observed_modb_cleaning_cycles": EXPECTED_INPERIOD_CLEANINGS,
            "inperiod_baseline_source": "SAME_DAY_POST_CLEAN",
            "rain_creates_cycle": False,
        },
        "reconstruction": {
            "days": int(len(recon)),
            "valid_days": int(valid.sum()),
            "coverage": coverage,
            "missing_days": int((~valid).sum()),
            "C_WAPP_distribution": finite_quantiles(c),
            "S_soil_distribution": finite_quantiles(s),
            "C_gt_1_days": int((c > 1.0).sum()),
            "C_gt_1_fraction_valid": float((c > 1.0).mean()),
            "S_lt_0_days": int((s < 0.0).sum()),
            "S_lt_0_fraction_valid": float((s < 0.0).mean()),
            "C_le_0_days": int((c <= 0.0).sum()),
            "observational_clipping_applied": False,
        },
        "modb_cleaning_days": {
            "days": int(len(clean_days)),
            "state_source_counts": {
                str(k): int(v)
                for k, v in clean_days[
                    "cleanliness_observation_source"
                ].value_counts().to_dict().items()
            },
            "C_equals_1_by_definition": True,
            "mixed_daily_ratio_relative_to_baseline_distribution": finite_quantiles(
                mixed_rel_diff
            ),
            "note": (
                "The all-day corrected ratio on a ModB cleaning date can mix pre- "
                "and post-clean states; it is retained only as a diagnostic."
            ),
        },
        "rain": {
            "rain_days": int(len(rain_days)),
            "rain_days_with_valid_cleanliness": int(
                np.isfinite(
                    pd.to_numeric(rain_days["C_WAPP"], errors="coerce")
                ).sum()
            ),
            "rain_resets_baseline": False,
        },
        "scheduled_maintenance": {
            "days": int(len(maintenance_days)),
            "dates": maintenance_days["date"].astype(str).tolist(),
            "used_to_create_cycle": False,
        },
        "cycle_summary": {
            "cycles": int(len(cycle_summary)),
            "cycles_with_any_C_gt_1": int(
                (cycle_summary["days_C_gt_1"] > 0).sum()
            ),
            "B_j_distribution": finite_quantiles(cycle_summary["B_j"]),
            "cycle_S_max_distribution": finite_quantiles(
                cycle_summary["S_max"]
            ),
        },
        "gates": {
            "coverage_gate": MIN_REQUIRED_COVERAGE,
            "coverage_pass": bool(coverage >= MIN_REQUIRED_COVERAGE),
            "all_27_baselines_resolved": True,
            "all_26_inperiod_cleaning_days_postclean_only": True,
            "rain_reset_absent": True,
            "observational_clipping_absent": True,
        },
        "notes": [
            "This is an observational WAPP cleanliness/soiling reconstruction, not yet a power-loss state.",
            "C_WAPP > 1 and S_soil < 0 are retained and quantified rather than clipped.",
            "The 26 in-period ModB manual-cleaning dates use their post-clean baseline observation, not the mixed all-day ratio.",
            "The first report-period cycle is anchored by the 2021-08-08 pre-period provenance and its first valid observed day.",
            "Passing this stage is necessary but not sufficient: official monthly-rate / sawtooth / rain-response validation is deferred to P2-0B-5.5c.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P2-0B-5.5b daily WAPP cleanliness reconstruction."
    )
    p.add_argument("--daily-ratio", required=True, type=Path)
    p.add_argument("--cycle-inventory", required=True, type=Path)
    p.add_argument("--baseline-candidates", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0b_5_5b_daily_cleanliness_reconstruction_v1"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    daily_path = args.daily_ratio.expanduser().resolve()
    cycles_path = args.cycle_inventory.expanduser().resolve()
    candidates_path = args.baseline_candidates.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for p in [daily_path, cycles_path, candidates_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    print("[1/6] Read P2-0B-5.5a outputs")
    daily, cycles, cand = load_inputs(
        daily_path,
        cycles_path,
        candidates_path,
    )

    print("[2/6] Validate frozen 5.5a primary baseline result")
    primary = validate_upstream(daily, cycles, cand)

    print("[3/6] Assign all 730 report days to 27 ModB cycles")
    recon = assign_cycles(daily, cycles, primary)

    print("[4/6] Construct C_WAPP and S_soil without clipping")
    # Construction occurs in assign_cycles; this explicit stage marker is
    # retained so the run log mirrors the scientific protocol.

    print("[5/6] Build cycle-level audit + scientific gates")
    cycle_summary = build_cycle_summary(recon)
    summary = make_summary(
        daily_path=daily_path,
        cycles_path=cycles_path,
        candidates_path=candidates_path,
        recon=recon,
        cycle_summary=cycle_summary,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    recon.to_csv(
        out_dir / "daily_cleanliness_reconstruction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cycle_summary.to_csv(
        out_dir / "cycle_reconstruction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[6/6] Done")
    print(out_dir / "daily_cleanliness_reconstruction.csv")
    print(out_dir / "cycle_reconstruction_audit.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: C_WAPP/S_soil are reconstructed but NOT frozen; "
        "P2-0B-5.5c external/report validation is still required."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
