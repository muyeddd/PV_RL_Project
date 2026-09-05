#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0B-5.5c
WAPP Malanville report-level validation of reconstructed daily cleanliness.

Purpose
-------
Validate the P2-0B-5.5b observational reconstruction against the independent
Malanville WAPP final report before freezing the site-level soiling state.

This script does NOT tune the reconstruction. It evaluates a predeclared,
transparent analogue of the report's rate calculation:

1) authoritative ModB manual cleaning days and observed rain days divide the
   time series into dry accumulation runs;
2) missing days also break a run;
3) for each dry run with >=3 valid days, the primary soiling-rate estimate is
       100 * (S_last - S_first) / elapsed_days   [%/day]
   with negative values set to 0 because the report's "soiling rate" is an
   accumulation rate and reported monthly values are non-negative;
4) the run rate is assigned to all days in that dry run, while rain, ModB
   cleaning, missing, and too-short runs receive 0 for monthly aggregation;
5) monthly rates are calendar-day means of the assigned daily rates.

Why this is only an analogue
----------------------------
The WAPP report states that ModB cleanings start new intervals and that rain,
strong wind/gust, or high diffuse irradiance may split intervals using analyst
thresholds. Those numeric wind/DHI thresholds are not published in the report.
Therefore exact reproduction of the report's proprietary/manual segmentation
is not claimed. The primary validation uses only objectively observed ModB
cleanings + precipitation, with no post-hoc threshold tuning.

Precommitted scientific gates
-----------------------------
- monthly Spearman rank agreement >= 0.80
- monthly rate MAE <= 0.10 %/day
- overall rate within +/-0.05 %/day of official 0.21 %/day
- no rain reset is introduced
- no clipping of observational C_WAPP / S_soil

Additional diagnostics
----------------------
- OLS dry-run slope sensitivity
- rain-event partial-recovery statistics
- manual-cleaning sawtooth/reset statistics

Inputs
------
P2-0B-5.5b daily_cleanliness_reconstruction.csv

Outputs
-------
monthly_rate_validation.csv
dry_run_audit.csv
rain_response_audit.csv
cleaning_reset_audit.csv
audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MONTHLY_SPEARMAN_GATE = 0.80
MONTHLY_MAE_GATE_PCT_PER_DAY = 0.10
OVERALL_RATE_TARGET_PCT_PER_DAY = 0.21
OVERALL_RATE_TOL_PCT_PER_DAY = 0.05
MIN_DRY_RUN_DAYS = 3

OFFICIAL_MONTHLY_RATE_PCT_PER_DAY = {
    "2021-08": 0.00,
    "2021-09": 0.00,
    "2021-10": 0.07,
    "2021-11": 0.26,
    "2021-12": 0.41,
    "2022-01": 0.25,
    "2022-02": 0.44,
    "2022-03": 0.59,
    "2022-04": 0.26,
    "2022-05": 0.20,
    "2022-06": 0.06,
    "2022-07": 0.00,
    "2022-08": 0.00,
    "2022-09": 0.00,
    "2022-10": 0.00,
    "2022-11": 0.18,
    "2022-12": 0.37,
    "2023-01": 0.36,
    "2023-02": 0.41,
    "2023-03": 0.49,
    "2023-04": 0.18,
    "2023-05": 0.23,
    "2023-06": 0.25,
    "2023-07": 0.10,
    "2023-08": 0.03,
}


def finite_quantiles(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
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


def load_reconstruction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date",
        "cycle_id",
        "B_j",
        "C_WAPP",
        "S_soil",
        "rain_mm_day",
        "rain_day",
        "modb_manual_cleaning_day",
        "observational_clipping_applied",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing reconstruction columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) != 730:
        raise RuntimeError(f"Expected 730 report days, found {len(df)}.")
    if df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in reconstruction.")

    expected = pd.date_range("2021-08-09", "2023-08-08", freq="D")
    if not np.array_equal(
        df["date"].to_numpy(dtype="datetime64[ns]"),
        expected.to_numpy(dtype="datetime64[ns]"),
    ):
        raise RuntimeError("Reconstruction does not exactly cover report period.")

    clipping = df["observational_clipping_applied"].fillna(False).astype(bool)
    if clipping.any():
        raise RuntimeError("Observational clipping was applied upstream; frozen protocol violated.")

    c = pd.to_numeric(df["C_WAPP"], errors="coerce")
    s = pd.to_numeric(df["S_soil"], errors="coerce")
    both = np.isfinite(c) & np.isfinite(s)
    if not np.allclose(
        (1.0 - c[both]).to_numpy(dtype=float),
        s[both].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("S_soil != 1 - C_WAPP for finite observations.")

    return df


def build_dry_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Primary deterministic segmentation.

    A dry accumulation day must be:
    - finite S_soil;
    - not a rain day;
    - not a ModB manual-cleaning day.

    Rain/cleaning/missing days break runs. Each qualifying calendar-consecutive
    run is assigned one primary endpoint rate and one OLS sensitivity rate.
    """
    s = pd.to_numeric(df["S_soil"], errors="coerce").to_numpy(dtype=float)
    rain = df["rain_day"].fillna(False).astype(bool).to_numpy()
    clean = df["modb_manual_cleaning_day"].fillna(False).astype(bool).to_numpy()

    eligible = np.isfinite(s) & ~rain & ~clean

    assigned_endpoint = np.zeros(len(df), dtype=float)
    assigned_ols = np.zeros(len(df), dtype=float)
    run_id_by_day = np.full(len(df), -1, dtype=int)

    rows = []
    run_id = 0
    i = 0

    while i < len(df):
        if not eligible[i]:
            i += 1
            continue

        j = i
        while (
            j + 1 < len(df)
            and eligible[j + 1]
            and (df.loc[j + 1, "date"] - df.loc[j, "date"]) == pd.Timedelta(days=1)
        ):
            j += 1

        idx = np.arange(i, j + 1)
        dates = df.loc[idx, "date"]
        y = s[idx]
        n = len(idx)
        elapsed = int((dates.iloc[-1] - dates.iloc[0]).days)

        if n >= MIN_DRY_RUN_DAYS and elapsed > 0:
            endpoint_raw = 100.0 * (y[-1] - y[0]) / float(elapsed)
            endpoint_rate = max(0.0, float(endpoint_raw))

            x = (dates - dates.iloc[0]).dt.days.to_numpy(dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
            ols_raw = 100.0 * slope
            ols_rate = max(0.0, ols_raw)

            assigned_endpoint[idx] = endpoint_rate
            assigned_ols[idx] = ols_rate
            run_id_by_day[idx] = run_id
        else:
            endpoint_raw = np.nan
            endpoint_rate = 0.0
            ols_raw = np.nan
            ols_rate = 0.0
            run_id_by_day[idx] = run_id

        rows.append(
            {
                "run_id": run_id,
                "start_date": str(dates.iloc[0].date()),
                "end_date": str(dates.iloc[-1].date()),
                "n_days": int(n),
                "elapsed_days": int(elapsed),
                "S_start": float(y[0]),
                "S_end": float(y[-1]),
                "endpoint_rate_raw_pct_per_day": endpoint_raw,
                "endpoint_rate_primary_pct_per_day": endpoint_rate,
                "ols_rate_raw_pct_per_day": ols_raw,
                "ols_rate_sensitivity_pct_per_day": ols_rate,
                "primary_rate_truncated_at_zero": bool(
                    np.isfinite(endpoint_raw) and endpoint_raw < 0.0
                ),
                "meets_min_run_days": bool(n >= MIN_DRY_RUN_DAYS),
            }
        )

        run_id += 1
        i = j + 1

    day_rates = df[["date", "cycle_id", "rain_day", "modb_manual_cleaning_day"]].copy()
    day_rates["S_soil"] = s
    day_rates["dry_run_id"] = run_id_by_day
    day_rates["rate_primary_pct_per_day"] = assigned_endpoint
    day_rates["rate_ols_sensitivity_pct_per_day"] = assigned_ols
    day_rates["rate_assignment_zero_due_event_or_missing"] = ~eligible

    return pd.DataFrame(rows), day_rates


def monthly_validation(day_rates: pd.DataFrame) -> pd.DataFrame:
    work = day_rates.copy()
    work["month"] = work["date"].dt.strftime("%Y-%m")

    monthly = (
        work.groupby("month", as_index=False)
        .agg(
            reconstructed_rate_pct_per_day=("rate_primary_pct_per_day", "mean"),
            reconstructed_rate_ols_sensitivity_pct_per_day=(
                "rate_ols_sensitivity_pct_per_day",
                "mean",
            ),
            calendar_days=("date", "size"),
            rain_days=("rain_day", "sum"),
            cleaning_days=("modb_manual_cleaning_day", "sum"),
        )
    )

    official = pd.DataFrame(
        {
            "month": list(OFFICIAL_MONTHLY_RATE_PCT_PER_DAY.keys()),
            "official_rate_pct_per_day": list(
                OFFICIAL_MONTHLY_RATE_PCT_PER_DAY.values()
            ),
        }
    )

    out = official.merge(monthly, on="month", how="left", validate="one_to_one")
    if out["reconstructed_rate_pct_per_day"].isna().any():
        raise RuntimeError("Missing reconstructed monthly rate.")

    out["error_pct_per_day"] = (
        out["reconstructed_rate_pct_per_day"]
        - out["official_rate_pct_per_day"]
    )
    out["abs_error_pct_per_day"] = out["error_pct_per_day"].abs()
    out["error_ols_sensitivity_pct_per_day"] = (
        out["reconstructed_rate_ols_sensitivity_pct_per_day"]
        - out["official_rate_pct_per_day"]
    )
    out["abs_error_ols_sensitivity_pct_per_day"] = (
        out["error_ols_sensitivity_pct_per_day"].abs()
    )
    return out


def rain_response_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group consecutive rain days into rain events and compare nearest finite
    S_soil before/after, excluding events crossed by a ModB manual cleaning.
    """
    rain = df["rain_day"].fillna(False).astype(bool).to_numpy()
    s = pd.to_numeric(df["S_soil"], errors="coerce").to_numpy(dtype=float)
    clean = df["modb_manual_cleaning_day"].fillna(False).astype(bool).to_numpy()
    rain_mm = pd.to_numeric(df["rain_mm_day"], errors="coerce").fillna(0.0).to_numpy()

    rows = []
    event_id = 0
    i = 0
    while i < len(df):
        if not rain[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(df) and rain[j + 1]:
            j += 1

        # nearest finite pre/post within 3 calendar days
        pre_idx = None
        for k in range(i - 1, max(-1, i - 4), -1):
            if k >= 0 and np.isfinite(s[k]):
                pre_idx = k
                break

        post_idx = None
        for k in range(j + 1, min(len(df), j + 4)):
            if np.isfinite(s[k]):
                post_idx = k
                break

        cleaning_confounded = bool(
            clean[max(0, i - 1):min(len(df), j + 2)].any()
        )

        if pre_idx is not None and post_idx is not None and not cleaning_confounded:
            delta_s = float(s[post_idx] - s[pre_idx])
            resolved = True
        else:
            delta_s = np.nan
            resolved = False

        total_rain = float(rain_mm[i:j + 1].sum())
        if total_rain < 1.0:
            rain_bin = "<1mm"
        elif total_rain < 5.0:
            rain_bin = "1-5mm"
        else:
            rain_bin = ">=5mm"

        rows.append(
            {
                "rain_event_id": event_id,
                "start_date": str(df.loc[i, "date"].date()),
                "end_date": str(df.loc[j, "date"].date()),
                "rain_days": int(j - i + 1),
                "rain_mm_total": total_rain,
                "rain_bin": rain_bin,
                "pre_date": (
                    str(df.loc[pre_idx, "date"].date())
                    if pre_idx is not None
                    else None
                ),
                "post_date": (
                    str(df.loc[post_idx, "date"].date())
                    if post_idx is not None
                    else None
                ),
                "S_pre": float(s[pre_idx]) if pre_idx is not None else np.nan,
                "S_post": float(s[post_idx]) if post_idx is not None else np.nan,
                "delta_S_post_minus_pre": delta_s,
                "soiling_decreased_after_rain": bool(
                    resolved and delta_s < 0.0
                ),
                "resolved": resolved,
                "cleaning_confounded": cleaning_confounded,
            }
        )
        event_id += 1
        i = j + 1

    return pd.DataFrame(rows)


def cleaning_reset_audit(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_numeric(df["S_soil"], errors="coerce").to_numpy(dtype=float)
    clean_idx = np.flatnonzero(
        df["modb_manual_cleaning_day"].fillna(False).astype(bool).to_numpy()
    )

    rows = []
    for idx in clean_idx:
        # nearest finite prior day within 3 days
        pre_idx = None
        for k in range(idx - 1, max(-1, idx - 4), -1):
            if k >= 0 and np.isfinite(s[k]):
                pre_idx = k
                break

        rows.append(
            {
                "cleaning_date": str(df.loc[idx, "date"].date()),
                "pre_date": (
                    str(df.loc[pre_idx, "date"].date())
                    if pre_idx is not None
                    else None
                ),
                "S_pre": float(s[pre_idx]) if pre_idx is not None else np.nan,
                "S_cleaning_day": float(s[idx]) if np.isfinite(s[idx]) else np.nan,
                "reset_magnitude": (
                    float(s[pre_idx] - s[idx])
                    if pre_idx is not None and np.isfinite(s[idx])
                    else np.nan
                ),
                "preclean_positive_soiling": bool(
                    pre_idx is not None and s[pre_idx] > 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_rain(rain_audit: pd.DataFrame) -> dict:
    resolved = rain_audit[rain_audit["resolved"]].copy()
    by_bin = {}
    for b, g in resolved.groupby("rain_bin"):
        delta = pd.to_numeric(g["delta_S_post_minus_pre"], errors="coerce")
        by_bin[str(b)] = {
            "events": int(len(g)),
            "delta_S_distribution": finite_quantiles(delta),
            "fraction_soiling_decreased": float((delta < 0.0).mean())
            if len(g)
            else np.nan,
        }
    return {
        "events_total": int(len(rain_audit)),
        "events_resolved_unconfounded": int(len(resolved)),
        "delta_S_distribution": finite_quantiles(
            resolved["delta_S_post_minus_pre"]
        ),
        "fraction_soiling_decreased": float(
            pd.to_numeric(
                resolved["delta_S_post_minus_pre"], errors="coerce"
            ).lt(0.0).mean()
        )
        if len(resolved)
        else np.nan,
        "by_rain_bin": by_bin,
        "note": (
            "Rain-response statistics are physical diagnostics, not numeric "
            "pass/fail gates because the WAPP report does not publish a "
            "universal rain-cleaning threshold."
        ),
    }


def make_summary(
    recon_path: Path,
    recon: pd.DataFrame,
    dry_runs: pd.DataFrame,
    day_rates: pd.DataFrame,
    monthly: pd.DataFrame,
    rain_audit: pd.DataFrame,
    cleaning_audit: pd.DataFrame,
) -> dict:
    official = monthly["official_rate_pct_per_day"]
    est = monthly["reconstructed_rate_pct_per_day"]
    est_ols = monthly["reconstructed_rate_ols_sensitivity_pct_per_day"]

    spearman = float(official.corr(est, method="spearman"))
    mae = float((est - official).abs().mean())

    spearman_ols = float(official.corr(est_ols, method="spearman"))
    mae_ols = float((est_ols - official).abs().mean())

    overall = float(day_rates["rate_primary_pct_per_day"].mean())
    overall_ols = float(
        day_rates["rate_ols_sensitivity_pct_per_day"].mean()
    )
    overall_error = overall - OVERALL_RATE_TARGET_PCT_PER_DAY

    gate_rank = bool(spearman >= MONTHLY_SPEARMAN_GATE)
    gate_mae = bool(mae <= MONTHLY_MAE_GATE_PCT_PER_DAY)
    gate_overall = bool(
        abs(overall_error) <= OVERALL_RATE_TOL_PCT_PER_DAY
    )

    eligible_runs = dry_runs[dry_runs["meets_min_run_days"]].copy()
    positive_raw_fraction = float(
        (
            pd.to_numeric(
                eligible_runs["endpoint_rate_raw_pct_per_day"],
                errors="coerce",
            )
            > 0.0
        ).mean()
    ) if len(eligible_runs) else np.nan

    clean_valid = cleaning_audit[
        np.isfinite(
            pd.to_numeric(cleaning_audit["S_pre"], errors="coerce")
        )
    ].copy()

    top_official = monthly.sort_values(
        "official_rate_pct_per_day", ascending=False
    ).head(5)["month"].tolist()
    top_recon = monthly.sort_values(
        "reconstructed_rate_pct_per_day", ascending=False
    ).head(5)["month"].tolist()

    return {
        "stage": "P2-0B-5.5c",
        "validation_only": True,
        "cleanliness_frozen": False,
        "soiling_state_frozen": False,
        "power_loss_bridge_generated": False,
        "rl_state_generated": False,
        "input": {
            "daily_cleanliness_reconstruction": str(recon_path),
            "days": int(len(recon)),
            "valid_soiling_days": int(
                np.isfinite(
                    pd.to_numeric(recon["S_soil"], errors="coerce")
                ).sum()
            ),
        },
        "official_reference": {
            "source": "WAPP Malanville final two-year measurement report, Table 11",
            "months": int(len(OFFICIAL_MONTHLY_RATE_PCT_PER_DAY)),
            "overall_rate_pct_per_day": OVERALL_RATE_TARGET_PCT_PER_DAY,
            "monthly_rates_pct_per_day": OFFICIAL_MONTHLY_RATE_PCT_PER_DAY,
            "methodology_note": (
                "Report splits near-constant-soiling intervals at ModB "
                "cleanings and may additionally use rain, wind/gust and high "
                "diffuse irradiance events. Numeric wind/DHI thresholds are "
                "not published, so exact proprietary/manual segmentation is "
                "not claimed here."
            ),
        },
        "primary_validation_estimator": {
            "min_dry_run_days": MIN_DRY_RUN_DAYS,
            "boundaries": [
                "ModB manual cleaning day",
                "observed precipitation day",
                "missing observation",
            ],
            "run_rate": (
                "max(0, 100*(S_last-S_first)/elapsed_days) [%/day]"
            ),
            "calendar_event_days_assigned_rate": 0.0,
            "dry_runs_total": int(len(dry_runs)),
            "dry_runs_meeting_min_length": int(
                dry_runs["meets_min_run_days"].sum()
            ),
            "fraction_eligible_runs_positive_raw_slope": positive_raw_fraction,
            "rate_distribution_eligible_runs": finite_quantiles(
                eligible_runs["endpoint_rate_primary_pct_per_day"]
            ),
        },
        "monthly_validation": {
            "spearman": spearman,
            "spearman_gate": MONTHLY_SPEARMAN_GATE,
            "spearman_pass": gate_rank,
            "mae_pct_per_day": mae,
            "mae_gate_pct_per_day": MONTHLY_MAE_GATE_PCT_PER_DAY,
            "mae_pass": gate_mae,
            "overall_reconstructed_pct_per_day": overall,
            "overall_official_pct_per_day": OVERALL_RATE_TARGET_PCT_PER_DAY,
            "overall_error_pct_per_day": overall_error,
            "overall_tolerance_pct_per_day": OVERALL_RATE_TOL_PCT_PER_DAY,
            "overall_pass": gate_overall,
            "top5_official_months": top_official,
            "top5_reconstructed_months": top_recon,
        },
        "ols_sensitivity": {
            "monthly_spearman": spearman_ols,
            "monthly_mae_pct_per_day": mae_ols,
            "overall_rate_pct_per_day": overall_ols,
        },
        "rain_response": summarize_rain(rain_audit),
        "manual_cleaning_sawtooth": {
            "cleaning_events": int(len(cleaning_audit)),
            "events_with_valid_preclean_state": int(len(clean_valid)),
            "preclean_S_distribution": finite_quantiles(
                clean_valid["S_pre"]
            ),
            "reset_magnitude_distribution": finite_quantiles(
                clean_valid["reset_magnitude"]
            ),
            "fraction_preclean_positive_soiling": float(
                clean_valid["preclean_positive_soiling"].mean()
            )
            if len(clean_valid)
            else np.nan,
            "cleaning_day_S_is_zero_by_cycle_baseline_definition": True,
        },
        "gates": {
            "monthly_spearman_pass": gate_rank,
            "monthly_mae_pass": gate_mae,
            "overall_rate_pass": gate_overall,
            "all_precommitted_quantitative_gates_pass": bool(
                gate_rank and gate_mae and gate_overall
            ),
            "rain_reset_introduced": False,
            "observational_clipping_introduced": False,
        },
        "notes": [
            "Validation targets are not used to retune B_j or the 5.5b reconstruction.",
            "Exact reproduction of WAPP's analyst-selected wind/DHI sub-interval thresholds is not claimed because numeric thresholds are not published.",
            "Rain diagnostics assess partial natural recovery without forcing cleanliness to 1.",
            "If a precommitted quantitative gate fails, stop and diagnose methodology/semantic mismatch rather than tuning to the report target.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P2-0B-5.5c report-level validation of WAPP cleanliness."
    )
    p.add_argument(
        "--daily-cleanliness",
        required=True,
        type=Path,
        help="P2-0B-5.5b daily_cleanliness_reconstruction.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0b_5_5c_report_validation_v1"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    recon_path = args.daily_cleanliness.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    if not recon_path.exists():
        raise FileNotFoundError(recon_path)

    print("[1/7] Read + validate frozen P2-0B-5.5b reconstruction")
    recon = load_reconstruction(recon_path)

    print("[2/7] Build deterministic dry accumulation runs")
    dry_runs, day_rates = build_dry_runs(recon)

    print("[3/7] Compare reconstructed monthly rates with WAPP Table 11")
    monthly = monthly_validation(day_rates)

    print("[4/7] Audit rain-event partial recovery")
    rain_audit = rain_response_audit(recon)

    print("[5/7] Audit ModB manual-cleaning sawtooth resets")
    cleaning_audit = cleaning_reset_audit(recon)

    print("[6/7] Evaluate precommitted scientific gates")
    summary = make_summary(
        recon_path=recon_path,
        recon=recon,
        dry_runs=dry_runs,
        day_rates=day_rates,
        monthly=monthly,
        rain_audit=rain_audit,
        cleaning_audit=cleaning_audit,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(
        out_dir / "monthly_rate_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dry_runs.to_csv(
        out_dir / "dry_run_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rain_audit.to_csv(
        out_dir / "rain_response_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cleaning_audit.to_csv(
        out_dir / "cleaning_reset_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[7/7] Done")
    print(out_dir / "monthly_rate_validation.csv")
    print(out_dir / "dry_run_audit.csv")
    print(out_dir / "rain_response_audit.csv")
    print(out_dir / "cleaning_reset_audit.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: This is validation only. Do not freeze the WAPP soiling "
        "state unless the scientific gates are reviewed and passed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
