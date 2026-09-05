#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0B-5.4c-2
WAPP Malanville ModA correction implementation + event-level validation.

This stage applies ONLY the already-audited ModA reference-module correction.
It does NOT construct final WAPP cleanliness, S_soil, power loss, or RL states.

Primary correction convention
-----------------------------
The cleaning-jump estimator from P2-0B-5.4a is

    s_j = 1 - Q_pre / Q_post,  Q = ModA / ModB.

Interpreting s_j as fractional attenuation of ModA immediately before cleaning,

    ModA_raw(t) = ModA_clean(t) * (1 - s_A(t)),

so the physically consistent inverse correction is

    ModA_corr(t) = ModA_raw(t) / (1 - s_A(t)).

For each P2-0B-5.4c-1 interval with
status == ELIGIBLE_FOR_LINEAR_CORRECTION,

    s_A(t) = s_j * (t - t_prev) / (t_clean - t_prev)

on the OPEN interval after the previous visit's last Cleaning pulse and before
the current strong event's first Cleaning pulse.

Outside those intervals, ModA_corr == ModA_raw exactly.

A secondary approximation, ModA_raw * (1 + s_A(t)), is evaluated only as a
formula-sensitivity benchmark and is NOT exported as the primary correction.

Frozen upstream provenance
--------------------------
- nearest-one placebo null P95 threshold:
  0.009297729719041414
- expected strong events: 66
- c-1 expected statuses:
  64 ELIGIBLE_FOR_LINEAR_CORRECTION
  2  SKIP_RAIN_NONLINEARITY

Outputs
-------
1) moda_corrected_segments.csv
   Only timestamps actually modified by the primary correction.
2) event_jump_validation.csv
   Raw vs corrected pre/post Q jump for every eligible event, plus the
   multiply-(1+s) sensitivity result.
3) applied_intervals.csv
   Compact applied-interval provenance.
4) audit_summary.json

The original WAPP CSV files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "Timestamp",
    "GHI",
    "DNI",
    "DHI",
    "ModA",
    "ModB",
    "Tamb",
    "RH",
    "WS",
    "WSgust",
    "WSstdev",
    "WD",
    "WDstdev",
    "BP",
    "Cleaning",
    "Precipitation",
    "TModA",
    "TModB",
    "Comments",
]

EXPECTED_YEAR1_SHA256 = (
    "7f15922f01de97eb6a8b1477f0357e1dd3460c2918a64f7f007622a08063bed3"
)
EXPECTED_YEAR2_SHA256 = (
    "d85310c0a722184502714845abec945f64c854529b2470c57f3303447eb4fc52"
)

FROZEN_THRESHOLD = 0.009297729719041414
EXPECTED_STRONG_EVENTS = 66
EXPECTED_ELIGIBLE_INTERVALS = 64
EXPECTED_RAIN_SKIPS = 2

PRIMARY_IRRADIANCE_THRESHOLD_WM2 = 100.0
PRIMARY_WINDOW_MIN = 30
EVENT_BUFFER_MIN = 15

ELIGIBLE_STATUS = "ELIGIBLE_FOR_LINEAR_CORRECTION"
RAIN_SKIP_STATUS = "SKIP_RAIN_NONLINEARITY"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wapp_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="cp1252",
        skiprows=[1],
        low_memory=False,
    )
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected schema in {path}\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Found:    {list(df.columns)}"
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")
    numeric_cols = [c for c in EXPECTED_COLUMNS if c not in ("Timestamp", "Comments")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def validate_combined(df: pd.DataFrame) -> None:
    if df["Timestamp"].duplicated().any():
        raise RuntimeError("Combined data contains duplicate timestamps.")
    diffs = df["Timestamp"].diff().dropna()
    bad = diffs != pd.Timedelta(minutes=1)
    if bad.any():
        raise RuntimeError(
            f"Combined data is not strictly 1-minute continuous; "
            f"non-1min steps={int(bad.sum())}"
        )


def finite_quantiles(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return {}
    probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q = x.quantile(probs)
    out = {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
    }
    for p, v in q.items():
        out[f"q{int(round(100*p)):02d}"] = float(v)
    out["max"] = float(x.max())
    return out


def robust_window_q(
    indexed: pd.DataFrame,
    moda_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    w = indexed.loc[start:end].copy()
    if w.empty:
        return {"n": 0, "q_median": np.nan}

    moda_raw = pd.to_numeric(w["ModA"], errors="coerce").to_numpy(dtype=float)
    moda_used = pd.to_numeric(w[moda_column], errors="coerce").to_numpy(dtype=float)
    modb = pd.to_numeric(w["ModB"], errors="coerce").to_numpy(dtype=float)
    rain = (
        pd.to_numeric(w["Precipitation"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # Keep exactly the same row-support logic as the frozen 5.4a primary event
    # audit: eligibility threshold is evaluated on ORIGINAL ModA.
    valid = (
        np.isfinite(moda_raw)
        & np.isfinite(moda_used)
        & np.isfinite(modb)
        & (moda_raw > PRIMARY_IRRADIANCE_THRESHOLD_WM2)
        & (modb > 0.0)
        & (rain <= 0.0)
    )

    if valid.sum() == 0:
        return {"n": 0, "q_median": np.nan}

    q = moda_used[valid] / modb[valid]
    return {"n": int(valid.sum()), "q_median": float(np.median(q))}


def validate_interval_input(audit: pd.DataFrame) -> pd.DataFrame:
    required = {
        "current_event_id",
        "current_date",
        "current_first_pulse",
        "current_last_pulse",
        "current_gain",
        "correction_start",
        "correction_end",
        "status",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise RuntimeError(f"Interval audit missing columns: {sorted(missing)}")

    if len(audit) != EXPECTED_STRONG_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_STRONG_EVENTS} strong intervals, found {len(audit)}"
        )

    status_counts = audit["status"].value_counts().to_dict()
    if int(status_counts.get(ELIGIBLE_STATUS, 0)) != EXPECTED_ELIGIBLE_INTERVALS:
        raise RuntimeError(
            "Eligible-interval count changed from frozen c-1 result: "
            f"{status_counts}"
        )
    if int(status_counts.get(RAIN_SKIP_STATUS, 0)) != EXPECTED_RAIN_SKIPS:
        raise RuntimeError(
            "Rain-skip count changed from frozen c-1 result: "
            f"{status_counts}"
        )

    audit = audit.copy()
    for c in [
        "current_first_pulse",
        "current_last_pulse",
        "correction_start",
        "correction_end",
    ]:
        audit[c] = pd.to_datetime(audit[c], errors="raise")
    audit["current_gain"] = pd.to_numeric(
        audit["current_gain"], errors="raise"
    )

    eligible = (
        audit[audit["status"].eq(ELIGIBLE_STATUS)]
        .sort_values("correction_start")
        .reset_index(drop=True)
    )

    starts = eligible["correction_start"].to_numpy(dtype="datetime64[ns]")
    ends = eligible["correction_end"].to_numpy(dtype="datetime64[ns]")
    if len(eligible) > 1 and np.any(starts[1:] < ends[:-1]):
        raise RuntimeError("Eligible correction intervals overlap.")

    if not np.all(eligible["current_gain"].to_numpy() > FROZEN_THRESHOLD):
        raise RuntimeError("An eligible interval does not exceed frozen P95 threshold.")

    if np.any(eligible["current_gain"].to_numpy() >= 1.0):
        raise RuntimeError("Invalid current_gain >= 1 encountered.")

    return audit


def apply_primary_and_sensitivity(
    combined: pd.DataFrame,
    interval_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = combined.copy()

    raw = df["ModA"].to_numpy(dtype=float)
    corr = raw.copy()
    corr_mul = raw.copy()

    frac = np.zeros(len(df), dtype=float)
    factor = np.ones(len(df), dtype=float)
    event_id = np.full(len(df), -1, dtype=np.int32)

    ts = df["Timestamp"].to_numpy(dtype="datetime64[ns]")

    eligible = interval_audit[
        interval_audit["status"].eq(ELIGIBLE_STATUS)
    ].copy()

    applied_rows = []

    for row in eligible.itertuples(index=False):
        start = pd.Timestamp(row.correction_start)
        end = pd.Timestamp(row.correction_end)
        gain = float(row.current_gain)
        eid = int(row.current_event_id)

        if not (0.0 < gain < 1.0):
            raise RuntimeError(f"Invalid gain for event {eid}: {gain}")
        if not start < end:
            raise RuntimeError(f"Invalid interval for event {eid}: {start} -> {end}")

        # Open interval: after previous Cleaning pulse, before current first pulse.
        mask = (ts > np.datetime64(start)) & (ts < np.datetime64(end))
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise RuntimeError(f"No minute rows found in eligible interval event {eid}")

        total_seconds = (end - start).total_seconds()
        elapsed_seconds = (
            df.loc[idx, "Timestamp"] - start
        ).dt.total_seconds().to_numpy(dtype=float)

        u = elapsed_seconds / total_seconds
        s_t = gain * u

        if np.any(s_t < 0.0) or np.any(s_t >= gain + 1e-12):
            raise RuntimeError(f"Invalid interpolation fraction for event {eid}")
        if np.any(s_t >= 1.0):
            raise RuntimeError(f"Correction fraction >=1 for event {eid}")

        primary_factor = 1.0 / (1.0 - s_t)

        corr[idx] = raw[idx] * primary_factor
        corr_mul[idx] = raw[idx] * (1.0 + s_t)
        frac[idx] = s_t
        factor[idx] = primary_factor
        event_id[idx] = eid

        applied_rows.append(
            {
                "current_event_id": eid,
                "current_date": str(row.current_date),
                "correction_start": start,
                "correction_end": end,
                "current_gain": gain,
                "modified_minutes": int(idx.size),
                "max_applied_fraction": float(s_t.max()),
                "max_primary_factor": float(primary_factor.max()),
                "formula": "ModA_corr = ModA_raw / (1 - s_A(t))",
            }
        )

    df["ModA_corr"] = corr
    df["ModA_corr_mul1p_sensitivity"] = corr_mul
    df["ModA_correction_fraction"] = frac
    df["ModA_correction_factor"] = factor
    df["ModA_correction_event_id"] = event_id

    # Hard invariants.
    outside = event_id < 0
    if not np.array_equal(corr[outside], raw[outside], equal_nan=True):
        raise RuntimeError("ModA changed outside eligible intervals.")
    if np.nanmin(factor) < 1.0:
        raise RuntimeError("Primary correction factor below 1.")
    if not np.all(np.isfinite(factor)):
        raise RuntimeError("Non-finite primary correction factor.")

    applied = pd.DataFrame(applied_rows)
    return df, applied


def build_event_validation(
    corrected: pd.DataFrame,
    interval_audit: pd.DataFrame,
) -> pd.DataFrame:
    indexed = corrected.set_index("Timestamp", drop=False)

    rows = []
    eligible = interval_audit[
        interval_audit["status"].eq(ELIGIBLE_STATUS)
    ].copy()

    for row in eligible.itertuples(index=False):
        first = pd.Timestamp(row.current_first_pulse)
        last = pd.Timestamp(row.current_last_pulse)

        pre_end = first - pd.Timedelta(minutes=EVENT_BUFFER_MIN)
        pre_start = pre_end - pd.Timedelta(minutes=PRIMARY_WINDOW_MIN - 1)
        post_start = last + pd.Timedelta(minutes=EVENT_BUFFER_MIN)
        post_end = post_start + pd.Timedelta(minutes=PRIMARY_WINDOW_MIN - 1)

        pre_raw = robust_window_q(indexed, "ModA", pre_start, pre_end)
        post_raw = robust_window_q(indexed, "ModA", post_start, post_end)

        pre_corr = robust_window_q(indexed, "ModA_corr", pre_start, pre_end)
        post_corr = robust_window_q(indexed, "ModA_corr", post_start, post_end)

        pre_mul = robust_window_q(
            indexed, "ModA_corr_mul1p_sensitivity", pre_start, pre_end
        )
        post_mul = robust_window_q(
            indexed, "ModA_corr_mul1p_sensitivity", post_start, post_end
        )

        for name, x in [
            ("pre_raw", pre_raw),
            ("post_raw", post_raw),
            ("pre_corr", pre_corr),
            ("post_corr", post_corr),
            ("pre_mul", pre_mul),
            ("post_mul", post_mul),
        ]:
            if x["n"] <= 0 or not np.isfinite(x["q_median"]):
                raise RuntimeError(
                    f"Invalid validation window {name} for event "
                    f"{int(row.current_event_id)}"
                )

        raw_gain_recalc = 1.0 - pre_raw["q_median"] / post_raw["q_median"]
        residual_primary = 1.0 - pre_corr["q_median"] / post_corr["q_median"]
        residual_mul1p = 1.0 - pre_mul["q_median"] / post_mul["q_median"]

        rows.append(
            {
                "current_event_id": int(row.current_event_id),
                "current_date": str(row.current_date),
                "frozen_current_gain": float(row.current_gain),
                "raw_gain_recalculated": float(raw_gain_recalc),
                "gain_recalc_abs_error": float(
                    abs(raw_gain_recalc - float(row.current_gain))
                ),
                "n_pre": int(pre_raw["n"]),
                "n_post": int(post_raw["n"]),
                "q_pre_raw": pre_raw["q_median"],
                "q_post_raw": post_raw["q_median"],
                "q_pre_primary_corr": pre_corr["q_median"],
                "q_post_primary_corr": post_corr["q_median"],
                "residual_gain_primary": float(residual_primary),
                "abs_residual_gain_primary": float(abs(residual_primary)),
                "q_pre_mul1p": pre_mul["q_median"],
                "q_post_mul1p": post_mul["q_median"],
                "residual_gain_mul1p": float(residual_mul1p),
                "abs_residual_gain_mul1p": float(abs(residual_mul1p)),
                "primary_abs_jump_reduction_fraction": float(
                    1.0
                    - abs(residual_primary)
                    / max(abs(raw_gain_recalc), 1e-15)
                ),
                "mul1p_abs_jump_reduction_fraction": float(
                    1.0
                    - abs(residual_mul1p)
                    / max(abs(raw_gain_recalc), 1e-15)
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values("current_event_id").reset_index(drop=True)

    # Frozen gain reconstruction should reproduce the upstream values to CSV precision.
    if out["gain_recalc_abs_error"].max() > 1e-10:
        raise RuntimeError(
            "Could not reproduce frozen 5.4a gain values; max abs error="
            f"{out['gain_recalc_abs_error'].max()}"
        )

    return out


def make_summary(
    y1_path: Path,
    y2_path: Path,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    combined: pd.DataFrame,
    interval_audit: pd.DataFrame,
    corrected: pd.DataFrame,
    applied: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    modified = corrected["ModA_correction_event_id"] >= 0
    primary_abs = validation["abs_residual_gain_primary"]
    mul_abs = validation["abs_residual_gain_mul1p"]
    raw_abs = validation["raw_gain_recalculated"].abs()

    official = validation[
        validation["current_date"].astype(str).eq("2022-03-29")
    ]

    if len(official) == 1:
        rr = official.iloc[0]
        official_summary = {
            "present": True,
            "raw_gain": float(rr["raw_gain_recalculated"]),
            "residual_gain_primary": float(rr["residual_gain_primary"]),
            "residual_gain_mul1p": float(rr["residual_gain_mul1p"]),
            "primary_jump_reduction_fraction": float(
                rr["primary_abs_jump_reduction_fraction"]
            ),
        }
    else:
        official_summary = {"present": False}

    return {
        "stage": "P2-0B-5.4c-2",
        "moda_correction_applied": True,
        "final_cleanliness_generated": False,
        "soiling_state_generated": False,
        "rl_state_generated": False,
        "input": {
            "year1_path": str(y1_path),
            "year1_sha256": sha256_file(y1_path),
            "year1_rows": int(len(y1)),
            "year2_path": str(y2_path),
            "year2_sha256": sha256_file(y2_path),
            "year2_rows": int(len(y2)),
            "combined_rows": int(len(combined)),
        },
        "frozen_rule": {
            "threshold": FROZEN_THRESHOLD,
            "interval_status_used": ELIGIBLE_STATUS,
            "eligible_intervals": int(len(applied)),
            "rain_skipped_intervals": int(
                interval_audit["status"].eq(RAIN_SKIP_STATUS).sum()
            ),
            "formula_primary": "ModA_corr = ModA_raw / (1 - s_A(t))",
            "linear_fraction_definition": (
                "s_A(t)=current_gain*(t-correction_start)/"
                "(correction_end-correction_start)"
            ),
            "interval_is_open": True,
        },
        "application": {
            "modified_minute_rows": int(modified.sum()),
            "modified_fraction_of_two_year_minutes": float(modified.mean()),
            "max_correction_fraction": float(
                corrected.loc[modified, "ModA_correction_fraction"].max()
            ),
            "max_correction_factor": float(
                corrected.loc[modified, "ModA_correction_factor"].max()
            ),
            "moda_raw_distribution_modified": finite_quantiles(
                corrected.loc[modified, "ModA"]
            ),
            "moda_corr_distribution_modified": finite_quantiles(
                corrected.loc[modified, "ModA_corr"]
            ),
        },
        "event_validation": {
            "events": int(len(validation)),
            "raw_abs_gain_distribution": finite_quantiles(raw_abs),
            "primary_abs_residual_distribution": finite_quantiles(primary_abs),
            "mul1p_abs_residual_distribution": finite_quantiles(mul_abs),
            "median_primary_jump_reduction_fraction": float(
                validation["primary_abs_jump_reduction_fraction"].median()
            ),
            "median_mul1p_jump_reduction_fraction": float(
                validation["mul1p_abs_jump_reduction_fraction"].median()
            ),
            "fraction_primary_better_than_mul1p": float((primary_abs < mul_abs).mean()),
            "fraction_primary_reduces_abs_jump": float((primary_abs < raw_abs).mean()),
            "official_example_2022_03_29": official_summary,
        },
        "notes": [
            "Original WAPP CSVs are never modified.",
            "Only c-1 intervals explicitly marked ELIGIBLE_FOR_LINEAR_CORRECTION are changed.",
            "Rain-confounded intervals remain uncorrected.",
            "Outside eligible intervals, ModA_corr equals raw ModA exactly.",
            "The primary division-by-(1-s) formula is the exact inverse of the frozen cleaning-jump definition s=1-Q_pre/Q_post under proportional attenuation.",
            "The multiply-by-(1+s) formula is evaluated only as a small-s approximation sensitivity benchmark.",
            "This is reference-module correction only; final WAPP cleanliness is not yet constructed.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "P2-0B-5.4c-2 apply audited ModA correction and validate cleaning jumps."
        )
    )
    p.add_argument("--year1", required=True, type=Path)
    p.add_argument("--year2", required=True, type=Path)
    p.add_argument("--interval-audit", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0b_5_4c2_moda_correction_v1"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    y1_path = args.year1.expanduser().resolve()
    y2_path = args.year2.expanduser().resolve()
    interval_path = args.interval_audit.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for p in [y1_path, y2_path, interval_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    print("[1/8] Verify frozen input hashes")
    y1_hash = sha256_file(y1_path)
    y2_hash = sha256_file(y2_path)
    if y1_hash != EXPECTED_YEAR1_SHA256:
        raise RuntimeError(f"Year1 SHA256 mismatch: {y1_hash}")
    if y2_hash != EXPECTED_YEAR2_SHA256:
        raise RuntimeError(f"Year2 SHA256 mismatch: {y2_hash}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[2/8] Read Year1")
    y1 = read_wapp_csv(y1_path)

    print("[3/8] Read Year2")
    y2 = read_wapp_csv(y2_path)

    print("[4/8] Merge + continuity audit")
    combined = (
        pd.concat([y1, y2], ignore_index=True)
        .sort_values("Timestamp", kind="stable")
        .reset_index(drop=True)
    )
    validate_combined(combined)

    print("[5/8] Read + validate frozen c-1 interval audit")
    interval_audit = pd.read_csv(interval_path, encoding="utf-8-sig")
    interval_audit = validate_interval_input(interval_audit)

    print("[6/8] Apply primary correction + formula sensitivity")
    corrected, applied = apply_primary_and_sensitivity(
        combined=combined,
        interval_audit=interval_audit,
    )

    print("[7/8] Event-level jump validation")
    validation = build_event_validation(
        corrected=corrected,
        interval_audit=interval_audit,
    )

    # Minimal scientific invariant: the physically exact primary formula must
    # reduce the median absolute cleaning discontinuity relative to raw.
    if (
        validation["abs_residual_gain_primary"].median()
        >= validation["raw_gain_recalculated"].abs().median()
    ):
        raise RuntimeError(
            "Primary correction did not reduce median absolute cleaning jump."
        )

    print("[8/8] Write outputs")
    modified = corrected["ModA_correction_event_id"] >= 0
    segment_cols = [
        "Timestamp",
        "ModA",
        "ModA_corr",
        "ModB",
        "GHI",
        "Precipitation",
        "Cleaning",
        "ModA_correction_fraction",
        "ModA_correction_factor",
        "ModA_correction_event_id",
    ]
    corrected.loc[modified, segment_cols].to_csv(
        out_dir / "moda_corrected_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    applied.to_csv(
        out_dir / "applied_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation.to_csv(
        out_dir / "event_jump_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = make_summary(
        y1_path=y1_path,
        y2_path=y2_path,
        y1=y1,
        y2=y2,
        combined=combined,
        interval_audit=interval_audit,
        corrected=corrected,
        applied=applied,
        validation=validation,
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "moda_corrected_segments.csv")
    print(out_dir / "applied_intervals.csv")
    print(out_dir / "event_jump_validation.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: ModA reference correction only; final cleanliness and RL state are NOT generated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
