#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0B-5.4a
WAPP Malanville ModA-cleaning event inventory and pre/post sensitivity audit.

Purpose
-------
This stage is DIAGNOSTIC ONLY. It does NOT apply ModA correction.

It:
1) reads Year1 + Year2 official 1-minute QC CSVs;
2) verifies strict 1-minute continuity;
3) groups Cleaning pulses by local calendar date (official maintenance-visit unit);
4) records first/last Cleaning pulse, pulse count, event span, daily rainfall;
5) flags official ModB-cleaning dates and scheduled-maintenance dates;
6) audits pre/post ModA/ModB ratio around each cleaning visit under several
   irradiance thresholds and window lengths;
7) estimates candidate ModA cleaning jumps WITHOUT choosing a gain threshold;
8) exports all event-level and sensitivity-level diagnostics.

Scientific boundary
-------------------
Official WAPP/CSPS methodology states that ModA soiling at cleaning can be
detected by a sudden signal increase in 1-minute data, and strong ModA soiling
may be corrected by backward linear interpolation toward the previous cleaning.
Rain/dust-storm/other non-linear periods must not be treated mechanically.

Therefore this script follows "distribution first, threshold later":
- NO hard "0.5% / 1% / 2%" ModA-gain threshold is applied.
- NO ModA correction is applied.
- ModB-cleaning dates are flagged as confounded for ModA-only jump estimation.
- Rain-near-event and scheduled-maintenance dates are flagged as confounded.
- Pre/post support and MAD are exported instead of silently deleting events.

Inputs
------
Official WAPP Malanville Year1 / Year2 QC CSVs.
CSV encoding: cp1252; the second row contains units and is skipped.

Outputs
-------
event_inventory.csv
prepost_sensitivity.csv
audit_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

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

# Candidate sensitivity grid. These are NOT frozen correction parameters.
IRRADIANCE_THRESHOLDS_WM2 = (100.0, 200.0, 300.0)
WINDOW_LENGTHS_MIN = (30, 60, 120)

# Keep a small gap between the button-pulse span and pre/post windows so that
# the actual cleaning operation itself does not contaminate the medians.
EVENT_BUFFER_MIN = 15

# Diagnostic rain-confounding window only; not a deletion rule for the dataset.
RAIN_NEAR_EVENT_HOURS = 3

# Dates identified in the official final Malanville station report.
SCHEDULED_MAINTENANCE_DATES = {
    pd.Timestamp("2022-05-19").date(),
    pd.Timestamp("2022-09-17").date(),
    pd.Timestamp("2023-03-07").date(),
}

# Official ModB cleaning dates from Table 12 of the final two-year report.
# 2021-08-08 precedes the public Year1 CSV start and is retained here for provenance.
MODB_CLEANING_DATES = [
    pd.Timestamp(x).date()
    for x in [
        "2021-08-08",
        "2021-09-01",
        "2021-10-01",
        "2021-11-01",
        "2021-11-09",
        "2021-12-01",
        "2021-12-31",
        "2022-02-01",
        "2022-03-01",
        "2022-04-01",
        "2022-05-03",
        "2022-05-31",
        "2022-06-03",
        "2022-07-01",
        "2022-08-01",
        "2022-09-01",
        "2022-10-03",
        "2022-11-01",
        "2022-11-30",
        "2022-12-29",
        "2023-02-01",
        "2023-03-01",
        "2023-04-04",
        "2023-05-01",
        "2023-06-01",
        "2023-07-01",
        "2023-08-02",
    ]
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_mad(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)))


def read_wapp_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="cp1252",
        skiprows=[1],  # second row is the units row
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


def build_event_inventory(df: pd.DataFrame) -> pd.DataFrame:
    clean_mask = df["Cleaning"].fillna(0).to_numpy(dtype=float) > 0.0
    pulses = df.loc[clean_mask, ["Timestamp", "Cleaning"]].copy()
    if pulses.empty:
        raise RuntimeError("No Cleaning pulses found.")

    pulses["date"] = pulses["Timestamp"].dt.date

    rows = []
    for event_id, (event_date, g) in enumerate(pulses.groupby("date", sort=True), start=1):
        first_ts = g["Timestamp"].min()
        last_ts = g["Timestamp"].max()

        day_start = pd.Timestamp(event_date)
        day_end = day_start + pd.Timedelta(days=1)
        day = df[(df["Timestamp"] >= day_start) & (df["Timestamp"] < day_end)]

        rain_same_day = float(day["Precipitation"].fillna(0).sum())

        near_start = first_ts - pd.Timedelta(hours=RAIN_NEAR_EVENT_HOURS)
        near_end = last_ts + pd.Timedelta(hours=RAIN_NEAR_EVENT_HOURS)
        near = df[(df["Timestamp"] >= near_start) & (df["Timestamp"] <= near_end)]
        rain_near_mm = float(near["Precipitation"].fillna(0).sum())

        rows.append(
            {
                "event_id": event_id,
                "date": str(event_date),
                "first_pulse": first_ts,
                "last_pulse": last_ts,
                "pulse_count": int(len(g)),
                "pulse_span_min": float((last_ts - first_ts) / pd.Timedelta(minutes=1)),
                "rain_same_day_mm": rain_same_day,
                "rain_near_event_mm": rain_near_mm,
                "rain_near_event": bool(rain_near_mm > 0),
                "is_modb_cleaning_date": bool(event_date in MODB_CLEANING_DATES),
                "is_scheduled_maintenance": bool(
                    event_date in SCHEDULED_MAINTENANCE_DATES
                ),
            }
        )

    events = pd.DataFrame(rows)

    # Metadata-only eligibility: no signal-amplitude threshold is used here.
    events["eligible_metadata_only"] = (
        ~events["is_modb_cleaning_date"]
        & ~events["is_scheduled_maintenance"]
        & ~events["rain_near_event"]
    )

    return events


def compute_window_stats(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    irradiance_threshold: float,
) -> dict:
    w = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].copy()

    valid = (
        np.isfinite(w["ModA"].to_numpy(dtype=float))
        & np.isfinite(w["ModB"].to_numpy(dtype=float))
        & (w["ModA"].to_numpy(dtype=float) > irradiance_threshold)
        & (w["ModB"].to_numpy(dtype=float) > 0.0)
        & (w["Precipitation"].fillna(0).to_numpy(dtype=float) <= 0.0)
    )
    w = w.loc[valid].copy()

    if w.empty:
        return {
            "n": 0,
            "q_median": np.nan,
            "q_mad": np.nan,
            "q_iqr": np.nan,
            "ghi_median": np.nan,
            "moda_median": np.nan,
            "modb_median": np.nan,
        }

    q = w["ModA"] / w["ModB"]
    return {
        "n": int(len(w)),
        "q_median": float(q.median()),
        "q_mad": robust_mad(q),
        "q_iqr": float(q.quantile(0.75) - q.quantile(0.25)),
        "ghi_median": float(w["GHI"].median()),
        "moda_median": float(w["ModA"].median()),
        "modb_median": float(w["ModB"].median()),
    }


def build_prepost_sensitivity(
    df: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    for ev in events.itertuples(index=False):
        first_ts = pd.Timestamp(ev.first_pulse)
        last_ts = pd.Timestamp(ev.last_pulse)

        for threshold in IRRADIANCE_THRESHOLDS_WM2:
            for window_min in WINDOW_LENGTHS_MIN:
                pre_end = first_ts - pd.Timedelta(minutes=EVENT_BUFFER_MIN)
                pre_start = pre_end - pd.Timedelta(minutes=window_min - 1)

                post_start = last_ts + pd.Timedelta(minutes=EVENT_BUFFER_MIN)
                post_end = post_start + pd.Timedelta(minutes=window_min - 1)

                pre = compute_window_stats(
                    df, pre_start, pre_end, irradiance_threshold=threshold
                )
                post = compute_window_stats(
                    df, post_start, post_end, irradiance_threshold=threshold
                )

                if (
                    pre["n"] > 0
                    and post["n"] > 0
                    and np.isfinite(pre["q_median"])
                    and np.isfinite(post["q_median"])
                    and post["q_median"] > 0
                    and pre["q_median"] > 0
                ):
                    # Candidate estimate of ModA soiling just before cleaning:
                    # if Q_pre=0.97 and Q_post=1.00, estimate = 0.03.
                    moda_soiling_est = 1.0 - pre["q_median"] / post["q_median"]

                    # Relative increase of Q itself, retained separately for diagnostics.
                    q_signal_gain_rel = post["q_median"] / pre["q_median"] - 1.0
                else:
                    moda_soiling_est = np.nan
                    q_signal_gain_rel = np.nan

                rows.append(
                    {
                        "event_id": int(ev.event_id),
                        "date": ev.date,
                        "first_pulse": ev.first_pulse,
                        "last_pulse": ev.last_pulse,
                        "pulse_count": int(ev.pulse_count),
                        "pulse_span_min": float(ev.pulse_span_min),
                        "is_modb_cleaning_date": bool(ev.is_modb_cleaning_date),
                        "is_scheduled_maintenance": bool(
                            ev.is_scheduled_maintenance
                        ),
                        "rain_near_event": bool(ev.rain_near_event),
                        "rain_near_event_mm": float(ev.rain_near_event_mm),
                        "eligible_metadata_only": bool(ev.eligible_metadata_only),
                        "irradiance_threshold_wm2": float(threshold),
                        "window_min": int(window_min),
                        "event_buffer_min": EVENT_BUFFER_MIN,
                        "pre_start": pre_start,
                        "pre_end": pre_end,
                        "post_start": post_start,
                        "post_end": post_end,
                        "n_pre": pre["n"],
                        "n_post": post["n"],
                        "q_pre_median": pre["q_median"],
                        "q_post_median": post["q_median"],
                        "q_pre_mad": pre["q_mad"],
                        "q_post_mad": post["q_mad"],
                        "q_pre_iqr": pre["q_iqr"],
                        "q_post_iqr": post["q_iqr"],
                        "ghi_pre_median": pre["ghi_median"],
                        "ghi_post_median": post["ghi_median"],
                        "moda_pre_median": pre["moda_median"],
                        "moda_post_median": post["moda_median"],
                        "modb_pre_median": pre["modb_median"],
                        "modb_post_median": post["modb_median"],
                        "moda_soiling_est": moda_soiling_est,
                        "q_signal_gain_rel": q_signal_gain_rel,
                    }
                )

    return pd.DataFrame(rows)


def finite_quantiles(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return {}
    qs = x.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        **{f"q{int(q*100):02d}": float(v) for q, v in qs.items()},
        "max": float(x.max()),
    }


def make_summary(
    y1_path: Path,
    y2_path: Path,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    combined: pd.DataFrame,
    events: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> dict:
    combos = []
    for (thr, win), g in sensitivity.groupby(
        ["irradiance_threshold_wm2", "window_min"], sort=True
    ):
        meta = g[g["eligible_metadata_only"]].copy()
        both_support = meta[(meta["n_pre"] > 0) & (meta["n_post"] > 0)]
        support_10 = meta[(meta["n_pre"] >= 10) & (meta["n_post"] >= 10)]
        support_20 = meta[(meta["n_pre"] >= 20) & (meta["n_post"] >= 20)]

        combos.append(
            {
                "irradiance_threshold_wm2": float(thr),
                "window_min": int(win),
                "metadata_eligible_events": int(len(meta)),
                "events_with_any_prepost_support": int(len(both_support)),
                "events_with_npre_npost_ge10": int(len(support_10)),
                "events_with_npre_npost_ge20": int(len(support_20)),
                "moda_soiling_est_distribution_ge10": finite_quantiles(
                    support_10["moda_soiling_est"]
                ),
                "q_pre_mad_distribution_ge10": finite_quantiles(
                    support_10["q_pre_mad"]
                ),
                "q_post_mad_distribution_ge10": finite_quantiles(
                    support_10["q_post_mad"]
                ),
            }
        )

    event_span = finite_quantiles(events["pulse_span_min"])

    return {
        "stage": "P2-0B-5.4a",
        "diagnostic_only": True,
        "moda_correction_applied": False,
        "gain_threshold_applied": False,
        "input": {
            "year1_path": str(y1_path),
            "year1_sha256": sha256_file(y1_path),
            "year1_rows": int(len(y1)),
            "year2_path": str(y2_path),
            "year2_sha256": sha256_file(y2_path),
            "year2_rows": int(len(y2)),
            "combined_rows": int(len(combined)),
            "combined_start": str(combined["Timestamp"].iloc[0]),
            "combined_end": str(combined["Timestamp"].iloc[-1]),
        },
        "cleaning_inventory": {
            "cleaning_pulses": int((combined["Cleaning"].fillna(0) > 0).sum()),
            "unique_cleaning_dates": int(len(events)),
            "official_expected_maintenance_visits": 504,
            "modb_cleaning_dates_in_inventory": int(
                events["is_modb_cleaning_date"].sum()
            ),
            "scheduled_maintenance_dates_in_inventory": int(
                events["is_scheduled_maintenance"].sum()
            ),
            "rain_near_event_dates": int(events["rain_near_event"].sum()),
            "metadata_eligible_events": int(events["eligible_metadata_only"].sum()),
            "pulse_span_min_distribution": event_span,
        },
        "sensitivity_grid": {
            "irradiance_thresholds_wm2": list(IRRADIANCE_THRESHOLDS_WM2),
            "window_lengths_min": list(WINDOW_LENGTHS_MIN),
            "event_buffer_min": EVENT_BUFFER_MIN,
            "rain_near_event_hours": RAIN_NEAR_EVENT_HOURS,
        },
        "combo_summaries": combos,
        "notes": [
            "Cleaning pulses are grouped by unique local calendar date because this reproduces the official maintenance-visit count.",
            "For each date, pre-window ends before the first pulse and post-window begins after the last pulse.",
            "Q = ModA/ModB is used only to diagnose ModA cleaning jumps.",
            "moda_soiling_est = 1 - Q_pre/Q_post; no amplitude threshold is applied.",
            "ModB-cleaning, scheduled-maintenance, and rain-near-event dates are flagged as metadata confounds.",
            "No ModA correction is performed in this stage.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P2-0B-5.4a ModA cleaning-event inventory and pre/post sensitivity audit."
    )
    p.add_argument("--year1", required=True, type=Path)
    p.add_argument("--year2", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0b_5_4a_moda_event_audit_v1"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    y1_path = args.year1.expanduser().resolve()
    y2_path = args.year2.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for p in (y1_path, y2_path):
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Read Year1")
    y1 = read_wapp_csv(y1_path)
    print("[2/7] Read Year2")
    y2 = read_wapp_csv(y2_path)

    print("[3/7] Merge + continuity audit")
    combined = (
        pd.concat([y1, y2], ignore_index=True)
        .sort_values("Timestamp", kind="stable")
        .reset_index(drop=True)
    )
    validate_combined(combined)

    print("[4/7] Build Cleaning-event inventory")
    events = build_event_inventory(combined)

    print("[5/7] Compute pre/post sensitivity grid")
    sensitivity = build_prepost_sensitivity(combined, events)

    print("[6/7] Write outputs")
    events.to_csv(out_dir / "event_inventory.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(
        out_dir / "prepost_sensitivity.csv", index=False, encoding="utf-8-sig"
    )

    summary = make_summary(
        y1_path, y2_path, y1, y2, combined, events, sensitivity
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[7/7] Done")
    print(out_dir / "event_inventory.csv")
    print(out_dir / "prepost_sensitivity.csv")
    print(out_dir / "audit_summary.json")
    print("IMPORTANT: diagnostic only; no ModA correction and no gain threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
