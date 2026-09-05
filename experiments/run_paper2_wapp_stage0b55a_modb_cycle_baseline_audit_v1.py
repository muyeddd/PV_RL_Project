#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0B-5.5a
WAPP Malanville ModB cleaning-cycle + baseline-candidate audit.

Purpose
-------
Build the 27 ModB manual-cleaning cycles and audit cycle-specific clean
baseline candidates B_j using the already frozen ModA correction.

This stage is DIAGNOSTIC ONLY:
- it reconstructs full ModA_corr by patching the frozen c-2 corrected segments;
- it computes daily corrected ratio R_d^corr = median(ModB / ModA_corr);
- it builds 27 ModB cleaning cycles;
- it evaluates baseline-support rules Nmin in {20, 30, 60};
- it DOES NOT yet freeze B_j;
- it DOES NOT yet compute final C_d^WAPP;
- it DOES NOT yet compute S_d^soil;
- it DOES NOT construct any RL state.

Frozen principles inherited from upstream
-----------------------------------------
1) Full ModA_corr is reconstructed from raw WAPP data + frozen
   moda_corrected_segments.csv (single source of truth).
2) Sample eligibility is based on ORIGINAL ModA, not corrected ModA.
3) Daily ratio uses corrected ModA in the denominator.
4) Daily filter:
   - local solar-noon +/-120 min
   - raw ModA > 200 W/m2
   - finite positive ModA / ModA_corr / ModB
   - Cleaning pulse +/-30 min excluded
   - active precipitation + 30 min post-rain excluded
   - rain days themselves are retained
   - no ratio clipping
5) Only authoritative Table-12 ModB manual-clean dates create cycle boundaries.
   Rain NEVER resets a cycle.
6) 2021-08-08 is a pre-period clean anchor, not an observed in-period cleaning
   event. Cycle 0 therefore uses the first valid report-period day as its
   baseline candidate.
7) For in-period cycles, baseline candidate is the earliest trustworthy
   post-clean observation:
   a) same-day post-clean solar-window samples after last Cleaning pulse +30 min;
   b) if support is insufficient, first subsequent valid day before the next
      ModB cleaning boundary.
   We never choose the maximum ratio or the day closest to 1.
8) C>1 / S<0 are NOT treated here and will not be clipped later at the
   observational reconstruction layer.

Outputs
-------
cycle_inventory.csv
daily_corrected_ratio_audit.csv
baseline_candidates.csv
audit_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

# Frozen Malanville location / local-standard-time metadata.
LATITUDE_DEG = 11.7827
LONGITUDE_DEG_EAST = 3.3735
UTC_OFFSET_HOURS = 1.0

SOLAR_NOON_HALF_WINDOW_MIN = 120
DAILY_RAW_MODA_THRESHOLD_WM2 = 200.0
CLEANING_BUFFER_MIN = 30
POST_RAIN_BUFFER_MIN = 30

BASELINE_SUPPORT_GRID = (20, 30, 60)
PRIMARY_BASELINE_SUPPORT = 30

EXPECTED_CORRECTED_SEGMENT_ROWS = 128705

SCHEDULED_MAINTENANCE_DATES = {
    pd.Timestamp("2022-05-19").date(),
    pd.Timestamp("2022-09-17").date(),
    pd.Timestamp("2023-03-07").date(),
}

# Table 12 authoritative ModB manual-clean boundaries.
# 2021-08-08 is deliberately retained as a pre-period clean anchor.
MODB_CLEANING_BOUNDARIES = [
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


def robust_mad(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def equation_of_time_minutes(day_of_year: int) -> float:
    """
    Standard approximate equation of time:
    B = 2*pi*(N-81)/364
    EoT = 9.87 sin(2B) - 7.53 cos(B) - 1.5 sin(B), minutes.
    """
    b = 2.0 * math.pi * (day_of_year - 81.0) / 364.0
    return 9.87 * math.sin(2.0 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def local_solar_noon(day: pd.Timestamp) -> pd.Timestamp:
    """
    Local standard-time solar noon.

    Time correction factor (minutes):
        TC = 4*(longitude - local_standard_meridian) + EoT
    Solar time = local clock time + TC
    Therefore local-clock solar noon = 12:00 - TC.
    """
    day = pd.Timestamp(day).normalize()
    eot = equation_of_time_minutes(int(day.dayofyear))
    lstm = 15.0 * UTC_OFFSET_HOURS
    tc = 4.0 * (LONGITUDE_DEG_EAST - lstm) + eot
    return day + pd.Timedelta(hours=12) - pd.Timedelta(minutes=tc)


def patch_frozen_moda_corr(
    combined: pd.DataFrame,
    corrected_segments_path: Path,
) -> tuple[pd.DataFrame, dict]:
    seg = pd.read_csv(corrected_segments_path, encoding="utf-8-sig")
    required = {
        "Timestamp",
        "ModA",
        "ModA_corr",
        "ModA_correction_event_id",
    }
    missing = required.difference(seg.columns)
    if missing:
        raise RuntimeError(
            f"moda_corrected_segments missing columns: {sorted(missing)}"
        )

    seg["Timestamp"] = pd.to_datetime(seg["Timestamp"], errors="raise")

    if len(seg) != EXPECTED_CORRECTED_SEGMENT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_CORRECTED_SEGMENT_ROWS} corrected-segment rows, "
            f"found {len(seg)}"
        )
    if seg["Timestamp"].duplicated().any():
        raise RuntimeError("Corrected segments contain duplicate timestamps.")

    df = combined.copy()
    df["ModA_corr"] = df["ModA"].to_numpy(dtype=float)

    raw_idx = pd.Index(df["Timestamp"])
    loc = raw_idx.get_indexer(seg["Timestamp"])
    if np.any(loc < 0):
        raise RuntimeError(
            f"{int((loc < 0).sum())} corrected-segment timestamps not found in raw data."
        )
    if len(np.unique(loc)) != len(loc):
        raise RuntimeError("Corrected-segment timestamps map non-uniquely.")

    raw_at_patch = df.loc[loc, "ModA"].to_numpy(dtype=float)
    seg_raw = pd.to_numeric(seg["ModA"], errors="raise").to_numpy(dtype=float)
    if not np.allclose(raw_at_patch, seg_raw, rtol=0.0, atol=1e-10, equal_nan=True):
        raise RuntimeError(
            "Raw ModA values in corrected segments do not reproduce source CSV."
        )

    corr_values = pd.to_numeric(seg["ModA_corr"], errors="raise").to_numpy(dtype=float)
    df.loc[loc, "ModA_corr"] = corr_values

    outside = np.ones(len(df), dtype=bool)
    outside[loc] = False
    if not np.array_equal(
        df.loc[outside, "ModA_corr"].to_numpy(dtype=float),
        df.loc[outside, "ModA"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise RuntimeError("ModA_corr differs from raw ModA outside frozen segments.")

    patch_info = {
        "segment_rows": int(len(seg)),
        "unique_segment_timestamps": int(seg["Timestamp"].nunique()),
        "min_timestamp": str(seg["Timestamp"].min()),
        "max_timestamp": str(seg["Timestamp"].max()),
        "numerically_changed_rows": int(
            np.sum(
                ~np.isclose(
                    raw_at_patch,
                    corr_values,
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                )
            )
        ),
    }
    return df, patch_info


def build_exclusion_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    ts = df["Timestamp"]
    n = len(df)

    cleaning_pulse = (
        pd.to_numeric(df["Cleaning"], errors="coerce").fillna(0.0).to_numpy() > 0.0
    )
    rain_active = (
        pd.to_numeric(df["Precipitation"], errors="coerce").fillna(0.0).to_numpy()
        > 0.0
    )

    # +/- 30 minutes around any Cleaning pulse.
    clean_series = pd.Series(cleaning_pulse.astype(np.int8), index=ts)
    cleaning_near = (
        clean_series.rolling(
            f"{2*CLEANING_BUFFER_MIN+1}min",
            center=True,
            min_periods=1,
        )
        .max()
        .to_numpy(dtype=float)
        > 0.0
    )

    # Active rain + 30 minutes after latest active-rain minute.
    rain_series = pd.Series(rain_active.astype(np.int8), index=ts)
    rain_or_post = (
        rain_series.rolling(
            f"{POST_RAIN_BUFFER_MIN+1}min",
            min_periods=1,
        )
        .max()
        .to_numpy(dtype=float)
        > 0.0
    )

    # Solar-noon window for each minute.
    dates = ts.dt.normalize()
    unique_days = pd.Index(dates.unique()).sort_values()
    bounds = {}
    for day in unique_days:
        noon = local_solar_noon(day)
        bounds[pd.Timestamp(day)] = (
            noon - pd.Timedelta(minutes=SOLAR_NOON_HALF_WINDOW_MIN),
            noon + pd.Timedelta(minutes=SOLAR_NOON_HALF_WINDOW_MIN),
        )

    solar_window = np.zeros(n, dtype=bool)
    for day, (lo, hi) in bounds.items():
        day_mask = dates.eq(day).to_numpy()
        tday = ts.to_numpy(dtype="datetime64[ns]")
        solar_window[day_mask] = (
            (tday[day_mask] >= np.datetime64(lo))
            & (tday[day_mask] <= np.datetime64(hi))
        )

    moda_raw = pd.to_numeric(df["ModA"], errors="coerce").to_numpy(dtype=float)
    moda_corr = pd.to_numeric(df["ModA_corr"], errors="coerce").to_numpy(dtype=float)
    modb = pd.to_numeric(df["ModB"], errors="coerce").to_numpy(dtype=float)

    finite_positive = (
        np.isfinite(moda_raw)
        & np.isfinite(moda_corr)
        & np.isfinite(modb)
        & (moda_raw > 0.0)
        & (moda_corr > 0.0)
        & (modb > 0.0)
    )

    valid_daily = (
        solar_window
        & finite_positive
        & (moda_raw > DAILY_RAW_MODA_THRESHOLD_WM2)
        & ~cleaning_near
        & ~rain_or_post
    )

    return {
        "cleaning_pulse": cleaning_pulse,
        "cleaning_near": cleaning_near,
        "rain_active": rain_active,
        "rain_or_post": rain_or_post,
        "solar_window": solar_window,
        "finite_positive": finite_positive,
        "valid_daily": valid_daily,
    }


def daily_corrected_ratio(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "date": df["Timestamp"].dt.date.astype(str),
            "Timestamp": df["Timestamp"],
            "valid": masks["valid_daily"],
            "ratio_corr": (
                pd.to_numeric(df["ModB"], errors="coerce")
                / pd.to_numeric(df["ModA_corr"], errors="coerce")
            ),
            "rain_mm": pd.to_numeric(
                df["Precipitation"], errors="coerce"
            ).fillna(0.0),
            "cleaning_pulse": masks["cleaning_pulse"],
        }
    )

    report_days = pd.date_range("2021-08-09", "2023-08-08", freq="D")
    rows = []

    for day in report_days:
        ds = str(day.date())
        g = work[work["date"].eq(ds)]
        gv = g[g["valid"]].copy()

        ratio = pd.to_numeric(gv["ratio_corr"], errors="coerce").to_numpy(dtype=float)
        ratio = ratio[np.isfinite(ratio)]

        if ratio.size:
            med = float(np.median(ratio))
            mad = robust_mad(ratio)
            q25, q75 = np.quantile(ratio, [0.25, 0.75])
            iqr = float(q75 - q25)
        else:
            med = mad = iqr = np.nan

        rows.append(
            {
                "date": ds,
                "n_valid": int(ratio.size),
                "ratio_corr_median": med,
                "ratio_corr_mad": mad,
                "ratio_corr_iqr": iqr,
                "rain_mm_day": float(g["rain_mm"].sum()),
                "has_cleaning_pulse": bool(g["cleaning_pulse"].any()),
                "scheduled_maintenance": bool(
                    day.date() in SCHEDULED_MAINTENANCE_DATES
                ),
            }
        )

    return pd.DataFrame(rows)


def build_cycle_inventory(
    event_inventory_path: Path,
) -> pd.DataFrame:
    events = pd.read_csv(event_inventory_path, encoding="utf-8-sig")
    required = {"date", "first_pulse", "last_pulse", "pulse_count"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"event_inventory missing columns: {sorted(missing)}")

    events["date"] = events["date"].astype(str)
    events["first_pulse"] = pd.to_datetime(events["first_pulse"], errors="raise")
    events["last_pulse"] = pd.to_datetime(events["last_pulse"], errors="raise")

    in_period_boundaries = MODB_CLEANING_BOUNDARIES[1:]
    rows = []

    for j, start_date in enumerate(MODB_CLEANING_BOUNDARIES):
        start = pd.Timestamp(start_date)
        if j + 1 < len(MODB_CLEANING_BOUNDARIES):
            next_date = MODB_CLEANING_BOUNDARIES[j + 1]
            end_date = str(
                (pd.Timestamp(next_date) - pd.Timedelta(days=1)).date()
            )
        else:
            end_date = "2023-08-08"

        if j == 0:
            anchor_type = "PREPERIOD_TABLE12_ANCHOR"
            first_pulse = pd.NaT
            last_pulse = pd.NaT
            pulse_count = 0
            observed_cleaning = False
        else:
            match = events[events["date"].eq(start_date)]
            if len(match) != 1:
                raise RuntimeError(
                    f"Expected exactly one event_inventory row for ModB cleaning "
                    f"date {start_date}, found {len(match)}"
                )
            rr = match.iloc[0]
            anchor_type = "OBSERVED_TABLE12_MODB_CLEANING"
            first_pulse = rr["first_pulse"]
            last_pulse = rr["last_pulse"]
            pulse_count = int(rr["pulse_count"])
            observed_cleaning = True

        rows.append(
            {
                "cycle_id": j,
                "cycle_start_boundary": start_date,
                "cycle_report_start": (
                    "2021-08-09" if j == 0 else start_date
                ),
                "cycle_report_end": end_date,
                "anchor_type": anchor_type,
                "observed_cleaning_event": observed_cleaning,
                "cleaning_first_pulse": first_pulse,
                "cleaning_last_pulse": last_pulse,
                "cleaning_pulse_count": pulse_count,
            }
        )

    cycles = pd.DataFrame(rows)

    if len(cycles) != 27:
        raise RuntimeError(f"Expected 27 ModB cycles, found {len(cycles)}")
    if cycles["observed_cleaning_event"].sum() != 26:
        raise RuntimeError("Expected 26 observed in-period ModB cleaning events.")

    return cycles


def same_day_post_clean_candidate(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    cleaning_last_pulse: pd.Timestamp,
) -> dict:
    if pd.isna(cleaning_last_pulse):
        return {"n": 0, "baseline": np.nan, "date": None}

    day = pd.Timestamp(cleaning_last_pulse).normalize()
    noon = local_solar_noon(day)
    solar_lo = noon - pd.Timedelta(minutes=SOLAR_NOON_HALF_WINDOW_MIN)
    solar_hi = noon + pd.Timedelta(minutes=SOLAR_NOON_HALF_WINDOW_MIN)

    start = max(
        pd.Timestamp(cleaning_last_pulse)
        + pd.Timedelta(minutes=CLEANING_BUFFER_MIN),
        solar_lo,
    )
    end = solar_hi

    if start > end:
        return {"n": 0, "baseline": np.nan, "date": str(day.date())}

    ts = df["Timestamp"].to_numpy(dtype="datetime64[ns]")
    daymask = (ts >= np.datetime64(start)) & (ts <= np.datetime64(end))

    moda_raw = pd.to_numeric(df["ModA"], errors="coerce").to_numpy(dtype=float)
    moda_corr = pd.to_numeric(df["ModA_corr"], errors="coerce").to_numpy(dtype=float)
    modb = pd.to_numeric(df["ModB"], errors="coerce").to_numpy(dtype=float)

    # Reuse the frozen quality logic, but do not use cleaning_near because the
    # window itself already starts after last pulse +30 min.
    valid = (
        daymask
        & masks["finite_positive"]
        & (moda_raw > DAILY_RAW_MODA_THRESHOLD_WM2)
        & ~masks["rain_or_post"]
    )

    ratio = modb[valid] / moda_corr[valid]
    ratio = ratio[np.isfinite(ratio)]

    return {
        "n": int(ratio.size),
        "baseline": float(np.median(ratio)) if ratio.size else np.nan,
        "date": str(day.date()),
        "start": str(start),
        "end": str(end),
    }


def first_valid_subsequent_day_candidate(
    daily: pd.DataFrame,
    cycle_start: str,
    cycle_end: str,
    min_support: int,
    strictly_after_start: bool,
) -> dict:
    d = daily.copy()
    d["date_ts"] = pd.to_datetime(d["date"])
    start = pd.Timestamp(cycle_start)
    end = pd.Timestamp(cycle_end)

    if strictly_after_start:
        sel = d[
            (d["date_ts"] > start)
            & (d["date_ts"] <= end)
            & (d["n_valid"] >= min_support)
            & np.isfinite(d["ratio_corr_median"])
        ]
    else:
        sel = d[
            (d["date_ts"] >= start)
            & (d["date_ts"] <= end)
            & (d["n_valid"] >= min_support)
            & np.isfinite(d["ratio_corr_median"])
        ]

    if len(sel) == 0:
        return {"resolved": False}

    r = sel.iloc[0]
    return {
        "resolved": True,
        "date": str(r["date"]),
        "n": int(r["n_valid"]),
        "baseline": float(r["ratio_corr_median"]),
        "mad": float(r["ratio_corr_mad"]),
        "iqr": float(r["ratio_corr_iqr"]),
    }


def build_baseline_candidates(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    daily: pd.DataFrame,
    cycles: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for cyc in cycles.itertuples(index=False):
        for min_support in BASELINE_SUPPORT_GRID:
            if int(cyc.cycle_id) == 0:
                cand = first_valid_subsequent_day_candidate(
                    daily=daily,
                    cycle_start="2021-08-09",
                    cycle_end=cyc.cycle_report_end,
                    min_support=min_support,
                    strictly_after_start=False,
                )
                if cand["resolved"]:
                    rows.append(
                        {
                            "cycle_id": int(cyc.cycle_id),
                            "min_support": int(min_support),
                            "cycle_start_boundary": cyc.cycle_start_boundary,
                            "cycle_report_end": cyc.cycle_report_end,
                            "baseline_resolved": True,
                            "baseline_source": "PREPERIOD_ANCHOR_FIRST_VALID_DAY",
                            "baseline_date": cand["date"],
                            "baseline_n": cand["n"],
                            "baseline_value": cand["baseline"],
                            "baseline_mad": cand["mad"],
                            "baseline_iqr": cand["iqr"],
                            "same_day_postclean_n": np.nan,
                            "fallback_days_after_cleaning": (
                                pd.Timestamp(cand["date"])
                                - pd.Timestamp("2021-08-09")
                            ).days,
                        }
                    )
                else:
                    rows.append(
                        {
                            "cycle_id": int(cyc.cycle_id),
                            "min_support": int(min_support),
                            "cycle_start_boundary": cyc.cycle_start_boundary,
                            "cycle_report_end": cyc.cycle_report_end,
                            "baseline_resolved": False,
                            "baseline_source": "UNRESOLVED",
                        }
                    )
                continue

            same = same_day_post_clean_candidate(
                df=df,
                masks=masks,
                cleaning_last_pulse=pd.Timestamp(cyc.cleaning_last_pulse),
            )

            if same["n"] >= min_support and np.isfinite(same["baseline"]):
                rows.append(
                    {
                        "cycle_id": int(cyc.cycle_id),
                        "min_support": int(min_support),
                        "cycle_start_boundary": cyc.cycle_start_boundary,
                        "cycle_report_end": cyc.cycle_report_end,
                        "baseline_resolved": True,
                        "baseline_source": "SAME_DAY_POST_CLEAN",
                        "baseline_date": same["date"],
                        "baseline_n": same["n"],
                        "baseline_value": same["baseline"],
                        "baseline_mad": np.nan,
                        "baseline_iqr": np.nan,
                        "same_day_postclean_n": same["n"],
                        "same_day_postclean_start": same.get("start"),
                        "same_day_postclean_end": same.get("end"),
                        "fallback_days_after_cleaning": 0,
                    }
                )
            else:
                cand = first_valid_subsequent_day_candidate(
                    daily=daily,
                    cycle_start=cyc.cycle_report_start,
                    cycle_end=cyc.cycle_report_end,
                    min_support=min_support,
                    strictly_after_start=True,
                )

                if cand["resolved"]:
                    rows.append(
                        {
                            "cycle_id": int(cyc.cycle_id),
                            "min_support": int(min_support),
                            "cycle_start_boundary": cyc.cycle_start_boundary,
                            "cycle_report_end": cyc.cycle_report_end,
                            "baseline_resolved": True,
                            "baseline_source": "FIRST_SUBSEQUENT_VALID_DAY",
                            "baseline_date": cand["date"],
                            "baseline_n": cand["n"],
                            "baseline_value": cand["baseline"],
                            "baseline_mad": cand["mad"],
                            "baseline_iqr": cand["iqr"],
                            "same_day_postclean_n": same["n"],
                            "same_day_postclean_start": same.get("start"),
                            "same_day_postclean_end": same.get("end"),
                            "fallback_days_after_cleaning": (
                                pd.Timestamp(cand["date"])
                                - pd.Timestamp(cyc.cycle_start_boundary)
                            ).days,
                        }
                    )
                else:
                    rows.append(
                        {
                            "cycle_id": int(cyc.cycle_id),
                            "min_support": int(min_support),
                            "cycle_start_boundary": cyc.cycle_start_boundary,
                            "cycle_report_end": cyc.cycle_report_end,
                            "baseline_resolved": False,
                            "baseline_source": "UNRESOLVED",
                            "same_day_postclean_n": same["n"],
                        }
                    )

    return pd.DataFrame(rows)


def finite_quantiles(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
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


def make_summary(
    y1_path: Path,
    y2_path: Path,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    combined: pd.DataFrame,
    patch_info: dict,
    daily: pd.DataFrame,
    cycles: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict:
    by_support = []
    for support in BASELINE_SUPPORT_GRID:
        g = candidates[candidates["min_support"].eq(support)].copy()
        resolved = g[g["baseline_resolved"].fillna(False)].copy()
        by_support.append(
            {
                "min_support": int(support),
                "cycles": int(len(g)),
                "resolved_cycles": int(len(resolved)),
                "unresolved_cycles": int(len(g) - len(resolved)),
                "source_counts": {
                    str(k): int(v)
                    for k, v in resolved["baseline_source"].value_counts().to_dict().items()
                },
                "baseline_distribution": finite_quantiles(
                    resolved["baseline_value"]
                ),
                "fallback_days_distribution": finite_quantiles(
                    resolved["fallback_days_after_cleaning"]
                ),
            }
        )

    pivot = candidates.pivot(
        index="cycle_id",
        columns="min_support",
        values="baseline_value",
    )

    diffs = {}
    for other in [20, 60]:
        common = pivot[[PRIMARY_BASELINE_SUPPORT, other]].dropna()
        if len(common):
            diff = (common[PRIMARY_BASELINE_SUPPORT] - common[other]).abs()
            diffs[f"abs_diff_30_vs_{other}"] = finite_quantiles(diff)
            diffs[f"spearman_30_vs_{other}"] = float(
                common[PRIMARY_BASELINE_SUPPORT].corr(
                    common[other], method="spearman"
                )
            )

    primary = candidates[
        candidates["min_support"].eq(PRIMARY_BASELINE_SUPPORT)
    ].copy()

    return {
        "stage": "P2-0B-5.5a",
        "diagnostic_only": True,
        "baseline_frozen": False,
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
            "moda_corr_patch": patch_info,
        },
        "daily_filter": {
            "local_solar_noon_half_window_min": SOLAR_NOON_HALF_WINDOW_MIN,
            "raw_moda_threshold_wm2": DAILY_RAW_MODA_THRESHOLD_WM2,
            "cleaning_buffer_min": CLEANING_BUFFER_MIN,
            "post_rain_buffer_min": POST_RAIN_BUFFER_MIN,
            "eligibility_uses_raw_moda": True,
            "ratio_denominator": "ModA_corr",
            "days": int(len(daily)),
            "days_observed": int(
                (daily["n_valid"] > 0).sum()
            ),
            "coverage": float((daily["n_valid"] > 0).mean()),
            "n_valid_distribution": finite_quantiles(daily["n_valid"]),
            "ratio_corr_distribution_observed_days": finite_quantiles(
                daily.loc[daily["n_valid"] > 0, "ratio_corr_median"]
            ),
        },
        "cycle_inventory": {
            "cycles": int(len(cycles)),
            "preperiod_anchor_cycles": int(
                (cycles["anchor_type"] == "PREPERIOD_TABLE12_ANCHOR").sum()
            ),
            "observed_modb_cleaning_cycles": int(
                cycles["observed_cleaning_event"].sum()
            ),
            "first_boundary": MODB_CLEANING_BOUNDARIES[0],
            "last_boundary": MODB_CLEANING_BOUNDARIES[-1],
            "rain_creates_cycle": False,
        },
        "baseline_support_audit": by_support,
        "support_sensitivity": diffs,
        "primary_30min_candidate": {
            "resolved_cycles": int(
                primary["baseline_resolved"].fillna(False).sum()
            ),
            "unresolved_cycles": int(
                (~primary["baseline_resolved"].fillna(False)).sum()
            ),
            "source_counts": {
                str(k): int(v)
                for k, v in primary.loc[
                    primary["baseline_resolved"].fillna(False),
                    "baseline_source",
                ].value_counts().to_dict().items()
            },
        },
        "notes": [
            "2021-08-08 is treated only as a pre-period clean anchor for cycle 0.",
            "Only Table-12 ModB manual cleaning dates create new cycles; rain never resets baseline.",
            "Baseline candidates are earliest trustworthy post-clean observations, not maxima and not values chosen to be close to 1.",
            "Same-day baseline uses samples after the LAST Cleaning pulse +30 min and within the local solar-noon window.",
            "If same-day support is insufficient, the first subsequent valid day before the next ModB cleaning boundary is used.",
            "Support levels 20/30/60 are audited; 30 is a predeclared primary candidate but is not frozen in this stage.",
            "No C_d^WAPP or S_d^soil is computed here.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P2-0B-5.5a ModB cleaning-cycle and baseline-candidate audit."
    )
    p.add_argument("--year1", required=True, type=Path)
    p.add_argument("--year2", required=True, type=Path)
    p.add_argument("--corrected-segments", required=True, type=Path)
    p.add_argument("--event-inventory", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0b_5_5a_modb_cycle_baseline_audit_v1"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    y1_path = args.year1.expanduser().resolve()
    y2_path = args.year2.expanduser().resolve()
    corrected_segments_path = args.corrected_segments.expanduser().resolve()
    event_inventory_path = args.event_inventory.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for p in [
        y1_path,
        y2_path,
        corrected_segments_path,
        event_inventory_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    print("[1/9] Verify frozen input hashes")
    if sha256_file(y1_path) != EXPECTED_YEAR1_SHA256:
        raise RuntimeError("Year1 SHA256 mismatch.")
    if sha256_file(y2_path) != EXPECTED_YEAR2_SHA256:
        raise RuntimeError("Year2 SHA256 mismatch.")

    print("[2/9] Read Year1")
    y1 = read_wapp_csv(y1_path)

    print("[3/9] Read Year2")
    y2 = read_wapp_csv(y2_path)

    print("[4/9] Merge + continuity audit")
    combined = (
        pd.concat([y1, y2], ignore_index=True)
        .sort_values("Timestamp", kind="stable")
        .reset_index(drop=True)
    )
    validate_combined(combined)

    print("[5/9] Reconstruct full frozen ModA_corr")
    combined, patch_info = patch_frozen_moda_corr(
        combined, corrected_segments_path
    )

    print("[6/9] Build frozen daily corrected-ratio audit")
    masks = build_exclusion_masks(combined)
    daily = daily_corrected_ratio(combined, masks)

    print("[7/9] Build 27 ModB cleaning cycles")
    cycles = build_cycle_inventory(event_inventory_path)

    print("[8/9] Build baseline-support candidates (20/30/60)")
    candidates = build_baseline_candidates(
        combined,
        masks,
        daily,
        cycles,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    cycles.to_csv(
        out_dir / "cycle_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        out_dir / "daily_corrected_ratio_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidates.to_csv(
        out_dir / "baseline_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = make_summary(
        y1_path=y1_path,
        y2_path=y2_path,
        y1=y1,
        y2=y2,
        combined=combined,
        patch_info=patch_info,
        daily=daily,
        cycles=cycles,
        candidates=candidates,
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[9/9] Done")
    print(out_dir / "cycle_inventory.csv")
    print(out_dir / "daily_corrected_ratio_audit.csv")
    print(out_dir / "baseline_candidates.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: baseline audit only; B_j is NOT frozen and "
        "C_WAPP / S_soil / RL state are NOT generated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
