#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P2-0B-5.3: WAPP Malanville two-year daily raw-ratio audit.
Diagnostic only: no ModA correction, no final cleanliness, no S_soil, no RL.
"""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "Timestamp","GHI","DNI","DHI","ModA","ModB","Tamb","RH","WS","WSgust",
    "WSstdev","WD","WDstdev","BP","Cleaning","Precipitation","TModA","TModB","Comments"
]

LONGITUDE_DEG_E = 3.37352
TIMEZONE_UTC_HOURS = 1.0
SOLAR_NOON_HALF_WINDOW_MIN = 120
MODA_THRESHOLD_WM2 = 200.0
CLEANING_BUFFER_MIN = 30
POST_RAIN_BUFFER_MIN = 30

SCHEDULED_MAINTENANCE_DATES = {
    pd.Timestamp("2022-05-19").date(),
    pd.Timestamp("2022-09-17").date(),
    pd.Timestamp("2023-03-07").date(),
}

MODB_CLEANING_DATES = [pd.Timestamp(x).date() for x in [
    "2021-08-08","2021-09-01","2021-10-01","2021-11-01","2021-11-09","2021-12-01",
    "2021-12-31","2022-02-01","2022-03-01","2022-04-01","2022-05-03","2022-05-31",
    "2022-06-03","2022-07-01","2022-08-01","2022-09-01","2022-10-03","2022-11-01",
    "2022-11-30","2022-12-29","2023-02-01","2023-03-01","2023-04-04","2023-05-01",
    "2023-06-01","2023-07-01","2023-08-02"
]]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_wapp_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp1252", skiprows=[1], low_memory=False)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema in {path}\nFound: {list(df.columns)}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")
    for c in [x for x in EXPECTED_COLUMNS if x not in ("Timestamp","Comments")]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df

def eot_minutes(doy):
    gamma = 2*np.pi/365.0*(doy-1.0)
    return 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                   -0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))

def local_solar_noon_minutes(doy):
    return 720.0 - 4.0*LONGITUDE_DEG_E - eot_minutes(doy) + 60.0*TIMEZONE_UTC_HOURS

def robust_mad(x):
    a = np.asarray(x, dtype=float)
    m = np.nanmedian(a)
    return float(np.nanmedian(np.abs(a-m)))

def assign_cycle_id(d: date) -> int:
    idx = -1
    for i, cd in enumerate(MODB_CLEANING_DATES):
        if cd <= d:
            idx = i
        else:
            break
    return idx

def build_masks(df):
    out = df.copy()
    ts = out["Timestamp"]
    out["local_date"] = ts.dt.date

    minute = ts.dt.hour.to_numpy()*60 + ts.dt.minute.to_numpy()
    noon = local_solar_noon_minutes(ts.dt.dayofyear.to_numpy(dtype=float))
    dist = np.abs(minute-noon)
    dist = np.minimum(dist, 1440-dist)

    out["solar_noon_minute"] = noon
    out["solar_noon_distance_min"] = dist
    out["in_solar_noon_window"] = dist <= SOLAR_NOON_HALF_WINDOW_MIN

    out["valid_mod_signals"] = (
        np.isfinite(out["ModA"]) & np.isfinite(out["ModB"]) &
        (out["ModA"] > 0) & (out["ModB"] > 0)
    )
    out["passes_moda_threshold"] = out["ModA"] > MODA_THRESHOLD_WM2

    clean = (out["Cleaning"].fillna(0) > 0).astype(int)
    out["cleaning_pulse"] = clean.astype(bool)
    out["cleaning_buffer_excluded"] = (
        clean.rolling(2*CLEANING_BUFFER_MIN+1, center=True, min_periods=1).max().astype(bool)
    )

    rain = (out["Precipitation"].fillna(0) > 0).astype(int)
    out["rain_active"] = rain.astype(bool)
    out["rain_or_postrain_excluded"] = (
        rain.rolling(POST_RAIN_BUFFER_MIN+1, min_periods=1).max().astype(bool)
    )

    out["scheduled_maintenance_day"] = out["local_date"].isin(SCHEDULED_MAINTENANCE_DATES)
    out["modb_cleaning_day"] = out["local_date"].isin(MODB_CLEANING_DATES)
    out["cycle_id"] = [assign_cycle_id(d) for d in out["local_date"]]
    out["ratio_raw"] = out["ModB"] / out["ModA"]

    out["valid_for_daily_ratio"] = (
        out["in_solar_noon_window"] &
        out["valid_mod_signals"] &
        out["passes_moda_threshold"] &
        ~out["cleaning_buffer_excluded"] &
        ~out["rain_or_postrain_excluded"]
    )
    return out

def daily_summary(df):
    valid = df[df["valid_for_daily_ratio"]].copy()
    all_days = pd.date_range(df["Timestamp"].min().normalize(),
                             df["Timestamp"].max().normalize(), freq="D")
    base = pd.DataFrame({"date": all_days.date})

    whole = df.groupby("local_date").agg(
        rain_mm=("Precipitation","sum"),
        cleaning_pulse_count=("cleaning_pulse","sum"),
        maintenance_visit_day=("cleaning_pulse","max"),
        scheduled_maintenance=("scheduled_maintenance_day","max"),
        modb_cleaning=("modb_cleaning_day","max"),
        cycle_id=("cycle_id","first"),
        n_total=("Timestamp","size"),
        n_solar_window=("in_solar_noon_window","sum"),
        n_cleaning_excluded=("cleaning_buffer_excluded","sum"),
        n_rain_excluded=("rain_or_postrain_excluded","sum"),
    ).reset_index().rename(columns={"local_date":"date"})

    stats = valid.groupby("local_date")["ratio_raw"].agg(
        n_valid="size",
        ratio_median_raw="median",
        ratio_mean_raw="mean",
        ratio_std_raw="std",
        ratio_q05=lambda x: x.quantile(0.05),
        ratio_q25=lambda x: x.quantile(0.25),
        ratio_q75=lambda x: x.quantile(0.75),
        ratio_q95=lambda x: x.quantile(0.95),
        ratio_min_raw="min",
        ratio_max_raw="max",
    ).reset_index().rename(columns={"local_date":"date"})
    mad = (
        valid.groupby("local_date")["ratio_raw"]
        .apply(robust_mad)
        .reset_index(name="ratio_mad_raw")
        .rename(columns={"local_date": "date"})
    )
    stats = stats.merge(mad, on="date", how="left")
    stats["ratio_iqr_raw"] = stats["ratio_q75"] - stats["ratio_q25"]

    out = base.merge(whole, on="date", how="left").merge(stats, on="date", how="left")
    out["is_observed"] = out["n_valid"].fillna(0) > 0

    def qflag(r):
        if bool(r.get("scheduled_maintenance", False)): return "SCHEDULED_MAINTENANCE"
        if pd.isna(r.get("n_valid")) or r.get("n_valid",0) == 0: return "NO_VALID_RATIO"
        if r["n_valid"] < 30: return "LOW_SUPPORT"
        return "OK"
    out["quality_flag"] = out.apply(qflag, axis=1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year1", required=True, type=Path)
    ap.add_argument("--year2", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/paper2_uncertainty_rl_v1/p2_0b_5_3_daily_ratio_audit_v1"))
    args = ap.parse_args()

    y1p, y2p = args.year1.resolve(), args.year2.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Read Year1")
    y1 = read_wapp_csv(y1p)
    print("[2/6] Read Year2")
    y2 = read_wapp_csv(y2p)

    print("[3/6] Merge + continuity audit")
    df = pd.concat([y1,y2], ignore_index=True).sort_values("Timestamp").reset_index(drop=True)
    if df["Timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps found.")
    diffs = df["Timestamp"].diff().dropna()
    if (diffs != pd.Timedelta(minutes=1)).any():
        raise RuntimeError("Combined data is not strictly 1-minute continuous.")

    print("[4/6] Build frozen v1 masks")
    df = build_masks(df)
    print("[5/6] Daily audit")
    daily = daily_summary(df)

    report_days = daily[(daily["date"] >= pd.Timestamp("2021-08-09").date()) &
                        (daily["date"] <= pd.Timestamp("2023-08-08").date())]

    ratio = df.loc[df["valid_for_daily_ratio"], "ratio_raw"]
    summary = {
        "stage": "P2-0B-5.3",
        "diagnostic_only": True,
        "moda_correction_applied": False,
        "input": {
            "year1_path": str(y1p),
            "year1_sha256": sha256_file(y1p),
            "year1_rows": int(len(y1)),
            "year2_path": str(y2p),
            "year2_sha256": sha256_file(y2p),
            "year2_rows": int(len(y2)),
        },
        "combined": {
            "rows": int(len(df)),
            "start": str(df["Timestamp"].iloc[0]),
            "end": str(df["Timestamp"].iloc[-1]),
            "duplicate_timestamps": int(df["Timestamp"].duplicated().sum()),
            "non_1min_steps": int((diffs != pd.Timedelta(minutes=1)).sum()),
        },
        "filter_v1": {
            "solar_noon_half_window_min": 120,
            "moda_threshold_wm2": 200,
            "cleaning_buffer_min": 30,
            "post_rain_buffer_min": 30,
            "scheduled_maintenance_dates": sorted(str(x) for x in SCHEDULED_MAINTENANCE_DATES),
        },
        "minute_counts": {
            "cleaning_pulses": int(df["cleaning_pulse"].sum()),
            "rain_active": int(df["rain_active"].sum()),
            "valid_for_daily_ratio": int(df["valid_for_daily_ratio"].sum()),
        },
        "raw_ratio": {
            "n": int(ratio.size),
            "median": float(ratio.median()),
            "q01": float(ratio.quantile(0.01)),
            "q99": float(ratio.quantile(0.99)),
            "min": float(ratio.min()),
            "max": float(ratio.max()),
            "fraction_gt_1": float((ratio > 1).mean()),
        },
        "report_period": {
            "days": int(len(report_days)),
            "days_observed": int(report_days["is_observed"].sum()),
            "coverage": float(report_days["is_observed"].mean()),
            "quality_flag_counts": {str(k): int(v) for k,v in report_days["quality_flag"].value_counts().items()},
        },
        "notes": [
            "ratio_raw = ModB/ModA before ModA correction",
            "do not use ratio_median_raw as final cleanliness or RL state",
            "no clipping is applied",
            "scheduled-maintenance days are retained for diagnostic visibility and flagged; they must not drive later soiling-model parameter fitting",
            "no fixed high-MAD rejection threshold is imposed at this stage; MAD/IQR are exported for audit first",
        ],
    }

    daily.to_csv(outdir/"daily_raw_ratio_audit.csv", index=False, encoding="utf-8-sig")
    with (outdir/"audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[6/6] Done")
    print(outdir/"daily_raw_ratio_audit.csv")
    print(outdir/"audit_summary.json")
    print("IMPORTANT: PRE-ModA-correction diagnostic only.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
