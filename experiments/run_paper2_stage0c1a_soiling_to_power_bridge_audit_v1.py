#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0C-1A
WAPP soiling-state -> relative-power-loss semantic bridge audit.

DIAGNOSTIC ONLY. This stage does NOT freeze L_power and does NOT call the
Paper1 perception emulator.

Primary candidate:
    S_phys = clip(S_obs, 0, 1)
    gamma_P = -0.0043 / degC
    deltaT_bias = median_clean_days(TModB - TModA)
    TModB_star = TModB - deltaT_bias

    f_A = 1 + gamma_P * (TModA - 25)
    f_B = 1 + gamma_P * (TModB_star - 25)

    L_power_candidate = 1 - (1 - S_phys) * f_B / f_A

The common irradiance and module nominal power cancel in the clean/soiled
relative-power ratio. The script therefore does not reinterpret WAPP ModA/ModB
signals as absolute irradiance measurements.

Frozen upstream daily sample eligibility is reproduced:
- local solar noon +/-120 min
- raw ModA > 200 W/m2
- finite positive raw ModA/ModB
- Cleaning pulse +/-30 min excluded
- active precipitation +30 min post-rain excluded

Outputs:
- daily_power_bridge_audit.csv
- audit_summary.json
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
    "Timestamp", "GHI", "DNI", "DHI", "ModA", "ModB", "Tamb", "RH",
    "WS", "WSgust", "WSstdev", "WD", "WDstdev", "BP", "Cleaning",
    "Precipitation", "TModA", "TModB", "Comments",
]

YEAR1_SHA256 = "7f15922f01de97eb6a8b1477f0357e1dd3460c2918a64f7f007622a08063bed3"
YEAR2_SHA256 = "d85310c0a722184502714845abec945f64c854529b2470c57f3303447eb4fc52"

LONGITUDE_DEG_EAST = 3.3735
UTC_OFFSET_HOURS = 1.0

SOLAR_NOON_HALF_WINDOW_MIN = 120
RAW_MODA_THRESHOLD_WM2 = 200.0
CLEANING_BUFFER_MIN = 30
POST_RAIN_BUFFER_MIN = 30

# Phaesun Sun Plus 30 S datasheet: Pmax temperature coefficient -0.43 %/degC.
GAMMA_PMAX_PER_C = -0.0043

EXPECTED_DAYS = 730
MIN_TEMP_DAY_COVERAGE = 0.95


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wapp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp1252", skiprows=[1], low_memory=False)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected WAPP schema in {path}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")
    for col in EXPECTED_COLUMNS:
        if col not in ("Timestamp", "Comments"):
            df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def validate_continuity(df: pd.DataFrame) -> None:
    if df["Timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps in combined WAPP data.")
    diffs = df["Timestamp"].diff().dropna()
    bad = diffs != pd.Timedelta(minutes=1)
    if bad.any():
        raise RuntimeError(f"Non-1-minute steps: {int(bad.sum())}")


def equation_of_time_minutes(day_of_year: int) -> float:
    b = 2.0 * math.pi * (day_of_year - 81.0) / 364.0
    return 9.87 * math.sin(2*b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def local_solar_noon(day: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(day).normalize()
    eot = equation_of_time_minutes(int(day.dayofyear))
    local_standard_meridian = 15.0 * UTC_OFFSET_HOURS
    tc = 4.0 * (LONGITUDE_DEG_EAST - local_standard_meridian) + eot
    return day + pd.Timedelta(hours=12) - pd.Timedelta(minutes=tc)


def build_frozen_daily_mask(df: pd.DataFrame) -> np.ndarray:
    ts = df["Timestamp"]
    n = len(df)

    cleaning = pd.to_numeric(
        df["Cleaning"], errors="coerce"
    ).fillna(0).to_numpy() > 0
    rain = pd.to_numeric(
        df["Precipitation"], errors="coerce"
    ).fillna(0).to_numpy() > 0

    clean_series = pd.Series(cleaning.astype(np.int8), index=ts)
    cleaning_near = (
        clean_series.rolling(
            f"{2*CLEANING_BUFFER_MIN+1}min",
            center=True,
            min_periods=1,
        ).max().to_numpy() > 0
    )

    rain_series = pd.Series(rain.astype(np.int8), index=ts)
    rain_or_post = (
        rain_series.rolling(
            f"{POST_RAIN_BUFFER_MIN+1}min",
            min_periods=1,
        ).max().to_numpy() > 0
    )

    dates = ts.dt.normalize()
    solar_window = np.zeros(n, dtype=bool)
    ts_np = ts.to_numpy(dtype="datetime64[ns]")

    for day in pd.Index(dates.unique()).sort_values():
        noon = local_solar_noon(day)
        lo = np.datetime64(noon - pd.Timedelta(minutes=SOLAR_NOON_HALF_WINDOW_MIN))
        hi = np.datetime64(noon + pd.Timedelta(minutes=SOLAR_NOON_HALF_WINDOW_MIN))
        day_mask = dates.eq(day).to_numpy()
        solar_window[day_mask] = (
            (ts_np[day_mask] >= lo) & (ts_np[day_mask] <= hi)
        )

    moda = pd.to_numeric(df["ModA"], errors="coerce").to_numpy(dtype=float)
    modb = pd.to_numeric(df["ModB"], errors="coerce").to_numpy(dtype=float)

    finite_positive = (
        np.isfinite(moda)
        & np.isfinite(modb)
        & (moda > 0)
        & (modb > 0)
    )

    return (
        solar_window
        & finite_positive
        & (moda > RAW_MODA_THRESHOLD_WM2)
        & ~cleaning_near
        & ~rain_or_post
    )


def daily_temperatures(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    work = pd.DataFrame({
        "date": df["Timestamp"].dt.strftime("%Y-%m-%d"),
        "eligible_ratio_mask": mask,
        "TModA": pd.to_numeric(df["TModA"], errors="coerce"),
        "TModB": pd.to_numeric(df["TModB"], errors="coerce"),
    })

    rows = []
    for day in pd.date_range("2021-08-09", "2023-08-08", freq="D"):
        ds = str(day.date())
        g = work[work["date"].eq(ds)]
        base = g[g["eligible_ratio_mask"]]
        ta = base["TModA"].to_numpy(dtype=float)
        tb = base["TModB"].to_numpy(dtype=float)
        ok = np.isfinite(ta) & np.isfinite(tb)
        ta = ta[ok]
        tb = tb[ok]
        rows.append({
            "date": ds,
            "n_ratio_mask": int(len(base)),
            "n_temp_pair": int(ok.sum()),
            "TModA_median_C": float(np.median(ta)) if len(ta) else np.nan,
            "TModB_median_C": float(np.median(tb)) if len(tb) else np.nan,
            "deltaT_B_minus_A_C": (
                float(np.median(tb) - np.median(ta)) if len(ta) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def qstats(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.5]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "max": float(x.max()),
    }


def load_cleanliness(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "S_soil", "C_WAPP", "n_valid_daily_ratio",
        "modb_manual_cleaning_day", "observational_clipping_applied",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing 5.5b columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if len(df) != EXPECTED_DAYS:
        raise RuntimeError(f"Expected {EXPECTED_DAYS} days, found {len(df)}")
    if df["date"].duplicated().any():
        raise RuntimeError("Duplicate date in 5.5b reconstruction.")
    if df["observational_clipping_applied"].fillna(False).astype(bool).any():
        raise RuntimeError("Upstream observational clipping detected.")
    return df


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "P2-0C-1A semantic bridge audit: "
            "WAPP S_soil -> relative power loss."
        )
    )
    p.add_argument("--year1", required=True, type=Path)
    p.add_argument("--year2", required=True, type=Path)
    p.add_argument("--daily-cleanliness", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0c_1a_soiling_to_power_bridge_audit_v1"
        ),
    )
    args = p.parse_args()

    year1 = args.year1.expanduser().resolve()
    year2 = args.year2.expanduser().resolve()
    clean_path = args.daily_cleanliness.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for path in (year1, year2, clean_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/8] Verify frozen WAPP hashes")
    if sha256_file(year1) != YEAR1_SHA256:
        raise RuntimeError("Year1 SHA256 mismatch")
    if sha256_file(year2) != YEAR2_SHA256:
        raise RuntimeError("Year2 SHA256 mismatch")

    print("[2/8] Read + merge two-year WAPP data")
    y1 = read_wapp(year1)
    y2 = read_wapp(year2)
    raw = pd.concat([y1, y2], ignore_index=True).sort_values(
        "Timestamp", kind="stable"
    ).reset_index(drop=True)
    validate_continuity(raw)

    print("[3/8] Reproduce frozen daily eligibility mask")
    frozen_mask = build_frozen_daily_mask(raw)

    print("[4/8] Aggregate daily paired module temperatures")
    temp = daily_temperatures(raw, frozen_mask)

    print("[5/8] Load + align frozen 5.5b soiling reconstruction")
    clean = load_cleanliness(clean_path)
    merged = clean.merge(temp, on="date", how="left", validate="one_to_one")

    n1 = pd.to_numeric(
        merged["n_valid_daily_ratio"], errors="raise"
    ).to_numpy()
    n2 = pd.to_numeric(
        merged["n_ratio_mask"], errors="raise"
    ).to_numpy()
    if not np.array_equal(n1, n2):
        raise RuntimeError(
            f"Frozen daily-mask N mismatch on {int((n1 != n2).sum())} days."
        )

    print("[6/8] Estimate independent clean-day temperature-channel offset")
    clean_days = merged[
        merged["modb_manual_cleaning_day"].fillna(False).astype(bool)
    ].copy()
    clean_delta = pd.to_numeric(
        clean_days["deltaT_B_minus_A_C"], errors="coerce"
    )
    clean_delta = clean_delta[np.isfinite(clean_delta)]
    if len(clean_delta) != 26:
        raise RuntimeError(
            f"Expected 26 finite clean-day temperature pairs, found {len(clean_delta)}"
        )
    deltaT_bias = float(np.median(clean_delta))

    print("[7/8] Compute bridge candidates without L clipping")
    s_obs = pd.to_numeric(
        merged["S_soil"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_s = np.isfinite(s_obs)

    s_phys = np.full(len(merged), np.nan, dtype=float)
    s_phys[finite_s] = np.clip(s_obs[finite_s], 0.0, 1.0)

    ta = pd.to_numeric(
        merged["TModA_median_C"], errors="coerce"
    ).to_numpy(dtype=float)
    tb = pd.to_numeric(
        merged["TModB_median_C"], errors="coerce"
    ).to_numpy(dtype=float)
    tb_star = tb - deltaT_bias

    fA = 1.0 + GAMMA_PMAX_PER_C * (ta - 25.0)
    fB_raw = 1.0 + GAMMA_PMAX_PER_C * (tb - 25.0)
    fB_star = 1.0 + GAMMA_PMAX_PER_C * (tb_star - 25.0)

    valid_bridge = (
        np.isfinite(s_phys)
        & np.isfinite(fA)
        & np.isfinite(fB_star)
        & (fA > 0)
        & (fB_star > 0)
    )

    l_identity = np.full(len(merged), np.nan)
    l_temp_raw = np.full(len(merged), np.nan)
    l_temp_biascorr = np.full(len(merged), np.nan)

    l_identity[valid_bridge] = s_phys[valid_bridge]
    l_temp_raw[valid_bridge] = (
        1.0
        - (1.0 - s_phys[valid_bridge])
        * fB_raw[valid_bridge] / fA[valid_bridge]
    )
    l_temp_biascorr[valid_bridge] = (
        1.0
        - (1.0 - s_phys[valid_bridge])
        * fB_star[valid_bridge] / fA[valid_bridge]
    )

    merged["S_soil_observed"] = s_obs
    merged["S_soil_physical_projected"] = s_phys
    merged["S_projection_lower_applied"] = finite_s & (s_obs < 0)
    merged["S_projection_upper_applied"] = finite_s & (s_obs > 1)
    merged["deltaT_clean_bias_C"] = deltaT_bias
    merged["TModB_bias_corrected_C"] = tb_star
    merged["power_temp_factor_A"] = fA
    merged["power_temp_factor_B_raw"] = fB_raw
    merged["power_temp_factor_B_biascorr"] = fB_star
    merged["L_identity_Sphys"] = l_identity
    merged["L_temp_raw_candidate"] = l_temp_raw
    merged["L_power_candidate"] = l_temp_biascorr
    merged["bridge_valid"] = valid_bridge

    bridge_coverage = float(valid_bridge.mean())
    if bridge_coverage < MIN_TEMP_DAY_COVERAGE:
        raise RuntimeError(
            f"Bridge day coverage {bridge_coverage:.6f} < {MIN_TEMP_DAY_COVERAGE:.2f}"
        )

    clean_mask = merged[
        "modb_manual_cleaning_day"
    ].fillna(False).astype(bool).to_numpy()
    abs_diff = np.abs(l_temp_biascorr - l_identity)

    summary = {
        "stage": "P2-0C-1A",
        "diagnostic_only": True,
        "power_loss_bridge_frozen": False,
        "paper1_emulator_called": False,
        "rl_state_generated": False,
        "semantic_boundary": {
            "wapp_state": "relative soiling / effective-irradiance attenuation",
            "paper1_target": "DeepSolarEye relative electrical power loss L",
            "direct_identity_assumed": False,
        },
        "inputs": {
            "year1_sha256": sha256_file(year1),
            "year2_sha256": sha256_file(year2),
            "days": int(len(merged)),
        },
        "temperature_audit": {
            "valid_bridge_days": int(valid_bridge.sum()),
            "bridge_day_coverage": bridge_coverage,
            "clean_days_used_for_offset": int(clean_mask.sum()),
            "raw_deltaT_all_days_C": qstats(
                merged["deltaT_B_minus_A_C"]
            ),
            "raw_deltaT_clean_days_C": qstats(
                merged.loc[clean_mask, "deltaT_B_minus_A_C"]
            ),
            "estimated_clean_day_deltaT_bias_C": deltaT_bias,
        },
        "physical_soiling_projection": {
            "observational_state_preserved": True,
            "projection_used_only_for_power_bridge": True,
            "lower_projection_days": int(
                (finite_s & (s_obs < 0)).sum()
            ),
            "upper_projection_days": int(
                (finite_s & (s_obs > 1)).sum()
            ),
            "S_observed_distribution": qstats(s_obs),
            "S_physical_distribution": qstats(s_phys),
        },
        "power_bridge": {
            "module": "Phaesun Sun Plus 30 S, 30 W",
            "gamma_Pmax_per_C": GAMMA_PMAX_PER_C,
            "formula": (
                "L=1-(1-S_phys)*(1+gamma*(TModB_star-25))/"
                "(1+gamma*(TModA-25))"
            ),
            "absolute_irradiance_required": False,
            "nominal_power_cancels_in_ratio": True,
            "temperature_factor_A": qstats(fA[valid_bridge]),
            "temperature_factor_B_biascorr": qstats(
                fB_star[valid_bridge]
            ),
            "L_identity_distribution": qstats(
                l_identity[valid_bridge]
            ),
            "L_temp_raw_distribution": qstats(
                l_temp_raw[valid_bridge]
            ),
            "L_power_candidate_distribution": qstats(
                l_temp_biascorr[valid_bridge]
            ),
            "abs_difference_candidate_vs_identity": qstats(
                abs_diff[valid_bridge]
            ),
            "candidate_negative_days": int(
                (l_temp_biascorr[valid_bridge] < 0).sum()
            ),
            "candidate_gt1_days": int(
                (l_temp_biascorr[valid_bridge] > 1).sum()
            ),
            "candidate_clean_day_distribution": qstats(
                l_temp_biascorr[clean_mask & valid_bridge]
            ),
        },
        "gates": {
            "bridge_coverage_gate": MIN_TEMP_DAY_COVERAGE,
            "bridge_coverage_pass": bool(
                bridge_coverage >= MIN_TEMP_DAY_COVERAGE
            ),
            "all_temperature_factors_positive": bool(
                np.all(fA[valid_bridge] > 0)
                and np.all(fB_star[valid_bridge] > 0)
            ),
        },
        "notes": [
            "This stage does not claim measured electrical power at Malanville.",
            "The bridge is a physically motivated relative-power proxy aligned to DeepSolarEye label semantics.",
            "Negative observational S_soil values are preserved and projected to zero only for the physical bridge candidate.",
            "L_power_candidate is intentionally not clipped in this diagnostic stage.",
            "Paper1 support-domain / nearest-neighbor audit is deferred to P2-0C-2.",
        ],
    }

    print("[8/8] Write audit outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(
        out_dir / "daily_power_bridge_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(out_dir / "daily_power_bridge_audit.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: diagnostic only; do NOT freeze L_power and do NOT call "
        "the Paper1 emulator until this audit is reviewed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
