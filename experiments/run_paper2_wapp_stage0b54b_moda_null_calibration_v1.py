#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0B-5.4b
WAPP Malanville placebo-event null calibration for ModA cleaning jumps.

Purpose
-------
This stage calibrates the natural/false short-window jump distribution of

    Q = ModA / ModB

using dates with NO Cleaning event, while preserving the clock time and event
span of each real, metadata-clean ModA cleaning event.

It is DIAGNOSTIC ONLY:
- NO ModA correction is applied.
- NO final gain threshold is frozen.
- NO final cleanliness / soiling state is generated.
- NO RL state is generated.

Frozen primary event-audit configuration inherited from P2-0B-5.4a
-------------------------------------------------------------------
- ModA > 100 W/m^2
- 30 min pre-window
- 30 min post-window
- 15 min gap from Cleaning pulse span
- n_pre >= 20 and n_post >= 20
- real-event pulse_span < 30 min
- exclude ModB cleaning dates
- exclude rain-near-event dates
- exclude scheduled-maintenance dates directly by calendar date

Placebo matching
----------------
For each eligible real event:
1) preserve its first-pulse local clock time and exact pulse span;
2) search no-Cleaning control dates within +/-45 calendar days;
3) directly exclude all scheduled-maintenance and official ModB-cleaning dates;
4) require zero precipitation within +/-3 h of the pseudo event;
5) require the same 100 W/m^2 and n_pre/n_post >=20 support rule;
6) keep the 5 nearest valid control dates.

Two null distributions are exported:
- nearest-one null: match_rank == 1 (one placebo per real event);
- pooled null: up to five nearest placebos per real event.

This avoids choosing 0.5%, 1%, or 2% before seeing the empirical false-jump
distribution. The script reports P90/P95/P97.5/P99 null quantiles and how many
real events would exceed each candidate quantile, but DOES NOT select one.

Inputs
------
Official WAPP Malanville Year1 / Year2 QC CSVs, plus the frozen outputs from
P2-0B-5.4a:
- event_inventory.csv
- prepost_sensitivity.csv

Outputs
-------
real_primary_events.csv
null_matches.csv
threshold_audit.csv
audit_summary.json
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

# Frozen primary configuration from P2-0B-5.4a.
PRIMARY_IRRADIANCE_THRESHOLD_WM2 = 100.0
PRIMARY_WINDOW_MIN = 30
EVENT_BUFFER_MIN = 15
MIN_SUPPORT = 20
MAX_LOCAL_EVENT_SPAN_MIN = 30.0  # strict: pulse_span_min < 30
RAIN_NEAR_EVENT_HOURS = 3

# Null-matching design.
MATCH_RADIUS_DAYS = 45
MAX_NULL_MATCHES_PER_REAL_EVENT = 5

SCHEDULED_MAINTENANCE_DATES = {
    pd.Timestamp("2022-05-19").date(),
    pd.Timestamp("2022-09-17").date(),
    pd.Timestamp("2023-03-07").date(),
}

# Official Table-12 dates. 2021-08-08 is pre-period provenance/boundary only.
MODB_CLEANING_DATES = {
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
}


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


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def finite_quantiles(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return {}
    probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
    q = x.quantile(probs)
    out = {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
    }
    for p, v in q.items():
        label = "q97_5" if abs(p - 0.975) < 1e-12 else f"q{int(round(p * 100)):02d}"
        out[label] = float(v)
    out["max"] = float(x.max())
    out["fraction_positive"] = float((x > 0).mean())
    return out


def compute_window_stats(
    indexed: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    irradiance_threshold: float,
) -> dict:
    try:
        w = indexed.loc[start:end]
    except KeyError:
        w = indexed[(indexed.index >= start) & (indexed.index <= end)]

    if w.empty:
        return {
            "n": 0,
            "q_median": np.nan,
            "q_mad": np.nan,
            "ghi_median": np.nan,
            "moda_median": np.nan,
            "modb_median": np.nan,
        }

    moda = w["ModA"].to_numpy(dtype=float)
    modb = w["ModB"].to_numpy(dtype=float)
    precip = w["Precipitation"].fillna(0).to_numpy(dtype=float)

    valid = (
        np.isfinite(moda)
        & np.isfinite(modb)
        & (moda > irradiance_threshold)
        & (modb > 0.0)
        & (precip <= 0.0)
    )
    w = w.loc[valid]

    if w.empty:
        return {
            "n": 0,
            "q_median": np.nan,
            "q_mad": np.nan,
            "ghi_median": np.nan,
            "moda_median": np.nan,
            "modb_median": np.nan,
        }

    q = (w["ModA"] / w["ModB"]).astype(float)
    med = float(q.median())
    mad = float(np.median(np.abs(q.to_numpy(dtype=float) - med)))

    return {
        "n": int(len(w)),
        "q_median": med,
        "q_mad": mad,
        "ghi_median": float(w["GHI"].median()),
        "moda_median": float(w["ModA"].median()),
        "modb_median": float(w["ModB"].median()),
    }


def rain_near_pseudo_event_mm(
    indexed: pd.DataFrame,
    first_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
) -> float:
    start = first_ts - pd.Timedelta(hours=RAIN_NEAR_EVENT_HOURS)
    end = last_ts + pd.Timedelta(hours=RAIN_NEAR_EVENT_HOURS)
    try:
        w = indexed.loc[start:end]
    except KeyError:
        w = indexed[(indexed.index >= start) & (indexed.index <= end)]
    if w.empty:
        return 0.0
    return float(w["Precipitation"].fillna(0).sum())


def reconstruct_primary_real_events(
    event_inventory: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    e = event_inventory.copy()
    s = sensitivity.copy()

    for col in [
        "eligible_metadata_only",
        "is_modb_cleaning_date",
        "is_scheduled_maintenance",
        "rain_near_event",
    ]:
        if col in e.columns:
            e[col] = as_bool(e[col])
        if col in s.columns:
            s[col] = as_bool(s[col])

    primary = s[
        np.isclose(
            pd.to_numeric(s["irradiance_threshold_wm2"], errors="coerce"),
            PRIMARY_IRRADIANCE_THRESHOLD_WM2,
        )
        & (
            pd.to_numeric(s["window_min"], errors="coerce")
            == PRIMARY_WINDOW_MIN
        )
    ].copy()

    if len(primary) != len(e):
        raise RuntimeError(
            "P2-0B-5.4a primary sensitivity row count does not match "
            f"event inventory: primary={len(primary)}, events={len(e)}"
        )

    primary["date"] = primary["date"].astype(str)
    primary["date_obj"] = pd.to_datetime(primary["date"]).dt.date

    primary["eligible_direct_date"] = ~primary["date_obj"].isin(
        SCHEDULED_MAINTENANCE_DATES
    )

    keep = (
        as_bool(primary["eligible_metadata_only"])
        & primary["eligible_direct_date"]
        & (pd.to_numeric(primary["n_pre"], errors="coerce") >= MIN_SUPPORT)
        & (pd.to_numeric(primary["n_post"], errors="coerce") >= MIN_SUPPORT)
        & (
            pd.to_numeric(primary["pulse_span_min"], errors="coerce")
            < MAX_LOCAL_EVENT_SPAN_MIN
        )
        & np.isfinite(
            pd.to_numeric(primary["moda_soiling_est"], errors="coerce")
        )
    )
    real = primary.loc[keep].copy()

    real["first_pulse"] = pd.to_datetime(real["first_pulse"], errors="raise")
    real["last_pulse"] = pd.to_datetime(real["last_pulse"], errors="raise")

    wanted = [
        "event_id",
        "date",
        "first_pulse",
        "last_pulse",
        "pulse_count",
        "pulse_span_min",
        "n_pre",
        "n_post",
        "q_pre_median",
        "q_post_median",
        "q_pre_mad",
        "q_post_mad",
        "ghi_pre_median",
        "ghi_post_median",
        "moda_soiling_est",
    ]
    return real[wanted].sort_values("event_id").reset_index(drop=True)


def report_calendar_days(combined: pd.DataFrame) -> list[pd.Timestamp]:
    # The two-year report covers 2021-08-09 through 2023-08-08 inclusive.
    start_day = combined["Timestamp"].iloc[0].normalize()
    end_day = combined["Timestamp"].iloc[-1].normalize()
    if combined["Timestamp"].iloc[-1] == end_day:
        end_day = end_day - pd.Timedelta(days=1)
    return list(pd.date_range(start_day, end_day, freq="D"))


def build_control_days(combined: pd.DataFrame) -> list[pd.Timestamp]:
    cleaning_dates = set(
        combined.loc[
            combined["Cleaning"].fillna(0).to_numpy(dtype=float) > 0.0,
            "Timestamp",
        ].dt.date
    )

    control_days = []
    for day in report_calendar_days(combined):
        d = day.date()
        if d in cleaning_dates:
            continue
        if d in SCHEDULED_MAINTENANCE_DATES:
            continue
        if d in MODB_CLEANING_DATES:
            continue
        control_days.append(day)
    return control_days


def pseudo_event_for_day(
    real_row: pd.Series,
    control_day: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    real_first = pd.Timestamp(real_row["first_pulse"])
    real_last = pd.Timestamp(real_row["last_pulse"])

    tod = real_first - real_first.normalize()
    span = real_last - real_first

    pseudo_first = control_day.normalize() + tod
    pseudo_last = pseudo_first + span
    return pseudo_first, pseudo_last


def build_null_matches(
    indexed: pd.DataFrame,
    real_events: pd.DataFrame,
    control_days: list[pd.Timestamp],
) -> pd.DataFrame:
    rows = []

    for real in real_events.itertuples(index=False):
        real_date = pd.Timestamp(real.date).normalize()

        candidate_days = [
            d
            for d in control_days
            if 0 < abs((d - real_date).days) <= MATCH_RADIUS_DAYS
        ]
        candidate_days.sort(
            key=lambda d: (abs((d - real_date).days), d)
        )

        rank = 0
        for control_day in candidate_days:
            pseudo_first, pseudo_last = pseudo_event_for_day(
                pd.Series(real._asdict()), control_day
            )

            # Keep the complete pre/post construction inside one control date.
            pre_end = pseudo_first - pd.Timedelta(minutes=EVENT_BUFFER_MIN)
            pre_start = pre_end - pd.Timedelta(minutes=PRIMARY_WINDOW_MIN - 1)
            post_start = pseudo_last + pd.Timedelta(minutes=EVENT_BUFFER_MIN)
            post_end = post_start + pd.Timedelta(minutes=PRIMARY_WINDOW_MIN - 1)

            if not (
                pre_start.normalize()
                == pseudo_first.normalize()
                == pseudo_last.normalize()
                == post_end.normalize()
            ):
                continue

            rain_mm = rain_near_pseudo_event_mm(
                indexed, pseudo_first, pseudo_last
            )
            if rain_mm > 0:
                continue

            pre = compute_window_stats(
                indexed,
                pre_start,
                pre_end,
                PRIMARY_IRRADIANCE_THRESHOLD_WM2,
            )
            post = compute_window_stats(
                indexed,
                post_start,
                post_end,
                PRIMARY_IRRADIANCE_THRESHOLD_WM2,
            )

            if pre["n"] < MIN_SUPPORT or post["n"] < MIN_SUPPORT:
                continue
            if (
                not np.isfinite(pre["q_median"])
                or not np.isfinite(post["q_median"])
                or pre["q_median"] <= 0
                or post["q_median"] <= 0
            ):
                continue

            gain = 1.0 - pre["q_median"] / post["q_median"]

            rank += 1
            rows.append(
                {
                    "real_event_id": int(real.event_id),
                    "real_event_date": str(real.date),
                    "real_gain": float(real.moda_soiling_est),
                    "real_pulse_span_min": float(real.pulse_span_min),
                    "match_rank": rank,
                    "control_date": str(control_day.date()),
                    "calendar_gap_days": int(abs((control_day - real_date).days)),
                    "pseudo_first": pseudo_first,
                    "pseudo_last": pseudo_last,
                    "rain_near_pseudo_mm": rain_mm,
                    "n_pre": pre["n"],
                    "n_post": post["n"],
                    "q_pre_median": pre["q_median"],
                    "q_post_median": post["q_median"],
                    "q_pre_mad": pre["q_mad"],
                    "q_post_mad": post["q_mad"],
                    "ghi_pre_median": pre["ghi_median"],
                    "ghi_post_median": post["ghi_median"],
                    "null_gain": float(gain),
                }
            )

            if rank >= MAX_NULL_MATCHES_PER_REAL_EVENT:
                break

    return pd.DataFrame(rows)


def threshold_rows(
    real_events: pd.DataFrame,
    null_matches: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    null_sets = {
        "nearest_one": null_matches.loc[
            null_matches["match_rank"] == 1, "null_gain"
        ],
        "pooled_up_to_5": null_matches["null_gain"],
    }

    real_gain = pd.to_numeric(real_events["moda_soiling_est"], errors="coerce")
    real_gain = real_gain[np.isfinite(real_gain)]

    for null_name, null_s in null_sets.items():
        x = pd.to_numeric(null_s, errors="coerce")
        x = x[np.isfinite(x)]
        if x.empty:
            continue

        for q in [0.90, 0.95, 0.975, 0.99]:
            threshold = float(x.quantile(q))
            rows.append(
                {
                    "null_set": null_name,
                    "quantile": q,
                    "candidate_threshold": threshold,
                    "null_n": int(len(x)),
                    "real_n": int(len(real_gain)),
                    "real_events_exceeding": int((real_gain > threshold).sum()),
                    "real_fraction_exceeding": float(
                        (real_gain > threshold).mean()
                    ),
                    "official_2022_03_29_exceeds": bool(
                        (
                            real_events.loc[
                                real_events["date"].astype(str) == "2022-03-29",
                                "moda_soiling_est",
                            ]
                            > threshold
                        ).any()
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_summary(
    y1_path: Path,
    y2_path: Path,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    combined: pd.DataFrame,
    event_inventory: pd.DataFrame,
    real_events: pd.DataFrame,
    control_days: list[pd.Timestamp],
    null_matches: pd.DataFrame,
    threshold_audit: pd.DataFrame,
) -> dict:
    match_counts = (
        null_matches.groupby("real_event_id").size()
        if not null_matches.empty
        else pd.Series(dtype=int)
    )

    nearest = (
        null_matches.loc[null_matches["match_rank"] == 1]
        if not null_matches.empty
        else null_matches
    )

    official = real_events[
        real_events["date"].astype(str) == "2022-03-29"
    ]

    return {
        "stage": "P2-0B-5.4b",
        "diagnostic_only": True,
        "moda_correction_applied": False,
        "final_gain_threshold_frozen": False,
        "input": {
            "year1_path": str(y1_path),
            "year1_sha256": sha256_file(y1_path),
            "year1_rows": int(len(y1)),
            "year2_path": str(y2_path),
            "year2_sha256": sha256_file(y2_path),
            "year2_rows": int(len(y2)),
            "combined_rows": int(len(combined)),
            "event_inventory_rows": int(len(event_inventory)),
        },
        "primary_real_event_rule": {
            "irradiance_threshold_wm2": PRIMARY_IRRADIANCE_THRESHOLD_WM2,
            "prepost_window_min": PRIMARY_WINDOW_MIN,
            "event_buffer_min": EVENT_BUFFER_MIN,
            "min_pre_support": MIN_SUPPORT,
            "min_post_support": MIN_SUPPORT,
            "pulse_span_rule": f"< {MAX_LOCAL_EVENT_SPAN_MIN:g} min",
            "direct_scheduled_maintenance_exclusion": sorted(
                str(x) for x in SCHEDULED_MAINTENANCE_DATES
            ),
            "real_events_after_all_primary_rules": int(len(real_events)),
            "real_gain_distribution": finite_quantiles(
                real_events["moda_soiling_est"]
            ),
        },
        "null_matching": {
            "control_calendar_days": int(len(control_days)),
            "match_radius_days": MATCH_RADIUS_DAYS,
            "max_matches_per_real_event": MAX_NULL_MATCHES_PER_REAL_EVENT,
            "preserve_real_clock_time": True,
            "preserve_real_event_span": True,
            "rain_near_event_hours": RAIN_NEAR_EVENT_HOURS,
            "real_events_with_at_least_1_match": int((match_counts >= 1).sum()),
            "real_events_with_at_least_3_matches": int((match_counts >= 3).sum()),
            "real_events_with_5_matches": int((match_counts >= 5).sum()),
            "real_events_without_match": int(len(real_events) - (match_counts >= 1).sum()),
            "total_null_matches": int(len(null_matches)),
            "calendar_gap_days_distribution": finite_quantiles(
                null_matches["calendar_gap_days"]
                if not null_matches.empty
                else pd.Series(dtype=float)
            ),
        },
        "null_distributions": {
            "nearest_one": finite_quantiles(
                nearest["null_gain"]
                if not nearest.empty
                else pd.Series(dtype=float)
            ),
            "pooled_up_to_5": finite_quantiles(
                null_matches["null_gain"]
                if not null_matches.empty
                else pd.Series(dtype=float)
            ),
        },
        "official_example_2022_03_29": {
            "present_in_primary_real_events": bool(len(official) > 0),
            "moda_soiling_est": (
                float(official["moda_soiling_est"].iloc[0])
                if len(official) > 0
                else None
            ),
        },
        "threshold_audit": threshold_audit.to_dict(orient="records"),
        "notes": [
            "No fixed 0.5%, 1%, or 2% threshold is imposed.",
            "Nearest-one null uses one placebo per real event to reduce pseudo-replication.",
            "Pooled-up-to-5 null is retained as a sensitivity distribution.",
            "Pseudo events preserve the real event's clock time and pulse span.",
            "Control dates contain no Cleaning pulse and directly exclude scheduled-maintenance and Table-12 ModB-cleaning dates.",
            "All pseudo events require zero rain within +/-3 h and the same primary pre/post support rule.",
            "This stage reports candidate null quantiles only; final threshold selection is deferred to the next gate.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "P2-0B-5.4b placebo-event null calibration for ModA cleaning jumps."
        )
    )
    p.add_argument("--year1", required=True, type=Path)
    p.add_argument("--year2", required=True, type=Path)
    p.add_argument("--event-inventory", required=True, type=Path)
    p.add_argument("--prepost-sensitivity", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0b_5_4b_moda_null_calibration_v1"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    y1_path = args.year1.expanduser().resolve()
    y2_path = args.year2.expanduser().resolve()
    event_path = args.event_inventory.expanduser().resolve()
    sensitivity_path = args.prepost_sensitivity.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for p in [y1_path, y2_path, event_path, sensitivity_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Read Year1")
    y1 = read_wapp_csv(y1_path)
    print("[2/8] Read Year2")
    y2 = read_wapp_csv(y2_path)

    print("[3/8] Merge + continuity audit")
    combined = (
        pd.concat([y1, y2], ignore_index=True)
        .sort_values("Timestamp", kind="stable")
        .reset_index(drop=True)
    )
    validate_combined(combined)
    indexed = combined.set_index("Timestamp", drop=False)

    print("[4/8] Read frozen P2-0B-5.4a outputs")
    event_inventory = pd.read_csv(event_path, encoding="utf-8-sig")
    sensitivity = pd.read_csv(sensitivity_path, encoding="utf-8-sig")
    if len(event_inventory) != 504:
        raise RuntimeError(
            f"Expected 504 event-inventory rows, found {len(event_inventory)}"
        )

    print("[5/8] Reconstruct primary real-event set")
    real_events = reconstruct_primary_real_events(
        event_inventory, sensitivity
    )
    control_days = build_control_days(combined)

    print("[6/8] Build matched placebo-event null distribution")
    null_matches = build_null_matches(
        indexed, real_events, control_days
    )
    if null_matches.empty:
        raise RuntimeError("No valid placebo-event matches were found.")

    print("[7/8] Build threshold audit + write outputs")
    threshold_audit = threshold_rows(real_events, null_matches)

    real_events.to_csv(
        out_dir / "real_primary_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    null_matches.to_csv(
        out_dir / "null_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_audit.to_csv(
        out_dir / "threshold_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = make_summary(
        y1_path=y1_path,
        y2_path=y2_path,
        y1=y1,
        y2=y2,
        combined=combined,
        event_inventory=event_inventory,
        real_events=real_events,
        control_days=control_days,
        null_matches=null_matches,
        threshold_audit=threshold_audit,
    )
    with (out_dir / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[8/8] Done")
    print(out_dir / "real_primary_events.csv")
    print(out_dir / "null_matches.csv")
    print(out_dir / "threshold_audit.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: diagnostic only; no ModA correction and no final gain threshold."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
