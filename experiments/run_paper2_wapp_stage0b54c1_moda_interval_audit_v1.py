#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0B-5.4c-1
WAPP Malanville ModA backward-correction interval audit.

Purpose
-------
Construct and audit the time intervals that WOULD be eligible for ModA
backward linear correction after the P2-0B-5.4a/5.4b event and null-calibration
gates.

This stage is DIAGNOSTIC ONLY:
- it does NOT modify ModA;
- it does NOT generate ModA_corr;
- it does NOT generate final cleanliness / soiling state;
- it does NOT generate any RL state.

Frozen upstream rule
--------------------
A current ModA cleaning event can become a strong-correction candidate only if
it is already present in P2-0B-5.4b real_primary_events.csv and

    moda_soiling_est > P95(nearest-one placebo null)

with the frozen threshold

    0.009297729719041414  (~0.9298%).

The upstream real-primary-event set already enforces:
- non-ModB-cleaning date;
- non-scheduled-maintenance date;
- no rain near the cleaning event;
- ModA > 100 W/m^2;
- 30 min pre/post windows;
- 15 min event buffer;
- n_pre >= 20 and n_post >= 20;
- pulse_span < 30 min.

Scientific interval rule
------------------------
For each strong current event:
1) Find the immediately preceding OMT Cleaning visit from the 504-date event
   inventory. The official campaign procedure states that ModA is cleaned at
   every OMT visit, so that preceding visit is the physical reset anchor.
2) Define the candidate backward-correction interval from the LAST Cleaning
   pulse of the previous visit to the FIRST Cleaning pulse of the current visit.
3) Audit precipitation throughout the full interval. If rain occurs, automatic
   linear backward correction is NOT allowed in this stage, because the official
   report explicitly singles out rain / dust storms / other non-linear events.
4) Detect any scheduled-maintenance calendar date lying inside the interval
   without being represented by the previous Cleaning anchor. This is important
   for 2022-09-17, which is an official scheduled-maintenance date but had no
   Cleaning pulse in the event inventory.
5) Export wind / gust / diffuse-radiation diagnostics, but impose NO new
   post-hoc thresholds on them. They are review variables only.

No final correction formula is chosen here. In particular, whether later
ModA_corr is implemented as division by (1-s(t)) or another report-compatible
multiplicative convention is deferred to P2-0B-5.4c-2 and must be validated
before modifying the two-year signal.

Inputs
------
- Official WAPP Malanville Year1 and Year2 QC CSVs
- P2-0B-5.4a event_inventory.csv
- P2-0B-5.4b real_primary_events.csv
- P2-0B-5.4b threshold_audit.csv

Outputs
-------
- correction_interval_audit.csv
- audit_summary.json
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

FROZEN_THRESHOLD = 0.009297729719041414
FROZEN_NULL_SET = "nearest_one"
FROZEN_NULL_QUANTILE = 0.95

SCHEDULED_MAINTENANCE_DATES = {
    pd.Timestamp("2022-05-19").date(),
    pd.Timestamp("2022-09-17").date(),
    pd.Timestamp("2023-03-07").date(),
}

# Review-only diagnostics. No threshold is applied.
DIAGNOSTIC_GHI_MIN_WM2 = 100.0


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
    return out


def parse_date_set_inside_interval(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    dates: set,
    previous_anchor_date,
) -> list[str]:
    found = []
    for d in sorted(dates):
        if d == previous_anchor_date:
            continue
        day_start = pd.Timestamp(d)
        day_end = day_start + pd.Timedelta(days=1)
        if day_end > start_ts and day_start < end_ts:
            found.append(str(d))
    return found


def interval_weather_diagnostics(
    indexed: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict:
    w = indexed[(indexed.index > start_ts) & (indexed.index < end_ts)].copy()

    if w.empty:
        return {
            "interval_rows": 0,
            "rain_total_mm": 0.0,
            "rain_active_minutes": 0,
            "rain_days": 0,
            "ws_max": np.nan,
            "wsgust_max": np.nan,
            "dhi_max": np.nan,
            "dhi_over_ghi_q95": np.nan,
            "daylight_diag_rows": 0,
        }

    precip = pd.to_numeric(w["Precipitation"], errors="coerce").fillna(0.0)
    rain_mask = precip > 0.0

    ghi_all = pd.to_numeric(w["GHI"], errors="coerce")
    daylight = w[np.isfinite(ghi_all) & (ghi_all > DIAGNOSTIC_GHI_MIN_WM2)].copy()

    if len(daylight) > 0:
        ghi = pd.to_numeric(daylight["GHI"], errors="coerce")
        dhi = pd.to_numeric(daylight["DHI"], errors="coerce")
        ratio = dhi / ghi
        ratio = ratio[np.isfinite(ratio)]
        dhi_over_ghi_q95 = float(ratio.quantile(0.95)) if len(ratio) else np.nan
    else:
        dhi_over_ghi_q95 = np.nan

    rain_dates = set(w.index[rain_mask].date) if rain_mask.any() else set()

    return {
        "interval_rows": int(len(w)),
        "rain_total_mm": float(precip.sum()),
        "rain_active_minutes": int(rain_mask.sum()),
        "rain_days": int(len(rain_dates)),
        "ws_max": float(pd.to_numeric(w["WS"], errors="coerce").max()),
        "wsgust_max": float(pd.to_numeric(w["WSgust"], errors="coerce").max()),
        "dhi_max": float(pd.to_numeric(w["DHI"], errors="coerce").max()),
        "dhi_over_ghi_q95": dhi_over_ghi_q95,
        "daylight_diag_rows": int(len(daylight)),
    }


def validate_threshold_audit(path: Path, strong_count: int) -> dict:
    t = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "null_set",
        "quantile",
        "candidate_threshold",
        "real_events_exceeding",
    }
    missing = required.difference(t.columns)
    if missing:
        raise RuntimeError(f"threshold_audit missing columns: {sorted(missing)}")

    row = t[
        (t["null_set"].astype(str) == FROZEN_NULL_SET)
        & np.isclose(
            pd.to_numeric(t["quantile"], errors="coerce"),
            FROZEN_NULL_QUANTILE,
        )
    ]
    if len(row) != 1:
        raise RuntimeError(
            "Could not uniquely identify frozen nearest-one P95 threshold row."
        )

    threshold = float(row["candidate_threshold"].iloc[0])
    expected_count = int(row["real_events_exceeding"].iloc[0])

    if not np.isclose(threshold, FROZEN_THRESHOLD, rtol=0.0, atol=1e-14):
        raise RuntimeError(
            f"Frozen threshold mismatch: file={threshold}, expected={FROZEN_THRESHOLD}"
        )
    if expected_count != strong_count:
        raise RuntimeError(
            f"Strong-event count mismatch: threshold_audit={expected_count}, "
            f"reconstructed={strong_count}"
        )

    return {
        "null_set": FROZEN_NULL_SET,
        "quantile": FROZEN_NULL_QUANTILE,
        "threshold": threshold,
        "expected_strong_events": expected_count,
    }


def build_interval_audit(
    combined: pd.DataFrame,
    event_inventory: pd.DataFrame,
    real_primary: pd.DataFrame,
) -> pd.DataFrame:
    e = event_inventory.copy()
    r = real_primary.copy()

    e["first_pulse"] = pd.to_datetime(e["first_pulse"], errors="raise")
    e["last_pulse"] = pd.to_datetime(e["last_pulse"], errors="raise")
    e["date"] = e["date"].astype(str)

    for c in [
        "is_modb_cleaning_date",
        "is_scheduled_maintenance",
        "rain_near_event",
        "eligible_metadata_only",
    ]:
        if c in e.columns:
            e[c] = as_bool(e[c])

    r["first_pulse"] = pd.to_datetime(r["first_pulse"], errors="raise")
    r["last_pulse"] = pd.to_datetime(r["last_pulse"], errors="raise")
    r["date"] = r["date"].astype(str)
    r["moda_soiling_est"] = pd.to_numeric(r["moda_soiling_est"], errors="raise")

    strong = r[r["moda_soiling_est"] > FROZEN_THRESHOLD].copy()
    strong = strong.sort_values("first_pulse").reset_index(drop=True)

    e = e.sort_values("first_pulse").reset_index(drop=True)
    event_pos = {int(row.event_id): i for i, row in e.iterrows()}

    indexed = combined.set_index("Timestamp", drop=False)

    rows = []
    for cur in strong.itertuples(index=False):
        current_event_id = int(cur.event_id)

        if current_event_id not in event_pos:
            raise RuntimeError(
                f"Strong event_id {current_event_id} not found in event inventory."
            )

        pos = event_pos[current_event_id]
        if pos == 0:
            rows.append(
                {
                    "current_event_id": current_event_id,
                    "current_date": cur.date,
                    "current_first_pulse": cur.first_pulse,
                    "current_last_pulse": cur.last_pulse,
                    "current_gain": float(cur.moda_soiling_est),
                    "status": "SKIP_NO_PREVIOUS_CLEANING_ANCHOR",
                }
            )
            continue

        prev = e.iloc[pos - 1]

        intermediate = e[
            (e["first_pulse"] > pd.Timestamp(prev["first_pulse"]))
            & (e["first_pulse"] < pd.Timestamp(cur.first_pulse))
        ]
        if len(intermediate) != 0:
            raise RuntimeError(
                f"Unexpected intermediate Cleaning event(s) for current event "
                f"{current_event_id}."
            )

        start_ts = pd.Timestamp(prev["last_pulse"])
        end_ts = pd.Timestamp(cur.first_pulse)
        if start_ts >= end_ts:
            raise RuntimeError(
                f"Non-positive correction interval for event {current_event_id}: "
                f"{start_ts} -> {end_ts}"
            )

        weather = interval_weather_diagnostics(indexed, start_ts, end_ts)

        scheduled_inside = parse_date_set_inside_interval(
            start_ts=start_ts,
            end_ts=end_ts,
            dates=SCHEDULED_MAINTENANCE_DATES,
            previous_anchor_date=pd.Timestamp(prev["date"]).date(),
        )

        rain_confounded = weather["rain_active_minutes"] > 0
        scheduled_confounded = len(scheduled_inside) > 0
        prev_anchor_long_span = float(prev["pulse_span_min"]) >= 30.0

        if scheduled_confounded:
            status = "SKIP_SCHEDULED_MAINTENANCE_INSIDE_INTERVAL"
        elif rain_confounded:
            status = "SKIP_RAIN_NONLINEARITY"
        else:
            status = "ELIGIBLE_FOR_LINEAR_CORRECTION"

        rows.append(
            {
                "current_event_id": current_event_id,
                "current_date": cur.date,
                "current_first_pulse": cur.first_pulse,
                "current_last_pulse": cur.last_pulse,
                "current_pulse_span_min": float(cur.pulse_span_min),
                "current_gain": float(cur.moda_soiling_est),
                "current_n_pre": int(cur.n_pre),
                "current_n_post": int(cur.n_post),
                "previous_event_id": int(prev["event_id"]),
                "previous_date": str(prev["date"]),
                "previous_first_pulse": prev["first_pulse"],
                "previous_last_pulse": prev["last_pulse"],
                "previous_pulse_count": int(prev["pulse_count"]),
                "previous_pulse_span_min": float(prev["pulse_span_min"]),
                "previous_is_modb_cleaning_date": bool(prev["is_modb_cleaning_date"]),
                "previous_is_scheduled_maintenance": bool(
                    prev["is_scheduled_maintenance"]
                ),
                "previous_rain_near_event": bool(prev["rain_near_event"]),
                "previous_anchor_long_span_review": bool(prev_anchor_long_span),
                "correction_start": start_ts,
                "correction_end": end_ts,
                "interval_hours": float(
                    (end_ts - start_ts) / pd.Timedelta(hours=1)
                ),
                "interval_days": float(
                    (end_ts - start_ts) / pd.Timedelta(days=1)
                ),
                "scheduled_maintenance_inside_interval": ";".join(scheduled_inside),
                "scheduled_maintenance_inside_count": int(len(scheduled_inside)),
                **weather,
                "status": status,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["current_first_pulse", "current_event_id"]
    ).reset_index(drop=True)


def make_summary(
    y1_path: Path,
    y2_path: Path,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    combined: pd.DataFrame,
    event_inventory: pd.DataFrame,
    real_primary: pd.DataFrame,
    interval_audit: pd.DataFrame,
    threshold_provenance: dict,
) -> dict:
    status_counts = (
        interval_audit["status"].value_counts(dropna=False).to_dict()
        if len(interval_audit)
        else {}
    )

    eligible = interval_audit[
        interval_audit["status"] == "ELIGIBLE_FOR_LINEAR_CORRECTION"
    ].copy()

    official = interval_audit[
        interval_audit["current_date"].astype(str) == "2022-03-29"
    ].copy()

    def official_dict() -> dict:
        if len(official) != 1:
            return {"present": False}
        row = official.iloc[0]
        return {
            "present": True,
            "current_gain": float(row["current_gain"]),
            "previous_date": str(row["previous_date"]),
            "interval_hours": float(row["interval_hours"]),
            "rain_total_mm": float(row["rain_total_mm"]),
            "scheduled_maintenance_inside_interval": str(
                row["scheduled_maintenance_inside_interval"]
            ),
            "status": str(row["status"]),
        }

    return {
        "stage": "P2-0B-5.4c-1",
        "diagnostic_only": True,
        "moda_correction_applied": False,
        "moda_corr_generated": False,
        "final_cleanliness_generated": False,
        "input": {
            "year1_path": str(y1_path),
            "year1_sha256": sha256_file(y1_path),
            "year1_rows": int(len(y1)),
            "year2_path": str(y2_path),
            "year2_sha256": sha256_file(y2_path),
            "year2_rows": int(len(y2)),
            "combined_rows": int(len(combined)),
            "event_inventory_rows": int(len(event_inventory)),
            "real_primary_event_rows": int(len(real_primary)),
        },
        "frozen_threshold_provenance": threshold_provenance,
        "interval_definition": {
            "start": "last Cleaning pulse of immediately preceding OMT visit",
            "end": "first Cleaning pulse of current strong event",
            "rain_rule": (
                "any precipitation inside the full open interval -> "
                "no automatic linear correction"
            ),
            "scheduled_maintenance_rule": (
                "scheduled-maintenance date inside interval without being the "
                "recorded previous anchor -> no automatic linear correction"
            ),
            "wind_gust_diffuse_metrics": "diagnostic only; no threshold applied",
        },
        "counts": {
            "strong_events_above_frozen_p95": int(len(interval_audit)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "eligible_linear_intervals": int(len(eligible)),
            "rain_confounded_intervals": int(
                (interval_audit["status"] == "SKIP_RAIN_NONLINEARITY").sum()
            ),
            "scheduled_maintenance_confounded_intervals": int(
                (
                    interval_audit["status"]
                    == "SKIP_SCHEDULED_MAINTENANCE_INSIDE_INTERVAL"
                ).sum()
            ),
            "previous_anchor_long_span_review": int(
                interval_audit["previous_anchor_long_span_review"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
            if "previous_anchor_long_span_review" in interval_audit.columns
            else 0,
        },
        "all_strong_interval_duration_hours": finite_quantiles(
            interval_audit["interval_hours"]
        ),
        "eligible_interval_duration_hours": finite_quantiles(
            eligible["interval_hours"]
        ),
        "all_strong_gain_distribution": finite_quantiles(
            interval_audit["current_gain"]
        ),
        "eligible_gain_distribution": finite_quantiles(
            eligible["current_gain"]
        ),
        "rain_total_mm_distribution_all_strong": finite_quantiles(
            interval_audit["rain_total_mm"]
        ),
        "official_example_2022_03_29": official_dict(),
        "notes": [
            "This stage does not modify ModA.",
            "Previous Cleaning visit is used as the physical ModA reset anchor because the campaign procedure states that ModA is cleaned at every OMT visit.",
            "Rain inside the candidate interval blocks automatic linear backward correction in this stage.",
            "Scheduled-maintenance dates are checked directly by calendar date so the unrecorded Cleaning-pulse case on 2022-09-17 cannot be silently crossed.",
            "Wind, gust and diffuse-radiation metrics are exported only for review; no post-hoc thresholds are imposed.",
            "The exact multiplicative ModA correction formula is deliberately deferred to P2-0B-5.4c-2.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "P2-0B-5.4c-1 audit candidate ModA backward-correction intervals."
        )
    )
    p.add_argument("--year1", required=True, type=Path)
    p.add_argument("--year2", required=True, type=Path)
    p.add_argument("--event-inventory", required=True, type=Path)
    p.add_argument("--real-primary-events", required=True, type=Path)
    p.add_argument("--threshold-audit", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0b_5_4c1_moda_interval_audit_v1"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    y1_path = args.year1.expanduser().resolve()
    y2_path = args.year2.expanduser().resolve()
    event_path = args.event_inventory.expanduser().resolve()
    real_path = args.real_primary_events.expanduser().resolve()
    threshold_path = args.threshold_audit.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for p in [y1_path, y2_path, event_path, real_path, threshold_path]:
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

    print("[4/8] Read frozen 5.4a / 5.4b outputs")
    event_inventory = pd.read_csv(event_path, encoding="utf-8-sig")
    real_primary = pd.read_csv(real_path, encoding="utf-8-sig")

    if len(event_inventory) != 504:
        raise RuntimeError(
            f"Expected 504 event-inventory rows, found {len(event_inventory)}"
        )
    if len(real_primary) != 301:
        raise RuntimeError(
            f"Expected 301 real-primary-event rows, found {len(real_primary)}"
        )

    strong_count = int(
        (
            pd.to_numeric(real_primary["moda_soiling_est"], errors="coerce")
            > FROZEN_THRESHOLD
        ).sum()
    )

    print("[5/8] Validate frozen P95 threshold provenance")
    threshold_provenance = validate_threshold_audit(
        threshold_path, strong_count=strong_count
    )

    print("[6/8] Build correction-interval audit")
    interval_audit = build_interval_audit(
        combined=combined,
        event_inventory=event_inventory,
        real_primary=real_primary,
    )

    print("[7/8] Write outputs")
    interval_audit.to_csv(
        out_dir / "correction_interval_audit.csv",
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
        real_primary=real_primary,
        interval_audit=interval_audit,
        threshold_provenance=threshold_provenance,
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[8/8] Done")
    print(out_dir / "correction_interval_audit.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: interval audit only; ModA is NOT modified and ModA_corr is NOT generated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
