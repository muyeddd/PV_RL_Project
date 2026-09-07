#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1C-1R-B2
Rain Support Integrity Audit for the selected R3 OLS + empirical-residual model.

PURPOSE
-------
R3 was selected over R0/R4 by the predeclared one-step + LOCO recursive rules.
This audit does NOT re-select or tune R3. It determines exactly where R3 is
data-supported and how often the 365-day WAIT-only stress simulation invokes
the rain model outside that support.

SCIENTIFIC DISTINCTION
----------------------
Tier A -- DIRECT_DAILY_SUPPORT:
    Current source state lies inside the source-L range actually used to fit
    the daily rain transition model.

Tier B -- EVENT_DIRECTION_REFERENCE_ONLY:
    Current state exceeds daily-transition fitting support but lies within the
    S_pre range observed in resolved/unconfounded rain events. Event data
    support the *direction* of stronger removal at higher state, but are not
    treated as equivalent daily-transition training samples.

Tier C -- OUTSIDE_DAILY_AND_EVENT_REFERENCE:
    Current state exceeds both the daily-transition fitting range and the
    event-level S_pre reference range. This is genuine long-horizon model
    extrapolation.

A second, independent support axis is the Paper1-derived perception deployment
domain:
    historical WAPP max L = max(valid L_power_proxy)
    local-margin reference = historical max + 0.01

This script deliberately does NOT clip hidden states to any historical or
perception maximum.

AUDITS
------
1) Exact bidirectional OOT source-range support:
   YEAR1 -> YEAR2 and YEAR2 -> YEAR1.
   Count every out-of-range query and quantify exceedance magnitude.

2) Full-data daily-rain source-state support:
   sample count, quantiles, exact min/max.

3) Resolved/unconfounded event-level S_pre reference support:
   sample count, quantiles, exact max.
   This is only a physical-direction reference, NOT merged into R3 training.

4) Exact reproduction of the selected R3 365-day WAIT stress streams:
   seeds match P2-1C-1R-B when parent_env_seed=420001:
       YEAR1 historical-first: 1020001
       YEAR1 zero-start:       1021001
       YEAR2 historical-first: 1030001
       YEAR2 zero-start:       1031001

5) For EVERY simulated rain-affected transition, quantify:
   - fraction inside direct daily support;
   - fraction above daily support;
   - fraction in event-reference-only tier;
   - fraction beyond both supports;
   - fraction above perception historical max / +0.01 margin;
   - source-state distribution;
   - implied R3 mean rain response;
   - lower physical projection frequency.

PREDECLARED DIAGNOSTIC GATES
----------------------------
These gates do not alter the selected R3 model.

A) STRICT_OOT_RANGE_SUPPORT:
   PASS only if 100% of OOT test source states lie within the corresponding
   training-fold source-state range.

B) MILD_OOT_BOUNDARY_EXCEPTION:
   If strict support fails, the exception is labelled "mild" only if every
   out-of-range query is no more than +0.01 above the training maximum or
   -0.01 below the training minimum. This does NOT convert the strict gate
   to PASS; it only quantifies severity.

C) DIRECT_365_RAIN_SUPPORT:
   Diagnostic PASS only if <=5% of R3 rain-model calls in each primary
   historical-first stress scenario are above the full-data daily-rain fitting
   maximum. This is a conservative diagnostic threshold, not a physical law.

D) PERCEPTION_STRESS_SUPPORT:
   Diagnostic PASS only if <=5% of state-days in each primary historical-first
   stress scenario exceed historical_max + 0.01.

The 365-day WAIT episode remains an extrapolation stress test, not field
validation, because historical manual-clean intervals are far shorter.

NO CLEAN / NO REWARD / NO ONLINE PERCEPTION / NO GYM / NO PPO.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CALENDAR_DAYS = 730
EXPECTED_VALID_STATES = 729
EXPECTED_DRY = 494
EXPECTED_RAIN = 201
EXPECTED_CLEAN_DAYS = 26

YEAR_LABELS = ("YEAR1", "YEAR2")
EXPECTED_DAYS_PER_YEAR = 365

DEFAULT_N_STRESS_TRAJECTORIES = 1000
DEFAULT_PARENT_ENV_SEED = 420001

LOCAL_MARGIN = 0.01
DIRECT_STRESS_SUPPORT_THRESHOLD = 0.05
PERCEPTION_STRESS_THRESHOLD = 0.05


def parse_bool_series(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.isna().any() or (~x.isin([0, 1])).any():
            raise RuntimeError(f"{name}: invalid numeric boolean values.")
        return x.astype(int).astype(bool)

    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "y": True, "n": False,
    }
    x = s.astype(str).str.strip().str.lower()
    y = x.map(mapping)
    if y.isna().any():
        bad = sorted(x[y.isna()].unique().tolist())[:10]
        raise RuntimeError(f"{name}: unparseable values {bad}")
    return y.astype(bool)


def qstats(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)].to_numpy(dtype=float)
    if len(x) == 0:
        return {}
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "q50": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
        "max": float(np.max(x)),
    }


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slopes = []
    for i in range(len(x) - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        good = np.abs(dx) > 1e-15
        if np.any(good):
            slopes.extend((dy[good] / dx[good]).tolist())
    return float(np.median(slopes)) if slopes else float("nan")


def load_ledger(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "audit_period", "state_valid", "L_power_proxy",
        "rain_day", "modb_manual_cleaning_day", "scheduled_maintenance",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Master ledger missing: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()

    for c in [
        "state_valid", "rain_day",
        "modb_manual_cleaning_day", "scheduled_maintenance",
    ]:
        df[c] = parse_bool_series(df[c], c)

    df["L_power_proxy"] = pd.to_numeric(
        df["L_power_proxy"], errors="coerce"
    )

    if df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in master ledger.")

    return df.sort_values("date").reset_index(drop=True)


def load_transitions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "transition_index", "source_date", "dest_date",
        "transition_class", "delta_L_power_proxy",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Transition audit missing: {sorted(missing)}")

    for c in ["source_date", "dest_date"]:
        df[c] = pd.to_datetime(df[c], errors="raise").dt.normalize()

    df["delta_L_power_proxy"] = pd.to_numeric(
        df["delta_L_power_proxy"], errors="coerce"
    )

    return df.sort_values("transition_index").reset_index(drop=True)


def load_rain_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "rain_event_id", "start_date", "end_date",
        "S_pre", "S_post", "resolved", "cleaning_confounded",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Rain-event audit missing: {sorted(missing)}")

    for c in ["start_date", "end_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()

    for c in ["resolved", "cleaning_confounded"]:
        df[c] = parse_bool_series(df[c], c)

    for c in ["S_pre", "S_post"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def attach_context(
    transitions: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    src = ledger[
        ["date", "audit_period", "L_power_proxy"]
    ].rename(
        columns={
            "date": "source_date",
            "audit_period": "source_period",
            "L_power_proxy": "source_L",
        }
    )
    dst = ledger[
        ["date", "L_power_proxy"]
    ].rename(
        columns={
            "date": "dest_date",
            "L_power_proxy": "dest_L",
        }
    )

    x = transitions.merge(
        src, on="source_date", how="left", validate="many_to_one"
    ).merge(
        dst, on="dest_date", how="left", validate="many_to_one"
    )
    return x


def fit_r3(train_rain: pd.DataFrame) -> dict:
    g = train_rain[
        np.isfinite(train_rain["source_L"])
        & np.isfinite(train_rain["delta_L_power_proxy"])
    ].copy()

    x = g["source_L"].to_numpy(dtype=float)
    y = g["delta_L_power_proxy"].to_numpy(dtype=float)

    if len(x) < 3:
        raise RuntimeError("Insufficient rain training rows.")

    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)

    return {
        "n_train": int(len(x)),
        "intercept": float(intercept),
        "slope": float(slope),
        "residuals": residuals,
        "source_min": float(np.min(x)),
        "source_max": float(np.max(x)),
        "source_distribution": qstats(x),
        "residual_distribution": qstats(residuals),
    }


def exact_oot_support(rain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    query_rows = []

    for train_period, test_period in [
        ("YEAR1", "YEAR2"),
        ("YEAR2", "YEAR1"),
    ]:
        train = rain[rain["source_period"].eq(train_period)].copy()
        test = rain[rain["source_period"].eq(test_period)].copy()

        train_min = float(train["source_L"].min())
        train_max = float(train["source_L"].max())

        outside_distances = []

        for _, r in test.iterrows():
            q = float(r["source_L"])
            below = q < train_min
            above = q > train_max

            if below:
                distance = float(train_min - q)
                signed_distance = float(q - train_min)
                direction = "BELOW_TRAIN_MIN"
            elif above:
                distance = float(q - train_max)
                signed_distance = float(q - train_max)
                direction = "ABOVE_TRAIN_MAX"
            else:
                distance = 0.0
                signed_distance = 0.0
                direction = "IN_RANGE"

            if distance > 0:
                outside_distances.append(distance)

            query_rows.append({
                "train_period": train_period,
                "test_period": test_period,
                "transition_index": int(r["transition_index"]),
                "source_date": r["source_date"],
                "dest_date": r["dest_date"],
                "source_L": q,
                "train_source_min": train_min,
                "train_source_max": train_max,
                "support_direction": direction,
                "in_train_source_range": bool(distance == 0.0),
                "outside_distance": distance,
                "signed_outside_distance": signed_distance,
                "within_local_margin_if_outside": bool(
                    distance <= LOCAL_MARGIN
                ) if distance > 0 else True,
            })

        n = int(len(test))
        n_out = int(sum(d > 0 for d in [
            row["outside_distance"]
            for row in query_rows
            if row["train_period"] == train_period
            and row["test_period"] == test_period
        ]))

        max_out = float(max(outside_distances)) if outside_distances else 0.0

        fold_rows.append({
            "train_period": train_period,
            "test_period": test_period,
            "train_n": int(len(train)),
            "test_n": n,
            "train_source_min": train_min,
            "train_source_max": train_max,
            "test_in_range_n": int(n - n_out),
            "test_out_of_range_n": n_out,
            "test_in_range_fraction": float((n - n_out) / n),
            "max_outside_distance": max_out,
            "all_outside_within_0p01_margin": bool(
                max_out <= LOCAL_MARGIN
            ),
            "strict_100pct_range_support_pass": bool(n_out == 0),
        })

    return pd.DataFrame(fold_rows), pd.DataFrame(query_rows)


def get_year_calendar(
    ledger: pd.DataFrame,
    year_label: str,
) -> pd.DataFrame:
    y = ledger[
        ledger["audit_period"].eq(year_label)
    ].copy().sort_values("date").reset_index(drop=True)

    if len(y) != EXPECTED_DAYS_PER_YEAR:
        raise RuntimeError(
            f"{year_label}: expected 365 days, got {len(y)}"
        )
    return y


def classify_support_tier(
    source_state: np.ndarray,
    daily_min: float,
    daily_max: float,
    event_min: float,
    event_max: float,
) -> np.ndarray:
    s = np.asarray(source_state, dtype=float)
    tier = np.full(len(s), "OUTSIDE_DAILY_AND_EVENT_REFERENCE", dtype=object)

    direct = (s >= daily_min) & (s <= daily_max)
    tier[direct] = "DIRECT_DAILY_SUPPORT"

    event_reference_only = (
        (~direct)
        & (s >= event_min)
        & (s <= event_max)
    )
    tier[event_reference_only] = "EVENT_DIRECTION_REFERENCE_ONLY"

    return tier


def simulate_r3_stress_with_call_audit(
    year_df: pd.DataFrame,
    dry_pool: np.ndarray,
    r3: dict,
    n_trajectories: int,
    env_seed: int,
    daily_min: float,
    daily_max: float,
    event_min: float,
    event_max: float,
    hist_max: float,
    perception_margin_max: float,
) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(env_seed)

    n_days = len(year_df)
    states = np.empty((n_trajectories, n_days), dtype=float)

    initial_state = float(year_df.attrs["initial_state"])
    states[:, 0] = initial_state

    rain_calendar = year_df["rain_day"].to_numpy(dtype=bool)

    call_parts = []
    lower_projection_total = 0
    upper_projection_total = 0
    total_transitions = 0

    residuals = np.asarray(r3["residuals"], dtype=float)

    for t in range(n_days - 1):
        is_rain = bool(rain_calendar[t] or rain_calendar[t + 1])
        source = states[:, t].copy()

        if is_rain:
            idx = rng.integers(0, len(residuals), size=n_trajectories)
            mu_delta = (
                float(r3["intercept"])
                + float(r3["slope"]) * source
            )
            delta = mu_delta + residuals[idx]

            raw = source + delta
            tier = classify_support_tier(
                source,
                daily_min=daily_min,
                daily_max=daily_max,
                event_min=event_min,
                event_max=event_max,
            )

            call_parts.append(
                pd.DataFrame({
                    "day_index": int(t),
                    "source_date": year_df.loc[t, "date"],
                    "dest_date": year_df.loc[t + 1, "date"],
                    "trajectory_id": np.arange(
                        n_trajectories, dtype=int
                    ),
                    "source_L": source,
                    "support_tier": tier,
                    "above_daily_max": source > daily_max,
                    "above_daily_max_plus_0p01": (
                        source > daily_max + LOCAL_MARGIN
                    ),
                    "above_event_max": source > event_max,
                    "above_historical_max": source > hist_max,
                    "above_perception_margin_max": (
                        source > perception_margin_max
                    ),
                    "r3_mu_delta": mu_delta,
                    "raw_next_L": raw,
                    "lower_projection_needed": raw < 0.0,
                    "upper_projection_needed": raw > 1.0,
                })
            )
        else:
            idx = rng.integers(0, len(dry_pool), size=n_trajectories)
            delta = dry_pool[idx]
            raw = source + delta

        lower_projection_total += int((raw < 0.0).sum())
        upper_projection_total += int((raw > 1.0).sum())
        total_transitions += n_trajectories

        states[:, t + 1] = np.clip(raw, 0.0, 1.0)

    calls = (
        pd.concat(call_parts, ignore_index=True)
        if call_parts else pd.DataFrame()
    )

    above_hist = states > hist_max
    above_margin = states > perception_margin_max

    summary = {
        "n_trajectories": int(n_trajectories),
        "calendar_days": int(n_days),
        "in_horizon_transitions": int(n_days - 1),
        "rain_affected_transition_days": int(
            (rain_calendar[:-1] | rain_calendar[1:]).sum()
        ),
        "rain_model_calls": int(len(calls)),
        "state_day_fraction_above_historical_max": float(
            above_hist.mean()
        ),
        "state_day_fraction_above_perception_margin_max": float(
            above_margin.mean()
        ),
        "trajectory_fraction_ever_above_historical_max": float(
            above_hist.any(axis=1).mean()
        ),
        "trajectory_fraction_ever_above_perception_margin_max": float(
            above_margin.any(axis=1).mean()
        ),
        "terminal_state_distribution": qstats(states[:, -1]),
        "trajectory_max_distribution": qstats(states.max(axis=1)),
        "all_state_distribution": qstats(states.ravel()),
        "all_transition_lower_projection_fraction": float(
            lower_projection_total / total_transitions
        ),
        "all_transition_upper_projection_fraction": float(
            upper_projection_total / total_transitions
        ),
    }

    if len(calls):
        tier_counts = calls["support_tier"].value_counts().to_dict()
        summary.update({
            "rain_call_source_state_distribution": qstats(calls["source_L"]),
            "rain_call_direct_daily_support_fraction": float(
                (calls["support_tier"] == "DIRECT_DAILY_SUPPORT").mean()
            ),
            "rain_call_event_reference_only_fraction": float(
                (
                    calls["support_tier"]
                    == "EVENT_DIRECTION_REFERENCE_ONLY"
                ).mean()
            ),
            "rain_call_outside_daily_and_event_fraction": float(
                (
                    calls["support_tier"]
                    == "OUTSIDE_DAILY_AND_EVENT_REFERENCE"
                ).mean()
            ),
            "rain_call_above_daily_max_fraction": float(
                calls["above_daily_max"].mean()
            ),
            "rain_call_above_daily_max_plus_0p01_fraction": float(
                calls["above_daily_max_plus_0p01"].mean()
            ),
            "rain_call_above_event_max_fraction": float(
                calls["above_event_max"].mean()
            ),
            "rain_call_above_historical_max_fraction": float(
                calls["above_historical_max"].mean()
            ),
            "rain_call_above_perception_margin_max_fraction": float(
                calls["above_perception_margin_max"].mean()
            ),
            "rain_call_lower_projection_fraction": float(
                calls["lower_projection_needed"].mean()
            ),
            "rain_call_upper_projection_fraction": float(
                calls["upper_projection_needed"].mean()
            ),
            "rain_call_r3_mu_delta_distribution": qstats(
                calls["r3_mu_delta"]
            ),
            "support_tier_counts": {
                str(k): int(v) for k, v in tier_counts.items()
            },
        })

    return summary, calls


def r3_response_curve(
    r3: dict,
    daily_max: float,
    event_max: float,
    hist_max: float,
    perception_margin_max: float,
) -> pd.DataFrame:
    grid = sorted(set([
        0.0,
        0.01,
        0.03,
        float(daily_max),
        float(daily_max + LOCAL_MARGIN),
        0.10,
        float(event_max),
        float(hist_max),
        float(perception_margin_max),
        0.20,
        0.30,
        0.50,
    ]))

    rows = []
    for L in grid:
        mu_delta = float(r3["intercept"] + r3["slope"] * L)
        mu_next = float(L + mu_delta)
        removal_fraction = (
            float(-mu_delta / L)
            if L > 0 and mu_delta < 0
            else np.nan
        )
        rows.append({
            "source_L": float(L),
            "r3_mean_delta": mu_delta,
            "r3_mean_next_before_residual": mu_next,
            "implied_mean_removal_fraction_if_negative": removal_fraction,
            "inside_daily_training_range": bool(
                r3["source_min"] <= L <= r3["source_max"]
            ),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-1C-1R-B2 rain support integrity audit."
    )
    p.add_argument("--master-ledger", required=True, type=Path)
    p.add_argument("--transition-audit", required=True, type=Path)
    p.add_argument("--rain-event-audit", required=True, type=Path)
    p.add_argument(
        "--n-stress-trajectories",
        type=int,
        default=DEFAULT_N_STRESS_TRAJECTORIES,
    )
    p.add_argument(
        "--parent-env-seed",
        type=int,
        default=DEFAULT_PARENT_ENV_SEED,
        help=(
            "Use 420001 to reproduce the R3 stress streams from "
            "P2-1C-1R-B."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1c1r_b2_rain_support_integrity_audit_v1"
        ),
    )
    args = p.parse_args()

    if args.n_stress_trajectories < 200:
        raise ValueError("--n-stress-trajectories must be >=200.")

    lp = args.master_ledger.expanduser().resolve()
    tp = args.transition_audit.expanduser().resolve()
    rp = args.rain_event_audit.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()

    for path in [lp, tp, rp]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/10] Load frozen ledger, transitions, and rain-event audit")
    ledger = load_ledger(lp)
    transitions = load_transitions(tp)
    rain_events = load_rain_events(rp)

    if len(ledger) != EXPECTED_CALENDAR_DAYS:
        raise RuntimeError("Calendar day count mismatch.")
    if int(ledger["state_valid"].sum()) != EXPECTED_VALID_STATES:
        raise RuntimeError("Valid-state count mismatch.")
    if int(ledger["modb_manual_cleaning_day"].sum()) != EXPECTED_CLEAN_DAYS:
        raise RuntimeError("Manual-clean day count mismatch.")

    print("[2/10] Attach source/destination state context")
    x = attach_context(transitions, ledger)
    dry = x[x["transition_class"].eq("DRY_NATURAL")].copy()
    rain = x[x["transition_class"].eq("RAIN_AFFECTED")].copy()

    if len(dry) != EXPECTED_DRY:
        raise RuntimeError(f"Dry count mismatch: {len(dry)}")
    if len(rain) != EXPECTED_RAIN:
        raise RuntimeError(f"Rain count mismatch: {len(rain)}")

    print("[3/10] Audit exact bidirectional OOT source-range support")
    oot_folds, oot_queries = exact_oot_support(rain)

    strict_oot_pass = bool(
        oot_folds["strict_100pct_range_support_pass"].all()
    )

    outside = oot_queries[
        ~oot_queries["in_train_source_range"]
    ].copy()

    mild_exception = bool(
        len(outside) > 0
        and outside["within_local_margin_if_outside"].all()
    )

    print("[4/10] Fit selected full-data R3 and define direct daily support")
    r3 = fit_r3(rain)

    daily_min = float(r3["source_min"])
    daily_max = float(r3["source_max"])

    print("[5/10] Audit resolved/unconfounded event-level S_pre reference")
    rr = rain_events[
        rain_events["resolved"]
        & (~rain_events["cleaning_confounded"])
        & np.isfinite(rain_events["S_pre"])
        & np.isfinite(rain_events["S_post"])
    ].copy()

    if len(rr) == 0:
        raise RuntimeError("No resolved/unconfounded finite rain events.")

    event_min = float(rr["S_pre"].min())
    event_max = float(rr["S_pre"].max())
    event_rho = float(
        rr["S_pre"].corr(
            rr["S_pre"] - rr["S_post"],
            method="spearman",
        )
    )

    valid_hist = ledger.loc[
        ledger["state_valid"]
        & np.isfinite(ledger["L_power_proxy"]),
        "L_power_proxy",
    ].to_numpy(dtype=float)

    hist_max = float(np.max(valid_hist))
    perception_margin_max = float(hist_max + LOCAL_MARGIN)

    support_reference = {
        "daily_rain_training_source_L": qstats(rain["source_L"]),
        "daily_rain_source_min": daily_min,
        "daily_rain_source_max": daily_max,
        "resolved_unconfounded_event_n": int(len(rr)),
        "event_S_pre_distribution": qstats(rr["S_pre"]),
        "event_S_pre_min": event_min,
        "event_S_pre_max": event_max,
        "event_Spre_vs_removal_spearman": event_rho,
        "historical_valid_state_distribution": qstats(valid_hist),
        "historical_state_max": hist_max,
        "perception_margin_max": perception_margin_max,
        "semantic_warning": (
            "Event-level S_pre is a physical-direction reference only. "
            "It is NOT merged with daily-transition samples and does not "
            "convert Tier B into direct daily-model support."
        ),
    }

    print("[6/10] Build R3 response-curve extrapolation diagnostic")
    response_curve = r3_response_curve(
        r3=r3,
        daily_max=daily_max,
        event_max=event_max,
        hist_max=hist_max,
        perception_margin_max=perception_margin_max,
    )

    print("[7/10] Reproduce R3 365-day WAIT stress and audit every rain call")
    dry_pool = dry["delta_L_power_proxy"].to_numpy(dtype=float)

    scenario_rows = []
    call_parts = []

    # Exact seed convention inherited from P2-1C-1R-B:
    # stress_root = parent_env_seed + 500000
    # R3 candidate index = 1 => +100000
    stress_root = int(args.parent_env_seed + 500000)

    for yidx, year_label in enumerate(YEAR_LABELS):
        ydf0 = get_year_calendar(ledger, year_label)
        hist_first = float(ydf0["L_power_proxy"].iloc[0])

        for iidx, (init_mode, init_state) in enumerate([
            ("HISTORICAL_FIRST_DAY", hist_first),
            ("CLEAN_START_ZERO", 0.0),
        ]):
            scenario_seed = int(
                stress_root
                + 100000
                + 10000 * yidx
                + 1000 * iidx
            )

            ydf = ydf0.copy()
            ydf.attrs["initial_state"] = float(init_state)

            s, calls = simulate_r3_stress_with_call_audit(
                year_df=ydf,
                dry_pool=dry_pool,
                r3=r3,
                n_trajectories=args.n_stress_trajectories,
                env_seed=scenario_seed,
                daily_min=daily_min,
                daily_max=daily_max,
                event_min=event_min,
                event_max=event_max,
                hist_max=hist_max,
                perception_margin_max=perception_margin_max,
            )

            row = {
                "year_label": year_label,
                "init_mode": init_mode,
                "initial_state": float(init_state),
                "scenario_env_seed": scenario_seed,
                **s,
            }
            scenario_rows.append(row)

            if len(calls):
                calls["year_label"] = year_label
                calls["init_mode"] = init_mode
                calls["scenario_env_seed"] = scenario_seed
                call_parts.append(calls)

    scenario_df = pd.DataFrame(scenario_rows)
    call_df = pd.concat(call_parts, ignore_index=True)

    print("[8/10] Apply predeclared support-integrity diagnostics")
    primary = scenario_df[
        scenario_df["init_mode"].eq("HISTORICAL_FIRST_DAY")
    ].copy()

    direct_365_pass = bool(
        (
            primary["rain_call_above_daily_max_fraction"]
            <= DIRECT_STRESS_SUPPORT_THRESHOLD
        ).all()
    )

    perception_stress_pass = bool(
        (
            primary["state_day_fraction_above_perception_margin_max"]
            <= PERCEPTION_STRESS_THRESHOLD
        ).all()
    )

    technical_gates = {
        "calendar_count_pass": bool(
            len(ledger) == EXPECTED_CALENDAR_DAYS
        ),
        "valid_state_count_pass": bool(
            int(ledger["state_valid"].sum())
            == EXPECTED_VALID_STATES
        ),
        "dry_count_pass": bool(len(dry) == EXPECTED_DRY),
        "rain_count_pass": bool(len(rain) == EXPECTED_RAIN),
        "manual_clean_count_pass": bool(
            int(ledger["modb_manual_cleaning_day"].sum())
            == EXPECTED_CLEAN_DAYS
        ),
        "r3_slope_negative_pass": bool(r3["slope"] < 0),
        "event_direction_reference_pass": bool(event_rho > 0),
    }
    technical_gates["all_technical_gates_pass"] = bool(
        all(technical_gates.values())
    )

    support_gates = {
        "strict_100pct_bidirectional_OOT_range_support_pass": strict_oot_pass,
        "all_OOT_exceptions_within_0p01_margin": mild_exception,
        "direct_365_rain_support_pass": direct_365_pass,
        "perception_365_stress_support_pass": perception_stress_pass,
    }

    if strict_oot_pass:
        oot_interpretation = "STRICT_RANGE_SUPPORT_PASS"
    elif mild_exception:
        oot_interpretation = (
            "STRICT_RANGE_SUPPORT_FAIL_BUT_ALL_EXCEPTIONS_ARE_WITHIN_0P01"
        )
    else:
        oot_interpretation = (
            "STRICT_RANGE_SUPPORT_FAIL_WITH_NONTRIVIAL_BOUNDARY_EXTRAPOLATION"
        )

    if direct_365_pass:
        long_horizon_rain_interpretation = (
            "R3_RAIN_CALLS_ARE_MOSTLY_WITHIN_DIRECT_DAILY_SUPPORT"
        )
    else:
        long_horizon_rain_interpretation = (
            "R3_365_WAIT_FREQUENTLY_USES_RAIN_MODEL_OUTSIDE_DIRECT_DAILY_SUPPORT"
        )

    if perception_stress_pass:
        perception_interpretation = (
            "365_WAIT_STRESS_MOSTLY_WITHIN_PERCEPTION_MARGIN"
        )
    else:
        perception_interpretation = (
            "365_WAIT_STRESS_FREQUENTLY_EXCEEDS_PERCEPTION_MARGIN"
        )

    print("[9/10] Build scientific audit summary")
    audit_summary = {
        "stage": "P2-1C-1R-B2",
        "audit_only": True,
        "selected_rain_model_re_tuned": False,
        "selected_rain_model": "R3_OLS_RESIDUAL",
        "r3_full_data": {
            "n_train": int(r3["n_train"]),
            "intercept": float(r3["intercept"]),
            "slope": float(r3["slope"]),
            "source_min": daily_min,
            "source_max": daily_max,
            "residual_distribution": r3["residual_distribution"],
        },
        "support_reference": support_reference,
        "oot_range_support": {
            "fold_summary": oot_folds.to_dict(orient="records"),
            "out_of_range_query_n": int(len(outside)),
            "out_of_range_query_fraction": float(
                len(outside) / len(oot_queries)
            ),
            "out_of_range_distance_distribution": qstats(
                outside["outside_distance"]
            ) if len(outside) else {},
            "interpretation": oot_interpretation,
        },
        "stress_scenarios": scenario_df.to_dict(orient="records"),
        "support_gates": support_gates,
        "technical_gates": technical_gates,
        "scientific_interpretation": {
            "long_horizon_rain_support": long_horizon_rain_interpretation,
            "perception_stress_support": perception_interpretation,
            "event_reference_role": (
                "Event-level support may corroborate the direction of state-"
                "dependent rain cleaning beyond the daily-fit range, but it "
                "must not be presented as direct training support for R3."
            ),
            "freeze_rule": (
                "R3 may be frozen as the primary empirically supported rain "
                "candidate for one-step and observed cleaning-cycle horizons "
                "if prior R-B selection remains valid. Whether it can be called "
                "a directly supported 365-day no-clean kernel depends on the "
                "direct_365_rain_support diagnostic; extreme WAIT remains a "
                "stress/extrapolation case regardless."
            ),
            "rl_rule": (
                "Do not begin Point/UA PPO merely because R3 is selected. "
                "The next environment stage must explicitly control/audit policy "
                "state occupancy relative to the perception-supported domain."
            ),
        },
        "next_step": (
            "Review support integrity. If R3 is supported on historical/LOCO "
            "horizons but 365-day WAIT remains substantially extrapolative, "
            "freeze R3 with a scope-qualified claim and design the counterfactual "
            "environment so that policy-domain support occupancy is audited "
            "explicitly before online perception/PPO."
        ),
    }

    print("[10/10] Write outputs")
    out.mkdir(parents=True, exist_ok=True)

    oot_folds.to_csv(
        out / "oot_range_support_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oot_queries.to_csv(
        out / "oot_range_support_query_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    response_curve.to_csv(
        out / "r3_response_curve_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_df.to_csv(
        out / "r3_wait365_support_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    call_df.to_csv(
        out / "r3_wait365_rain_call_support_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            audit_summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(out / "oot_range_support_fold_summary.csv")
    print(out / "oot_range_support_query_audit.csv")
    print(out / "r3_response_curve_audit.csv")
    print(out / "r3_wait365_support_scenario_summary.csv")
    print(out / "r3_wait365_rain_call_support_audit.csv")
    print(out / "audit_summary.json")
    print(
        "IMPORTANT: support audit only. No CLEAN, reward, online perception, "
        "Gym, or PPO should be added before scientific review."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
