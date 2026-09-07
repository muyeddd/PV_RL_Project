#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1C-1R-B
State-dependent rain repair candidate validation.

PURPOSE
-------
P2-1C-1 showed that the frozen one-step D0/R0 kernels become severely
out-of-domain when recursively rolled for a 365-day WAIT-only stress episode.

P2-1C-1R-A then showed two important facts:
1) D0/R0 still have meaningful recursive skill over historically observed
   post-clean horizons (roughly one month);
2) rain response has stable negative state dependence at both daily and
   event levels: higher pre-rain soiling -> more negative rain delta.

This script compares ONLY three predeclared rain candidates:

R0_GLOBAL
    delta ~ empirical rain-delta pool, independent of current state.

R3_OLS_RESIDUAL
    delta = alpha_OLS + beta_OLS * L_t + epsilon,
    epsilon sampled from the empirical training residual distribution.

R4_THEILSEN_RESIDUAL
    delta = alpha_TS + beta_TS * L_t + epsilon,
    beta_TS = Theil-Sen robust slope,
    alpha_TS = median(y - beta_TS*x),
    epsilon sampled from the empirical training residual distribution.

Dry dynamics remain the frozen D0_GLOBAL empirical kernel.

VALIDATION LAYERS
-----------------
Layer 1: bidirectional one-step out-of-time validation
    TRAIN YEAR1 -> TEST YEAR2
    TRAIN YEAR2 -> TEST YEAR1

Layer 2: leave-one-cleaning-cycle-out recursive validation
    Target cycle transitions are excluded from both dry and rain training pools.
    The historical rain calendar is replayed.
    Primary scoring excludes cycles containing scheduled-maintenance days.

Layer 3: 365-day WAIT-only extrapolation stress test
    This is explicitly a STRESS TEST, not historical validation, because the
    observed maximum no-manual-clean interval is only ~33 days.
    No historical-max clipping is allowed.

PREDECLARED REPAIR ELIGIBILITY
------------------------------
A repaired candidate (R3/R4) may replace R0 only if all are true:
1) fitted rain slope is negative in both OOT training folds and on full data;
2) one-step macro CRPS is within 5% of the best full-support candidate;
3) primary LOCO recursive CRPS is lower than R0;
4) primary LOCO median MAE is lower than R0;
5) primary LOCO |coverage90 - 0.90| is not worse than R0.

If both R3 and R4 are eligible:
- choose lower LOCO CRPS;
- if their LOCO CRPS values are within 2%, prefer R3 OLS for parsimony.

The 365-day stress test is NOT an automatic eligibility gate because it is
far beyond the historical intervention horizon. It is used to quantify how
the repair changes long-horizon support/extrapolation behaviour.

NO CLEAN / NO REWARD / NO ONLINE PERCEPTION / NO GYM / NO PPO.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DRY = 494
EXPECTED_RAIN = 201
EXPECTED_CLEAN_DAYS = 26
EXPECTED_CALENDAR_DAYS = 730
EXPECTED_VALID_STATES = 729

YEAR_LABELS = ("YEAR1", "YEAR2")
EXPECTED_DAYS_PER_YEAR = 365

DEFAULT_N_CYCLE_TRAJECTORIES = 1000
DEFAULT_N_STRESS_TRAJECTORIES = 1000
DEFAULT_ENV_SEED = 420001

MIN_LOO_DRY_POOL = 350
MIN_LOO_RAIN_POOL = 120
MIN_PRIMARY_CYCLES = 18

ONE_STEP_TOL = 0.05
REPAIRED_TIE_TOL = 0.02
NOMINAL_COVERAGE = 0.90

CANDIDATES = (
    "R0_GLOBAL",
    "R3_OLS_RESIDUAL",
    "R4_THEILSEN_RESIDUAL",
)


def parse_bool_series(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.isna().any() or (~x.isin([0, 1])).any():
            raise RuntimeError(f"{name}: invalid numeric boolean values.")
        return x.astype(int).astype(bool)
    mp = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "y": True, "n": False,
    }
    x = s.astype(str).str.strip().str.lower()
    y = x.map(mp)
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


def empirical_crps(samples: np.ndarray, y: float) -> float:
    """Exact empirical CRPS in O(n log n)."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    first = float(np.mean(np.abs(x - y)))
    i = np.arange(1, n + 1, dtype=float)
    half_pair = float(
        np.sum((2.0 * i - n - 1.0) * x) / (n * n)
    )
    return first - half_pair


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slopes = []
    for i in range(len(x) - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        ok = np.abs(dx) > 1e-15
        if np.any(ok):
            slopes.extend((dy[ok] / dx[ok]).tolist())
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

    df["L_power_proxy"] = pd.to_numeric(df["L_power_proxy"], errors="coerce")

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
        raise RuntimeError(f"Rain event audit missing: {sorted(missing)}")

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


def fit_rain_model(train_rain: pd.DataFrame, candidate: str) -> dict:
    g = train_rain[
        np.isfinite(train_rain["source_L"])
        & np.isfinite(train_rain["delta_L_power_proxy"])
    ].copy()

    x = g["source_L"].to_numpy(dtype=float)
    y = g["delta_L_power_proxy"].to_numpy(dtype=float)

    if len(y) < 3:
        raise RuntimeError(f"{candidate}: insufficient rain training rows.")

    if candidate == "R0_GLOBAL":
        return {
            "candidate": candidate,
            "n_train": int(len(y)),
            "intercept": 0.0,
            "slope": 0.0,
            "residuals": y.copy(),
            "residual_distribution": qstats(y),
            "train_source_L_min": float(np.min(x)),
            "train_source_L_max": float(np.max(x)),
        }

    if candidate == "R3_OLS_RESIDUAL":
        slope, intercept = np.polyfit(x, y, 1)

    elif candidate == "R4_THEILSEN_RESIDUAL":
        slope = theil_sen_slope(x, y)
        if not np.isfinite(slope):
            raise RuntimeError("Theil-Sen slope is not finite.")
        intercept = float(np.median(y - slope * x))

    else:
        raise KeyError(candidate)

    fitted = intercept + slope * x
    residuals = y - fitted

    return {
        "candidate": candidate,
        "n_train": int(len(y)),
        "intercept": float(intercept),
        "slope": float(slope),
        "residuals": residuals,
        "residual_distribution": qstats(residuals),
        "train_source_L_min": float(np.min(x)),
        "train_source_L_max": float(np.max(x)),
    }


def rain_next_samples(
    source_L: float,
    model: dict,
) -> np.ndarray:
    residuals = np.asarray(model["residuals"], dtype=float)
    if model["candidate"] == "R0_GLOBAL":
        raw_next = float(source_L) + residuals
    else:
        mu_delta = (
            float(model["intercept"])
            + float(model["slope"]) * float(source_L)
        )
        raw_next = float(source_L) + mu_delta + residuals

    # This is the physical simulator boundary, NOT historical-max clipping.
    return np.clip(raw_next, 0.0, 1.0)


def dry_next_samples(
    source_L: float,
    dry_pool: np.ndarray,
) -> np.ndarray:
    return np.clip(float(source_L) + dry_pool, 0.0, 1.0)


def one_step_cv(
    x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rain = x[x["transition_class"].eq("RAIN_AFFECTED")].copy()
    fold_rows = []
    pred_rows = []
    param_rows = []

    for train_period, test_period in [
        ("YEAR1", "YEAR2"),
        ("YEAR2", "YEAR1"),
    ]:
        train = rain[rain["source_period"].eq(train_period)].copy()
        test = rain[rain["source_period"].eq(test_period)].copy()

        for candidate in CANDIDATES:
            model = fit_rain_model(train, candidate)

            param_rows.append({
                "candidate": candidate,
                "train_period": train_period,
                "test_period": test_period,
                "n_train": int(model["n_train"]),
                "intercept": float(model["intercept"]),
                "slope": float(model["slope"]),
                "train_source_L_min": float(model["train_source_L_min"]),
                "train_source_L_max": float(model["train_source_L_max"]),
            })

            rows = []
            for _, r in test.iterrows():
                src = float(r["source_L"])
                obs_next = float(r["dest_L"])
                samples = rain_next_samples(src, model)

                q05, q50, q95 = np.quantile(
                    samples, [0.05, 0.50, 0.95]
                )
                in_range = bool(
                    model["train_source_L_min"]
                    <= src
                    <= model["train_source_L_max"]
                )

                rec = {
                    "candidate": candidate,
                    "train_period": train_period,
                    "test_period": test_period,
                    "transition_index": int(r["transition_index"]),
                    "source_date": r["source_date"],
                    "dest_date": r["dest_date"],
                    "source_L": src,
                    "observed_next_L": obs_next,
                    "train_source_range_contains_query": in_range,
                    "crps_next_state": empirical_crps(samples, obs_next),
                    "median_abs_error_next_state": float(abs(q50 - obs_next)),
                    "covered90_next_state": bool(q05 <= obs_next <= q95),
                    "width90_next_state": float(q95 - q05),
                    "pred_q05": float(q05),
                    "pred_q50": float(q50),
                    "pred_q95": float(q95),
                }
                rows.append(rec)
                pred_rows.append(rec)

            f = pd.DataFrame(rows)
            fold_rows.append({
                "candidate": candidate,
                "train_period": train_period,
                "test_period": test_period,
                "test_n": int(len(f)),
                "finite_prediction_pass": bool(
                    np.isfinite(
                        f[
                            [
                                "crps_next_state",
                                "median_abs_error_next_state",
                                "width90_next_state",
                            ]
                        ].to_numpy(dtype=float)
                    ).all()
                ),
                "query_in_train_source_range_fraction": float(
                    f["train_source_range_contains_query"].mean()
                ),
                "crps_mean": float(f["crps_next_state"].mean()),
                "median_abs_error_mean": float(
                    f["median_abs_error_next_state"].mean()
                ),
                "coverage90": float(
                    f["covered90_next_state"].astype(bool).mean()
                ),
                "width90_mean": float(f["width90_next_state"].mean()),
                "fitted_slope": float(model["slope"]),
            })

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    params = pd.DataFrame(param_rows)
    return folds, preds, params


def macro_one_step(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        g = folds[folds["candidate"].eq(candidate)].copy()
        rows.append({
            "candidate": candidate,
            "folds": int(len(g)),
            "full_finite_support": bool(
                len(g) == 2 and g["finite_prediction_pass"].all()
            ),
            "crps_macro": float(g["crps_mean"].mean()),
            "median_abs_error_macro": float(
                g["median_abs_error_mean"].mean()
            ),
            "coverage90_macro": float(g["coverage90"].mean()),
            "coverage90_abs_error": float(
                abs(g["coverage90"].mean() - NOMINAL_COVERAGE)
            ),
            "width90_macro": float(g["width90_mean"].mean()),
            "min_query_in_train_source_range_fraction": float(
                g["query_in_train_source_range_fraction"].min()
            ),
            "slope_negative_both_folds": bool(
                (g["fitted_slope"] < 0).all()
            ) if candidate != "R0_GLOBAL" else True,
        })
    return pd.DataFrame(rows)


def build_cleaning_cycles(ledger: pd.DataFrame) -> pd.DataFrame:
    clean_dates = ledger.loc[
        ledger["modb_manual_cleaning_day"], "date"
    ].sort_values().tolist()

    if len(clean_dates) != EXPECTED_CLEAN_DAYS:
        raise RuntimeError(
            f"Expected {EXPECTED_CLEAN_DAYS} clean days, got {len(clean_dates)}."
        )

    rows = []
    for i, start in enumerate(clean_dates):
        if i + 1 < len(clean_dates):
            end = clean_dates[i + 1] - pd.Timedelta(days=1)
        else:
            end = ledger["date"].max()

        seg = ledger[
            (ledger["date"] >= start)
            & (ledger["date"] <= end)
        ].copy()

        start_row = ledger[ledger["date"].eq(start)].iloc[0]
        rows.append({
            "cycle_id": int(i + 1),
            "cycle_start": start,
            "cycle_end": end,
            "cycle_calendar_days": int(len(seg)),
            "start_L": float(start_row["L_power_proxy"]),
            "maintenance_days": int(seg["scheduled_maintenance"].sum()),
            "contains_invalid_state": bool((~seg["state_valid"]).any()),
        })
    return pd.DataFrame(rows)


def simulate_cycle_candidate(
    cycle_row: pd.Series,
    ledger: pd.DataFrame,
    x: pd.DataFrame,
    candidate: str,
    n_trajectories: int,
    env_seed: int,
) -> tuple[pd.DataFrame, dict]:
    start = pd.Timestamp(cycle_row["cycle_start"])
    end = pd.Timestamp(cycle_row["cycle_end"])

    seg = ledger[
        (ledger["date"] >= start)
        & (ledger["date"] <= end)
    ].copy().sort_values("date").reset_index(drop=True)

    train = x[
        ~(
            (x["source_date"] >= start)
            & (x["source_date"] <= end)
        )
    ].copy()

    dry_pool = train.loc[
        train["transition_class"].eq("DRY_NATURAL"),
        "delta_L_power_proxy",
    ].to_numpy(dtype=float)

    train_rain = train[
        train["transition_class"].eq("RAIN_AFFECTED")
    ].copy()

    if len(dry_pool) < MIN_LOO_DRY_POOL:
        raise RuntimeError(
            f"Cycle {cycle_row['cycle_id']}: dry LOO pool too small."
        )
    if len(train_rain) < MIN_LOO_RAIN_POOL:
        raise RuntimeError(
            f"Cycle {cycle_row['cycle_id']}: rain LOO pool too small."
        )

    model = fit_rain_model(train_rain, candidate)
    start_L = float(cycle_row["start_L"])

    rng = np.random.default_rng(env_seed)
    states = np.empty((n_trajectories, len(seg)), dtype=float)
    states[:, 0] = start_L

    rain_flags = seg["rain_day"].to_numpy(dtype=bool)

    for t in range(len(seg) - 1):
        is_rain = bool(rain_flags[t] or rain_flags[t + 1])

        if is_rain:
            residuals = np.asarray(model["residuals"], dtype=float)
            idx = rng.integers(0, len(residuals), size=n_trajectories)

            if candidate == "R0_GLOBAL":
                delta = residuals[idx]
            else:
                mu_delta = (
                    float(model["intercept"])
                    + float(model["slope"]) * states[:, t]
                )
                delta = mu_delta + residuals[idx]
        else:
            idx = rng.integers(0, len(dry_pool), size=n_trajectories)
            delta = dry_pool[idx]

        states[:, t + 1] = np.clip(
            states[:, t] + delta, 0.0, 1.0
        )

    rows = []
    for d in range(1, len(seg)):
        obs_valid = bool(seg.loc[d, "state_valid"])
        obs = (
            float(seg.loc[d, "L_power_proxy"])
            if obs_valid
            else np.nan
        )
        pred = states[:, d]
        q05, q50, q95 = np.quantile(pred, [0.05, 0.50, 0.95])

        rec = {
            "candidate": candidate,
            "cycle_id": int(cycle_row["cycle_id"]),
            "date": seg.loc[d, "date"],
            "day_since_clean": int(d),
            "observed_state_valid": bool(
                obs_valid and np.isfinite(obs)
            ),
            "observed_L": obs,
            "pred_q05": float(q05),
            "pred_q50": float(q50),
            "pred_q95": float(q95),
            "pred_mean": float(np.mean(pred)),
            "pred_max": float(np.max(pred)),
            "loo_dry_pool_n": int(len(dry_pool)),
            "loo_rain_pool_n": int(len(train_rain)),
            "fitted_rain_slope": float(model["slope"]),
        }

        if rec["observed_state_valid"]:
            rec.update({
                "crps": empirical_crps(pred, obs),
                "median_abs_error": float(abs(q50 - obs)),
                "covered90": bool(q05 <= obs <= q95),
                "persistence_abs_error": float(abs(start_L - obs)),
            })
        else:
            rec.update({
                "crps": np.nan,
                "median_abs_error": np.nan,
                "covered90": False,
                "persistence_abs_error": np.nan,
            })
        rows.append(rec)

    day_df = pd.DataFrame(rows)
    valid = day_df[day_df["observed_state_valid"]].copy()

    summary = {
        "candidate": candidate,
        "cycle_id": int(cycle_row["cycle_id"]),
        "cycle_start": start.strftime("%Y-%m-%d"),
        "cycle_end": end.strftime("%Y-%m-%d"),
        "cycle_calendar_days": int(len(seg)),
        "primary_validation_cycle": bool(
            int(cycle_row["maintenance_days"]) == 0
        ),
        "contains_invalid_state": bool(
            cycle_row["contains_invalid_state"]
        ),
        "maintenance_days": int(cycle_row["maintenance_days"]),
        "evaluable_days": int(len(valid)),
        "rain_slope": float(model["slope"]),
        "rain_slope_negative": bool(
            model["slope"] < 0
        ) if candidate != "R0_GLOBAL" else True,
        "crps_mean": float(valid["crps"].mean())
        if len(valid) else np.nan,
        "median_abs_error_mean": float(
            valid["median_abs_error"].mean()
        ) if len(valid) else np.nan,
        "coverage90": float(
            valid["covered90"].astype(bool).mean()
        ) if len(valid) else np.nan,
        "persistence_abs_error_mean": float(
            valid["persistence_abs_error"].mean()
        ) if len(valid) else np.nan,
        "predicted_state_max": float(states.max()),
    }
    return day_df, summary


def loco_recursive_validation(
    ledger: pd.DataFrame,
    x: pd.DataFrame,
    n_trajectories: int,
    env_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycles = build_cleaning_cycles(ledger)
    day_parts = []
    cycle_rows = []

    for candidate_idx, candidate in enumerate(CANDIDATES):
        for _, cycle in cycles.iterrows():
            seed = int(
                env_seed
                + 100000 * candidate_idx
                + 1000 * int(cycle["cycle_id"])
            )
            d, s = simulate_cycle_candidate(
                cycle_row=cycle,
                ledger=ledger,
                x=x,
                candidate=candidate,
                n_trajectories=n_trajectories,
                env_seed=seed,
            )
            s["scenario_env_seed"] = seed
            day_parts.append(d)
            cycle_rows.append(s)

    days = pd.concat(day_parts, ignore_index=True)
    cycle_summary = pd.DataFrame(cycle_rows)

    macro_rows = []
    for candidate in CANDIDATES:
        cand_cycles = cycle_summary[
            cycle_summary["candidate"].eq(candidate)
        ]
        primary_ids = cand_cycles.loc[
            cand_cycles["primary_validation_cycle"],
            "cycle_id",
        ].astype(int)

        primary_days = days[
            days["candidate"].eq(candidate)
            & days["cycle_id"].isin(primary_ids)
            & days["observed_state_valid"]
        ].copy()

        macro_rows.append({
            "candidate": candidate,
            "primary_cycles": int(len(primary_ids)),
            "primary_evaluable_days": int(len(primary_days)),
            "crps_mean": float(primary_days["crps"].mean()),
            "median_abs_error_mean": float(
                primary_days["median_abs_error"].mean()
            ),
            "coverage90": float(
                primary_days["covered90"].astype(bool).mean()
            ),
            "coverage90_abs_error": float(
                abs(
                    primary_days["covered90"].astype(bool).mean()
                    - NOMINAL_COVERAGE
                )
            ),
            "persistence_abs_error_mean": float(
                primary_days["persistence_abs_error"].mean()
            ),
            "improves_over_persistence_mae": bool(
                primary_days["median_abs_error"].mean()
                < primary_days["persistence_abs_error"].mean()
            ),
            "negative_slope_fraction_primary_cycles": float(
                cand_cycles.loc[
                    cand_cycles["primary_validation_cycle"],
                    "rain_slope_negative",
                ].astype(bool).mean()
            ),
        })

    return (
        pd.DataFrame(macro_rows),
        cycle_summary,
        days,
    )


def get_year_calendar(
    ledger: pd.DataFrame,
    year_label: str,
) -> pd.DataFrame:
    y = ledger[
        ledger["audit_period"].eq(year_label)
    ].copy().sort_values("date").reset_index(drop=True)

    if len(y) != EXPECTED_DAYS_PER_YEAR:
        raise RuntimeError(
            f"{year_label}: expected 365 days, got {len(y)}."
        )
    return y


def simulate_stress(
    year_df: pd.DataFrame,
    dry_pool: np.ndarray,
    rain_model: dict,
    candidate: str,
    initial_state: float,
    n_trajectories: int,
    env_seed: int,
    hist_max: float,
    margin_max: float,
) -> dict:
    rng = np.random.default_rng(env_seed)
    states = np.empty(
        (n_trajectories, len(year_df)),
        dtype=float,
    )
    states[:, 0] = float(initial_state)

    lower_proj = 0
    upper_proj = 0
    n_transitions = 0

    rain = year_df["rain_day"].to_numpy(dtype=bool)

    for t in range(len(year_df) - 1):
        is_rain = bool(rain[t] or rain[t + 1])

        if is_rain:
            residuals = np.asarray(
                rain_model["residuals"], dtype=float
            )
            idx = rng.integers(
                0, len(residuals), size=n_trajectories
            )

            if candidate == "R0_GLOBAL":
                delta = residuals[idx]
            else:
                mu_delta = (
                    float(rain_model["intercept"])
                    + float(rain_model["slope"])
                    * states[:, t]
                )
                delta = mu_delta + residuals[idx]
        else:
            idx = rng.integers(
                0, len(dry_pool), size=n_trajectories
            )
            delta = dry_pool[idx]

        raw = states[:, t] + delta
        lower_proj += int((raw < 0).sum())
        upper_proj += int((raw > 1).sum())
        n_transitions += n_trajectories

        states[:, t + 1] = np.clip(raw, 0.0, 1.0)

    above_hist = states > hist_max
    above_margin = states > margin_max

    return {
        "state_day_fraction_above_historical_max": float(
            above_hist.mean()
        ),
        "state_day_fraction_above_margin_max": float(
            above_margin.mean()
        ),
        "trajectory_fraction_ever_above_historical_max": float(
            above_hist.any(axis=1).mean()
        ),
        "trajectory_fraction_ever_above_margin_max": float(
            above_margin.any(axis=1).mean()
        ),
        "terminal_distribution": qstats(states[:, -1]),
        "trajectory_max_distribution": qstats(
            states.max(axis=1)
        ),
        "all_state_distribution": qstats(states.ravel()),
        "lower_projection_fraction": float(
            lower_proj / n_transitions
        ),
        "upper_projection_fraction": float(
            upper_proj / n_transitions
        ),
    }


def stress_test(
    ledger: pd.DataFrame,
    x: pd.DataFrame,
    n_trajectories: int,
    env_seed: int,
) -> pd.DataFrame:
    dry_pool = x.loc[
        x["transition_class"].eq("DRY_NATURAL"),
        "delta_L_power_proxy",
    ].to_numpy(dtype=float)

    rain_train = x[
        x["transition_class"].eq("RAIN_AFFECTED")
    ].copy()

    valid_hist = ledger.loc[
        ledger["state_valid"]
        & np.isfinite(ledger["L_power_proxy"]),
        "L_power_proxy",
    ].to_numpy(dtype=float)

    hist_max = float(np.max(valid_hist))
    margin_max = float(hist_max + 0.01)

    rows = []
    for cidx, candidate in enumerate(CANDIDATES):
        model = fit_rain_model(rain_train, candidate)

        for yidx, year_label in enumerate(YEAR_LABELS):
            ydf = get_year_calendar(ledger, year_label)
            hist_first = float(ydf["L_power_proxy"].iloc[0])

            for iidx, (init_mode, init_value) in enumerate([
                ("HISTORICAL_FIRST_DAY", hist_first),
                ("CLEAN_START_ZERO", 0.0),
            ]):
                seed = int(
                    env_seed
                    + 100000 * cidx
                    + 10000 * yidx
                    + 1000 * iidx
                )
                s = simulate_stress(
                    year_df=ydf,
                    dry_pool=dry_pool,
                    rain_model=model,
                    candidate=candidate,
                    initial_state=init_value,
                    n_trajectories=n_trajectories,
                    env_seed=seed,
                    hist_max=hist_max,
                    margin_max=margin_max,
                )
                rows.append({
                    "candidate": candidate,
                    "year_label": year_label,
                    "init_mode": init_mode,
                    "initial_state": float(init_value),
                    "scenario_env_seed": seed,
                    "rain_model_intercept": float(
                        model["intercept"]
                    ),
                    "rain_model_slope": float(model["slope"]),
                    "historical_max": hist_max,
                    "margin_max": margin_max,
                    "state_day_fraction_above_historical_max": s[
                        "state_day_fraction_above_historical_max"
                    ],
                    "state_day_fraction_above_margin_max": s[
                        "state_day_fraction_above_margin_max"
                    ],
                    "trajectory_fraction_ever_above_historical_max": s[
                        "trajectory_fraction_ever_above_historical_max"
                    ],
                    "trajectory_fraction_ever_above_margin_max": s[
                        "trajectory_fraction_ever_above_margin_max"
                    ],
                    "terminal_median": s["terminal_distribution"]["q50"],
                    "terminal_q95": s["terminal_distribution"]["q95"],
                    "trajectory_max_median": s[
                        "trajectory_max_distribution"
                    ]["q50"],
                    "trajectory_max_q95": s[
                        "trajectory_max_distribution"
                    ]["q95"],
                    "all_state_median": s[
                        "all_state_distribution"
                    ]["q50"],
                    "all_state_q95": s[
                        "all_state_distribution"
                    ]["q95"],
                    "lower_projection_fraction": s[
                        "lower_projection_fraction"
                    ],
                    "upper_projection_fraction": s[
                        "upper_projection_fraction"
                    ],
                })
    return pd.DataFrame(rows)


def final_full_model_table(x: pd.DataFrame) -> pd.DataFrame:
    rain = x[
        x["transition_class"].eq("RAIN_AFFECTED")
    ].copy()

    rows = []
    for candidate in CANDIDATES:
        m = fit_rain_model(rain, candidate)
        rows.append({
            "candidate": candidate,
            "n_train": int(m["n_train"]),
            "intercept": float(m["intercept"]),
            "slope": float(m["slope"]),
            "slope_negative": bool(
                m["slope"] < 0
            ) if candidate != "R0_GLOBAL" else True,
            "train_source_L_min": float(
                m["train_source_L_min"]
            ),
            "train_source_L_max": float(
                m["train_source_L_max"]
            ),
            "residual_mean": float(
                np.mean(m["residuals"])
            ),
            "residual_median": float(
                np.median(m["residuals"])
            ),
            "residual_std": float(
                np.std(m["residuals"], ddof=1)
            ),
        })
    return pd.DataFrame(rows)


def choose_candidate(
    one_step_macro: pd.DataFrame,
    loco_macro: pd.DataFrame,
    full_models: pd.DataFrame,
) -> dict:
    merged = one_step_macro.merge(
        loco_macro,
        on="candidate",
        suffixes=("_one_step", "_loco"),
        validate="one_to_one",
    ).merge(
        full_models[
            ["candidate", "slope", "slope_negative"]
        ].rename(
            columns={
                "slope": "full_data_slope",
                "slope_negative": "full_data_slope_negative",
            }
        ),
        on="candidate",
        validate="one_to_one",
    )

    best_one_step = float(
        merged.loc[
            merged["full_finite_support"],
            "crps_macro",
        ].min()
    )
    one_step_limit = best_one_step * (1.0 + ONE_STEP_TOL)

    r0 = merged[
        merged["candidate"].eq("R0_GLOBAL")
    ].iloc[0]

    eligibility_rows = []
    for candidate in CANDIDATES:
        r = merged[
            merged["candidate"].eq(candidate)
        ].iloc[0]

        if candidate == "R0_GLOBAL":
            eligible = True
            reasons = ["BASELINE_REFERENCE"]
        else:
            checks = {
                "full_finite_support": bool(
                    r["full_finite_support"]
                ),
                "negative_slope_both_oot_folds": bool(
                    r["slope_negative_both_folds"]
                ),
                "negative_slope_full_data": bool(
                    r["full_data_slope_negative"]
                ),
                "one_step_within_5pct_best": bool(
                    r["crps_macro"] <= one_step_limit
                ),
                "loco_crps_better_than_R0": bool(
                    r["crps_mean"] < r0["crps_mean"]
                ),
                "loco_mae_better_than_R0": bool(
                    r["median_abs_error_mean"]
                    < r0["median_abs_error_mean"]
                ),
                "loco_coverage_not_worse_than_R0": bool(
                    r["coverage90_abs_error_loco"]
                    <= r0["coverage90_abs_error_loco"]
                ),
            }
            eligible = bool(all(checks.values()))
            reasons = [
                f"{k}={v}" for k, v in checks.items()
            ]

        eligibility_rows.append({
            "candidate": candidate,
            "repair_eligible": eligible,
            "eligibility_details": "; ".join(reasons),
        })

    elig = pd.DataFrame(eligibility_rows)
    merged = merged.merge(
        elig, on="candidate", validate="one_to_one"
    )

    repaired = merged[
        merged["candidate"].isin(
            ["R3_OLS_RESIDUAL", "R4_THEILSEN_RESIDUAL"]
        )
        & merged["repair_eligible"]
    ].copy()

    if len(repaired) == 0:
        recommendation = None
        decision = (
            "NO_REPAIRED_CANDIDATE_MEETS_PREDECLARED_ELIGIBILITY"
        )
    elif len(repaired) == 1:
        recommendation = str(
            repaired.iloc[0]["candidate"]
        )
        decision = "ONE_REPAIRED_CANDIDATE_ELIGIBLE"
    else:
        repaired = repaired.sort_values(
            "crps_mean", ascending=True
        )
        best = repaired.iloc[0]
        second = repaired.iloc[1]

        relative_gap = (
            float(second["crps_mean"] - best["crps_mean"])
            / float(best["crps_mean"])
        )

        if relative_gap <= REPAIRED_TIE_TOL:
            recommendation = "R3_OLS_RESIDUAL"
            decision = (
                "BOTH_ELIGIBLE_AND_WITHIN_2PCT_LOCO_CRPS_"
                "PREFER_OLS_FOR_PARSIMONY"
            )
        else:
            recommendation = str(best["candidate"])
            decision = "BOTH_ELIGIBLE_CHOOSE_LOWER_LOCO_CRPS"

    return {
        "best_one_step_crps": best_one_step,
        "one_step_5pct_limit": float(one_step_limit),
        "R0_loco_crps": float(r0["crps_mean"]),
        "R0_loco_mae": float(r0["median_abs_error_mean"]),
        "R0_loco_coverage90": float(r0["coverage90"]),
        "recommended_repaired_candidate": recommendation,
        "decision": decision,
        "eligibility_table": merged.to_dict(
            orient="records"
        ),
        "important_note": (
            "365-day WAIT stress is diagnostic only because the observed "
            "maximum no-manual-clean interval is ~33 days. It does not "
            "override the one-step + observed-horizon recursive gates."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-1C-1R-B rain-repair candidate validation."
    )
    p.add_argument("--master-ledger", required=True, type=Path)
    p.add_argument("--transition-audit", required=True, type=Path)
    p.add_argument("--rain-event-audit", required=True, type=Path)
    p.add_argument(
        "--n-cycle-trajectories",
        type=int,
        default=DEFAULT_N_CYCLE_TRAJECTORIES,
    )
    p.add_argument(
        "--n-stress-trajectories",
        type=int,
        default=DEFAULT_N_STRESS_TRAJECTORIES,
    )
    p.add_argument(
        "--env-seed",
        type=int,
        default=DEFAULT_ENV_SEED,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1c1r_b_rain_repair_candidate_validation_v1"
        ),
    )
    args = p.parse_args()

    if args.n_cycle_trajectories < 200:
        raise ValueError("--n-cycle-trajectories must be >=200.")
    if args.n_stress_trajectories < 200:
        raise ValueError("--n-stress-trajectories must be >=200.")

    lp = args.master_ledger.expanduser().resolve()
    tp = args.transition_audit.expanduser().resolve()
    rp = args.rain_event_audit.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()

    for path in [lp, tp, rp]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/11] Load frozen timeline / transition / rain-event assets")
    ledger = load_ledger(lp)
    transitions = load_transitions(tp)
    rain_events = load_rain_events(rp)

    if len(ledger) != EXPECTED_CALENDAR_DAYS:
        raise RuntimeError("Calendar day count mismatch.")
    if int(ledger["state_valid"].sum()) != EXPECTED_VALID_STATES:
        raise RuntimeError("Valid-state count mismatch.")

    print("[2/11] Attach source/destination hidden-state context")
    x = attach_context(transitions, ledger)

    dry_n = int(
        x["transition_class"].eq("DRY_NATURAL").sum()
    )
    rain_n = int(
        x["transition_class"].eq("RAIN_AFFECTED").sum()
    )
    if dry_n != EXPECTED_DRY or rain_n != EXPECTED_RAIN:
        raise RuntimeError(
            f"Transition counts mismatch: dry={dry_n}, rain={rain_n}"
        )

    print("[3/11] Run bidirectional one-step OOT validation")
    one_folds, one_preds, one_params = one_step_cv(x)
    one_macro = macro_one_step(one_folds)

    print("[4/11] Fit full-data candidate models")
    full_models = final_full_model_table(x)

    print("[5/11] Audit event-level rain direction as physical sanity reference")
    rr = rain_events[
        rain_events["resolved"]
        & (~rain_events["cleaning_confounded"])
    ].copy()
    rr = rr[
        np.isfinite(
            rr[["S_pre", "S_post"]].to_numpy(dtype=float)
        ).all(axis=1)
    ].copy()
    rr["removal"] = rr["S_pre"] - rr["S_post"]

    event_rho = float(
        rr["S_pre"].corr(rr["removal"], method="spearman")
    )
    event_slope = float(
        np.polyfit(
            rr["S_pre"].to_numpy(dtype=float),
            (rr["S_post"] - rr["S_pre"]).to_numpy(dtype=float),
            1,
        )[0]
    )

    print("[6/11] Run leave-one-cleaning-cycle-out recursive validation")
    loco_macro, loco_cycles, loco_days = loco_recursive_validation(
        ledger=ledger,
        x=x,
        n_trajectories=args.n_cycle_trajectories,
        env_seed=args.env_seed,
    )

    print("[7/11] Apply predeclared repair eligibility rules")
    selection = choose_candidate(
        one_step_macro=one_macro,
        loco_macro=loco_macro,
        full_models=full_models,
    )

    print("[8/11] Run 365-day WAIT-only extrapolation stress test")
    stress = stress_test(
        ledger=ledger,
        x=x,
        n_trajectories=args.n_stress_trajectories,
        env_seed=args.env_seed + 500000,
    )

    print("[9/11] Build primary technical/scientific gates")
    primary_cycles_n = int(
        loco_macro["primary_cycles"].min()
    )

    technical_gates = {
        "calendar_count_pass": bool(
            len(ledger) == EXPECTED_CALENDAR_DAYS
        ),
        "valid_state_count_pass": bool(
            int(ledger["state_valid"].sum())
            == EXPECTED_VALID_STATES
        ),
        "dry_count_pass": bool(dry_n == EXPECTED_DRY),
        "rain_count_pass": bool(rain_n == EXPECTED_RAIN),
        "clean_day_count_pass": bool(
            int(ledger["modb_manual_cleaning_day"].sum())
            == EXPECTED_CLEAN_DAYS
        ),
        "one_step_all_candidates_finite_pass": bool(
            one_macro["full_finite_support"].all()
        ),
        "primary_cycle_count_pass": bool(
            primary_cycles_n >= MIN_PRIMARY_CYCLES
        ),
        "event_level_rain_direction_pass": bool(
            event_rho > 0 and event_slope < 0
        ),
    }
    technical_gates["all_technical_gates_pass"] = bool(
        all(technical_gates.values())
    )

    repaired_recommended = (
        selection["recommended_repaired_candidate"]
        is not None
    )

    print("[10/11] Build audit summary")
    audit_summary = {
        "stage": "P2-1C-1R-B",
        "candidate_validation_only": True,
        "dry_kernel": "D0_GLOBAL_EMPIRICAL_FIXED",
        "rain_candidates": list(CANDIDATES),
        "clean_action_built": False,
        "online_perception_built": False,
        "reward_built": False,
        "rl_started": False,
        "validation_layers": {
            "layer1": "BIDIRECTIONAL_ONE_STEP_OOT",
            "layer2": "LEAVE_ONE_CLEANING_CYCLE_OUT_RECURSIVE",
            "layer3": "365_DAY_WAIT_ONLY_EXTRAPOLATION_STRESS",
        },
        "predeclared_selection_rules": {
            "one_step_CRPS_tolerance_from_best": ONE_STEP_TOL,
            "repaired_candidate_must_improve_R0_LOCO_CRPS": True,
            "repaired_candidate_must_improve_R0_LOCO_MAE": True,
            "repaired_candidate_LOCO_coverage_must_not_be_worse": True,
            "negative_slope_required_both_OOT_folds": True,
            "negative_slope_required_full_data": True,
            "R3_R4_tie_tolerance_LOCO_CRPS": REPAIRED_TIE_TOL,
            "stress_test_is_eligibility_gate": False,
        },
        "one_step_macro": one_macro.to_dict(orient="records"),
        "full_data_models": full_models.to_dict(orient="records"),
        "event_level_sanity": {
            "resolved_unconfounded_events": int(len(rr)),
            "spearman_Spre_vs_removal": event_rho,
            "OLS_slope_Spre_vs_deltaS": event_slope,
            "direction_consistent_with_state_dependent_cleaning": bool(
                event_rho > 0 and event_slope < 0
            ),
        },
        "loco_recursive_macro": loco_macro.to_dict(orient="records"),
        "candidate_selection": selection,
        "stress_test_summary": stress.to_dict(orient="records"),
        "technical_gates": technical_gates,
        "scientific_readiness": {
            "repaired_candidate_recommended": repaired_recommended,
            "ready_to_freeze_repaired_rain_kernel": bool(
                repaired_recommended
                and technical_gates["all_technical_gates_pass"]
            ),
        },
        "next_step_if_repaired_candidate_recommended": (
            "Scientifically review the selected candidate and its 365-day "
            "stress behaviour. If acceptable, freeze the repaired rain kernel "
            "and rerun P2-1C-1 WAIT-only audit using the repaired dynamics. "
            "Only after that should P2-1C-2 online counterfactual perception begin."
        ),
        "next_step_if_none_recommended": (
            "Do not add model complexity automatically. Inspect which eligibility "
            "gate failed and reconsider whether the 365-day counterfactual horizon "
            "needs a different physically supported environment formulation."
        ),
    }

    print("[11/11] Write outputs")
    out.mkdir(parents=True, exist_ok=True)

    one_folds.to_csv(
        out / "one_step_oot_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    one_macro.to_csv(
        out / "one_step_oot_candidate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    one_preds.to_csv(
        out / "one_step_oot_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    one_params.to_csv(
        out / "one_step_oot_fitted_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    full_models.to_csv(
        out / "full_data_candidate_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loco_macro.to_csv(
        out / "loco_recursive_candidate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loco_cycles.to_csv(
        out / "loco_recursive_cycle_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loco_days.to_csv(
        out / "loco_recursive_day_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stress.to_csv(
        out / "wait365_stress_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            audit_summary, f,
            ensure_ascii=False, indent=2
        )

    print(out / "one_step_oot_candidate_summary.csv")
    print(out / "full_data_candidate_parameters.csv")
    print(out / "loco_recursive_candidate_summary.csv")
    print(out / "wait365_stress_summary.csv")
    print(out / "audit_summary.json")
    print(
        "IMPORTANT: candidate validation only. Do NOT add CLEAN, reward, "
        "online perception, Gym, or PPO before scientific review."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
