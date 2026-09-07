#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1C-1R-A
Long-horizon failure diagnosis for recursive natural dynamics.

CONTEXT
-------
P2-1B-2 showed that D0_GLOBAL (dry) and R0_GLOBAL (rain) are reasonable
ONE-STEP empirical stochastic benchmarks.

P2-1C-1 then showed that recursively replaying those kernels for a 365-day
WAIT-only episode produces severe support drift:
the model spends far too much time above the historical WAPP/perception
deployment domain.

This script does NOT "repair" the model. It diagnoses *why* the recursive
failure occurs and distinguishes two fundamentally different possibilities:

A) D0/R0 are already poor over historically observed no-clean intervals.
   -> Natural dynamics themselves need repair before RL.

B) D0/R0 are acceptable over historically observed cleaning-cycle horizons,
   but fail only when extrapolated to a 365-day no-clean counterfactual.
   -> The main problem is long-horizon extrapolation / state-support mismatch,
      not necessarily one-step dynamics.

PRIMARY DIAGNOSTICS
-------------------
1) Daily transition state dependence:
   delta_L ~ source_L
   for dry and rain transitions, overall and YEAR1/YEAR2.

2) Robust slope diagnostics:
   OLS slope, R^2, Pearson, Spearman, Theil-Sen slope.

3) Source-state support:
   transition counts and delta distributions in fixed source-L bands.

4) Event-level rain response:
   S_post-S_pre ~ S_pre, including cross-year direction consistency.

5) Historical intervention horizon:
   authoritative ModB cleaning-cycle lengths and the longest observed
   no-manual-clean interval.

6) Leave-one-cleaning-cycle-out recursive validation:
   for each historical post-clean cycle:
      - exclude that cycle's transitions from D0/R0 pools;
      - start from the observed post-clean state;
      - replay the actual historical rain calendar for that cycle;
      - recursively simulate an empirical stochastic state distribution;
      - compare the predictive distribution against observed WAPP states.
   This tests recursive behaviour only over horizons actually represented in
   the field data, without using the target cycle's transitions in its kernels.

IMPORTANT
---------
- No historical-max clipping is introduced.
- No CLEAN action model is added.
- No online perception, reward, Gym, or PPO is built.
- This is diagnosis only; it does not auto-select a repaired kernel.
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
EXPECTED_CLEAN_DAYS = 26
EXPECTED_DRY = 494
EXPECTED_RAIN = 201
EXPECTED_RAIN_EVENTS = 69

YEAR2_START = pd.Timestamp("2022-08-09")

STATE_BINS = [-np.inf, 0.01, 0.03, 0.06, 0.10, 0.16, np.inf]
STATE_LABELS = [
    "<0.01",
    "0.01-0.03",
    "0.03-0.06",
    "0.06-0.10",
    "0.10-0.16",
    ">=0.16",
]

DEFAULT_N_CYCLE_TRAJECTORIES = 1000
DEFAULT_ENV_SEED = 410001

# Technical support gates for leave-one-cycle-out kernels.
MIN_LOO_DRY_POOL = 350
MIN_LOO_RAIN_POOL = 120
MIN_PRIMARY_CYCLES = 18

# Only used to label diagnostics, not to silently pass/fail scientific claims.
RECURSIVE_COVERAGE_TARGET = 0.90


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


def empirical_crps(samples: np.ndarray, y: float) -> float:
    """
    O(n log n) exact empirical CRPS:
      mean|X-y| - 0.5 E|X-X'|
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    first = float(np.mean(np.abs(x - y)))
    i = np.arange(1, n + 1, dtype=float)
    second_half_pairwise = float(
        np.sum((2.0 * i - n - 1.0) * x) / (n * n)
    )
    return first - second_half_pairwise


def ols_stats(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.std(x) <= 0:
        return {
            "n": int(len(x)),
            "intercept": np.nan,
            "slope": np.nan,
            "r2": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "theil_sen_slope": np.nan,
        }

    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pearson = pd.Series(x).corr(pd.Series(y), method="pearson")
    spearman = pd.Series(x).corr(pd.Series(y), method="spearman")

    # Robust Theil-Sen slope: median of all valid pairwise slopes.
    # n <= 494 here, so O(n^2) remains cheap.
    slopes = []
    for i in range(len(x) - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        good = np.abs(dx) > 1e-15
        if np.any(good):
            slopes.extend((dy[good] / dx[good]).tolist())
    ts = float(np.median(slopes)) if slopes else np.nan

    return {
        "n": int(len(x)),
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "pearson": float(pearson) if np.isfinite(pearson) else np.nan,
        "spearman": float(spearman) if np.isfinite(spearman) else np.nan,
        "theil_sen_slope": ts,
    }


def load_ledger(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date",
        "audit_period",
        "state_valid",
        "L_power_proxy",
        "rain_day",
        "modb_manual_cleaning_day",
        "scheduled_maintenance",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Master ledger missing: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    for c in [
        "state_valid",
        "rain_day",
        "modb_manual_cleaning_day",
        "scheduled_maintenance",
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
        "transition_index",
        "source_date",
        "dest_date",
        "source_state_valid",
        "dest_state_valid",
        "source_manual_clean_day",
        "dest_manual_clean_day",
        "source_rain_day",
        "dest_rain_day",
        "source_scheduled_maintenance",
        "dest_scheduled_maintenance",
        "rain_adjacent",
        "maintenance_adjacent",
        "transition_class",
        "delta_L_power_proxy",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Transition audit missing: {sorted(missing)}")

    for c in ["source_date", "dest_date"]:
        df[c] = pd.to_datetime(df[c], errors="raise").dt.normalize()

    for c in [
        "source_state_valid",
        "dest_state_valid",
        "source_manual_clean_day",
        "dest_manual_clean_day",
        "source_rain_day",
        "dest_rain_day",
        "source_scheduled_maintenance",
        "dest_scheduled_maintenance",
        "rain_adjacent",
        "maintenance_adjacent",
    ]:
        df[c] = parse_bool_series(df[c], c)

    df["delta_L_power_proxy"] = pd.to_numeric(
        df["delta_L_power_proxy"], errors="coerce"
    )
    return df.sort_values("transition_index").reset_index(drop=True)


def load_rain_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "rain_event_id",
        "start_date",
        "end_date",
        "rain_days",
        "rain_mm_total",
        "S_pre",
        "S_post",
        "delta_S_post_minus_pre",
        "resolved",
        "cleaning_confounded",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Rain-event audit missing: {sorted(missing)}")

    for c in ["start_date", "end_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()

    for c in ["resolved", "cleaning_confounded"]:
        df[c] = parse_bool_series(df[c], c)

    for c in [
        "rain_days",
        "rain_mm_total",
        "S_pre",
        "S_post",
        "delta_S_post_minus_pre",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def attach_state_context(
    transitions: pd.DataFrame, ledger: pd.DataFrame
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

    x["state_band"] = pd.cut(
        x["source_L"],
        bins=STATE_BINS,
        labels=STATE_LABELS,
        right=False,
    ).astype(str)

    rain_pattern = np.full(
        len(x), "NOT_RAIN_AFFECTED", dtype=object
    )
    sr = x["source_rain_day"].to_numpy(dtype=bool)
    dr = x["dest_rain_day"].to_numpy(dtype=bool)
    rain_pattern[(~sr) & dr] = "ONSET"
    rain_pattern[sr & dr] = "CONTINUATION"
    rain_pattern[sr & (~dr)] = "POST_RAIN"
    x["rain_pattern"] = rain_pattern

    return x


def dependence_rows(
    data: pd.DataFrame,
    family: str,
    x_col: str,
    y_col: str,
) -> list[dict]:
    rows = []

    subsets = [("ALL", data)]
    if "source_period" in data.columns:
        for period in ["YEAR1", "YEAR2"]:
            subsets.append(
                (period, data[data["source_period"].eq(period)])
            )

    for label, g in subsets:
        g = g[
            np.isfinite(g[x_col])
            & np.isfinite(g[y_col])
        ].copy()

        st = ols_stats(
            g[x_col].to_numpy(dtype=float),
            g[y_col].to_numpy(dtype=float),
        )

        rows.append({
            "family": family,
            "subset": label,
            "x": x_col,
            "y": y_col,
            **st,
            "x_distribution": json.dumps(
                qstats(g[x_col]), ensure_ascii=False
            ),
            "y_distribution": json.dumps(
                qstats(g[y_col]), ensure_ascii=False
            ),
        })
    return rows


def state_band_summary(data: pd.DataFrame, family: str) -> pd.DataFrame:
    rows = []
    for period_label, pg in [
        ("ALL", data),
        ("YEAR1", data[data["source_period"].eq("YEAR1")]),
        ("YEAR2", data[data["source_period"].eq("YEAR2")]),
    ]:
        for band in STATE_LABELS:
            g = pg[pg["state_band"].eq(band)]
            if len(g) == 0:
                continue
            d = g["delta_L_power_proxy"]
            rows.append({
                "family": family,
                "period": period_label,
                "state_band": band,
                "n": int(len(g)),
                "source_L_mean": float(g["source_L"].mean()),
                "source_L_median": float(g["source_L"].median()),
                "delta_mean": float(d.mean()),
                "delta_median": float(d.median()),
                "delta_q05": float(d.quantile(0.05)),
                "delta_q95": float(d.quantile(0.95)),
                "fraction_delta_negative": float((d < 0).mean()),
                "fraction_delta_positive": float((d > 0).mean()),
            })
    return pd.DataFrame(rows)


def build_cleaning_cycles(ledger: pd.DataFrame) -> pd.DataFrame:
    clean_dates = ledger.loc[
        ledger["modb_manual_cleaning_day"], "date"
    ].sort_values().tolist()

    if len(clean_dates) != EXPECTED_CLEAN_DAYS:
        raise RuntimeError(
            f"Expected {EXPECTED_CLEAN_DAYS} manual clean dates, "
            f"got {len(clean_dates)}."
        )

    rows = []
    for i, start in enumerate(clean_dates):
        if i + 1 < len(clean_dates):
            next_clean = clean_dates[i + 1]
            end = next_clean - pd.Timedelta(days=1)
            days_to_next_clean = int((next_clean - start).days)
            no_clean_days_between = max(days_to_next_clean - 1, 0)
        else:
            next_clean = pd.NaT
            end = ledger["date"].max()
            days_to_next_clean = np.nan
            no_clean_days_between = int((end - start).days)

        seg = ledger[
            (ledger["date"] >= start)
            & (ledger["date"] <= end)
        ].copy()

        valid_L = seg.loc[
            seg["state_valid"] & np.isfinite(seg["L_power_proxy"]),
            "L_power_proxy",
        ]

        start_row = ledger[ledger["date"].eq(start)].iloc[0]
        rows.append({
            "cycle_id": int(i + 1),
            "cycle_start": start,
            "cycle_end": end,
            "next_manual_clean": next_clean,
            "cycle_calendar_days": int(len(seg)),
            "days_to_next_clean": (
                int(days_to_next_clean)
                if np.isfinite(days_to_next_clean)
                else np.nan
            ),
            "no_clean_days_between_manual_cleans": int(no_clean_days_between),
            "start_period": str(start_row["audit_period"]),
            "start_L": float(start_row["L_power_proxy"])
            if np.isfinite(start_row["L_power_proxy"]) else np.nan,
            "valid_observed_state_days": int(len(valid_L)),
            "rain_days": int(seg["rain_day"].sum()),
            "maintenance_days": int(seg["scheduled_maintenance"].sum()),
            "contains_invalid_state": bool((~seg["state_valid"]).any()),
            "observed_L_max": float(valid_L.max()) if len(valid_L) else np.nan,
            "observed_L_terminal": float(valid_L.iloc[-1]) if len(valid_L) else np.nan,
        })

    return pd.DataFrame(rows)


def simulate_cycle_loo(
    cycle_row: pd.Series,
    ledger: pd.DataFrame,
    transitions: pd.DataFrame,
    n_trajectories: int,
    env_seed: int,
) -> tuple[pd.DataFrame, dict]:
    start = pd.Timestamp(cycle_row["cycle_start"])
    end = pd.Timestamp(cycle_row["cycle_end"])

    seg = ledger[
        (ledger["date"] >= start)
        & (ledger["date"] <= end)
    ].copy().sort_values("date").reset_index(drop=True)

    # Leave-one-cycle-out: remove target cycle source dates from training pools.
    train = transitions[
        ~(
            (transitions["source_date"] >= start)
            & (transitions["source_date"] <= end)
        )
    ].copy()

    dry_pool = train.loc[
        train["transition_class"].eq("DRY_NATURAL"),
        "delta_L_power_proxy",
    ].to_numpy(dtype=float)

    rain_pool = train.loc[
        train["transition_class"].eq("RAIN_AFFECTED"),
        "delta_L_power_proxy",
    ].to_numpy(dtype=float)

    if not np.isfinite(dry_pool).all() or not np.isfinite(rain_pool).all():
        raise RuntimeError("LOO kernel contains non-finite values.")

    start_L = float(cycle_row["start_L"])
    if not np.isfinite(start_L):
        raise RuntimeError(
            f"Cycle {cycle_row['cycle_id']}: start state is not finite."
        )

    rng = np.random.default_rng(env_seed)

    n_days = len(seg)
    states = np.empty((n_trajectories, n_days), dtype=float)
    states[:, 0] = start_L

    rain = seg["rain_day"].to_numpy(dtype=bool)

    for t in range(n_days - 1):
        is_rain_affected = bool(rain[t] or rain[t + 1])
        pool = rain_pool if is_rain_affected else dry_pool
        idx = rng.integers(0, len(pool), size=n_trajectories)
        raw = states[:, t] + pool[idx]
        states[:, t + 1] = np.clip(raw, 0.0, 1.0)

    rows = []
    for d in range(1, n_days):
        obs_valid = bool(seg.loc[d, "state_valid"])
        obs = float(seg.loc[d, "L_power_proxy"]) if obs_valid else np.nan

        pred = states[:, d]
        q05, q50, q95 = np.quantile(pred, [0.05, 0.50, 0.95])

        rec = {
            "cycle_id": int(cycle_row["cycle_id"]),
            "date": seg.loc[d, "date"],
            "day_since_clean": int(d),
            "observed_state_valid": obs_valid and np.isfinite(obs),
            "observed_L": obs,
            "pred_q05": float(q05),
            "pred_q50": float(q50),
            "pred_q95": float(q95),
            "pred_mean": float(np.mean(pred)),
            "pred_max": float(np.max(pred)),
            "loo_dry_pool_n": int(len(dry_pool)),
            "loo_rain_pool_n": int(len(rain_pool)),
        }

        if rec["observed_state_valid"]:
            rec.update({
                "crps": empirical_crps(pred, obs),
                "median_abs_error": float(abs(q50 - obs)),
                "covered90": bool(q05 <= obs <= q95),
                "zero_start_persistence_abs_error": float(abs(start_L - obs)),
            })
        else:
            rec.update({
                "crps": np.nan,
                "median_abs_error": np.nan,
                "covered90": False,
                "zero_start_persistence_abs_error": np.nan,
            })
        rows.append(rec)

    day_df = pd.DataFrame(rows)

    valid = day_df[day_df["observed_state_valid"]].copy()

    # Primary cycle = no scheduled-maintenance ambiguity.
    primary_cycle = bool(int(cycle_row["maintenance_days"]) == 0)

    summary = {
        "cycle_id": int(cycle_row["cycle_id"]),
        "cycle_start": start.strftime("%Y-%m-%d"),
        "cycle_end": end.strftime("%Y-%m-%d"),
        "cycle_calendar_days": int(n_days),
        "primary_validation_cycle": primary_cycle,
        "contains_invalid_state": bool(cycle_row["contains_invalid_state"]),
        "maintenance_days": int(cycle_row["maintenance_days"]),
        "loo_dry_pool_n": int(len(dry_pool)),
        "loo_rain_pool_n": int(len(rain_pool)),
        "loo_pool_gate_pass": bool(
            len(dry_pool) >= MIN_LOO_DRY_POOL
            and len(rain_pool) >= MIN_LOO_RAIN_POOL
        ),
        "evaluable_days": int(len(valid)),
        "crps_mean": float(valid["crps"].mean()) if len(valid) else np.nan,
        "median_abs_error_mean": float(
            valid["median_abs_error"].mean()
        ) if len(valid) else np.nan,
        "coverage90": float(
            valid["covered90"].astype(bool).mean()
        ) if len(valid) else np.nan,
        "persistence_abs_error_mean": float(
            valid["zero_start_persistence_abs_error"].mean()
        ) if len(valid) else np.nan,
        "median_prediction_improves_over_persistence": bool(
            len(valid)
            and valid["median_abs_error"].mean()
            < valid["zero_start_persistence_abs_error"].mean()
        ),
        "predicted_state_max_over_all_trajectories": float(
            states.max()
        ),
    }

    return day_df, summary


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-1C-1R-A recursive failure diagnosis."
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
        "--env-seed",
        type=int,
        default=DEFAULT_ENV_SEED,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1c1r_a_recursive_failure_diagnosis_v1"
        ),
    )
    args = p.parse_args()

    if args.n_cycle_trajectories < 200:
        raise ValueError("--n-cycle-trajectories must be >= 200.")

    lp = args.master_ledger.expanduser().resolve()
    tp = args.transition_audit.expanduser().resolve()
    rp = args.rain_event_audit.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()

    for path in [lp, tp, rp]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/10] Load frozen P2-1A/P2-0B assets")
    ledger = load_ledger(lp)
    transitions = load_transitions(tp)
    rain_events = load_rain_events(rp)

    if len(ledger) != EXPECTED_CALENDAR_DAYS:
        raise RuntimeError(
            f"Expected {EXPECTED_CALENDAR_DAYS} ledger days, got {len(ledger)}"
        )

    if int(ledger["state_valid"].sum()) != EXPECTED_VALID_STATES:
        raise RuntimeError(
            f"Expected {EXPECTED_VALID_STATES} valid states."
        )

    print("[2/10] Attach source-state context to dry/rain transitions")
    x = attach_state_context(transitions, ledger)
    dry = x[x["transition_class"].eq("DRY_NATURAL")].copy()
    rain = x[x["transition_class"].eq("RAIN_AFFECTED")].copy()

    if len(dry) != EXPECTED_DRY:
        raise RuntimeError(f"Expected {EXPECTED_DRY} dry rows, got {len(dry)}")
    if len(rain) != EXPECTED_RAIN:
        raise RuntimeError(f"Expected {EXPECTED_RAIN} rain rows, got {len(rain)}")

    print("[3/10] Diagnose signed state dependence of daily transitions")
    dependence = []
    dependence += dependence_rows(
        dry, "DRY_DAILY", "source_L", "delta_L_power_proxy"
    )
    dependence += dependence_rows(
        rain, "RAIN_DAILY", "source_L", "delta_L_power_proxy"
    )

    dep_df = pd.DataFrame(dependence)

    print("[4/10] Audit source-state support bands")
    band_df = pd.concat(
        [
            state_band_summary(dry, "DRY_DAILY"),
            state_band_summary(rain, "RAIN_DAILY"),
        ],
        ignore_index=True,
    )

    print("[5/10] Diagnose event-level rain state dependence")
    rr = rain_events[
        rain_events["resolved"]
        & (~rain_events["cleaning_confounded"])
    ].copy()
    finite = np.isfinite(
        rr[["S_pre", "S_post", "delta_S_post_minus_pre"]].to_numpy(dtype=float)
    ).all(axis=1)
    rr = rr.loc[finite].copy()

    rr["source_period"] = np.where(
        rr["start_date"] < YEAR2_START, "YEAR1", "YEAR2"
    )
    rr["delta_S"] = rr["S_post"] - rr["S_pre"]
    rr["removal"] = rr["S_pre"] - rr["S_post"]

    rain_event_dep = pd.DataFrame(
        dependence_rows(
            rr,
            "RAIN_EVENT",
            "S_pre",
            "delta_S",
        )
    )

    print("[6/10] Build authoritative manual-cleaning cycle audit")
    cycle_df = build_cleaning_cycles(ledger)

    clean_dates = ledger.loc[
        ledger["modb_manual_cleaning_day"], "date"
    ].sort_values()

    pre_first_clean_days = int(
        (clean_dates.iloc[0] - ledger["date"].min()).days
    )

    gap_values = cycle_df[
        "no_clean_days_between_manual_cleans"
    ].to_numpy(dtype=float)

    cycle_horizon_audit = {
        "manual_clean_days": int(len(clean_dates)),
        "pre_first_clean_days_from_dataset_start": pre_first_clean_days,
        "cycle_calendar_days_distribution": qstats(
            cycle_df["cycle_calendar_days"]
        ),
        "no_clean_days_between_manual_cleans_distribution": qstats(
            gap_values
        ),
        "max_no_clean_days_between_manual_cleans": int(
            np.nanmax(gap_values)
        ),
        "cycles_ge_30_calendar_days": int(
            (cycle_df["cycle_calendar_days"] >= 30).sum()
        ),
        "cycles_ge_45_calendar_days": int(
            (cycle_df["cycle_calendar_days"] >= 45).sum()
        ),
        "cycles_ge_60_calendar_days": int(
            (cycle_df["cycle_calendar_days"] >= 60).sum()
        ),
        "cycles_ge_90_calendar_days": int(
            (cycle_df["cycle_calendar_days"] >= 90).sum()
        ),
        "key_interpretation": (
            "365-day WAIT-only is directly unsupported by historical manual-clean "
            "intervals if the observed maximum no-clean interval is far shorter."
        ),
    }

    print("[7/10] Run leave-one-cleaning-cycle-out recursive validation")
    day_parts = []
    cycle_summaries = []

    for i, row in cycle_df.iterrows():
        scenario_seed = int(args.env_seed + 1000 * int(row["cycle_id"]))
        day_part, cyc_sum = simulate_cycle_loo(
            cycle_row=row,
            ledger=ledger,
            transitions=transitions,
            n_trajectories=args.n_cycle_trajectories,
            env_seed=scenario_seed,
        )
        cyc_sum["scenario_env_seed"] = scenario_seed
        day_parts.append(day_part)
        cycle_summaries.append(cyc_sum)

    cycle_day_df = pd.concat(day_parts, ignore_index=True)
    cycle_val_df = pd.DataFrame(cycle_summaries)

    print("[8/10] Aggregate recursive validation over observed horizons")
    primary_cycle_ids = cycle_val_df.loc[
        cycle_val_df["primary_validation_cycle"],
        "cycle_id",
    ].astype(int)

    primary_days = cycle_day_df[
        cycle_day_df["cycle_id"].isin(primary_cycle_ids)
        & cycle_day_df["observed_state_valid"]
    ].copy()

    all_valid_days = cycle_day_df[
        cycle_day_df["observed_state_valid"]
    ].copy()

    primary_cycles_n = int(
        cycle_val_df["primary_validation_cycle"].sum()
    )
    loo_pool_gate = bool(
        cycle_val_df["loo_pool_gate_pass"].all()
    )

    recursive_primary = {
        "primary_cycle_definition": (
            "post-clean cycles with zero scheduled-maintenance days; invalid "
            "observation days are omitted from scoring but remain in simulation"
        ),
        "primary_cycles": primary_cycles_n,
        "primary_evaluable_days": int(len(primary_days)),
        "mean_crps": float(primary_days["crps"].mean())
        if len(primary_days) else np.nan,
        "mean_median_abs_error": float(
            primary_days["median_abs_error"].mean()
        ) if len(primary_days) else np.nan,
        "coverage90": float(
            primary_days["covered90"].astype(bool).mean()
        ) if len(primary_days) else np.nan,
        "coverage90_abs_error_from_nominal": float(
            abs(primary_days["covered90"].astype(bool).mean()
                - RECURSIVE_COVERAGE_TARGET)
        ) if len(primary_days) else np.nan,
        "persistence_abs_error_mean": float(
            primary_days["zero_start_persistence_abs_error"].mean()
        ) if len(primary_days) else np.nan,
        "median_prediction_improves_over_persistence": bool(
            len(primary_days)
            and primary_days["median_abs_error"].mean()
            < primary_days["zero_start_persistence_abs_error"].mean()
        ),
        "all_cycle_evaluable_days": int(len(all_valid_days)),
        "all_cycle_coverage90": float(
            all_valid_days["covered90"].astype(bool).mean()
        ) if len(all_valid_days) else np.nan,
        "all_cycle_mean_median_abs_error": float(
            all_valid_days["median_abs_error"].mean()
        ) if len(all_valid_days) else np.nan,
        "loo_kernel_pool_gate_pass": loo_pool_gate,
    }

    print("[9/10] Build diagnosis and technical gates")
    rain_daily_all = dep_df[
        (dep_df["family"].eq("RAIN_DAILY"))
        & (dep_df["subset"].eq("ALL"))
    ].iloc[0]
    rain_daily_y1 = dep_df[
        (dep_df["family"].eq("RAIN_DAILY"))
        & (dep_df["subset"].eq("YEAR1"))
    ].iloc[0]
    rain_daily_y2 = dep_df[
        (dep_df["family"].eq("RAIN_DAILY"))
        & (dep_df["subset"].eq("YEAR2"))
    ].iloc[0]

    rain_event_all = rain_event_dep[
        rain_event_dep["subset"].eq("ALL")
    ].iloc[0]

    daily_rain_state_effect = {
        "overall_slope": float(rain_daily_all["slope"]),
        "overall_spearman": float(rain_daily_all["spearman"]),
        "overall_theil_sen_slope": float(
            rain_daily_all["theil_sen_slope"]
        ),
        "year1_slope": float(rain_daily_y1["slope"]),
        "year2_slope": float(rain_daily_y2["slope"]),
        "year1_spearman": float(rain_daily_y1["spearman"]),
        "year2_spearman": float(rain_daily_y2["spearman"]),
        "slope_same_negative_direction_across_years": bool(
            rain_daily_y1["slope"] < 0
            and rain_daily_y2["slope"] < 0
        ),
        "spearman_same_negative_direction_across_years": bool(
            rain_daily_y1["spearman"] < 0
            and rain_daily_y2["spearman"] < 0
        ),
    }

    event_rain_state_effect = {
        "n": int(rain_event_all["n"]),
        "slope": float(rain_event_all["slope"]),
        "spearman_Spre_vs_deltaS": float(
            rain_event_all["spearman"]
        ),
        "spearman_Spre_vs_removal": float(
            -rain_event_all["spearman"]
        ),
        "theil_sen_slope": float(
            rain_event_all["theil_sen_slope"]
        ),
    }

    technical_gates = {
        "calendar_count_pass": bool(
            len(ledger) == EXPECTED_CALENDAR_DAYS
        ),
        "valid_state_count_pass": bool(
            int(ledger["state_valid"].sum()) == EXPECTED_VALID_STATES
        ),
        "clean_day_count_pass": bool(
            int(ledger["modb_manual_cleaning_day"].sum())
            == EXPECTED_CLEAN_DAYS
        ),
        "dry_transition_count_pass": bool(
            len(dry) == EXPECTED_DRY
        ),
        "rain_transition_count_pass": bool(
            len(rain) == EXPECTED_RAIN
        ),
        "rain_event_count_pass": bool(
            len(rain_events) == EXPECTED_RAIN_EVENTS
        ),
        "loo_kernel_pool_pass": loo_pool_gate,
        "primary_cycle_support_pass": bool(
            primary_cycles_n >= MIN_PRIMARY_CYCLES
        ),
    }
    technical_gates["all_technical_gates_pass"] = bool(
        all(technical_gates.values())
    )

    # Diagnosis labels only. Do NOT auto-freeze repaired dynamics.
    if (
        recursive_primary["median_prediction_improves_over_persistence"]
        and recursive_primary["coverage90"] >= 0.80
    ):
        observed_horizon_label = (
            "CURRENT_KERNELS_HAVE_MEANINGFUL_RECURSIVE_SKILL_ON_OBSERVED_HORIZONS"
        )
    else:
        observed_horizon_label = (
            "CURRENT_KERNELS_WEAK_EVEN_ON_OBSERVED_HORIZONS"
        )

    if (
        daily_rain_state_effect["overall_slope"] < 0
        and event_rain_state_effect["slope"] < 0
    ):
        rain_state_dependence_label = (
            "DAILY_AND_EVENT_LEVEL_BOTH_SUPPORT_NEGATIVE_DELTA_WITH_HIGHER_STATE"
        )
    else:
        rain_state_dependence_label = (
            "RAIN_STATE_DEPENDENCE_NOT_CONSISTENT_ACROSS_DAILY_AND_EVENT_LEVEL"
        )

    audit_summary = {
        "stage": "P2-1C-1R-A",
        "diagnosis_only": True,
        "repaired_transition_model_fitted": False,
        "clean_action_built": False,
        "online_perception_built": False,
        "reward_built": False,
        "rl_started": False,
        "inputs": {
            "calendar_days": int(len(ledger)),
            "valid_states": int(ledger["state_valid"].sum()),
            "manual_clean_days": int(
                ledger["modb_manual_cleaning_day"].sum()
            ),
            "dry_transitions": int(len(dry)),
            "rain_daily_transitions": int(len(rain)),
            "rain_event_rows": int(len(rain_events)),
            "resolved_unconfounded_rain_events": int(len(rr)),
        },
        "daily_rain_state_effect": daily_rain_state_effect,
        "event_rain_state_effect": event_rain_state_effect,
        "cycle_horizon_audit": cycle_horizon_audit,
        "recursive_observed_horizon_validation": recursive_primary,
        "diagnosis_labels": {
            "observed_horizon_recursive_skill": observed_horizon_label,
            "rain_state_dependence": rain_state_dependence_label,
        },
        "technical_gates": technical_gates,
        "interpretation_rules": {
            "case_A": (
                "If D0/R0 are weak even on leave-one-cycle-out historical horizons, "
                "repair natural dynamics before any online perception work."
            ),
            "case_B": (
                "If D0/R0 are reasonably calibrated on historical cleaning-cycle "
                "horizons but fail only at 365-day WAIT, the dominant problem is "
                "counterfactual horizon/support extrapolation. Do not distort rain "
                "physics merely to keep states under the historical maximum."
            ),
            "rain_rule": (
                "A negative daily delta-vs-source-state slope consistent with the "
                "event-level S_pre/removal relationship justifies evaluating a "
                "continuous state-dependent rain candidate in P2-1C-1R-B, but does "
                "not by itself prove that candidate should be frozen."
            ),
            "next_step": (
                "Review this diagnosis first. Only then define the small repaired "
                "candidate set and require both one-step OOT and recursive validation."
            ),
        },
    }

    print("[10/10] Write diagnostic outputs")
    out.mkdir(parents=True, exist_ok=True)

    dep_df.to_csv(
        out / "transition_state_dependence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    band_df.to_csv(
        out / "transition_state_band_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rr.to_csv(
        out / "rain_event_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rain_event_dep.to_csv(
        out / "rain_event_state_dependence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cycle_df.to_csv(
        out / "manual_cleaning_cycle_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cycle_val_df.to_csv(
        out / "cycle_recursive_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cycle_day_df.to_csv(
        out / "cycle_recursive_day_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, ensure_ascii=False, indent=2)

    print(out / "transition_state_dependence.csv")
    print(out / "transition_state_band_summary.csv")
    print(out / "rain_event_state_dependence.csv")
    print(out / "manual_cleaning_cycle_summary.csv")
    print(out / "cycle_recursive_validation_summary.csv")
    print(out / "cycle_recursive_day_validation.csv")
    print(out / "audit_summary.json")
    print(
        "IMPORTANT: diagnosis only. Do NOT fit a repaired kernel, add CLEAN, "
        "online perception, reward, Gym, or PPO before review."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
