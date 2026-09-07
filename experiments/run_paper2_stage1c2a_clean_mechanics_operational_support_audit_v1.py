#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1C-2A
CLEAN Mechanics + Operational Support Audit v1

SCIENTIFIC PURPOSE
------------------
Natural Dynamics v2 is now scope-qualified and frozen as:
    Dry  : D0_GLOBAL empirical delta kernel
    Rain : R3_OLS_RESIDUAL
           delta = alpha + beta * L_source + empirical residual

The 365-day WAIT-only experiment is intentionally retained as an extreme
counterfactual stress case. It frequently leaves the Paper1-derived perception
deployment domain.

Before building online perception, reward, Gym, or PPO, this stage asks a more
relevant operational question:

    Can a simple, non-optimised CLEAN intervention keep the long-horizon
    counterfactual system inside the perception-supported operating domain?

This is NOT policy optimisation and NOT an RL baseline contest.
True-state threshold rules are used only as physical/support probes.

TIMING CONTRACT
---------------
For each day t:
    pre_action_state L_t
        -> threshold CLEAN/WAIT decision
        -> post_action_state
        -> natural dry/rain transition
        -> L_{t+1}

CLEAN is immediate and removes a fraction eta of current soiling:
    L_post = (1 - eta) * L_pre

Natural rain response is evaluated at the POST-ACTION state, because the
physical rain transition occurs after the intervention in the frozen daily
contract.

The agent/probe does NOT observe future rain. Threshold action uses only L_t.
Historical ModB cleaning dates never trigger counterfactual CLEAN.

PRIMARY SETTINGS
----------------
True-state support-probe thresholds:
    0.05, 0.08, 0.10, 0.12, 0.15

Cleaning-efficiency matrix:
    eta = 0.90, 0.95, 1.00
Primary eta:
    0.95

Reference policy:
    WAIT_ONLY

Each policy is simulated with COMMON RANDOM NUMBERS (CRN):
for a given year/initialisation scenario, the exact same empirical innovation
index is used at each day and trajectory for every threshold/eta policy.

PRIMARY SUPPORT GATES
---------------------
For a threshold probe at eta=0.95 to be called SUPPORT-SAFE, BOTH YEAR1 and
YEAR2 historical-first scenarios must satisfy:

1) pre-action state-day fraction above perception margin <= 1%
   perception margin = historical WAPP max + 0.01

2) rain-model call fraction above R3 direct daily source-state maximum <= 5%

3) no upper physical projection to L=1

These are support-integrity diagnostics, not claims of physical law.

If at least one eta=0.95 threshold is support-safe:
    operational_support_controllability = PASS

The largest support-safe threshold is reported only as a
"support-probe upper reference". It is NOT frozen as the final cleaning policy.

ADDITIONAL AUDITS
-----------------
- exact CLEAN formula identity
- threshold-action identity
- clean counts and inter-clean gaps
- consecutive-clean frequency
- cleaning/rain coincidence (diagnostic; no future-rain avoidance)
- pre-action vs post-action support occupancy
- lower/upper physical projection rates
- sensitivity to eta = 0.90/0.95/1.00
- YEAR1/YEAR2 consistency
- historical-first vs clean-start-zero sensitivity
- CRN reproducibility
- WAIT_ONLY regression against the already audited R3 stress values

NO REWARD / NO ONLINE PERCEPTION / NO GYM / NO PPO.
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


EXPECTED_CALENDAR_DAYS = 730
EXPECTED_VALID_STATES = 729
EXPECTED_DRY = 494
EXPECTED_RAIN = 201
EXPECTED_CLEAN_DAYS = 26
EXPECTED_DAYS_PER_YEAR = 365

YEAR_LABELS = ("YEAR1", "YEAR2")

THRESHOLDS = (0.05, 0.08, 0.10, 0.12, 0.15)
ETAS = (0.90, 0.95, 1.00)
PRIMARY_ETA = 0.95

DEFAULT_N_TRAJECTORIES = 1000
DEFAULT_PARENT_ENV_SEED = 420001

PERCEPTION_LOCAL_MARGIN = 0.01
PRE_ACTION_MARGIN_GATE = 0.01
RAIN_DIRECT_SUPPORT_GATE = 0.05

# Frozen full-data R3 parameters from P2-1C-1R-B/B2.
EXPECTED_R3_INTERCEPT = 0.00090811689781017
EXPECTED_R3_SLOPE = -0.48383380267297343
PARAM_TOL = 1e-12

# WAIT_ONLY regression values from P2-1C-1R-B2, historical-first only.
WAIT_REGRESSION = {
    "YEAR1": {
        "seed": 1020001,
        "margin_fraction": 0.3172876712328767,
        "hist_fraction": 0.33512054794520546,
        "terminal_median": 0.004780516323453693,
    },
    "YEAR2": {
        "seed": 1030001,
        "margin_fraction": 0.2487835616438356,
        "hist_fraction": 0.26043561643835617,
        "terminal_median": 0.0018227846504571142,
    },
}


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


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


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

    return transitions.merge(
        src, on="source_date", how="left", validate="many_to_one"
    ).merge(
        dst, on="dest_date", how="left", validate="many_to_one"
    )


def fit_r3(rain: pd.DataFrame) -> dict:
    g = rain[
        np.isfinite(rain["source_L"])
        & np.isfinite(rain["delta_L_power_proxy"])
    ].copy()

    x = g["source_L"].to_numpy(dtype=float)
    y = g["delta_L_power_proxy"].to_numpy(dtype=float)

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


def scenario_seed(
    parent_env_seed: int,
    year_label: str,
    init_mode: str,
) -> int:
    # Exact R3 stress-seed convention inherited from P2-1C-1R-B/B2.
    stress_root = int(parent_env_seed + 500000)
    r3_candidate_offset = 100000
    yidx = 0 if year_label == "YEAR1" else 1
    iidx = 0 if init_mode == "HISTORICAL_FIRST_DAY" else 1

    return int(
        stress_root
        + r3_candidate_offset
        + 10000 * yidx
        + 1000 * iidx
    )


def build_crn_indices(
    year_df: pd.DataFrame,
    dry_pool_n: int,
    rain_residual_n: int,
    n_trajectories: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """
    Reproduce the exact RNG-call structure used in P2-1C-1R-B:
    one rng.integers call per transition day, from the pool actually used that day.
    The resulting innovation index arrays are then shared across ALL policies.
    """
    rng = np.random.default_rng(seed)
    rain = year_df["rain_day"].to_numpy(dtype=bool)
    rain_affected = rain[:-1] | rain[1:]

    indices = []
    for is_rain in rain_affected:
        pool_n = rain_residual_n if is_rain else dry_pool_n
        idx = rng.integers(0, pool_n, size=n_trajectories)
        indices.append(idx.astype(np.int64, copy=False))

    stacked = np.vstack(indices)

    # Exact reproducibility audit.
    rng2 = np.random.default_rng(seed)
    indices2 = []
    for is_rain in rain_affected:
        pool_n = rain_residual_n if is_rain else dry_pool_n
        indices2.append(
            rng2.integers(0, pool_n, size=n_trajectories).astype(
                np.int64, copy=False
            )
        )
    stacked2 = np.vstack(indices2)

    audit = {
        "seed": int(seed),
        "transition_days": int(len(rain_affected)),
        "rain_affected_transition_days": int(rain_affected.sum()),
        "dry_transition_days": int((~rain_affected).sum()),
        "index_hash": sha256_array(stacked),
        "same_seed_exact_reproduction": bool(
            np.array_equal(stacked, stacked2)
        ),
    }

    return indices, rain_affected, audit


def policy_id(threshold: float | None, eta: float | None) -> str:
    if threshold is None:
        return "WAIT_ONLY"
    return f"THR_{threshold:.2f}_ETA_{eta:.2f}"


def interclean_gaps(actions: np.ndarray) -> np.ndarray:
    """
    actions: bool vector length 365 for one trajectory.
    Return day-index gaps between consecutive CLEAN actions.
    """
    idx = np.flatnonzero(actions)
    if len(idx) < 2:
        return np.array([], dtype=float)
    return np.diff(idx).astype(float)


def simulate_policy(
    year_df: pd.DataFrame,
    initial_state: float,
    threshold: float | None,
    eta: float | None,
    dry_pool: np.ndarray,
    r3: dict,
    innovation_indices: list[np.ndarray],
    rain_affected: np.ndarray,
    hist_max: float,
    perception_margin_max: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    n_trajectories = len(innovation_indices[0])
    n_days = len(year_df)

    states_pre = np.empty((n_trajectories, n_days), dtype=float)
    states_post = np.empty((n_trajectories, n_days), dtype=float)
    actions = np.zeros((n_trajectories, n_days), dtype=bool)

    lower_proj_count = 0
    upper_proj_count = 0
    natural_transition_count = 0

    clean_formula_max_abs_error = 0.0
    threshold_action_mismatch_count = 0

    rain_call_n = 0
    rain_call_above_direct_n = 0
    rain_call_above_hist_n = 0
    rain_call_above_margin_n = 0
    rain_call_lower_proj_n = 0
    rain_call_upper_proj_n = 0

    states_pre[:, 0] = float(initial_state)

    for t in range(n_days):
        pre = states_pre[:, t]

        if threshold is None:
            clean = np.zeros(n_trajectories, dtype=bool)
            post = pre.copy()
        else:
            clean = pre >= float(threshold)
            actions[:, t] = clean

            expected_clean = pre >= float(threshold)
            threshold_action_mismatch_count += int(
                np.sum(clean != expected_clean)
            )

            post = pre.copy()
            post[clean] = (1.0 - float(eta)) * pre[clean]

            if np.any(clean):
                formula_err = np.max(
                    np.abs(
                        post[clean]
                        - (1.0 - float(eta)) * pre[clean]
                    )
                )
                clean_formula_max_abs_error = max(
                    clean_formula_max_abs_error,
                    float(formula_err),
                )

        states_post[:, t] = post

        # Terminal day still has an action/post-action state, but no t->t+1 transition.
        if t == n_days - 1:
            break

        idx = innovation_indices[t]

        if rain_affected[t]:
            residual = np.asarray(r3["residuals"], dtype=float)[idx]
            mu_delta = (
                float(r3["intercept"])
                + float(r3["slope"]) * post
            )
            delta = mu_delta + residual

            rain_call_n += n_trajectories
            rain_call_above_direct_n += int(
                np.sum(post > float(r3["source_max"]))
            )
            rain_call_above_hist_n += int(
                np.sum(post > hist_max)
            )
            rain_call_above_margin_n += int(
                np.sum(post > perception_margin_max)
            )
        else:
            delta = dry_pool[idx]

        raw_next = post + delta

        lower = raw_next < 0.0
        upper = raw_next > 1.0

        lower_proj_count += int(lower.sum())
        upper_proj_count += int(upper.sum())
        natural_transition_count += n_trajectories

        if rain_affected[t]:
            rain_call_lower_proj_n += int(lower.sum())
            rain_call_upper_proj_n += int(upper.sum())

        states_pre[:, t + 1] = np.clip(raw_next, 0.0, 1.0)

    pre_above_hist = states_pre > hist_max
    pre_above_margin = states_pre > perception_margin_max
    post_above_hist = states_post > hist_max
    post_above_margin = states_post > perception_margin_max

    clean_counts = actions.sum(axis=1).astype(int)
    traj_max_pre = states_pre.max(axis=1)
    traj_max_post = states_post.max(axis=1)

    all_gaps = []
    consecutive_pairs = 0
    total_clean_pairs = 0

    for i in range(n_trajectories):
        gaps = interclean_gaps(actions[i])
        if len(gaps):
            all_gaps.extend(gaps.tolist())
            consecutive_pairs += int(np.sum(gaps == 1))
            total_clean_pairs += int(len(gaps))

    all_gaps = np.asarray(all_gaps, dtype=float)

    # Count action/rain coincidence for non-terminal days.
    rain_action_mask = np.zeros(n_days, dtype=bool)
    rain_action_mask[:-1] = rain_affected
    clean_on_rain_affected = int(
        np.sum(actions & rain_action_mask[None, :])
    )
    total_clean_actions = int(actions.sum())

    summary = {
        "policy_id": policy_id(threshold, eta),
        "threshold": (
            float(threshold) if threshold is not None else np.nan
        ),
        "eta": float(eta) if eta is not None else np.nan,
        "n_trajectories": int(n_trajectories),
        "calendar_days": int(n_days),
        "in_horizon_transitions": int(n_days - 1),

        "pre_action_state_distribution": qstats(states_pre.ravel()),
        "post_action_state_distribution": qstats(states_post.ravel()),

        "pre_action_state_day_fraction_above_historical_max": float(
            pre_above_hist.mean()
        ),
        "pre_action_state_day_fraction_above_perception_margin": float(
            pre_above_margin.mean()
        ),
        "post_action_state_day_fraction_above_historical_max": float(
            post_above_hist.mean()
        ),
        "post_action_state_day_fraction_above_perception_margin": float(
            post_above_margin.mean()
        ),

        "trajectory_fraction_ever_pre_action_above_perception_margin": float(
            pre_above_margin.any(axis=1).mean()
        ),
        "trajectory_max_pre_distribution": qstats(traj_max_pre),
        "trajectory_max_post_distribution": qstats(traj_max_post),

        "clean_count_distribution": qstats(clean_counts),
        "clean_count_mean": float(np.mean(clean_counts)),
        "clean_count_median": float(np.median(clean_counts)),
        "clean_count_min": int(np.min(clean_counts)),
        "clean_count_max": int(np.max(clean_counts)),
        "total_clean_actions": total_clean_actions,

        "interclean_gap_distribution": qstats(all_gaps),
        "consecutive_clean_pair_fraction": float(
            consecutive_pairs / total_clean_pairs
        ) if total_clean_pairs > 0 else 0.0,
        "clean_action_on_rain_affected_fraction": float(
            clean_on_rain_affected / total_clean_actions
        ) if total_clean_actions > 0 else 0.0,

        "rain_model_calls": int(rain_call_n),
        "rain_call_above_direct_daily_support_fraction": float(
            rain_call_above_direct_n / rain_call_n
        ) if rain_call_n > 0 else np.nan,
        "rain_call_above_historical_max_fraction": float(
            rain_call_above_hist_n / rain_call_n
        ) if rain_call_n > 0 else np.nan,
        "rain_call_above_perception_margin_fraction": float(
            rain_call_above_margin_n / rain_call_n
        ) if rain_call_n > 0 else np.nan,
        "rain_call_lower_projection_fraction": float(
            rain_call_lower_proj_n / rain_call_n
        ) if rain_call_n > 0 else np.nan,
        "rain_call_upper_projection_fraction": float(
            rain_call_upper_proj_n / rain_call_n
        ) if rain_call_n > 0 else np.nan,

        "all_transition_lower_projection_fraction": float(
            lower_proj_count / natural_transition_count
        ),
        "all_transition_upper_projection_fraction": float(
            upper_proj_count / natural_transition_count
        ),

        "threshold_action_mismatch_count": int(
            threshold_action_mismatch_count
        ),
        "clean_formula_max_abs_error": float(
            clean_formula_max_abs_error
        ),

        "terminal_pre_state_distribution": qstats(
            states_pre[:, -1]
        ),
        "terminal_post_state_distribution": qstats(
            states_post[:, -1]
        ),
    }

    traj_rows = []
    for i in range(n_trajectories):
        gaps = interclean_gaps(actions[i])
        traj_rows.append({
            "policy_id": summary["policy_id"],
            "trajectory_id": int(i),
            "clean_count": int(clean_counts[i]),
            "max_pre_action_state": float(traj_max_pre[i]),
            "max_post_action_state": float(traj_max_post[i]),
            "fraction_pre_action_above_hist_max": float(
                pre_above_hist[i].mean()
            ),
            "fraction_pre_action_above_perception_margin": float(
                pre_above_margin[i].mean()
            ),
            "ever_pre_action_above_perception_margin": bool(
                pre_above_margin[i].any()
            ),
            "median_interclean_gap": (
                float(np.median(gaps)) if len(gaps) else np.nan
            ),
            "min_interclean_gap": (
                float(np.min(gaps)) if len(gaps) else np.nan
            ),
            "terminal_pre_state": float(states_pre[i, -1]),
            "terminal_post_state": float(states_post[i, -1]),
        })
    traj_df = pd.DataFrame(traj_rows)

    day_rows = []
    for d in range(n_days):
        day_rows.append({
            "policy_id": summary["policy_id"],
            "day_index": int(d),
            "date": year_df.loc[d, "date"],
            "pre_mean": float(np.mean(states_pre[:, d])),
            "pre_q50": float(np.quantile(states_pre[:, d], 0.50)),
            "pre_q95": float(np.quantile(states_pre[:, d], 0.95)),
            "pre_q99": float(np.quantile(states_pre[:, d], 0.99)),
            "pre_max": float(np.max(states_pre[:, d])),
            "post_q50": float(np.quantile(states_post[:, d], 0.50)),
            "post_q95": float(np.quantile(states_post[:, d], 0.95)),
            "clean_fraction": float(actions[:, d].mean()),
            "pre_fraction_above_perception_margin": float(
                pre_above_margin[:, d].mean()
            ),
        })
    day_df = pd.DataFrame(day_rows)

    return summary, traj_df, day_df


def wait_regression_gate(
    row: dict,
    year_label: str,
    seed: int,
) -> dict:
    ref = WAIT_REGRESSION[year_label]

    checks = {
        "seed_exact": bool(seed == ref["seed"]),
        "margin_fraction_exact": bool(
            abs(
                row[
                    "pre_action_state_day_fraction_above_perception_margin"
                ]
                - ref["margin_fraction"]
            ) <= 1e-15
        ),
        "hist_fraction_exact": bool(
            abs(
                row[
                    "pre_action_state_day_fraction_above_historical_max"
                ]
                - ref["hist_fraction"]
            ) <= 1e-15
        ),
        "terminal_median_exact": bool(
            abs(
                row["terminal_pre_state_distribution"]["q50"]
                - ref["terminal_median"]
            ) <= 1e-15
        ),
    }
    checks["all_wait_regression_pass"] = bool(all(checks.values()))
    return checks


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-1C-2A CLEAN mechanics + operational support audit."
    )
    p.add_argument("--master-ledger", required=True, type=Path)
    p.add_argument("--transition-audit", required=True, type=Path)
    p.add_argument(
        "--n-trajectories",
        type=int,
        default=DEFAULT_N_TRAJECTORIES,
    )
    p.add_argument(
        "--parent-env-seed",
        type=int,
        default=DEFAULT_PARENT_ENV_SEED,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1c2a_clean_mechanics_operational_support_audit_v1"
        ),
    )
    args = p.parse_args()

    if args.n_trajectories < 200:
        raise ValueError("--n-trajectories must be >=200.")

    lp = args.master_ledger.expanduser().resolve()
    tp = args.transition_audit.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()

    for path in [lp, tp]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/11] Load frozen timeline and natural-dynamics assets")
    ledger = load_ledger(lp)
    transitions = load_transitions(tp)

    if len(ledger) != EXPECTED_CALENDAR_DAYS:
        raise RuntimeError("Calendar day count mismatch.")
    if int(ledger["state_valid"].sum()) != EXPECTED_VALID_STATES:
        raise RuntimeError("Valid-state count mismatch.")
    if int(ledger["modb_manual_cleaning_day"].sum()) != EXPECTED_CLEAN_DAYS:
        raise RuntimeError("Manual-clean day count mismatch.")

    print("[2/11] Reconstruct frozen D0 + R3 Natural Dynamics v2")
    x = attach_context(transitions, ledger)
    dry = x[x["transition_class"].eq("DRY_NATURAL")].copy()
    rain = x[x["transition_class"].eq("RAIN_AFFECTED")].copy()

    if len(dry) != EXPECTED_DRY:
        raise RuntimeError(f"Dry count mismatch: {len(dry)}")
    if len(rain) != EXPECTED_RAIN:
        raise RuntimeError(f"Rain count mismatch: {len(rain)}")

    dry_pool = dry["delta_L_power_proxy"].to_numpy(dtype=float)
    r3 = fit_r3(rain)

    param_gate = bool(
        abs(r3["intercept"] - EXPECTED_R3_INTERCEPT) <= PARAM_TOL
        and abs(r3["slope"] - EXPECTED_R3_SLOPE) <= PARAM_TOL
    )

    if not param_gate:
        raise RuntimeError(
            "Frozen R3 parameters drifted from P2-1C-1R-B/B2."
        )

    valid_hist = ledger.loc[
        ledger["state_valid"]
        & np.isfinite(ledger["L_power_proxy"]),
        "L_power_proxy",
    ].to_numpy(dtype=float)

    hist_max = float(np.max(valid_hist))
    perception_margin_max = float(
        hist_max + PERCEPTION_LOCAL_MARGIN
    )

    print("[3/11] Build historical cleaning-frequency reference")
    clean_dates = ledger.loc[
        ledger["modb_manual_cleaning_day"], "date"
    ].sort_values().tolist()

    clean_gaps = np.diff(
        np.array(clean_dates, dtype="datetime64[D]")
    ).astype("timedelta64[D]").astype(int)

    historical_clean_reference = {
        "manual_clean_days_2yr": int(len(clean_dates)),
        "manual_clean_days_per_year_equivalent": float(
            len(clean_dates) / 2.0
        ),
        "clean_to_clean_gap_distribution_days": qstats(clean_gaps),
        "note": (
            "Historical manual cleaning frequency is descriptive only; "
            "it is not used as an optimisation target or hard policy gate."
        ),
    }

    print("[4/11] Pre-generate common-random-number innovation streams")
    scenario_assets = {}
    crn_rows = []

    for year_label in YEAR_LABELS:
        ydf = get_year_calendar(ledger, year_label)
        hist_first = float(ydf["L_power_proxy"].iloc[0])

        for init_mode, init_state in [
            ("HISTORICAL_FIRST_DAY", hist_first),
            ("CLEAN_START_ZERO", 0.0),
        ]:
            seed = scenario_seed(
                args.parent_env_seed,
                year_label,
                init_mode,
            )
            indices, rain_affected, crn_audit = build_crn_indices(
                year_df=ydf,
                dry_pool_n=len(dry_pool),
                rain_residual_n=len(r3["residuals"]),
                n_trajectories=args.n_trajectories,
                seed=seed,
            )

            key = (year_label, init_mode)
            scenario_assets[key] = {
                "year_df": ydf,
                "initial_state": init_state,
                "seed": seed,
                "indices": indices,
                "rain_affected": rain_affected,
            }

            crn_rows.append({
                "year_label": year_label,
                "init_mode": init_mode,
                **crn_audit,
            })

    crn_df = pd.DataFrame(crn_rows)

    print("[5/11] Simulate WAIT_ONLY regression reference")
    scenario_summary_rows = []
    traj_parts = []
    day_parts = []
    wait_regression = {}

    for (year_label, init_mode), asset in scenario_assets.items():
        summary, traj_df, day_df = simulate_policy(
            year_df=asset["year_df"],
            initial_state=asset["initial_state"],
            threshold=None,
            eta=None,
            dry_pool=dry_pool,
            r3=r3,
            innovation_indices=asset["indices"],
            rain_affected=asset["rain_affected"],
            hist_max=hist_max,
            perception_margin_max=perception_margin_max,
        )

        summary.update({
            "year_label": year_label,
            "init_mode": init_mode,
            "scenario_env_seed": int(asset["seed"]),
        })
        scenario_summary_rows.append(summary)

        traj_df["year_label"] = year_label
        traj_df["init_mode"] = init_mode
        traj_df["scenario_env_seed"] = int(asset["seed"])
        traj_parts.append(traj_df)

        day_df["year_label"] = year_label
        day_df["init_mode"] = init_mode
        day_df["scenario_env_seed"] = int(asset["seed"])
        day_parts.append(day_df)

        if init_mode == "HISTORICAL_FIRST_DAY":
            wait_regression[year_label] = wait_regression_gate(
                summary,
                year_label,
                asset["seed"],
            )

    if not all(
        x["all_wait_regression_pass"]
        for x in wait_regression.values()
    ):
        raise RuntimeError(
            "WAIT_ONLY regression failed; abort before CLEAN audit."
        )

    print("[6/11] Simulate full threshold x cleaning-efficiency probe matrix")
    for eta in ETAS:
        for threshold in THRESHOLDS:
            for (year_label, init_mode), asset in scenario_assets.items():
                summary, traj_df, day_df = simulate_policy(
                    year_df=asset["year_df"],
                    initial_state=asset["initial_state"],
                    threshold=threshold,
                    eta=eta,
                    dry_pool=dry_pool,
                    r3=r3,
                    innovation_indices=asset["indices"],
                    rain_affected=asset["rain_affected"],
                    hist_max=hist_max,
                    perception_margin_max=perception_margin_max,
                )

                summary.update({
                    "year_label": year_label,
                    "init_mode": init_mode,
                    "scenario_env_seed": int(asset["seed"]),
                })
                scenario_summary_rows.append(summary)

                traj_df["year_label"] = year_label
                traj_df["init_mode"] = init_mode
                traj_df["scenario_env_seed"] = int(asset["seed"])
                traj_parts.append(traj_df)

                day_df["year_label"] = year_label
                day_df["init_mode"] = init_mode
                day_df["scenario_env_seed"] = int(asset["seed"])
                day_parts.append(day_df)

    scenario_df = pd.DataFrame(scenario_summary_rows)
    trajectory_df = pd.concat(traj_parts, ignore_index=True)
    daywise_df = pd.concat(day_parts, ignore_index=True)

    print("[7/11] Audit CLEAN mechanics invariants")
    clean_rows = scenario_df[
        scenario_df["policy_id"].ne("WAIT_ONLY")
    ].copy()

    mechanics_gates = {
        "threshold_action_identity_pass": bool(
            (clean_rows["threshold_action_mismatch_count"] == 0).all()
        ),
        "clean_formula_identity_pass": bool(
            (clean_rows["clean_formula_max_abs_error"] <= 1e-15).all()
        ),
        "all_states_finite_pass": bool(
            np.isfinite(
                trajectory_df[
                    [
                        "max_pre_action_state",
                        "max_post_action_state",
                        "terminal_pre_state",
                        "terminal_post_state",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "no_upper_projection_pass": bool(
            (
                scenario_df[
                    "all_transition_upper_projection_fraction"
                ] == 0.0
            ).all()
        ),
        "crn_same_seed_reproducibility_pass": bool(
            crn_df["same_seed_exact_reproduction"].all()
        ),
    }
    mechanics_gates["all_mechanics_gates_pass"] = bool(
        all(mechanics_gates.values())
    )

    print("[8/11] Evaluate operational-support safety at primary eta=0.95")
    primary = scenario_df[
        np.isclose(
            pd.to_numeric(scenario_df["eta"], errors="coerce"),
            PRIMARY_ETA,
            equal_nan=False,
        )
        & scenario_df["init_mode"].eq("HISTORICAL_FIRST_DAY")
    ].copy()

    support_rows = []
    for threshold in THRESHOLDS:
        g = primary[
            np.isclose(
                pd.to_numeric(primary["threshold"], errors="coerce"),
                threshold,
            )
        ].copy()

        if set(g["year_label"]) != set(YEAR_LABELS):
            raise RuntimeError(
                f"Missing primary scenarios for threshold {threshold}."
            )

        per_year = {}
        year_passes = []

        for _, r in g.iterrows():
            y = str(r["year_label"])
            margin_frac = float(
                r[
                    "pre_action_state_day_fraction_above_perception_margin"
                ]
            )
            rain_out_frac = float(
                r[
                    "rain_call_above_direct_daily_support_fraction"
                ]
            )
            upper_proj = float(
                r["all_transition_upper_projection_fraction"]
            )

            y_pass = bool(
                margin_frac <= PRE_ACTION_MARGIN_GATE
                and rain_out_frac <= RAIN_DIRECT_SUPPORT_GATE
                and upper_proj == 0.0
            )
            year_passes.append(y_pass)

            per_year[y] = {
                "pre_action_margin_exceed_fraction": margin_frac,
                "rain_call_above_direct_support_fraction": rain_out_frac,
                "upper_projection_fraction": upper_proj,
                "clean_count_mean": float(r["clean_count_mean"]),
                "clean_count_median": float(r["clean_count_median"]),
                "pre_action_state_q99": float(
                    r["pre_action_state_distribution"]["q99"]
                ),
                "pre_action_state_max": float(
                    r["pre_action_state_distribution"]["max"]
                ),
                "support_safe": y_pass,
            }

        support_rows.append({
            "threshold": float(threshold),
            "eta": PRIMARY_ETA,
            "YEAR1_margin_exceed_fraction": per_year["YEAR1"][
                "pre_action_margin_exceed_fraction"
            ],
            "YEAR2_margin_exceed_fraction": per_year["YEAR2"][
                "pre_action_margin_exceed_fraction"
            ],
            "YEAR1_rain_call_above_direct_support_fraction": per_year[
                "YEAR1"
            ]["rain_call_above_direct_support_fraction"],
            "YEAR2_rain_call_above_direct_support_fraction": per_year[
                "YEAR2"
            ]["rain_call_above_direct_support_fraction"],
            "YEAR1_clean_count_mean": per_year["YEAR1"][
                "clean_count_mean"
            ],
            "YEAR2_clean_count_mean": per_year["YEAR2"][
                "clean_count_mean"
            ],
            "YEAR1_pre_action_q99": per_year["YEAR1"][
                "pre_action_state_q99"
            ],
            "YEAR2_pre_action_q99": per_year["YEAR2"][
                "pre_action_state_q99"
            ],
            "support_safe_both_years": bool(all(year_passes)),
        })

    support_df = pd.DataFrame(support_rows)

    safe_thresholds = support_df.loc[
        support_df["support_safe_both_years"],
        "threshold",
    ].to_numpy(dtype=float)

    operational_support_controllability = bool(
        len(safe_thresholds) > 0
    )
    support_probe_upper_reference = (
        float(np.max(safe_thresholds))
        if len(safe_thresholds)
        else None
    )

    print("[9/11] Build eta-sensitivity and behavioural diagnostics")
    eta_rows = []
    clean_primary_all_eta = scenario_df[
        scenario_df["policy_id"].ne("WAIT_ONLY")
        & scenario_df["init_mode"].eq("HISTORICAL_FIRST_DAY")
    ].copy()

    for eta in ETAS:
        for threshold in THRESHOLDS:
            g = clean_primary_all_eta[
                np.isclose(
                    pd.to_numeric(
                        clean_primary_all_eta["eta"],
                        errors="coerce",
                    ),
                    eta,
                )
                & np.isclose(
                    pd.to_numeric(
                        clean_primary_all_eta["threshold"],
                        errors="coerce",
                    ),
                    threshold,
                )
            ].copy()

            eta_rows.append({
                "eta": float(eta),
                "threshold": float(threshold),
                "mean_clean_count_across_years": float(
                    g["clean_count_mean"].mean()
                ),
                "max_margin_exceed_fraction_across_years": float(
                    g[
                        "pre_action_state_day_fraction_above_perception_margin"
                    ].max()
                ),
                "max_rain_call_above_direct_support_fraction_across_years": float(
                    g[
                        "rain_call_above_direct_daily_support_fraction"
                    ].max()
                ),
                "max_consecutive_clean_pair_fraction_across_years": float(
                    g["consecutive_clean_pair_fraction"].max()
                ),
                "mean_clean_on_rain_affected_fraction_across_years": float(
                    g["clean_action_on_rain_affected_fraction"].mean()
                ),
                "mean_lower_projection_fraction_across_years": float(
                    g["all_transition_lower_projection_fraction"].mean()
                ),
            })

    eta_df = pd.DataFrame(eta_rows)

    print("[10/11] Build audit summary and PASS/FAIL decision")
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
        "frozen_R3_parameter_regression_pass": param_gate,
        "WAIT_ONLY_regression_YEAR1_pass": bool(
            wait_regression["YEAR1"]["all_wait_regression_pass"]
        ),
        "WAIT_ONLY_regression_YEAR2_pass": bool(
            wait_regression["YEAR2"]["all_wait_regression_pass"]
        ),
    }
    technical_gates["all_technical_gates_pass"] = bool(
        all(technical_gates.values())
    )

    stage_pass = bool(
        technical_gates["all_technical_gates_pass"]
        and mechanics_gates["all_mechanics_gates_pass"]
        and operational_support_controllability
    )

    audit_summary = {
        "stage": "P2-1C-2A",
        "audit_only": True,
        "clean_mechanics_built": True,
        "reward_built": False,
        "online_perception_built": False,
        "gym_built": False,
        "rl_started": False,
        "frozen_natural_dynamics": {
            "dry": "D0_GLOBAL_EMPIRICAL",
            "rain": "R3_OLS_RESIDUAL",
            "r3_intercept": float(r3["intercept"]),
            "r3_slope": float(r3["slope"]),
            "r3_direct_daily_source_max": float(r3["source_max"]),
            "physical_projection": "[0,1]",
        },
        "clean_contract": {
            "action_timing": (
                "observe pre-action hidden state for support probe -> "
                "CLEAN/WAIT -> post-action state -> natural transition"
            ),
            "clean_formula": "L_post=(1-eta)*L_pre",
            "thresholds": list(THRESHOLDS),
            "etas": list(ETAS),
            "primary_eta": PRIMARY_ETA,
            "future_rain_used_by_action": False,
            "historical_manual_clean_dates_trigger_action": False,
            "terminal_day_action_exists": True,
            "terminal_day_natural_transition_exists": False,
        },
        "support_reference": {
            "historical_valid_state_max": hist_max,
            "perception_margin_max": perception_margin_max,
            "pre_action_margin_gate": PRE_ACTION_MARGIN_GATE,
            "r3_direct_daily_source_max": float(r3["source_max"]),
            "rain_direct_support_gate": RAIN_DIRECT_SUPPORT_GATE,
            "important_note": (
                "Pre-action state is the relevant observation-domain state "
                "because perception occurs before CLEAN."
            ),
        },
        "historical_cleaning_reference": historical_clean_reference,
        "common_random_numbers": crn_df.to_dict(orient="records"),
        "wait_regression": wait_regression,
        "mechanics_gates": mechanics_gates,
        "technical_gates": technical_gates,
        "primary_eta_support_table": support_df.to_dict(orient="records"),
        "operational_support_decision": {
            "operational_support_controllability_pass": (
                operational_support_controllability
            ),
            "support_safe_thresholds_eta_0p95": (
                [float(x) for x in safe_thresholds]
            ),
            "support_probe_upper_reference": (
                support_probe_upper_reference
            ),
            "support_probe_upper_reference_is_final_policy": False,
            "interpretation": (
                "PASS means at least one simple true-state CLEAN probe can keep "
                "the 365-day operating trajectory predominantly inside both the "
                "perception margin and the R3 direct rain-support diagnostic. "
                "This establishes operational controllability, not optimality."
                if operational_support_controllability
                else
                "FAIL means even simple CLEAN probes cannot keep the long-horizon "
                "operating trajectory within the declared support diagnostics. "
                "Do not proceed to online perception/PPO."
            ),
        },
        "eta_sensitivity": eta_df.to_dict(orient="records"),
        "stage_pass": stage_pass,
        "next_step_if_pass": (
            "Freeze CLEAN mechanics at eta=0.95 for the primary environment "
            "(retain eta=0.90/1.00 as sensitivity), keep threshold probes as "
            "diagnostic only, then proceed to P2-1C-2B online counterfactual "
            "perception with explicit policy-state support occupancy auditing."
        ),
        "next_step_if_fail": (
            "Do not tune thresholds post hoc to force a pass. Inspect whether "
            "failure is driven by perception-domain occupancy, R3 rain support, "
            "or CLEAN mechanics before any online perception or RL work."
        ),
    }

    print("[11/11] Write outputs")
    out.mkdir(parents=True, exist_ok=True)

    scenario_df.to_csv(
        out / "policy_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    support_df.to_csv(
        out / "primary_eta_support_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eta_df.to_csv(
        out / "eta_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trajectory_df.to_csv(
        out / "trajectory_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daywise_df.to_csv(
        out / "daywise_policy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    crn_df.to_csv(
        out / "common_random_number_audit.csv",
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

    print(out / "policy_scenario_summary.csv")
    print(out / "primary_eta_support_table.csv")
    print(out / "eta_sensitivity_summary.csv")
    print(out / "trajectory_summary.csv")
    print(out / "daywise_policy_summary.csv")
    print(out / "common_random_number_audit.csv")
    print(out / "audit_summary.json")
    print(
        "IMPORTANT: support/control audit only. "
        "Do NOT add reward, online perception, Gym, or PPO before review."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
