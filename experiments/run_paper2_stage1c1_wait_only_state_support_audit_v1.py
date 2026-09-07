#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1C-1
WAIT-only long-horizon state-support and boundary audit.

GOAL
----
Before adding CLEAN, reward, Gym, or PPO, test whether the frozen one-step
empirical natural-dynamics kernels remain scientifically usable when rolled
forward for a full 365-day counterfactual horizon.

Frozen kernels from P2-1B-2:
    D0_GLOBAL: empirical DRY_NATURAL delta_L distribution
    R0_GLOBAL: empirical RAIN_AFFECTED delta_L distribution

WHY THIS AUDIT IS NECESSARY
---------------------------
A model can perform well for one-step OOT prediction but still become unstable
when recursively simulated for hundreds of days. In particular, a positive
mean dry drift can accumulate and drive WAIT-only trajectories outside the
historical/perception-supported state domain.

This stage therefore checks:
1) exact calendar and kernel integrity;
2) long-horizon state drift under WAIT only;
3) how often simulated states exceed the historical WAPP maximum;
4) how often they exceed the +0.01 perception-neighbourhood margin;
5) lower/upper physical-boundary projection frequency;
6) YEAR1/YEAR2 calendar-template differences;
7) sensitivity to initial state;
8) deterministic reproducibility under a fixed environment RNG seed.

NO CLEAN / NO REWARD / NO PERCEPTION / NO PPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CALENDAR_DAYS = 730
EXPECTED_VALID_STATES = 729
EXPECTED_INVALID_DATES = {"2022-06-29"}

EXPECTED_DRY_KERNEL_N = 494
EXPECTED_RAIN_KERNEL_N = 201

YEAR_LABELS = ("YEAR1", "YEAR2")
EXPECTED_DAYS_PER_YEAR = 365

DEFAULT_N_TRAJECTORIES = 1000
DEFAULT_ENV_SEED = 310001

# Diagnostic support thresholds declared before seeing the result.
# These are not claims of physical law; they determine whether the current
# perception-supported domain is adequate for the planned online emulator.
HISTORICAL_SUPPORT_WARN = 0.01     # >1% state-days above historical max: concern
HISTORICAL_SUPPORT_FAIL = 0.05     # >5% state-days above historical max: mismatch
MARGIN_SUPPORT_FAIL = 0.01         # >1% above hist_max+0.01: local-support problem


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
        raise RuntimeError(f"{name}: unparseable boolean values {bad}")
    return y.astype(bool)


def qstats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
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
        raise RuntimeError(f"Master ledger missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    for col in [
        "state_valid",
        "rain_day",
        "modb_manual_cleaning_day",
        "scheduled_maintenance",
    ]:
        df[col] = parse_bool_series(df[col], col)

    df["L_power_proxy"] = pd.to_numeric(df["L_power_proxy"], errors="coerce")

    if df["date"].duplicated().any():
        raise RuntimeError("Master ledger contains duplicate dates.")
    return df.sort_values("date").reset_index(drop=True)


def load_transition_audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "transition_index",
        "source_date",
        "dest_date",
        "transition_class",
        "delta_L_power_proxy",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Transition audit missing columns: {sorted(missing)}")

    df["source_date"] = pd.to_datetime(
        df["source_date"], errors="raise"
    ).dt.normalize()
    df["dest_date"] = pd.to_datetime(
        df["dest_date"], errors="raise"
    ).dt.normalize()
    df["delta_L_power_proxy"] = pd.to_numeric(
        df["delta_L_power_proxy"], errors="coerce"
    )
    return df.sort_values("transition_index").reset_index(drop=True)


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def calendar_integrity(ledger: pd.DataFrame) -> dict:
    expected_dates = pd.date_range(
        ledger["date"].min(), ledger["date"].max(), freq="D"
    )
    invalid_dates = set(
        ledger.loc[~ledger["state_valid"], "date"].dt.strftime("%Y-%m-%d")
    )

    year_counts = (
        ledger.groupby("audit_period")["date"].size().to_dict()
    )

    return {
        "calendar_days": int(len(ledger)),
        "calendar_consecutive": bool(
            len(expected_dates) == len(ledger)
            and np.array_equal(expected_dates.to_numpy(), ledger["date"].to_numpy())
        ),
        "valid_states": int(ledger["state_valid"].sum()),
        "invalid_states": int((~ledger["state_valid"]).sum()),
        "invalid_dates": sorted(invalid_dates),
        "invalid_date_set_exact": bool(invalid_dates == EXPECTED_INVALID_DATES),
        "year_counts": {str(k): int(v) for k, v in year_counts.items()},
        "year_count_gate_pass": bool(
            all(year_counts.get(y, 0) == EXPECTED_DAYS_PER_YEAR for y in YEAR_LABELS)
        ),
    }


def build_kernels(transitions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    dry = transitions.loc[
        transitions["transition_class"].eq("DRY_NATURAL"),
        "delta_L_power_proxy",
    ].to_numpy(dtype=float)

    rain = transitions.loc[
        transitions["transition_class"].eq("RAIN_AFFECTED"),
        "delta_L_power_proxy",
    ].to_numpy(dtype=float)

    if not np.isfinite(dry).all():
        raise RuntimeError("Dry kernel contains non-finite values.")
    if not np.isfinite(rain).all():
        raise RuntimeError("Rain kernel contains non-finite values.")

    audit = {
        "dry_n": int(len(dry)),
        "rain_n": int(len(rain)),
        "dry_count_pass": bool(len(dry) == EXPECTED_DRY_KERNEL_N),
        "rain_count_pass": bool(len(rain) == EXPECTED_RAIN_KERNEL_N),
        "dry_distribution": qstats(dry),
        "rain_distribution": qstats(rain),
    }
    return dry, rain, audit


def get_year_calendar(ledger: pd.DataFrame, year_label: str) -> pd.DataFrame:
    y = ledger.loc[
        ledger["audit_period"].eq(year_label)
    ].copy().sort_values("date").reset_index(drop=True)

    if len(y) != EXPECTED_DAYS_PER_YEAR:
        raise RuntimeError(
            f"{year_label}: expected {EXPECTED_DAYS_PER_YEAR} rows, got {len(y)}."
        )

    expected = pd.date_range(y["date"].iloc[0], periods=len(y), freq="D")
    if not np.array_equal(expected.to_numpy(), y["date"].to_numpy()):
        raise RuntimeError(f"{year_label}: calendar is not consecutive.")
    return y


def rain_affected_flags(year_df: pd.DataFrame) -> np.ndarray:
    """
    There are 365 decision days but only 364 in-horizon t->t+1 transitions.
    The terminal day earns a terminal-day reward in the future RL environment,
    but no next-state transition is required inside the episode.
    """
    rain = year_df["rain_day"].to_numpy(dtype=bool)
    return rain[:-1] | rain[1:]


def simulate_wait_only(
    year_df: pd.DataFrame,
    dry_kernel: np.ndarray,
    rain_kernel: np.ndarray,
    n_trajectories: int,
    initial_state: float,
    env_seed: int,
) -> dict:
    n_days = len(year_df)
    n_steps = n_days - 1
    rain_flags = rain_affected_flags(year_df)

    rng = np.random.default_rng(env_seed)

    states = np.empty((n_trajectories, n_days), dtype=np.float64)
    raw_next = np.empty((n_trajectories, n_steps), dtype=np.float64)
    sampled_delta = np.empty((n_trajectories, n_steps), dtype=np.float64)
    lower_proj = np.zeros((n_trajectories, n_steps), dtype=bool)
    upper_proj = np.zeros((n_trajectories, n_steps), dtype=bool)

    states[:, 0] = float(initial_state)

    for t in range(n_steps):
        pool = rain_kernel if rain_flags[t] else dry_kernel
        idx = rng.integers(0, len(pool), size=n_trajectories)
        delta = pool[idx]
        raw = states[:, t] + delta

        sampled_delta[:, t] = delta
        raw_next[:, t] = raw
        lower_proj[:, t] = raw < 0.0
        upper_proj[:, t] = raw > 1.0

        states[:, t + 1] = np.clip(raw, 0.0, 1.0)

    return {
        "states": states,
        "raw_next": raw_next,
        "sampled_delta": sampled_delta,
        "lower_proj": lower_proj,
        "upper_proj": upper_proj,
        "rain_flags": rain_flags,
        "state_hash": sha256_array(states),
    }


def first_true_index(mask: np.ndarray) -> np.ndarray:
    """
    Return first True column index per row; -1 if never True.
    """
    any_true = mask.any(axis=1)
    idx = np.argmax(mask, axis=1)
    return np.where(any_true, idx, -1)


def summarise_simulation(
    sim: dict,
    year_df: pd.DataFrame,
    year_label: str,
    init_label: str,
    initial_state: float,
    hist_max: float,
    margin_max: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    states = sim["states"]
    lower_proj = sim["lower_proj"]
    upper_proj = sim["upper_proj"]

    above_hist = states > hist_max
    above_margin = states > margin_max
    at_zero = np.isclose(states, 0.0, atol=1e-15, rtol=0.0)

    first_hist = first_true_index(above_hist)
    first_margin = first_true_index(above_margin)

    traj_rows = []
    dates = year_df["date"].reset_index(drop=True)
    for i in range(states.shape[0]):
        hidx = int(first_hist[i])
        midx = int(first_margin[i])
        traj_rows.append({
            "year_label": year_label,
            "init_mode": init_label,
            "trajectory_id": int(i),
            "initial_state": float(initial_state),
            "terminal_state": float(states[i, -1]),
            "max_state": float(states[i].max()),
            "mean_state": float(states[i].mean()),
            "fraction_state_days_above_hist_max": float(above_hist[i].mean()),
            "fraction_state_days_above_margin_max": float(above_margin[i].mean()),
            "fraction_state_days_at_zero": float(at_zero[i].mean()),
            "any_above_hist_max": bool(above_hist[i].any()),
            "any_above_margin_max": bool(above_margin[i].any()),
            "first_above_hist_day_index": hidx if hidx >= 0 else np.nan,
            "first_above_hist_date": (
                dates.iloc[hidx].strftime("%Y-%m-%d") if hidx >= 0 else ""
            ),
            "first_above_margin_day_index": midx if midx >= 0 else np.nan,
            "first_above_margin_date": (
                dates.iloc[midx].strftime("%Y-%m-%d") if midx >= 0 else ""
            ),
            "lower_projection_count": int(lower_proj[i].sum()),
            "upper_projection_count": int(upper_proj[i].sum()),
        })
    traj_df = pd.DataFrame(traj_rows)

    day_rows = []
    for d in range(states.shape[1]):
        x = states[:, d]
        day_rows.append({
            "year_label": year_label,
            "init_mode": init_label,
            "day_index": int(d),
            "date": dates.iloc[d].strftime("%Y-%m-%d"),
            "state_mean": float(np.mean(x)),
            "state_q05": float(np.quantile(x, 0.05)),
            "state_q50": float(np.quantile(x, 0.50)),
            "state_q95": float(np.quantile(x, 0.95)),
            "state_q99": float(np.quantile(x, 0.99)),
            "state_max": float(np.max(x)),
            "fraction_above_hist_max": float(np.mean(x > hist_max)),
            "fraction_above_margin_max": float(np.mean(x > margin_max)),
            "fraction_at_zero": float(np.mean(np.isclose(x, 0.0, atol=1e-15))),
        })
    day_df = pd.DataFrame(day_rows)

    state_day_hist_frac = float(above_hist.mean())
    state_day_margin_frac = float(above_margin.mean())

    if state_day_hist_frac < HISTORICAL_SUPPORT_WARN:
        support_level = "GREEN_LT_1PCT_ABOVE_HIST_MAX"
    elif state_day_hist_frac <= HISTORICAL_SUPPORT_FAIL:
        support_level = "AMBER_1_TO_5PCT_ABOVE_HIST_MAX"
    else:
        support_level = "RED_GT_5PCT_ABOVE_HIST_MAX"

    summary = {
        "year_label": year_label,
        "init_mode": init_label,
        "n_trajectories": int(states.shape[0]),
        "n_calendar_days": int(states.shape[1]),
        "n_in_horizon_transitions": int(states.shape[1] - 1),
        "initial_state": float(initial_state),
        "rain_affected_transition_count": int(sim["rain_flags"].sum()),
        "dry_transition_count": int((~sim["rain_flags"]).sum()),
        "all_state_distribution": qstats(states.ravel()),
        "terminal_state_distribution": qstats(states[:, -1]),
        "trajectory_max_distribution": qstats(states.max(axis=1)),
        "state_day_fraction_above_historical_max": state_day_hist_frac,
        "state_day_fraction_above_margin_max": state_day_margin_frac,
        "trajectory_fraction_ever_above_historical_max": float(
            above_hist.any(axis=1).mean()
        ),
        "trajectory_fraction_ever_above_margin_max": float(
            above_margin.any(axis=1).mean()
        ),
        "state_day_fraction_at_zero": float(at_zero.mean()),
        "trajectory_fraction_with_any_lower_projection": float(
            lower_proj.any(axis=1).mean()
        ),
        "lower_projection_fraction_of_transitions": float(lower_proj.mean()),
        "trajectory_fraction_with_any_upper_projection": float(
            upper_proj.any(axis=1).mean()
        ),
        "upper_projection_fraction_of_transitions": float(upper_proj.mean()),
        "support_excursion_level": support_level,
        "state_hash": sim["state_hash"],
    }
    return summary, traj_df, day_df


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-1C-1 WAIT-only long-horizon support/boundary audit."
    )
    p.add_argument("--master-ledger", required=True, type=Path)
    p.add_argument("--transition-audit", required=True, type=Path)
    p.add_argument(
        "--n-trajectories",
        type=int,
        default=DEFAULT_N_TRAJECTORIES,
        help="Trajectories per year per initialization mode.",
    )
    p.add_argument(
        "--env-seed",
        type=int,
        default=DEFAULT_ENV_SEED,
        help="Environment RNG seed for reproducibility audit.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1c1_wait_only_state_support_audit_v1"
        ),
    )
    args = p.parse_args()

    if args.n_trajectories < 100:
        raise ValueError("--n-trajectories must be >=100 for this audit.")

    ledger_path = args.master_ledger.expanduser().resolve()
    trans_path = args.transition_audit.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for path in [ledger_path, trans_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/9] Load frozen master ledger and transition audit")
    ledger = load_ledger(ledger_path)
    transitions = load_transition_audit(trans_path)

    print("[2/9] Audit calendar integrity")
    cal_audit = calendar_integrity(ledger)

    print("[3/9] Reconstruct frozen D0/R0 empirical kernels")
    dry_kernel, rain_kernel, kernel_audit = build_kernels(transitions)

    valid_hist = ledger.loc[
        ledger["state_valid"] & np.isfinite(ledger["L_power_proxy"]),
        "L_power_proxy",
    ].to_numpy(dtype=float)

    if valid_hist.size != EXPECTED_VALID_STATES:
        raise RuntimeError(
            f"Expected {EXPECTED_VALID_STATES} valid historical states, "
            f"got {valid_hist.size}."
        )

    hist_max = float(np.max(valid_hist))
    hist_min = float(np.min(valid_hist))
    margin_max = float(hist_max + 0.01)

    support_reference = {
        "historical_valid_state_n": int(valid_hist.size),
        "historical_state_distribution": qstats(valid_hist),
        "historical_min": hist_min,
        "historical_max": hist_max,
        "perception_local_margin_max": margin_max,
        "note": (
            "Historical max is an audited deployment-domain reference, not a "
            "physical hard cap. +0.01 corresponds to the frozen local-support "
            "neighbourhood scale used by the Paper1-derived perception emulator."
        ),
    }

    print("[4/9] Build YEAR1/YEAR2 calendar templates")
    calendars = {y: get_year_calendar(ledger, y) for y in YEAR_LABELS}

    # Primary initialization: actual first-day physical loss proxy of each calendar.
    # Sensitivity: perfectly clean start (L=0).
    init_modes = {}
    for y, ydf in calendars.items():
        first_l = float(ydf["L_power_proxy"].iloc[0])
        if not np.isfinite(first_l):
            raise RuntimeError(f"{y}: first-day L_power_proxy is not finite.")
        init_modes[y] = {
            "HISTORICAL_FIRST_DAY": first_l,
            "CLEAN_START_ZERO": 0.0,
        }

    all_summaries = []
    traj_parts = []
    day_parts = []
    reproducibility_rows = []

    print("[5/9] Run WAIT-only long-horizon simulations")
    for y_idx, y in enumerate(YEAR_LABELS):
        ydf = calendars[y]
        for init_idx, (init_label, initial_state) in enumerate(init_modes[y].items()):
            # Separate deterministic streams for year/init-mode while remaining
            # reproducible from one root env seed.
            scenario_seed = int(
                args.env_seed + 10000 * y_idx + 1000 * init_idx
            )

            sim1 = simulate_wait_only(
                year_df=ydf,
                dry_kernel=dry_kernel,
                rain_kernel=rain_kernel,
                n_trajectories=args.n_trajectories,
                initial_state=initial_state,
                env_seed=scenario_seed,
            )

            summary, traj_df, day_df = summarise_simulation(
                sim=sim1,
                year_df=ydf,
                year_label=y,
                init_label=init_label,
                initial_state=initial_state,
                hist_max=hist_max,
                margin_max=margin_max,
            )
            summary["scenario_env_seed"] = scenario_seed
            all_summaries.append(summary)
            traj_parts.append(traj_df)
            day_parts.append(day_df)

            # Exact reproducibility check: same inputs + same seed => same states.
            sim2 = simulate_wait_only(
                year_df=ydf,
                dry_kernel=dry_kernel,
                rain_kernel=rain_kernel,
                n_trajectories=args.n_trajectories,
                initial_state=initial_state,
                env_seed=scenario_seed,
            )

            same_seed_exact = bool(
                np.array_equal(sim1["states"], sim2["states"])
                and sim1["state_hash"] == sim2["state_hash"]
            )

            # Different seed should produce a genuinely different stochastic path set.
            sim3 = simulate_wait_only(
                year_df=ydf,
                dry_kernel=dry_kernel,
                rain_kernel=rain_kernel,
                n_trajectories=args.n_trajectories,
                initial_state=initial_state,
                env_seed=scenario_seed + 1,
            )
            different_seed_different = bool(
                not np.array_equal(sim1["states"], sim3["states"])
                and sim1["state_hash"] != sim3["state_hash"]
            )

            reproducibility_rows.append({
                "year_label": y,
                "init_mode": init_label,
                "scenario_env_seed": scenario_seed,
                "same_seed_exact_reproduction": same_seed_exact,
                "different_seed_changes_trajectories": different_seed_different,
                "state_hash": sim1["state_hash"],
            })

    print("[6/9] Aggregate support and boundary diagnostics")
    scenario_df = pd.DataFrame(all_summaries)
    traj_df = pd.concat(traj_parts, ignore_index=True)
    day_df = pd.concat(day_parts, ignore_index=True)
    repro_df = pd.DataFrame(reproducibility_rows)

    # Primary support decision uses HISTORICAL_FIRST_DAY initialisation only.
    primary = scenario_df[
        scenario_df["init_mode"].eq("HISTORICAL_FIRST_DAY")
    ].copy()

    primary_hist_frac = float(
        np.average(
            primary["state_day_fraction_above_historical_max"],
            weights=primary["n_trajectories"] * primary["n_calendar_days"],
        )
    )
    primary_margin_frac = float(
        np.average(
            primary["state_day_fraction_above_margin_max"],
            weights=primary["n_trajectories"] * primary["n_calendar_days"],
        )
    )
    primary_upper_proj_frac = float(
        np.average(
            primary["upper_projection_fraction_of_transitions"],
            weights=primary["n_trajectories"] * primary["n_in_horizon_transitions"],
        )
    )

    if primary_hist_frac < HISTORICAL_SUPPORT_WARN:
        primary_support_level = "GREEN"
    elif primary_hist_frac <= HISTORICAL_SUPPORT_FAIL:
        primary_support_level = "AMBER"
    else:
        primary_support_level = "RED"

    support_ready_for_online_perception = bool(
        primary_hist_frac <= HISTORICAL_SUPPORT_FAIL
        and primary_margin_frac <= MARGIN_SUPPORT_FAIL
        and np.isclose(primary_upper_proj_frac, 0.0)
    )

    print("[7/9] Evaluate P2-1C-1 gates")
    calendar_gate = bool(
        cal_audit["calendar_days"] == EXPECTED_CALENDAR_DAYS
        and cal_audit["calendar_consecutive"]
        and cal_audit["valid_states"] == EXPECTED_VALID_STATES
        and cal_audit["invalid_date_set_exact"]
        and cal_audit["year_count_gate_pass"]
    )

    kernel_gate = bool(
        kernel_audit["dry_count_pass"]
        and kernel_audit["rain_count_pass"]
    )

    reproducibility_gate = bool(
        repro_df["same_seed_exact_reproduction"].all()
        and repro_df["different_seed_changes_trajectories"].all()
    )

    physical_boundary_gate = bool(
        np.isfinite(traj_df[
            [
                "terminal_state",
                "max_state",
                "mean_state",
            ]
        ].to_numpy(dtype=float)).all()
        and (traj_df["max_state"] <= 1.0 + 1e-12).all()
        and (traj_df["terminal_state"] >= -1e-12).all()
    )

    all_technical_gates = bool(
        calendar_gate
        and kernel_gate
        and reproducibility_gate
        and physical_boundary_gate
    )

    # Scientific readiness is intentionally stricter than mere code correctness.
    all_primary_gates = bool(
        all_technical_gates
        and support_ready_for_online_perception
    )

    print("[8/9] Build audit summary")
    audit_summary = {
        "stage": "P2-1C-1",
        "audit_only": True,
        "wait_only": True,
        "clean_action_built": False,
        "reward_built": False,
        "online_perception_built": False,
        "rl_started": False,
        "contract_note": (
            "A 365-day decision horizon contains 365 decision/reward days but "
            "364 in-horizon natural t->t+1 transitions. The terminal day does "
            "not require a next state inside the episode."
        ),
        "simulation_design": {
            "n_trajectories_per_year_per_init_mode": int(args.n_trajectories),
            "year_templates": list(YEAR_LABELS),
            "initialization_modes": [
                "HISTORICAL_FIRST_DAY",
                "CLEAN_START_ZERO",
            ],
            "root_env_seed": int(args.env_seed),
            "kernel_dry": "D0_GLOBAL_EMPIRICAL",
            "kernel_rain": "R0_GLOBAL_EMPIRICAL",
            "historical_manual_cleaning_used_as_action": False,
            "scheduled_maintenance_used_as_intervention": False,
            "rain_kernel_rule": "source_rain_day OR dest_rain_day",
            "physical_projection": "[0,1]",
        },
        "calendar_audit": cal_audit,
        "kernel_audit": kernel_audit,
        "support_reference": support_reference,
        "scenario_summaries": all_summaries,
        "primary_support_assessment": {
            "initialization": "HISTORICAL_FIRST_DAY",
            "state_day_fraction_above_historical_max": primary_hist_frac,
            "state_day_fraction_above_margin_max": primary_margin_frac,
            "upper_projection_fraction_of_transitions": primary_upper_proj_frac,
            "support_level": primary_support_level,
            "diagnostic_thresholds": {
                "green_if_hist_exceed_fraction_lt": HISTORICAL_SUPPORT_WARN,
                "red_if_hist_exceed_fraction_gt": HISTORICAL_SUPPORT_FAIL,
                "margin_fail_if_fraction_gt": MARGIN_SUPPORT_FAIL,
            },
            "support_ready_for_online_perception": (
                support_ready_for_online_perception
            ),
        },
        "primary_gates": {
            "calendar_integrity_pass": calendar_gate,
            "kernel_integrity_pass": kernel_gate,
            "reproducibility_pass": reproducibility_gate,
            "physical_boundary_invariant_pass": physical_boundary_gate,
            "technical_gates_pass": all_technical_gates,
            "support_ready_for_online_perception": (
                support_ready_for_online_perception
            ),
            "all_primary_gates_pass": all_primary_gates,
        },
        "interpretation_rule": {
            "if_pass": (
                "The frozen D0/R0 kernels are long-horizon stable enough, within "
                "the declared support thresholds, to proceed to P2-1C-2 online "
                "counterfactual perception audit. Do not add CLEAN/PPO yet."
            ),
            "if_fail": (
                "Do not mask the failure with clipping at the historical maximum. "
                "Diagnose long-horizon drift/support mismatch and revise the "
                "counterfactual dynamics construction before online perception."
            ),
        },
    }

    print("[9/9] Write outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_df.to_csv(
        out_dir / "scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    traj_df.to_csv(
        out_dir / "trajectory_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    day_df.to_csv(
        out_dir / "daywise_state_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    repro_df.to_csv(
        out_dir / "reproducibility_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "scenario_summary.csv")
    print(out_dir / "trajectory_summary.csv")
    print(out_dir / "daywise_state_summary.csv")
    print(out_dir / "reproducibility_audit.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: P2-1C-1 is WAIT-only. "
        "Do NOT add CLEAN, reward, online perception, Gym, or PPO before review."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
