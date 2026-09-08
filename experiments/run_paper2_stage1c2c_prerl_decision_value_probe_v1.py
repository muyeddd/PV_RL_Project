#!/usr/bin/env python
"""P2-1C-2C-v1: smoke-only pre-RL matched-budget sequential probe.

No experiment executes on import. Only --mode smoke is implemented. Threshold
calibration uses clean counts, never decision-value metrics. It is a finite
smoke diagnostic, not formal threshold selection or evidence of UA superiority.
The frozen v1.1 class DOES expose sample_one; no sampler is extracted here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STAGE = "P2-1C-2C-v1"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUT = BASE / "p2_1c_2c_prerl_decision_value_probe_v1"
FROZEN = {
    "experiments/run_paper2_stage1c1r_b_rain_repair_candidate_validation_v1.py":
        "c1430b7b4ccac776433294489984b07af0c82c5f",
    "experiments/run_paper2_stage1c1r_b2_rain_support_integrity_audit_v1.py":
        "54d7ade07113736d7eaca08f49c0d36bca34eb36",
    "experiments/run_paper2_stage1c2a_clean_mechanics_operational_support_audit_v1.py":
        "3be8472e8963e0ee97b6173d54916a92dbc05277",
    "experiments/run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1.py":
        "6f940272ce1a262f325af310c9e72ed0adc401eb",
}
K_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
BUDGETS = [2, 4, 6, 8, 10, 12]
DEV_PERCEPTION_SEED = 20260906
FORMAL_PERCEPTION_SEEDS = [20260907, 20260908, 20260909, 20260910, 20260911]
DEV_ENV_ROOTS = {"YEAR1": 1020001, "YEAR2": 1030001}
FORMAL_ENV_PARENT_SEED = 620001
FORMAL_ENV_ROOTS = {"YEAR1": 1220001, "YEAR2": 1230001}
EXPECTED_R3_INTERCEPT = 0.00090811689781017
EXPECTED_R3_SLOPE = -0.48383380267297343
EXPECTED_R3_SOURCE_L_MAX = 0.0490334865821413
EXPECTED_R3_RESIDUAL_STD = 0.004494874318643408
DEV_TRAJECTORIES = 300
FORMAL_TRAJECTORIES = 1000
ETA = 0.95
SMOKE_N = 20
SMOKE_K = 1.0
SMOKE_BUDGET = 6
FORMAL_CLEAN_COUNT_TOLERANCE = 0.05
PARSIMONY_TIE_PERCENTAGE_POINTS = 0.25
POLICIES = ("True-State", "Point", "UA")
OBSERVATIONS = {"True-State": ["L_true"], "Point": ["q50"],
                "UA": ["q50", "width"]}
GATE_NAMES = [
    "source_provenance_pass", "frozen_sampler_regression_pass",
    "no_reward_pass", "no_future_weather_pass", "no_formal_seed_tuning_pass",
    "env_crn_policy_independent_pass", "perception_crn_policy_independent_pass",
    "point_ua_prefix_identity_pass", "clean_eta_exact_pass",
    "state_projection_semantics_pass", "metric_identity_pass",
    "output_completeness_pass",
]
CATEGORICAL = ["source_sample_id", "source_block_id", "source_date",
               "candidate_samples", "candidate_dates", "candidate_blocks",
               "lower_clipped"]
NUMERIC = ["q50", "lower", "upper", "width"]
METRIC_COLUMNS = ["year", "trajectory", "policy", "perception_seed",
                  "env_seed", "smoke_only_k", "smoke_only_budget_target",
                  "threshold", "J_post", "J_pre", "P_L_pre_gt_0.05",
                  "P_L_pre_gt_0.10", "P_L_pre_gt_0.15", "P95_L_pre",
                  "max_L_pre", "clean_count", "consecutive_clean_pairs",
                  "max_consecutive_clean_run"]


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT, text=True, stderr=subprocess.PIPE,
    ).strip()


def metadata(*args: str):
    try:
        return git(*args)
    except (OSError, subprocess.CalledProcessError):
        return None


def load_frozen():
    """Verify all four files before executing any frozen module code."""
    records = []
    for path, expected in FROZEN.items():
        actual = git("hash-object", path)
        records.append({"path": path, "expected_git_blob": expected,
                        "actual_git_blob": actual})
        if actual != expected:
            raise RuntimeError(f"Frozen source provenance mismatch: {path}")
    path = list(FROZEN)[-1]
    spec = importlib.util.spec_from_file_location("_p2_frozen_online_v11", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    # Avoid writing __pycache__ next to frozen sources at runtime.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module, records


def query(frozen, emulator, L, year, trajectory, day):
    seed = frozen.perception_substream_seed(
        DEV_PERCEPTION_SEED, year, trajectory, day)
    return emulator.sample_one(float(L), np.random.default_rng(seed))


def same_tuple(left, right):
    return (all(left[k] == right[k] for k in CATEGORICAL)
            and all(np.isfinite(left[k]) and np.isfinite(right[k])
                    and abs(left[k] - right[k]) <= 1e-12 for k in NUMERIC))


def sampler_regression(frozen, emulator, bridge, bank):
    """Independent frozen-bank oracle; DEV only, no formal-seed generation.

    Historical continuous RNG is used ONLY for historical-bank regression.
    Sequential policies always use query's per-day substream instead.
    """
    generated = frozen.generate_historical_path(emulator, bridge, DEV_PERCEPTION_SEED)
    reference = bank.loc[bank.perception_seed.eq(DEV_PERCEPTION_SEED)].copy()
    reference = reference.set_index("date").loc[generated["date"]].reset_index()
    reference = reference.rename(columns={"source_block10_id": "source_block_id"})
    exact = {k: bool(np.array_equal(generated[k].to_numpy(),
                                    reference[k].to_numpy())) for k in CATEGORICAL}
    differences = {k: float(np.max(np.abs(generated[k].to_numpy(dtype=float)
                            - reference[k].to_numpy(dtype=float)))) for k in NUMERIC}
    passed = (len(generated) == 729 and all(exact.values())
              and all(np.isfinite(v) and v <= 1e-12 for v in differences.values()))
    if not passed:
        raise RuntimeError(f"Frozen sampler regression FAIL: {exact}, {differences}")
    return {"oracle": "frozen historical bank, DEV rows only",
            "sampler": "direct frozen OnlineBlock10Emulator.sample_one",
            "rows": len(generated), "categorical_exact": exact,
            "numeric_max_abs_diff": differences, "tolerance": 1e-12}


def policy_score(policy, observation):
    # This function's entire information boundary is a numeric observation.
    if policy in ("True-State", "Point"):
        return observation[0]
    if policy == "UA":
        return observation[0] + SMOKE_K * observation[1]
    raise ValueError(policy)


def rollout(frozen, emulator, asset, policy, threshold):
    n, days = SMOKE_N, len(asset["calendar"])
    pre = np.empty((n, days))
    post = np.empty_like(pre)
    raw_next = np.empty((n, days - 1))
    actions = np.zeros((n, days), dtype=bool)
    q50 = np.full_like(pre, np.nan)
    width = np.full_like(pre, np.nan)
    pre[:, 0] = asset["initial"]
    for day in range(days):
        if policy == "True-State":
            # Reference is independent of perception support and sampler calls.
            actions[:, day] = pre[:, day] >= threshold
        elif policy in ("Point", "UA"):
            for traj in range(n):
                rec = query(frozen, emulator, pre[traj, day], asset["year"], traj, day)
                q50[traj, day], width[traj, day] = rec["q50"], rec["width"]
                obs = (rec["q50"],) if policy == "Point" else (rec["q50"], rec["width"])
                actions[traj, day] = policy_score(policy, obs) >= threshold
        else:
            raise ValueError(policy)
        post[:, day] = pre[:, day]
        clean = actions[:, day]
        post[clean, day] = (1.0 - ETA) * pre[clean, day]
        # Metrics settle on the arrays above, before the next natural transition.
        if day == days - 1:
            break
        idx = asset["indices"][day]
        if asset["rain"][day]:
            r3 = asset["r3"]
            delta = (float(r3["intercept"]) + float(r3["slope"]) * post[:, day]
                     + np.asarray(r3["residuals"], dtype=float)[idx])
        else:
            delta = asset["dry"][idx]
        raw_next[:, day] = post[:, day] + delta
        pre[:, day + 1] = np.clip(raw_next[:, day], 0.0, 1.0)
    return {"pre": pre, "post": post, "raw_next": raw_next,
            "actions": actions, "q50": q50, "width": width}


def calibrate_smoke(frozen, emulator, asset, policy):
    """6 fixed bisection proposals; retain closest observed mean clean count.

    Smoke-only directional threshold search assumes approximate local
    monotonicity only for engineering calibration; all proposals and
    budget residuals are reported. Formal DEV calibration will be
    specified separately.
    Unsupported queries stop the stage, with no fallback or widening.
    """
    low, high = 0.0, (2.0 if policy == "UA" else 1.0)
    best = None
    history = []
    for iteration in range(6):
        threshold = (low + high) / 2.0
        result = rollout(frozen, emulator, asset, policy, threshold)
        count = float(result["actions"].sum(axis=1).mean())
        error = abs(count - SMOKE_BUDGET)
        history.append({"iteration": iteration, "threshold": threshold,
                        "mean_clean_count": count,
                        "budget_residual": count - SMOKE_BUDGET,
                        "absolute_budget_residual": error,
                        "relative_target_error": error / SMOKE_BUDGET})
        rank = (error, threshold)
        if best is None or rank < best[0]:
            best = (rank, threshold, result)
        if count > SMOKE_BUDGET:
            low = threshold
        else:
            high = threshold
    return best[1], best[2], history


def metrics(asset, policy, threshold, result):
    rows = []
    for traj, (pre, post, clean) in enumerate(zip(
            result["pre"], result["post"], result["actions"])):
        longest = run = 0
        for action in clean:
            run = run + 1 if action else 0
            longest = max(longest, run)
        rows.append({"year": asset["year"], "trajectory": traj, "policy": policy,
                     "perception_seed": DEV_PERCEPTION_SEED, "env_seed": asset["seed"],
                     "smoke_only_k": SMOKE_K if policy == "UA" else None,
                     "smoke_only_budget_target": SMOKE_BUDGET, "threshold": threshold,
                     "J_post": float(post.sum()), "J_pre": float(pre.sum()),
                     **{f"P_L_pre_gt_{t:.2f}": float(np.mean(pre > t))
                        for t in (0.05, 0.10, 0.15)},
                     "P95_L_pre": float(np.quantile(pre, 0.95)),
                     "max_L_pre": float(pre.max()), "clean_count": int(clean.sum()),
                     "consecutive_clean_pairs": int(np.sum(clean[:-1] & clean[1:])),
                     "max_consecutive_clean_run": longest})
    return rows


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke"], required=True)
    timeline = BASE / "p2_1a_timeline_intervention_audit_v1"
    parser.add_argument("--master-ledger", type=Path,
                        default=timeline / "environment_master_ledger.csv")
    parser.add_argument("--transition-audit", type=Path,
                        default=timeline / "transition_audit.csv")
    parser.add_argument("--paper1-cqr", type=Path, required=True)
    parser.add_argument("--wapp-power-bridge", type=Path, required=True)
    parser.add_argument("--frozen-trajectory-bank", type=Path,
                        default=BASE / "p2_0c_3c_perception_trajectory_bank_v1/trajectory_bank.csv")
    args = parser.parse_args()
    # A previous smoke result is immutable. No overwrite or resume mode.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        OUT.mkdir(exist_ok=False)
    except FileExistsError:
        print(f"Refusing existing output directory: {OUT}", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    manifest = {
        "stage": STAGE, "title": "Pre-RL Matched-Budget Sequential Decision-Value Probe",
        "mode": args.mode, "utc_timestamp": now.isoformat(),
        "local_timestamp": now.astimezone().isoformat(),
        "git_branch": metadata("branch", "--show-current"),
        "head_commit": metadata("rev-parse", "HEAD"),
        "frozen_sources": [{"path": p, "git_blob": h} for p, h in FROZEN.items()],
        "protocol_constants": {
            "K_GRID": K_GRID, "BUDGETS": BUDGETS,
            "DEV_PERCEPTION_SEED": DEV_PERCEPTION_SEED,
            "FORMAL_PERCEPTION_SEEDS": FORMAL_PERCEPTION_SEEDS,
            "DEV_ENV_ROOTS": DEV_ENV_ROOTS, "FORMAL_ENV_ROOTS": FORMAL_ENV_ROOTS,
            "DEV_TRAJECTORIES": DEV_TRAJECTORIES, "FORMAL_TRAJECTORIES": FORMAL_TRAJECTORIES,
            "PRIMARY_METRIC": "J_post = sum(L_post)",
            "SECONDARY": ["J_pre", "P(L_pre > 0.05)", "P(L_pre > 0.10)",
                          "P(L_pre > 0.15)", "P95(L_pre)", "max(L_pre)",
                          "clean_count", "consecutive_clean diagnostics"],
            "formal_clean_count_tolerance": FORMAL_CLEAN_COUNT_TOLERANCE,
            "k_selection": "DEV only; one global k_star across both years and all budgets",
            "parsimony_tie": "mean relative improvement difference <0.25 percentage point: smaller k",
            "parsimony_tie_percentage_points": PARSIMONY_TIE_PERCENTAGE_POINTS,
            "eta": ETA, "dry": "D0_GLOBAL empirical innovations", "rain": "R3_OLS_RESIDUAL",
        },
        "daily_order": ["L_pre", "online perception (Point/UA only)", "action", "L_post",
                        "metric settlement", "natural transition", "next L"],
        "observation_schema": OBSERVATIONS,
        "scores": {"True-State": "L_true", "Point": "q50", "UA": "q50 + k * width"},
        "smoke_only": {"trajectories_per_year": SMOKE_N, "k": SMOKE_K,
                       "budget_target": SMOKE_BUDGET, "perception_seed": DEV_PERCEPTION_SEED,
                       "threshold_search": "6 bisection proposals per policy/year; count-only",
                       "threshold_search_bounds": {"True-State": [0, 1], "Point": [0, 1], "UA": [0, 2]},
                       "threshold_tie": "smaller threshold", "initialization": "HISTORICAL_FIRST_DAY"},
        "formal_selected_k": None, "formal_selected_budget": None,
        "formal_execution_implemented": False,
        "formal_env_parent_seed": FORMAL_ENV_PARENT_SEED,
        "formal_env_roots": FORMAL_ENV_ROOTS,
        # Repository-wide text search (including outputs) before adopting these
        # seeds found no historical usage; this records that review, not a runtime scan.
        "historical_seed_collision_check": "PASS",
        "note": "1120001/1130001 were rejected because previously used in R4 diagnostics.",
        "randomness": {"environment": "frozen indices shared by all policies, year/trajectory/day",
                       "environment_roots_are_final_scenario_seeds": True,
                       "perception": "frozen perception_substream_seed(seed, year, trajectory, day)"},
    }
    gates = {name: False for name in GATE_NAMES}
    summary = {"stage": STAGE, "mode": "smoke", "gates": gates,
               "stage_pass": False, "formal_selected_k": None,
               "budget_calibration": [], "errors": []}
    rows = []
    try:
        frozen, records = load_frozen()
        manifest["verified_sources"] = records
        for year, seed in FORMAL_ENV_ROOTS.items():
            if frozen.scenario_seed(FORMAL_ENV_PARENT_SEED, year) != seed:
                raise RuntimeError(f"Formal environment scenario seed mismatch: {year}")
        inputs = {name: getattr(args, name).expanduser().resolve() for name in
                  ("master_ledger", "transition_audit", "paper1_cqr",
                   "wapp_power_bridge", "frozen_trajectory_bank")}
        manifest["inputs"] = {k: {"path": str(p), "sha256": frozen.sha256_file(p)}
                              for k, p in inputs.items()}
        # Checked P2-1A, P2-1C-2A and P2-1C-2B (including v1.1) outputs:
        # no historical file SHA256 for these two CSVs was recorded. CRN array
        # hashes are not file hashes and must not be presented as such.
        for name in ("master_ledger", "transition_audit"):
            manifest["inputs"][name].update({
                "historical_expected_sha256": None,
                "historical_hash_status": "no_historical_expected_hash",
                "historical_artifacts_checked": ["P2-1A", "P2-1C-2A", "P2-1C-2B", "P2-1C-2B-v1.1"],
            })
        for name in ("paper1_cqr", "wapp_power_bridge", "frozen_trajectory_bank"):
            if manifest["inputs"][name]["sha256"] != frozen.EXPECTED_HASHES[name]:
                raise RuntimeError(f"Frozen input hash mismatch: {name}")
        ledger = frozen.load_ledger(inputs["master_ledger"])
        transitions = frozen.load_transitions(inputs["transition_audit"])
        source = frozen.load_paper1_cqr(inputs["paper1_cqr"])
        bridge = frozen.load_wapp_power_bridge(inputs["wapp_power_bridge"])
        bank = frozen.load_frozen_trajectory_bank(inputs["frozen_trajectory_bank"])
        valid = ledger.loc[ledger.state_valid, ["date", "L_power_proxy"]]
        if (len(ledger) != 730 or len(valid) != 729
                or not np.array_equal(valid.date.dt.strftime("%Y-%m-%d"), bridge.date)
                or not np.array_equal(valid.L_power_proxy.to_numpy(float), bridge.L_power_proxy.to_numpy(float))):
            raise RuntimeError("Ledger/bridge provenance alignment failed")
        # Structural constants are recorded in the historical P2-1A summary.
        if (not np.array_equal(ledger.date, pd.date_range("2021-08-09", "2023-08-08"))
                or int(ledger.rain_day.sum()) != 143
                or int(ledger.modb_manual_cleaning_day.sum()) != 26
                or len(transitions) != 729
                or not np.array_equal(transitions.transition_index, np.arange(729))
                or not np.array_equal(transitions.source_date, ledger.date.iloc[:-1])
                or not np.array_equal(transitions.dest_date, ledger.date.iloc[1:])
                or transitions.transition_class.value_counts().to_dict() != {
                    "DRY_NATURAL": 494, "RAIN_AFFECTED": 201,
                    "MANUAL_CLEAN_CONTAMINATED": 26, "MAINTENANCE_ADJACENT": 6,
                    "INVALID_GAP": 2}):
            raise RuntimeError("Historical ledger/transition structure regression failed")
        x = frozen.attach_context(transitions, ledger)
        natural = x.transition_class.isin(["DRY_NATURAL", "RAIN_AFFECTED"])
        if not np.allclose(x.loc[natural, "delta_L_power_proxy"],
                           x.loc[natural, "dest_L"] - x.loc[natural, "source_L"],
                           atol=1e-12, rtol=0):
            raise RuntimeError("Natural transition delta/ledger regression failed")
        dry = x.loc[x.transition_class.eq("DRY_NATURAL"), "delta_L_power_proxy"].to_numpy(float)
        rain = x.loc[x.transition_class.eq("RAIN_AFFECTED")]
        if (len(dry) != 494 or len(rain) != 201 or not np.isfinite(dry).all()
                or not np.isfinite(rain[["source_L", "delta_L_power_proxy"]]
                                   .to_numpy(float)).all()):
            raise RuntimeError("Frozen transition pool counts/values failed")
        r3 = frozen.fit_r3(rain)
        if len(r3["residuals"]) != 201 or not np.isfinite(r3["residuals"]).all():
            raise RuntimeError("Frozen R3 residual pool counts/values failed")
        r3_actual = {
            "intercept": float(r3["intercept"]),
            "slope": float(r3["slope"]),
            "source_L_max": float(rain["source_L"].max()),
            "residual_std": float(np.std(r3["residuals"], ddof=1)),
        }
        r3_expected = {
            "intercept": EXPECTED_R3_INTERCEPT,
            "slope": EXPECTED_R3_SLOPE,
            "source_L_max": EXPECTED_R3_SOURCE_L_MAX,
            "residual_std": EXPECTED_R3_RESIDUAL_STD,
        }
        r3_checks = {name: bool(np.isclose(value, r3_expected[name],
                                          atol=1e-15, rtol=0))
                     for name, value in r3_actual.items()}
        manifest["dynamics_regression"] = {
            "r3_intercept": r3["intercept"], "r3_slope": r3["slope"],
            "frozen_r3_parameter_regression_pass": all(r3_checks.values()),
            "r3_parameter_checks": r3_checks,
            "r3_actual": r3_actual, "r3_expected": r3_expected,
            "r3_parameter_atol": 1e-15, "r3_parameter_rtol": 0,
            "r3_residual_std_ddof": 1,
            "dry_pool_current_array_sha256": frozen.sha256_array(dry),
            "r3_residual_current_array_sha256": frozen.sha256_array(r3["residuals"]),
            "array_hash_role": "current recording, not historical expected hashes",
        }
        if not all(r3_checks.values()):
            raise RuntimeError(f"Frozen R3 parameter regression failed: {r3_checks}")
        gates[GATE_NAMES[0]] = True
        emulator = frozen.OnlineBlock10Emulator(source)
        manifest["frozen_perception_constants"] = {
            "local_radius": frozen.LOCAL_RADIUS,
            "min_local_samples": frozen.MIN_LOCAL_SAMPLES,
            "min_local_dates": frozen.MIN_LOCAL_DATES,
            "kernel_bandwidth": frozen.KERNEL_BANDWIDTH,
            "block_minutes": frozen.BLOCK_MINUTES,
            "fallback": False, "conditioning": ["true_L"],
            "original_csv_row_order": True,
        }
        summary["sampler_regression"] = sampler_regression(frozen, emulator, bridge, bank)
        gates[GATE_NAMES[1]] = True  # Fail closed before any probe rollout.
        # Executable capability boundaries: action sees exactly the declared tuples.
        gates[GATE_NAMES[2]] = (policy_score.__code__.co_names == ("SMOKE_K", "ValueError"))
        gates[GATE_NAMES[3]] = OBSERVATIONS == {
            "True-State": ["L_true"], "Point": ["q50"], "UA": ["q50", "width"]}
        gates[GATE_NAMES[4]] = (args.mode == "smoke" and DEV_PERCEPTION_SEED == 20260906
            and DEV_PERCEPTION_SEED not in FORMAL_PERCEPTION_SEEDS
            and manifest["formal_selected_k"] is None)
        if not all(gates[n] for n in GATE_NAMES[:5]):
            raise RuntimeError("Pre-probe contract gate failed")
        checks = {name: True for name in GATE_NAMES[5:11]}
        prefix_queries = 0
        for year, seed in DEV_ENV_ROOTS.items():
            calendar = frozen.get_year_calendar(ledger, year)
            # Root values already equal scenario_seed(420001, year); do not add offsets twice.
            if seed != frozen.scenario_seed(420001, year):
                raise RuntimeError("DEV environment scenario seed mismatch")
            indices, rain_days = frozen.build_env_crn_indices(calendar, len(dry), len(r3["residuals"]), SMOKE_N, seed)
            indices2, rain2 = frozen.build_env_crn_indices(calendar, len(dry), len(r3["residuals"]), SMOKE_N, seed)
            checks[GATE_NAMES[5]] &= (np.array_equal(indices, indices2) and np.array_equal(rain_days, rain2))
            for idx in indices:
                idx.setflags(write=False)
            rain_days.setflags(write=False)
            asset = {"calendar": calendar, "initial": float(calendar.L_power_proxy.iloc[0]),
                     "indices": indices, "rain": rain_days, "dry": dry, "r3": r3,
                     "year": year, "seed": seed}
            if not np.isfinite(asset["initial"]) or not 0 <= asset["initial"] <= 1:
                raise RuntimeError("Invalid historical first state")
            results = {}
            for policy in POLICIES:
                threshold, result, history = calibrate_smoke(frozen, emulator, asset, policy)
                results[policy] = result
                count = float(result["actions"].sum(axis=1).mean())
                summary["budget_calibration"].append({"year": year, "policy": policy,
                    "smoke_only_threshold": threshold, "mean_clean_count": count,
                    "relative_target_error": abs(count - SMOKE_BUDGET) / SMOKE_BUDGET,
                    "within_5pct_diagnostic": abs(count - SMOKE_BUDGET) / SMOKE_BUDGET <= 0.05,
                    "proposals": history})
                pre, post, clean = result["pre"], result["post"], result["actions"]
                checks[GATE_NAMES[8]] &= bool(np.array_equal(post[clean], (1.0 - ETA) * pre[clean])
                    and np.array_equal(post[~clean], pre[~clean]) and ETA == 0.95)
                checks[GATE_NAMES[9]] &= bool(np.isfinite(pre).all() and np.isfinite(post).all()
                    and np.array_equal(pre[:, 1:], np.clip(result["raw_next"], 0, 1)))
                # Independent frozen full-rollout oracle for natural/projection mechanics.
                if policy == "True-State":
                    ref_pre, ref_actions = frozen.simulate_threshold_hidden_states(
                        calendar, asset["initial"], threshold, ETA, dry, r3, indices, rain_days)
                    checks[GATE_NAMES[9]] &= bool(np.array_equal(pre, ref_pre)
                                                  and np.array_equal(clean, ref_actions))
                policy_rows = metrics(asset, policy, threshold, result)
                for traj, row in enumerate(policy_rows):
                    expected_post = sum(float(v) for v in post[traj])
                    saved = ETA * float(pre[traj][clean[traj]].sum())
                    checks[GATE_NAMES[10]] &= bool(
                        abs(row["J_post"] - expected_post) <= 1e-10
                        and abs(row["J_pre"] - sum(float(v) for v in pre[traj])) <= 1e-10
                        and abs(row["J_pre"] - row["J_post"] - saved) <= 1e-10
                        and row["clean_count"] == int(np.count_nonzero(clean[traj]))
                        and all(row[f"P_L_pre_gt_{t:.2f}"] ==
                                sum(float(v) > t for v in pre[traj]) / len(calendar)
                                for t in (0.05, 0.10, 0.15))
                        and row["P95_L_pre"] == float(np.percentile(pre[traj], 95))
                        and row["max_L_pre"] == max(pre[traj])
                        and row["consecutive_clean_pairs"] == sum(
                            bool(a and b) for a, b in zip(clean[traj][:-1], clean[traj][1:])))
                rows.extend(policy_rows)
            point, ua = results["Point"], results["UA"]
            same_state = point["pre"] == ua["pre"]
            prefix_queries += int(same_state.sum())
            checks[GATE_NAMES[7]] &= bool(np.array_equal(point["q50"][same_state], ua["q50"][same_state])
                and np.array_equal(point["width"][same_state], ua["width"][same_state]))
            # Replay actual queries in reverse order, independent of policy run order.
            for traj in reversed(range(SMOKE_N)):
                for day in reversed(range(len(calendar))):
                    L = point["pre"][traj, day]
                    a = query(frozen, emulator, L, year, traj, day)
                    b = query(frozen, emulator, L, year, traj, day)
                    checks[GATE_NAMES[6]] &= bool(same_tuple(a, b)
                        and a["q50"] == point["q50"][traj, day]
                        and a["width"] == point["width"][traj, day])
        checks[GATE_NAMES[7]] &= prefix_queries >= 2 * SMOKE_N
        summary["same_hidden_state_point_ua_queries"] = prefix_queries
        gates.update({k: bool(v) for k, v in checks.items()})
        summary["matched_budget_smoke_diagnostic_pass"] = all(
            r["within_5pct_diagnostic"] for r in summary["budget_calibration"])
        summary["interpretation"] = (
            "Smoke engineering checks only. Budget mismatches are reported; no formal "
            "decision-value conclusion or selected k/budget is produced.")
    except Exception as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    # On failure, still emit the manifest, all gates, and a possibly empty CSV.
    try:
        dump(OUT / "protocol_manifest.json", manifest)
        frame = pd.DataFrame(rows, columns=METRIC_COLUMNS)
        with (OUT / "smoke_trajectory_metrics.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            frame.to_csv(stream, index=False)
        loaded = pd.read_csv(OUT / "smoke_trajectory_metrics.csv")
        gates[GATE_NAMES[11]] = bool(len(loaded) == 2 * SMOKE_N * len(POLICIES)
            and list(loaded.columns) == METRIC_COLUMNS
            and not loaded.duplicated(["year", "trajectory", "policy"]).any()
            and np.isfinite(loaded[[c for c in METRIC_COLUMNS if c not in
                ("year", "policy", "smoke_only_k")]].to_numpy(float)).all()
            and json.loads((OUT / "protocol_manifest.json").read_text(encoding="utf-8"))["mode"] == "smoke")
        summary["stage_pass"] = bool(all(gates.values()) and not summary["errors"])
        summary["numbered_gates"] = {f"G{i}": {"name": name, "pass": gates[name]}
                                       for i, name in enumerate(GATE_NAMES, 1)}
        dump(OUT / "smoke_audit_summary.json", summary)
        reread = json.loads((OUT / "smoke_audit_summary.json").read_text(encoding="utf-8"))
        if reread != summary:
            raise RuntimeError("Summary serialization verification failed")
    except Exception as exc:
        print(f"Output failure: {exc}", file=sys.stderr)
        return 2
    for i, name in enumerate(GATE_NAMES, 1):
        print(f"G{i} {name}: {'PASS' if gates[name] else 'FAIL'}")
    print(f"stage_pass={summary['stage_pass']}")
    for error in summary["errors"]:
        print(error, file=sys.stderr)
    return 0 if summary["stage_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
