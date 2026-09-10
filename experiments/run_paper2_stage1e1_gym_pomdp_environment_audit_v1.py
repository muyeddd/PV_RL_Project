#!/usr/bin/env python
"""P2-1E-1-v1: Gym/POMDP Environment Contract and Regression Audit.

Explicit execution only. Both modes use development EpisodeSpecs exclusively.
No training, optimization, policy comparison, selection, or contract tuning.
The two checkers are actually executed inside run_checkers, never at import.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from dataclasses import asdict
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import paper2_gym_pomdp_env_v1 as core


STAGE = "P2-1E-1-v1"
TITLE = "Gym/POMDP Environment Contract and Regression Audit"
SELF = "experiments/run_paper2_stage1e1_gym_pomdp_environment_audit_v1.py"
ENV_FILE = "experiments/paper2_gym_pomdp_env_v1.py"
OUTPUT = core.BASE / "p2_1e_1_gym_pomdp_environment_audit_v1"
ENV_ROOTS = {"YEAR1": 1020001, "YEAR2": 1030001}
POPULATION_SIZE = 300
PERCEPTION_SEED = 20260906
POLICIES = ("NEVER_CLEAN", "DAILY_CLEAN", "TRUE_THRESHOLD_0.05")
MODES = ("True-State", "Point", "UA")
ATOL = 1e-12
RTOL = 0.0
EXPECTED_VERSIONS = {"Python": "3.11.15", "gymnasium": "1.2.3", "stable_baselines3": "2.7.1",
                     "torch": "2.10.0+cu130", "numpy": "2.3.5", "pandas": "2.3.3"}
INFO_KEYS = {"date", "day_index", "observation_mode", "action_name", "terminated_reason",
             "terminal_observation_sentinel", "episode_return", "clean_count"}
PHYSICS_COLUMNS = ["L_pre", "action", "L_post", "E_clean_GHI", "soiling_cost", "cleaning_cost",
                   "total_cost", "raw_reward", "reward_scale", "final_reward", "L_next"]
GATES = dict(enumerate([
    "reward_checkpoint_ancestry", "reward_predecessor_blob", "frozen_reward_artifact_hashes",
    "frozen_reward_contract_exact", "natural_helper_provenance", "perception_helper_provenance",
    "timeline_energy_integrity", "dependency_versions_exact", "Gymnasium_checker_pass", "SB3_checker_pass",
    "action_observation_spaces_exact", "reset_semantics", "365_steps_364_transitions", "termination_truncation",
    "CLEAN_WAIT_mechanics_exact", "natural_dynamics_CRN_exact", "Gym_seed_scientific_independence",
    "reward_daily_annual_regression", "latent_continuity_2022_06_29", "True_State_no_perception",
    "Point_UA_q50_season_pairing", "no_GHI_future_weather_info_leakage", "perception_support_fail_closed",
    "observation_mode_physics_independence", "illegal_action_state_immutability", "output_completeness",
    "both_current_files_committed_unchanged",
], start=1))


def check(condition, message):
    core.require(condition, message)


def gate(audit, number, condition, detail):
    audit["gates"][GATES[number]] = bool(condition)
    audit["gate_details"][f"G{number}"] = detail
    check(condition, f"G{number} {GATES[number]}: {detail}")


def close(actual, expected):
    return bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=False))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_csv(payload):
    return pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", float_precision="round_trip")


def development_spec(year, trajectory):
    check(year in ENV_ROOTS and isinstance(trajectory, int) and 0 <= trajectory < 10, "Audit development episode restriction")
    return core.EpisodeSpec(year=year, environment_root=ENV_ROOTS[year], trajectory_id=trajectory,
                            population_size=POPULATION_SIZE, perception_seed=PERCEPTION_SEED)


def dependency_versions():
    result = {"Python": platform.python_version()}
    for name in EXPECTED_VERSIONS:
        if name != "Python":
            result[name] = metadata.version(name)
    return result


def provenance(audit, manifest):
    head = core.git("rev-parse", "HEAD")
    records = {path: core.script_record(path, head) for path in (SELF, ENV_FILE)}
    gate(audit, 27, len(records) == 2, records)
    manifest.update(current_HEAD=head, current_files=records)
    actual = dependency_versions()
    manifest["dependency_versions"] = actual
    # Strict in both modes: smoke implementation checks should not hide drift.
    gate(audit, 8, actual == EXPECTED_VERSIONS, {"expected": EXPECTED_VERSIONS, "actual": actual})


def asset_gates(assets, audit, manifest):
    p = assets.provenance
    gate(audit, 1, p["checkpoint_ancestor"] and p["checkpoint"] == core.CHECKPOINT, p["checkpoint"])
    gate(audit, 2, p["reward_script"]["git_blob"] == core.REWARD_BLOB, p["reward_script"])
    gate(audit, 3, p["reward_hashes"] == core.REWARD_HASHES, p["reward_hashes"])
    gate(audit, 4, assets.lambda_main == core.LAMBDA_MAIN and assets.eta == .95 and assets.reward_scale == 1., assets.reward_contract)
    gate(audit, 5, all(p["helpers"][name]["git_blob"] == core.HELPERS[name][1] for name in ("mechanics", "natural")),
         {"helpers": {name: p["helpers"][name] for name in ("mechanics", "natural")}, "regression": p["natural_regression"]})
    gate(audit, 6, p["helpers"]["perception"]["git_blob"] == core.HELPERS["perception"][1], p["helpers"]["perception"])
    gate(audit, 7, len(assets.ledger) == len(assets.energy) == 730 and len(assets.transitions) == 729, p["assets"])
    manifest["frozen_provenance"] = p


def space_check(env):
    check(isinstance(env.action_space, core.gym.spaces.Discrete) and env.action_space.n == 2
          and env.action_space.start == 0, "G11 exact Discrete(2)")
    low = [0, 0, -1, -1] if env.observation_mode == "UA" else [0, -1, -1]
    high = [1] * len(low)
    space = env.observation_space
    check(isinstance(space, core.gym.spaces.Box) and space.shape == (len(low),)
          and space.dtype == np.float32 and np.array_equal(space.low, np.array(low, dtype=np.float32))
          and np.array_equal(space.high, np.array(high, dtype=np.float32)), "G11 exact observation Box")


def observation_check(env, observation, day, latent, terminal=False):
    check(isinstance(observation, np.ndarray) and observation.dtype == np.float32
          and env.observation_space.contains(observation) and np.isfinite(observation).all(), "G11 finite in-space observation")
    if terminal:
        check(np.array_equal(observation, np.zeros(env.observation_space.shape, dtype=np.float32)), "G14 terminal zero sentinel")
        return
    date = env._calendar.date.iloc[day]
    phase = 2 * np.pi * (date.dayofyear - 1) / 365
    check(np.array_equal(observation[-2:], np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32)), "G22 real date seasonal encoding")
    if env.observation_mode == "True-State":
        check(observation[0] == np.float32(latent), "G11 true latent channel")
    else:
        spec = env.episode_spec
        key = (int(spec.perception_seed), spec.year, int(spec.trajectory_id), day, float(latent))
        q50, width = env._assets._perception_cache[key]
        content = [q50, width] if env.observation_mode == "UA" else [q50]
        check(np.array_equal(observation[:-2], np.asarray(content, dtype=np.float32)),
              "G22 observation contains only the frozen q50/width channel")


def info_check(env, info, day, terminal=False):
    check(isinstance(info, dict) and set(info) <= INFO_KEYS, "G22 public info allowlist")
    check(info["date"] == env._calendar.date.iloc[day].strftime("%Y-%m-%d")
          and info["day_index"] == day and info["observation_mode"] == env.observation_mode
          and info["terminal_observation_sentinel"] is terminal, "G22 public info semantics")
    if not terminal:
        check(not {"episode_return", "clean_count", "terminated_reason"}.intersection(info), "G22 terminal-only diagnostics")


def illegal_actions(env):
    bad = [-1, 2, .5, float("nan"), np.array([0]), np.array([0, 1]), np.array([[1]]), np.array(0)]
    before = env._audit_snapshot()
    counts = (env._assets.perception_sample_count, env._assets.bank_generation_count)
    for action in bad:
        try:
            env.step(action)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"G25 illegal action accepted: {action!r}")
        check(env._audit_snapshot() == before and counts == (env._assets.perception_sample_count, env._assets.bank_generation_count),
              "G25 illegal action changed scientific state/counters")
    return len(bad)


def run_episode(env, policy, frozen, gym_seed, supplied_actions=None):
    space_check(env)
    spec_before = env.episode_spec
    obs, info = env.reset(seed=gym_seed)
    snapshot = env._audit_snapshot()
    check(snapshot["day_index"] == 0 and snapshot["L_pre"] == env._initial and not snapshot["terminated"]
          and snapshot["transition_count"] == snapshot["clean_count"] == 0 and not snapshot["needs_reset"], "G12 reset semantics")
    info_check(env, info, 0)
    invalid_count = illegal_actions(env)
    observations, rows, actions = [], [], []
    for day in range(365):
        before = env._audit_snapshot()
        check(before["day_index"] == day and type(before["L_pre"]) is float, "G13 internal float64 state/day")
        observation_check(env, obs, day, before["L_pre"])
        observations.append(obs.copy())
        if supplied_actions is not None:
            action = int(supplied_actions[day])
        elif policy == "NEVER_CLEAN":
            action = 0
        elif policy == "DAILY_CLEAN":
            action = 1
        else:
            check(policy == "TRUE_THRESHOLD_0.05", "Fixed trace policy")
            action = int(before["L_pre"] >= .05)  # Audit probe, never a policy observation.
        # Exercise both Python and NumPy integer actions without changing sequence.
        outward_action = np.int64(action) if day % 2 else action
        result = env.step(outward_action)
        check(isinstance(result, tuple) and len(result) == 5, "Gymnasium five-tuple")
        obs, reward, terminated, truncated, info = result
        check(type(reward) is float and type(terminated) is bool and type(truncated) is bool,
              "Outward Gym scalar types")
        check(terminated is (day == 364) and truncated is False, "G14 exact termination/truncation")
        info_check(env, info, day, terminal=terminated)
        after = env._audit_snapshot()
        last = after["last_reward_components"]
        check(last["day_index"] == day and last["L_pre"] == before["L_pre"] and last["action"] == action,
              "G15 pre-action identity")
        check(close(last["L_post"], (.05 * before["L_pre"]) if action else before["L_pre"]), "G15 CLEAN/WAIT identity")
        check(close(last["soiling_cost"], last["E_clean_GHI"] * last["L_post"])
              and close(last["cleaning_cost"], core.LAMBDA_MAIN * action)
              and close(last["total_cost"], last["soiling_cost"] + last["cleaning_cost"])
              and close(reward, -last["total_cost"]) and last["reward_scale"] == 1.
              and last["raw_reward"] == last["final_reward"] == reward, "G18 daily frozen reward algebra")
        check(after["transition_count"] == min(day + 1, 364), "G13 transition count")
        if day < 364:
            idx, post = int(env._indices[day]), last["L_post"]
            r3 = env._assets.r3
            raw = (post + (r3["intercept"] + r3["slope"] * post) + r3["residuals"][idx]
                   if env._rain[day] else post + env._assets.dry[idx])
            check(last["L_next"] == float(np.clip(raw, 0., 1.)) == after["L_pre"]
                  and last["transition_executed"] is True, "G16 exact indexed natural dynamics")
        else:
            check(last["L_next"] is None and last["transition_executed"] is False, "G14 no fake next state")
            observation_check(env, obs, day, after["L_pre"], terminal=True)
            check(info["episode_return"] == after["episode_return"] and info["clean_count"] == after["clean_count"], "Terminal totals")
        actions.append(action)
        rows.append(last)
    final_snapshot = env._audit_snapshot()
    try:
        env.step(0)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("G14 step after terminal did not require reset")
    check(env._audit_snapshot() == final_snapshot, "Step after terminal changed state")
    check(env.episode_spec == spec_before and final_snapshot["transition_count"] == 364, "G12 fixed scientific episode")
    query_count = final_snapshot["perception_query_count"]
    check(query_count == (0 if env.observation_mode == "True-State" else 365), "G20 observation-day perception query count")
    trace = pd.DataFrame(rows)
    reference = frozen.reset_index(drop=True)
    check(len(reference) == len(trace) == 365 and trace.date.tolist() == reference.date.tolist(), "G18 exact frozen dates")
    differences = {}
    for column in PHYSICS_COLUMNS:
        actual = trace[column].to_numpy(float)
        expected = reference[column].to_numpy(float)
        if column == "L_next":
            check(np.isnan(actual[-1]) and np.isnan(expected[-1]), "Terminal frozen next-state semantics")
            actual, expected = actual[:-1], expected[:-1]
        check(np.isfinite(actual).all() and close(actual, expected), f"G18 frozen daily regression: {column}")
        differences[column] = float(np.max(np.abs(actual - expected)))
    for column in ("rain_affected_transition", "transition_executed"):
        check(np.array_equal(trace[column], reference[column]), f"G16 frozen transition mask: {column}")
    annual = float(trace.final_reward.sum())
    check(close(annual, reference.final_reward.sum()) and close(final_snapshot["episode_return"], annual)
          and close(annual, -(trace.soiling_cost.sum() + core.LAMBDA_MAIN * trace.action.sum())), "G18 annual reward identity")
    check(final_snapshot["clean_count"] == int(reference.action.sum()), "G18 annual clean count")
    if env.episode_spec.year == "YEAR1":
        missing = trace.loc[trace.date.eq("2022-06-29")]
        check(len(missing) == 1 and np.isfinite(missing[["L_pre", "L_post", "E_clean_GHI", "final_reward"]].to_numpy()).all()
              and missing.transition_executed.all(), "G19 retained latent continuity day")
    summary = {**asdict(env.episode_spec), "observation_mode": env.observation_mode, "policy": policy,
               "gym_seed": gym_seed, "steps": len(trace), "natural_transition_count": final_snapshot["transition_count"],
               "perception_query_count": query_count, "annual_reward": annual, "frozen_annual_reward": float(reference.final_reward.sum()),
               "annual_abs_diff": abs(annual - float(reference.final_reward.sum())), "clean_count": final_snapshot["clean_count"],
               "max_daily_abs_diff": max(differences.values()), "indices_sha256": final_snapshot["indices_sha256"],
               "full_bank_sha256": final_snapshot["full_bank_sha256"], "invalid_actions_rejected": invalid_count,
               "regression_pass": True}
    return trace, np.stack(observations), actions, summary


def frozen_population(assets):
    frame = read_csv(assets.reward_payloads["daily_reward_trace.csv"])
    keys = ["year", "trajectory_id", "policy", "day_index"]
    expected = {(year, tid, policy, day) for year in ENV_ROOTS for tid in range(10) for policy in POLICIES for day in range(365)}
    check(len(frame) == 21900 and set(frame[keys].itertuples(index=False, name=None)) == expected, "Exact frozen trace population")
    return frame


def audit_episodes(assets, n, audit):
    frozen = frozen_population(assets)
    summary_rows, pairs, representatives = [], [], []
    true_guard_episodes = 0
    for year in ENV_ROOTS:
        for tid in range(n):
            spec = development_spec(year, tid)
            for policy in POLICIES:
                reference = frozen.loc[frozen.year.eq(year) & frozen.trajectory_id.eq(tid) & frozen.policy.eq(policy)].sort_values("day_index")
                results = {}
                shared_actions = None
                for mode in MODES:
                    with ExitStack() as guard:
                        before_counts = (assets.perception_construction_count, assets.perception_sample_count)
                        if mode == "True-State":
                            guard.enter_context(patch.object(assets, "ensure_perception", side_effect=AssertionError("True-State constructed perception")))
                            guard.enter_context(patch.object(assets, "perception_record", side_effect=AssertionError("True-State queried perception")))
                        env = core.Paper2CleaningEnv(assets, spec, mode)
                        first = run_episode(env, policy, reference, 1, shared_actions)
                        # Same env, different Gym seed, identical externally supplied actions.
                        second = run_episode(env, policy, reference, 999, first[2])
                        pd.testing.assert_frame_equal(first[0], second[0], check_exact=True)
                        check(np.array_equal(first[1], second[1]), "G17 Gym seed changed scientific observations")
                        check(first[3]["indices_sha256"] == second[3]["indices_sha256"], "G17 Gym seed changed innovations")
                        if mode == "True-State":
                            shared_actions = first[2]
                            check(before_counts == (assets.perception_construction_count, assets.perception_sample_count), "G20 guarded zero perception")
                            true_guard_episodes += 2
                        summary_rows.extend([first[3], second[3]])
                        results[mode] = first
                        if tid == 0:
                            representative = first[0].copy()
                            representative.insert(2, "policy", policy)
                            representative.insert(3, "observation_mode", mode)
                            representatives.append(representative)
                        reference_crn = assets.reward_protocol["CRN"][year]
                        check(env._full_bank_hash == reference_crn["full_bank_sha256"]
                              and env._bank_report == reference_crn["full_bank"] and env._initial == reference_crn["initial_L"]
                              and not env._indices.flags.writeable, "G16 frozen CRN bank and initial state")
                        env.close()
                for mode in ("Point", "UA"):
                    pd.testing.assert_frame_equal(results["True-State"][0], results[mode][0], check_exact=True)
                    check(results[mode][3]["indices_sha256"] == results["True-State"][3]["indices_sha256"], "G24 shared environment indices")
                point, ua = results["Point"][1], results["UA"][1]
                check(point.shape == (365, 3) and ua.shape == (365, 4), "G21 only UA adds width")
                delta = np.abs(point - ua[:, [0, 2, 3]])
                check(np.array_equal(point, ua[:, [0, 2, 3]]), "G21 exact Point-UA q50 and season pairing")
                pairs.append({"year": year, "trajectory_id": tid, "policy": policy, "days": 365,
                              "max_abs_q50_diff": float(delta[:, 0].max()), "max_abs_sin_diff": float(delta[:, 1].max()),
                              "max_abs_cos_diff": float(delta[:, 2].max()), "q50_season_bitwise_equal": point.tobytes() == ua[:, [0, 2, 3]].copy(order="C").tobytes(),
                              "UA_width_min": float(ua[:, 1].min()), "UA_width_max": float(ua[:, 1].max()), "pairing_pass": True})
    summaries = pd.DataFrame(summary_rows)
    check(len(summaries) == 2 * n * 3 * 3 * 2 and len(pairs) == 2 * n * 3, "Exact audit episode population")
    check(assets.bank_generation_count == 2 and len(assets._banks) == 2, "G16 generate each year/root bank once across modes/resets")
    gate(audit, 11, True, "All observation/action spaces and all 365 decision observations checked")
    gate(audit, 12, True, "Fixed constructor EpisodeSpec; day0 historical latent; resets reproduce")
    gate(audit, 13, summaries.steps.eq(365).all() and summaries.natural_transition_count.eq(364).all(), {"episodes": len(summaries)})
    gate(audit, 14, True, "365th step terminated; never truncated; zero sentinel; post-terminal step raises without mutation")
    gate(audit, 15, True, "Every step: WAIT identity or CLEAN .05 identity at float64 atol=1e-12")
    gate(audit, 16, True, {"full_bank_shape": [364, 300], "generated_banks": assets.bank_generation_count,
                          "all_columns_read_only": True, "indexed_natural_and_frozen_trace_regression": True})
    gate(audit, 17, True, "Same env reset(seed=1/999), full hidden/reward/observation trajectories exactly equal")
    gate(audit, 18, summaries.regression_pass.all(), {"daily_atol": ATOL, "rtol": RTOL,
                                                  "max_daily_abs_diff": float(summaries.max_daily_abs_diff.max()),
                                                  "max_annual_abs_diff": float(summaries.annual_abs_diff.max())})
    day = assets.ledger.loc[assets.ledger.date.eq(pd.Timestamp("2022-06-29"))]
    check(len(day) == 1 and not bool(day.state_valid.iloc[0]) and not bool(day.bridge_valid.iloc[0])
          and not np.isfinite(day.L_power_proxy.iloc[0]) and np.isfinite(assets.energy.loc[pd.Timestamp("2022-06-29")])
          and assets.energy.loc[pd.Timestamp("2022-06-29")] > 0, "G19 observed missing state, valid realized energy")
    gate(audit, 19, True, {"date": "2022-06-29", "settlement": "simulator latent L plus frozen realized E_clean_GHI",
                          "all_YEAR1_episodes_retained_date": True, "observed_state_interpolation": False})
    gate(audit, 20, true_guard_episodes == 2 * n * 3 * 2,
         {"guarded_full_True_State_episodes": true_guard_episodes, "True_State_queries": 0, "Point_UA_queries_per_episode": 365})
    gate(audit, 21, all(row["q50_season_bitwise_equal"] for row in pairs), {"paired_episodes": len(pairs), "all_differences": 0})
    gate(audit, 22, True, {"public_info_allowlist": sorted(INFO_KEYS), "all_observations_schema_checked": True,
                          "E_clean_GHI_role": "REALIZED_REWARD_SETTLEMENT_ONLY", "hidden_data_access": "private audit accessor only"})
    gate(audit, 24, True, "Every mode: identical externally supplied actions, exact hidden dynamics/reward and indices")
    gate(audit, 25, summaries.invalid_actions_rejected.eq(8).all(), "Invalid scalars/arrays rejected before state or counter mutation")
    return summaries, pd.DataFrame(pairs), pd.concat(representatives, ignore_index=True)


def support_failure_probe(assets, audit):
    """Fault injection verifies exception adaptation and no retry/fallback.

    This does not change frozen source files or model parameters. Real unsupported
    queries in any episode/checker still abort the stage; none are suppressed.
    """
    assets.ensure_perception()
    contexts = []
    for mode in ("Point", "UA"):
        env = core.Paper2CleaningEnv(assets, development_spec("YEAR1", 0), mode)
        # Isolate memoization only for the negative test, restoring it afterwards.
        with patch.object(assets, "_perception_cache", {}):
            with patch.object(assets.emulator, "sample_one", side_effect=RuntimeError("Unsupported query injected support-contract probe")) as sampler:
                try:
                    env.reset(seed=1)
                except core.PerceptionSupportError as exc:
                    context = exc.context
                    required = {"year", "trajectory_id", "day_index", "date", "L_true", "perception_seed", "local_rows", "local_dates", "candidate_blocks"}
                    check(required <= set(context) and context["year"] == "YEAR1" and context["trajectory_id"] == 0
                          and context["day_index"] == 0 and context["perception_seed"] == PERCEPTION_SEED
                          and context["local_rows"] is not None and sampler.call_count == 1, "G23 exception context/no retry")
                    contexts.append({"mode": mode, "context": context, "sampler_attempts": sampler.call_count})
                else:
                    raise RuntimeError("G23 unsupported perception did not fail closed")
                try:
                    env.step(0)
                except RuntimeError:
                    pass
                else:
                    raise RuntimeError("G23 unsupported reset left a usable episode")
        env.close()
    gate(audit, 23, len(contexts) == 2, {"negative_test": "injected frozen Unsupported query; real failures remain fatal", "probes": contexts})


def run_checkers(assets, audit):
    # SB3 is imported ONLY for its environment checker, never a learner.
    from gymnasium.utils import env_checker as gym_checker
    from stable_baselines3.common import env_checker as sb3_checker

    results = []
    for mode in MODES:
        for label, checker in (("gymnasium", gym_checker.check_env), ("stable_baselines3", sb3_checker.check_env)):
            env = core.Paper2CleaningEnv(assets, development_spec("YEAR1", 0), mode)
            failure = None
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    checker(env)
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                finally:
                    env.close()
            results.append({"checker": label, "observation_mode": mode, "passed": failure is None,
                            "hard_failure": failure, "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured]})
    audit["checker_results"] = results
    # Preserve evidence for both checker gates even when one hard-fails.
    for number, label in ((9, "gymnasium"), (10, "stable_baselines3")):
        subset = [row for row in results if row["checker"] == label]
        audit["gates"][GATES[number]] = len(subset) == 3 and all(row["passed"] for row in subset)
        audit["gate_details"][f"G{number}"] = subset
    check(audit["gates"][GATES[9]] and audit["gates"][GATES[10]], "Gymnasium/SB3 checker hard failure; warnings recorded separately")
    return {"results": results, "warnings_are_hard_failures": False, "actual_check_env_calls": len(results)}


def static_audit():
    """Supplement runtime evidence with prohibited-call/import checks."""
    for path in (SELF, ENV_FILE):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("stable_baselines3"):
                check(node.module == "stable_baselines3.common" and [alias.name for alias in node.names] == ["env_checker"], "SB3 checker-only import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                check(node.func.attr not in {"learn", "backward", "minimize"}, "No training or optimization")
    tree = ast.parse((ROOT / ENV_FILE).read_text(encoding="utf-8"))
    env = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Paper2CleaningEnv")
    reset = next(node for node in env.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
    seed_reads = [node for node in ast.walk(reset) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "seed"]
    check(len(seed_reads) == 1, "Gym seed only forwarded to super reset")
    return {"passed": True, "SB3_usage": "env_checker only", "Gym_seed_loaded_once": True}


def protocol(mode):
    return {"stage": STAGE, "title": TITLE, "mode": mode, "expected_versions": EXPECTED_VERSIONS,
            "development_only": True, "environment_roots": ENV_ROOTS, "population_size": POPULATION_SIZE,
            "trajectory_ids": list(range(10 if mode == "formal" else 2)), "perception_seed": PERCEPTION_SEED,
            "observation_modes": list(MODES), "trace_policies": list(POLICIES), "gym_reset_test_seeds": [1, 999],
            "scientific_seed_semantics": "EpisodeSpec fixes year/root/trajectory/population/perception; Gym seed only initializes superclass RNG",
            "audit_action_source": "True-State fixed probes; identical externally supplied actions replayed in Point/UA",
            "observation_dtype": "float32", "internal_physics_reward_dtype": "float64", "reward_atol": ATOL, "reward_rtol": RTOL,
            "seasonal_encoding": "2*pi*(date.dayofyear-1)/365", "terminal_observation": "zero float32 vector; no new decision state",
            "public_info_allowlist": sorted(INFO_KEYS), "E_clean_GHI_role": "REALIZED_REWARD_SETTLEMENT_ONLY",
            "no_training": True, "no_policy_comparison": True, "no_tuning": True, "required_gates": GATES}


def publish(out, manifest, api, summaries, pairs, trace, checkers, audit, assets):
    check(all(value for name, value in audit["gates"].items() if name != GATES[26]), "All implementation gates required before outputs")
    frames = {"episode_regression_summary.csv": summaries, "observation_pairing_summary.csv": pairs,
              "representative_step_trace.csv": trace}
    payloads = {"protocol_manifest.json": encode(manifest), "environment_api_summary.json": encode(api),
                "checker_summary.json": encode(checkers)}
    payloads.update({name: frame.to_csv(index=False).encode("utf-8-sig") for name, frame in frames.items()})
    index = {"hashes": {name: core.sha(data) for name, data in payloads.items()},
             "excluded": ["output_hashes.json", "audit_summary.json"],
             "semantics": "Final audit certifies actual disk readback; partial manifests alone never freeze the environment"}
    payloads["output_hashes.json"] = encode(index)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (out / name).open("xb") as stream:
            stream.write(data)
    verified = {}
    for name, expected in payloads.items():
        actual = (out / name).read_bytes()
        check(actual == expected and core.sha(actual) == core.sha(expected), f"Output bytes/hash: {name}")
        if name in index["hashes"]:
            check(core.sha(actual) == index["hashes"][name], f"Hash index: {name}")
        if name in frames:
            reread = read_csv(actual)
            check(reread.columns.tolist() == frames[name].columns.tolist() and len(reread) == len(frames[name]), f"CSV schema: {name}")
            pd.testing.assert_frame_equal(frames[name].reset_index(drop=True), reread, check_dtype=False, check_exact=True)
        else:
            check(json.loads(actual) == json.loads(expected), f"JSON schema/content: {name}")
        verified[name] = core.sha(actual)
    # Final provenance recheck; no changes may be hidden by cached inputs/modules.
    check(core.git("rev-parse", "HEAD") == manifest["current_HEAD"], "HEAD changed during audit")
    for path, record in manifest["current_files"].items():
        check(core.script_record(path, manifest["current_HEAD"]) == record, "Current file changed during audit")
    check(core.script_record(core.REWARD_SCRIPT, manifest["current_HEAD"], core.REWARD_BLOB) == assets.provenance["reward_script"], "Reward script changed")
    for name, (path, blob) in core.HELPERS.items():
        check(core.script_record(path, manifest["current_HEAD"], blob) == assets.provenance["helpers"][name], "Helper changed during audit")
    for name, digest in core.REWARD_HASHES.items():
        check(core.sha((core.REWARD_DIRECTORY / name).read_bytes()) == digest, "Reward artifact changed during audit")
    for path, digest in core.ASSET_HASHES.values():
        check(core.sha(path.read_bytes()) == digest, "Timeline/energy changed during audit")
    for name, record in assets.provenance["perception_sources"].items():
        check(assets.perception.sha256_file(Path(record["path"])) == assets.perception.EXPECTED_HASHES[name] == record["sha256"], "Perception source changed during audit")
    gate(audit, 26, set(p.name for p in out.iterdir()) == set(payloads), {"verified_non_audit_output_sha256": verified,
                                                                     "verification": "actual write/readback/hash/schema/content"})
    audit["stage_pass"] = all(audit["gates"].values())
    audit["failed_gates"] = [name for name, passed in audit["gates"].items() if not passed]
    audit["scientific_status"] = api["scientific_status"]
    data = encode(audit)
    with (out / "audit_summary.json").open("xb") as stream:
        stream.write(data)
    check((out / "audit_summary.json").read_bytes() == data, "Final audit readback")


def main():
    parser = argparse.ArgumentParser(description=f"{STAGE}: {TITLE}")
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    parser.add_argument("--paper1-cqr", type=Path, default=core.DEFAULT_PAPER1_CQR)
    parser.add_argument("--wapp-power-bridge", type=Path, default=core.DEFAULT_WAPP_BRIDGE)
    args = parser.parse_args()
    audit = {"stage": STAGE, "mode": args.mode, "stage_pass": False,
             "gates": {name: False for name in GATES.values()}, "gate_details": {}}
    manifest = protocol(args.mode)
    out = OUTPUT / args.mode
    try:
        check(not out.exists(), f"Immutable output directory exists: {out}")
        provenance(audit, manifest)
        manifest["static_audit"] = static_audit()
        assets = core.Paper2EnvAssets(paper1_cqr=args.paper1_cqr, wapp_power_bridge=args.wapp_power_bridge)
        asset_gates(assets, audit, manifest)
        check(assets.emulator is None and assets.perception_construction_count == 0, "True-State assets do not construct perception")
        summaries, pairs, trace = audit_episodes(assets, 10 if args.mode == "formal" else 2, audit)
        support_failure_probe(assets, audit)
        checkers = run_checkers(assets, audit)
        formal = args.mode == "formal"
        api = {"stage": STAGE, "mode": args.mode,
               "scientific_status": "GYM_POMDP_ENVIRONMENT_PASS_FROZEN" if formal else "SMOKE_ONLY_NO_ENV_FREEZE",
               "environment_contract_status": "PASS_FROZEN" if formal else "SMOKE_ONLY_NO_ENV_FREEZE",
               "module": ENV_FILE, "class": "Paper2CleaningEnv", "episode_spec": "fixed at construction; no scheduler",
               "observation_modes": {"True-State": ["L_true", "sin_DOY", "cos_DOY"], "Point": ["q50", "sin_DOY", "cos_DOY"],
                                     "UA": ["q50", "width", "sin_DOY", "cos_DOY"]},
               "observation_dtype": "float32", "actions": {"0": "WAIT", "1": "CLEAN"},
               "settlements": 365, "natural_transitions": 364, "terminated_last_step": True, "truncated": False,
               "terminal_observation": "zero float32 sentinel with observation-space shape",
               "reward": {"predecessor": "P2-1D-3", "lambda_c_main": assets.lambda_main, "eta": assets.eta,
                          "scale": assets.reward_scale, "formula": "-[E_clean_GHI * L_post + lambda_c_main * action]"},
               "perception": "frozen OnlineBlock10Emulator; shared day substream record; no fallback",
               "declaration": "NO ENVIRONMENT CONTRACT TUNING AFTER P2-1E-1" if formal else "SMOKE ONLY: NO ENVIRONMENT FREEZE",
               "downstream": ["non-RL baselines", "True-State PPO", "Point PPO", "UA-PPO"],
               "downstream_requirement": "Import experiments/paper2_gym_pomdp_env_v1.py; never copy environment logic into training scripts",
               "freeze_effective_only_with": "Same-directory final audit_summary.json: stage_pass=True, failed_gates=[], scientific_status=GYM_POMDP_ENVIRONMENT_PASS_FROZEN"}
        publish(out, manifest, api, summaries, pairs, trace, checkers, audit, assets)
    except Exception as exc:
        audit.update(stage_pass=False, scientific_status="FAIL_CLOSED_NO_ENV_FREEZE", execution_error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, core.PerceptionSupportError):
            audit["perception_support_context"] = exc.context
        audit["failed_gates"] = [name for name, passed in audit["gates"].items() if not passed]
        # Failure evidence goes to stdout; immutable partial directories have no
        # successful publication marker and must never be silently overwritten.
        print(encode(audit).decode("utf-8"))
        return 1
    print(encode({"stage": STAGE, "stage_pass": audit["stage_pass"], "scientific_status": audit["scientific_status"], "output": str(out)}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
