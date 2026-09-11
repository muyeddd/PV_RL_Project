#!/usr/bin/env python
"""P2-2A-2-v1: Held-Out Formal Non-RL Baseline Evaluation.

Explicit execution only, after human review and an unchanged Git checkpoint.
Smoke is implementation regression, never selection or a formal claim.
No environment construction, scientific import, or output write at import time.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys


STAGE = "P2-2A-2-v1"
TITLE = "Held-Out Formal Non-RL Baseline Evaluation"
PROVENANCE_CORRECTION = (
    "One P2-2A-1 selected_baselines.json expected SHA256 was corrected from a protocol "
    "transcription error after independent hashlib + frozen hash-index + certutil "
    "verification; frozen source artifact was not modified."
)
ANNUAL_REGRESSION_TOLERANCE_NOTE = (
    "E1 annual reward regression uses atol=1e-12, rtol=0 after an independent diagnostic "
    "confirmed exact daily trajectory and exact CRN identity. The observed exact-equality "
    "discrepancy was 2.842170943040401e-14 and arose from floating-point annual accumulation order."
)
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUTPUT = BASE / "p2_2a_2_heldout_nonrl_baseline_evaluation_v1"
PRE = BASE / "p2_2a_1_nonrl_baseline_selection_freeze_v1"
SELF = "experiments/run_paper2_stage2a2_heldout_nonrl_baseline_evaluation_v1.py"
PRE_SCRIPT = "experiments/run_paper2_stage2a1_nonrl_baseline_selection_freeze_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
POLICY_SOURCE = "experiments/run_paper2_stage1d2b_reward_behaviour_audit_v1.py"
CHECKPOINT = "43659869ac45b8a7f35c27608c80f4a75c5d479f"
SCRIPT_PINS = {
    PRE_SCRIPT: "3f633f87a152b5b815e18dc29134e01e45d955ef",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    POLICY_SOURCE: "f4400a94f5df4c46d975a3944182f084206ef37a",
}
PRE_HASHES = {
    "audit_summary.json": "12e9b36995693e35e7672a3adcc8f5585d2d5711ef2420d2adcd5938ca5f3dc5",
    "baseline_candidate_table.csv": "c34216772f9b9f8fdf32b3bf3db1df5861a644b7fc0a20384efc6ca1cee184f1",
    # Protocol transcription error independently verified by hashlib, frozen
    # hash-index and certutil; frozen source artifact was not modified.
    "selected_baselines.json": "5bf1c8aa3042010d947eac684114ef1e031b9c3726983c33bc760d1fa70c537d",
    "protocol_manifest.json": "6d9eb076891bb0eb1c19febcdca982bce71b2fb82d75324d68c53a917ee95352",
    "source_artifact_hashes.json": "5a2b0b15f7439c5f3c33ad5e2bf63e9bd4db02b7c42560daec20510614c5f995",
    "output_hashes.json": "b90fdb72e61b4fe52b731b588671dfb054e7b245840c2f77976a0d0034784061",
}
# Validation oracle only: the rollout population MUST come from frozen JSON.
EXPECTED_BASELINES = ("NEVER_CLEAN", "DAILY_CLEAN", "PERIODIC_14", "TRUE_THRESHOLD_0.05")
ROLES = dict(zip(EXPECTED_BASELINES, (
    "NO_MAINTENANCE_DIAGNOSTIC", "OVER_MAINTENANCE_DEGENERATE_CONTROL",
    "DEPLOYABLE_CALENDAR_BASELINE", "TRUE_STATE_INTERPRETABLE_BASELINE")))
REFERENCE = "TRUE_THRESHOLD_0.05"
COMPARISONS = ("PERIODIC_14", "NEVER_CLEAN", "DAILY_CLEAN")
DECLARATION = "NO BASELINE RESELECTION FROM HELD-OUT RESULTS"
SMOKE_STATUS = "SMOKE_ONLY_NO_FORMAL_CLAIM"
FORMAL_STATUS = "HELDOUT_NONRL_BASELINE_EVALUATION_PASS_FROZEN"
REGRESSION_PINS = {
    "p2_1d_2b_reward_behaviour_audit_v1/formal/trajectory_policy_base_metrics.csv":
        "baac9aec1108d387d0afaa94e083cc142ac13ea6a901ab94f70c6e554e2d5e47",
    "p2_1d_3_final_reward_contract_audit_v1/formal/daily_reward_trace.csv":
        "de0d78496a94c507e78080ac5c5748ed40b467b6e1079da0514814f1cfe81f01",
    "p2_1e_1_gym_pomdp_environment_audit_v1_1/formal/episode_regression_summary.csv":
        "95b8fd67c89bc2790a49e93536df80bceb8cbea0abb9c1fd0030e9541e3f39bb",
}
EPISODE_COLUMNS = ["year", "trajectory_id", "policy", "environment_root", "J_total", "J_soiling",
                   "J_cleaning", "N_clean", "episode_return", "step_count", "natural_transition_count",
                   "max_L_pre", "mean_L_pre", "perception_query_count", "indices_sha256", "full_bank_sha256"]
TRACE_COLUMNS = ["year", "trajectory_id", "date", "day_index", "policy", "L_pre", "action", "L_post",
                 "E_clean_GHI", "J_soiling_step", "J_cleaning_step", "cost_step", "reward", "L_next",
                 "transition_applied"]
GATES = dict(enumerate([
    "predecessor_provenance_exact", "frozen_baseline_set_declaration", "core_blob_exact",
    "reward_contract_exact", "frozen_policy_semantics", "mode_roots_isolated", "population_exact",
    "365_decisions", "364_transitions", "zero_perception_queries", "no_formal_perception_seeds",
    "paired_CRN_identity", "cost_decomposition", "return_identity", "finite_annual_metrics",
    "four_policies_per_key", "no_duplicate_episode", "formal_roots_exact_or_smoke_excluded",
    "no_baseline_search", "aggregate_counts", "CVaR_worst_five_percent", "paired_exact_keys",
    "representative_trace_complete", "development_formal_roles_separate", "no_perception_fields",
    "sources_unchanged", "HEAD_unchanged", "self_committed_unchanged", "exclusive_readback_complete",
    "status_declaration_exact",
], 1))


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def gate(audit, number, condition, detail):
    audit["gates"][f"G{number}"] = bool(condition)
    audit["gate_details"][f"G{number}"] = detail
    require(condition, f"G{number} {GATES[number]}: {detail}")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
                          check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def record_script(path, head, expected=None):
    blob = git("rev-parse", "--verify", f"{head}:{path}")
    require(blob == git("hash-object", f"--path={path}", path), f"Working tree differs: {path}")
    require(expected is None or blob == expected, f"Frozen blob differs: {path}")
    return {"path": path, "git_blob": blob, "sha256": sha((ROOT / path).read_bytes())}


def read_pinned(path, expected, sources):
    payload = path.read_bytes()
    require(sha(payload) == expected, f"Frozen SHA256 mismatch: {path}; expected={expected}, actual={sha(payload)}")
    sources[str(path)] = {"sha256": expected}
    return payload


def csv_rows(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "CSV header unique")
    rows = list(reader)
    require(all(set(row) == set(reader.fieldnames) and None not in row.values() for row in rows), "CSV schema")
    return rows


def mode_population(mode):
    # Sole scientific-root provider. No CLI root/seed/population overrides.
    if mode == "smoke":
        return {"YEAR1": 1020001, "YEAR2": 1030001}, tuple(range(3))
    require(mode == "formal", "Explicit smoke/formal mode required")
    return {"YEAR1": 1220001, "YEAR2": 1230001}, tuple(range(300))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    require(spec is not None and spec.loader is not None, f"Module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by the frozen core's dataclass.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def static_audit():
    """Read-only AST audit; never imports scientific code or runs a rollout."""
    tree = ast.parse((ROOT / SELF).read_text(encoding="utf-8"))
    forbidden = {"select_family", "select_main", "freeze_selection", "simulate", "rollout", "build_crn_indices",
                 "OnlineBlock10Emulator", "sample_one", "perception_record", "ensure_perception",
                 "default_rng", "PPO", "learn", "backward", "minimize", "eval", "exec", "__import__"}
    calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
             for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    require(not calls.intersection(forbidden), "Prohibited search/scientific implementation/training call")
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    require(set(imports) <= {"__future__", "argparse", "ast", "csv", "hashlib", "importlib.util", "io",
                             "json", "math", "pathlib", "subprocess", "sys", "numpy"}, "Import allowlist")
    population = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mode_population")
    branch = population.body[0]
    require(isinstance(branch, ast.If) and ast.unparse(branch.test) == "mode == 'smoke'", "Smoke early branch")
    constants = {n.value for n in ast.walk(branch) if isinstance(n, ast.Constant) and isinstance(n.value, int)}
    require(constants == {1020001, 1030001, 3}, "Smoke branch contains development roots only")
    require(isinstance(branch.body[0], ast.Return), "Smoke returns before formal root path")
    spec_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "EpisodeSpec"]
    require(len(spec_calls) == 1 and any(k.arg == "perception_seed" and isinstance(k.value, ast.Constant)
            and k.value.value is None for k in spec_calls[0].keywords), "Only seed=None EpisodeSpec")
    assignments = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "policies" for target in n.targets)]
    require(assignments == ["policies = tuple(selected['formal_nonrl_baselines'])"], "Frozen JSON is sole policy population source")
    return {"imports": imports, "prohibited_calls": [], "smoke_early_return": True,
            "policy_population_source": "load_inputs: frozen selected_baselines.json/formal_nonrl_baselines"}


def load_inputs(audit, protocol, sources):
    head = git("rev-parse", "HEAD")
    require(git("rev-parse", "--verify", f"{CHECKPOINT}^{{commit}}") == CHECKPOINT, "Checkpoint exists")
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    records = {p: record_script(p, head, blob) for p, blob in SCRIPT_PINS.items()}
    for path, blob in SCRIPT_PINS.items():
        require(git("rev-parse", f"{CHECKPOINT}:{path}") == blob, "Checkpoint tree exact")
    own = record_script(SELF, head)
    gate(audit, 28, True, own)
    payloads = {name: read_pinned(PRE / name, digest, sources) for name, digest in PRE_HASHES.items()}
    prior = json.loads(payloads["audit_summary.json"])
    require(prior.get("stage_pass") is True and prior.get("failed_gates") == []
            and prior.get("scientific_status") == "NONRL_BASELINE_SET_PASS_FROZEN", "Frozen predecessor PASS")
    index = json.loads(payloads["output_hashes.json"])["hashes"]
    require(set(index) == set(PRE_HASHES) - {"audit_summary.json", "output_hashes.json"}
            and all(index[n] == PRE_HASHES[n] for n in index), "Predecessor hash index exact")
    gate(audit, 1, True, {"checkpoint": CHECKPOINT, "ancestor": True, "artifact_hashes": PRE_HASHES})
    selected = json.loads(payloads["selected_baselines.json"])
    policies = tuple(selected["formal_nonrl_baselines"])
    gate(audit, 2, policies == EXPECTED_BASELINES and selected.get("declaration") ==
         "NO BASELINE RESELECTION AFTER P2-2A-1" and selected.get("baseline_roles") == ROLES, list(policies))
    prior_protocol = json.loads(payloads["protocol_manifest.json"])
    require(prior_protocol.get("selection_occurred_before_formal_roots") is True
            and prior_protocol["formal_environment"]["status"] == "RESERVED_NOT_ACCESSED_IN_P2_2A_1"
            and prior_protocol["formal_environment"]["first_access_stage"] == "P2-2A-2", "Selection did not access held-out roots")
    protocol.update(current_HEAD=head, current_script=own, script_provenance=records,
                    formal_nonrl_baselines=list(policies), baseline_roles=selected["baseline_roles"],
                    selection_stage_did_not_access_heldout_roots=True)
    gate(audit, 3, records[CORE]["git_blob"] == SCRIPT_PINS[CORE], records[CORE])
    core = load_module("_p2_2a2_frozen_core", CORE)
    helper = load_module("_p2_2a2_frozen_policy", POLICY_SOURCE)
    # Only helper.action is called. Its module-level code has no experiment side effects.
    require(helper.action.__code__.co_varnames[:3] == ("policy", "day_index", "L_pre"), "Pure policy interface")
    import numpy as np
    for day in range(365):
        for p, expected in (("NEVER_CLEAN", 0), ("DAILY_CLEAN", 1), ("PERIODIC_14", int((day + 1) % 14 == 0))):
            require(int(helper.action(p, day, np.float64(.05))) == expected, "Frozen action schedule")
    for latent, expected in ((np.nextafter(.05, -np.inf), 0), (.05, 1), (np.nextafter(.05, np.inf), 1)):
        require(int(helper.action(REFERENCE, 0, np.float64(latent))) == expected, "Inclusive threshold exact boundary")
    gate(audit, 5, True, {"policy_semantics_source_file": POLICY_SOURCE,
         "policy_semantics_source_git_blob": SCRIPT_PINS[POLICY_SOURCE], "function": "action(policy, day_index, L_pre)",
         "rules": {"NEVER_CLEAN": "0", "DAILY_CLEAN": "1", "PERIODIC_14": "(day_index + 1) % 14 == 0; first day_index=13",
                   REFERENCE: "pre-action float64 L_pre >= 0.05"}})
    protocol["policy_semantics"] = audit["gate_details"]["G5"]
    # Snapshot every direct frozen core helper/input before asset construction.
    for path, digest in core.ASSET_HASHES.values():
        read_pinned(path, digest, sources)
    for name, digest in core.REWARD_HASHES.items():
        read_pinned(core.REWARD_DIRECTORY / name, digest, sources)
    for path, blob in [*core.HELPERS.values(), (core.REWARD_SCRIPT, core.REWARD_BLOB)]:
        records[path] = record_script(path, head, blob)
    assets = core.Paper2EnvAssets()
    gate(audit, 4, assets.lambda_main == selected["main_lambda_c"] == core.LAMBDA_MAIN == 0.19925428989934618
         and assets.eta == selected["eta"] == core.ETA == .95
         and assets.reward_scale == selected["reward_scale"] == core.REWARD_SCALE == 1.0,
         assets.reward_contract)
    protocol["reward_contract"] = assets.reward_contract
    protocol["environment_asset_provenance"] = assets.provenance
    return policies, core, helper.action, assets


def regression_inputs(mode, sources):
    if mode != "smoke":
        return None
    frames = {name: csv_rows(read_pinned(BASE / name, digest, sources)) for name, digest in REGRESSION_PINS.items()}
    base, daily, episodes = frames.values()
    annual = {(r["year"], int(r["trajectory_id"]), r["policy"]): r for r in base}
    trace = {(r["year"], int(r["trajectory_id"]), r["policy"], int(r["day_index"])): r for r in daily}
    env = {(r["year"], int(r["trajectory_id"]), r["policy"]): r for r in episodes
           if r["observation_mode"] == "True-State" and r["gym_seed"] == "1"}
    return annual, trace, env


def close(actual, expected):
    return math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=0)


def run_episode(core, assets, action_rule, policy, year, root, tid, regression):
    import numpy as np
    spec = core.EpisodeSpec(year=year, environment_root=root, trajectory_id=tid,
                            population_size=300, perception_seed=None)
    env = core.Paper2CleaningEnv(assets, spec, observation_mode="True-State")
    rows = []
    key = (year, tid, policy)
    try:
        obs, _ = env.reset()
        initial = env._audit_snapshot()
        for day in range(365):
            before = env._audit_snapshot()
            require(before["day_index"] == day and obs.shape == (3,) and np.isfinite(obs).all(), "True-State day/observation")
            require(obs[0] == np.float32(before["L_pre"]), "Current state agrees with observation")
            # Frozen D2B and E1 use float64 current L_pre. Only this present-state
            # scalar and calendar day reach the policy, never CRN or future data.
            action = int(action_rule(policy, day, np.float64(before["L_pre"])))
            obs, reward, terminated, truncated, info = env.step(action)
            after = env._audit_snapshot()
            last = after["last_reward_components"]
            require(not truncated and terminated == (day == 364), "Exact 365-decision terminal semantics")
            require(last["day_index"] == day and last["action"] == action and last["L_pre"] == before["L_pre"], "Pre-action state")
            require(last["L_post"] == ((1 - assets.eta) * last["L_pre"] if action else last["L_pre"]), "Same-day CLEAN before settlement")
            require(last["soiling_cost"] == last["E_clean_GHI"] * last["L_post"]
                    and last["cleaning_cost"] == assets.lambda_main * action
                    and last["total_cost"] == last["soiling_cost"] + last["cleaning_cost"]
                    and reward == last["final_reward"] == -last["total_cost"], "Frozen reward mechanics")
            require(last["transition_executed"] == (day < 364) and (last["L_next"] is None) == (day == 364), "364 transitions")
            require(after["perception_query_count"] == 0 and after["perception_seed"] is None, "No perception queries/seeds")
            require(all(after[n] == initial[n] for n in ("indices_sha256", "full_bank_sha256", "environment_root")), "CRN stable within episode")
            if regression is not None and policy != "PERIODIC_14":
                frozen = regression[1][(*key, day)]
                for column in ("L_pre", "L_post", "E_clean_GHI", "soiling_cost", "cleaning_cost", "total_cost", "raw_reward", "final_reward"):
                    require(last[column] == float(frozen[column]), f"Exact frozen daily regression {key}/{day}/{column}")
                require(action == int(frozen["action"]) and last["date"] == frozen["date"]
                        and str(last["transition_executed"]) == frozen["transition_executed"], "Frozen action/calendar/transition exact")
                require(last["L_next"] == (float(frozen["L_next"]) if day < 364 else None), "Exact frozen next state")
            rows.append({"year": year, "trajectory_id": tid, "date": last["date"], "day_index": day, "policy": policy,
                         **{n: last[n] for n in ("L_pre", "action", "L_post", "E_clean_GHI", "L_next")},
                         "J_soiling_step": last["soiling_cost"], "J_cleaning_step": last["cleaning_cost"],
                         "cost_step": last["total_cost"], "reward": reward, "transition_applied": last["transition_executed"]})
        total = math.fsum(r["cost_step"] for r in rows)
        soiling = math.fsum(r["J_soiling_step"] for r in rows)
        cleaning = math.fsum(r["J_cleaning_step"] for r in rows)
        metrics = {"year": year, "trajectory_id": tid, "policy": policy, "environment_root": root,
                   "J_total": total, "J_soiling": soiling, "J_cleaning": cleaning,
                   "N_clean": sum(r["action"] for r in rows), "episode_return": after["episode_return"],
                   "step_count": len(rows), "natural_transition_count": after["transition_count"],
                   "max_L_pre": max(r["L_pre"] for r in rows), "mean_L_pre": math.fsum(r["L_pre"] for r in rows) / 365,
                   "perception_query_count": after["perception_query_count"],
                   "indices_sha256": after["indices_sha256"], "full_bank_sha256": after["full_bank_sha256"]}
        require(metrics["N_clean"] == after["clean_count"] == info["clean_count"], "Clean count exact")
        require(close(total, soiling + cleaning) and close(after["episode_return"], -total)
                and close(cleaning, assets.lambda_main * metrics["N_clean"]), "Annual reward identities atol=1e-12 rtol=0")
        require(all(math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))), "Annual metrics finite")
        if regression is not None:
            frozen = regression[0][key]
            for column in ("J_soiling", "mean_L_pre", "max_L_pre"):
                require(close(metrics[column], frozen[column]), f"D2B annual regression {key}/{column}")
            require(metrics["N_clean"] == int(frozen["N_clean"]), "D2B cleaning exact incl PERIODIC_14")
            if policy != "PERIODIC_14":
                e1 = regression[2][key]
                require(close(after["episode_return"], float(e1["annual_reward"]))
                        and all(metrics[n] == e1[n] for n in ("indices_sha256", "full_bank_sha256")), "E1 annual reward tolerance / exact CRN regression")
        return metrics, rows
    finally:
        env.close()


def aggregate(episodes, policies, n, audit):
    import numpy as np
    rows = []
    for scope in ("YEAR1", "YEAR2", "JOINT"):
        for policy in policies:
            block = [r for r in episodes if r["policy"] == policy and (scope == "JOINT" or r["year"] == scope)]
            count = len(block)
            require(count == n * (2 if scope == "JOINT" else 1), "Aggregate count")
            cost = np.array([r["J_total"] for r in block])
            clean = np.array([r["N_clean"] for r in block])
            worst_n = math.ceil(.05 * count)
            cvar = float(np.sort(cost)[::-1][:worst_n].mean())
            require(close(cvar, math.fsum(sorted(cost.tolist(), reverse=True)[:worst_n]) / worst_n), "CVaR exact worst-count definition")
            rows.append({"scope": scope, "policy": policy, "N_episodes": count,
                         "mean_J_total": float(cost.mean()), "std_J_total": float(cost.std(ddof=1)),
                         "median_J_total": float(np.median(cost)),
                         **{f"P{q:02d}_J_total": float(np.percentile(cost, q, method="linear")) for q in (5, 25, 75, 95)},
                         "CVaR95_J_total": cvar, "CVaR95_episode_count": worst_n,
                         "mean_J_soiling": math.fsum(r["J_soiling"] for r in block) / count,
                         "mean_J_cleaning": math.fsum(r["J_cleaning"] for r in block) / count,
                         "mean_N_clean": float(clean.mean()), "median_N_clean": float(np.median(clean)),
                         "P05_N_clean": float(np.percentile(clean, 5, method="linear")),
                         "P95_N_clean": float(np.percentile(clean, 95, method="linear"))})
    gate(audit, 20, len(rows) == 12, "Exact per-policy YEAR1/YEAR2/JOINT counts")
    gate(audit, 21, True, "UPPER_COST_TAIL; descending order; worst ceil(0.05*N); no quantile-tie filtering")
    return rows


def paired(episodes, n, audit):
    import numpy as np
    rows = []
    for scope in ("YEAR1", "YEAR2", "JOINT"):
        maps = {p: {(r["year"], r["trajectory_id"]): r["J_total"] for r in episodes
                    if r["policy"] == p and (scope == "JOINT" or r["year"] == scope)} for p in (*COMPARISONS, REFERENCE)}
        ref = maps[REFERENCE]
        for policy in COMPARISONS:
            require(maps[policy].keys() == ref.keys() and len(ref) == n * (2 if scope == "JOINT" else 1), "Paired exact keys")
            delta = np.array([maps[policy][key] - ref[key] for key in sorted(ref)])
            rows.append({"scope": scope, "policy": policy, "reference": REFERENCE, "N_pairs": len(delta),
                         "paired_mean_difference": float(delta.mean()), "paired_median_difference": float(np.median(delta)),
                         "win_fraction_threshold": float(np.mean(delta > 0)), "loss_fraction_threshold": float(np.mean(delta < 0)),
                         "tie_fraction": float(np.mean(delta == 0))})
    gate(audit, 22, len(rows) == 9, "Delta_J = policy - threshold; >0 threshold wins; exact-zero ties")
    return rows


def evaluate(mode, policies, core, action_rule, assets, regression, audit, protocol):
    roots, tids = mode_population(mode)
    require(len(tids) == (3 if mode == "smoke" else 300), "Mode-specific trajectory count")
    episodes, traces = [], []
    for year, root in roots.items():
        for tid in tids:
            crn = []
            for policy in policies:
                metric, rows = run_episode(core, assets, action_rule, policy, year, root, tid, regression)
                episodes.append(metric)
                crn.append((root, tid, metric["indices_sha256"], metric["full_bank_sha256"]))
                if tid == 0:
                    traces.extend(rows)
            require(len(set(crn)) == 1, "Four policies share exact scientific CRN identity")
    keys = [(r["year"], r["trajectory_id"], r["policy"]) for r in episodes]
    expected = {(year, tid, p) for year in roots for tid in tids for p in policies}
    gate(audit, 6, all(r["environment_root"] == roots[r["year"]] and r["trajectory_id"] in tids for r in episodes),
         {"mode": mode, "roots": roots, "trajectory_ids": list(tids), "actual_episode_roots_checked": True})
    gate(audit, 18, set(assets._banks) == {(year, root, 300) for year, root in roots.items()}
         and assets.bank_generation_count == 2, "Exactly two mode-specific scientific banks; no other roots generated")
    gate(audit, 7, len(episodes) == 2 * len(tids) * 4 and set(keys) == expected,
         {"episodes": len(episodes), "decision_steps": sum(r["step_count"] for r in episodes)})
    for number, column, value in ((8, "step_count", 365), (9, "natural_transition_count", 364), (10, "perception_query_count", 0)):
        gate(audit, number, all(r[column] == value for r in episodes), {column: value})
    gate(audit, 11, assets.perception_construction_count == assets.perception_sample_count == 0
         and assets.emulator is None and not assets._perception_cache, "No perception seeds supplied; zero construction/samples")
    gate(audit, 12, True, "Exact (root, trajectory, indices_sha256, full_bank_sha256) compared across policies")
    gate(audit, 13, all(close(r["J_total"], r["J_soiling"] + r["J_cleaning"]) for r in episodes), "atol=1e-12, rtol=0")
    gate(audit, 14, all(close(r["episode_return"], -r["J_total"]) for r in episodes), "atol=1e-12, rtol=0")
    gate(audit, 15, all(math.isfinite(v) for r in episodes for v in r.values() if isinstance(v, (int, float))), "All annual numeric metrics finite")
    gate(audit, 16, set(keys) == expected, "Four complete policies per year/trajectory")
    gate(audit, 17, len(keys) == len(set(keys)), "Unique annual episode keys")
    trace_keys = {(r["year"], r["trajectory_id"], r["policy"], r["day_index"]) for r in traces}
    gate(audit, 23, len(traces) == len(trace_keys) == 2920 and trace_keys ==
         {(y, 0, p, d) for y in roots for p in policies for d in range(365)}, "Both years trajectory 0; four policies; 365 days")
    protocol.update(environment_roots=roots, trajectory_ids=list(tids), population_size=300,
                    annual_episodes=len(episodes), decision_steps=len(episodes) * 365,
                    HELD_OUT_ROOTS_FIRST_ACCESSED_IN_P2_2A_2=(mode == "formal"),
                    perception_query_count=0, perception_seed=None,
                    perception_seed_role="PERCEPTION_SEED_UNUSED_BY_TRUE_STATE")
    return episodes, traces


def publish(out, tables, audit, protocol, sources):
    payloads = {"protocol_manifest.json": encode(protocol), "source_artifact_hashes.json": encode(sources)}
    for name, (columns, rows) in tables.items():
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        payloads[name] = stream.getvalue().encode("utf-8-sig")
    index = {"hashes": {n: sha(p) for n, p in payloads.items()}, "excluded": ["output_hashes.json", "audit_summary.json"]}
    payloads["output_hashes.json"] = encode(index)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (out / name).open("xb") as file:
            file.write(data)
    for name, data in payloads.items():
        actual = (out / name).read_bytes()
        require(actual == data and sha(actual) == sha(data), f"Output readback SHA256: {name}")
        if name in tables:
            columns, rows = tables[name]
            require(csv_rows(actual) == [{k: "" if r[k] is None else str(r[k]) for k in columns} for r in rows], "CSV schema/content readback")
        else:
            require(json.loads(actual) == json.loads(data), "JSON schema/content readback")
        if name in index["hashes"]:
            require(sha(actual) == index["hashes"][name], "Hash index coverage")
    gate(audit, 26, all(sha(Path(p).read_bytes()) == rec["sha256"] for p, rec in sources.items())
         and all(record_script(p, protocol["current_HEAD"], rec["git_blob"]) == rec
                 for p, rec in protocol["script_provenance"].items()), "All frozen sources unchanged during run")
    gate(audit, 27, git("rev-parse", "HEAD") == protocol["current_HEAD"], "HEAD unchanged")
    gate(audit, 28, record_script(SELF, protocol["current_HEAD"]) == protocol["current_script"], "Committed script unchanged throughout run")
    gate(audit, 29, {p.name for p in out.iterdir()} == set(payloads),
         {"exclusive_write": True, "readback_schema_hashes": {n: sha(p) for n, p in payloads.items()},
          "audit_last": True, "hash_index_excludes_self_and_final_audit": True})
    require(set(audit["gates"]) == {f"G{i}" for i in GATES} and all(audit["gates"].values()), "All 30 gates required")
    audit.update(stage_pass=True, failed_gates=[], scientific_status=protocol["scientific_status"], declaration=DECLARATION)
    data = encode(audit)
    with (out / "audit_summary.json").open("xb") as file:
        file.write(data)
    require((out / "audit_summary.json").read_bytes() == data
            and {p.name for p in out.iterdir()} == set(payloads) | {"audit_summary.json"}, "Final audit readback/completeness")


def main():
    parser = argparse.ArgumentParser(description=f"{STAGE}: {TITLE}; no baseline reselection")
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    mode = parser.parse_args().mode
    audit = {"stage": STAGE, "provenance_correction": PROVENANCE_CORRECTION, "stage_pass": False, "gates": {f"G{i}": False for i in GATES}, "gate_details": {}}
    audit["annual_regression_tolerance_note"] = ANNUAL_REGRESSION_TOLERANCE_NOTE
    sources = {}
    protocol = {"stage": STAGE, "title": TITLE, "mode": mode, "required_gates": GATES,
                "provenance_correction": PROVENANCE_CORRECTION,
                "annual_regression_tolerance_note": ANNUAL_REGRESSION_TOLERANCE_NOTE,
                "scientific_status": SMOKE_STATUS if mode == "smoke" else FORMAL_STATUS, "declaration": DECLARATION,
                "observation_mode": "True-State", "observation": "[L_true, sin_DOY, cos_DOY]",
                "policy_state_precision": "Frozen D2B float64 current L_pre via private audit accessor; no future/CRN inputs to action",
                "settlement_order_source": CORE + ":Paper2CleaningEnv.step; CLEAN before same-day reward; terminal day has no transition",
                "projection_counts": "NOT_EXPOSED_BY_FROZEN_ENVIRONMENT; omitted, not inferred from boundary hits",
                "CVaR_direction": "UPPER_COST_TAIL", "CVaR95_definition": "Mean worst ceil(0.05*N) costs sorted descending",
                "quantile_method": "linear", "std_ddof": 1, "paired_delta": "J_policy - J_TRUE_THRESHOLD_0.05",
                "paired_comparisons": list(COMPARISONS), "primary_reference": REFERENCE,
                "paired_sign": "Delta_J > 0 means threshold is better (lower cost)",
                "tie_rule": "exact Delta_J == 0", "bootstrap": "NOT_IMPLEMENTED_IN_V1; estimation may be added in later statistics stage",
                "ranking_role": "DESCRIPTIVE_ONLY_NO_RESELECTION", "development_formal_diagnostic": "NOT_PERFORMED_IN_V1",
                "interpretation": "held-out counterfactual policy evaluation under the frozen field-grounded empirical environment" if mode == "formal"
                    else "Implementation regression only; no selection, formal claim, or policy comparison conclusion",
                "NEVER_CLEAN_context": "Long-term no-maintenance / extrapolation-stress reference",
                "PPO_stop_gate": "After sufficient training, if True-State PPO is clearly worse than TRUE_THRESHOLD_0.05, stop Point-PPO / UA-PPO; inspect PPO implementation, training budget, hyperparameters, exploration, normalization and convergence. Do not interpret this as uncertainty lacks value.",
                "smoke_regression_scope": "D2B annual metrics all four policies; D3 exact daily traces and E1 exact annual return/CRN for NEVER_CLEAN, DAILY_CLEAN, TRUE_THRESHOLD_0.05. PERIODIC_14 has no frozen daily trace: annual metrics plus exact 365-day schedule and per-step reward mechanics."}
    try:
        out = OUTPUT / mode
        require(not out.exists(), f"Immutable output directory already exists: {out}")
        gate(audit, 19, True, static_audit())
        policies, core, action_rule, assets = load_inputs(audit, protocol, sources)
        regression = regression_inputs(mode, sources)
        episodes, traces = evaluate(mode, policies, core, action_rule, assets, regression, audit, protocol)
        summaries = aggregate(episodes, policies, len(mode_population(mode)[1]), audit)
        comparisons = paired(episodes, len(mode_population(mode)[1]), audit)
        gate(audit, 24, (regression is not None) == (mode == "smoke"), "Smoke only development regression; formal only held-out evaluation; no reselection")
        gate(audit, 25, not {"q50", "width", "perception_seed"}.intersection(EPISODE_COLUMNS + TRACE_COLUMNS),
             "No perception-dependent metric/trace fields; protocol records seed=None and zero queries")
        gate(audit, 30, protocol["scientific_status"] == (SMOKE_STATUS if mode == "smoke" else FORMAL_STATUS)
             and protocol["declaration"] == DECLARATION, "Exact mode-specific status/declaration")
        tables = {"policy_episode_metrics.csv": (EPISODE_COLUMNS, episodes),
                  "policy_aggregate_summary.csv": (list(summaries[0]), summaries),
                  "paired_comparison_summary.csv": (list(comparisons[0]), comparisons),
                  "representative_step_trace.csv": (TRACE_COLUMNS, traces)}
        publish(out, tables, audit, protocol, sources)
    except Exception as exc:
        audit.update(stage_pass=False, scientific_status="FAIL_CLOSED_NO_FORMAL_CLAIM",
                     execution_error=f"{type(exc).__name__}: {exc}",
                     failed_gates=[k for k, value in audit["gates"].items() if not value])
        print(encode(audit).decode("utf-8"))
        return 1
    print(encode({"stage": STAGE, "scientific_status": audit["scientific_status"], "output": str(out)}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
