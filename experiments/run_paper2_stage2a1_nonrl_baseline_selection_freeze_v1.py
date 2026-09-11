#!/usr/bin/env python
"""P2-2A-1-v1: Pre-RL Non-RL Baseline Selection Freeze.

Explicit execution only; selection from frozen development summaries, no rollout.
Only standard-library imports. No environment/perception/learner is imported.
The script must be committed unchanged before an audit can publish final PASS.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from importlib import metadata
import io
import json
import math
from pathlib import Path
import platform
import subprocess


STAGE = "P2-2A-1-v1"
TITLE = "Pre-RL Non-RL Baseline Selection Freeze"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
SELF = "experiments/run_paper2_stage2a1_nonrl_baseline_selection_freeze_v1.py"
OUTPUT = BASE / "p2_2a_1_nonrl_baseline_selection_freeze_v1"
D2B = BASE / "p2_1d_2b_reward_behaviour_audit_v1/formal"
E1 = BASE / "p2_1e_1_gym_pomdp_environment_audit_v1_1/formal"
CHECKPOINT = "8300ee867bd34985e9c03e8e0f39aa1d9116aed2"
D2B_CHECKPOINT = "0abece7317e8b10c9855803905de9cc7e33ee4c4"
CORE_FILE = "experiments/paper2_gym_pomdp_env_v1.py"
E1_FILE = "experiments/run_paper2_stage1e1_gym_pomdp_environment_audit_v1_1.py"
SCRIPT_PINS = {
    CORE_FILE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    E1_FILE: "6f66a3f8786b48746e7084457722bd7e490e468b",
    "experiments/run_paper2_stage1e1_gym_pomdp_environment_audit_v1.py": "4db745164fc8a6a0aa9d5b17fcd6eb8e5214bce0",
    "experiments/run_paper2_stage1d2b_reward_behaviour_audit_v1.py": "f4400a94f5df4c46d975a3944182f084206ef37a",
}
D2B_HASHES = {
    "lambda_policy_summary.csv": "ce5a43f03495ec140f0e83e56ec872c713de20cf26aebcaa5b236a73e39f4300",
    "main_lambda_manifest.json": "12ce7a5eac2ef1596a4ad3409b7a446fde71ebfe7d58860a535526503b8226db",
    "lambda_classification_summary.csv": "d9ceca392f4054e8a3170241d36d1ccb9e7f69839b259c9aa1ca78f9c10c1841",
    "protocol_manifest.json": "6f0c8f573fc29149f5b0d04615e6ebb66991c39724771945af1eda76e99a8581",
    "audit_summary.json": "99aba74d388b11d02a29c85f775b94f523e38d4d3372f864ec57c11e9e4aa0ac",
}
E1_HASHES = {
    "audit_summary.json": "e1e65e746a965aa115f17c8a6a170c62061dd9ff0d56e8f1a64f4a16aa67dc88",
    "protocol_manifest.json": "4e0f1ee5e9c4fcc1387e10dcb223a2b57db920ace994c9b15cba8e1aaffda646",
    "environment_api_summary.json": "0890037d31d82fd446aefea33c5259daf79d7f9ed792fcaca97a1ebf9d64cb76",
    "checker_summary.json": "2a6c5dc30743d095dc5ecfc9fd398e51be74b236e8bf2a2a06f2d26c0924b34d",
    "episode_regression_summary.csv": "95b8fd67c89bc2790a49e93536df80bceb8cbea0abb9c1fd0030e9541e3f39bb",
    "observation_pairing_summary.csv": "d083a5bb33e4f3880ec9a3b98efca3d9d0c9828c4332b6520c1fa1578bbe83bb",
    "representative_step_trace.csv": "0e70b004b249732bb716b7e9f7268328ac17cede8ea2b1474b4abdd6a567ff39",
    "output_hashes.json": "cbad9c06de056fa3d5874095ac296b31712a42109fb5a743c1ac6d9a65fb16af",
}
HASH_CORRECTION = "Two P2-1E expected SHA256 values corrected from protocol transcription errors; frozen source artifacts were not modified."
EXTREMES = ("NEVER_CLEAN", "DAILY_CLEAN")
PERIODIC = ("PERIODIC_7", "PERIODIC_14", "PERIODIC_30", "PERIODIC_60", "PERIODIC_90", "PERIODIC_180")
THRESHOLDS = ("TRUE_THRESHOLD_0.05", "TRUE_THRESHOLD_0.08", "TRUE_THRESHOLD_0.10", "TRUE_THRESHOLD_0.12", "TRUE_THRESHOLD_0.15")
CANDIDATES = EXTREMES + PERIODIC + THRESHOLDS
SCOPES = ("YEAR1", "YEAR2", "JOINT")
EXPECTED_PERIODIC = "PERIODIC_14"  # Regression only, never passed to selector.
EXPECTED_THRESHOLD = "TRUE_THRESHOLD_0.05"  # Regression only.
EXPECTED_LAMBDA = 0.19925428989934618
ETA = 0.95
REWARD_SCALE = 1.0
SELECTION_METRIC = "JOINT mean_J_total"
TIE_RULE = "EXACT_EQUAL_COST; FIRST_IN_FROZEN_CANDIDATE_ORDER"
# Metadata only: never passed to a scientific function or used to construct RNGs.
RESERVED_FORMAL_ENVIRONMENT_ROOTS = {"YEAR1": 1220001, "YEAR2": 1230001}
RESERVED_FORMAL_PERCEPTION_SEEDS = (20260907, 20260908, 20260909, 20260910, 20260911)
EXPECTED_DEPENDENCIES = {"Python": "3.11.15", "gymnasium": "1.2.3", "stable_baselines3": "2.7.1",
                         "torch": "2.10.0+cu130", "numpy": "2.3.5", "pandas": "2.3.3"}
COLUMNS = ["policy", "family", "candidate_order", "YEAR1_mean_J_total", "YEAR2_mean_J_total",
           "JOINT_mean_J_total", "selected_within_family", "formal_baseline_retained", "role",
           "selection_metric", "selection_source_stage"]
GATES = dict(enumerate([
    "P2_1D_2B_frozen_provenance_exact", "main_lambda_exact", "candidate_policy_population_exact",
    "periodic_family_exact", "threshold_family_exact", "scope_costs_finite_uniquely_mapped",
    "periodic_mechanical_JOINT_selection", "threshold_mechanical_JOINT_selection",
    "expected_periodic_regression", "expected_threshold_regression", "extreme_controls_retained",
    "historical_cleaning_reference_only", "P2_1E_formal_PASS_FROZEN_hashes_exact",
    "core_environment_historical_blob_exact", "formal_environment_roots_reserved_only",
    "formal_perception_seeds_reserved_only", "no_rollout_no_CRN_no_environment_instantiation",
    "no_PPO_learner_optimizer_training", "JOINT_cost_exclusive_selection", "deterministic_exact_ties",
    "formal_baseline_set_complete", "reward_contract_unchanged", "output_completeness",
    "current_script_committed_unchanged",
], start=1))


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def gate(audit, number, condition, detail):
    audit["gates"][GATES[number]] = bool(condition)
    audit["gate_details"][f"G{number}"] = detail
    check(condition, f"G{number} {GATES[number]}: {detail}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def script_record(path, head, expected=None, checkpoint=None):
    actual = git("rev-parse", "--verify", f"{head}:{path}")
    working = git("hash-object", f"--path={path}", path)
    check(actual == working and (expected is None or actual == expected), f"Committed unchanged script: {path}")
    frozen = git("rev-parse", "--verify", f"{checkpoint}:{path}") if checkpoint else None
    check(checkpoint is None or frozen == actual, f"Historical checkpoint blob: {path}")
    return {"path": path, "HEAD_blob": actual, "working_tree_blob": working,
            "checkpoint_blob": frozen, "expected_blob": expected, "sha256": sha((ROOT / path).read_bytes())}


def ancestry(checkpoint, head):
    check(git("rev-parse", "--verify", f"{checkpoint}^{{commit}}") == checkpoint, "Checkpoint exists")
    git("merge-base", "--is-ancestor", checkpoint, head)
    return {"checkpoint": checkpoint, "HEAD": head, "exists": True, "ancestor": True}


def parse_csv(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    columns = reader.fieldnames
    check(columns and len(set(columns)) == len(columns), "Unique CSV schema required")
    rows = list(reader)
    check(all(set(row) == set(columns) and all(value is not None for value in row.values()) for row in rows), "Malformed CSV row")
    return columns, rows


def read_sources(directory, hashes, provenance):
    result = {}
    for name, expected in hashes.items():
        path = directory / name
        check(path.is_file(), f"Missing frozen source: {path}")
        payload = path.read_bytes()
        actual = sha(payload)
        check(actual == expected, f"Frozen SHA256 mismatch: {path}")
        if name.endswith(".csv"):
            schema, value = parse_csv(payload)
            count = len(value)
        else:
            value = json.loads(payload)
            check(isinstance(value, dict), f"JSON object required: {path}")
            schema, count = list(value), 1
        provenance[str(path)] = {"path": str(path), "sha256": actual, "expected_sha256": expected,
                                 "schema": schema, "row_count": count, "verified": True}
        result[name] = value
    return result


def load_inputs(audit, protocol):
    head = git("rev-parse", "HEAD")
    protocol["ancestry"] = [ancestry(CHECKPOINT, head), ancestry(D2B_CHECKPOINT, head)]
    own = script_record(SELF, head)
    gate(audit, 24, own["HEAD_blob"] == own["working_tree_blob"], own)
    records = {path: script_record(path, head, blob, CHECKPOINT) for path, blob in SCRIPT_PINS.items()}
    protocol.update(current_HEAD=head, current_script=own, script_provenance=records)
    sources = {}
    d2b = read_sources(D2B, D2B_HASHES, sources)
    e1 = read_sources(E1, E1_HASHES, sources)
    protocol["source_artifacts"] = sources
    prior, manifest = d2b["audit_summary.json"], d2b["main_lambda_manifest.json"]
    frozen_protocol = d2b["protocol_manifest.json"]
    gate(audit, 1, prior.get("stage_pass") is True and prior.get("failed_gates") == []
         and prior.get("scientific_status") == "ROBUST_INTERIOR_MAIN_LAMBDA_SELECTED"
         and manifest.get("scientific_status") == "ROBUST_INTERIOR_MAIN_LAMBDA_SELECTED"
         and frozen_protocol.get("environment_population") == "DEVELOPMENT_ONLY"
         and frozen_protocol.get("environment_roots") == {"YEAR1": 1020001, "YEAR2": 1030001}
         and frozen_protocol.get("formal_environment_roots_accessed") is False,
         "Five pinned P2-1D-2B artifacts; pinned script and checkpoint ancestry; development population")
    gate(audit, 2, manifest.get("main_multiplier") == 2 and manifest.get("main_lambda_c") == EXPECTED_LAMBDA
         and manifest.get("selection_eligibility") == "support_validated_robust_interior", "Frozen main lambda validation only")
    env_audit, api = e1["audit_summary.json"], e1["environment_api_summary.json"]
    env_protocol = e1["protocol_manifest.json"]
    index = e1["output_hashes.json"]
    check(set(index["hashes"]) == set(E1_HASHES) - {"audit_summary.json", "output_hashes.json"}, "Frozen environment hash-index coverage")
    check(all(E1_HASHES[name] == digest for name, digest in index["hashes"].items()), "Environment hash index exact")
    gate(audit, 13, len(E1_HASHES) == 8 and env_audit.get("stage_pass") is True
         and env_audit.get("failed_gates") == [] and env_audit.get("scientific_status") == "GYM_POMDP_ENVIRONMENT_PASS_FROZEN"
         and api.get("environment_contract_status") == "PASS_FROZEN"
         and api.get("scientific_status") == "GYM_POMDP_ENVIRONMENT_PASS_FROZEN"
         and env_protocol.get("current_HEAD") == CHECKPOINT,
         {"artifacts": E1_HASHES, "hash_correction": HASH_CORRECTION})
    gate(audit, 14, records[CORE_FILE]["HEAD_blob"] == SCRIPT_PINS[CORE_FILE], records[CORE_FILE])
    reward = api["reward"]
    gate(audit, 22, reward.get("lambda_c_main") == manifest["main_lambda_c"] == EXPECTED_LAMBDA
         and reward.get("eta") == frozen_protocol.get("ETA") == ETA
         and reward.get("scale") == REWARD_SCALE
         and api.get("declaration") == "NO ENVIRONMENT CONTRACT TUNING AFTER P2-1E-1",
         {"reward": reward, "reward_recalculated": False})
    protocol["reward_contract"] = dict(reward)
    historical = manifest["historical_clean_reference"]
    gate(audit, 12, historical.get("role") == "BEHAVIOURAL_PLAUSIBILITY_REFERENCE_ONLY"
         and historical.get("used_for_selection") is False and historical.get("observed_count") == 26
         and historical.get("years") == 2 and frozen_protocol.get("historical_cleaning_used_for_selection") is False,
         {"original_frozen_fields": historical, "role_here": "BEHAVIOURAL_REFERENCE_ONLY",
          "not_a_policy_target_or_reproduction_claim": True})
    protocol["historical_cleaning"] = {"source_fields": historical, "role": "BEHAVIOURAL_REFERENCE_ONLY", "used_for_selection": False}
    # Document pre-existing support evidence; never expose it to the selector.
    protocol["prior_support_reference"] = {"evidence": env_protocol["support_reference"], "used_for_selection": False}
    protocol["v1_failure_diagnostic"] = {"evidence": env_protocol["known_v1_natural_support_diagnostic"], "role": "DIAGNOSTIC_ONLY", "used_for_selection": False}
    actual_versions = {"Python": platform.python_version()}
    for name in EXPECTED_DEPENDENCIES:
        if name != "Python":
            actual_versions[name] = metadata.version(name)  # Distribution metadata, no package import.
    protocol["dependency_provenance"] = {"expected": EXPECTED_DEPENDENCIES, "actual": actual_versions,
                                         "matches_expected": actual_versions == EXPECTED_DEPENDENCIES,
                                         "role": "RECORDED_ONLY; selection uses standard library"}
    return d2b, manifest


def candidate_costs(d2b, main_multiplier, main_lambda, audit):
    # Non-main multiplier performance remains unconverted CSV strings.
    rows = [row for row in d2b["lambda_policy_summary.csv"] if float(row["multiplier"]) == main_multiplier]
    expected = {(scope, policy) for scope in SCOPES for policy in CANDIDATES}
    keys = [(row["scope"], row["policy"]) for row in rows]
    gate(audit, 3, len(rows) == 39 and len(set(keys)) == len(keys) and set(keys) == expected
         and d2b["protocol_manifest.json"]["active_policies"] == list(CANDIDATES), "13 policies x three scopes; main multiplier only")
    gate(audit, 4, {row["policy"] for row in rows if row["policy"].startswith("PERIODIC_")} == set(PERIODIC), list(PERIODIC))
    gate(audit, 5, {row["policy"] for row in rows if row["policy"].startswith("TRUE_THRESHOLD_")} == set(THRESHOLDS), list(THRESHOLDS))
    costs = {}
    for row in rows:
        check(float(row["lambda_c"]) == main_lambda, "Main-row lambda exact")
        value = float(row["mean_J_total"])
        check(math.isfinite(value) and value >= 0, "Finite nonnegative frozen mean_J_total")
        costs[(row["scope"], row["policy"])] = value
    gate(audit, 6, set(costs) == expected, "YEAR1/YEAR2/JOINT uniquely mapped; no reward recalculation")
    return costs


def select_family(ordered_joint_costs):
    """Only input: ordered (policy, JOINT mean_J_total) pairs. No other metric."""
    check(len(ordered_joint_costs) > 0, "Empty candidate family")
    minimum = min(cost for _, cost in ordered_joint_costs)
    tied = [policy for policy, cost in ordered_joint_costs if cost == minimum]
    return {"selected": tied[0], "minimum_JOINT_mean_J_total": minimum,
            "tie_detected": len(tied) > 1, "tied_candidates": tied, "tie_rule": TIE_RULE}


def verify_selection(result, ordered_joint_costs):
    selected_cost = dict(ordered_joint_costs)[result["selected"]]
    tied = [name for name, value in ordered_joint_costs if value == selected_cost]
    return (all(selected_cost <= value for _, value in ordered_joint_costs)
            and result["selected"] == tied[0] and result["tied_candidates"] == tied
            and result["tie_detected"] == (len(tied) > 1) and result["tie_rule"] == TIE_RULE)


def freeze_selection(costs, manifest, audit):
    periodic_input = [(policy, costs[("JOINT", policy)]) for policy in PERIODIC]
    threshold_input = [(policy, costs[("JOINT", policy)]) for policy in THRESHOLDS]
    periodic = select_family(periodic_input)
    threshold = select_family(threshold_input)
    gate(audit, 7, verify_selection(periodic, periodic_input), periodic)
    gate(audit, 8, verify_selection(threshold, threshold_input), threshold)
    # Expected answers appear only AFTER mechanical selection.
    gate(audit, 9, periodic["selected"] == EXPECTED_PERIODIC, periodic["selected"])
    gate(audit, 10, threshold["selected"] == EXPECTED_THRESHOLD, threshold["selected"])
    gate(audit, 20, verify_selection(periodic, periodic_input) and verify_selection(threshold, threshold_input),
         {"periodic": periodic, "threshold": threshold, "tie_comparison": "exact float equality; no tolerance or rounding"})
    retained = list(EXTREMES) + [periodic["selected"], threshold["selected"]]
    gate(audit, 11, retained[:2] == list(EXTREMES), "Extreme controls retained unconditionally, no ranking")
    rows = []
    for family, policies in (("EXTREME_CONTROL", EXTREMES), ("PERIODIC", PERIODIC), ("TRUE_STATE_THRESHOLD", THRESHOLDS)):
        for order, policy in enumerate(policies):
            role = ("NO_MAINTENANCE_DIAGNOSTIC" if policy == EXTREMES[0] else
                    "OVER_MAINTENANCE_DEGENERATE_CONTROL" if policy == EXTREMES[1] else
                    "DEPLOYABLE_CALENDAR_BASELINE" if family == "PERIODIC" else "TRUE_STATE_INTERPRETABLE_BASELINE")
            rows.append({"policy": policy, "family": family, "candidate_order": order,
                         **{f"{scope}_mean_J_total": costs[(scope, policy)] for scope in SCOPES},
                         "selected_within_family": "NOT_APPLICABLE" if family == "EXTREME_CONTROL" else str(policy in retained),
                         "formal_baseline_retained": policy in retained, "role": role,
                         "selection_metric": SELECTION_METRIC, "selection_source_stage": "P2-1D-2B"})
    gate(audit, 21, len(retained) == len(set(retained)) == 4
         and {row["policy"] for row in rows if row["formal_baseline_retained"]} == set(retained), retained)
    selected = {"stage": STAGE, "selection_metric": SELECTION_METRIC,
                "main_lambda_c": manifest["main_lambda_c"], "eta": ETA, "reward_scale": REWARD_SCALE,
                "periodic_candidates": list(PERIODIC), "threshold_candidates": list(THRESHOLDS),
                "selected_periodic": periodic["selected"], "selected_true_threshold": threshold["selected"],
                "formal_nonrl_baselines": retained, "tie_evidence": {"periodic": periodic, "threshold": threshold},
                "baseline_roles": {row["policy"]: row["role"] for row in rows if row["formal_baseline_retained"]},
                "historical_cleaning_role": "BEHAVIOURAL_REFERENCE_ONLY", "selection_dataset_role": "PRE-RL DEVELOPMENT ONLY",
                "declaration": "NO BASELINE RESELECTION AFTER P2-2A-1",
                "freeze_effective_only_with": "Same-directory final audit_summary.json: stage_pass=True, failed_gates=[], scientific_status=NONRL_BASELINE_SET_PASS_FROZEN"}
    return rows, selected


def static_prohibition_audit(audit):
    tree = ast.parse((ROOT / SELF).read_text(encoding="utf-8"))
    allowed_imports = {"__future__", "argparse", "ast", "csv", "hashlib", "importlib", "io", "json", "math", "pathlib", "platform", "subprocess"}
    imports, calls = [], []
    forbidden_calls = {"Paper2CleaningEnv", "EpisodeSpec", "Paper2EnvAssets", "build_crn_indices", "sample_one",
                       "perception_substream_seed", "default_rng", "PPO", "learn", "optimizer", "backward", "minimize", "eval", "exec", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            calls.append(name)
    check(all(name.split(".")[0] in allowed_imports for name in imports), "Standard-library-only imports")
    check(not set(calls).intersection(forbidden_calls), "Prohibited scientific/training call")
    # Reserved namespaces can only be defined at module scope and referenced by
    # the protocol metadata function; never fed into execution or selection.
    for name, number in (("RESERVED_FORMAL_ENVIRONMENT_ROOTS", 15), ("RESERVED_FORMAL_PERCEPTION_SEEDS", 16)):
        users = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                 and any(isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id == name for child in ast.walk(node))]
        gate(audit, number, users == ["protocol"], {"readers": users, "role": "RESERVED_NOT_ACCESSED"})
    gate(audit, 17, not set(calls).intersection(forbidden_calls), {"imports": imports, "rollout_or_CRN_calls": []})
    gate(audit, 18, all(name.split(".")[0] in allowed_imports for name in imports), "No learner, torch, Gym or environment import")
    selector = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_family")
    loaded = {node.id for node in ast.walk(selector) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    gate(audit, 19, loaded <= {"ordered_joint_costs", "check", "len", "min", "cost", "_", "minimum", "policy", "tied", "TIE_RULE"},
         {"selector_inputs": [arg.arg for arg in selector.args.args], "loaded_names": sorted(loaded), "metric": SELECTION_METRIC})


def protocol():
    return {"stage": STAGE, "title": TITLE, "execution_role": "SELECTION FREEZE ONLY", "required_gates": GATES,
            "hash_correction": HASH_CORRECTION, "selection_occurred_before_PPO": True,
            "selection_occurred_before_formal_roots": True,
            "chronology_scope": "Pre-RL protocol declaration; this process accesses only pinned predecessor development artifacts, not later experiment results",
            "selection_source": "P2-1D-2B frozen development CRN results", "selection_metric": SELECTION_METRIC,
            "candidate_families": {"extremes": EXTREMES, "periodic": PERIODIC, "threshold": THRESHOLDS},
            "candidate_order_origin": 0, "tie_rule": TIE_RULE,
            "formal_environment": {"roots": RESERVED_FORMAL_ENVIRONMENT_ROOTS, "population_size": 300,
                                   "trajectory_ids": list(range(300)), "annual_episodes_per_policy": 600,
                                   "status": "RESERVED_NOT_ACCESSED_IN_P2_2A_1", "first_access_stage": "P2-2A-2"},
            "formal_perception": {"seeds": RESERVED_FORMAL_PERCEPTION_SEEDS, "status": "RESERVED_NOT_ACCESSED"},
            "development_perception_seed": {"value": 20260906, "role": "DOCUMENTATION_ONLY_NOT_USED_HERE"},
            "RL_training_seeds": "DO NOT ASSIGN RL TRAINING SEEDS; design separately in P2-2C",
            "PPO_imported": False, "Gym_rollout_executed": False, "new_CRN_generated": False,
            "reward_recalculated": False, "baseline_candidate_grid_changed": False,
            "historical_cleaning_role": "BEHAVIOURAL_REFERENCE_ONLY; not a target or claim of reproducing real operations",
            "NEVER_CLEAN_role": "NO_MAINTENANCE_DIAGNOSTIC; long-term extrapolation stress, no perception full-episode support claim",
            "DAILY_CLEAN_role": "OVER_MAINTENANCE_DEGENERATE_CONTROL",
            "periodic_information": "calendar only; no state, perception or future weather",
            "threshold_information": "latent true state; interpretable oracle-information baseline, not deployable Point/UA policy",
            "downstream_environment_module": CORE_FILE, "downstream_environment_rule": "P2-2A-2 and PPO must import the same frozen module; never copy environment logic",
            "P2_2A_2_metrics": {"primary": "mean J_total", "secondary": ["mean J_soiling", "mean N_clean"],
                               "distributional": ["median J_total", "P05 J_total", "P95 J_total"], "recommended": "CVaR95(J_total)",
                               "distributional_metrics_used_for_selection": False},
            "PPO_stop_gate": "After adequate training and fair budgets, True-State PPO should approach or exceed TRUE_THRESHOLD_0.05. If clearly worse, stop Point/UA PPO and inspect implementation, budget, hyperparameters, exploration, normalization and convergence; do not infer that uncertainty lacks value.",
            "DP_role": "P2-2B separately: Finite-Horizon Clairvoyant DP Oracle. Future GHI/rain/exogenous realizations imply CLAIRVOYANT_ORACLE, not an information-fair deployable competitor. True-State PPO need not beat it; gap is planner/oracle reference.",
            "next_stage": "P2-2A-2 Held-Out Formal Non-RL Baseline Evaluation; no PPO before completion",
            "declaration": "NO BASELINE RESELECTION AFTER P2-2A-1"}


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def publish(protocol_data, rows, selected, audit):
    check(all(value for name, value in audit["gates"].items() if name != GATES[23]), "All implementation gates required before publication")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    payloads = {"baseline_candidate_table.csv": stream.getvalue().encode("utf-8-sig"),
                "selected_baselines.json": encode(selected), "protocol_manifest.json": encode(protocol_data),
                "source_artifact_hashes.json": encode(protocol_data["source_artifacts"])}
    index = {"hashes": {name: sha(data) for name, data in payloads.items()},
             "excluded": ["output_hashes.json", "audit_summary.json"]}
    payloads["output_hashes.json"] = encode(index)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (OUTPUT / name).open("xb") as output:
            output.write(data)
    verified = {}
    for name, expected in payloads.items():
        actual = (OUTPUT / name).read_bytes()
        check(actual == expected and sha(actual) == sha(expected), f"Disk readback/hash: {name}")
        if name.endswith(".csv"):
            columns, reread = parse_csv(actual)
            check(columns == COLUMNS and len(reread) == 13, "Persisted candidate CSV schema/population")
            check(reread == [{key: str(row[key]) for key in COLUMNS} for row in rows], "Persisted candidate CSV values")
        else:
            check(json.loads(actual) == json.loads(expected), f"Persisted JSON schema/content: {name}")
        if name in index["hashes"]:
            check(sha(actual) == index["hashes"][name], "Output hash-index consistency")
        verified[name] = sha(actual)
    for record in protocol_data["source_artifacts"].values():
        check(sha(Path(record["path"]).read_bytes()) == record["sha256"], "Frozen source changed during execution")
    head = git("rev-parse", "HEAD")
    check(head == protocol_data["current_HEAD"], "HEAD changed during execution")
    check(script_record(SELF, head) == protocol_data["current_script"], "Current script changed during execution")
    for path, expected in SCRIPT_PINS.items():
        check(script_record(path, head, expected, CHECKPOINT) == protocol_data["script_provenance"][path], "Frozen script changed during execution")
    gate(audit, 23, {p.name for p in OUTPUT.iterdir()} == set(payloads),
         {"verified_non_audit_SHA256": verified, "verification": "exclusive write + actual readback + schema/content/hash"})
    audit["stage_pass"] = all(audit["gates"].values())
    audit["failed_gates"] = [name for name, value in audit["gates"].items() if not value]
    audit["scientific_status"] = "NONRL_BASELINE_SET_PASS_FROZEN"
    audit["declaration"] = "NO BASELINE RESELECTION AFTER P2-2A-1"
    data = encode(audit)
    with (OUTPUT / "audit_summary.json").open("xb") as output:
        output.write(data)
    check((OUTPUT / "audit_summary.json").read_bytes() == data, "Final audit readback")


def main():
    parser = argparse.ArgumentParser(description=f"{STAGE}: {TITLE}; frozen-result selection only, no modes")
    parser.parse_args()
    audit = {"stage": STAGE, "stage_pass": False, "gates": {name: False for name in GATES.values()},
             "gate_details": {}, "hash_correction": HASH_CORRECTION}
    protocol_data = protocol()
    try:
        check(not OUTPUT.exists(), f"Immutable output directory already exists: {OUTPUT}")
        static_prohibition_audit(audit)
        d2b, manifest = load_inputs(audit, protocol_data)
        costs = candidate_costs(d2b, manifest["main_multiplier"], manifest["main_lambda_c"], audit)
        rows, selected = freeze_selection(costs, manifest, audit)
        publish(protocol_data, rows, selected, audit)
    except Exception as exc:
        audit.update(stage_pass=False, scientific_status="FAIL_CLOSED_NO_BASELINE_FREEZE",
                     execution_error=f"{type(exc).__name__}: {exc}")
        audit["failed_gates"] = [name for name, value in audit["gates"].items() if not value]
        print(encode(audit).decode("utf-8"))  # Never overwrite immutable failure evidence.
        return 1
    print(encode({"stage": STAGE, "stage_pass": audit["stage_pass"], "scientific_status": audit["scientific_status"],
                  "output": str(OUTPUT)}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
