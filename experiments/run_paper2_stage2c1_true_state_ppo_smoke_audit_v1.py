"""P2-2C-1-v1: explicit engineering smoke only, never scientific selection.

Human source audit and Git checkpoint are required before execution. Importing
this script only defines metadata/functions; run_smoke() is the sole entry point
that loads assets, constructs trajectories, trains, evaluates or writes outputs.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess


STAGE = "P2-2C-1-v1"
TITLE = "True-State PPO Implementation and Smoke Training Audit"
STATUS = "TRUE_STATE_PPO_SMOKE_IMPLEMENTATION_PASS"
DECLARATION = "SMOKE IMPLEMENTATION ONLY; NO HYPERPARAMETER SELECTION; NO FORMAL PPO PERFORMANCE CLAIM"
ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage2c1_true_state_ppo_smoke_audit_v1.py"
RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
PREDECESSOR = "experiments/run_paper2_stage2c0_true_state_ppo_protocol_freeze_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
CHECKPOINT = "66c462c3635398d75a4cbab7abe7f5e4ce13f5d9"
PINS = {
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    PREDECESSOR: "6afdea8bf90061be78a06eeae8bd63f5515c66ba",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
}
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
FROZEN = BASE / "p2_2c_0_true_state_ppo_protocol_freeze_v1/formal"
OUTPUT = BASE / "p2_2c_1_true_state_ppo_smoke_audit_v1/smoke"
HASHES = {
    "audit_summary.json": "efeeedfab8510710ded8c34d85bf374d539dbbb2e22e2244d6003daa530c1ef8",
    "ppo_protocol.json": "f67fae8119d3a96606e01944853e612794f0e1d47c0fbf7a146cf9a38c394eb2",
    "ppo_candidate_configs.csv": "7953de644dc16e1906427ace113d6affbc4de9796994141381ea1d2e103ab454",
    "environment_role_partition.json": "c47d27556a5072b6a67831a27e9dd9ca3d6db28d080530bb54305bb57aa8acda",
    "rl_seed_registry.json": "9e6142d5555c60f63fa8275f345505a9653b4840c3a30d95204091299b500165",
    "training_schedule_contract.json": "3b803f06f64ea9b1487927c633f95b9819a04a72c04e5e4de8defd736276b534",
    "selection_contract.json": "7040e04b5c3eba16947929005ed1bf575c8b94dbbf048f1865e10e4e6c854344",
    "evaluation_contract.json": "737f32423ced093f3bd2a789a56898193d4e55ae215e13d5b8fb4cdfe753320e",
    "source_artifact_hashes.json": "1278b8636ee7fe6347700f20166cb06357a898a563344a175a9bae0688f71718",
    "output_hashes.json": "5aef22c890f977141d8ee5a88df9089063f2867213f854d826d855a33ef83b79",
}
GATES = (
    "predecessor_checkpoint", "protocol_blob", "protocol_audit_blob", "formal_hashes", "formal_status",
    "core_blob", "dependency_versions", "single_protocol_source", "True_State_observation", "WAIT_CLEAN_actions",
    "subseeds", "canonical_population", "unique_full_cycle", "schedule_reproduction", "config_independent_schedule",
    "smoke_CONFIG_A", "diagnostic_budget", "frozen_PPO_parameters", "no_normalization", "training_roots_only",
    "training_prefix_no_replacement", "zero_formal_access", "zero_perception", "learn_completion", "finite_parameters",
    "final_save_reload", "fixed_development_eval", "episode_horizon", "finite_valid_observations_actions_rewards_states",
    "reward_sign", "engineering_only", "source_HEAD_publication_integrity",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
                          check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def dependency_versions():
    return {"Python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("gymnasium", "stable_baselines3", "torch", "numpy", "pandas")}}


def static_review(sources):
    """Conservative AST checks plus mandatory human review; no code execution.

    Training is intentionally allowed only in train_final_checkpoint. Metadata
    containing forbidden names is harmless; executable references are rejected.
    """
    banned = {"VecNormalize", "EvalCallback", "StopTrainingOnRewardThreshold", "StopTrainingOnNoModelImprovement",
              "RewardWrapper", "ObservationWrapper", "OnlineBlock10Emulator", "ensure_perception", "perception_record",
              "best_model_save_path", "selected_config", "action_masks", "q50", "width", "oracle_action",
              "eval", "exec", "__import__", "exec_module", "import_module"}
    permitted_imports = {"__future__", "dataclasses", "datetime", "functools", "hashlib", "pathlib", "random", "time",
                         "gymnasium", "numpy", "paper2_ppo_protocol_v1", "paper2_gym_pomdp_env_v1",
                         "torch", "stable_baselines3", "argparse", "ast", "csv", "importlib.metadata",
                         "io", "json", "platform", "subprocess", "paper2_true_state_ppo_runner_v1"}
    trees = {name: ast.parse(text, filename=name) for name, text in sources.items()}
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                require(node.id not in banned and not node.id.startswith("StopTraining"), f"Banned name {node.id}")
            if isinstance(node, ast.Attribute):
                require(node.attr not in banned and not node.attr.startswith("StopTraining"), f"Banned attribute {node.attr}")
            if isinstance(node, ast.keyword):
                require(node.arg not in banned, "Banned keyword")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                require(all(x in permitted_imports for x in names), f"Import outside audited surface: {names}")
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                require(str(node.value) not in ("1220001", "1230001"), "No executable formal-root literals in these files")
        for node in tree.body:
            if isinstance(node, ast.Expr):
                require(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), "Import-time expression")
            elif isinstance(node, ast.If):
                require(name == SELF and ast.dump(node.test) == ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body), "Only CLI main guard")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                calls = [x for x in ast.walk(node) if isinstance(x, ast.Call)]
                require(all(isinstance(x.func, ast.Name) and x.func.id == "Path" or
                            isinstance(x.func, ast.Attribute) and x.func.attr == "resolve" for x in calls), "No runtime construction at import")
            else:
                require(isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)), "Unexpected top-level statement")
    runner = trees[RUNNER]
    require(any(isinstance(n, ast.Import) and any(a.name == "paper2_ppo_protocol_v1" for a in n.names)
                for n in runner.body), "Runner must import frozen protocol")
    learn_calls = []
    constructors = []
    for func in (n for n in runner.body if isinstance(n, ast.FunctionDef)):
        for call in (n for n in ast.walk(func) if isinstance(n, ast.Call)):
            if isinstance(call.func, ast.Attribute) and call.func.attr == "learn":
                learn_calls.append(func.name)
                require(not any(k.arg == "callback" for k in call.keywords), "No model-selection/early-stop callback")
            if isinstance(call.func, ast.Attribute) and call.func.attr in ("EpisodeSpec", "Paper2CleaningEnv"):
                constructors.append((func.name, call.func.attr))
            if isinstance(call.func, ast.Attribute) and call.func.attr == "predict":
                require(func.name == "evaluate_deterministic" and any(k.arg == "deterministic" for k in call.keywords), "Causal deterministic prediction")
    require(learn_calls == ["train_final_checkpoint"], "Exactly one explicit training path")
    require(constructors == [("construct_core", "EpisodeSpec"), ("construct_core", "Paper2CleaningEnv")], "Single guarded core constructor")
    guard = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "construct_core")
    require(isinstance(guard.body[0], ast.Expr) and isinstance(guard.body[0].value, ast.Call)
            and isinstance(guard.body[0].value.func, ast.Attribute) and guard.body[0].value.func.attr == "authorize", "Authorize before EpisodeSpec")
    return {"AST_checked": True, "learn_sites": learn_calls, "core_constructor": "construct_core after ledger.authorize",
            "normalization_or_selection_references": False, "formal_root_literal_paths": False}


def csv_bytes(rows):
    require(rows and all(set(row) == set(rows[0]) for row in rows), "Consistent nonempty CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def run_smoke():
    # Fail before loading assets/training if output already exists or sources uncommitted.
    require(not OUTPUT.exists(), "Exclusive output directory already exists")
    head = git("rev-parse", "HEAD")
    gates = {}

    def gate(number, passed):
        gates[f"G{number}"] = bool(passed)
        require(passed, f"G{number}: {GATES[number - 1]}")

    sources = {}

    def capture(path, digest=None):
        path = Path(path)
        key = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        require(digest is None or sha(data) == digest, f"Frozen hash: {key}")
        require(key not in sources or sources[key] == sha(data), f"Source changed: {key}")
        sources[key] = sha(data)
        return data

    gate(1, git("rev-parse", "--verify", f"{CHECKPOINT}^{{commit}}") == CHECKPOINT)
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for number, name in ((2, PROTOCOL), (3, PREDECESSOR), (6, CORE)):
        gate(number, git("rev-parse", f"{CHECKPOINT}:{name}") == git("rev-parse", f"{head}:{name}")
             == git("hash-object", f"--path={name}", name) == PINS[name])
        capture(ROOT / name)
    for name in (RUNNER, SELF):
        require(git("rev-parse", "--verify", f"{head}:{name}") == git("hash-object", f"--path={name}", name),
                "Both current files must be committed unchanged before smoke")
        capture(ROOT / name)
    frozen_docs = {name: capture(FROZEN / name, digest) for name, digest in HASHES.items()}
    gate(4, len(frozen_docs) == 10)
    audit = json.loads(frozen_docs["audit_summary.json"])
    gate(5, audit.get("stage_pass") is True and audit.get("failed_gates") == []
         and audit.get("scientific_status") == "TRUE_STATE_PPO_PROTOCOL_PASS_FROZEN"
         and audit.get("declaration") == "TRUE-STATE PPO PROTOCOL PRE-REGISTERED BEFORE PPO TRAINING; NO FORMAL PPO PERFORMANCE ACCESSED; NO POINT/UA HYPERPARAMETER RESELECTION")
    frozen_payload = json.loads(frozen_docs["ppo_protocol.json"])
    versions = dependency_versions()
    gate(7, versions == dict(frozen_payload["dependencies"]))
    static = static_review({name: (ROOT / name).read_text(encoding="utf-8") for name in (RUNNER, SELF)})

    # Imports below cannot happen until exact provenance/dependency checks passed.
    import paper2_ppo_protocol_v1 as frozen
    import paper2_true_state_ppo_runner_v1 as runner
    p = runner.get_protocol()
    gate(8, frozen.protocol_payload(p) == frozen_payload and static["AST_checked"])
    parent_seed, config = p.seeds.development[0], p.candidates[0]
    development = runner.partition("DEVELOPMENT")
    eval_specs = tuple((year, development.trajectory_ids[0]) for year in development.years)
    subseeds = runner.derive_frozen_rl_seeds(parent_seed)
    children = runner.child_seeds(parent_seed)
    env_roots = {root for part in p.partitions for root in part.roots}
    registry = json.loads(frozen_docs["rl_seed_registry.json"])
    gate(11, subseeds == next(x for x in registry["substreams"] if x["parent_seed"] == parent_seed)
         and subseeds == runner.derive_frozen_rl_seeds(parent_seed)
         and len(set(children.values())) == 2 and not set(children.values()) & env_roots)
    canonical = frozen.canonical_training_specs()
    gate(12, len(canonical) == 600 and len(set(canonical)) == 600
         and json.loads(json.dumps(canonical)) == json.loads(frozen_docs["training_schedule_contract.json"])["canonical_specs"])

    def cycle(config_id):
        schedule = runner.build_training_schedule(parent_seed, config_id)
        return [schedule.next_assignment() for _ in range(len(canonical))]

    cycle_a, repeat_a, cycle_b = cycle("CONFIG_A"), cycle("CONFIG_A"), cycle("CONFIG_B")
    keys = {runner.assignment_key(x) for x in cycle_a}
    gate(13, len(keys) == len(cycle_a) == 600 and keys == {(x[0], x[1], x[2]) for x in canonical})
    gate(14, cycle_a == repeat_a)
    gate(15, cycle_a == cycle_b)
    gate(16, config.name == "CONFIG_A" and parent_seed == 510001)
    gate(17, runner.DIAGNOSTIC_SMOKE_BUDGET == 4096 and 4096 // config.n_steps == 2
         and 4096 % config.n_steps == 0 and 4096 != p.development_budget)
    gate(19, not p.observation_normalization and not p.reward_normalization
         and not static["normalization_or_selection_references"])

    # Capture frozen asset inputs BEFORE asset loading, including helper source.
    core = runner.core
    for name, digest in core.REWARD_HASHES.items():
        capture(core.REWARD_DIRECTORY / name, digest)
    capture(ROOT / core.REWARD_SCRIPT)
    for path, blob in core.HELPERS.values():
        capture(ROOT / path)
        require(git("rev-parse", f"{head}:{path}") == git("hash-object", f"--path={path}", path) == blob, "Frozen helper")
    for path, digest in core.ASSET_HASHES.values():
        capture(path, digest)
    # Exclusive directory claim before expensive work. Failed runs retain evidence,
    # never a PASS summary; reruns cannot overwrite either complete or partial output.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    assets = runner.load_assets()
    ledger = runner.AccessLedger()
    env = runner.build_scheduled_true_state_env(assets, parent_seed, ledger, config.name)
    try:
        gate(9, env.current.observation_mode == "True-State" and env.observation_space.shape == (3,)
             and p.observations[0] == ("True-State", ("L_true", "sin_DOY", "cos_DOY")))
        gate(10, env.action_space.n == 2 and env.action_space.start == 0 and p.actions == ((0, "WAIT"), (1, "CLEAN")))
        bundle = runner.build_ppo_model(env, config.name, parent_seed)
        gate(18, runner.verify_model_contract(bundle.model, config.name, parent_seed))
        training = runner.train_final_checkpoint(bundle, OUTPUT / "final_smoke_model.zip", purpose="smoke")
        gate(24, training["total_timesteps"] == 4096 == env.total_steps and training["rollout_updates"] == 2
             and training["scientific_budget_used"] is False)
        gate(25, training["model_parameter_finite"] and training["optimizer_steps_checked"] > 0)
        reloaded = runner.reload_final_checkpoint(OUTPUT / "final_smoke_model.zip", bundle)
        gate(26, reloaded is not bundle.model and runner.parameters_finite(reloaded)
             and runner.file_sha256(OUTPUT / "final_smoke_model.zip") == training["model_sha256"])
        assignments = env.assignments
        canonical_keys = {(x[0], x[1], x[2]) for x in canonical}
        train_records = [x for x in ledger.records if x["role"] == "TRAINING"]
        gate(20, len(train_records) == len(assignments) and all(runner.assignment_key(x) in canonical_keys for x in train_records))
        prefix = [runner.assignment_key(x) for x in assignments]
        gate(21, len(prefix) <= len(cycle_a) and len(prefix) == len(set(prefix))
             and prefix == [runner.assignment_key(x) for x in cycle_a[:len(prefix)]]
             and sum(x["steps"] for x in assignments) == 4096
             and env.episodes_completed == 4096 // p.decision_reward_steps
             and assignments[-1]["steps"] == 4096 % p.decision_reward_steps)
        metrics = runner.evaluate_deterministic(reloaded, assets, eval_specs, ledger)
        expected_eval_keys = {(y, r, development.trajectory_ids[0]) for y, r in zip(development.years, development.roots)}
        eval_records = [x for x in ledger.records if x["role"] == "DEVELOPMENT"]
        gate(27, len(metrics) == len(eval_records) == 2
             and {runner.assignment_key(x) for x in metrics} == expected_eval_keys
             and {runner.assignment_key(x) for x in eval_records} == expected_eval_keys)
        gate(28, all(x["step_count"] == p.decision_reward_steps and x["natural_transition_count"] == p.natural_transitions for x in metrics))
        gate(29, env.observations_checked == env.total_steps + len(assignments)
             and all(x["all_observations_finite"] and x["all_rewards_finite"] and x["all_states_finite"]
                     and x["action_min"] >= 0 and x["action_max"] <= 1 and x["deterministic_action"] for x in metrics))
        gate(30, all(x["reward_sign_regression_diff"] <= 1e-12 for x in metrics))
        # Cross-check constructor records against actual frozen core bank cache.
        banks = [{"year": y, "environment_root": r, "population_size": n} for y, r, n in assets._banks]
        allowed_roots = set(runner.partition("TRAINING").roots) | set(development.roots)
        gate(22, ledger.heldout_ppo_trajectory_access_count == 0 and ledger.denied_accesses == 0
             and all(x["environment_root"] in allowed_roots for x in banks))
        gate(23, env.perception_query_count == 0 and all(x["perception_query_count"] == 0 for x in assignments + metrics)
             and assets.perception_construction_count == assets.perception_sample_count == 0 and assets.emulator is None)
        gate(31, parent_seed not in p.seeds.final and config.name == "CONFIG_A" and not training["scientific_budget_used"]
             and STATUS == "TRUE_STATE_PPO_SMOKE_IMPLEMENTATION_PASS"
             and DECLARATION == "SMOKE IMPLEMENTATION ONLY; NO HYPERPARAMETER SELECTION; NO FORMAL PPO PERFORMANCE CLAIM")
        manifest = runner.model_manifest(bundle, training)
    finally:
        env.close()

    summary_training = {**training, "model_path": "final_smoke_model.zip", "config_id": config.name,
        "learning_rate": config.learning_rate, "n_steps": config.n_steps,
        "parent_rl_seed": parent_seed, "learner_seed": children["learner"], "schedule_seed": children["training_schedule"],
        "episodes_started": len(assignments), "episodes_completed": env.episodes_completed,
        "training_roots": list(runner.partition("TRAINING").roots), "development_eval_roots": list(development.roots),
        "formal_root_access_count": ledger.heldout_ppo_trajectory_access_count,
        "perception_query_count": env.perception_query_count + sum(x["perception_query_count"] for x in metrics),
        "scientific_budget_used": False, "observations_checked": env.observations_checked,
        "last_episode_incomplete": not assignments[-1]["completed"]}
    protocol_manifest = {"stage": STAGE, "title": TITLE, "mode": "smoke", "declaration": DECLARATION,
        "budget_role": "DIAGNOSTIC_SMOKE_BUDGET", "scientific_budget_used": False,
        "config_id": config.name, "parent_rl_seed": parent_seed, "total_timesteps": runner.DIAGNOSTIC_SMOKE_BUDGET,
        "frozen_protocol_sha256": HASHES["ppo_protocol.json"], "substreams": subseeds,
        "development_eval_specs": [runner.development_assignment(y, t) for y, t in eval_specs],
        "observation": dict(p.observations)["True-State"], "observation_dtype": str(env.observation_space.dtype),
        "observation_semantics": "every decision observation compared to private L_pre and date-derived sin/cos; terminal zero sentinel checked separately",
        "deterministic_evaluation": True, "schedule_rng": dict(p.schedule)["rng"],
        "full_cycle_audit_is_metadata_only": True, "no_performance_threshold": True,
        "model_use": "diagnostic only; never development candidate, final or formal model",
        "wall_clock_scope": "PPO.learn only; excludes assets, save/reload and evaluation",
        "trajectory_access_log": ledger.records, "generated_banks": banks, "static_review": static}
    documents = {"smoke_protocol_manifest.json": protocol_manifest,
                 "smoke_training_summary.json": summary_training, "model_manifest.json": manifest,
                 "source_artifact_hashes.json": {"HEAD": head, "checkpoint": CHECKPOINT,
                    "git_blobs": {name: git("rev-parse", f"{head}:{name}") for name in (SELF, RUNNER, *PINS)},
                    "sha256": sources, "dependencies": versions, "core_assets_provenance": assets.provenance}}
    content = {name: encode(doc) for name, doc in documents.items()}
    content.update({"schedule_cycle_audit.csv": csv_bytes(cycle_a),
                    "training_episode_assignments.csv": csv_bytes(assignments),
                    "smoke_eval_metrics.csv": csv_bytes(metrics)})
    output_hashes = {name: sha(data) for name, data in content.items()}
    output_hashes["final_smoke_model.zip"] = training["model_sha256"]
    content["output_hashes.json"] = encode({"hashes": output_hashes,
        "exclusions": {"output_hashes.json": "hash recorded in audit_summary to avoid self-reference",
                       "audit_summary.json": "last publication marker; pin externally after acceptance"}})
    for name, data in content.items():
        with (OUTPUT / name).open("xb") as stream:
            stream.write(data)
    require({x.name for x in OUTPUT.iterdir()} == set(content) | {"final_smoke_model.zip"}, "Exclusive inventory")
    for name, data in content.items():
        readback = (OUTPUT / name).read_bytes()
        require(readback == data, f"Exact output readback: {name}")
        if name.endswith(".json"):
            require(json.loads(readback) == json.loads(data), f"JSON readback/schema: {name}")
        else:
            rows = list(csv.DictReader(io.StringIO(readback.decode())))
            require(rows and all(set(r) == set(rows[0]) and None not in r.values() for r in rows), f"CSV schema: {name}")
    index = json.loads((OUTPUT / "output_hashes.json").read_bytes())["hashes"]
    require(index == {n: runner.file_sha256(OUTPUT / n) for n in output_hashes}, "Output hash completeness")
    gate(32, git("rev-parse", "HEAD") == head and all(runner.file_sha256(ROOT / n) == digest for n, digest in sources.items()))
    require(set(gates) == {f"G{i}" for i in range(1, 33)} and all(gates.values()), "All 32 gates")
    summary = {"stage": STAGE, "title": TITLE, "mode": "smoke", "stage_pass": True,
               "scientific_status": STATUS, "declaration": DECLARATION, "HEAD": head,
               "gates": gates, "gate_names": {f"G{i}": n for i, n in enumerate(GATES, 1)}, "failed_gates": [],
               "scientific_budget_used": False, "heldout_ppo_trajectory_access_count": ledger.heldout_ppo_trajectory_access_count,
               "perception_query_count": summary_training["perception_query_count"],
               "output_sha256": {**output_hashes, "output_hashes.json": sha(content["output_hashes.json"])}}
    data = encode(summary)
    # Publish the completion marker only after all gates/readbacks/hashes succeed.
    with (OUTPUT / "audit_summary.json").open("xb") as stream:
        stream.write(data)
    require((OUTPUT / "audit_summary.json").read_bytes() == data, "Final summary readback")
    require(len(list(OUTPUT.iterdir())) == 10, "Final ten-file inventory")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke",))
    parser.parse_args()
    print(json.dumps(run_smoke(), indent=2))


if __name__ == "__main__":
    main()
