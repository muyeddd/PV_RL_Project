"""P2-2C-3-v1: True-State PPO Final-Seed Training and Development Sanity Audit.

Import is inert. Explicit --mode formal, committed unchanged source, a clean
Git tree, exact predecessor provenance and a new stage directory are required.
There is no continuation, reselection, retry or held-out evaluation entry point.
Sanity failure preserves all five models and evidence, never a PASS marker.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage2c3_true_state_ppo_final_seed_sanity_v1.py"
PREDECESSOR = "experiments/run_paper2_stage2c2_true_state_ppo_development_selection_v1.py"
SMOKE_SOURCE = "experiments/run_paper2_stage2c1_true_state_ppo_smoke_audit_v1.py"
RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
CHECKPOINT = "06846d6d141a68a3fc9e40c282e99dda07992c0e"
PINS = {
    PREDECESSOR: "0ed5bf31aa6a6aaae312a7a0dea170cd050bef3d",
    RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
}
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
FROZEN = BASE / "p2_2c_2_true_state_ppo_development_selection_v1/formal"
STAGE_DIRECTORY = BASE / "p2_2c_3_true_state_ppo_final_seed_sanity_v1"
OUTPUT = STAGE_DIRECTORY / "formal"
PREDECESSOR_HASHES = {
    "audit_summary.json": "44c2bdea5f476bea9498b514072ad5a3cfb8fa69bcdba575a32a6d25b50dec18",
    "selected_config.json": "3cc4bdc4393936f736e659fbbe0ded37adb39ebb0871ccf1b5269fdefe4e9a67",
    "candidate_selection_summary.csv": "bfd25ac752c1a7148df1aa70e952441a7b819ec1724d90b36b3df3028dbb4569",
    "development_run_summary.csv": "3a39d999590126ca6e8dc94bc9b917ec941572747f394297851e0522d6560016",
    "source_artifact_hashes.json": "1002ddabf2c28434d06818bf6d791b9487ed09ce82c0f3dc928c51a9b03c6020",
    "protocol_manifest.json": "f4d36bf2deffd321316c350a137d25ef49acd4baba87dd89be4a2ee360817260",
    "output_hashes.json": "80da9d4e50c2470af8c15e6ddb98fbdbb98fe831be5cc00d28c641f2b300a648",
}
STAGE = "P2-2C-3-v1"
TITLE = "True-State PPO Final-Seed Training and Development Sanity Audit"
PASS_STATUS = "TRUE_STATE_PPO_FINAL_SEED_SANITY_PASS_FROZEN"
FAIL_STATUS = "TRUE_STATE_PPO_FINAL_SEED_SANITY_FAIL"
DECLARATION = ("FIVE FINAL TRUE-STATE PPO MODELS FROZEN AFTER DEVELOPMENT SANITY; "
               "NO FORMAL HELD-OUT PPO PERFORMANCE ACCESSED; THESE EXACT MODEL SHA256S ONLY ARE ELIGIBLE "
               "FOR LATER FORMAL EVALUATION; NO RETRAINING OR SEED REPLACEMENT")
MODEL_DECLARATION = ("THESE EXACT FIVE MODEL SHA256S ARE THE ONLY TRUE-STATE PPO MODELS "
                     "ELIGIBLE FOR LATER FORMAL HELD-OUT EVALUATION")
RUN_GATES = (
    "fresh_model", "fresh_schedule", "fresh_substreams", "True_State_only", "final_budget_exact",
    "development_budget_and_seeds_unused", "training_roots_only", "frozen_schedule_exact",
    "final_checkpoint_only", "no_early_stopping_best_model_callback_or_rollback",
    "parameters_finite", "save_reload_exact", "full_development_population",
    "zero_formal_access", "zero_perception_access", "valid_actions",
    "finite_observations_rewards_states", "reward_sign_regression",
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


def csv_bytes(rows):
    require(rows and all(set(r) == set(rows[0]) for r in rows), "Consistent nonempty CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


def write_exclusive(path, data):
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, f"Exact readback: {path.name}")


def sanity_input(rows):
    return [{k: r[k] for k in ("parent_rl_seed", "mean_J_total")} for r in rows]


def sanity_result(seed_means, protocol):
    """Hard gates consume only the five full-population seed means and contract.

    Cost-distribution tails, cleaning counts and descriptive gaps cannot enter
    either decision. Sample standard deviation describes five training seeds.
    """
    e = protocol.evaluation
    seeds = protocol.seeds.final
    require(len(seed_means) == len(seeds) == e.replication_n == 5, "Exactly five final seeds")
    values_by_seed = {}
    for row in seed_means:
        require(set(row) == {"parent_rl_seed", "mean_J_total"}, "Sanity input whitelist")
        seed, value = int(row["parent_rl_seed"]), float(row["mean_J_total"])
        require(seed not in values_by_seed and math.isfinite(value), "Unique finite seed mean")
        values_by_seed[seed] = value
    require(set(values_by_seed) == set(seeds), "Exact frozen final seed population")
    values = [values_by_seed[s] for s in seeds]
    mean = sum(values) / len(values)
    std = statistics.stdev(values)
    require(math.isfinite(mean) and math.isfinite(std), "Finite seed-level statistics")
    gate_a = mean <= e.gate_a_max
    gate_b_by_seed = {str(s): values_by_seed[s] < e.periodic_mean for s in seeds}
    gate_b = all(gate_b_by_seed.values())
    return {"threshold_baseline": e.threshold_baseline, "threshold_mean": e.threshold_mean,
            "tolerance_multiplier": e.tolerance_multiplier, "gate_a_max": e.gate_a_max,
            "periodic_baseline": e.periodic_baseline, "periodic_mean": e.periodic_mean,
            "each_seed_development_mean_J_total": {str(s): values_by_seed[s] for s in seeds},
            "five_seed_mean_J_total": mean, "five_seed_std": std, "std_ddof": 1,
            "five_seed_median": statistics.median(values), "five_seed_min": min(values),
            "five_seed_max": max(values), "five_seed_range": max(values) - min(values),
            "gate_a_definition": e.gate_a, "gate_b_definition": e.gate_b,
            "gate_a_pass": gate_a, "gate_b_pass": gate_b, "gate_b_by_seed": gate_b_by_seed,
            "overall_sanity_pass": gate_a and gate_b,
            "scientific_status": PASS_STATUS if gate_a and gate_b else FAIL_STATUS,
            "instruction": "MODELS FROZEN; LATER FORMAL ACCESS REQUIRES ITS OWN AUDIT" if gate_a and gate_b
                           else "STOP BEFORE POINT/UA",
            "each_seed_gap_vs_threshold": {str(s): values_by_seed[s] - e.threshold_mean for s in seeds},
            "descriptive_five_seed_gap_vs_threshold": mean - e.threshold_mean,
            "gap_role": "DESCRIPTIVE ONLY; positive means higher cost; no model or seed selection",
            "statistical_unit": e.replication_unit, "independent_replications": e.replication_n,
            "environment_episode_role": "within-fixed-policy environment evaluation distribution"}


def static_review(source):
    """Conservative execution-surface audit, complemented by frozen blob checks."""
    tree = ast.parse(source)
    banned = {"learn", "predict", "PPO", "EpisodeSpec", "Paper2CleaningEnv", "Paper2EnvAssets",
              "ensure_perception", "perception_record", "VecNormalize", "EvalCallback", "select_candidate",
              "selector_input", "construct_core", "build_true_state_eval_env", "exec", "eval", "__import__",
              "load_frozen_module", "load", "save", "unlink", "rmdir", "getattr", "setattr",
              "import_module", "exec_module"}
    training_calls = []
    runner_api = {"get_protocol", "candidate", "child_seeds", "derive_frozen_rl_seeds", "partition",
                  "development_assignment", "assignment_key", "load_assets", "AccessLedger",
                  "build_scheduled_true_state_env", "build_training_schedule", "build_ppo_model",
                  "verify_model_contract", "train_final_checkpoint", "reload_final_checkpoint",
                  "evaluate_deterministic", "parameters_finite", "file_sha256", "model_manifest"}
    allowed_imports = {"__future__", "argparse", "ast", "csv", "hashlib", "io", "json", "math", "pathlib",
                       "statistics", "subprocess", "paper2_ppo_protocol_v1", "paper2_true_state_ppo_runner_v1",
                       "run_paper2_stage2c1_true_state_ppo_smoke_audit_v1",
                       "run_paper2_stage2c2_true_state_ppo_development_selection_v1"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            require(name not in banned and not name.startswith("StopTraining"), f"Forbidden execution: {name}")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "runner":
                require(node.attr in runner_api, "Only development-safe frozen runner API")
            if node.value.id == "development":
                require(node.attr in {"summarize", "SMOKE", "SMOKE_AUDIT_SHA256"}, "No predecessor execution or reselection")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            require(all(n in allowed_imports for n in names), "Audited imports only")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "train_final_checkpoint":
                training_calls.append(node)
                require(len(node.keywords) == 1 and node.keywords[0].arg == "purpose"
                        and isinstance(node.keywords[0].value, ast.Constant)
                        and node.keywords[0].value.value == "final", "Final training only")
            if node.func.attr == "partition":
                require(len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value in ("TRAINING", "DEVELOPMENT"), "Allowed partitions only")
    require(len(training_calls) == 1, "Single final training call site")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            require(all(n in allowed_imports and not n.startswith(("paper2", "run_paper2")) for n in names),
                    "Scientific imports must be lazy")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                require((isinstance(call.func, ast.Name) and call.func.id == "Path") or
                        (isinstance(call.func, ast.Attribute) and call.func.attr == "resolve"), "Inert initializer")
        elif isinstance(node, ast.FunctionDef):
            require(not node.decorator_list and not node.args.defaults and not any(node.args.kw_defaults), "Inert definition")
        elif isinstance(node, ast.Expr):
            require(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), "Module docstring only")
        else:
            require(isinstance(node, ast.If) and ast.dump(node.test) ==
                    ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body), "Only CLI main guard")
    return True


def read_predecessors(head):
    """Read-only provenance: no assets, models, selector, or trajectory execution.

    Pin the accepted C2 audit, all its output hashes, and its inherited source
    inventory. A model archive is hashed as opaque bytes, never deserialized.
    No Oracle/held-out performance artifact is followed through older manifests.
    """
    sources, blobs, gates = {}, {}, {}

    def capture(path, digest):
        path = Path(path).resolve()
        key = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        require(sha(data) == digest, f"Frozen SHA256: {key}")
        require(key not in sources or sources[key] == digest, f"Source changed: {key}")
        sources[key] = digest
        return data

    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    docs = {n: capture(FROZEN / n, digest) for n, digest in PREDECESSOR_HASHES.items()}
    audit = json.loads(docs["audit_summary.json"])
    require(audit.get("stage_pass") is True and audit.get("failed_gates") == []
            and audit.get("scientific_status") == "TRUE_STATE_PPO_DEVELOPMENT_SELECTION_PASS_FROZEN"
            and audit.get("HEAD") == CHECKPOINT and audit.get("run_count") == 8
            and audit.get("evaluation_episode_count") == 4800 and audit.get("mode") == "formal"
            and audit.get("heldout_ppo_trajectory_access_count") == audit.get("perception_query_count") == 0
            and audit.get("gates") and all(v is True for v in audit["gates"].values()), "P2-2C-2 accepted audit exact")
    for name, digest in audit["output_sha256"].items():
        path = (FROZEN / name).resolve()
        require(path.is_relative_to(FROZEN), "Predecessor output path containment")
        capture(path, digest)
    index = json.loads(docs["output_hashes.json"])["hashes"]
    require(index == {n: h for n, h in audit["output_sha256"].items() if n != "output_hashes.json"},
            "Predecessor hash index exact")
    provenance = json.loads(docs["source_artifact_hashes.json"])
    require(provenance["HEAD"] == CHECKPOINT, "Predecessor provenance HEAD exact")
    for name, digest in provenance["sha256"].items():
        capture(ROOT / name, digest)
    require(all(provenance["git_blobs"].get(n) == b for n, b in PINS.items()), "Frozen source pins exact")
    for name, blob in provenance["git_blobs"].items():
        require(git("rev-parse", f"{CHECKPOINT}:{name}") == git("rev-parse", f"{head}:{name}")
                == git("rev-parse", f":{name}") == git("hash-object", f"--path={name}", name) == blob,
                f"Frozen committed source: {name}")
        blobs[name] = blob
    # These modules are inert and have now been verified against accepted blobs.
    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as smoke
    import run_paper2_stage2c2_true_state_ppo_development_selection_v1 as development
    old_docs = {n: capture(smoke.FROZEN / n, h) for n, h in smoke.HASHES.items()}
    for label, data, status in (
        ("P2_2C_0_PASS_FROZEN", old_docs["audit_summary.json"], "TRUE_STATE_PPO_PROTOCOL_PASS_FROZEN"),
        ("P2_2C_1_PASS", capture(development.SMOKE / "audit_summary.json", development.SMOKE_AUDIT_SHA256),
         "TRUE_STATE_PPO_SMOKE_IMPLEMENTATION_PASS"),
    ):
        previous = json.loads(data)
        require(previous.get("stage_pass") is True and previous.get("failed_gates") == []
                and previous.get("scientific_status") == status and previous.get("gates")
                and all(v is True for v in previous["gates"].values()), label)
        gates[label] = True
    selected = json.loads(docs["selected_config.json"])
    # Read the accepted identity; regression assertions do not select a winner.
    selected_id = selected["selected_config"]
    require(selected_id == audit["selected_config"] == "CONFIG_A"
            and selected["declaration"] == "NO HYPERPARAMETER RESELECTION AFTER P2-2C-2", "Frozen selected identity")
    rows = csv_rows(docs["candidate_selection_summary.csv"])
    marked = [r for r in rows if r["selected"] == "True"]
    require(len(rows) == 4 and len(marked) == 1 and marked[0]["config_id"] == selected_id
            and float(marked[0]["primary_score"]) == selected["selected_primary_score"]
            and float(marked[0]["worst_seed_dev_mean_J_total"]) == selected["selected_worst_seed_score"],
            "Accepted selection artifacts agree; no scores ranked or selector invoked")
    require(len(csv_rows(docs["development_run_summary.csv"])) == audit["run_count"], "Predecessor run count")
    payload = json.loads(old_docs["ppo_protocol.json"])
    require(json.loads(docs["protocol_manifest.json"])["protocol"] == payload, "Predecessor protocol identity")
    gates.update({"P2_2C_2_PASS_FROZEN": True, "predecessor_provenance_exact": True,
                  "selected_config_artifact_exact": True, "selected_config_frozen_no_reselection": True,
                  "frozen_runner_protocol_environment_blobs": True})
    return {"sources": sources, "blobs": blobs, "gates": gates, "selected_config": selected_id,
            "selected_artifact": selected, "protocol_payload": payload,
            "seed_registry": json.loads(old_docs["rl_seed_registry.json"]),
            "canonical_training_specs": json.loads(old_docs["training_schedule_contract.json"])["canonical_specs"]}


def run_formal():
    require(not STAGE_DIRECTORY.exists(), "Stage output directory must not exist; no resume or overwrite")
    require(git("status", "--porcelain", "--untracked-files=all") == "", "Clean working tree and index required")
    head = git("rev-parse", "HEAD")
    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(self_blob == git("hash-object", f"--path={SELF}", SELF) == git("rev-parse", f":{SELF}"),
            "New script must be committed unchanged")
    source = (ROOT / SELF).read_bytes()
    require(static_review(source.decode("utf-8")), "Static boundaries")
    prior = read_predecessors(head)
    sources, blobs, gates = prior["sources"], prior["blobs"], prior["gates"]
    sources[SELF], blobs[SELF] = sha(source), self_blob

    def gate(name, condition):
        require(name not in gates, f"Duplicate gate: {name}")
        gates[name] = bool(condition)
        require(condition, name)

    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as smoke
    import run_paper2_stage2c2_true_state_ppo_development_selection_v1 as development
    versions = smoke.dependency_versions()
    gate("dependency_versions", versions == dict(prior["protocol_payload"]["dependencies"]))
    runner_review = smoke.static_review({n: (ROOT / n).read_text(encoding="utf-8") for n in (RUNNER, SMOKE_SOURCE)})
    import paper2_ppo_protocol_v1 as frozen
    import paper2_true_state_ppo_runner_v1 as runner
    p = runner.get_protocol()
    gate("frozen_protocol_exact", frozen.protocol_payload(p) == prior["protocol_payload"])
    c = runner.candidate(prior["selected_config"])
    gate("selected_config_expected_regression", c.name == prior["selected_config"] == "CONFIG_A"
         and c.learning_rate == 0.0003 and c.n_steps == 2048)
    gate("exactly_five_final_seeds", len(p.seeds.final) == len(set(p.seeds.final)) == 5
         and not set(p.seeds.final) & set(p.seeds.development))
    gate("final_budget_contract", p.final_budget == p.selection.final_checkpoint
         and p.final_budget != p.development_budget and p.final_budget % c.n_steps == 0)
    gate("statistical_unit", p.evaluation.replication_unit == "RL TRAINING SEED"
         and p.evaluation.replication_n == len(p.seeds.final) == 5)
    children = {seed: runner.child_seeds(seed) for seed in p.seeds.final}
    registered = {r["parent_seed"]: r for r in prior["seed_registry"]["substreams"]}
    gate("frozen_independent_substreams", all(runner.derive_frozen_rl_seeds(s) == registered[s] for s in p.seeds.final)
         and len({v for pair in children.values() for v in pair.values()}) == 2 * len(p.seeds.final))
    part = runner.partition("DEVELOPMENT")
    specs = tuple((year, tid) for year in part.years for tid in part.trajectory_ids)
    eval_assignments = [runner.development_assignment(y, t) for y, t in specs]
    eval_keys = [runner.assignment_key(r) for r in eval_assignments]
    gate("full_development_specs", len(specs) == len(set(specs)) == part.population_size
         == p.evaluation.episodes_per_model == 600 and len(part.years) == 2)
    canonical = frozen.canonical_training_specs()
    gate("training_population_exact", json.loads(json.dumps(canonical)) == prior["canonical_training_specs"])
    training_keys = {(y, r, t) for y, r, t, _, _ in canonical}
    allowed_banks = {(y, r, len(q.trajectory_ids)) for q in (runner.partition("TRAINING"), part)
                     for y, r in zip(q.years, q.roots)}
    gate("pretraining_publication_integrity", git("rev-parse", "HEAD") == head
         and git("status", "--porcelain", "--untracked-files=all") == ""
         and all(sha((ROOT / n).read_bytes()) == h for n, h in sources.items()))
    # Exclusive claim occurs only after all preflight checks; no assets above.
    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    provenance = {"HEAD": head, "predecessor_checkpoint": CHECKPOINT, "sha256": sources,
                  "git_blobs": blobs, "dependencies": versions, "predecessor_pins": PREDECESSOR_HASHES}
    # Publish provenance immediately so even a machine/implementation failure retains it.
    provenance_bytes = encode(provenance)
    write_exclusive(OUTPUT / "source_artifact_hashes.json", provenance_bytes)
    assets = runner.load_assets()
    runs, episodes, model_records, run_hashes = [], [], [], {}
    for seed in p.seeds.final:
        directory = OUTPUT / "runs" / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        model_path = directory / "final_model.zip"
        ledger = runner.AccessLedger()
        env = runner.build_scheduled_true_state_env(assets, seed, ledger, c.name)
        local = {}

        def check(name, condition):
            require(name in RUN_GATES and name not in local, f"Unknown/duplicate run gate: {name}")
            local[name] = bool(condition)
            gate(f"seed_{seed}:{name}", condition)

        try:
            check("fresh_schedule", not env.started and not env.assignments and env.total_steps == 0
                  and env.schedule.parent_seed == seed and env.schedule.ordinal == 1)
            check("fresh_substreams", runner.child_seeds(seed) == children[seed]
                  and runner.derive_frozen_rl_seeds(seed) == registered[seed])
            check("True_State_only", env.current.observation_mode == "True-State"
                  and env.observation_space.shape == (dict(p.observation_dimensions)["True-State"],))
            bundle = runner.build_ppo_model(env, c.name, seed)
            check("fresh_model", not bundle.training_finished and bundle.model.num_timesteps == 0
                  and runner.verify_model_contract(bundle.model, c.name, seed))
            training = runner.train_final_checkpoint(bundle, model_path, purpose="final")
            check("final_budget_exact", training["total_timesteps"] == env.total_steps == p.final_budget
                  and training["rollout_updates"] == p.final_budget // c.n_steps)
            check("development_budget_and_seeds_unused", seed in p.seeds.final and seed not in p.seeds.development
                  and training["total_timesteps"] != p.development_budget)
            check("final_checkpoint_only", bundle.training_finished and training["final_checkpoint_only"])
            check("no_early_stopping_best_model_callback_or_rollback", not p.selection.early_stopping
                  and not p.selection.best_model_selection and runner_review["learn_sites"] == ["train_final_checkpoint"])
            check("parameters_finite", training["model_parameter_finite"] and training["optimizer_steps_checked"] > 0
                  and runner.parameters_finite(bundle.model))
            loaded = runner.reload_final_checkpoint(model_path, bundle)
            check("save_reload_exact", loaded is not bundle.model and runner.parameters_finite(loaded)
                  and runner.file_sha256(model_path) == training["model_sha256"])
            assignments = env.assignments
            train_records = [r for r in ledger.records if r["role"] == "TRAINING"]
            check("training_roots_only", len(train_records) == len(assignments)
                  and all(runner.assignment_key(r) in training_keys for r in train_records)
                  and sum(r["steps"] for r in assignments) == p.final_budget)
            schedule = runner.build_training_schedule(seed, c.name)
            expected = [schedule.next_assignment() for _ in assignments]
            check("frozen_schedule_exact", [{k: r[k] for k in expected[0]} for r in assignments] == expected
                  and env.episodes_completed == p.final_budget // p.decision_reward_steps)
            metrics = runner.evaluate_deterministic(loaded, assets, specs, ledger)
            evaluation_records = [r for r in ledger.records if r["role"] == "DEVELOPMENT"]
            check("full_development_population", len(metrics) == len(evaluation_records) == p.evaluation.episodes_per_model
                  and [runner.assignment_key(r) for r in metrics] == eval_keys
                  and [runner.assignment_key(r) for r in evaluation_records] == eval_keys
                  and all(r["deterministic_action"] and r["step_count"] == p.decision_reward_steps
                          and r["natural_transition_count"] == p.natural_transitions for r in metrics))
            check("zero_formal_access", ledger.heldout_ppo_trajectory_access_count == ledger.denied_accesses == 0
                  and set(assets._banks) <= allowed_banks
                  and len(ledger.records) == len(train_records) + len(evaluation_records))
            check("zero_perception_access", env.perception_query_count == 0
                  and all(r["perception_query_count"] == 0 for r in assignments + metrics)
                  and assets.perception_construction_count == assets.perception_sample_count == 0 and assets.emulator is None)
            # Frozen core.step rejects every invalid training action before changing state.
            actions = {a for a, _ in p.actions}
            check("valid_actions", all(r["action_min"] in actions and r["action_max"] in actions for r in metrics))
            check("finite_observations_rewards_states", env.observations_checked == env.total_steps + len(assignments)
                  and all(r["all_observations_finite"] and r["all_rewards_finite"] and r["all_states_finite"] for r in metrics))
            check("reward_sign_regression", all(r["reward_sign_regression_diff"] <= p.selection.atol for r in metrics))
            require(set(local) == set(RUN_GATES) and all(local.values()), "Complete run audit")
            record = {"selected_config": c.name, "parent_rl_seed": seed, "learner_seed": children[seed]["learner"],
                      "schedule_seed": children[seed]["training_schedule"], "learning_rate": c.learning_rate,
                      "n_steps": c.n_steps, "total_timesteps": training["total_timesteps"],
                      "rollout_updates": training["rollout_updates"],
                      "wall_clock_training_seconds": training["wall_clock_seconds"],
                      "final_model_sha256": training["model_sha256"], "evaluation_episode_count": len(metrics),
                      **development.summarize(metrics), "formal_access_count": ledger.heldout_ppo_trajectory_access_count,
                      "perception_query_count": 0, "finite_parameters": True, "final_checkpoint_only": True}
            runs.append(record)
            episodes.extend({"selected_config": c.name, "parent_rl_seed": seed, **r} for r in metrics)
            model_records.append({"parent_rl_seed": seed, "model_path": model_path.relative_to(OUTPUT).as_posix(),
                                  "sha256": training["model_sha256"], "training_budget": training["total_timesteps"]})
            evidence = encode({"run_summary": record, "gates": local, "training_assignments": assignments,
                               "trajectory_access_log": ledger.records, "model_manifest": runner.model_manifest(bundle, training)})
            write_exclusive(directory / "run_audit.json", evidence)
            run_hashes[(directory / "run_audit.json").relative_to(OUTPUT).as_posix()] = sha(evidence)
            run_hashes[model_path.relative_to(OUTPUT).as_posix()] = training["model_sha256"]
        finally:
            env.close()
        del loaded, bundle, env

    gate("exactly_five_independent_runs", [r["parent_rl_seed"] for r in runs] == list(p.seeds.final)
         and len(runs) == 5 and all(r["selected_config"] == c.name for r in runs))
    gate("exactly_3000_evaluation_rows", len(episodes) == len(p.seeds.final) * p.evaluation.episodes_per_model == 3000)
    result = sanity_result(sanity_input(runs), p)
    seed_values = [r["mean_J_total"] for r in runs]
    gate("gate_a_identity", result["gate_a_pass"] == (sum(seed_values) / len(seed_values) <= p.evaluation.gate_a_max)
         and result["gate_a_max"] == p.evaluation.gate_a_max)
    gate("gate_b_identity", result["gate_b_pass"] == all(v < p.evaluation.periodic_mean for v in seed_values)
         and result["periodic_mean"] == p.evaluation.periodic_mean)
    # Do not abort on poor performance until every required diagnostic is published.
    gates["gate_a_PASS"], gates["gate_b_PASS"] = result["gate_a_pass"], result["gate_b_pass"]
    manifest = {"selected_config": c.name, "final_seeds": p.seeds.final, "training_budget": p.final_budget,
                "models": model_records, "sanity_pass": result["overall_sanity_pass"],
                "eligibility_requires": "PASS audit_summary.json plus matching complete output hashes; later formal audit required",
                "declaration": MODEL_DECLARATION if result["overall_sanity_pass"] else
                               "DIAGNOSTIC MODELS ONLY; NOT ELIGIBLE FOR FORMAL EVALUATION; STOP BEFORE POINT/UA",
                "no_retraining_or_seed_replacement": True}
    protocol_manifest = {"stage": STAGE, "title": TITLE, "mode": "formal", "protocol": prior["protocol_payload"],
        "selected_config_authority": {"path": str((FROZEN / "selected_config.json").relative_to(ROOT)),
                                      "sha256": PREDECESSOR_HASHES["selected_config.json"], "selected_config": c.name},
        "training_purpose": "final", "training_budget": p.final_budget, "final_seeds": p.seeds.final,
        "substreams": [registered[s] for s in p.seeds.final], "development_specs": eval_assignments,
        "observation_mode": "True-State", "deterministic_evaluation": p.evaluation.deterministic,
        "statistical_unit": p.evaluation.replication_unit, "independent_replications": p.evaluation.replication_n,
        "environment_episode_role": result["environment_episode_role"],
        "statistics_helper": PREDECESSOR + ":summarize; inherits baseline/Oracle cost_summary definition",
        "CVaR95_definition": "Descending cost; mean worst ceil(0.05*N) episodes", "std_ddof": 1,
        "quantile_method": "linear", "runner_static_review": runner_review,
        "generated_banks": [list(k) for k in assets._banks], "core_assets_provenance": assets.provenance,
        "wall_clock_scope": "PPO.learn only; excludes loading, saving and evaluation"}
    content = {"final_seed_run_summary.csv": csv_bytes(runs), "final_seed_episode_metrics.csv": csv_bytes(episodes),
               "final_models_manifest.json": encode(manifest), "sanity_gate_result.json": encode(result),
               "protocol_manifest.json": encode(protocol_manifest)}
    for name, data in content.items():
        write_exclusive(OUTPUT / name, data)
    content["source_artifact_hashes.json"] = provenance_bytes
    hashes = {**run_hashes, **{n: sha(data) for n, data in content.items()}}
    hash_index = encode({"hashes": hashes, "exclusions": {
        "output_hashes.json": "SHA256 recorded in final audit_summary or failure_summary; no self-reference",
        "audit_summary.json": "PASS-only last publication marker; pin externally after acceptance",
        "failure_summary.json": "Only on sanity failure; diagnostic marker, never a PASS audit"}})
    write_exclusive(OUTPUT / "output_hashes.json", hash_index)
    content["output_hashes.json"] = hash_index
    inventory = {x.relative_to(OUTPUT).as_posix() for x in OUTPUT.rglob("*") if x.is_file()}
    gate("output_inventory_complete", inventory == set(content) | set(run_hashes))
    for name, expected in content.items():
        actual = (OUTPUT / name).read_bytes()
        require(actual == expected, f"Exact output readback: {name}")
        if name.endswith(".json"):
            require(json.loads(actual) == json.loads(expected), f"JSON readback: {name}")
        else:
            require(csv_rows(actual) == csv_rows(expected), f"CSV readback: {name}")
    readback_runs = csv_rows((OUTPUT / "final_seed_run_summary.csv").read_bytes())
    readback_episodes = csv_rows((OUTPUT / "final_seed_episode_metrics.csv").read_bytes())
    gate("output_readback_exact", len(readback_runs) == 5 and len(readback_episodes) == 3000
         and sanity_result(sanity_input(readback_runs), p) == json.loads((OUTPUT / "sanity_gate_result.json").read_bytes()))
    # Reconstruct every reported distribution from serialized episode rows.
    for row in readback_runs:
        block = [e for e in readback_episodes if e["parent_rl_seed"] == row["parent_rl_seed"]]
        require(len(block) == p.evaluation.episodes_per_model, "Readback episode population")
        stats = development.summarize(block)
        require(all(float(row[k]) == value for k, value in stats.items()), "Run metrics reconstructed from episode CSV")
    gate("final_model_SHA256_frozen", len(model_records) == len(p.seeds.final)
         and len({m["model_path"] for m in model_records}) == len(p.seeds.final)
         and all(runner.file_sha256(OUTPUT / m["model_path"]) == m["sha256"] for m in model_records))
    index = json.loads((OUTPUT / "output_hashes.json").read_bytes())["hashes"]
    gate("output_hashes_complete", set(index) == inventory - {"output_hashes.json"}
         and index == {n: runner.file_sha256(OUTPUT / n) for n in index})
    gate("source_unchanged", all(sha((ROOT / n).read_bytes()) == h for n, h in sources.items())
         and all(git("rev-parse", f"{head}:{n}") == git("rev-parse", f":{n}")
                 == git("hash-object", f"--path={n}", n) == b for n, b in blobs.items()))
    gate("HEAD_unchanged_and_clean", git("rev-parse", "HEAD") == head
         and git("status", "--porcelain", "--untracked-files=all") == "")
    require(sum(":" in n for n in gates) == len(RUN_GATES) * len(p.seeds.final), "Complete five-run gates")
    failed = [n for n, ok in gates.items() if not ok]
    summary = {"stage": STAGE, "title": TITLE, "mode": "formal", "HEAD": head,
               "stage_pass": result["overall_sanity_pass"], "scientific_status": result["scientific_status"],
               "declaration": DECLARATION if result["overall_sanity_pass"] else "STOP BEFORE POINT/UA",
               "selected_config": c.name, "run_count": len(runs), "evaluation_episode_count": len(episodes),
               "statistical_unit": p.evaluation.replication_unit, "independent_replications": p.evaluation.replication_n,
               "heldout_ppo_trajectory_access_count": sum(r["formal_access_count"] for r in runs),
               "perception_query_count": sum(r["perception_query_count"] for r in runs),
               "gates": gates, "failed_gates": failed, "final_models": model_records,
               "output_sha256": {**hashes, "output_hashes.json": sha(hash_index)}}
    require(not (OUTPUT / "audit_summary.json").exists(), "No preexisting PASS marker")
    if not result["overall_sanity_pass"]:
        require(failed and set(failed) <= {"gate_a_PASS", "gate_b_PASS"}, "Only scientific sanity failure here")
        write_exclusive(OUTPUT / "failure_summary.json", encode(summary))
        return summary
    require(not failed and all(gates.values()), "Every provenance, run, sanity and output gate required")
    gate("audit_summary_published_last", {x.relative_to(OUTPUT).as_posix() for x in OUTPUT.rglob("*") if x.is_file()} == inventory)
    # Sole PASS marker; all five checkpoints are eligible only with this document.
    write_exclusive(OUTPUT / "audit_summary.json", encode(summary))
    require({x.relative_to(OUTPUT).as_posix() for x in OUTPUT.rglob("*") if x.is_file()}
            == inventory | {"audit_summary.json"}, "Final output inventory")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    result = run_formal()
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["stage_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
