"""P2-2C-2-v1: frozen development-only candidate selection.

Import is inert. Only explicit --mode formal invokes the eight independent runs.
An exclusive stage directory and committed unchanged source are mandatory.
Failure retains diagnostic evidence; there is no retry, resume or overwrite path.
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
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage2c2_true_state_ppo_development_selection_v1.py"
SMOKE_SOURCE = "experiments/run_paper2_stage2c1_true_state_ppo_smoke_audit_v1.py"
RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
CHECKPOINT = "687cb0836496bdc9acc274ba19c2f39ef2aae972"
SMOKE_BLOB = "7d5465ad5019a4c19b14545a016a46720ff81df9"
SMOKE_AUDIT_SHA256 = "84a2a594ecc375a822a810f01e3694d90cd2d277050933d7218e053ffafd2737"
STATISTICS_SOURCE = "experiments/run_paper2_stage2b2_heldout_clairvoyant_oracle_evaluation_v1.py"
STATISTICS_BLOB = "7995fcc20a15b6c208701a62aca0ec721e7338ce"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
SMOKE = BASE / "p2_2c_1_true_state_ppo_smoke_audit_v1/smoke"
STAGE_DIRECTORY = BASE / "p2_2c_2_true_state_ppo_development_selection_v1"
OUTPUT = STAGE_DIRECTORY / "formal"
STAGE = "P2-2C-2-v1"
STATUS = "TRUE_STATE_PPO_DEVELOPMENT_SELECTION_PASS_FROZEN"
DECLARATION = ("TRUE-STATE PPO CONFIG SELECTED ON FROZEN DEVELOPMENT POPULATION ONLY; "
               "NO FORMAL HELD-OUT PPO PERFORMANCE ACCESSED; NO RESELECTION AFTER THIS STAGE")
RUN_GATES = (
    "fresh_model", "fresh_schedule", "True_State_only", "development_budget_exact",
    "final_budget_unused", "training_roots_only", "config_independent_schedule_semantics",
    "final_checkpoint_only", "no_early_stopping", "no_best_model_selection",
    "no_performance_callback_or_rollback", "model_parameters_finite", "final_save_reload_exact",
    "development_evaluation_roots_only", "exactly_600_episodes", "zero_formal_access",
    "zero_perception_access", "all_observations_finite", "all_actions_valid",
    "all_rewards_finite", "all_states_finite", "reward_sign_regression",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
                          cwd=ROOT, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def csv_bytes(rows):
    require(rows and all(set(r) == set(rows[0]) for r in rows), "Nonempty consistent CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


def summarize(metrics):
    """Descriptive statistics only; never supplied to selector except mean_J_total.

    Same definition as pinned STATISTICS_SOURCE:cost_summary (lines 389-404)
    and baseline stage2a2:aggregate: sample std, linear P95, descending worst
    ceil(0.05*N) costs. No quantile-threshold tie expansion. Neither old stage
    is imported/executed and no Oracle performance artifact is read.
    """
    import numpy as np
    costs = np.asarray([r["J_total"] for r in metrics], dtype=float)
    cleans = np.asarray([r["N_clean"] for r in metrics], dtype=float)
    require(len(costs) > 1 and np.isfinite(costs).all() and np.isfinite(cleans).all(), "Finite metrics")
    worst = math.ceil(.05 * len(costs))
    return {"mean_J_total": float(costs.mean()), "std_J_total": float(costs.std(ddof=1)),
            "median_J_total": float(np.median(costs)),
            "P95_J_total": float(np.percentile(costs, 95, method="linear")),
            "CVaR95_J_total": float(np.sort(costs)[::-1][:worst].mean()),
            "mean_N_clean": float(cleans.mean())}


def select_candidate(seed_means, protocol):
    """Only (config_id, parent_rl_seed, mean_J_total) may cross this boundary.

    Global minimum -> absolute tolerance -> worst seed -> absolute tolerance
    -> frozen order. In particular, this is not chained pairwise isclose.
    """
    order = tuple(c.name for c in protocol.candidates)
    seeds = protocol.seeds.development
    require(order == protocol.selection.tie_breaks[1:] and protocol.selection.rtol == 0,
            "Frozen tie order and absolute-only tolerance")
    require(len(seed_means) == len(order) * len(seeds), "Exact selector population")
    scores = {}
    for row in seed_means:
        require(set(row) == {"config_id", "parent_rl_seed", "mean_J_total"}, "Selector input whitelist")
        key = row["config_id"], int(row["parent_rl_seed"])
        value = float(row["mean_J_total"])
        require(key not in scores and math.isfinite(value), "Unique finite seed score")
        scores[key] = value
    require(set(scores) == {(c, s) for c in order for s in seeds}, "Exact candidate/seed keys")
    rows = []
    for config in order:
        values = [scores[config, s] for s in seeds]
        primary = sum(values) / len(values)
        require(math.isfinite(primary), "Finite primary score")
        rows.append({"config_id": config,
                     **{f"seed_{s}_mean_J_total": scores[config, s] for s in seeds},
                     "primary_score": primary, "worst_seed_dev_mean_J_total": max(values)})
    minimum = min(r["primary_score"] for r in rows)
    for row in rows:
        row["primary_tie_survivor"] = abs(row["primary_score"] - minimum) <= protocol.selection.atol
    worst_minimum = min(r["worst_seed_dev_mean_J_total"] for r in rows if r["primary_tie_survivor"])
    for row in rows:
        row["worst_seed_tie_survivor"] = (row["primary_tie_survivor"] and
            abs(row["worst_seed_dev_mean_J_total"] - worst_minimum) <= protocol.selection.atol)
    selected = next(r["config_id"] for r in rows if r["worst_seed_tie_survivor"])
    for row in rows:
        row["selected"] = row["config_id"] == selected
    return rows


def selector_input(rows):
    return [{k: row[k] for k in ("config_id", "parent_rl_seed", "mean_J_total")} for row in rows]


def static_review(source):
    """Review orchestration boundaries; pinned runner owns all scientific execution."""
    tree = ast.parse(source)
    banned = {"learn", "predict", "PPO", "EpisodeSpec", "Paper2CleaningEnv", "Paper2EnvAssets",
              "ensure_perception", "VecNormalize", "EvalCallback", "StopTrainingOnRewardThreshold",
              "exec", "eval", "__import__", "load_frozen_module"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            require(name not in banned, f"Forbidden scientific execution: {name}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "train_final_checkpoint":
            require(len(node.keywords) == 1 and node.keywords[0].arg == "purpose"
                    and isinstance(node.keywords[0].value, ast.Constant)
                    and node.keywords[0].value.value == "development", "Development-only training")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module]
            require(all(not n.startswith("paper2") and not n.startswith("run_paper2") for n in names), "Lazy scientific imports")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for call in (x for x in ast.walk(node) if isinstance(x, ast.Call)):
                require((isinstance(call.func, ast.Name) and call.func.id == "Path") or
                        (isinstance(call.func, ast.Attribute) and call.func.attr == "resolve"), "Import-time work")
        elif isinstance(node, ast.FunctionDef):
            require(not node.decorator_list and not node.args.defaults and not any(node.args.kw_defaults), "Inert function definitions")
        elif isinstance(node, ast.Expr):
            require(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), "Only module docstring")
        else:
            require(isinstance(node, ast.If) and ast.dump(node.test) ==
                    ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body), "Only main guard")
    selector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_candidate")
    require(not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(selector)), "Pure selector")
    return True


def run_formal():
    require(not STAGE_DIRECTORY.exists(), "Stage output directory must not exist; no resume/overwrite")
    head = git("rev-parse", "HEAD")
    sources, blobs, gates = {}, {}, {}

    def gate(name, passed):
        require(name not in gates, f"Duplicate gate: {name}")
        gates[name] = bool(passed)
        require(passed, name)

    def capture(path, digest=None):
        key = Path(path).relative_to(ROOT).as_posix()
        data = Path(path).read_bytes()
        actual = sha(data)
        require(digest is None or actual == digest, f"Frozen SHA256: {key}")
        require(key not in sources or sources[key] == actual, f"Changed source: {key}")
        sources[key] = actual
        return data

    def committed(name, expected=None):
        blob = git("rev-parse", "--verify", f"{head}:{name}")
        require(blob == git("hash-object", f"--path={name}", name)
                == git("rev-parse", f":{name}"), f"Committed unchanged working tree/index: {name}")
        if expected is not None:
            require(blob == expected == git("rev-parse", f"{CHECKPOINT}:{name}"), f"Frozen blob: {name}")
        blobs[name] = blob
        capture(ROOT / name)

    committed(SELF)
    committed(SMOKE_SOURCE, SMOKE_BLOB)
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    gate("source_publication_integrity", True)
    # Validated inert predecessor supplies its frozen P2-2C-0 SHA256 pins and IO helpers.
    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as smoke
    frozen_docs = {n: capture(smoke.FROZEN / n, h) for n, h in smoke.HASHES.items()}
    previous = json.loads(frozen_docs["audit_summary.json"])
    gate("P2_2C_0_PASS_FROZEN", previous.get("stage_pass") is True and previous.get("failed_gates") == []
         and previous.get("scientific_status") == "TRUE_STATE_PPO_PROTOCOL_PASS_FROZEN")
    smoke_audit = json.loads(capture(SMOKE / "audit_summary.json", SMOKE_AUDIT_SHA256))
    gate("P2_2C_1_smoke_PASS", smoke_audit.get("stage_pass") is True and smoke_audit.get("failed_gates") == []
         and smoke_audit.get("scientific_status") == "TRUE_STATE_PPO_SMOKE_IMPLEMENTATION_PASS")
    smoke_docs = {n: capture(SMOKE / n, h) for n, h in smoke_audit["output_sha256"].items()}
    smoke_provenance = json.loads(smoke_docs["source_artifact_hashes.json"])
    require(smoke_provenance["HEAD"] == smoke_audit["HEAD"] == CHECKPOINT, "Smoke checkpoint exact")
    for name, digest in smoke_provenance["sha256"].items():
        capture(ROOT / name, digest)
    for name, blob in smoke_provenance["git_blobs"].items():
        committed(name, blob)
    for record in smoke_provenance["core_assets_provenance"]["helpers"].values():
        committed(record["path"], record["git_blob"])
    reward_record = smoke_provenance["core_assets_provenance"]["reward_script"]
    committed(reward_record["path"], reward_record["git_blob"])
    committed(STATISTICS_SOURCE, STATISTICS_BLOB)
    gate("predecessor_provenance_exact", True)
    for name, path in (("runner", RUNNER), ("protocol", PROTOCOL), ("environment", CORE)):
        gate(f"frozen_{name}_blob_exact", blobs[path] == smoke_provenance["git_blobs"][path])
    frozen_payload = json.loads(frozen_docs["ppo_protocol.json"])
    versions = smoke.dependency_versions()
    gate("dependency_versions", versions == dict(frozen_payload["dependencies"]))
    gate("static_orchestration_boundaries", static_review((ROOT / SELF).read_text(encoding="utf-8")))
    runner_static = smoke.static_review({n: (ROOT / n).read_text(encoding="utf-8") for n in (RUNNER, SMOKE_SOURCE)})

    import paper2_ppo_protocol_v1 as frozen
    import paper2_true_state_ppo_runner_v1 as runner
    p = runner.get_protocol()
    gate("exact_frozen_protocol_reproduction", frozen.protocol_payload(p) == frozen_payload)
    gate("exactly_4_candidates", len(p.candidates) == 4)
    gate("exactly_2_development_seeds", len(p.seeds.development) == 2
         and not set(p.seeds.development) & set(p.seeds.final))
    development = runner.partition("DEVELOPMENT")
    specs = tuple((y, t) for y in development.years for t in development.trajectory_ids)
    eval_assignments = [runner.development_assignment(y, t) for y, t in specs]
    eval_keys = [runner.assignment_key(r) for r in eval_assignments]
    gate("frozen_development_population", len(specs) == len(set(specs)) == development.population_size
         == p.evaluation.episodes_per_model == 600 and len(development.years) == 2)
    canonical = frozen.canonical_training_specs()
    require(json.loads(json.dumps(canonical)) == json.loads(frozen_docs["training_schedule_contract.json"])["canonical_specs"],
            "Frozen training population identity")
    training_keys = {(y, root, tid) for y, root, tid, _, _ in canonical}
    # No assets or trajectory construction above this point.
    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    assets = runner.load_assets()
    runs, episodes, evidence_hashes, schedule_prefixes = [], [], {}, {}
    for config in p.candidates:
        c = runner.candidate(config.name)
        for seed in p.seeds.development:
            label = f"{c.name}/seed_{seed}"
            directory = OUTPUT / "runs" / c.name / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            model_path = directory / "final_model.zip"
            ledger = runner.AccessLedger()
            env = runner.build_scheduled_true_state_env(assets, seed, ledger, c.name)
            local = {}

            def check(name, passed):
                require(name in RUN_GATES and name not in local, f"Unknown/duplicate run gate: {name}")
                local[name] = bool(passed)
                gate(f"{label}:{name}", passed)

            try:
                check("fresh_schedule", not env.started and env.total_steps == 0 and env.assignments == []
                      and env.schedule.ordinal == 1 and env.schedule.parent_seed == seed)
                check("True_State_only", env.current.observation_mode == "True-State"
                      and env.observation_space.shape == (dict(p.observation_dimensions)["True-State"],))
                bundle = runner.build_ppo_model(env, c.name, seed)
                check("fresh_model", not bundle.training_finished and bundle.model.num_timesteps == 0
                      and runner.verify_model_contract(bundle.model, c.name, seed))
                training = runner.train_final_checkpoint(bundle, model_path, purpose="development")
                check("development_budget_exact", training["total_timesteps"] == env.total_steps
                      == p.development_budget == p.selection.development_checkpoint
                      and training["rollout_updates"] == p.development_budget // c.n_steps
                      and p.development_budget % c.n_steps == 0)
                check("final_budget_unused", seed not in p.seeds.final
                      and training["total_timesteps"] != p.final_budget)
                check("final_checkpoint_only", training["final_checkpoint_only"] and bundle.training_finished)
                check("no_early_stopping", not p.selection.early_stopping and env.total_steps == p.development_budget)
                check("no_best_model_selection", not p.selection.best_model_selection and runner_static["AST_checked"])
                check("no_performance_callback_or_rollback", runner_static["learn_sites"] == ["train_final_checkpoint"])
                check("model_parameters_finite", training["model_parameter_finite"]
                      and training["optimizer_steps_checked"] > 0 and runner.parameters_finite(bundle.model))
                loaded = runner.reload_final_checkpoint(model_path, bundle)
                check("final_save_reload_exact", loaded is not bundle.model and runner.parameters_finite(loaded)
                      and runner.file_sha256(model_path) == training["model_sha256"])
                assignments = env.assignments
                train_records = [r for r in ledger.records if r["role"] == "TRAINING"]
                check("training_roots_only", len(train_records) == len(assignments)
                      and all(runner.assignment_key(r) in training_keys for r in train_records)
                      and sum(r["steps"] for r in assignments) == p.development_budget)
                schedule = runner.build_training_schedule(seed, c.name)
                expected = [schedule.next_assignment() for _ in assignments]
                actual = [{k: r[k] for k in expected[0]} for r in assignments]
                reference = schedule_prefixes.setdefault(seed, expected)
                check("config_independent_schedule_semantics", actual == expected == reference
                      and env.episodes_completed == p.development_budget // p.decision_reward_steps)
                metrics = runner.evaluate_deterministic(loaded, assets, specs, ledger)
                eval_records = [r for r in ledger.records if r["role"] == "DEVELOPMENT"]
                check("development_evaluation_roots_only", [runner.assignment_key(r) for r in metrics] == eval_keys
                      and [runner.assignment_key(r) for r in eval_records] == eval_keys)
                check("exactly_600_episodes", len(metrics) == len(eval_records) == p.evaluation.episodes_per_model == 600
                      and all(r["step_count"] == p.decision_reward_steps and r["natural_transition_count"] == p.natural_transitions
                              and r["deterministic_action"] for r in metrics))
                allowed_banks = {(y, r, len(part.trajectory_ids)) for part in (runner.partition("TRAINING"), development)
                                 for y, r in zip(part.years, part.roots)}
                check("zero_formal_access", ledger.heldout_ppo_trajectory_access_count == ledger.denied_accesses == 0
                      and set(assets._banks) <= allowed_banks and len(ledger.records) == len(train_records) + len(eval_records))
                check("zero_perception_access", env.perception_query_count == 0
                      and all(r["perception_query_count"] == 0 for r in assignments + metrics)
                      and assets.perception_construction_count == assets.perception_sample_count == 0 and assets.emulator is None)
                for field in ("observations", "rewards", "states"):
                    check(f"all_{field}_finite", env.observations_checked == env.total_steps + len(assignments)
                          and all(r[f"all_{field}_finite"] for r in metrics))
                # Frozen core.step rejects invalid training actions before any state mutation.
                valid_actions = {a for a, _ in p.actions}
                check("all_actions_valid", all(r["action_min"] in valid_actions and r["action_max"] in valid_actions for r in metrics))
                check("reward_sign_regression", all(r["reward_sign_regression_diff"] <= p.selection.atol for r in metrics))
                require(set(local) == set(RUN_GATES) and all(local.values()), "Complete run gates")
                children = runner.child_seeds(seed)
                record = {"config_id": c.name, "parent_rl_seed": seed, "learner_seed": children["learner"],
                          "schedule_seed": children["training_schedule"], "learning_rate": c.learning_rate,
                          "n_steps": c.n_steps, "total_timesteps": training["total_timesteps"],
                          "rollout_updates": training["rollout_updates"],
                          "wall_clock_training_seconds": training["wall_clock_seconds"],
                          "model_sha256": training["model_sha256"], "evaluation_episode_count": len(metrics),
                          **summarize(metrics), "formal_access_count": ledger.heldout_ppo_trajectory_access_count,
                          "perception_query_count": 0, "finite_parameters": True, "final_checkpoint_only": True}
                runs.append(record)
                episodes.extend({"config_id": c.name, "parent_rl_seed": seed, **r} for r in metrics)
                evidence = encode({"run_summary": record, "gates": local, "training_assignments": assignments,
                                   "trajectory_access_log": ledger.records, "model_manifest": runner.model_manifest(bundle, training)})
                with (directory / "run_audit.json").open("xb") as stream:
                    stream.write(evidence)
                require((directory / "run_audit.json").read_bytes() == evidence, "Run evidence readback")
                evidence_hashes[(directory / "run_audit.json").relative_to(OUTPUT).as_posix()] = sha(evidence)
                evidence_hashes[model_path.relative_to(OUTPUT).as_posix()] = training["model_sha256"]
            finally:
                env.close()
            del loaded, bundle, env

    gate("exactly_8_runs", len(runs) == len(p.candidates) * len(p.seeds.development) == 8)
    gate("evaluation_rows_exactly_4800", len(episodes) == 4800)
    selection = select_candidate(selector_input(runs), p)
    winner = next(r for r in selection if r["selected"])
    # Independent audit of score identities and survivor sets, using only seed means.
    for row in selection:
        values = [next(r["mean_J_total"] for r in runs if r["config_id"] == row["config_id"]
                       and r["parent_rl_seed"] == s) for s in p.seeds.development]
        require(row["primary_score"] == sum(values) / len(values), "Primary score identity")
        require(row["worst_seed_dev_mean_J_total"] == max(values), "Worst-seed score identity")
    gate("primary_score_identity", True)
    gate("worst_seed_score_identity", True)
    primary_min = min(r["primary_score"] for r in selection)
    first = [r for r in selection if abs(r["primary_score"] - primary_min) <= p.selection.atol]
    worst_min = min(r["worst_seed_dev_mean_J_total"] for r in first)
    second = [r for r in first if abs(r["worst_seed_dev_mean_J_total"] - worst_min) <= p.selection.atol]
    gate("tie_rule_identity", winner["config_id"] == second[0]["config_id"]
         and all(r["primary_tie_survivor"] == (r in first) and r["worst_seed_tie_survivor"] == (r in second) for r in selection))
    gate("selection_metric_identity", all(summarize([e for e in episodes if e["config_id"] == r["config_id"]
         and e["parent_rl_seed"] == r["parent_rl_seed"]])["mean_J_total"] == r["mean_J_total"] for r in runs))
    gate("prohibited_metrics_not_used_by_selector", all(set(r) == {"config_id", "parent_rl_seed", "mean_J_total"}
                                                       for r in selector_input(runs)))
    selected = {"selected_config": winner["config_id"], "selected_primary_score": winner["primary_score"],
                "selected_worst_seed_score": winner["worst_seed_dev_mean_J_total"],
                "selection_metric_definition": p.selection.primary_metric,
                "tie_procedure": {"algorithm": p.selection.tie_algorithm, "atol": p.selection.atol,
                                  "rtol": p.selection.rtol, "tie_breaks": p.selection.tie_breaks},
                "candidate_order": [c.name for c in p.candidates], "development_seeds": p.seeds.development,
                "development_budget": p.development_budget, "declaration": "NO HYPERPARAMETER RESELECTION AFTER P2-2C-2"}
    documents = {
        "selected_config.json": selected,
        "protocol_manifest.json": {"stage": STAGE, "mode": "formal", "declaration": DECLARATION,
            "protocol": frozen_payload, "frozen_protocol_sha256": smoke.HASHES["ppo_protocol.json"],
            "development_specs": eval_assignments, "training_purpose": "development",
            "observation_mode": "True-State", "deterministic_evaluation": p.evaluation.deterministic,
            "statistics_source": STATISTICS_SOURCE + ":cost_summary", "std_ddof": 1, "quantile_method": "linear",
            "CVaR95_definition": "Descending cost; mean worst ceil(0.05*N) episodes",
            "generated_banks": [list(k) for k in assets._banks], "runner_static_review": runner_static},
        "source_artifact_hashes.json": {"HEAD": head, "predecessor_checkpoint": CHECKPOINT,
            "sha256": sources, "git_blobs": blobs, "dependencies": versions, "core_assets_provenance": assets.provenance},
    }
    content = {n: encode(d) for n, d in documents.items()}
    content.update({"development_run_summary.csv": csv_bytes(runs), "development_episode_metrics.csv": csv_bytes(episodes),
                    "candidate_selection_summary.csv": csv_bytes(selection)})
    hashes = {**evidence_hashes, **{n: sha(data) for n, data in content.items()}}
    content["output_hashes.json"] = encode({"hashes": hashes, "exclusions": {
        "output_hashes.json": "SHA256 in audit_summary; avoid self-reference",
        "audit_summary.json": "last publication marker; pin externally after acceptance"}})
    for name, data in content.items():
        with (OUTPUT / name).open("xb") as stream:
            stream.write(data)
    inventory = {x.relative_to(OUTPUT).as_posix() for x in OUTPUT.rglob("*") if x.is_file()}
    gate("output_inventory_complete", inventory == set(content) | set(evidence_hashes))
    for name, data in content.items():
        actual = (OUTPUT / name).read_bytes()
        require(actual == data, f"Exact output readback: {name}")
        if name.endswith(".json"):
            require(json.loads(actual) == json.loads(data), f"JSON readback: {name}")
        else:
            require(csv_rows(actual) == csv_rows(data), f"CSV readback: {name}")
    readback_runs = csv_rows((OUTPUT / "development_run_summary.csv").read_bytes())
    gate("selection_reproducible_from_frozen_run_summary", select_candidate(selector_input(readback_runs), p) == selection)
    gate("output_readback_exact", len(readback_runs) == 8
         and len(csv_rows((OUTPUT / "development_episode_metrics.csv").read_bytes())) == 4800
         and len(csv_rows((OUTPUT / "candidate_selection_summary.csv").read_bytes())) == 4)
    index = json.loads((OUTPUT / "output_hashes.json").read_bytes())["hashes"]
    gate("output_hash_completeness", set(index) == inventory - {"output_hashes.json"}
         and index == {n: runner.file_sha256(OUTPUT / n) for n in index})
    gate("source_unchanged", all(sha((ROOT / n).read_bytes()) == h for n, h in sources.items()))
    for name, blob in blobs.items():
        require(git("rev-parse", f"{head}:{name}") == git("hash-object", f"--path={name}", name)
                == git("rev-parse", f":{name}") == blob, f"Publication source integrity: {name}")
    gate("HEAD_unchanged", git("rev-parse", "HEAD") == head)
    gate("audit_summary_published_last", not (OUTPUT / "audit_summary.json").exists()
         and inventory == set(content) | set(evidence_hashes))
    require(all(gates.values()) and sum(":" in name for name in gates) == len(RUN_GATES) * len(runs), "All gates required")
    summary = {"stage": STAGE, "mode": "formal", "stage_pass": True, "scientific_status": STATUS,
               "declaration": DECLARATION, "HEAD": head, "gates": gates, "failed_gates": [],
               "selected_config": winner["config_id"], "run_count": len(runs), "evaluation_episode_count": len(episodes),
               "heldout_ppo_trajectory_access_count": sum(r["formal_access_count"] for r in runs),
               "perception_query_count": sum(r["perception_query_count"] for r in runs),
               "output_sha256": {**hashes, "output_hashes.json": sha(content["output_hashes.json"])}}
    # Last and sole completion marker: never created on an earlier gate failure.
    data = encode(summary)
    with (OUTPUT / "audit_summary.json").open("xb") as stream:
        stream.write(data)
    require((OUTPUT / "audit_summary.json").read_bytes() == data, "Final marker readback")
    require({x.relative_to(OUTPUT).as_posix() for x in OUTPUT.rglob("*") if x.is_file()}
            == inventory | {"audit_summary.json"}, "Final inventory")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2))


if __name__ == "__main__":
    main()
