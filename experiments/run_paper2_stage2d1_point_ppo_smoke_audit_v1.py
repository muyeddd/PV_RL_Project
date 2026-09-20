"""P2-2D-1-v1: Point-PPO Perception-Aware Runner and Smoke Audit.

ENGINEERING SMOKE ONLY. Import is inert; only --mode smoke executes. No
performance selection, UA training, warm start, retry or formal evaluation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage2d1_point_ppo_smoke_audit_v1.py"
BRIDGE = "experiments/paper2_perception_ppo_runner_v1.py"
PREDECESSOR = "experiments/run_paper2_stage2c3_true_state_ppo_final_seed_sanity_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
CHECKPOINT = "7183335a4792735859488d2061d4c398b9fc0464"
PREDECESSOR_BLOB = "3b760c593b9a1a144efdaf725f301b6ee174bd75"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
FROZEN = BASE / "p2_2c_3_true_state_ppo_final_seed_sanity_v1/formal"
STAGE_DIRECTORY = BASE / "p2_2d_1_point_ppo_smoke_audit_v1"
OUTPUT = STAGE_DIRECTORY / "smoke"
PINS = {
    "audit_summary.json": "355a51634c7d99fc590d671dfb547aff4bc0e94a1eae53d9aa0fdebbf7e549d4",
    "sanity_gate_result.json": "4705bf2cb9751b8289c2281ebf0e66dcffda96f0f121f272bacbb67c60101cf7",
    "final_models_manifest.json": "418dd790f0e2b9d878fc469236268b6c8c6a5c77964ceae13065b6177ddef65f",
    "final_seed_run_summary.csv": "3e873356353458b1086afb00b0d55517f7c3ccf5b1067bcea2499a164a67b3ee",
    "protocol_manifest.json": "3a458becb3d75661ade478777c65f71797bec8db2fb7828e32c9e94f95f8e6d4",
    "source_artifact_hashes.json": "a8118cc3c64dcc1cbaf5c58f9d401604d095505dc38ec0ea27da331412524a0e",
    "output_hashes.json": "f8877ea07814479e0a3e6e8aca5b5474b9b0753bfbeff3a5091b74351c5ad51c",
}
STAGE = "P2-2D-1-v1"
STATUS = "POINT_PPO_SMOKE_IMPLEMENTATION_PASS"
DECLARATION = ("POINT-PPO PERCEPTION-AWARE RUNNER ENGINEERING SMOKE PASS; CONFIG_A AND FROZEN DEVELOPMENT "
               "PERCEPTION CONTRACT INHERITED; NO POINT PERFORMANCE CLAIM; NO UA TRAINING; NO FORMAL HELD-OUT ACCESS")
COMPATIBILITY_REASON = ("Frozen True-State runner intentionally contains: exact ScheduledTrueStateEnv guard; "
    "True-State-only construct_core; no-perception audit. This runner is provenance evidence for P2-2C and is "
    "therefore not generalized in place. Point/UA use a new compatibility layer consuming the same frozen PPO protocol.")
CONSTRUCTION_DECLARATION = ("NO NEW PPO HYPERPARAMETER OR TRAINING RULE INTRODUCED; "
                           "ONLY THE OBSERVATION-MODE-COMPATIBILITY CONSTRUCTOR IS NEW.")
REUSED = ("get_protocol", "candidate", "partition", "derive_frozen_rl_seeds", "child_seeds", "build_training_schedule",
          "verify_model_contract", "parameters_finite", "train_final_checkpoint", "reload_final_checkpoint",
          "file_sha256", "model_manifest", "ModelBundle", "load_assets")


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
    require(rows and all(set(r) == set(rows[0]) for r in rows), "Nonempty consistent CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_exclusive(path, data):
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, f"Exact readback: {path.name}")


def capture(sources, path, digest):
    path = Path(path).resolve()
    data = path.read_bytes()
    actual = sha(data)
    require(digest is None or actual == digest, f"Pinned source/artifact: {path.name}")
    key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    require(key not in sources or sources[key] == actual, f"Source changed: {key}")
    sources[key] = actual
    return data


def read_predecessors(head):
    """Read-only frozen evidence; archives hashed, never loaded as models."""
    sources = {}
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    require(git("rev-parse", f"{head}:{PREDECESSOR}") == git("hash-object", f"--path={PREDECESSOR}", PREDECESSOR)
            == PREDECESSOR_BLOB, "Frozen P2-2C-3 source")
    docs = {n: capture(sources, FROZEN / n, h) for n, h in PINS.items()}
    audit = json.loads(docs["audit_summary.json"])
    require(audit.get("stage_pass") is True and audit.get("failed_gates") == []
            and audit.get("scientific_status") == "TRUE_STATE_PPO_FINAL_SEED_SANITY_PASS_FROZEN"
            and audit.get("HEAD") == CHECKPOINT and audit.get("selected_config") == "CONFIG_A"
            and audit.get("run_count") == 5 and audit.get("evaluation_episode_count") == 3000
            and audit.get("heldout_ppo_trajectory_access_count") == audit.get("perception_query_count") == 0
            and audit.get("gates") and all(v is True for v in audit["gates"].values()), "Accepted True-State sanity predecessor")
    for name, digest in audit["output_sha256"].items():
        path = (FROZEN / name).resolve()
        require(path.is_relative_to(FROZEN), "Predecessor path containment")
        capture(sources, path, digest)
    require(json.loads(docs["output_hashes.json"])["hashes"] ==
            {n: h for n, h in audit["output_sha256"].items() if n != "output_hashes.json"}, "Predecessor output hash inventory")
    provenance = json.loads(docs["source_artifact_hashes.json"])
    require(provenance["HEAD"] == CHECKPOINT, "Predecessor publication HEAD")
    for name, digest in provenance["sha256"].items():
        capture(sources, ROOT / name, digest)
    blobs = provenance["git_blobs"]
    for name, blob in blobs.items():
        require(git("rev-parse", f"{CHECKPOINT}:{name}") == git("rev-parse", f"{head}:{name}")
                == git("rev-parse", f":{name}") == git("hash-object", f"--path={name}", name) == blob, "Frozen source blob: " + name)
    import run_paper2_stage2c3_true_state_ppo_final_seed_sanity_v1 as previous
    inherited = previous.read_predecessors(head)
    for name, digest in inherited["sources"].items():
        capture(sources, ROOT / name, digest)
    models = json.loads(docs["final_models_manifest.json"])
    sanity = json.loads(docs["sanity_gate_result.json"])
    require(models["selected_config"] == audit["selected_config"] == inherited["selected_config"]
            and models["sanity_pass"] is True and sanity["overall_sanity_pass"] is True
            and sanity["gate_a_pass"] is True and sanity["gate_b_pass"] is True
            and len(models["models"]) == 5, "Inherited frozen configuration and five-model evidence")
    require(json.loads(docs["protocol_manifest.json"])["protocol"] == inherited["protocol_payload"], "Protocol inheritance identity")
    return {"sources": sources, "blobs": blobs, "selected_config": inherited["selected_config"],
            "protocol": inherited["protocol_payload"], "predecessor_gates": inherited["gates"]}


def static_review(bridge_source, stage_source, true_source):
    """AST equivalence plus direct-alias and no-bypass checks; no model execution."""
    bridge, stage, frozen = (ast.parse(s) for s in (bridge_source, stage_source, true_source))

    def function(tree, name):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def construction_body(func):
        start = next(i for i, n in enumerate(func.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Tuple)
                     and [x.id for x in n.targets[0].elts] == ["common", "architecture"])
        return ast.dump(ast.Module(body=func.body[start:-1], type_ignores=[]))

    original = construction_body(function(frozen, "build_ppo_model"))
    require(original == construction_body(function(bridge, "build_perception_ppo_model")),
            "PPO constructor body, seeds, thread count, kwargs and defaults must be structurally identical")
    aliases = {n.targets[0].id: n.value.attr for n in bridge.body if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name) and n.value.value.id == "true_runner"}
    require(all(aliases.get(n) == n for n in REUSED), "Direct frozen API aliases")
    for name in ("train_final_checkpoint", "reload_final_checkpoint", "verify_model_contract"):
        func = function(frozen, name)
        require(not any(isinstance(n, ast.Name) and n.id in {"ScheduledTrueStateEnv", "audit_observation", "construct_core"}
                        for n in ast.walk(func)), "No hidden True-State restriction in reused API")
    bundle = next(n for n in frozen.body if isinstance(n, ast.ClassDef) and n.name == "ModelBundle")
    require(all(isinstance(n, ast.AnnAssign) for n in bundle.body), "Frozen bundle has annotations only; no runtime type enforcement")
    banned = {"set_env", "__class__", "setattr", "exec", "eval", "__import__", "learn", "save", "load",
              "sample_one", "perception_substream_seed", "dry_next_samples", "rain_next_samples", "settlement",
              "select_candidate", "VecNormalize", "EvalCallback", "build_ppo_model"}
    for tree in (bridge, stage):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                require(name not in banned and not name.startswith("StopTraining"), "Forbidden bypass/scientific duplication: " + name)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                require(not (isinstance(node.value, ast.Name) and node.value.id in ("true_runner", "core", "runner")), "No frozen-module mutation")
        # Import-time calls limited to path metadata, inert decorators/dataclass fields.
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                    require((isinstance(call.func, ast.Name) and call.func.id == "Path") or
                            (isinstance(call.func, ast.Attribute) and call.func.attr == "resolve"), "No import-time runtime work")
            elif isinstance(node, ast.Expr):
                require(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), "Only module docstring")
            elif isinstance(node, ast.If):
                require(tree is stage and ast.dump(node.test) == ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body), "Only main guard")
            else:
                require(isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)), "Inert declarations")
    training = [n for n in ast.walk(stage) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "train_final_checkpoint"]
    require(len(training) == 1 and len(training[0].keywords) == 1 and training[0].keywords[0].arg == "purpose"
            and isinstance(training[0].keywords[0].value, ast.Constant) and training[0].keywords[0].value.value == "smoke", "Smoke training only")
    scheduled_calls = [n for n in ast.walk(stage) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "build_scheduled_perception_env"]
    require(len(scheduled_calls) == 1 and isinstance(scheduled_calls[0].args[-1], ast.Constant)
            and scheduled_calls[0].args[-1].value == "Point", "No UA training entry point in this stage")
    constructor = function(bridge, "construct_perception_core")
    require(isinstance(constructor.body[0], ast.Expr) and isinstance(constructor.body[0].value, ast.Call)
            and isinstance(constructor.body[0].value.func, ast.Attribute)
            and constructor.body[0].value.func.attr == "authorize", "Ledger authorizes before EpisodeSpec")
    # Same prediction call and output metric mapping as frozen True-State evaluation.
    old_eval = function(frozen, "evaluate_deterministic")
    new_eval = function(bridge, "evaluate_perception_deterministic")
    for attr in ("predict", "append"):
        def calls(func):
            return [ast.dump(n) for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == attr]
        require(calls(old_eval) == calls(new_eval), "Frozen deterministic prediction/accumulation/metrics identity")
    # Stronger comparison: after removing only the new perception query assertion
    # and adapting the two mode-specific calls, the entire shell must be identical.
    class NormalizePerceptionShell(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "build_perception_eval_env":
                node.func.id = "build_true_state_eval_env"
                node.args.pop()
            elif isinstance(node.func, ast.Name) and node.func.id == "audit_perception_observation":
                node.func.id = "audit_observation"
                node.args.pop()
            return node

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and any(isinstance(n, ast.Constant)
                    and n.value == "One perception query per decision observation" for n in ast.walk(node)):
                return None
            return self.generic_visit(node)

    normalized = NormalizePerceptionShell().visit(ast.parse(ast.unparse(new_eval)).body[0])
    normalized.name, normalized.args, normalized.body[0] = old_eval.name, old_eval.args, old_eval.body[0]
    require(ast.dump(normalized) == ast.dump(old_eval), "Full mechanically mirrored evaluation shell identity")
    return {"PPO_CONSTRUCTION_EQUIVALENCE_GATE": True, "constructor_AST_sha256": sha(original.encode()),
            "direct_frozen_API_reuse": list(REUSED), "evaluation_shell_identity": True,
            "no_frozen_mutation": True, "no_new_PPO_parameters": True}


def pairing_regression(runner, assets, ledger):
    """Tiny fixed external WAIT/CLEAN trace, no model/policy calls or UA training."""
    p = runner.get_protocol()
    part = runner.partition("DEVELOPMENT")
    year, tid = part.years[0], part.trajectory_ids[0]
    trace = [action for action, _ in p.actions]
    point = runner.build_perception_eval_env(assets, year, tid, ledger, "Point")
    try:
        ua = runner.build_perception_eval_env(assets, year, tid, ledger, "UA")
        try:
            point_obs, point_info = point.reset()
            before = assets.perception_sample_count
            ua_obs, ua_info = ua.reset()
            require(assets.perception_sample_count == before, "UA reset reused Point sample")
            point_fields, ua_fields = dict(p.observations)["Point"], dict(p.observations)["UA"]
            checks = []
            for index in range(len(trace) + 1):
                a = runner.audit_perception_observation(point, point_obs, point_info)
                b = runner.audit_perception_observation(ua, ua_obs, ua_info)
                require(a == b, "Paired hidden dynamics, settlement, query count and environment identity")
                require(runner.np.array_equal(point_obs, ua_obs[[ua_fields.index(f) for f in point_fields]])
                        and runner.np.isfinite(ua_obs[ua_fields.index("width")]), "Shared q50/season; UA adds only finite width")
                checks.append({"day_index": a["day_index"], "hidden_state_exact": True, "q50_season_exact": True,
                               "UA_width_finite": True, "sample_reused": True})
                if index < len(trace):
                    point_result = point.step(trace[index])
                    before = assets.perception_sample_count
                    ua_result = ua.step(trace[index])
                    require(assets.perception_sample_count == before, "UA step reused Point record; no independent draw")
                    require(point_result[1:4] == ua_result[1:4] and not point_result[2] and not point_result[3], "Paired rewards and terminal flags exact")
                    point_obs, point_info = point_result[0], point_result[4]
                    ua_obs, ua_info = ua_result[0], ua_result[4]
            return {"passed": True, "year": year, "trajectory_id": tid, "environment_root": point.episode_spec.environment_root,
                    "perception_seed": point.episode_spec.perception_seed, "fixed_external_action_trace": trace,
                    "checks": checks, "perception_queries_per_mode": a["perception_query_count"],
                    "role": "PARTIAL-EPISODE INTEGRATION ONLY; NO UA TRAINING OR PERFORMANCE CLAIM",
                    "same_assets_cache": point._assets is ua._assets, "policy_name_in_substream": False}
        finally:
            ua.close()
    finally:
        point.close()


def run_smoke():
    require(not STAGE_DIRECTORY.exists(), "Exclusive stage directory; no overwrite/resume/retry")
    require(git("status", "--porcelain", "--untracked-files=all") == "", "Committed unchanged clean sources required")
    head = git("rev-parse", "HEAD")
    prior = read_predecessors(head)
    sources, blobs = prior["sources"], prior["blobs"]
    for name in (SELF, BRIDGE):
        blob = git("rev-parse", "--verify", f"{head}:{name}")
        require(blob == git("hash-object", f"--path={name}", name) == git("rev-parse", f":{name}"), "Both new files committed unchanged")
        blobs[name] = blob
        capture(sources, ROOT / name, None)
    static = static_review((ROOT / BRIDGE).read_text(encoding="utf-8"), (ROOT / SELF).read_text(encoding="utf-8"),
                           (ROOT / TRUE_RUNNER).read_text(encoding="utf-8"))
    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as smoke
    versions = smoke.dependency_versions()
    require(versions == dict(prior["protocol"]["dependencies"]), "Frozen runtime dependencies")
    import paper2_perception_ppo_runner_v1 as runner
    import paper2_ppo_protocol_v1 as frozen
    p = runner.get_protocol()
    require(frozen.protocol_payload(p) == prior["protocol"], "Exact frozen protocol reproduction")
    authority = runner.read_perception_authority()
    for name, digest in runner.AUTHORITY_HASHES.items():
        capture(sources, runner.AUTHORITY / name, digest)
    for name, record in authority["current_files"].items():
        capture(sources, ROOT / name, record["sha256"])
        require(git("rev-parse", f"{head}:{name}") == git("hash-object", f"--path={name}", name)
                == git("rev-parse", f":{name}") == record["git_blob"], "Accepted P2-1E-1 source unchanged")
        blobs[name] = record["git_blob"]
    require(runner.get_perception_contract().development_seed == authority["perception_seed"], "Mechanical DEV seed authority")
    require(runner.inherited_config_id() == prior["selected_config"] == p.candidates[0].name, "Inherited config and frozen smoke contract")
    capture(sources, runner.SELECTED_ARTIFACT, runner.SELECTED_SHA256)
    require(git("rev-parse", "HEAD") == head and all(sha((ROOT / n).read_bytes()) == h for n, h in sources.items()), "Pre-execution source integrity")
    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    try:
        return execute_smoke(runner, prior, head, versions, static, authority)
    except runner.core.PerceptionSupportError as exc:
        # Diagnostic evidence only. Preserve exact exception; never retry/resample.
        write_exclusive(OUTPUT / "perception_support_failure.json", encode({"stage_pass": False,
            "scientific_status": "POINT_PPO_SMOKE_PERCEPTION_SUPPORT_FAIL", "context": exc.context,
            "error": str(exc), "HEAD": head, "source_sha256": sources,
            "action": "STOP; NO RETRY, RESAMPLING, FALLBACK OR PERCEPTION PARAMETER CHANGE"}))
        raise


def execute_smoke(runner, prior, head, versions, static, authority):
    sources, blobs, gates = prior["sources"], prior["blobs"], {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate")
        gates[name] = bool(condition)
        require(condition, name)

    p = runner.get_protocol()
    c, seed = runner.candidate(prior["selected_config"]), p.seeds.development[0]
    gate("predecessor_and_authority_provenance_exact", all(prior["predecessor_gates"].values()))
    gate("direct_frozen_train_reload_bundle", runner.train_final_checkpoint is runner.true_runner.train_final_checkpoint
         and runner.reload_final_checkpoint is runner.true_runner.reload_final_checkpoint and runner.ModelBundle is runner.true_runner.ModelBundle)
    assets = runner.load_assets()
    # Pin external perception inputs before constructing any perception environment.
    for name, path in (("paper1_cqr", assets.paper1_cqr), ("wapp_power_bridge", assets.wapp_power_bridge_path)):
        capture(sources, path, assets.perception.EXPECTED_HASHES[name])
    ledger = runner.AccessLedger()
    pairing = pairing_regression(runner, assets, ledger)
    gate("Point_UA_shared_perception_pairing", pairing["passed"] and pairing["same_assets_cache"])
    env = runner.build_scheduled_perception_env(assets, seed, ledger, c.name, "Point")
    try:
        gate("fresh_Point_model_schedule_inputs", env.observation_mode == "Point" and not env.started
             and not env.assignments and env.schedule.parent_seed == seed and env.schedule.ordinal == 1)
        bundle = runner.build_perception_ppo_model(env, c.name, seed)
        gate("PPO_CONSTRUCTION_EQUIVALENCE_GATE", static["PPO_CONSTRUCTION_EQUIVALENCE_GATE"]
             and runner.verify_model_contract(bundle.model, c.name, seed) and c.name == prior["selected_config"]
             and not p.observation_normalization and not p.reward_normalization
             and bundle.model.num_timesteps == 0 and not bundle.training_finished)
        construction = {"gate_pass": True, "declaration": CONSTRUCTION_DECLARATION,
            "compatibility_reason": COMPATIBILITY_REASON, "config_id": c.name,
            "learning_rate": c.learning_rate, "n_steps": c.n_steps, "common": dict(p.common),
            "architecture": dict(p.architecture), "learner_seed": runner.child_seeds(seed)["learner"],
            "torch_num_threads": p.torch_num_threads, "device": p.device,
            "observation_normalization": p.observation_normalization, "reward_normalization": p.reward_normalization,
            "schedule_contract": dict(p.schedule), "optimizer": "Same PPO constructor defaults and pinned SB3 version as True-State",
            "verify_model_contract": True, "static_equivalence": static}
        write_exclusive(OUTPUT / "ppo_construction_contract.json", encode(construction))
        training = runner.train_final_checkpoint(bundle, OUTPUT / "final_smoke_model.zip", purpose="smoke")
        gate("diagnostic_budget_optimizer_finite_final_only", training["total_timesteps"] == env.total_steps == runner.DIAGNOSTIC_SMOKE_BUDGET
             and training["rollout_updates"] == runner.DIAGNOSTIC_SMOKE_BUDGET // c.n_steps
             and training["scientific_budget_used"] is False and training["optimizer_steps_checked"] > 0
             and training["final_checkpoint_only"] and runner.parameters_finite(bundle.model))
        loaded = runner.reload_final_checkpoint(OUTPUT / "final_smoke_model.zip", bundle)
        gate("save_reload_exact", loaded is not bundle.model and runner.parameters_finite(loaded)
             and runner.file_sha256(OUTPUT / "final_smoke_model.zip") == training["model_sha256"])
        schedule = runner.build_training_schedule(seed, c.name)
        expected = [{**schedule.next_assignment(), "perception_seed": runner.get_perception_contract().development_seed} for _ in env.assignments]
        gate("frozen_schedule_semantics", [{k: r[k] for k in expected[0]} for r in env.assignments] == expected
             and sum(r["steps"] for r in env.assignments) == env.total_steps)
        train_records = [r for r in ledger.records if r["role"] == "TRAINING"]
        gate("TRAINING_roots_only", len(train_records) == len(env.assignments)
             and all(r["environment_root"] in runner.partition("TRAINING").roots and r["observation_mode"] == "Point" for r in train_records))
        gate("training_perception_query_identity", env.perception_query_count == sum(r["perception_query_count"] for r in env.assignments)
             == env.total_steps + len(env.assignments) - env.episodes_completed
             and all(r["perception_query_count"] == r["natural_transition_count"] + 1 for r in env.assignments))
        part = runner.partition("DEVELOPMENT")
        specs = tuple((y, part.trajectory_ids[0]) for y in part.years)
        metrics = runner.evaluate_perception_deterministic(loaded, assets, specs, ledger, "Point")
        gate("fixed_development_smoke_evaluation", len(metrics) == len(specs)
             and [(r["year"], r["trajectory_id"]) for r in metrics] == list(specs)
             and all(r["step_count"] == p.decision_reward_steps and r["natural_transition_count"] == p.natural_transitions
                     and r["perception_query_count"] == p.decision_reward_steps for r in metrics))
        gate("finite_valid_obs_rewards_states_actions", env.observations_checked == env.total_steps + len(env.assignments)
             and all(r["all_observations_finite"] and r["all_rewards_finite"] and r["all_states_finite"]
                     and r["action_min"] in dict(p.actions) and r["action_max"] in dict(p.actions) for r in metrics))
        gate("reward_sign", all(r["reward_sign_regression_diff"] <= p.selection.atol for r in metrics))
        gate("Point_observation_boundary", env.observation_space.shape == (dict(p.observation_dimensions)["Point"],)
             and dict(p.observations)["Point"] == ("q50", "sin_DOY", "cos_DOY"))
        manifest_model = runner.model_manifest(bundle, training)
        assignments = env.assignments
        training_query_count = env.perception_query_count
    finally:
        env.close()
    allowed_banks = {(y, root, len(part.trajectory_ids)) for part in (runner.partition("TRAINING"), runner.partition("DEVELOPMENT"))
                     for y, root in zip(part.years, part.roots)}
    gate("zero_formal_trajectory_and_perception_seed_access", ledger.heldout_ppo_trajectory_access_count == 0
         and ledger.formal_perception_seed_access_count == ledger.denied_accesses == 0
         and set(assets._banks) <= allowed_banks
         and all(k[0] == runner.get_perception_contract().development_seed for k in assets._perception_cache))
    gate("perception_actually_queried_no_fallback", training_query_count > 0 and assets.perception_construction_count == 1
         and assets.perception_sample_count > 0 and assets.emulator is not None)
    total_queries = training_query_count + sum(r["perception_query_count"] for r in metrics) + 2 * pairing["perception_queries_per_mode"]
    training_summary = {**training, "model_path": "final_smoke_model.zip", "config_id": c.name, "parent_rl_seed": seed,
        **runner.child_seeds(seed), "observation_mode": "Point", "perception_seed": runner.get_perception_contract().development_seed,
        "perception_query_count": training_query_count, "scientific_budget_used": False, "model_manifest": manifest_model,
        "declaration": "ENGINEERING SMOKE ONLY; NOT FOR SELECTION OR POINT PERFORMANCE CLAIM"}
    manifest = {"stage": STAGE, "mode": "smoke", "declaration": DECLARATION, "compatibility_reason": COMPATIBILITY_REASON,
        "protocol": prior["protocol"], "selected_config": c.name, "development_eval_specs": list(specs),
        "perception_authority": {"artifact": str(runner.AUTHORITY.relative_to(ROOT)), "sha256": runner.AUTHORITY_HASHES,
            "development_seed": runner.get_perception_contract().development_seed,
            "PPO_protocol_perception_seed": p.seeds.perception_seed,
            "resolution": "Use PPO seed if non-None and equal to accepted artifact; otherwise use accepted P2-1E-1 seed"},
        "public_info_allowlist": authority["public_info_allowlist"],
        "query_identity": "reset:1; each successful nonterminal step:+1; terminal:+0; complete episode:365",
        "unsupported_behavior": "Unmodified PerceptionSupportError propagates; stop, no retry/resample/fallback",
        "trajectory_access_log": ledger.records, "generated_banks": [list(k) for k in assets._banks], "static_review": static}
    documents = {"smoke_training_summary.json": training_summary, "point_ua_pairing_regression.json": pairing,
        "protocol_manifest.json": manifest, "source_artifact_hashes.json": {"HEAD": head, "checkpoint": CHECKPOINT,
            "sha256": sources, "git_blobs": blobs, "dependencies": versions, "core_assets_provenance": assets.provenance}}
    content = {n: encode(d) for n, d in documents.items()}
    content.update({"smoke_evaluation_metrics.csv": csv_bytes(metrics), "training_episode_assignments.csv": csv_bytes(assignments)})
    for name, data in content.items():
        write_exclusive(OUTPUT / name, data)
    content["ppo_construction_contract.json"] = encode(construction)
    hashes = {n: sha(data) for n, data in content.items()}
    hashes["final_smoke_model.zip"] = training["model_sha256"]
    content["output_hashes.json"] = encode({"hashes": hashes, "exclusions": {
        "output_hashes.json": "SHA256 in audit_summary; avoid self-reference", "audit_summary.json": "Last PASS marker; pin externally"}})
    write_exclusive(OUTPUT / "output_hashes.json", content["output_hashes.json"])
    inventory = {x.name for x in OUTPUT.iterdir()}
    gate("output_inventory_readback_hashes", inventory == set(content) | {"final_smoke_model.zip"}
         and all((OUTPUT / n).read_bytes() == data for n, data in content.items())
         and json.loads((OUTPUT / "output_hashes.json").read_bytes())["hashes"] ==
             {n: runner.file_sha256(OUTPUT / n) for n in hashes})
    gate("source_HEAD_publication_integrity", git("rev-parse", "HEAD") == head
         and git("status", "--porcelain", "--untracked-files=all") == ""
         and all(sha((ROOT / n).read_bytes()) == h for n, h in sources.items())
         and all(git("rev-parse", f"{head}:{n}") == git("hash-object", f"--path={n}", n)
                 == git("rev-parse", f":{n}") == b for n, b in blobs.items()))
    gate("audit_summary_published_last", not (OUTPUT / "audit_summary.json").exists())
    require(all(gates.values()), "All engineering gates required")
    summary = {"stage": STAGE, "mode": "smoke", "stage_pass": True, "scientific_status": STATUS,
        "declaration": DECLARATION, "HEAD": head, "gates": gates, "failed_gates": [], "selected_config": c.name,
        "scientific_budget_used": False, "UA_training_runs": 0, "heldout_ppo_trajectory_access_count": ledger.heldout_ppo_trajectory_access_count,
        "formal_perception_seed_access_count": ledger.formal_perception_seed_access_count, "perception_query_count": total_queries,
        "output_sha256": {**hashes, "output_hashes.json": sha(content["output_hashes.json"])}}
    write_exclusive(OUTPUT / "audit_summary.json", encode(summary))
    require({x.name for x in OUTPUT.iterdir()} == inventory | {"audit_summary.json"}, "Final inventory")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke",))
    parser.parse_args()
    print(json.dumps(run_smoke(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
