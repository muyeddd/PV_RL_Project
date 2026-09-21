
"""P2-2D-2-v1: Shielded Point-PPO engineering smoke audit.

This stage is the first PPO training run after the frozen q50-only
set-membership shield was proven feasible and integrated into the Gym adapter.

ENGINEERING SMOKE ONLY:
* Point observation stays [q50, sin_DOY, cos_DOY].
* CONFIG_A, architecture, optimizer defaults, PPO protocol, RL seed derivation,
  training schedule and diagnostic smoke budget are inherited unchanged.
* PPO actions are proposals. The frozen q50-only shield maps nonterminal
  proposals to executed actions before the frozen core settles reward/physics.
* No UA training, no performance selection, no hyperparameter reselection,
  no warm start, no retry, no formal-held-out access, no RANDOM_TEST or
  SEALED_DATES access.
* The old unshielded P2-2D-1 support-failure artifact is preserved read-only.
* Import is inert; only --mode smoke executes.
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
SELF = "experiments/run_paper2_stage2d2_shielded_point_ppo_smoke_audit_v1.py"
RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
BASE_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
ADAPTER_STAGE = "experiments/run_paper2_stage1f1b_shielded_gym_adapter_audit_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
ADAPTER_AUTHORITY = BASE / "p2_1f_1b_shielded_gym_adapter_audit_v1/formal"
OLD_FAILURE = BASE / "p2_2d_1_point_ppo_smoke_audit_v1/smoke/perception_support_failure.json"
STAGE_DIRECTORY = BASE / "p2_2d_2_shielded_point_ppo_smoke_audit_v1"
OUTPUT = STAGE_DIRECTORY / "smoke"

CHECKPOINT = "83ddbbc54b0ca242208ea539418403d8639714d5"
PINNED_BLOBS = {
    RUNNER: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    ADAPTER_STAGE: "ea9fb0aacdc9cd083df9e238bd6d99097e37fb4c",
    BASE_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
}

STAGE = "P2-2D-2-v1"
STATUS = "SHIELDED_POINT_PPO_SMOKE_PASS"
DECLARATION = (
    "SHIELDED POINT-PPO ENGINEERING SMOKE PASS; FROZEN CONFIG_A, POINT "
    "OBSERVATION, BLOCK10 V1 PERCEPTION, Q50-ONLY SET-MEMBERSHIP SHIELD, PPO "
    "TRAINING RULES, REWARD AND PHYSICS INHERITED; SAVE/RELOAD AND DEVELOPMENT "
    "DETERMINISTIC EVALUATION COMPLETE; NO UA TRAINING OR FORMAL ACCESS"
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def encode(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def csv_bytes(rows):
    require(
        rows and all(set(row) == set(rows[0]) for row in rows),
        "Nonempty consistent CSV schema",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_exclusive(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
    require(Path(path).read_bytes() == data, "Exact output readback: " + Path(path).name)


def capture_source(sources, path):
    path = Path(path).resolve()
    digest = sha_file(path)
    key = (
        path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else str(path)
    )
    require(
        key not in sources or sources[key] == digest,
        "Source changed during capture: " + key,
    )
    sources[key] = digest
    return digest


def verify_adapter_predecessor(head):
    """Read-only P2-1F-1B authority plus source-blob pinning."""
    audit_path = ADAPTER_AUTHORITY / "audit_summary.json"
    summary_path = ADAPTER_AUTHORITY / "adapter_integration_summary.json"
    require(audit_path.exists() and summary_path.exists(), "P2-1F-1B authority exists")

    audit = json.loads(audit_path.read_bytes())
    summary = json.loads(summary_path.read_bytes())
    require(
        audit["stage"] == "P2-1F-1B-v1"
        and audit["HEAD"] == CHECKPOINT
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "SHIELDED_GYM_ADAPTER_INTEGRATION_PASS_FROZEN"
        and audit["failed_gates"] == []
        and audit["PPO_training_runs"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-1F-1B audit authority",
    )
    require(
        summary["adapter_integration_pass"] is True
        and summary["failed_gates"] == []
        and all(summary["gates"].values())
        and summary["PPO_training_runs"] == 0,
        "All P2-1F-1B adapter gates passed",
    )
    for name, digest in audit["output_sha256"].items():
        require(
            sha_file(ADAPTER_AUTHORITY / name) == digest,
            "P2-1F-1B output hash: " + name,
        )

    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned predecessor source unchanged: " + name,
        )
    return audit, summary


def verify_old_failure_preserved():
    """Freeze the prior unshielded failure as negative evidence."""
    require(OLD_FAILURE.exists(), "Original P2-2D-1 failure artifact exists")
    data = OLD_FAILURE.read_bytes()
    doc = json.loads(data)
    require(
        doc.get("stage_pass") is False
        and doc.get("scientific_status") == "POINT_PPO_SMOKE_PERCEPTION_SUPPORT_FAIL"
        and "NO RETRY" in str(doc.get("action", "")),
        "Accepted original unshielded Point-PPO support failure",
    )
    return {
        "path": OLD_FAILURE.relative_to(ROOT).as_posix(),
        "sha256": sha_bytes(data),
        "scientific_status": doc["scientific_status"],
        "context": doc.get("context"),
    }


def static_contract():
    """No bypass to unshielded PPO; smoke training only; Point only."""
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    training_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "train_final_checkpoint"
    ]
    require(len(training_calls) == 1, "Exactly one frozen training call")
    call = training_calls[0]
    require(
        len(call.keywords) == 1
        and call.keywords[0].arg == "purpose"
        and isinstance(call.keywords[0].value, ast.Constant)
        and call.keywords[0].value.value == "smoke",
        "Diagnostic smoke budget only",
    )

    names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    require(
        "build_shielded_scheduled_perception_env" in names
        and "build_shielded_perception_ppo_model" in names
        and "evaluate_shielded_perception_deterministic" in names,
        "Only shielded Point-PPO runtime entry points",
    )
    require(
        "build_scheduled_perception_env" not in names
        and "build_perception_ppo_model" not in names
        and "evaluate_perception_deterministic" not in names,
        "No unshielded Point-PPO bypass",
    )
    require(
        "learn" not in names
        and "save" not in names
        and "load" not in names,
        "No direct PPO learn/save/load bypass",
    )

    protected_role_calls = []
    for node in calls:
        values = list(node.args) + [kw.value for kw in node.keywords]
        for value in values:
            if (
                isinstance(value, ast.Constant)
                and value.value in ("FORMAL HELD-OUT", "RANDOM_TEST", "SEALED_DATES")
            ):
                protected_role_calls.append(value.value)
    require(not protected_role_calls, "No protected-role constructor/call literal")

    # Any explicit observation-mode literal passed to shielded runtime helpers
    # must be Point. UA is mentioned only in declarations/zero-run reporting.
    shield_runtime = {
        "build_shielded_scheduled_perception_env",
        "evaluate_shielded_perception_deterministic",
    }
    for node in calls:
        if isinstance(node.func, ast.Attribute) and node.func.attr in shield_runtime:
            literals = [
                arg.value
                for arg in list(node.args) + [kw.value for kw in node.keywords]
                if isinstance(arg, ast.Constant) and arg.value in ("Point", "UA")
            ]
            require("UA" not in literals, "No UA runtime entry in Point smoke")

    return {
        "single_smoke_training_call": True,
        "shielded_runtime_only": True,
        "no_unshielded_runtime_bypass": True,
        "no_direct_learn_save_load": True,
        "no_protected_role_call": True,
        "Point_only_runtime": True,
    }


def run_smoke():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive stage directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean sources required",
    )
    head = git("rev-parse", "HEAD")
    adapter_audit, adapter_summary = verify_adapter_predecessor(head)
    old_failure = verify_old_failure_preserved()
    old_failure_hash_before = old_failure["sha256"]

    # New source itself must be committed unchanged.
    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(
        self_blob
        == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "New smoke source committed unchanged",
    )

    sources = {}
    for name in (
        SELF,
        RUNNER,
        BASE_RUNNER,
        SHIELD,
        CORE,
        TRUE_RUNNER,
        ADAPTER_STAGE,
    ):
        capture_source(sources, ROOT / name)

    static = static_contract()

    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as smoke_authority
    import paper2_ppo_protocol_v1 as frozen
    import paper2_shielded_perception_ppo_runner_v1 as runner

    versions = smoke_authority.dependency_versions()
    p = runner.get_protocol()
    require(
        versions == dict(p.dependencies),
        "Frozen runtime dependency versions",
    )
    require(
        frozen.protocol_payload(p) == frozen.protocol_payload(frozen.Protocol()),
        "Exact frozen PPO protocol",
    )
    require(
        runner.inherited_config_id() == "CONFIG_A"
        == p.candidates[0].name,
        "Inherited CONFIG_A; no reselection",
    )
    authority = runner.read_shield_authority()
    require(
        authority.feasibility_audit[
            "support_compatible_shield_feasibility_pass"
        ]
        is True,
        "Frozen shield authority available",
    )

    assets = runner.load_assets()
    assets.ensure_perception()

    # Pin external perception inputs before any training environment is reset.
    for name, path in (
        ("paper1_cqr", assets.paper1_cqr),
        ("wapp_power_bridge", assets.wapp_power_bridge_path),
    ):
        expected = assets.perception.EXPECTED_HASHES[name]
        require(sha_file(path) == expected, "Frozen perception input: " + name)
        capture_source(sources, path)

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    try:
        return execute_smoke(
            runner=runner,
            assets=assets,
            protocol_payload=frozen.protocol_payload(p),
            head=head,
            versions=versions,
            static=static,
            sources=sources,
            self_blob=self_blob,
            adapter_audit=adapter_audit,
            adapter_summary=adapter_summary,
            old_failure=old_failure,
            old_failure_hash_before=old_failure_hash_before,
        )
    except Exception as exc:
        if not (OUTPUT / "execution_failure.json").exists():
            write_exclusive(
                OUTPUT / "execution_failure.json",
                encode(
                    {
                        "stage": STAGE,
                        "stage_pass": False,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "action":
                            "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/PERCEPTION OR SHIELD TUNING",
                        "HEAD": head,
                    }
                ),
            )
        raise


def execute_smoke(
    *,
    runner,
    assets,
    protocol_payload,
    head,
    versions,
    static,
    sources,
    self_blob,
    adapter_audit,
    adapter_summary,
    old_failure,
    old_failure_hash_before,
):
    gates = {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate")
        gates[name] = bool(condition)
        require(condition, name)

    p = runner.get_protocol()
    config = runner.candidate(runner.inherited_config_id())
    rl_seed = p.seeds.development[0]

    gate(
        "adapter_and_shield_predecessor_exact",
        adapter_audit["stage_pass"] is True
        and adapter_summary["adapter_integration_pass"] is True
        and all(adapter_summary["gates"].values()),
    )
    gate("static_smoke_contract", all(static.values()))
    gate(
        "frozen_Point_observation_contract",
        dict(p.observations)["Point"] == ("q50", "sin_DOY", "cos_DOY")
        and dict(p.observation_dimensions)["Point"] == 3,
    )
    gate(
        "frozen_CONFIG_A_and_smoke_budget",
        config.name == "CONFIG_A"
        and config.n_steps == 2048
        and config.learning_rate == 3e-4
        and runner.DIAGNOSTIC_SMOKE_BUDGET == 4096,
    )

    ledger = runner.AccessLedger()
    env = runner.build_shielded_scheduled_perception_env(
        assets=assets,
        rl_seed=rl_seed,
        ledger=ledger,
        config_id=config.name,
        observation_mode="Point",
    )
    # Use the just-created assets object consistently from here on.
    training_assets = env.assets

    try:
        gate(
            "fresh_shielded_Point_training_environment",
            env.observation_mode == "Point"
            and env.observation_space.shape == (3,)
            and env.schedule.parent_seed == rl_seed
            and not env.started
            and not env.failed
            and not env.assignments,
        )

        bundle = runner.build_shielded_perception_ppo_model(
            env, config.name, rl_seed
        )
        gate(
            "frozen_PPO_constructor_contract",
            runner.verify_model_contract(bundle.model, config.name, rl_seed)
            and bundle.model.num_timesteps == 0
            and not bundle.training_finished
            and not p.observation_normalization
            and not p.reward_normalization,
        )

        construction = {
            "config_id": config.name,
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "diagnostic_smoke_budget": runner.DIAGNOSTIC_SMOKE_BUDGET,
            "common": dict(p.common),
            "architecture": dict(p.architecture),
            "parent_rl_seed": rl_seed,
            "child_seeds": runner.child_seeds(rl_seed),
            "observation_mode": "Point",
            "policy_observation": list(dict(p.observations)["Point"]),
            "shield":
                "frozen q50-only set-membership predictive shield",
            "shield_inputs": ["q50_float32_history", "executed_action_history"],
            "reward_shaping": False,
            "scientific_budget_used": False,
        }
        write_exclusive(
            OUTPUT / "ppo_construction_contract.json", encode(construction)
        )

        training = runner.train_final_checkpoint(
            bundle,
            OUTPUT / "final_smoke_model.zip",
            purpose="smoke",
        )
        gate(
            "diagnostic_budget_optimizer_and_parameters_finite",
            training["total_timesteps"]
            == env.total_steps
            == runner.DIAGNOSTIC_SMOKE_BUDGET
            and training["rollout_updates"]
            == runner.DIAGNOSTIC_SMOKE_BUDGET // config.n_steps
            and training["scientific_budget_used"] is False
            and training["optimizer_steps_checked"] > 0
            and training["final_checkpoint_only"] is True
            and runner.parameters_finite(bundle.model),
        )

        shield_training = env._shield_audit_snapshot()
        expected_terminal_passthrough = env.episodes_completed
        gate(
            "training_shield_accounting",
            shield_training["total_terminal_passthrough"]
            == expected_terminal_passthrough
            and shield_training["total_safety_filtered_decisions"]
            + shield_training["total_terminal_passthrough"]
            == env.total_steps
            and shield_training["total_executed_clean"]
            == shield_training["total_proposed_clean"]
            + shield_training["total_shield_interventions"]
            and shield_training["total_free_wait"]
            + shield_training["total_proposed_clean"]
            + shield_training["total_shield_interventions"]
            == env.total_steps,
        )
        gate(
            "training_schedule_and_perception_queries",
            env.perception_query_count
            == env.total_steps + len(env.assignments) - env.episodes_completed
            and env.observations_checked
            == env.total_steps + len(env.assignments)
            and all(
                row["perception_query_count"]
                == row["natural_transition_count"] + 1
                for row in env.assignments
            ),
        )
        gate(
            "TRAINING_roots_only",
            len([r for r in ledger.records if r["role"] == "TRAINING"])
            == len(env.assignments)
            and all(
                r["environment_root"] in runner.partition("TRAINING").roots
                and r["observation_mode"] == "Point"
                for r in ledger.records
                if r["role"] == "TRAINING"
            ),
        )

        loaded = runner.reload_final_checkpoint(
            OUTPUT / "final_smoke_model.zip", bundle
        )
        gate(
            "save_reload_exact_and_finite",
            loaded is not bundle.model
            and runner.parameters_finite(loaded)
            and runner.file_sha256(OUTPUT / "final_smoke_model.zip")
            == training["model_sha256"],
        )

        dev = runner.partition("DEVELOPMENT")
        specs = tuple(
            (year, dev.trajectory_ids[0]) for year in dev.years
        )
        metrics = runner.evaluate_shielded_perception_deterministic(
            loaded,
            training_assets,
            specs,
            ledger,
            "Point",
        )
        gate(
            "fixed_DEVELOPMENT_deterministic_evaluation",
            len(metrics) == 2
            and [(r["year"], r["trajectory_id"]) for r in metrics]
            == list(specs)
            and all(
                r["step_count"] == p.decision_reward_steps
                and r["natural_transition_count"] == p.natural_transitions
                and r["perception_query_count"] == p.decision_reward_steps
                for r in metrics
            ),
        )
        gate(
            "finite_eval_and_reward_cost_identity",
            all(
                r["all_observations_finite"]
                and r["all_rewards_finite"]
                and r["all_states_finite"]
                and r["reward_sign_regression_diff"] <= p.selection.atol
                for r in metrics
            ),
        )
        gate(
            "eval_proposal_execution_and_shield_accounting",
            all(
                r["N_clean"] >= r["N_clean_proposed"]
                and r["N_clean"] - r["N_clean_proposed"]
                == r["shield_interventions"]
                and 0.0 <= r["shield_intervention_fraction"] <= 1.0
                and r["proposed_action_min"] in dict(p.actions)
                and r["proposed_action_max"] in dict(p.actions)
                and r["executed_action_min"] in dict(p.actions)
                and r["executed_action_max"] in dict(p.actions)
                for r in metrics
            ),
        )

        manifest_model = runner.model_manifest(bundle, training)
        assignments = [dict(row) for row in env.assignments]
        training_query_count = env.perception_query_count
        training_shield_summary = {
            **shield_training,
            "intervention_fraction":
                shield_training["total_shield_interventions"]
                / shield_training["total_safety_filtered_decisions"],
            "completed_episodes": env.episodes_completed,
            "assignment_count": len(env.assignments),
            "total_steps": env.total_steps,
        }
    finally:
        env.close()

    allowed_roots = (
        set(runner.partition("TRAINING").roots)
        | set(runner.partition("DEVELOPMENT").roots)
    )
    gate(
        "zero_formal_trajectory_and_perception_seed_access",
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0
        and all(
            record["environment_root"] in allowed_roots
            for record in ledger.records
        ),
    )
    gate(
        "perception_queried_without_extension_or_fallback",
        training_query_count > 0
        and training_assets.perception_construction_count == 1
        and training_assets.perception_sample_count > 0
        and training_assets.emulator is not None,
    )
    gate(
        "old_unshielded_failure_preserved",
        OLD_FAILURE.exists()
        and sha_file(OLD_FAILURE) == old_failure_hash_before,
    )

    total_queries = training_query_count + sum(
        row["perception_query_count"] for row in metrics
    )
    training_summary = {
        **training,
        "model_path": "final_smoke_model.zip",
        "config_id": config.name,
        "parent_rl_seed": rl_seed,
        **runner.child_seeds(rl_seed),
        "observation_mode": "Point",
        "perception_seed": runner.get_perception_contract().development_seed,
        "perception_query_count": training_query_count,
        "scientific_budget_used": False,
        "model_manifest": manifest_model,
        "shield_training": training_shield_summary,
        "declaration":
            "ENGINEERING SHIELDED POINT-PPO SMOKE ONLY; NOT FOR MODEL SELECTION OR PERFORMANCE CLAIM",
    }
    protocol_manifest = {
        "stage": STAGE,
        "mode": "smoke",
        "declaration": DECLARATION,
        "protocol": protocol_payload,
        "selected_config": config.name,
        "development_eval_specs": [list(x) for x in specs],
        "policy_observation": ["q50", "sin_DOY", "cos_DOY"],
        "policy_action_semantics":
            "PPO proposal -> frozen q50-only shield -> executed action -> frozen reward/physics",
        "terminal_day_semantics":
            "day 364 proposal passes unchanged; no t->t+1 safety transition",
        "shield_authority_stage": "P2-1F-1A-v1",
        "adapter_authority_stage": "P2-1F-1B-v1",
        "old_unshielded_failure": old_failure,
        "trajectory_access_log": ledger.records,
        "static_contract": static,
        "prohibitions": [
            "UA training",
            "hyperparameter reselection",
            "perception extension",
            "shield tuning",
            "retry/reseed",
            "formal held-out access",
            "RANDOM_TEST access",
            "SEALED_DATES access",
        ],
    }
    source_manifest = {
        "HEAD": head,
        "checkpoint": CHECKPOINT,
        "source_sha256": sources,
        "new_source_git_blob": self_blob,
        "pinned_predecessor_blobs": PINNED_BLOBS,
        "dependencies": versions,
        "core_assets_provenance": training_assets.provenance,
        "old_failure_sha256": old_failure_hash_before,
    }

    documents = {
        "smoke_training_summary.json": training_summary,
        "protocol_manifest.json": protocol_manifest,
        "source_artifact_hashes.json": source_manifest,
    }
    content = {name: encode(doc) for name, doc in documents.items()}
    content["smoke_evaluation_metrics.csv"] = csv_bytes(metrics)
    content["training_episode_assignments.csv"] = csv_bytes(assignments)

    for name, data in content.items():
        write_exclusive(OUTPUT / name, data)

    content["ppo_construction_contract.json"] = encode(construction)
    # Already published before training; include its actual bytes in hash inventory.
    require(
        (OUTPUT / "ppo_construction_contract.json").read_bytes()
        == content["ppo_construction_contract.json"],
        "Construction contract exact readback",
    )

    hashes = {name: sha_bytes(data) for name, data in content.items()}
    hashes["final_smoke_model.zip"] = training["model_sha256"]
    output_hashes = encode(
        {
            "hashes": hashes,
            "exclusions": {
                "output_hashes.json":
                    "SHA256 is pinned in audit_summary; avoid self-reference",
                "audit_summary.json":
                    "Published last PASS marker",
            },
        }
    )
    write_exclusive(OUTPUT / "output_hashes.json", output_hashes)

    expected_inventory = (
        set(content)
        | {"final_smoke_model.zip", "output_hashes.json"}
    )
    gate(
        "output_inventory_and_readback_hashes",
        {p.name for p in OUTPUT.iterdir()} == expected_inventory
        and all(
            (OUTPUT / name).read_bytes() == data
            for name, data in content.items()
        )
        and json.loads(output_hashes)["hashes"]
        == {name: runner.file_sha256(OUTPUT / name) for name in hashes},
    )
    gate(
        "source_HEAD_and_old_failure_integrity",
        git("rev-parse", "HEAD") == head
        and git("status", "--porcelain", "--untracked-files=all") == ""
        and all(sha_file(ROOT / name) == digest for name, digest in sources.items())
        and OLD_FAILURE.exists()
        and sha_file(OLD_FAILURE) == old_failure_hash_before,
    )
    gate(
        "audit_summary_published_last",
        not (OUTPUT / "audit_summary.json").exists(),
    )
    require(all(gates.values()), "All shielded Point-PPO smoke gates required")

    summary = {
        "stage": STAGE,
        "mode": "smoke",
        "stage_pass": True,
        "scientific_status": STATUS,
        "declaration": DECLARATION,
        "HEAD": head,
        "gates": gates,
        "failed_gates": [],
        "selected_config": config.name,
        "scientific_budget_used": False,
        "PPO_training_runs": 1,
        "Point_training_runs": 1,
        "UA_training_runs": 0,
        "heldout_ppo_trajectory_access_count":
            ledger.heldout_ppo_trajectory_access_count,
        "formal_perception_seed_access_count":
            ledger.formal_perception_seed_access_count,
        "RANDOM_TEST_access_count": 0,
        "SEALED_DATES_access_count": 0,
        "perception_query_count": total_queries,
        "training_shield_interventions":
            training_shield_summary["total_shield_interventions"],
        "training_shield_intervention_fraction":
            training_shield_summary["intervention_fraction"],
        "development_evaluation_episode_count": len(metrics),
        "output_sha256": {
            **hashes,
            "output_hashes.json": sha_bytes(output_hashes),
        },
    }
    write_exclusive(OUTPUT / "audit_summary.json", encode(summary))
    require(
        {p.name for p in OUTPUT.iterdir()}
        == expected_inventory | {"audit_summary.json"},
        "Final output inventory",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke",))
    parser.parse_args()
    print(json.dumps(run_smoke(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
