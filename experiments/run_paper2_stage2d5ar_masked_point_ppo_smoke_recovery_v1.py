
"""P2-2D-5AR-v1: Masked Point-PPO smoke API-compatibility recovery.

Predecessor P2-2D-5A failed BEFORE model construction/training because
sb3-contrib==2.7.1 MaskablePPO.__init__ rejected the frozen ordinary-PPO kwarg
"use_sde".  This recovery changes no scientific setting.

The only compatibility rule is:
    assert frozen use_sde=False and sde_sample_freq=-1
    assert MaskablePPO.__init__ does not accept those two kwargs
    omit exactly those two kwargs at construction

Everything else remains frozen:
- Point observation [q50, sin_DOY, cos_DOY]
- q50-history action mask
- unchanged frozen q50 shield as runtime fail-safe
- reward / physics / cleaning mechanics
- CONFIG_A PPO hyperparameters and architecture
- development seed 510001 for smoke
- 4096 diagnostic interactions
- TRAINING gradients only; fixed DEVELOPMENT smoke evaluation only
- no UA, formal held-out, RANDOM_TEST or SEALED_DATES access
- no performance gate / no tuning / no retry.

Import is inert. Only --mode smoke executes.
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

SELF = (
    "experiments/"
    "run_paper2_stage2d5ar_masked_point_ppo_smoke_recovery_v1.py"
)
RECOVERY_RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1_1.py"

FAILED_STAGE_SOURCE = (
    "experiments/"
    "run_paper2_stage2d5a_masked_point_ppo_smoke_audit_v1.py"
)
FAILED_MASKED_RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1.py"
MASKED_REQUIREMENTS = "requirements-paper2-masked-v1.txt"

SHIELDED_RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
PERCEPTION_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
STAGE4B_SOURCE = (
    "experiments/"
    "run_paper2_stage2d4b_safe_action_distinction_decomposition_audit_v1.py"
)
BASE_REQUIREMENTS = "requirements.txt"

FAILED_HEAD = "482aeb612d344fa0f1cfd563bc55e9575364c3fb"
FAILED_OUTPUT = (
    ROOT
    / "outputs/paper2_uncertainty_rl_v1/"
      "p2_2d_5a_masked_point_ppo_smoke_audit_v1/smoke"
)
FAILED_EXECUTION_SHA256 = (
    "f5683ca497fc9a265d5769ba9bbd07a156ddac89da9ce3b4b86e4483b3b2a1e4"
)

PINNED_BLOBS = {
    SHIELDED_RUNNER: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    PERCEPTION_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
    STAGE4B_SOURCE: "e5b4006c9ddb1866e78df2335ccc855de2253b98",
    BASE_REQUIREMENTS: "430357153fbdc6b2b77d92d8040333d2de30135b",
    FAILED_MASKED_RUNNER: "be86a5e8a8a162c0d6be720c5340914ccc1ed82e",
    FAILED_STAGE_SOURCE: "2b80a1160bd4aceb749c8ab0fbe658c8a6c83485",
    MASKED_REQUIREMENTS: "8bb5ad1e5b6d84eb31b7a207aefde57cc0e3b88c",
}

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
STAGE4B = (
    BASE / "p2_2d_4b_safe_action_distinction_decomposition_audit_v1/formal"
)
STAGE_DIRECTORY = BASE / "p2_2d_5ar_masked_point_ppo_smoke_recovery_v1"
OUTPUT = STAGE_DIRECTORY / "smoke"

STAGE = "P2-2D-5AR-v1"
STATUS = "MASKED_POINT_PPO_SMOKE_API_RECOVERY_PASS"
DECLARATION = (
    "P2-2D-5A API-COMPATIBILITY RECOVERY ONLY: EXACT FROZEN NO-GSDE VALUES "
    "ARE ASSERTED AND ONLY MASKABLEPPO-UNSUPPORTED use_sde/sde_sample_freq "
    "CONSTRUCTOR KWARGS ARE OMITTED; NO MASK/SHIELD/PERCEPTION/REWARD/PPO "
    "HYPERPARAMETER/SEED/BUDGET CHANGE"
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def encode(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def csv_bytes(rows):
    require(rows, "Nonempty CSV required")
    fields = list(rows[0])
    require(
        all(set(row) == set(fields) for row in rows),
        "Consistent CSV schema",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_exclusive(path, data):
    path = Path(path)
    require(
        path.parent.resolve() == OUTPUT.resolve(),
        "Writes restricted to new recovery output directory",
    )
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, "Exact readback: " + path.name)


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def verify_failed_5a(head):
    require(
        FAILED_OUTPUT.is_dir(),
        "Failed P2-2D-5A smoke directory exists",
    )
    actual = {
        item.name
        for item in FAILED_OUTPUT.iterdir()
        if item.is_file()
    }
    require(
        actual == {"execution_failure.json"},
        "Failed P2-2D-5A directory remains evidence-only",
    )

    failure_path = FAILED_OUTPUT / "execution_failure.json"
    require(
        sha_file(failure_path) == FAILED_EXECUTION_SHA256,
        "Exact failed P2-2D-5A execution evidence SHA256",
    )
    failure = json.loads(failure_path.read_bytes())
    require(
        failure
        == {
            "HEAD": FAILED_HEAD,
            "action":
                "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/MASK/SHIELD/PERCEPTION/PPO TUNING",
            "error_type": "TypeError",
            "message":
                "MaskablePPO.__init__() got an unexpected keyword argument 'use_sde'",
            "stage": "P2-2D-5A-v1",
            "stage_pass": False,
        },
        "Exact failed P2-2D-5A content",
    )

    git("merge-base", "--is-ancestor", FAILED_HEAD, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{FAILED_HEAD}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Frozen predecessor source unchanged: " + name,
        )

    return {
        "failed_HEAD": FAILED_HEAD,
        "execution_failure_sha256": FAILED_EXECUTION_SHA256,
        "failure_message": failure["message"],
        "failed_stage_source_blob":
            PINNED_BLOBS[FAILED_STAGE_SOURCE],
        "failed_masked_runner_blob":
            PINNED_BLOBS[FAILED_MASKED_RUNNER],
        "read_only": True,
    }


def verify_4b():
    audit_path = STAGE4B / "audit_summary.json"
    require(audit_path.exists(), "P2-2D-4B authority exists")
    audit = json.loads(audit_path.read_bytes())
    require(
        audit["stage"] == "P2-2D-4B-v1"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "SAFE_ACTION_DISTINCTION_DECOMPOSITION_AUDIT_COMPLETE"
        and audit["counterfactual_anchor_count"] == 163
        and audit["immediate_distinguishable_count"] == 163
        and audit["economic_distinguishable_count"] == 163
        and audit["horizon_cost_distinguishable_count"] == 153
        and audit["clean_beneficial_count"] == 63
        and audit["wait_beneficial_count"] == 90
        and audit["tie_count"] == 10
        and audit["PPO_training_runs"] == 0
        and audit["PPO_model_loads"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-2D-4B route authority",
    )
    for name, digest in audit["output_sha256"].items():
        require(
            sha_file(STAGE4B / name) == digest,
            "P2-2D-4B output hash: " + name,
        )
    return {
        "audit_summary_sha256": sha_file(audit_path),
        "economic_distinguishable_count": 163,
        "horizon_cost_distinguishable_count": 153,
        "clean_beneficial_count": 63,
        "wait_beneficial_count": 90,
        "tie_count": 10,
    }


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    runner_source = (ROOT / RECOVERY_RUNNER).read_text(encoding="utf-8")
    tree = ast.parse(source)
    runner_tree = ast.parse(runner_source)

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }

    train_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "train_masked_final_checkpoint"
    ]
    require(
        len(train_calls) == 1,
        "Exactly one recovery smoke training call",
    )
    call = train_calls[0]
    require(
        len(call.keywords) == 1
        and call.keywords[0].arg == "purpose"
        and isinstance(call.keywords[0].value, ast.Constant)
        and call.keywords[0].value.value == "smoke",
        "Diagnostic smoke budget only",
    )

    require(
        "build_masked_scheduled_perception_env" in attrs
        and "build_masked_perception_ppo_model" in attrs
        and "evaluate_masked_perception_deterministic" in attrs,
        "Recovery masked runtime required",
    )
    require(
        "learn" not in attrs
        and "save" not in attrs
        and "load" not in attrs,
        "No direct training/save/load bypass",
    )

    pop_keys = {
        node.args[0].value
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "common"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    require(
        {"use_sde", "sde_sample_freq"} <= pop_keys,
        "Recovery runner mechanically omits exact unsupported kwargs",
    )
    require(
        'frozen_use_sde is False' in runner_source
        and 'frozen_sde_sample_freq == -1' in runner_source,
        "Recovery runner asserts frozen no-gSDE values",
    )
    require(
        "UNSUPPORTED_NO_GSDE_KWARGS" in runner_source
        and "MaskablePPO.__init__" in runner_source,
        "Explicit API-compatibility rationale present",
    )

    protected = []
    for node in calls:
        values = list(node.args) + [kw.value for kw in node.keywords]
        for value in values:
            if (
                isinstance(value, ast.Constant)
                and value.value
                in (
                    "FORMAL HELD-OUT",
                    "RANDOM_TEST",
                    "SEALED_DATES",
                    "UA",
                )
            ):
                protected.append(value.value)
    require(not protected, "No protected/UA runtime literal")

    require(
        "FAILED_EXECUTION_SHA256" in source
        and "verify_failed_5a" in source
        and "performance_diagnostics_not_pass_gates" in source,
        "Failed evidence pin and no-performance-gate contract",
    )

    return {
        "single_4096_smoke_training_call": True,
        "recovery_masked_runtime_only": True,
        "no_direct_learn_save_load": True,
        "exact_no_gSDE_values_asserted": True,
        "only_unsupported_gSDE_kwargs_omitted": True,
        "failed_5A_evidence_read_only": True,
        "Point_only_no_protected_runtime": True,
        "performance_diagnostics_not_pass_gates": True,
    }


def run_smoke():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive recovery directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean worktree required",
    )
    head = git("rev-parse", "HEAD")
    failed_before = verify_failed_5a(head)
    stage4b = verify_4b()

    for name in (SELF, RECOVERY_RUNNER):
        blob = git("rev-parse", "--verify", f"{head}:{name}")
        require(
            blob
            == git("hash-object", f"--path={name}", name)
            == git("rev-parse", f":{name}"),
            "Recovery source committed unchanged: " + name,
        )

    static = static_contract()

    import paper2_masked_perception_ppo_runner_v1_1 as runner
    import paper2_ppo_protocol_v1 as frozen

    api = runner.verify_maskable_api_contract()
    require(
        api["MaskablePPO_init_rejects_use_sde"] is True
        and api["MaskablePPO_init_rejects_sde_sample_freq"] is True
        and api["frozen_use_sde"] is False
        and api["frozen_sde_sample_freq"] == -1,
        "Exact API recovery premise",
    )

    p = runner.get_protocol()
    require(
        frozen.protocol_payload(p)
        == frozen.protocol_payload(frozen.Protocol()),
        "Exact frozen PPO protocol",
    )
    require(
        runner.inherited_config_id() == "CONFIG_A"
        and runner.DIAGNOSTIC_SMOKE_BUDGET == 4096
        and dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY")
        and dict(p.observation_dimensions)["Point"] == 3,
        "Frozen Point CONFIG_A smoke contract",
    )

    assets = runner.load_assets()
    assets.ensure_perception()

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        result = execute_smoke(
            runner=runner,
            assets=assets,
            protocol=frozen.protocol_payload(p),
            head=head,
            api=api,
            static=static,
            failed_before=failed_before,
            stage4b=stage4b,
        )
        require(
            verify_failed_5a(head)["execution_failure_sha256"]
            == failed_before["execution_failure_sha256"],
            "Failed P2-2D-5A evidence preserved after recovery",
        )
        return result
    except Exception as exc:
        if not (OUTPUT / "execution_failure.json").exists():
            write_exclusive(
                OUTPUT / "execution_failure.json",
                encode(
                    {
                        "stage": STAGE,
                        "stage_pass": False,
                        "HEAD": head,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "action":
                            "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/MASK/SHIELD/PERCEPTION/PPO TUNING",
                    }
                ),
            )
        raise


def execute_smoke(
    *,
    runner,
    assets,
    protocol,
    head,
    api,
    static,
    failed_before,
    stage4b,
):
    p = runner.get_protocol()
    config = runner.candidate(runner.inherited_config_id())
    rl_seed = p.seeds.development[0]

    gates = {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate: " + name)
        gates[name] = bool(condition)
        require(condition, name)

    gate(
        "failed_5A_API_evidence_exact",
        failed_before["execution_failure_sha256"]
        == FAILED_EXECUTION_SHA256,
    )
    gate(
        "P2_2D_4B_route_authority_exact",
        stage4b["economic_distinguishable_count"] == 163
        and stage4b["horizon_cost_distinguishable_count"] == 153
        and stage4b["clean_beneficial_count"] == 63
        and stage4b["wait_beneficial_count"] == 90,
    )
    gate("static_recovery_contract", all(static.values()))
    gate(
        "API_compatibility_recovery_exact",
        api["frozen_use_sde"] is False
        and api["frozen_sde_sample_freq"] == -1
        and api["MaskablePPO_init_rejects_use_sde"] is True
        and api["MaskablePPO_init_rejects_sde_sample_freq"] is True,
    )
    gate(
        "frozen_CONFIG_A_4096_smoke",
        config.name == "CONFIG_A"
        and config.learning_rate == 3e-4
        and config.n_steps == 2048
        and runner.DIAGNOSTIC_SMOKE_BUDGET == 4096,
    )

    ledger = runner.AccessLedger()
    env = runner.build_masked_scheduled_perception_env(
        assets=assets,
        rl_seed=rl_seed,
        ledger=ledger,
        config_id=config.name,
        observation_mode="Point",
    )

    try:
        gate(
            "fresh_masked_Point_training_env",
            env.observation_mode == "Point"
            and env.observation_space.shape == (3,)
            and env.action_space.n == 2
            and env.action_space.start == 0
            and env.schedule.parent_seed == rl_seed
            and not env.started
            and not env.failed,
        )

        bundle = runner.build_masked_perception_ppo_model(
            env, config.name, rl_seed
        )
        gate(
            "MaskablePPO_constructed_under_recovered_contract",
            runner.verify_masked_model_contract(
                bundle.model, config.name, rl_seed
            )
            and bundle.model.num_timesteps == 0
            and not bundle.training_finished,
        )

        construction = {
            "stage": STAGE,
            "runtime_algorithm": "MaskablePPO",
            "config_id": config.name,
            "parent_rl_seed": rl_seed,
            "child_seeds": runner.child_seeds(rl_seed),
            "observation_mode": "Point",
            "policy_observation":
                list(dict(p.observations)["Point"]),
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "diagnostic_smoke_budget": 4096,
            "API_recovery": {
                "frozen_use_sde": False,
                "frozen_sde_sample_freq": -1,
                "omitted_constructor_kwargs":
                    ["use_sde", "sde_sample_freq"],
                "all_other_PPO_semantics_unchanged": True,
            },
            "mask_semantics": {
                "nonterminal_WAIT_safe": [True, True],
                "nonterminal_WAIT_unsafe": [False, True],
                "terminal_day_364": [True, True],
            },
            "shield_remains_runtime_failsafe": True,
            "reward_shaping": False,
            "performance_gate": False,
        }
        write_exclusive(
            OUTPUT / "recovered_masked_ppo_construction.json",
            encode(construction),
        )

        training = runner.train_masked_final_checkpoint(
            bundle,
            OUTPUT / "final_masked_smoke_model.zip",
            purpose="smoke",
        )
        gate(
            "exact_4096_training_and_finite_parameters",
            training["total_timesteps"] == 4096
            and env.total_steps == 4096
            and training["scientific_budget_used"] is False
            and training["optimizer_steps_checked"] > 0
            and runner.parameters_finite(bundle.model),
        )

        mask_training = env._mask_audit_snapshot()
        gate(
            "zero_training_shield_interventions",
            mask_training["total_shield_interventions"] == 0
            and mask_training["mask_shield_intervention_invariant"] is True
            and mask_training["total_proposed_clean"]
            == mask_training["total_executed_clean"],
        )
        gate(
            "training_mask_partition_exact",
            mask_training["total_masked_safety_decisions"]
            + mask_training["total_terminal_passthrough"]
            == env.total_steps
            and mask_training["total_masked_safety_decisions"]
            == mask_training["total_forced_clean"]
            + mask_training["total_free_choice_decisions"]
            and mask_training["total_free_choice_decisions"]
            == mask_training["total_voluntary_clean"]
            + mask_training["total_mask_free_wait"]
            and mask_training["total_executed_clean"]
            == mask_training["total_forced_clean"]
            + mask_training["total_voluntary_clean"]
            + mask_training["total_terminal_clean"],
        )
        gate(
            "both_mask_patterns_exercised_in_training",
            mask_training["total_forced_clean"] > 0
            and mask_training["total_free_choice_decisions"] > 0,
        )

        loaded = runner.reload_masked_final_checkpoint(
            OUTPUT / "final_masked_smoke_model.zip",
            bundle,
        )
        gate(
            "masked_save_reload_exact",
            loaded is not bundle.model
            and runner.parameters_finite(loaded)
            and runner.file_sha256(
                OUTPUT / "final_masked_smoke_model.zip"
            )
            == training["model_sha256"],
        )

        dev = runner.partition("DEVELOPMENT")
        specs = tuple(
            (year, dev.trajectory_ids[0])
            for year in dev.years
        )
        metrics = runner.evaluate_masked_perception_deterministic(
            loaded,
            assets,
            specs,
            ledger,
            "Point",
        )
        gate(
            "two_fixed_DEVELOPMENT_smoke_evals",
            len(metrics) == 2
            and [
                (row["year"], row["trajectory_id"])
                for row in metrics
            ]
            == list(specs)
            and all(
                row["step_count"] == p.decision_reward_steps
                and row["natural_transition_count"]
                == p.natural_transitions
                and row["N_shield_interventions"] == 0
                and row["N_forced_clean"] + row["N_free_choice"]
                == p.natural_transitions
                and row["terminal_mask"] == [True, True]
                for row in metrics
            ),
        )
        gate(
            "eval_reward_and_mask_accounting_finite",
            all(
                row["all_observations_finite"]
                and row["all_rewards_finite"]
                and row["all_states_finite"]
                and row["reward_sign_regression_diff"]
                <= p.selection.atol
                and 0.0 <= row["forced_clean_fraction"] <= 1.0
                and 0.0 <= row["free_choice_fraction"] <= 1.0
                and 0.0
                <= row["voluntary_clean_fraction_given_free"]
                <= 1.0
                for row in metrics
            ),
        )

        assignments = [dict(row) for row in env.assignments]
        manifest = runner.masked_model_manifest(bundle, training)
    finally:
        env.close()

    allowed_roots = (
        set(runner.partition("TRAINING").roots)
        | set(runner.partition("DEVELOPMENT").roots)
    )
    gate(
        "zero_protected_access",
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0
        and all(
            row["environment_root"] in allowed_roots
            for row in ledger.records
        ),
    )

    performance_diagnostics_not_pass_gates = {
        "training_forced_clean":
            mask_training["total_forced_clean"],
        "training_voluntary_clean":
            mask_training["total_voluntary_clean"],
        "training_free_choice_decisions":
            mask_training["total_free_choice_decisions"],
        "eval_forced_clean":
            sum(row["N_forced_clean"] for row in metrics),
        "eval_voluntary_clean":
            sum(row["N_voluntary_clean"] for row in metrics),
        "eval_free_choice":
            sum(row["N_free_choice"] for row in metrics),
        "eval_mean_J_total":
            sum(row["J_total"] for row in metrics) / len(metrics),
        "role":
            "DESCRIPTIVE ENGINEERING-SMOKE DIAGNOSTICS ONLY; NOT A PASS GATE",
    }

    failed_after = verify_failed_5a(head)
    gate(
        "failed_5A_evidence_preserved",
        failed_after["execution_failure_sha256"]
        == failed_before["execution_failure_sha256"],
    )

    protocol_manifest = {
        "stage": STAGE,
        "mode": "smoke",
        "declaration": DECLARATION,
        "protocol": protocol,
        "runtime_algorithm": "MaskablePPO",
        "selected_config": config.name,
        "observation_mode": "Point",
        "action_semantics":
            "frozen q50 mask -> MaskablePPO -> unchanged frozen q50 shield -> frozen reward/physics",
        "API_recovery_only": True,
        "API_recovery_rule":
            "assert use_sde=False and sde_sample_freq=-1, omit only those unsupported constructor kwargs",
        "development_eval_specs": [list(x) for x in specs],
        "scientific_budget_used": False,
        "performance_diagnostics_as_gate": False,
        "trajectory_access_log": ledger.records,
    }
    training_summary = {
        **training,
        "stage": STAGE,
        "runtime_algorithm": "MaskablePPO",
        "observation_mode": "Point",
        "parent_rl_seed": rl_seed,
        "config_id": config.name,
        "mask_training_audit": mask_training,
        "model_manifest": manifest,
        "performance_diagnostics_not_pass_gates":
            performance_diagnostics_not_pass_gates,
    }
    provenance = {
        "HEAD": head,
        "failed_predecessor": failed_before,
        "P2_2D_4B": stage4b,
        "API_contract": api,
        "static_contract": static,
        "pinned_predecessor_blobs": PINNED_BLOBS,
    }

    content = {
        "protocol_manifest.json": encode(protocol_manifest),
        "recovery_provenance.json": encode(provenance),
        "masked_smoke_training_summary.json": encode(training_summary),
        "masked_smoke_evaluation_metrics.csv": csv_bytes(metrics),
        "training_episode_assignments.csv": csv_bytes(assignments),
        "recovered_masked_ppo_construction.json":
            (OUTPUT / "recovered_masked_ppo_construction.json").read_bytes(),
    }
    for name, data in content.items():
        if name == "recovered_masked_ppo_construction.json":
            continue
        write_exclusive(OUTPUT / name, data)

    hashes = {
        name: sha_bytes(data)
        for name, data in content.items()
    }
    hashes["final_masked_smoke_model.zip"] = training["model_sha256"]

    output_hashes = encode(
        {
            "hashes": hashes,
            "exclusions": {
                "output_hashes.json":
                    "SHA256 pinned in audit_summary; avoid self-reference",
                "audit_summary.json":
                    "Published last PASS marker",
            },
        }
    )
    write_exclusive(OUTPUT / "output_hashes.json", output_hashes)

    expected_inventory = (
        set(content)
        | {
            "final_masked_smoke_model.zip",
            "output_hashes.json",
        }
    )
    gate(
        "output_inventory_exact",
        {p.name for p in OUTPUT.iterdir()}
        == expected_inventory,
    )
    gate(
        "source_HEAD_and_worktree_unchanged",
        git("rev-parse", "HEAD") == head
        and git("status", "--porcelain", "--untracked-files=all") == "",
    )
    gate(
        "audit_summary_published_last",
        not (OUTPUT / "audit_summary.json").exists(),
    )
    require(all(gates.values()), "All P2-2D-5AR structural gates")

    audit = {
        "stage": STAGE,
        "mode": "smoke",
        "stage_pass": True,
        "scientific_status": STATUS,
        "declaration": DECLARATION,
        "HEAD": head,
        "gates": gates,
        "failed_gates": [],
        "runtime_algorithm": "MaskablePPO",
        "PPO_training_runs": 1,
        "Masked_Point_training_runs": 1,
        "UA_training_runs": 0,
        "training_timesteps": 4096,
        "scientific_budget_used": False,
        "training_shield_interventions":
            mask_training["total_shield_interventions"],
        "training_forced_clean":
            mask_training["total_forced_clean"],
        "training_voluntary_clean":
            mask_training["total_voluntary_clean"],
        "training_free_choice_decisions":
            mask_training["total_free_choice_decisions"],
        "evaluation_shield_interventions":
            sum(row["N_shield_interventions"] for row in metrics),
        "formal_RL_access_count": 0,
        "formal_perception_seed_access_count": 0,
        "RANDOM_TEST_access_count": 0,
        "SEALED_DATES_access_count": 0,
        "performance_diagnostics_not_pass_gates":
            performance_diagnostics_not_pass_gates,
        "output_sha256": {
            **hashes,
            "output_hashes.json": sha_bytes(output_hashes),
        },
        "audit_summary_published_last": True,
    }
    write_exclusive(OUTPUT / "audit_summary.json", encode(audit))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke",))
    parser.parse_args()
    print(json.dumps(run_smoke(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
