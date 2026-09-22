
"""P2-2D-5A-v1: q50 safety-action-masked Point-PPO engineering smoke audit.

Scientific role
---------------
This is an ENGINEERING SMOKE only.  It does not select an algorithm or make a
performance claim.

P2-2D-3R showed deterministic post-hoc-shield Point-PPO collapsed to
always-WAIT + shield.  P2-2D-4AR confirmed exact proposal/execution aliasing.
P2-2D-4B decomposed the safe-action counterfactuals and showed:
    163/163 immediate-economic distinction,
    153/163 horizon-cost distinction,
    63 CLEAN-beneficial, 90 WAIT-beneficial, 10 ties.

This stage therefore integrates PRE-ACTION q50 safety masking while retaining
the identical frozen q50 shield as a runtime fail-safe.

Nonterminal action mask:
    WAIT safe   -> [True, True]
    WAIT unsafe -> [False, True]
Terminal day 364:
    [True, True]

Point observation, BLOCK10 v1 perception, CONFIG_A PPO hyperparameters,
reward/physics/cleaning mechanics and q50 shield safety set are unchanged.

The only new runtime dependency is sb3-contrib==2.7.1, paired exactly with the
already-frozen stable-baselines3==2.7.1.

Import is inert.  Only --mode smoke executes training.
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

SELF = "experiments/run_paper2_stage2d5a_masked_point_ppo_smoke_audit_v1.py"
MASKED_RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1.py"
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

CHECKPOINT = "2e053879dee9ad1df15aaff3956ea641274022e7"
PINNED_BLOBS = {
    SHIELDED_RUNNER: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    PERCEPTION_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
    STAGE4B_SOURCE: "e5b4006c9ddb1866e78df2335ccc855de2253b98",
    BASE_REQUIREMENTS: "430357153fbdc6b2b77d92d8040333d2de30135b",
}

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
STAGE4B = BASE / "p2_2d_4b_safe_action_distinction_decomposition_audit_v1/formal"
STAGE_DIRECTORY = BASE / "p2_2d_5a_masked_point_ppo_smoke_audit_v1"
OUTPUT = STAGE_DIRECTORY / "smoke"

STAGE = "P2-2D-5A-v1"
STATUS = "MASKED_POINT_PPO_SMOKE_PASS"
DECLARATION = (
    "ENGINEERING Q50 SAFETY-ACTION-MASKED POINT-PPO SMOKE ONLY; "
    "MASKABLEPPO 2.7.1 USES THE FROZEN Q50-HISTORY SAFETY SET BEFORE ACTION "
    "SELECTION, WHILE THE UNCHANGED FROZEN SHIELD REMAINS A FAIL-SAFE; "
    "NO SCIENTIFIC MODEL SELECTION OR PERFORMANCE CLAIM"
)

EXPECTED_4B_HEAD = CHECKPOINT
EXPECTED_4B_OUTPUT_SHA256 = {
    "protocol_manifest.json":
        "059dc4fc4b3f62a81015839341bc38b87d6d47b0b55bbe662eeff9e7c9cc7f28",
    "predecessor_provenance.json":
        "8f9e7dbbb323cadc284e60168a6a6a8468a92b73bda6fd54f14bb9bd68665763",
    "distinction_summary.json":
        "ab812841687ee47bd778ed9a0f878d8ae186352fb866b31a72a46cbb0dcf4178",
    "distinction_row_audit.csv":
        "5debd51437fc551181126443b8be24d169e4b84f4059918ffae2954b0aaa015d",
    "distinction_crosstab.csv":
        "f69fbc48beefba8fb194aab7044faae032e13a1393b7e9adae032b037bfbabb9",
    "output_hashes.json":
        "820139e384d374046025d7708f11d19f47d2ebf72de15732883fc7a61d9ec802",
}


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
    require(rows, "Nonempty CSV")
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


def verify_4b_predecessor(head):
    audit_path = STAGE4B / "audit_summary.json"
    summary_path = STAGE4B / "distinction_summary.json"
    require(
        audit_path.exists() and summary_path.exists(),
        "P2-2D-4B predecessor outputs exist",
    )
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    summary = json.loads(summary_path.read_bytes())

    require(
        audit["stage"] == "P2-2D-4B-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "SAFE_ACTION_DISTINCTION_DECOMPOSITION_AUDIT_COMPLETE"
        and audit["HEAD"] == EXPECTED_4B_HEAD
        and audit["failed_gates"] == []
        and audit["counterfactual_anchor_count"] == 163
        and audit["immediate_distinguishable_count"] == 163
        and audit["physical_distinguishable_count"] == 66
        and audit["economic_distinguishable_count"] == 163
        and audit["joint_physical_and_economic_count"] == 66
        and audit["horizon_cost_distinguishable_count"] == 153
        and audit["clean_beneficial_count"] == 63
        and audit["wait_beneficial_count"] == 90
        and audit["tie_count"] == 10
        and audit["PPO_training_runs"] == 0
        and audit["PPO_model_loads"] == 0
        and audit["environment_runtime_calls"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-2D-4B authority",
    )
    require(
        summary["overall"]["economic_distinguishable_count"] == 163
        and summary["overall"]["horizon_cost_distinguishable_count"] == 153
        and summary["overall"]["clean_beneficial_count"] == 63
        and summary["overall"]["wait_beneficial_count"] == 90
        and summary["overall"]["tie_count"] == 10
        and summary["scientific_diagnostic"][
            "all_immediate_economic_actions_distinguishable"
        ]
        is True
        and summary["scientific_diagnostic"][
            "any_clean_beneficial_anchor"
        ]
        is True,
        "P2-2D-4B route rationale exact",
    )
    require(
        audit["output_sha256"] == EXPECTED_4B_OUTPUT_SHA256,
        "P2-2D-4B hash registry exact",
    )
    for name, digest in EXPECTED_4B_OUTPUT_SHA256.items():
        require(
            sha_file(STAGE4B / name) == digest,
            "P2-2D-4B output hash: " + name,
        )

    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned frozen source unchanged: " + name,
        )

    return {
        "audit_summary_sha256": sha_bytes(audit_bytes),
        "output_sha256": EXPECTED_4B_OUTPUT_SHA256,
        "counterfactual_anchor_count": 163,
        "economic_distinguishable_count": 163,
        "horizon_cost_distinguishable_count": 153,
        "clean_beneficial_count": 63,
        "wait_beneficial_count": 90,
        "tie_count": 10,
    }


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
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
    require(len(train_calls) == 1, "Exactly one masked smoke training call")
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
        "Masked runtime entry points required",
    )
    require(
        "build_shielded_scheduled_perception_env" not in attrs
        and "build_shielded_perception_ppo_model" not in attrs
        and "evaluate_shielded_perception_deterministic" not in attrs
        and "build_perception_ppo_model" not in attrs,
        "No post-hoc/unmasked model runtime bypass",
    )
    require(
        "learn" not in attrs
        and "save" not in attrs
        and "load" not in attrs,
        "No direct learn/save/load bypass",
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
                )
            ):
                protected.append(value.value)
    require(not protected, "No protected role call literal")

    masked_runtime = {
        "build_masked_scheduled_perception_env",
        "evaluate_masked_perception_deterministic",
    }
    for node in calls:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in masked_runtime
        ):
            literals = [
                value.value
                for value in list(node.args)
                + [kw.value for kw in node.keywords]
                if isinstance(value, ast.Constant)
                and value.value in ("Point", "UA")
            ]
            require("UA" not in literals, "Point-only smoke runtime")

    require(
        "total_shield_interventions" in source
        and "total_forced_clean" in source
        and "total_voluntary_clean" in source
        and "total_free_choice_decisions" in source
        and "performance_diagnostics_not_pass_gates" in source,
        "Mask/shield accounting and no-performance-gate contract",
    )

    return {
        "single_masked_smoke_training_call": True,
        "masked_runtime_only": True,
        "no_posthoc_or_unmasked_model_bypass": True,
        "no_direct_learn_save_load": True,
        "no_protected_role_call": True,
        "Point_only_runtime": True,
        "mask_shield_accounting_present": True,
        "performance_diagnostics_not_pass_gates": True,
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
    predecessor = verify_4b_predecessor(head)
    predecessor_hash_before = predecessor["audit_summary_sha256"]

    # Both new sources and the supplemental dependency pin must be committed.
    for name in (SELF, MASKED_RUNNER, MASKED_REQUIREMENTS):
        blob = git("rev-parse", "--verify", f"{head}:{name}")
        require(
            blob
            == git("hash-object", f"--path={name}", name)
            == git("rev-parse", f":{name}"),
            "New masked-PPO source committed unchanged: " + name,
        )

    sources = {}
    for name in (
        SELF,
        MASKED_RUNNER,
        MASKED_REQUIREMENTS,
        SHIELDED_RUNNER,
        PERCEPTION_RUNNER,
        SHIELD,
        PROTOCOL,
        CORE,
        TRUE_RUNNER,
        STAGE4B_SOURCE,
        BASE_REQUIREMENTS,
    ):
        capture_source(sources, ROOT / name)

    static = static_contract()

    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as version_authority
    import paper2_ppo_protocol_v1 as frozen
    import paper2_masked_perception_ppo_runner_v1 as runner

    base_versions = version_authority.dependency_versions()
    masked_versions = runner.masked_dependency_versions()
    api_contract = runner.verify_maskable_api_contract()

    p = runner.get_protocol()
    require(
        base_versions == dict(p.dependencies),
        "Frozen base dependency versions",
    )
    require(
        masked_versions
        == {
            "stable-baselines3": "2.7.1",
            "sb3-contrib": "2.7.1",
        },
        "Exact masked runtime dependency pair",
    )
    require(
        base_versions["stable_baselines3"] == "2.7.1"
        and masked_versions["stable-baselines3"] == "2.7.1",
        "SB3 dependency identity across frozen/masked runtime",
    )
    require(
        frozen.protocol_payload(p)
        == frozen.protocol_payload(frozen.Protocol()),
        "Exact frozen PPO protocol",
    )
    require(
        runner.inherited_config_id() == "CONFIG_A"
        == p.candidates[0].name,
        "Inherited CONFIG_A; no reselection",
    )
    require(
        dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY")
        and dict(p.observation_dimensions)["Point"] == 3,
        "Frozen Point observation",
    )
    authority = runner.read_shield_authority()
    require(
        authority.feasibility_audit[
            "support_compatible_shield_feasibility_pass"
        ]
        is True,
        "Frozen q50 shield authority",
    )

    assets = runner.load_assets()
    assets.ensure_perception()
    for name, path in (
        ("paper1_cqr", assets.paper1_cqr),
        ("wapp_power_bridge", assets.wapp_power_bridge_path),
    ):
        expected = assets.perception.EXPECTED_HASHES[name]
        require(
            sha_file(path) == expected,
            "Frozen perception input: " + name,
        )
        capture_source(sources, path)

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        return execute_smoke(
            runner=runner,
            assets=assets,
            protocol_payload=frozen.protocol_payload(p),
            head=head,
            base_versions=base_versions,
            masked_versions=masked_versions,
            api_contract=api_contract,
            static=static,
            sources=sources,
            predecessor=predecessor,
            predecessor_hash_before=predecessor_hash_before,
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
                            "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/MASK/SHIELD/PERCEPTION/PPO TUNING",
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
    base_versions,
    masked_versions,
    api_contract,
    static,
    sources,
    predecessor,
    predecessor_hash_before,
):
    gates = {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate: " + name)
        gates[name] = bool(condition)
        require(condition, name)

    p = runner.get_protocol()
    config = runner.candidate(runner.inherited_config_id())
    rl_seed = p.seeds.development[0]

    gate(
        "P2_2D_4B_mechanism_predecessor_exact",
        predecessor["economic_distinguishable_count"] == 163
        and predecessor["horizon_cost_distinguishable_count"] == 153
        and predecessor["clean_beneficial_count"] == 63
        and predecessor["wait_beneficial_count"] == 90,
    )
    gate("static_smoke_contract", all(static.values()))
    gate(
        "masked_dependency_and_API_contract",
        masked_versions["stable-baselines3"] == "2.7.1"
        and masked_versions["sb3-contrib"] == "2.7.1"
        and api_contract["MaskablePPO_learn_default_use_masking"] is True
        and api_contract["MaskablePPO_predict_accepts_action_masks"] is True,
    )
    gate(
        "frozen_Point_CONFIG_A_smoke_contract",
        config.name == "CONFIG_A"
        and config.n_steps == 2048
        and config.learning_rate == 3e-4
        and runner.DIAGNOSTIC_SMOKE_BUDGET == 4096
        and dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY"),
    )

    ledger = runner.AccessLedger()
    env = runner.build_masked_scheduled_perception_env(
        assets=assets,
        rl_seed=rl_seed,
        ledger=ledger,
        config_id=config.name,
        observation_mode="Point",
    )
    training_assets = env.assets

    try:
        gate(
            "fresh_masked_Point_training_environment",
            env.observation_mode == "Point"
            and env.observation_space.shape == (3,)
            and env.action_space.n == 2
            and env.action_space.start == 0
            and env.schedule.parent_seed == rl_seed
            and not env.started
            and not env.failed
            and not env.assignments,
        )

        bundle = runner.build_masked_perception_ppo_model(
            env, config.name, rl_seed
        )
        gate(
            "frozen_PPO_hyperparameters_with_MaskablePPO_runtime",
            runner.verify_masked_model_contract(
                bundle.model, config.name, rl_seed
            )
            and bundle.model.num_timesteps == 0
            and not bundle.training_finished
            and not p.observation_normalization
            and not p.reward_normalization,
        )

        construction = {
            "config_id": config.name,
            "runtime_algorithm": "MaskablePPO",
            "stable_baselines3_version":
                masked_versions["stable-baselines3"],
            "sb3_contrib_version": masked_versions["sb3-contrib"],
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "diagnostic_smoke_budget":
                runner.DIAGNOSTIC_SMOKE_BUDGET,
            "common": dict(p.common),
            "architecture": dict(p.architecture),
            "parent_rl_seed": rl_seed,
            "child_seeds": runner.child_seeds(rl_seed),
            "observation_mode": "Point",
            "policy_observation":
                list(dict(p.observations)["Point"]),
            "mask_input":
                "frozen q50-history set-membership shield belief only",
            "mask_semantics": {
                "WAIT_safe": [True, True],
                "WAIT_unsafe": [False, True],
                "terminal_day_364": [True, True],
            },
            "frozen_shield_remains_runtime_failsafe": True,
            "reward_shaping": False,
            "scientific_budget_used": False,
        }
        write_exclusive(
            OUTPUT / "masked_ppo_construction_contract.json",
            encode(construction),
        )

        training = runner.train_masked_final_checkpoint(
            bundle,
            OUTPUT / "final_masked_smoke_model.zip",
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

        mask_training = env._mask_audit_snapshot()
        gate(
            "training_zero_posthoc_shield_interventions",
            mask_training["total_shield_interventions"] == 0
            and mask_training["mask_shield_intervention_invariant"] is True
            and mask_training["total_proposed_clean"]
            == mask_training["total_executed_clean"],
        )
        gate(
            "training_mask_decision_partition",
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
            + mask_training["total_terminal_clean"]
            and mask_training["total_terminal_passthrough"]
            == mask_training["total_terminal_clean"]
            + mask_training["total_terminal_wait"],
        )
        gate(
            "training_exercises_forced_and_free_choice_masks",
            mask_training["total_forced_clean"] > 0
            and mask_training["total_free_choice_decisions"] > 0,
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
            len(
                [r for r in ledger.records if r["role"] == "TRAINING"]
            )
            == len(env.assignments)
            and all(
                r["environment_root"]
                in runner.partition("TRAINING").roots
                and r["observation_mode"] == "Point"
                for r in ledger.records
                if r["role"] == "TRAINING"
            ),
        )

        loaded = runner.reload_masked_final_checkpoint(
            OUTPUT / "final_masked_smoke_model.zip",
            bundle,
        )
        gate(
            "masked_save_reload_exact_and_finite",
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
            training_assets,
            specs,
            ledger,
            "Point",
        )
        gate(
            "fixed_DEVELOPMENT_masked_deterministic_evaluation",
            len(metrics) == 2
            and [
                (r["year"], r["trajectory_id"])
                for r in metrics
            ]
            == list(specs)
            and all(
                r["step_count"] == p.decision_reward_steps
                and r["natural_transition_count"]
                == p.natural_transitions
                and r["perception_query_count"]
                == p.decision_reward_steps
                and r["N_shield_interventions"] == 0
                and r["N_forced_clean"] + r["N_free_choice"]
                == p.natural_transitions
                and r["terminal_mask"] == [True, True]
                for r in metrics
            ),
        )
        gate(
            "finite_eval_reward_cost_and_mask_accounting",
            all(
                r["all_observations_finite"]
                and r["all_rewards_finite"]
                and r["all_states_finite"]
                and r["reward_sign_regression_diff"]
                <= p.selection.atol
                and 0.0 <= r["forced_clean_fraction"] <= 1.0
                and 0.0 <= r["free_choice_fraction"] <= 1.0
                and 0.0
                <= r["voluntary_clean_fraction_given_free"]
                <= 1.0
                for r in metrics
            ),
        )

        assignments = [dict(row) for row in env.assignments]
        training_query_count = env.perception_query_count
        model_manifest = runner.masked_model_manifest(
            bundle, training
        )
    finally:
        env.close()

    allowed_roots = (
        set(runner.partition("TRAINING").roots)
        | set(runner.partition("DEVELOPMENT").roots)
    )
    gate(
        "zero_formal_or_protected_access",
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

    # Scientific smoke diagnostics are deliberately NOT gates.
    total_eval_forced = sum(r["N_forced_clean"] for r in metrics)
    total_eval_voluntary = sum(
        r["N_voluntary_clean"] for r in metrics
    )
    total_eval_free = sum(r["N_free_choice"] for r in metrics)
    performance_diagnostics_not_pass_gates = {
        "training_forced_clean_count":
            mask_training["total_forced_clean"],
        "training_voluntary_clean_count":
            mask_training["total_voluntary_clean"],
        "training_free_choice_count":
            mask_training["total_free_choice_decisions"],
        "eval_forced_clean_count": total_eval_forced,
        "eval_voluntary_clean_count": total_eval_voluntary,
        "eval_free_choice_count": total_eval_free,
        "eval_mean_J_total":
            sum(r["J_total"] for r in metrics) / len(metrics),
        "role":
            "DESCRIPTIVE ENGINEERING-SMOKE DIAGNOSTICS ONLY; NOT A PASS GATE",
    }

    # Verify P2-2D-4B evidence remains unchanged after the run.
    predecessor_after = verify_4b_predecessor(head)
    gate(
        "P2_2D_4B_predecessor_preserved",
        predecessor_after["audit_summary_sha256"]
        == predecessor_hash_before,
    )

    protocol_manifest = {
        "stage": STAGE,
        "mode": "smoke",
        "declaration": DECLARATION,
        "protocol": protocol_payload,
        "selected_config": config.name,
        "runtime_algorithm": "MaskablePPO",
        "masked_dependencies": masked_versions,
        "development_eval_specs":
            [list(x) for x in specs],
        "policy_observation":
            ["q50", "sin_DOY", "cos_DOY"],
        "action_semantics":
            "frozen q50 mask -> MaskablePPO action -> unchanged frozen q50 shield fail-safe -> frozen reward/physics",
        "mask_information":
            ["q50_float32_history", "executed_action_history"],
        "mask_forbidden_inputs":
            [
                "current latent L",
                "UA width",
                "rain flag",
                "future weather",
            ],
        "terminal_day_semantics":
            "day 364 mask=[True,True]; no t->t+1 safety transition",
        "trajectory_access_log": ledger.records,
        "static_contract": static,
        "scientific_budget_used": False,
        "performance_diagnostics_as_gate": False,
        "prohibitions": [
            "UA training",
            "hyperparameter reselection",
            "reward shaping",
            "perception extension",
            "shield/kernel tuning",
            "retry/reseed",
            "formal held-out access",
            "RANDOM_TEST access",
            "SEALED_DATES access",
        ],
    }

    training_summary = {
        **training,
        "model_path": "final_masked_smoke_model.zip",
        "config_id": config.name,
        "runtime_algorithm": "MaskablePPO",
        "parent_rl_seed": rl_seed,
        **runner.child_seeds(rl_seed),
        "observation_mode": "Point",
        "perception_seed":
            runner.get_perception_contract().development_seed,
        "perception_query_count": training_query_count,
        "scientific_budget_used": False,
        "model_manifest": model_manifest,
        "mask_training_audit": mask_training,
        "performance_diagnostics_not_pass_gates":
            performance_diagnostics_not_pass_gates,
    }

    source_manifest = {
        "HEAD": head,
        "pre_mask_checkpoint": CHECKPOINT,
        "source_sha256": sources,
        "pinned_predecessor_blobs": PINNED_BLOBS,
        "base_dependencies": base_versions,
        "masked_dependencies": masked_versions,
        "maskable_API_contract": api_contract,
        "core_assets_provenance": training_assets.provenance,
        "P2_2D_4B_predecessor": predecessor,
    }

    documents = {
        "masked_smoke_training_summary.json": training_summary,
        "protocol_manifest.json": protocol_manifest,
        "source_artifact_hashes.json": source_manifest,
    }
    content = {
        name: encode(doc) for name, doc in documents.items()
    }
    content["masked_smoke_evaluation_metrics.csv"] = csv_bytes(
        metrics
    )
    content["training_episode_assignments.csv"] = csv_bytes(
        assignments
    )

    for name, data in content.items():
        write_exclusive(OUTPUT / name, data)

    construction_bytes = encode(construction)
    require(
        (OUTPUT / "masked_ppo_construction_contract.json").read_bytes()
        == construction_bytes,
        "Construction contract exact readback",
    )
    content["masked_ppo_construction_contract.json"] = (
        construction_bytes
    )

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
    write_exclusive(
        OUTPUT / "output_hashes.json",
        output_hashes,
    )

    expected_inventory = (
        set(content)
        | {
            "final_masked_smoke_model.zip",
            "output_hashes.json",
        }
    )
    gate(
        "output_inventory_and_hashes",
        {p.name for p in OUTPUT.iterdir()}
        == expected_inventory
        and all(
            (OUTPUT / name).read_bytes() == data
            for name, data in content.items()
        )
        and json.loads(output_hashes)["hashes"]
        == {
            name: runner.file_sha256(OUTPUT / name)
            for name in hashes
        },
    )
    gate(
        "source_HEAD_and_worktree_unchanged",
        git("rev-parse", "HEAD") == head
        and git("status", "--porcelain", "--untracked-files=all")
        == "",
    )
    gate(
        "audit_summary_published_last",
        not (OUTPUT / "audit_summary.json").exists(),
    )
    require(all(gates.values()), "All masked smoke structural gates")

    audit_summary = {
        "stage": STAGE,
        "mode": "smoke",
        "stage_pass": True,
        "scientific_status": STATUS,
        "declaration": DECLARATION,
        "HEAD": head,
        "gates": gates,
        "failed_gates": [],
        "runtime_algorithm": "MaskablePPO",
        "selected_config": config.name,
        "PPO_training_runs": 1,
        "Masked_Point_training_runs": 1,
        "UA_training_runs": 0,
        "scientific_budget_used": False,
        "training_timesteps":
            runner.DIAGNOSTIC_SMOKE_BUDGET,
        "training_shield_interventions":
            mask_training["total_shield_interventions"],
        "training_forced_clean":
            mask_training["total_forced_clean"],
        "training_voluntary_clean":
            mask_training["total_voluntary_clean"],
        "training_free_choice_decisions":
            mask_training["total_free_choice_decisions"],
        "evaluation_shield_interventions":
            sum(r["N_shield_interventions"] for r in metrics),
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
    write_exclusive(
        OUTPUT / "audit_summary.json",
        encode(audit_summary),
    )
    return audit_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("smoke",),
    )
    parser.parse_args()
    print(json.dumps(run_smoke(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
