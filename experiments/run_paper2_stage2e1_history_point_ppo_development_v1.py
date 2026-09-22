
"""P2-2E-1-v1: Masked History-Point PPO DEVELOPMENT characterization.

Scientific question
-------------------
P2-2D-5B showed that static Point observation

    [q50_t, sin_DOY_t, cos_DOY_t]

collapsed to WAIT under the frozen masked DEVELOPMENT protocol.
P2-2D-6A showed that adding instantaneous width did not recover active
decision behavior. P2-2E-0/0R then froze and validated the compact temporal
candidate family.

This stage tests ONLY the marginal value of compact temporal/control history:

    History-Point =
    [q50_t, delta_q50_t, tau_t, sin_DOY_t, cos_DOY_t]

where tau_t is the frozen normalized time since the most recent executed
in-episode CLEAN, or since episode start before the first in-episode CLEAN.

Everything else remains frozen:
- MaskablePPO / CONFIG_A
- DEVELOPMENT seeds 510001 and 510002
- 491,520 training steps per seed
- TRAINING / DEVELOPMENT populations
- BLOCK10-v1 Point perception
- q50-history safety mask and q50 shield
- reward, physics, network, optimizer and evaluation protocol.

No width or delta-width is available to the History-Point policy. The safety
mask/shield also remains completely temporal/UQ-free.

Each History-Point DEVELOPMENT episode is paired read-only with:
1) frozen static Point P2-2D-5B;
2) frozen static UA P2-2D-6A;
3) the frozen shield-only reference carried by P2-2D-5B.

Performance is descriptive only and is NEVER a structural PASS gate.
No retry/reseed/tuning, FINAL-seed training, formal-held-out, RANDOM_TEST or
SEALED_DATES access is allowed.

Import is inert. Only --mode formal performs training/evaluation.
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
import random
import statistics
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SELF = "experiments/run_paper2_stage2e1_history_point_ppo_development_v1.py"
TEMPORAL = "experiments/paper2_temporal_observation_v1.py"
RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1_1.py"
MASK_BASE = "experiments/paper2_masked_perception_ppo_runner_v1.py"
STAGE0 = (
    "experiments/"
    "run_paper2_stage2e0_temporal_partial_observability_design_freeze_v1.py"
)
STAGE0R = (
    "experiments/"
    "run_paper2_stage2e0r_temporal_partial_observability_design_recovery_v1.py"
)
STAGE5B = "experiments/run_paper2_stage2d5b_masked_point_ppo_development_v1.py"
STAGE6A = "experiments/run_paper2_stage2d6a_masked_ua_ppo_development_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
PERCEPTION_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
SHIELDED_RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
MASKED_REQUIREMENTS = "requirements-paper2-masked-v1.txt"
BASE_REQUIREMENTS = "requirements.txt"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
STAGE0R_FORMAL = (
    BASE / "p2_2e_0r_temporal_partial_observability_design_recovery_v1/formal"
)
POINT5B = BASE / "p2_2d_5b_masked_point_ppo_development_v1/formal"
UA6A = BASE / "p2_2d_6a_masked_ua_ppo_development_v1/formal"

STAGE_DIRECTORY = BASE / "p2_2e_1_history_point_ppo_development_v1"
OUTPUT = STAGE_DIRECTORY / "formal"

CHECKPOINT = "034076106fc6e1aa4165b0e780c2dd9bb449e16b"
STAGE0R_HEAD = CHECKPOINT

DEVELOPMENT_SEEDS = (510001, 510002)
DEVELOPMENT_BUDGET = 491520
DEVELOPMENT_EPISODES = 600
ATOL = 1e-12

PINNED_BLOBS = {
    TEMPORAL: "fec506636e53855be4cd79d8dd6702e9b71c674d",
    RUNNER: "9b1777ce4262efb95c677432ea52997352d76750",
    MASK_BASE: "be86a5e8a8a162c0d6be720c5340914ccc1ed82e",
    STAGE0: "c4e4baf44d675e69440fbc2cc9b9af6a358f0ff1",
    STAGE0R: "469ae2fd728c1fd2648841d0f5b7792716d0837e",
    STAGE6A: "9954d5b41444084697ca8fb92c1372f158160f31",
}

EXPECTED_0R_OUTPUT_HASHES = {
    "recovery_summary.json":
        "2e71d10713ae710fc0c17462d93cf2ab33a3faa998b12d008a91c4925e27d1d0",
    "corrected_construction_trace.csv":
        "366a7087d37f768d5959bc19a31432c428001b7fb5987bc10224346d81139975",
    "source_artifact_provenance.json":
        "a2d438b5cd1535b6bcb5d71f9d60767cbbbde12ed79372fd5cb64608dab8538f",
    "output_hashes.json":
        "2e09be69e1b38a12f26b702fdf178f990b62fec28efa1cc1a77fe6a1f46340b2",
}

STAGE = "P2-2E-1-v1"
STATUS = "HISTORY_POINT_PPO_DEVELOPMENT_CHARACTERIZATION_PASS"

SCHEDULE_FIELDS = (
    "episode_ordinal",
    "cycle_index",
    "position_in_cycle",
    "year",
    "trajectory_id",
    "environment_root",
    "population_size",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows):
    require(
        rows and all(set(row) == set(rows[0]) for row in rows),
        "Nonempty consistent CSV schema",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_exclusive(path, data):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, "Exact readback: " + path.name)


def verify_pinned_sources(head):
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned predecessor source changed: " + name,
        )


def verify_0r_authority():
    audit_path = STAGE0R_FORMAL / "audit_summary.json"
    require(audit_path.exists(), "P2-2E-0R audit exists")
    audit = json.loads(audit_path.read_bytes())

    require(
        audit["stage"] == "P2-2E-0R-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "TEMPORAL_PARTIAL_OBSERVABILITY_DESIGN_RECOVERY_PASS_FROZEN"
        and audit["HEAD"] == STAGE0R_HEAD
        and audit["failed_gates"] == []
        and all(audit["gates"].values())
        and audit["PPO_training_runs"] == 0
        and audit["PPO_model_loads"] == 0
        and audit["development_episode_pairs_audited"] == 4
        and audit["corrected_construction_trace_rows"] == 1460
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0
        and audit["temporal_observation_design_changed"] is False
        and audit["safety_mask_or_shield_changed"] is False
        and audit["output_sha256"] == EXPECTED_0R_OUTPUT_HASHES,
        "Accepted P2-2E-0R authority",
    )
    for name, digest in EXPECTED_0R_OUTPUT_HASHES.items():
        require(
            sha_file(STAGE0R_FORMAL / name) == digest,
            "P2-2E-0R output hash: " + name,
        )

    recovery = json.loads(
        (STAGE0R_FORMAL / "recovery_summary.json").read_bytes()
    )
    contract = recovery["temporal_observation_contract"]
    require(
        recovery["future_experiment_order"]
        == [
            "P2-2E-1 History-Point DEVELOPMENT",
            "P2-2E-2 History-UA DEVELOPMENT",
        ]
        and contract["modes"]["History-Point"]["features"]
        == [
            "q50",
            "delta_q50",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ]
        and contract["modes"]["History-Point"]["dimension"] == 5
        and contract["modes"]["History-UA"]["features"]
        == [
            "q50",
            "delta_q50",
            "width",
            "delta_width",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ]
        and contract["modes"]["History-UA"]["dimension"] == 7
        and contract["history_window_hyperparameter"] is None
        and contract["pre_episode_clean_age_imputed"] is False
        and contract["future_information_used"] is False
        and contract["latent_state_used"] is False
        and contract["safety_rule_changed"] is False,
        "Frozen temporal candidate family",
    )
    return sha_file(audit_path)


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }

    trains = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "train_masked_final_checkpoint"
    ]
    require(
        len(trains) == 1
        and len(trains[0].keywords) == 1
        and trains[0].keywords[0].arg == "purpose"
        and isinstance(trains[0].keywords[0].value, ast.Constant)
        and trains[0].keywords[0].value.value == "development",
        "Exactly one frozen DEVELOPMENT training call site",
    )
    require(
        "build_masked_scheduled_perception_env" in attrs
        and "build_masked_perception_eval_env" in attrs
        and "wrap_temporal_masked_env" in attrs,
        "Frozen masked Point base plus temporal wrapper runtime",
    )
    require(
        "learn" not in attrs
        and "save" not in attrs
        and "load" not in attrs,
        "No direct learn/save/load bypass",
    )

    # Training/evaluation base perception must be Point only. History-Point
    # policy features are supplied by the frozen temporal wrapper.
    for node in calls:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in (
                "build_masked_scheduled_perception_env",
                "build_masked_perception_eval_env",
            )
        ):
            literals = [
                value.value
                for value in (
                    list(node.args)
                    + [kw.value for kw in node.keywords]
                )
                if isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ]
            require(
                "UA" not in literals,
                "History-Point runtime cannot construct UA perception",
            )

    require(
        "performance_diagnostics_not_pass_gates" in source
        and "paired_comparison_descriptive_only" in source
        and "zero_training_shield_interventions" in source
        and "zero_evaluation_shield_interventions" in source,
        "Scientific outcomes separated from structural PASS gates",
    )

    import paper2_temporal_observation_v1 as temporal

    contract = temporal.temporal_contract()
    require(
        contract["modes"][temporal.HISTORY_POINT]["features"]
        == [
            "q50",
            "delta_q50",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ]
        and contract["modes"][temporal.HISTORY_POINT]["dimension"] == 5
        and contract["pre_episode_clean_age_imputed"] is False
        and contract["future_information_used"] is False
        and contract["latent_state_used"] is False,
        "Exact frozen History-Point contract",
    )

    # Mechanically prove the inherited mask does not read temporal/UQ fields.
    mask_source = (ROOT / MASK_BASE).read_text(encoding="utf-8")
    mask_tree = ast.parse(mask_source)
    method = None
    for item in mask_tree.body:
        if (
            isinstance(item, ast.ClassDef)
            and item.name == "Q50MaskedShieldedPerceptionEnv"
        ):
            method = next(
                (
                    child for child in item.body
                    if isinstance(child, ast.FunctionDef)
                    and child.name == "_compute_action_mask"
                ),
                None,
            )
    require(method is not None, "Frozen action-mask method exists")
    tokens = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.add(node.value)
    require(
        "width" not in tokens
        and "delta_width" not in tokens
        and "delta_q50" not in tokens
        and "tau" not in tokens
        and "belief_lower" in tokens
        and "belief_upper" in tokens
        and "action_safe" in tokens,
        "Safety mask remains q50-belief-only",
    )

    return {
        "single_History_Point_development_training_call_site": True,
        "Point_base_plus_frozen_temporal_wrapper_runtime": True,
        "History_Point_no_UA_runtime": True,
        "no_direct_learn_save_load": True,
        "History_Point_feature_contract_exact": True,
        "q50_only_temporal_UQ_free_safety_mask": True,
        "zero_intervention_contract_present": True,
        "performance_and_pairing_diagnostics_not_pass_gates": True,
    }


def build_history_point_model(runner, temporal, env, config_id, rl_seed):
    """Construct MaskablePPO with frozen CONFIG_A on the 5-D History-Point env."""
    import torch
    from sb3_contrib import MaskablePPO

    runner.verify_maskable_api_contract()
    p, c = runner.get_protocol(), runner.candidate(config_id)

    require(
        isinstance(env, temporal.TemporalMaskedObservationWrapper)
        and env.temporal_mode == temporal.HISTORY_POINT
        and env.base_observation_mode == "Point"
        and env.observation_space.shape == (5,)
        and env.schedule.parent_seed == rl_seed
        and not env.started
        and not env.failed
        and config_id == runner.inherited_config_id(),
        "Fresh matching History-Point temporal training environment",
    )
    require(
        env.action_space.n == 2
        and env.action_space.start == 0
        and not p.observation_normalization
        and not p.reward_normalization,
        "Frozen action/normalization contract",
    )

    policy, compatible_common = runner._split_maskable_common()
    architecture = dict(p.architecture)
    require(
        policy == "MlpPolicy"
        and architecture["activation"] == "Tanh",
        "Frozen policy architecture",
    )

    learner = runner.child_seeds(rl_seed)["learner"]
    torch.set_num_threads(p.torch_num_threads)
    random.seed(learner)
    np.random.seed(learner)
    torch.manual_seed(learner)

    model = MaskablePPO(
        policy=policy,
        env=env,
        learning_rate=c.learning_rate,
        n_steps=c.n_steps,
        policy_kwargs={
            "net_arch": {
                "pi": list(architecture["actor"]),
                "vf": list(architecture["critic"]),
            },
            "activation_fn": torch.nn.Tanh,
            "ortho_init": architecture["ortho_init"],
        },
        seed=learner,
        verbose=0,
        **compatible_common,
    )
    require(
        model.observation_space == env.observation_space
        and model.observation_space.shape == (5,)
        and model.action_space == env.action_space
        and runner.verify_masked_model_contract(
            model, c.name, rl_seed
        ),
        "Frozen MaskablePPO contract on History-Point observation space",
    )
    return runner.ModelBundle(
        model=model,
        env=env,
        config_id=c.name,
        parent_seed=rl_seed,
    )


def corrected_schedule_audit(runner, env, seed, config_id):
    schedule = runner.build_training_schedule(seed, config_id)
    expected = [
        schedule.next_assignment()
        for _ in env.assignments
    ]
    actual_latent = [
        {field: row[field] for field in SCHEDULE_FIELDS}
        for row in env.assignments
    ]
    expected_latent = [
        {field: row[field] for field in SCHEDULE_FIELDS}
        for row in expected
    ]
    perception_seed = runner.get_perception_contract().development_seed
    require(
        actual_latent == expected_latent,
        "Exact frozen latent training schedule",
    )
    require(
        all(
            row["perception_seed"] == perception_seed
            for row in env.assignments
        ),
        "Frozen injected DEVELOPMENT perception seed",
    )
    require(
        all(row["perception_seed"] is None for row in expected),
        "Raw True-State schedule perception seed remains None",
    )
    return {
        "assignment_count": len(env.assignments),
        "latent_schedule_exact": True,
        "runtime_perception_seed_exact": True,
        "runtime_injected_perception_seed": perception_seed,
        "compared_latent_fields": list(SCHEDULE_FIELDS),
    }


def summarize(rows):
    require(rows, "Nonempty evaluation rows")
    costs = [float(row["J_total"]) for row in rows]
    forced = [int(row["N_forced_clean"]) for row in rows]
    voluntary = [int(row["N_voluntary_clean"]) for row in rows]
    free = [int(row["N_free_choice"]) for row in rows]
    ratios = [
        float(row["voluntary_clean_fraction_given_free"])
        for row in rows
    ]
    interventions = [
        int(row["N_shield_interventions"]) for row in rows
    ]
    clean = [int(row["N_clean"]) for row in rows]
    ordered = sorted(costs)
    cvar_count = math.ceil(0.05 * len(ordered))
    return {
        "episode_count": len(rows),
        "mean_J_total": float(statistics.fmean(costs)),
        "std_J_total":
            float(statistics.stdev(costs)) if len(rows) > 1 else 0.0,
        "median_J_total": float(statistics.median(costs)),
        "P95_J_total":
            float(np.quantile(costs, 0.95, method="linear")),
        "CVaR95_J_total":
            float(statistics.fmean(ordered[-cvar_count:])),
        "mean_N_clean": float(statistics.fmean(clean)),
        "mean_N_forced_clean":
            float(statistics.fmean(forced)),
        "mean_N_voluntary_clean":
            float(statistics.fmean(voluntary)),
        "mean_N_free_choice":
            float(statistics.fmean(free)),
        "mean_voluntary_clean_fraction_given_free":
            float(statistics.fmean(ratios)),
        "total_N_clean": int(sum(clean)),
        "total_N_forced_clean": int(sum(forced)),
        "total_N_voluntary_clean": int(sum(voluntary)),
        "total_N_free_choice": int(sum(free)),
        "total_shield_interventions":
            int(sum(interventions)),
        "episodes_with_any_voluntary_clean":
            int(sum(value > 0 for value in voluntary)),
        "episodes_with_zero_voluntary_clean":
            int(sum(value == 0 for value in voluntary)),
    }


def evaluate_history_point(
    runner,
    temporal,
    model,
    assets,
    specs,
    ledger,
):
    """Deterministic 600-episode DEVELOPMENT evaluation on History-Point."""
    p = runner.get_protocol()
    rows = []

    for year, tid in specs:
        base_env = runner.build_masked_perception_eval_env(
            assets, year, tid, ledger, "Point"
        )
        env = temporal.wrap_temporal_masked_env(
            base_env, temporal.HISTORY_POINT
        )
        try:
            obs, info = env.reset()
            require(
                obs.shape == (5,)
                and obs.dtype == np.float32
                and np.isfinite(obs).all()
                and env.observation_space.contains(obs),
                "Finite exact History-Point reset observation",
            )
            temporal_snap = env._temporal_audit_snapshot()
            public_obs = np.asarray(
                temporal_snap["previous_public_observation"],
                dtype=np.float32,
            )
            snap = runner.audit_perception_observation(
                base_env.inner, public_obs, info
            )

            rewards, costs, states = [], [], []
            actions, masks = [], []

            for index in range(p.decision_reward_steps):
                action_mask = env.action_masks()
                require(
                    action_mask.shape == (2,)
                    and action_mask.dtype == np.bool_
                    and bool(action_mask[1]),
                    "Valid frozen q50 action mask",
                )
                action_array, _ = model.predict(
                    obs,
                    deterministic=p.evaluation.deterministic,
                    action_masks=action_mask,
                )
                array = np.asarray(action_array)
                require(
                    array.size == 1
                    and np.issubdtype(array.dtype, np.integer),
                    "Integer scalar masked History-Point action",
                )
                action = int(array.item())
                require(
                    env.action_space.contains(action)
                    and bool(action_mask[action]),
                    "MaskablePPO action must satisfy frozen mask",
                )

                obs, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    public_after = np.zeros(
                        base_env.observation_space.shape,
                        dtype=base_env.observation_space.dtype,
                    )
                else:
                    temporal_snap = env._temporal_audit_snapshot()
                    public_after = np.asarray(
                        temporal_snap["previous_public_observation"],
                        dtype=np.float32,
                    )

                snap = runner.audit_perception_observation(
                    base_env.inner,
                    public_after,
                    info,
                    terminal=terminated,
                )
                parts = snap["last_reward_components"]
                decision = env.last_shield_decision

                require(
                    decision is not None
                    and decision["proposed_action"] == action
                    and decision["executed_action"] == parts["action"],
                    "Masked History-Point action/executed-action identity",
                )
                if index < p.natural_transitions:
                    require(
                        decision["intervention"] is False
                        and decision["proposed_safe"] is True
                        and decision["executed_action"] == action,
                        "Zero post-hoc shield intervention after masking",
                    )

                rewards.append(float(reward))
                costs.append(float(parts["total_cost"]))
                states.append(float(parts["L_pre"]))
                actions.append(action)
                masks.append(tuple(bool(value) for value in action_mask))

                require(
                    obs.shape == (5,)
                    and obs.dtype == np.float32
                    and np.isfinite(obs).all()
                    and env.observation_space.contains(obs)
                    and not truncated
                    and terminated
                    == (index == p.decision_reward_steps - 1),
                    "Finite History-Point observation and exact horizon",
                )

            episode_return = sum(rewards)
            total_cost = sum(costs)
            mask_audit = env._mask_audit_snapshot()
            record = mask_audit["episode_records"][-1]

            require(
                episode_return
                == snap["episode_return"]
                == info["episode_return"]
                and abs(episode_return + total_cost)
                <= p.selection.atol,
                "Frozen return/cost sign",
            )
            require(
                sum(actions)
                == snap["clean_count"]
                == info["clean_count"]
                == record["executed_clean_count"],
                "History-Point cleaning-count identity",
            )
            require(
                mask_audit["total_shield_interventions"] == 0
                and record["interventions"] == 0,
                "Zero shield intervention invariant",
            )
            require(
                record["masked_safety_decisions"]
                == p.natural_transitions
                == record["forced_clean_count"]
                + record["free_choice_decisions"]
                and record["free_choice_decisions"]
                == record["voluntary_clean_count"]
                + record["mask_free_wait_count"]
                and record["executed_clean_count"]
                == record["forced_clean_count"]
                + record["voluntary_clean_count"]
                + record["terminal_clean_count"],
                "Exact History-Point mask/action accounting",
            )

            free = record["free_choice_decisions"]
            rows.append(
                {
                    "year": year,
                    "trajectory_id": tid,
                    "environment_root":
                        base_env.inner.episode_spec.environment_root,
                    "step_count": len(actions),
                    "natural_transition_count":
                        snap["transition_count"],
                    "J_total": total_cost,
                    "episode_return": episode_return,
                    "reward_sign_regression_diff":
                        abs(episode_return + total_cost),
                    "N_clean": snap["clean_count"],
                    "N_forced_clean":
                        record["forced_clean_count"],
                    "N_voluntary_clean":
                        record["voluntary_clean_count"],
                    "N_free_wait":
                        record["mask_free_wait_count"],
                    "N_free_choice": free,
                    "N_terminal_clean":
                        record["terminal_clean_count"],
                    "N_shield_interventions":
                        record["interventions"],
                    "forced_clean_fraction":
                        record["forced_clean_count"]
                        / p.natural_transitions,
                    "free_choice_fraction":
                        free / p.natural_transitions,
                    "voluntary_clean_fraction_given_free":
                        (
                            record["voluntary_clean_count"] / free
                            if free else 0.0
                        ),
                    "mean_L_pre": float(np.mean(states)),
                    "max_L_pre": max(states),
                    "action_min": min(actions),
                    "action_max": max(actions),
                    "forced_mask_count":
                        sum(mask == (False, True) for mask in masks),
                    "free_mask_count":
                        sum(mask == (True, True) for mask in masks[:-1]),
                    "terminal_mask": list(masks[-1]),
                    "all_observations_finite": True,
                    "all_rewards_finite": True,
                    "all_states_finite": True,
                    "deterministic_action":
                        p.evaluation.deterministic,
                    "perception_query_count":
                        snap["perception_query_count"],
                    "indices_sha256": snap["indices_sha256"],
                    "full_bank_sha256": snap["full_bank_sha256"],
                }
            )
        finally:
            env.close()

    return rows


def paired_summary(rows, delta_field):
    values = [float(row[delta_field]) for row in rows]
    lower = sum(value < -ATOL for value in values)
    higher = sum(value > ATOL for value in values)
    tie = len(values) - lower - higher
    return {
        "episode_count": len(values),
        "mean_delta": float(statistics.fmean(values)),
        "median_delta": float(statistics.median(values)),
        "min_delta": min(values),
        "max_delta": max(values),
        "History_Point_lower_cost_episode_count": int(lower),
        "History_Point_higher_cost_episode_count": int(higher),
        "tie_episode_count": int(tie),
        "History_Point_lower_cost_episode_fraction":
            float(lower / len(values)),
        "History_Point_higher_cost_episode_fraction":
            float(higher / len(values)),
        "tie_episode_fraction": float(tie / len(values)),
    }


def build_pairs(
    seed,
    history_rows,
    static_point,
    static_ua,
    shield_reference,
):
    rows = []
    seen = set()

    for history in history_rows:
        year = history["year"]
        tid = int(history["trajectory_id"])
        key = (seed, year, tid)
        require(key not in seen, "Unique paired episode key")
        seen.add(key)

        point = static_point[key]
        ua = static_ua[key]
        ref = shield_reference[key]

        j_history = float(history["J_total"])
        j_point = float(point["J_total"])
        j_ua = float(ua["J_total"])
        j_shield = float(ref["J_shield_only"])

        require(
            abs(j_point - float(ref["J_masked"])) <= ATOL,
            "Frozen static Point paired identity",
        )

        def relation(delta):
            if delta < -ATOL:
                return "HISTORY_POINT_LOWER_COST"
            if delta > ATOL:
                return "HISTORY_POINT_HIGHER_COST"
            return "COST_TIE_WITHIN_ATOL"

        d_point = j_history - j_point
        d_ua = j_history - j_ua
        d_shield = j_history - j_shield

        rows.append(
            {
                "parent_rl_seed": seed,
                "year": year,
                "trajectory_id": tid,
                "J_History_Point": j_history,
                "J_Static_Point": j_point,
                "J_Static_UA": j_ua,
                "J_shield_only": j_shield,
                "delta_J_History_Point_minus_Static_Point":
                    d_point,
                "delta_J_History_Point_minus_Static_UA":
                    d_ua,
                "delta_J_History_Point_minus_shield_only":
                    d_shield,
                "relation_vs_Static_Point": relation(d_point),
                "relation_vs_Static_UA": relation(d_ua),
                "relation_vs_shield_only": relation(d_shield),
                "N_clean_History_Point":
                    int(history["N_clean"]),
                "N_forced_clean_History_Point":
                    int(history["N_forced_clean"]),
                "N_voluntary_clean_History_Point":
                    int(history["N_voluntary_clean"]),
                "N_free_choice_History_Point":
                    int(history["N_free_choice"]),
                "voluntary_clean_fraction_given_free_History_Point":
                    float(
                        history[
                            "voluntary_clean_fraction_given_free"
                        ]
                    ),
                "N_shield_interventions_History_Point":
                    int(history["N_shield_interventions"]),
                "N_clean_Static_Point":
                    int(point["N_clean"]),
                "N_voluntary_clean_Static_Point":
                    int(point["N_voluntary_clean"]),
                "N_clean_Static_UA":
                    int(ua["N_clean"]),
                "N_voluntary_clean_Static_UA":
                    int(ua["N_voluntary_clean"]),
            }
        )

    require(
        len(rows) == DEVELOPMENT_EPISODES,
        "Complete 600-episode paired population",
    )
    return rows


def run_formal():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive P2-2E-1 directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean worktree required",
    )

    head = git("rev-parse", "HEAD")
    verify_pinned_sources(head)
    stage0r_sha = verify_0r_authority()

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(
        self_blob
        == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "P2-2E-1 source committed unchanged",
    )
    static = static_contract()

    import run_paper2_stage2d6a_masked_ua_ppo_development_v1 as stage6a
    import run_paper2_stage2e0_temporal_partial_observability_design_freeze_v1 as stage0
    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as version_authority
    import paper2_ppo_protocol_v1 as frozen
    import paper2_masked_perception_ppo_runner_v1_1 as runner
    import paper2_temporal_observation_v1 as temporal

    # Read-only frozen references.
    point_ref = stage6a.verify_point5b(head)
    stage6a_sha = stage0.verify_6a()

    ua_rows = read_csv(
        UA6A / "masked_ua_development_episode_metrics.csv"
    )
    require(len(ua_rows) == 1200, "Exact frozen static-UA population")
    static_ua = {}
    for row in ua_rows:
        key = (
            int(row["parent_rl_seed"]),
            row["year"],
            int(row["trajectory_id"]),
        )
        require(key not in static_ua, "Unique frozen static-UA key")
        static_ua[key] = row
    require(len(static_ua) == 1200, "Complete frozen static-UA map")

    p = runner.get_protocol()
    config = runner.candidate("CONFIG_A")
    base_versions = version_authority.dependency_versions()
    masked_versions = runner.masked_dependency_versions()
    api_contract = runner.verify_maskable_api_contract()

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
        "Exact masked dependency pair",
    )
    require(
        frozen.protocol_payload(p)
        == frozen.protocol_payload(frozen.Protocol()),
        "Exact immutable PPO protocol",
    )
    require(
        tuple(p.seeds.development) == DEVELOPMENT_SEEDS
        and p.development_budget == DEVELOPMENT_BUDGET
        and p.development_budget
        == p.selection.development_checkpoint
        and config.name
        == runner.inherited_config_id()
        == "CONFIG_A"
        and dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY"),
        "Frozen CONFIG_A Point DEVELOPMENT contract",
    )
    require(
        api_contract["frozen_use_sde"] is False
        and api_contract["frozen_sde_sample_freq"] == -1,
        "Frozen no-gSDE MaskablePPO semantics",
    )

    contract = temporal.temporal_contract()
    require(
        contract["modes"][temporal.HISTORY_POINT]["dimension"] == 5
        and contract["modes"][temporal.HISTORY_POINT]["features"]
        == [
            "q50",
            "delta_q50",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ],
        "Frozen History-Point observation contract",
    )

    assets = runner.load_assets()
    assets.ensure_perception()
    development = runner.partition("DEVELOPMENT")
    specs = tuple(
        (year, tid)
        for year in development.years
        for tid in development.trajectory_ids
    )
    require(
        len(specs)
        == len(set(specs))
        == development.population_size
        == DEVELOPMENT_EPISODES,
        "Full frozen 600-episode DEVELOPMENT population",
    )

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        return execute_formal(
            runner=runner,
            temporal=temporal,
            frozen=frozen,
            p=p,
            config=config,
            assets=assets,
            specs=specs,
            head=head,
            self_blob=self_blob,
            static=static,
            stage0r_sha=stage0r_sha,
            stage6a_sha=stage6a_sha,
            point_ref=point_ref,
            static_ua=static_ua,
            base_versions=base_versions,
            masked_versions=masked_versions,
            api_contract=api_contract,
        )
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
                            "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/HISTORY-REDESIGN/MASK/SHIELD/REWARD/PERCEPTION/PPO TUNING",
                    }
                ),
            )
        raise


def execute_formal(
    *,
    runner,
    temporal,
    frozen,
    p,
    config,
    assets,
    specs,
    head,
    self_blob,
    static,
    stage0r_sha,
    stage6a_sha,
    point_ref,
    static_ua,
    base_versions,
    masked_versions,
    api_contract,
):
    gates = {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate: " + name)
        gates[name] = bool(condition)
        require(condition, name)

    gate("P2_2E_0R_temporal_design_authority_exact", bool(stage0r_sha))
    gate("P2_2D_5B_static_Point_reference_exact", bool(point_ref))
    gate("P2_2D_6A_static_UA_reference_exact", bool(stage6a_sha))
    gate("static_History_Point_development_contract", all(static.values()))
    gate(
        "frozen_CONFIG_A_seed_budget_contract",
        config.name == "CONFIG_A"
        and tuple(p.seeds.development) == DEVELOPMENT_SEEDS
        and p.development_budget == DEVELOPMENT_BUDGET
        and config.learning_rate == 3e-4
        and config.n_steps == 2048,
    )

    run_rows = []
    history_episode_rows = []
    paired_rows = []
    model_paths = {}
    completion_paths = {}
    access_ledgers = []

    training_roots = set(runner.partition("TRAINING").roots)

    for seed in DEVELOPMENT_SEEDS:
        label = f"seed_{seed}"
        run_dir = OUTPUT / "runs" / label
        run_dir.mkdir(parents=True, exist_ok=False)
        model_path = run_dir / "final_model.zip"
        completion_path = run_dir / "training_completion.json"

        ledger = runner.AccessLedger()
        base_env = runner.build_masked_scheduled_perception_env(
            assets=assets,
            rl_seed=seed,
            ledger=ledger,
            config_id=config.name,
            observation_mode="Point",
        )
        env = temporal.wrap_temporal_masked_env(
            base_env,
            temporal.HISTORY_POINT,
        )
        loaded = None

        try:
            gate(
                f"{label}:fresh_History_Point_training_environment",
                env.temporal_mode == temporal.HISTORY_POINT
                and env.base_observation_mode == "Point"
                and env.observation_space.shape == (5,)
                and env.action_space.n == 2
                and env.action_space.start == 0
                and env.schedule.parent_seed == seed
                and not env.started
                and not env.failed
                and not env.assignments,
            )

            bundle = build_history_point_model(
                runner, temporal, env, config.name, seed
            )
            gate(
                f"{label}:frozen_MaskablePPO_contract",
                bundle.model.num_timesteps == 0
                and runner.verify_masked_model_contract(
                    bundle.model, config.name, seed
                ),
            )

            training = runner.train_masked_final_checkpoint(
                bundle,
                model_path,
                purpose="development",
            )
            gate(
                f"{label}:exact_DEVELOPMENT_budget",
                training["total_timesteps"]
                == env.total_steps
                == DEVELOPMENT_BUDGET
                and training["rollout_updates"]
                == DEVELOPMENT_BUDGET // config.n_steps
                and training["scientific_budget_used"] is True
                and training["final_checkpoint_only"] is True
                and training["optimizer_steps_checked"] > 0
                and runner.parameters_finite(bundle.model),
            )

            mask_training = env._mask_audit_snapshot()
            gate(
                f"{label}:zero_training_shield_interventions",
                mask_training["total_shield_interventions"] == 0
                and mask_training[
                    "mask_shield_intervention_invariant"
                ]
                is True
                and mask_training["total_proposed_clean"]
                == mask_training["total_executed_clean"],
            )
            gate(
                f"{label}:training_mask_accounting_exact",
                mask_training["total_masked_safety_decisions"]
                + mask_training["total_terminal_passthrough"]
                == env.total_steps
                and mask_training["total_masked_safety_decisions"]
                == mask_training["total_forced_clean"]
                + mask_training["total_free_choice_decisions"]
                and mask_training["total_free_choice_decisions"]
                == mask_training["total_voluntary_clean"]
                + mask_training["total_mask_free_wait"],
            )

            schedule_audit = corrected_schedule_audit(
                runner, env, seed, config.name
            )
            training_records = [
                row for row in ledger.records
                if row["role"] == "TRAINING"
            ]
            gate(
                f"{label}:TRAINING_roots_Point_perception_only",
                len(training_records) == len(env.assignments)
                and all(
                    row["environment_root"] in training_roots
                    and row["observation_mode"] == "Point"
                    for row in training_records
                ),
            )

            completion = {
                "stage": STAGE,
                "parent_rl_seed": seed,
                "policy_observation_mode": "History-Point",
                "base_perception_mode": "Point",
                "policy_observation": [
                    "q50",
                    "delta_q50",
                    "tau_since_last_clean_or_episode_start",
                    "sin_DOY",
                    "cos_DOY",
                ],
                "safety_mask_inputs":
                    ["q50-history belief lower/upper only"],
                "training": training,
                "model_sha256": sha_file(model_path),
                "model_manifest":
                    runner.masked_model_manifest(bundle, training),
                "temporal_observation_contract":
                    temporal.temporal_contract(),
                "mask_training_audit": mask_training,
                "corrected_schedule_audit": schedule_audit,
                "training_access_log": training_records,
                "performance_diagnostics_not_pass_gates": {
                    "training_forced_clean":
                        mask_training["total_forced_clean"],
                    "training_voluntary_clean":
                        mask_training["total_voluntary_clean"],
                    "training_free_choice":
                        mask_training[
                            "total_free_choice_decisions"
                        ],
                    "role":
                        "DESCRIPTIVE ONLY; NOT A PASS GATE",
                },
            }
            write_exclusive(
                completion_path,
                encode(completion),
            )

            loaded = runner.reload_masked_final_checkpoint(
                model_path, bundle
            )
            gate(
                f"{label}:save_reload_exact",
                loaded is not bundle.model
                and runner.parameters_finite(loaded)
                and loaded.observation_space.shape == (5,)
                and sha_file(model_path)
                == training["model_sha256"],
            )
        finally:
            env.close()

        require(loaded is not None, f"{label}: checkpoint reloaded")

        eval_ledger = runner.AccessLedger()
        rows = evaluate_history_point(
            runner,
            temporal,
            loaded,
            assets,
            specs,
            eval_ledger,
        )
        summary = summarize(rows)

        gate(
            f"{label}:full_600_History_Point_DEVELOPMENT_evaluation",
            summary["episode_count"] == DEVELOPMENT_EPISODES,
        )
        gate(
            f"{label}:zero_evaluation_shield_interventions",
            summary["total_shield_interventions"] == 0,
        )
        gate(
            f"{label}:DEVELOPMENT_Point_perception_only",
            eval_ledger.heldout_ppo_trajectory_access_count == 0
            and eval_ledger.formal_perception_seed_access_count == 0
            and eval_ledger.denied_accesses == 0
            and all(
                row["role"] == "DEVELOPMENT"
                and row["observation_mode"] == "Point"
                for row in eval_ledger.records
            ),
        )

        pairs = build_pairs(
            seed,
            rows,
            point_ref["point"],
            static_ua,
            point_ref["reference"],
        )
        point_pairs = paired_summary(
            pairs,
            "delta_J_History_Point_minus_Static_Point",
        )
        ua_pairs = paired_summary(
            pairs,
            "delta_J_History_Point_minus_Static_UA",
        )
        shield_pairs = paired_summary(
            pairs,
            "delta_J_History_Point_minus_shield_only",
        )

        run_rows.append(
            {
                "parent_rl_seed": seed,
                "config_id": config.name,
                "policy_observation_mode": "History-Point",
                "development_timesteps": DEVELOPMENT_BUDGET,
                "mean_J_total": summary["mean_J_total"],
                "std_J_total": summary["std_J_total"],
                "median_J_total": summary["median_J_total"],
                "P95_J_total": summary["P95_J_total"],
                "CVaR95_J_total": summary["CVaR95_J_total"],
                "mean_N_clean": summary["mean_N_clean"],
                "mean_N_forced_clean":
                    summary["mean_N_forced_clean"],
                "mean_N_voluntary_clean":
                    summary["mean_N_voluntary_clean"],
                "mean_N_free_choice":
                    summary["mean_N_free_choice"],
                "mean_voluntary_clean_fraction_given_free":
                    summary[
                        "mean_voluntary_clean_fraction_given_free"
                    ],
                "episodes_with_any_voluntary_clean":
                    summary[
                        "episodes_with_any_voluntary_clean"
                    ],
                "total_shield_interventions":
                    summary["total_shield_interventions"],
                "mean_delta_J_vs_Static_Point":
                    point_pairs["mean_delta"],
                "lower_cost_vs_Static_Point_fraction":
                    point_pairs[
                        "History_Point_lower_cost_episode_fraction"
                    ],
                "higher_cost_vs_Static_Point_fraction":
                    point_pairs[
                        "History_Point_higher_cost_episode_fraction"
                    ],
                "mean_delta_J_vs_Static_UA":
                    ua_pairs["mean_delta"],
                "mean_delta_J_vs_shield_only":
                    shield_pairs["mean_delta"],
                "lower_cost_vs_shield_fraction":
                    shield_pairs[
                        "History_Point_lower_cost_episode_fraction"
                    ],
            }
        )

        history_episode_rows.extend(
            [
                {"parent_rl_seed": seed, **row}
                for row in rows
            ]
        )
        paired_rows.extend(pairs)
        model_paths[seed] = model_path
        completion_paths[seed] = completion_path
        access_ledgers.extend((ledger, eval_ledger))

    gate(
        "all_runtime_access_TRAINING_or_DEVELOPMENT_Point_only",
        all(
            ledger.heldout_ppo_trajectory_access_count == 0
            and ledger.formal_perception_seed_access_count == 0
            and ledger.denied_accesses == 0
            and all(
                row["observation_mode"] == "Point"
                for row in ledger.records
            )
            for ledger in access_ledgers
        ),
    )
    gate(
        "paired_comparison_population_complete",
        len(history_episode_rows) == 1200
        and len(paired_rows) == 1200
        and all(
            row["N_shield_interventions_History_Point"] == 0
            for row in paired_rows
        ),
    )

    seed_summaries = {
        seed: summarize(
            [
                row
                for row in history_episode_rows
                if int(row["parent_rl_seed"]) == seed
            ]
        )
        for seed in DEVELOPMENT_SEEDS
    }

    def seed_pair_summary(seed, field):
        return paired_summary(
            [
                row for row in paired_rows
                if int(row["parent_rl_seed"]) == seed
            ],
            field,
        )

    point_pairs_by_seed = {
        seed: seed_pair_summary(
            seed,
            "delta_J_History_Point_minus_Static_Point",
        )
        for seed in DEVELOPMENT_SEEDS
    }
    ua_pairs_by_seed = {
        seed: seed_pair_summary(
            seed,
            "delta_J_History_Point_minus_Static_UA",
        )
        for seed in DEVELOPMENT_SEEDS
    }
    shield_pairs_by_seed = {
        seed: seed_pair_summary(
            seed,
            "delta_J_History_Point_minus_shield_only",
        )
        for seed in DEVELOPMENT_SEEDS
    }

    performance_diagnostics_not_pass_gates = {
        "each_seed_has_any_deterministic_voluntary_clean":
            all(
                seed_summaries[seed][
                    "episodes_with_any_voluntary_clean"
                ]
                > 0
                for seed in DEVELOPMENT_SEEDS
            ),
        "any_deterministic_voluntary_clean_across_both_seeds":
            any(
                seed_summaries[seed][
                    "episodes_with_any_voluntary_clean"
                ]
                > 0
                for seed in DEVELOPMENT_SEEDS
            ),
        "each_seed_mean_cost_below_Static_Point":
            all(
                point_pairs_by_seed[seed]["mean_delta"] < -ATOL
                for seed in DEVELOPMENT_SEEDS
            ),
        "each_seed_mean_cost_below_Static_UA":
            all(
                ua_pairs_by_seed[seed]["mean_delta"] < -ATOL
                for seed in DEVELOPMENT_SEEDS
            ),
        "each_seed_mean_cost_below_shield_only":
            all(
                shield_pairs_by_seed[seed]["mean_delta"] < -ATOL
                for seed in DEVELOPMENT_SEEDS
            ),
        "two_seed_mean_of_seed_mean_J_total":
            float(
                statistics.fmean(
                    seed_summaries[seed]["mean_J_total"]
                    for seed in DEVELOPMENT_SEEDS
                )
            ),
        "two_seed_mean_of_seed_mean_voluntary_clean":
            float(
                statistics.fmean(
                    seed_summaries[seed]["mean_N_voluntary_clean"]
                    for seed in DEVELOPMENT_SEEDS
                )
            ),
        "role":
            "DESCRIPTIVE DEVELOPMENT SCIENCE ONLY; NOT USED TO PASS, RETRY, RESEED, TUNE OR SELECT",
    }

    paired_comparison_descriptive_only = {
        "Static_Point_reference": "P2-2D-5B-v1",
        "Static_UA_reference": "P2-2D-6A-v1",
        "shield_only_reference":
            "frozen shield-only costs carried by P2-2D-5B paired comparison",
        "paired_episode_unit":
            "(parent_rl_seed, year, trajectory_id)",
        "History_Point_vs_Static_Point_by_seed": {
            str(seed): point_pairs_by_seed[seed]
            for seed in DEVELOPMENT_SEEDS
        },
        "History_Point_vs_Static_UA_by_seed": {
            str(seed): ua_pairs_by_seed[seed]
            for seed in DEVELOPMENT_SEEDS
        },
        "History_Point_vs_shield_only_by_seed": {
            str(seed): shield_pairs_by_seed[seed]
            for seed in DEVELOPMENT_SEEDS
        },
        "performance_gate": False,
    }

    development_summary = {
        "stage": STAGE,
        "runtime_algorithm": "MaskablePPO",
        "policy_observation_mode": "History-Point",
        "base_perception_mode": "Point",
        "policy_observation": [
            "q50",
            "delta_q50",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ],
        "static_Point_reference_observation":
            ["q50", "sin_DOY", "cos_DOY"],
        "only_policy_inputs_added_relative_to_Static_Point":
            ["delta_q50", "tau_since_last_clean_or_episode_start"],
        "uncertainty_inputs_present": False,
        "safety_mask_temporal_or_UQ_inputs_used": False,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "development_budget_per_seed": DEVELOPMENT_BUDGET,
        "evaluation_episodes_per_seed": DEVELOPMENT_EPISODES,
        "total_development_evaluation_episodes": 1200,
        "seed_level": run_rows,
        "performance_diagnostics_not_pass_gates":
            performance_diagnostics_not_pass_gates,
        "paired_comparison_descriptive_only":
            paired_comparison_descriptive_only,
        "claim_boundary":
            "DEVELOPMENT characterization of compact temporal Point history only; no FINAL-seed or protected-data claim",
    }

    protocol_manifest = {
        "stage": STAGE,
        "mode": "formal",
        "protocol": frozen.protocol_payload(p),
        "selected_config": config.name,
        "runtime_algorithm": "MaskablePPO",
        "policy_observation_mode": "History-Point",
        "base_perception_mode": "Point",
        "policy_observation": [
            "q50",
            "delta_q50",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ],
        "safety_mask_inputs":
            ["q50-history belief lower/upper only"],
        "safety_mask_temporal_or_UQ_inputs_used": False,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "PPO_training_runs": 2,
        "History_Point_training_runs": 2,
        "History_UA_training_runs": 0,
        "development_budget_per_seed": DEVELOPMENT_BUDGET,
        "full_development_episode_count_per_model":
            DEVELOPMENT_EPISODES,
        "required_shield_interventions_after_masking": 0,
        "performance_gates": None,
        "FINAL_seed_training_runs": 0,
        "formal_RL_access_count": 0,
        "formal_perception_seed_access_count": 0,
        "RANDOM_TEST_access_count": 0,
        "SEALED_DATES_access_count": 0,
    }

    provenance = {
        "HEAD": head,
        "new_source_git_blob": self_blob,
        "source_sha256": {
            SELF: sha_file(ROOT / SELF),
            TEMPORAL: sha_file(ROOT / TEMPORAL),
            RUNNER: sha_file(ROOT / RUNNER),
        },
        "pinned_predecessor_blobs": PINNED_BLOBS,
        "P2_2E_0R_audit_summary_sha256": stage0r_sha,
        "P2_2D_5B_audit_summary_sha256":
            point_ref["audit_summary_sha256"],
        "P2_2D_6A_audit_summary_sha256": stage6a_sha,
        "base_dependencies": base_versions,
        "masked_dependencies": masked_versions,
        "MaskablePPO_API_contract": api_contract,
        "core_assets_provenance": assets.provenance,
    }

    content = {
        "protocol_manifest.json": encode(protocol_manifest),
        "development_summary.json": encode(development_summary),
        "source_artifact_hashes.json": encode(provenance),
        "history_point_development_run_summary.csv":
            csv_bytes(run_rows),
        "history_point_development_episode_metrics.csv":
            csv_bytes(history_episode_rows),
        "paired_history_point_static_references.csv":
            csv_bytes(paired_rows),
    }

    for name, data in content.items():
        write_exclusive(OUTPUT / name, data)

    extra_paths = {}
    for seed in DEVELOPMENT_SEEDS:
        extra_paths[
            f"runs/seed_{seed}/training_completion.json"
        ] = completion_paths[seed]
        extra_paths[
            f"runs/seed_{seed}/final_model.zip"
        ] = model_paths[seed]

    hashes = {
        **{
            name: sha_bytes(data)
            for name, data in content.items()
        },
        **{
            name: sha_file(path)
            for name, path in extra_paths.items()
        },
    }

    output_hashes = encode(
        {
            "hashes": hashes,
            "read_only_predecessor_evidence": {
                "P2_2E_0R_audit_summary_sha256": stage0r_sha,
                "P2_2D_5B_audit_summary_sha256":
                    point_ref["audit_summary_sha256"],
                "P2_2D_6A_audit_summary_sha256": stage6a_sha,
            },
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
        | set(extra_paths)
        | {"output_hashes.json"}
    )
    actual_inventory = {
        item.relative_to(OUTPUT).as_posix()
        for item in OUTPUT.rglob("*")
        if item.is_file()
    }

    gate(
        "output_inventory_complete",
        actual_inventory == expected_inventory,
    )
    gate(
        "output_hashes_exact",
        json.loads(output_hashes)["hashes"]
        == {
            name: sha_file(OUTPUT / name)
            for name in hashes
        },
    )
    gate(
        "P2_2E_0R_evidence_preserved",
        verify_0r_authority() == stage0r_sha,
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

    require(all(gates.values()), "All P2-2E-1 structural gates")

    audit = {
        "stage": STAGE,
        "mode": "formal",
        "stage_pass": True,
        "scientific_status": STATUS,
        "HEAD": head,
        "gates": gates,
        "failed_gates": [],
        "runtime_algorithm": "MaskablePPO",
        "selected_config": config.name,
        "policy_observation_mode": "History-Point",
        "policy_observation": [
            "q50",
            "delta_q50",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ],
        "uncertainty_inputs_present": False,
        "safety_mask_temporal_or_UQ_inputs_used": False,
        "PPO_training_runs": 2,
        "History_Point_training_runs": 2,
        "History_UA_training_runs": 0,
        "development_budget_per_seed": DEVELOPMENT_BUDGET,
        "development_evaluation_episode_count": 1200,
        "training_shield_interventions": {
            str(seed):
                json.loads(
                    completion_paths[seed].read_bytes()
                )["mask_training_audit"][
                    "total_shield_interventions"
                ]
            for seed in DEVELOPMENT_SEEDS
        },
        "evaluation_shield_interventions": {
            str(seed):
                seed_summaries[seed][
                    "total_shield_interventions"
                ]
            for seed in DEVELOPMENT_SEEDS
        },
        "FINAL_seed_training_runs": 0,
        "formal_RL_access_count": 0,
        "formal_perception_seed_access_count": 0,
        "RANDOM_TEST_access_count": 0,
        "SEALED_DATES_access_count": 0,
        "performance_diagnostics_not_pass_gates":
            performance_diagnostics_not_pass_gates,
        "paired_comparison_descriptive_only":
            paired_comparison_descriptive_only,
        "output_sha256": {
            **hashes,
            "output_hashes.json": sha_bytes(output_hashes),
        },
        "audit_summary_published_last": True,
    }

    write_exclusive(
        OUTPUT / "audit_summary.json",
        encode(audit),
    )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("formal",),
    )
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
