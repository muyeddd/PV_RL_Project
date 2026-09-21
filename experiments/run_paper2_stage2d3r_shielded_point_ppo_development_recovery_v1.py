
"""P2-2D-3R-v1: recovery of Shielded Point-PPO DEVELOPMENT characterization.

Why this stage exists
---------------------
P2-2D-3-v1 completed the full 491,520-step DEVELOPMENT training run for seed
510001 and saved its final checkpoint, then stopped at a POST-TRAINING schedule
provenance assertion. The assertion compared the raw frozen True-State schedule
row (perception_seed=None) against the Point runtime schedule row, where the
already-frozen perception bridge mechanically injects DEV perception_seed
20260906. This is an audit-representation mismatch, not a PPO/shield/perception
failure.

Recovery policy
---------------
* Preserve the failed P2-2D-3-v1 directory unchanged.
* NEVER retrain seed 510001.
* Verify the exact orphaned seed-510001 checkpoint hash, 491,520 timesteps,
  CONFIG_A/model contract, learner seed and finite parameters, then evaluate it.
* Train seed 510002 exactly once at the original frozen DEVELOPMENT budget.
* Correct the schedule audit by comparing the latent schedule fields exactly and
  checking the injected perception seed separately.
* Evaluate both models on the full 600-episode DEVELOPMENT population.
* Evaluate the identical shield-only always-WAIT DEVELOPMENT baseline once.
* No hyperparameter reselection, retry, reseed, perception/shield tuning, FINAL
  seed training, FORMAL HELD-OUT access, RANDOM_TEST or SEALED_DATES access.

This is recovery/provenance completion, not a new model-selection stage.
Import is inert. Only --mode formal executes.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage2d3r_shielded_point_ppo_development_recovery_v1.py"
FAILED_STAGE_SOURCE = "experiments/run_paper2_stage2d3_shielded_point_ppo_development_v1.py"
RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
BASE_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
FAILED_STAGE = BASE / "p2_2d_3_shielded_point_ppo_development_v1/formal"
FAILED_EXECUTION = FAILED_STAGE / "execution_failure.json"
RECOVERED_MODEL = FAILED_STAGE / "runs/seed_510001/final_model.zip"

STAGE_DIRECTORY = BASE / "p2_2d_3r_shielded_point_ppo_development_recovery_v1"
OUTPUT = STAGE_DIRECTORY / "formal"

PREDECESSOR_HEAD = "aebacec91c50e290f3ee6d361614e8b3564e1dc3"
RECOVERED_MODEL_SHA256 = "d3db58a693b3697d36dbbf2ccd3edd9da255561ebdfcfc5c4cca04debbad8d43"
RECOVERED_SEED = 510001
NEW_TRAIN_SEED = 510002

PINNED_BLOBS = {
    FAILED_STAGE_SOURCE: "b0852c93122717e6976407f85551e94922c2e46b",
    RUNNER: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    BASE_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
}

STAGE = "P2-2D-3R-v1"
STATUS = "SHIELDED_POINT_PPO_DEVELOPMENT_RECOVERY_PASS"
DECLARATION = (
    "P2-2D-3 POST-TRAINING SCHEDULE-AUDIT REPRESENTATION MISMATCH RECOVERED "
    "WITHOUT RETRAINING SEED 510001; ITS EXACT FINAL CHECKPOINT IS REUSED, "
    "SEED 510002 IS TRAINED ONCE AT THE ORIGINAL FROZEN DEVELOPMENT BUDGET, "
    "AND BOTH MODELS PLUS THE SHIELD-ONLY BASELINE ARE EVALUATED ON THE FULL "
    "600-EPISODE DEVELOPMENT POPULATION; NO TUNING, RESEED OR FORMAL ACCESS"
)

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
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, "Exact readback: " + path.name)


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


def verify_failed_predecessor(head):
    require(
        FAILED_EXECUTION.exists() and RECOVERED_MODEL.exists(),
        "Failed predecessor evidence and recovered checkpoint exist",
    )
    failure_bytes = FAILED_EXECUTION.read_bytes()
    failure = json.loads(failure_bytes)
    require(
        failure == {
            "HEAD": PREDECESSOR_HEAD,
            "action":
                "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/HYPERPARAMETER, PERCEPTION OR SHIELD TUNING",
            "error_type": "RuntimeError",
            "message": "seed_510001:frozen_training_schedule_exact",
            "stage": "P2-2D-3-v1",
            "stage_pass": False,
        },
        "Exact P2-2D-3 failure authority",
    )
    require(
        sha_file(RECOVERED_MODEL) == RECOVERED_MODEL_SHA256,
        "Exact recovered seed-510001 checkpoint hash",
    )

    inventory = {
        item.relative_to(FAILED_STAGE).as_posix()
        for item in FAILED_STAGE.rglob("*")
        if item.is_file()
    }
    require(
        inventory == {
            "execution_failure.json",
            "runs/seed_510001/final_model.zip",
        },
        "Failed predecessor output inventory preserved exactly",
    )

    git("merge-base", "--is-ancestor", PREDECESSOR_HEAD, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{PREDECESSOR_HEAD}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned predecessor source unchanged: " + name,
        )
    return {
        "execution_failure_sha256": sha_bytes(failure_bytes),
        "recovered_model_sha256": RECOVERED_MODEL_SHA256,
        "inventory": sorted(inventory),
        "failure": failure,
    }


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attrs = {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }

    train_calls = [
        n
        for n in calls
        if isinstance(n.func, ast.Attribute)
        and n.func.attr == "train_final_checkpoint"
    ]
    require(len(train_calls) == 1, "Exactly one new PPO training call site")
    train_call = train_calls[0]
    require(
        len(train_call.keywords) == 1
        and train_call.keywords[0].arg == "purpose"
        and isinstance(train_call.keywords[0].value, ast.Constant)
        and train_call.keywords[0].value.value == "development",
        "Only frozen DEVELOPMENT budget",
    )

    require(
        "build_shielded_scheduled_perception_env" in attrs
        and "evaluate_shielded_perception_deterministic" in attrs,
        "Shielded runtime only",
    )
    require(
        "build_scheduled_perception_env" not in attrs
        and "evaluate_perception_deterministic" not in attrs,
        "No unshielded runtime bypass",
    )
    require(
        "learn" not in attrs and "save" not in attrs,
        "No direct learn/save bypass",
    )

    load_calls = [
        n
        for n in calls
        if isinstance(n.func, ast.Attribute) and n.func.attr == "load"
    ]
    require(len(load_calls) == 1, "Exactly one direct recovery load call")

    protected_role_calls = []
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
                protected_role_calls.append(value.value)
    require(
        not protected_role_calls,
        "No formal/protected/UA runtime constructor literal",
    )
    return {
        "exactly_one_new_development_training_call": True,
        "exactly_one_orphan_checkpoint_direct_load": True,
        "shielded_runtime_only": True,
        "no_unshielded_runtime_bypass": True,
        "no_direct_learn_or_save": True,
        "no_formal_protected_or_UA_runtime_call": True,
    }


def summarize_metrics(rows):
    require(rows, "Nonempty evaluation rows")
    costs = [float(r["J_total"]) for r in rows]
    cleans = [int(r["N_clean"]) for r in rows]
    proposed = [int(r["N_clean_proposed"]) for r in rows]
    interventions = [int(r["shield_interventions"]) for r in rows]
    intervention_fractions = [
        float(r["shield_intervention_fraction"]) for r in rows
    ]
    ordered = sorted(costs)
    cvar_count = math.ceil(0.05 * len(ordered))
    return {
        "episode_count": len(rows),
        "mean_J_total": float(statistics.fmean(costs)),
        "std_J_total":
            float(statistics.stdev(costs)) if len(costs) > 1 else 0.0,
        "median_J_total": float(statistics.median(costs)),
        "P95_J_total": float(np.quantile(costs, 0.95, method="linear")),
        "CVaR95_J_total":
            float(statistics.fmean(ordered[-cvar_count:])),
        "CVaR95_episode_count": int(cvar_count),
        "mean_N_clean_executed": float(statistics.fmean(cleans)),
        "mean_N_clean_proposed": float(statistics.fmean(proposed)),
        "mean_shield_interventions":
            float(statistics.fmean(interventions)),
        "mean_shield_intervention_fraction":
            float(statistics.fmean(intervention_fractions)),
        "total_N_clean_executed": int(sum(cleans)),
        "total_N_clean_proposed": int(sum(proposed)),
        "total_shield_interventions": int(sum(interventions)),
        "episodes_with_any_proposed_clean":
            int(sum(v > 0 for v in proposed)),
        "episodes_with_zero_proposed_clean":
            int(sum(v == 0 for v in proposed)),
        "episodes_with_any_shield_intervention":
            int(sum(v > 0 for v in interventions)),
        "executed_clean_minus_proposed_clean":
            int(sum(cleans) - sum(proposed)),
        "shield_share_of_executed_clean":
            0.0
            if sum(cleans) == 0
            else float(sum(interventions) / sum(cleans)),
    }


class AlwaysWaitModel:
    def predict(self, observation, deterministic=True):
        del observation, deterministic
        return np.asarray([0], dtype=np.int64), None


def load_recovered_checkpoint(runner, config):
    from stable_baselines3 import PPO

    model = PPO.load(
        str(RECOVERED_MODEL),
        device=runner.get_protocol().device,
    )
    require(
        model.num_timesteps == runner.get_protocol().development_budget,
        "Recovered checkpoint has exact 491,520 DEVELOPMENT timesteps",
    )
    require(
        runner.verify_model_contract(
            model, config.name, RECOVERED_SEED
        ),
        "Recovered checkpoint CONFIG_A/seed/model contract",
    )
    require(
        runner.parameters_finite(model),
        "Recovered checkpoint parameters finite",
    )
    require(
        model.observation_space.shape == (3,)
        and model.action_space.n == 2
        and model.action_space.start == 0,
        "Recovered checkpoint Point observation/action spaces",
    )
    return model


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
        all(row["perception_seed"] == perception_seed
            for row in env.assignments),
        "Runtime schedule has frozen injected DEV perception seed",
    )
    require(
        all(row["perception_seed"] is None for row in expected),
        "Raw True-State schedule perception_seed remains None by design",
    )
    return {
        "assignment_count": len(env.assignments),
        "latent_schedule_exact": True,
        "raw_schedule_perception_seed": None,
        "runtime_injected_perception_seed": perception_seed,
        "runtime_perception_seed_exact": True,
        "compared_latent_fields": list(SCHEDULE_FIELDS),
    }


def evaluate_full(runner, model, assets, specs, seed_label):
    ledger = runner.AccessLedger()
    rows = runner.evaluate_shielded_perception_deterministic(
        model, assets, specs, ledger, "Point"
    )
    require(
        len(rows) == 600
        and all(
            r["step_count"]
            == runner.get_protocol().decision_reward_steps
            and r["natural_transition_count"]
            == runner.get_protocol().natural_transitions
            and r["all_observations_finite"]
            and r["all_rewards_finite"]
            and r["all_states_finite"]
            and r["reward_sign_regression_diff"]
            <= runner.get_protocol().selection.atol
            and r["N_clean"] >= r["N_clean_proposed"]
            and r["N_clean"] - r["N_clean_proposed"]
            == r["shield_interventions"]
            for r in rows
        ),
        f"{seed_label}: complete finite DEVELOPMENT evaluation",
    )
    require(
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0,
        f"{seed_label}: DEVELOPMENT access only",
    )
    return rows, summarize_metrics(rows), ledger


def run_formal():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive recovery stage directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean sources required",
    )
    head = git("rev-parse", "HEAD")
    predecessor = verify_failed_predecessor(head)

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(
        self_blob
        == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "Recovery source committed unchanged",
    )

    sources = {}
    for name in (
        SELF,
        FAILED_STAGE_SOURCE,
        RUNNER,
        BASE_RUNNER,
        SHIELD,
        PROTOCOL,
        CORE,
        TRUE_RUNNER,
    ):
        capture_source(sources, ROOT / name)

    static = static_contract()

    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as version_authority
    import paper2_ppo_protocol_v1 as frozen
    import paper2_shielded_perception_ppo_runner_v1 as runner

    versions = version_authority.dependency_versions()
    p = runner.get_protocol()
    config = runner.candidate("CONFIG_A")
    require(versions == dict(p.dependencies), "Frozen dependencies")
    require(
        frozen.protocol_payload(p)
        == frozen.protocol_payload(frozen.Protocol()),
        "Exact frozen PPO protocol",
    )
    require(
        tuple(p.seeds.development)
        == (RECOVERED_SEED, NEW_TRAIN_SEED)
        and p.development_budget == 491520
        and p.final_budget == 983040
        and config.name == runner.inherited_config_id() == "CONFIG_A",
        "Frozen CONFIG_A/development seed/budget registry",
    )
    require(
        dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY")
        and dict(p.observation_dimensions)["Point"] == 3,
        "Frozen Point observation",
    )

    assets = runner.load_assets()
    assets.ensure_perception()
    for name, source_path in (
        ("paper1_cqr", assets.paper1_cqr),
        ("wapp_power_bridge", assets.wapp_power_bridge_path),
    ):
        require(
            sha_file(source_path)
            == assets.perception.EXPECTED_HASHES[name],
            "Frozen perception input: " + name,
        )
        capture_source(sources, source_path)

    development = runner.partition("DEVELOPMENT")
    specs = tuple(
        (year, tid)
        for year in development.years
        for tid in development.trajectory_ids
    )
    require(
        len(specs) == len(set(specs)) == 600,
        "Full 600-episode DEVELOPMENT population",
    )

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        return execute_recovery(
            runner=runner,
            frozen=frozen,
            p=p,
            config=config,
            assets=assets,
            specs=specs,
            head=head,
            versions=versions,
            sources=sources,
            self_blob=self_blob,
            static=static,
            predecessor=predecessor,
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
                            "STOP; PRESERVE BOTH ORIGINAL FAILURE AND RECOVERY EVIDENCE; NO RETRAIN OF 510001, NO RETRY/RESEED/TUNING",
                        "HEAD": head,
                    }
                ),
            )
        raise


def execute_recovery(
    *,
    runner,
    frozen,
    p,
    config,
    assets,
    specs,
    head,
    versions,
    sources,
    self_blob,
    static,
    predecessor,
):
    gates = {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate: " + name)
        gates[name] = bool(condition)
        require(condition, name)

    gate("failed_predecessor_preserved_exact", bool(predecessor))
    gate("static_recovery_contract", all(static.values()))

    raw_schedule = runner.build_training_schedule(
        RECOVERED_SEED, config.name
    ).next_assignment()
    gate(
        "recovery_reason_mechanically_reproduced",
        raw_schedule["perception_seed"] is None
        and runner.get_perception_contract().development_seed == 20260906,
    )
    recovery_reason = {
        "failed_gate":
            "seed_510001:frozen_training_schedule_exact",
        "raw_TrueState_schedule_perception_seed":
            raw_schedule["perception_seed"],
        "Point_runtime_injected_perception_seed":
            runner.get_perception_contract().development_seed,
        "scientific_interpretation":
            "post-training audit representation mismatch only; latent training schedule, PPO, shield and perception algorithms unchanged",
        "retraining_seed_510001": False,
    }
    write_exclusive(
        OUTPUT / "recovery_reason.json",
        encode(recovery_reason),
    )

    recovered_model = load_recovered_checkpoint(runner, config)
    gate(
        "seed_510001_checkpoint_recovered_without_training",
        recovered_model.num_timesteps == p.development_budget
        and sha_file(RECOVERED_MODEL)
        == RECOVERED_MODEL_SHA256,
    )
    recovered_rows, recovered_summary, recovered_ledger = evaluate_full(
        runner, recovered_model, assets, specs, "seed_510001"
    )
    gate(
        "seed_510001_full_DEVELOPMENT_evaluation",
        recovered_summary["episode_count"] == 600,
    )

    baseline_ledger = runner.AccessLedger()
    baseline_rows = runner.evaluate_shielded_perception_deterministic(
        AlwaysWaitModel(),
        assets,
        specs,
        baseline_ledger,
        "Point",
    )
    baseline_summary = summarize_metrics(baseline_rows)
    gate(
        "shield_only_baseline_complete",
        len(baseline_rows) == 600
        and baseline_summary["total_N_clean_proposed"] == 0
        and baseline_summary["total_N_clean_executed"]
        == baseline_summary["total_shield_interventions"]
        and baseline_ledger.heldout_ppo_trajectory_access_count == 0
        and baseline_ledger.formal_perception_seed_access_count == 0
        and baseline_ledger.denied_accesses == 0,
    )

    ledger = runner.AccessLedger()
    env = runner.build_shielded_scheduled_perception_env(
        assets,
        NEW_TRAIN_SEED,
        ledger,
        config.name,
        "Point",
    )
    new_dir = OUTPUT / "runs" / f"seed_{NEW_TRAIN_SEED}"
    new_dir.mkdir(parents=True, exist_ok=False)
    model_path = new_dir / "final_model.zip"
    loaded = None
    try:
        gate(
            "seed_510002_fresh_training_environment",
            env.observation_mode == "Point"
            and env.schedule.parent_seed == NEW_TRAIN_SEED
            and not env.started
            and not env.failed
            and not env.assignments,
        )
        bundle = runner.build_shielded_perception_ppo_model(
            env, config.name, NEW_TRAIN_SEED
        )
        gate(
            "seed_510002_frozen_PPO_contract",
            bundle.model.num_timesteps == 0
            and runner.verify_model_contract(
                bundle.model, config.name, NEW_TRAIN_SEED
            ),
        )
        training = runner.train_final_checkpoint(
            bundle,
            model_path,
            purpose="development",
        )
        gate(
            "seed_510002_exact_development_budget",
            training["total_timesteps"]
            == env.total_steps
            == p.development_budget
            and training["scientific_budget_used"] is True
            and training["rollout_updates"]
            == p.development_budget // config.n_steps
            and training["final_checkpoint_only"] is True
            and runner.parameters_finite(bundle.model),
        )

        shield_training = env._shield_audit_snapshot()
        gate(
            "seed_510002_training_shield_accounting",
            shield_training["total_safety_filtered_decisions"]
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
            "seed_510002_training_perception_query_accounting",
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

        training_records = [
            row for row in ledger.records if row["role"] == "TRAINING"
        ]
        gate(
            "seed_510002_TRAINING_roots_only",
            len(training_records) == len(env.assignments)
            and all(
                row["environment_root"]
                in runner.partition("TRAINING").roots
                and row["observation_mode"] == "Point"
                for row in training_records
            ),
        )

        schedule_audit = corrected_schedule_audit(
            runner, env, NEW_TRAIN_SEED, config.name
        )
        gate(
            "seed_510002_corrected_schedule_exact",
            schedule_audit["latent_schedule_exact"]
            and schedule_audit["runtime_perception_seed_exact"],
        )

        training_completion = {
            "stage": STAGE,
            "parent_rl_seed": NEW_TRAIN_SEED,
            "training": training,
            "model_sha256": sha_file(model_path),
            "model_manifest":
                runner.model_manifest(bundle, training),
            "shield_training": shield_training,
            "corrected_schedule_audit": schedule_audit,
            "training_assignment_count": len(env.assignments),
            "training_access_log": training_records,
        }
        write_exclusive(
            new_dir / "training_completion.json",
            encode(training_completion),
        )

        loaded = runner.reload_final_checkpoint(
            model_path, bundle
        )
        gate(
            "seed_510002_save_reload_exact",
            loaded is not bundle.model
            and runner.parameters_finite(loaded)
            and sha_file(model_path) == training["model_sha256"],
        )
    finally:
        env.close()

    require(loaded is not None, "Seed 510002 checkpoint reloaded")
    new_rows, new_summary, eval_ledger = evaluate_full(
        runner, loaded, assets, specs, "seed_510002"
    )
    gate(
        "seed_510002_full_DEVELOPMENT_evaluation",
        new_summary["episode_count"] == 600,
    )

    gate(
        "all_recovery_access_development_or_training_only",
        recovered_ledger.heldout_ppo_trajectory_access_count == 0
        and recovered_ledger.formal_perception_seed_access_count == 0
        and baseline_ledger.heldout_ppo_trajectory_access_count == 0
        and baseline_ledger.formal_perception_seed_access_count == 0
        and eval_ledger.heldout_ppo_trajectory_access_count == 0
        and eval_ledger.formal_perception_seed_access_count == 0
        and ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and recovered_ledger.denied_accesses == 0
        and baseline_ledger.denied_accesses == 0
        and eval_ledger.denied_accesses == 0
        and ledger.denied_accesses == 0,
    )

    seed_summaries = {
        RECOVERED_SEED: recovered_summary,
        NEW_TRAIN_SEED: new_summary,
    }
    run_rows = []
    for seed in (RECOVERED_SEED, NEW_TRAIN_SEED):
        summary = seed_summaries[seed]
        run_rows.append(
            {
                "parent_rl_seed": seed,
                "config_id": config.name,
                "checkpoint_origin":
                    "P2-2D-3-v1 recovered immutable checkpoint"
                    if seed == RECOVERED_SEED
                    else "P2-2D-3R-v1 newly trained once",
                "checkpoint_path":
                    RECOVERED_MODEL.relative_to(ROOT).as_posix()
                    if seed == RECOVERED_SEED
                    else model_path.relative_to(ROOT).as_posix(),
                "checkpoint_sha256":
                    RECOVERED_MODEL_SHA256
                    if seed == RECOVERED_SEED
                    else sha_file(model_path),
                "development_timesteps": p.development_budget,
                "training_runtime_metrics_available":
                    seed == NEW_TRAIN_SEED,
                "mean_J_total": summary["mean_J_total"],
                "std_J_total": summary["std_J_total"],
                "median_J_total": summary["median_J_total"],
                "P95_J_total": summary["P95_J_total"],
                "CVaR95_J_total": summary["CVaR95_J_total"],
                "mean_N_clean_executed":
                    summary["mean_N_clean_executed"],
                "mean_N_clean_proposed":
                    summary["mean_N_clean_proposed"],
                "mean_shield_interventions":
                    summary["mean_shield_interventions"],
                "mean_shield_intervention_fraction":
                    summary["mean_shield_intervention_fraction"],
                "episodes_with_any_proposed_clean":
                    summary["episodes_with_any_proposed_clean"],
                "episodes_with_zero_proposed_clean":
                    summary["episodes_with_zero_proposed_clean"],
                "mean_J_gap_vs_shield_only":
                    summary["mean_J_total"]
                    - baseline_summary["mean_J_total"],
                "mean_intervention_fraction_gap_vs_shield_only":
                    summary["mean_shield_intervention_fraction"]
                    - baseline_summary[
                        "mean_shield_intervention_fraction"
                    ],
            }
        )

    episode_rows = [
        {"parent_rl_seed": RECOVERED_SEED, **row}
        for row in recovered_rows
    ] + [
        {"parent_rl_seed": NEW_TRAIN_SEED, **row}
        for row in new_rows
    ]

    seed_mean_costs = [
        recovered_summary["mean_J_total"],
        new_summary["mean_J_total"],
    ]
    seed_mean_interventions = [
        recovered_summary["mean_shield_intervention_fraction"],
        new_summary["mean_shield_intervention_fraction"],
    ]
    development_summary = {
        "stage": STAGE,
        "statistical_unit": "RL TRAINING SEED",
        "independent_training_seed_count": 2,
        "development_seeds": [RECOVERED_SEED, NEW_TRAIN_SEED],
        "development_budget_per_seed": p.development_budget,
        "recovered_seed_retrained": False,
        "new_training_runs_in_recovery_stage": 1,
        "cumulative_development_models": 2,
        "evaluation_episodes_per_seed": 600,
        "two_seed_mean_of_seed_mean_J_total":
            float(statistics.fmean(seed_mean_costs)),
        "two_seed_std_of_seed_mean_J_total":
            float(statistics.stdev(seed_mean_costs)),
        "two_seed_mean_of_seed_intervention_fraction":
            float(statistics.fmean(seed_mean_interventions)),
        "seed_level": run_rows,
        "shield_only_baseline": baseline_summary,
        "readiness_diagnostics_descriptive_only": {
            "each_seed_proposes_clean_somewhere":
                all(
                    summary["total_N_clean_proposed"] > 0
                    for summary in seed_summaries.values()
                ),
            "each_seed_mean_cost_below_shield_only":
                all(
                    summary["mean_J_total"]
                    < baseline_summary["mean_J_total"]
                    for summary in seed_summaries.values()
                ),
            "each_seed_mean_intervention_fraction_below_shield_only":
                all(
                    summary["mean_shield_intervention_fraction"]
                    < baseline_summary[
                        "mean_shield_intervention_fraction"
                    ]
                    for summary in seed_summaries.values()
                ),
            "role":
                "DESCRIPTIVE ONLY; NOT USED TO TUNE, SELECT, RETRY OR RESEED",
        },
        "claim_boundary":
            "Recovered DEVELOPMENT characterization only; no final-seed or formal-held-out claim",
    }

    protocol_manifest = {
        "stage": STAGE,
        "mode": "formal",
        "declaration": DECLARATION,
        "protocol": frozen.protocol_payload(p),
        "selected_config": config.name,
        "observation_mode": "Point",
        "recovered_seed": RECOVERED_SEED,
        "newly_trained_seed": NEW_TRAIN_SEED,
        "seed_510001_retrained": False,
        "new_PPO_training_runs": 1,
        "development_budget_per_seed": p.development_budget,
        "full_development_episode_count_per_model": 600,
        "schedule_audit_semantics":
            "latent schedule fields compared exactly; Point bridge perception seed checked separately",
        "performance_gates": None,
        "FINAL_seed_training_runs": 0,
        "formal_RL_access_count": 0,
        "formal_perception_seed_access_count": 0,
        "RANDOM_TEST_access_count": 0,
        "SEALED_DATES_access_count": 0,
    }

    source_manifest = {
        "HEAD": head,
        "predecessor_HEAD": PREDECESSOR_HEAD,
        "source_sha256": sources,
        "new_source_git_blob": self_blob,
        "pinned_predecessor_blobs": PINNED_BLOBS,
        "dependencies": versions,
        "core_assets_provenance": assets.provenance,
        "failed_predecessor": predecessor,
    }

    documents = {
        "protocol_manifest.json": protocol_manifest,
        "development_summary.json": development_summary,
        "source_artifact_hashes.json": source_manifest,
    }
    content = {
        name: encode(doc) for name, doc in documents.items()
    }
    content["development_run_summary.csv"] = csv_bytes(run_rows)
    content["development_episode_metrics.csv"] = csv_bytes(episode_rows)
    content["shield_only_development_metrics.csv"] = csv_bytes(baseline_rows)

    for name, data in content.items():
        write_exclusive(OUTPUT / name, data)

    extra_paths = {
        "recovery_reason.json": OUTPUT / "recovery_reason.json",
        f"runs/seed_{NEW_TRAIN_SEED}/training_completion.json":
            new_dir / "training_completion.json",
        f"runs/seed_{NEW_TRAIN_SEED}/final_model.zip":
            model_path,
    }
    hashes = {
        **{name: sha_bytes(data) for name, data in content.items()},
        **{
            name: sha_file(path)
            for name, path in extra_paths.items()
        },
    }
    output_hashes = encode(
        {
            "hashes": hashes,
            "external_preserved_evidence": {
                RECOVERED_MODEL.relative_to(ROOT).as_posix():
                    RECOVERED_MODEL_SHA256,
                FAILED_EXECUTION.relative_to(ROOT).as_posix():
                    predecessor["execution_failure_sha256"],
            },
            "exclusions": {
                "output_hashes.json":
                    "SHA256 pinned in audit_summary; avoid self-reference",
                "audit_summary.json":
                    "Published last PASS marker",
            },
        }
    )
    write_exclusive(OUTPUT / "output_hashes.json", output_hashes)

    expected_inventory = set(content) | set(extra_paths) | {
        "output_hashes.json"
    }
    actual_inventory = {
        item.relative_to(OUTPUT).as_posix()
        for item in OUTPUT.rglob("*")
        if item.is_file()
    }
    gate(
        "recovery_output_inventory_complete",
        actual_inventory == expected_inventory,
    )
    gate(
        "recovery_output_hashes_exact",
        json.loads(output_hashes)["hashes"]
        == {
            name: sha_file(OUTPUT / name)
            for name in hashes
        },
    )
    gate(
        "original_failed_stage_unchanged",
        sha_file(FAILED_EXECUTION)
        == predecessor["execution_failure_sha256"]
        and sha_file(RECOVERED_MODEL)
        == RECOVERED_MODEL_SHA256,
    )
    gate(
        "source_HEAD_and_worktree_unchanged",
        git("rev-parse", "HEAD") == head
        and git("status", "--porcelain", "--untracked-files=all") == ""
        and all(
            sha_file(ROOT / name) == digest
            for name, digest in sources.items()
        ),
    )
    gate(
        "audit_summary_published_last",
        not (OUTPUT / "audit_summary.json").exists(),
    )
    require(all(gates.values()), "All recovery structural gates required")

    summary = {
        "stage": STAGE,
        "mode": "formal",
        "stage_pass": True,
        "scientific_status": STATUS,
        "declaration": DECLARATION,
        "HEAD": head,
        "gates": gates,
        "failed_gates": [],
        "selected_config": config.name,
        "new_PPO_training_runs": 1,
        "recovered_predecessor_training_runs": 1,
        "cumulative_development_model_count": 2,
        "Point_models": 2,
        "UA_training_runs": 0,
        "recovered_seed_retrained": False,
        "development_evaluation_episode_count": 1200,
        "shield_only_baseline_episode_count": 600,
        "final_seed_training_runs": 0,
        "heldout_ppo_trajectory_access_count": 0,
        "formal_perception_seed_access_count": 0,
        "RANDOM_TEST_access_count": 0,
        "SEALED_DATES_access_count": 0,
        "two_seed_mean_of_seed_mean_J_total":
            development_summary[
                "two_seed_mean_of_seed_mean_J_total"
            ],
        "two_seed_mean_of_seed_intervention_fraction":
            development_summary[
                "two_seed_mean_of_seed_intervention_fraction"
            ],
        "readiness_diagnostics_descriptive_only":
            development_summary[
                "readiness_diagnostics_descriptive_only"
            ],
        "output_sha256": {
            **hashes,
            "output_hashes.json": sha_bytes(output_hashes),
        },
        "external_preserved_evidence": {
            "failed_stage_execution_failure_sha256":
                predecessor["execution_failure_sha256"],
            "seed_510001_final_model_sha256":
                RECOVERED_MODEL_SHA256,
        },
        "audit_summary_published_last": True,
    }
    write_exclusive(OUTPUT / "audit_summary.json", encode(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
