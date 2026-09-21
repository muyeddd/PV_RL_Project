"""P2-2D-3-v1: Shielded Point-PPO DEVELOPMENT characterization.

This stage uses the already-selected CONFIG_A and the original frozen PPO
development protocol. It does NOT reselect hyperparameters.

Scientific role
---------------
* Train exactly the two frozen DEVELOPMENT RL seeds (510001, 510002).
* Use exactly the frozen development budget (491520 interactions per seed).
* Gradient training uses TRAINING roots only.
* Deterministic evaluation uses the full 600 DEVELOPMENT specs per model.
* Point observation remains [q50, sin_DOY, cos_DOY].
* Perception remains frozen BLOCK10 v1.
* The already-frozen q50-only set-membership shield remains unchanged.
* Save/reload final checkpoints only; no early stopping/best checkpoint.
* No FINAL RL seeds, FORMAL HELD-OUT roots, formal perception seed,
  RANDOM_TEST or SEALED_DATES access.
* A full DEVELOPMENT shield-only (always-WAIT proposal) baseline is evaluated
  once for context. It is descriptive and never tunes training.
* Performance and shield-dependence diagnostics are descriptive only in this
  stage. Structural/provenance gates determine stage PASS.

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
SELF = "experiments/run_paper2_stage2d3_shielded_point_ppo_development_v1.py"
RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
SMOKE_SOURCE = "experiments/run_paper2_stage2d2_shielded_point_ppo_smoke_audit_v1.py"
BASE_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
SMOKE = BASE / "p2_2d_2_shielded_point_ppo_smoke_audit_v1/smoke"
STAGE_DIRECTORY = BASE / "p2_2d_3_shielded_point_ppo_development_v1"
OUTPUT = STAGE_DIRECTORY / "formal"

CHECKPOINT = "6a276f2adc147828b543da1b4e56e018ff84b072"
PINNED_BLOBS = {
    RUNNER: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    SMOKE_SOURCE: "3bbd7e803de4ecc4dc5a97e7fa44ab765c35b4ab",
    BASE_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
}

STAGE = "P2-2D-3-v1"
STATUS = "SHIELDED_POINT_PPO_DEVELOPMENT_CHARACTERIZATION_PASS"
DECLARATION = (
    "TWO FROZEN DEVELOPMENT-SEED SHIELDED POINT-PPO MODELS TRAINED AT THE "
    "ORIGINAL DEVELOPMENT BUDGET AND EVALUATED DETERMINISTICALLY ON THE FULL "
    "600-EPISODE DEVELOPMENT POPULATION; CONFIG_A, POINT OBSERVATION, BLOCK10 "
    "V1 PERCEPTION, Q50-ONLY SHIELD, PPO RULES, REWARD AND PHYSICS UNCHANGED; "
    "NO HYPERPARAMETER RESELECTION, FINAL-SEED TRAINING OR FORMAL ACCESS"
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
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


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
    require(rows and all(set(row) == set(rows[0]) for row in rows), "Nonempty consistent CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
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
    key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    require(key not in sources or sources[key] == digest, "Source changed during capture: " + key)
    sources[key] = digest
    return digest


def verify_smoke_predecessor(head):
    require((SMOKE / "audit_summary.json").exists(), "P2-2D-2 smoke exists")
    audit = json.loads((SMOKE / "audit_summary.json").read_bytes())
    require(
        audit["stage"] == "P2-2D-2-v1"
        and audit["mode"] == "smoke"
        and audit["HEAD"] == CHECKPOINT
        and audit["stage_pass"] is True
        and audit["scientific_status"] == "SHIELDED_POINT_PPO_SMOKE_PASS"
        and audit["failed_gates"] == []
        and all(audit["gates"].values())
        and audit["selected_config"] == "CONFIG_A"
        and audit["scientific_budget_used"] is False
        and audit["PPO_training_runs"] == 1
        and audit["Point_training_runs"] == 1
        and audit["UA_training_runs"] == 0
        and audit["heldout_ppo_trajectory_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-2D-2 smoke authority",
    )
    for name, digest in audit["output_sha256"].items():
        require(sha_file(SMOKE / name) == digest, "P2-2D-2 output hash: " + name)

    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned predecessor source unchanged: " + name,
        )
    return audit


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    training_calls = [
        n for n in calls
        if isinstance(n.func, ast.Attribute)
        and n.func.attr == "train_final_checkpoint"
    ]
    require(len(training_calls) == 1, "One development training call site")
    call = training_calls[0]
    require(
        len(call.keywords) == 1
        and call.keywords[0].arg == "purpose"
        and isinstance(call.keywords[0].value, ast.Constant)
        and call.keywords[0].value.value == "development",
        "Frozen development budget only",
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
    require("learn" not in attrs and "save" not in attrs and "load" not in attrs,
            "No direct PPO learn/save/load bypass")

    protected_role_calls = []
    for node in calls:
        values = list(node.args) + [kw.value for kw in node.keywords]
        for value in values:
            if isinstance(value, ast.Constant) and value.value in (
                "FORMAL HELD-OUT", "RANDOM_TEST", "SEALED_DATES", "UA"
            ):
                protected_role_calls.append(value.value)
    require(not protected_role_calls, "No formal/protected/UA runtime constructor literal")
    return {
        "single_development_training_call_site": True,
        "shielded_runtime_only": True,
        "no_unshielded_runtime_bypass": True,
        "no_direct_learn_save_load": True,
        "no_formal_protected_or_UA_runtime_call": True,
    }


def summarize_metrics(rows):
    require(len(rows) > 0, "Nonempty evaluation rows")
    costs = [float(r["J_total"]) for r in rows]
    cleans = [int(r["N_clean"]) for r in rows]
    proposed = [int(r["N_clean_proposed"]) for r in rows]
    interventions = [int(r["shield_interventions"]) for r in rows]
    intervention_fractions = [float(r["shield_intervention_fraction"]) for r in rows]
    ordered = sorted(costs)
    cvar_count = math.ceil(0.05 * len(ordered))
    return {
        "episode_count": len(rows),
        "mean_J_total": float(statistics.fmean(costs)),
        "std_J_total": float(statistics.stdev(costs)) if len(rows) > 1 else 0.0,
        "median_J_total": float(statistics.median(costs)),
        "P95_J_total": float(np.quantile(costs, 0.95, method="linear")),
        "CVaR95_J_total": float(statistics.fmean(ordered[-cvar_count:])),
        "CVaR95_episode_count": int(cvar_count),
        "mean_N_clean_executed": float(statistics.fmean(cleans)),
        "mean_N_clean_proposed": float(statistics.fmean(proposed)),
        "mean_shield_interventions": float(statistics.fmean(interventions)),
        "mean_shield_intervention_fraction": float(statistics.fmean(intervention_fractions)),
        "total_N_clean_executed": int(sum(cleans)),
        "total_N_clean_proposed": int(sum(proposed)),
        "total_shield_interventions": int(sum(interventions)),
        "episodes_with_any_proposed_clean": int(sum(v > 0 for v in proposed)),
        "episodes_with_any_shield_intervention": int(sum(v > 0 for v in interventions)),
        "episodes_with_zero_proposed_clean": int(sum(v == 0 for v in proposed)),
        "executed_clean_minus_proposed_clean": int(sum(cleans) - sum(proposed)),
        "shield_share_of_executed_clean": 0.0 if sum(cleans) == 0 else float(sum(interventions) / sum(cleans)),
    }


class AlwaysWaitModel:
    def predict(self, observation, deterministic=True):
        del observation, deterministic
        return np.asarray([0], dtype=np.int64), None


def evaluate_shield_only_baseline(runner, assets, specs):
    ledger = runner.AccessLedger()
    rows = runner.evaluate_shielded_perception_deterministic(
        AlwaysWaitModel(), assets, specs, ledger, "Point"
    )
    require(
        len(rows) == len(specs)
        and all(r["N_clean_proposed"] == 0 for r in rows)
        and all(r["N_clean"] == r["shield_interventions"] for r in rows),
        "Exact shield-only always-WAIT baseline semantics",
    )
    require(
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0,
        "Shield-only baseline DEVELOPMENT access only",
    )
    return rows, summarize_metrics(rows), ledger


def run_formal():
    require(not STAGE_DIRECTORY.exists(), "Exclusive stage directory; no overwrite/resume/retry")
    require(git("status", "--porcelain", "--untracked-files=all") == "",
            "Committed unchanged clean sources required")
    head = git("rev-parse", "HEAD")
    smoke_audit = verify_smoke_predecessor(head)

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(
        self_blob == git("hash-object", f"--path={SELF}", SELF) == git("rev-parse", f":{SELF}"),
        "New development source committed unchanged",
    )

    sources = {}
    for name in (SELF, RUNNER, SMOKE_SOURCE, BASE_RUNNER, SHIELD, PROTOCOL, CORE, TRUE_RUNNER):
        capture_source(sources, ROOT / name)

    static = static_contract()

    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as version_authority
    import paper2_ppo_protocol_v1 as frozen
    import paper2_shielded_perception_ppo_runner_v1 as runner

    versions = version_authority.dependency_versions()
    p = runner.get_protocol()
    require(versions == dict(p.dependencies), "Frozen dependency versions")
    require(frozen.protocol_payload(p) == frozen.protocol_payload(frozen.Protocol()),
            "Exact immutable PPO protocol")
    require(runner.inherited_config_id() == "CONFIG_A", "CONFIG_A inherited; no reselection")

    config = runner.candidate("CONFIG_A")
    require(
        tuple(p.seeds.development) == (510001, 510002)
        and tuple(p.seeds.final) == (520001, 520002, 520003, 520004, 520005)
        and not set(p.seeds.development) & set(p.seeds.final),
        "Frozen disjoint development/final seed registry",
    )
    require(
        p.development_budget == 491520
        and p.final_budget == 983040
        and p.development_budget == p.selection.development_checkpoint
        and p.development_budget % config.n_steps == 0,
        "Frozen development budget",
    )
    require(
        dict(p.observations)["Point"] == ("q50", "sin_DOY", "cos_DOY")
        and dict(p.observation_dimensions)["Point"] == 3,
        "Frozen Point observation contract",
    )

    assets = runner.load_assets()
    assets.ensure_perception()
    for name, source_path in (
        ("paper1_cqr", assets.paper1_cqr),
        ("wapp_power_bridge", assets.wapp_power_bridge_path),
    ):
        require(sha_file(source_path) == assets.perception.EXPECTED_HASHES[name],
                "Frozen perception input: " + name)
        capture_source(sources, source_path)

    development = runner.partition("DEVELOPMENT")
    specs = tuple((year, tid) for year in development.years for tid in development.trajectory_ids)
    require(
        len(specs) == len(set(specs)) == development.population_size == p.evaluation.episodes_per_model == 600,
        "Full frozen DEVELOPMENT population",
    )

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        return execute_formal(
            runner=runner, frozen=frozen, p=p, config=config, assets=assets,
            specs=specs, development=development, head=head, versions=versions,
            sources=sources, self_blob=self_blob, static=static,
            smoke_audit=smoke_audit,
        )
    except Exception as exc:
        if not (OUTPUT / "execution_failure.json").exists():
            write_exclusive(
                OUTPUT / "execution_failure.json",
                encode({
                    "stage": STAGE,
                    "stage_pass": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "action": "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/HYPERPARAMETER, PERCEPTION OR SHIELD TUNING",
                    "HEAD": head,
                }),
            )
        raise


def execute_formal(*, runner, frozen, p, config, assets, specs, development,
                   head, versions, sources, self_blob, static, smoke_audit):
    gates = {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate: " + name)
        gates[name] = bool(condition)
        require(condition, name)

    gate("smoke_predecessor_exact", smoke_audit["stage_pass"] is True)
    gate("static_development_contract", all(static.values()))
    gate("frozen_config_and_budget",
         config.name == "CONFIG_A" and config.learning_rate == 3e-4
         and config.n_steps == 2048 and p.development_budget == 491520
         and p.final_budget == 983040)
    gate("exactly_two_development_seeds_final_seeds_unused",
         tuple(p.seeds.development) == (510001, 510002)
         and not set(p.seeds.development) & set(p.seeds.final))
    gate("full_development_population_contract",
         len(specs) == 600 and len(development.years) == 2
         and len(development.trajectory_ids) == 300)

    baseline_rows, baseline_summary, baseline_ledger = evaluate_shield_only_baseline(runner, assets, specs)
    gate("shield_only_baseline_complete",
         baseline_summary["episode_count"] == 600
         and baseline_summary["total_N_clean_proposed"] == 0
         and baseline_summary["total_shield_interventions"] == baseline_summary["total_N_clean_executed"])

    run_rows = []
    episode_rows = []
    evidence_hashes = {}
    training_roots = set(runner.partition("TRAINING").roots)
    development_roots = set(development.roots)

    for seed in p.seeds.development:
        label = f"seed_{seed}"
        directory = OUTPUT / "runs" / label
        directory.mkdir(parents=True, exist_ok=False)
        model_path = directory / "final_model.zip"

        ledger = runner.AccessLedger()
        env = runner.build_shielded_scheduled_perception_env(
            assets, seed, ledger, config.name, "Point"
        )
        local = {}

        def check(name, condition):
            require(name not in local, "Duplicate run gate: " + name)
            local[name] = bool(condition)
            gate(f"{label}:{name}", condition)

        try:
            check("fresh_training_environment",
                  env.observation_mode == "Point" and env.observation_space.shape == (3,)
                  and env.schedule.parent_seed == seed and not env.started
                  and not env.failed and not env.assignments)
            bundle = runner.build_shielded_perception_ppo_model(env, config.name, seed)
            check("frozen_PPO_model_contract",
                  bundle.model.num_timesteps == 0 and not bundle.training_finished
                  and runner.verify_model_contract(bundle.model, config.name, seed))

            training = runner.train_final_checkpoint(bundle, model_path, purpose="development")
            check("development_budget_exact",
                  training["total_timesteps"] == env.total_steps == p.development_budget
                  and training["rollout_updates"] == p.development_budget // config.n_steps
                  and training["scientific_budget_used"] is True)
            check("final_budget_and_final_seeds_unused",
                  seed not in p.seeds.final and training["total_timesteps"] != p.final_budget)
            check("parameters_finite_final_checkpoint_only",
                  training["model_parameter_finite"] and training["optimizer_steps_checked"] > 0
                  and training["final_checkpoint_only"] is True and bundle.training_finished
                  and runner.parameters_finite(bundle.model))

            shield_training = env._shield_audit_snapshot()
            check("training_shield_accounting",
                  shield_training["total_safety_filtered_decisions"]
                  + shield_training["total_terminal_passthrough"] == env.total_steps
                  and shield_training["total_executed_clean"]
                  == shield_training["total_proposed_clean"] + shield_training["total_shield_interventions"]
                  and shield_training["total_free_wait"] + shield_training["total_proposed_clean"]
                  + shield_training["total_shield_interventions"] == env.total_steps)
            check("training_perception_query_accounting",
                  env.perception_query_count == env.total_steps + len(env.assignments) - env.episodes_completed
                  and env.observations_checked == env.total_steps + len(env.assignments)
                  and all(row["perception_query_count"] == row["natural_transition_count"] + 1
                          for row in env.assignments))

            training_records = [r for r in ledger.records if r["role"] == "TRAINING"]
            check("TRAINING_roots_only",
                  len(training_records) == len(env.assignments)
                  and all(r["environment_root"] in training_roots and r["observation_mode"] == "Point"
                          for r in training_records))

            schedule = runner.build_training_schedule(seed, config.name)
            expected = [schedule.next_assignment() for _ in env.assignments]
            actual = [{k: row[k] for k in expected[0]} for row in env.assignments]
            check("frozen_training_schedule_exact", actual == expected)

            loaded = runner.reload_final_checkpoint(model_path, bundle)
            check("save_reload_exact_and_finite",
                  loaded is not bundle.model and runner.parameters_finite(loaded)
                  and runner.file_sha256(model_path) == training["model_sha256"])

            metrics = runner.evaluate_shielded_perception_deterministic(
                loaded, assets, specs, ledger, "Point"
            )
            evaluation_records = [r for r in ledger.records if r["role"] == "DEVELOPMENT"]
            check("full_600_DEVELOPMENT_evaluation",
                  len(metrics) == len(evaluation_records) == p.evaluation.episodes_per_model == 600
                  and all(r["step_count"] == p.decision_reward_steps
                          and r["natural_transition_count"] == p.natural_transitions
                          and r["deterministic_action"] for r in metrics))
            check("DEVELOPMENT_roots_only",
                  all(r["environment_root"] in development_roots and r["observation_mode"] == "Point"
                      for r in evaluation_records))
            check("finite_eval_reward_cost_identity",
                  all(r["all_observations_finite"] and r["all_rewards_finite"] and r["all_states_finite"]
                      and r["reward_sign_regression_diff"] <= p.selection.atol for r in metrics))
            check("eval_proposal_execution_shield_identity",
                  all(r["N_clean"] >= r["N_clean_proposed"]
                      and r["N_clean"] - r["N_clean_proposed"] == r["shield_interventions"]
                      and 0.0 <= r["shield_intervention_fraction"] <= 1.0 for r in metrics))
            check("zero_formal_or_protected_access",
                  ledger.heldout_ppo_trajectory_access_count == 0
                  and ledger.formal_perception_seed_access_count == 0
                  and ledger.denied_accesses == 0)

            summary = summarize_metrics(metrics)
            training_fraction = (
                shield_training["total_shield_interventions"]
                / shield_training["total_safety_filtered_decisions"]
            )
            children = runner.child_seeds(seed)
            run_row = {
                "config_id": config.name,
                "parent_rl_seed": int(seed),
                "learner_seed": int(children["learner"]),
                "schedule_seed": int(children["training_schedule"]),
                "total_timesteps": int(training["total_timesteps"]),
                "rollout_updates": int(training["rollout_updates"]),
                "wall_clock_training_seconds": float(training["wall_clock_seconds"]),
                "model_sha256": training["model_sha256"],
                "training_assignment_count": len(env.assignments),
                "training_completed_episodes": int(env.episodes_completed),
                "training_perception_query_count": int(env.perception_query_count),
                "training_proposed_clean": int(shield_training["total_proposed_clean"]),
                "training_executed_clean": int(shield_training["total_executed_clean"]),
                "training_shield_interventions": int(shield_training["total_shield_interventions"]),
                "training_shield_intervention_fraction": float(training_fraction),
                **summary,
                "mean_J_gap_vs_shield_only": float(summary["mean_J_total"] - baseline_summary["mean_J_total"]),
                "mean_intervention_fraction_gap_vs_shield_only": float(
                    summary["mean_shield_intervention_fraction"]
                    - baseline_summary["mean_shield_intervention_fraction"]
                ),
                "formal_access_count": int(ledger.heldout_ppo_trajectory_access_count),
                "formal_perception_seed_access_count": int(ledger.formal_perception_seed_access_count),
                "scientific_budget_used": True,
                "final_checkpoint_only": True,
            }
            run_rows.append(run_row)
            episode_rows.extend({"parent_rl_seed": int(seed), "config_id": config.name, **row}
                                for row in metrics)

            run_audit = {
                "stage": STAGE,
                "run": label,
                "gates": local,
                "run_summary": run_row,
                "training_assignments": env.assignments,
                "trajectory_access_log": ledger.records,
                "model_manifest": runner.model_manifest(bundle, training),
                "shield_training": shield_training,
            }
            run_audit_bytes = encode(run_audit)
            write_exclusive(directory / "run_audit.json", run_audit_bytes)
            evidence_hashes[(directory / "run_audit.json").relative_to(OUTPUT).as_posix()] = sha_bytes(run_audit_bytes)
            evidence_hashes[model_path.relative_to(OUTPUT).as_posix()] = training["model_sha256"]
        finally:
            env.close()

    gate("exactly_two_development_training_runs", len(run_rows) == 2)
    gate("exactly_1200_DEVELOPMENT_evaluation_episodes", len(episode_rows) == 1200)
    gate("all_run_gates_passed", all(value for name, value in gates.items() if name.startswith("seed_")))

    seed_means = [float(r["mean_J_total"]) for r in run_rows]
    seed_intervention = [float(r["mean_shield_intervention_fraction"]) for r in run_rows]
    development_summary = {
        "stage": STAGE,
        "statistical_unit": "RL TRAINING SEED",
        "independent_training_seed_count": 2,
        "development_seeds": list(p.seeds.development),
        "config_id": config.name,
        "development_budget_per_seed": p.development_budget,
        "evaluation_episodes_per_seed": 600,
        "two_seed_mean_of_seed_mean_J_total": float(statistics.fmean(seed_means)),
        "two_seed_std_of_seed_mean_J_total": float(statistics.stdev(seed_means)),
        "two_seed_mean_of_seed_intervention_fraction": float(statistics.fmean(seed_intervention)),
        "seed_level": [
            {
                "parent_rl_seed": r["parent_rl_seed"],
                "mean_J_total": r["mean_J_total"],
                "std_J_total": r["std_J_total"],
                "mean_N_clean_executed": r["mean_N_clean_executed"],
                "mean_N_clean_proposed": r["mean_N_clean_proposed"],
                "mean_shield_interventions": r["mean_shield_interventions"],
                "mean_shield_intervention_fraction": r["mean_shield_intervention_fraction"],
                "episodes_with_any_proposed_clean": r["episodes_with_any_proposed_clean"],
                "episodes_with_zero_proposed_clean": r["episodes_with_zero_proposed_clean"],
                "mean_J_gap_vs_shield_only": r["mean_J_gap_vs_shield_only"],
                "mean_intervention_fraction_gap_vs_shield_only": r["mean_intervention_fraction_gap_vs_shield_only"],
            }
            for r in run_rows
        ],
        "shield_only_baseline": baseline_summary,
        "readiness_diagnostics_descriptive_only": {
            "each_seed_proposes_clean_somewhere": all(r["total_N_clean_proposed"] > 0 for r in run_rows),
            "each_seed_mean_cost_below_shield_only": all(r["mean_J_total"] < baseline_summary["mean_J_total"] for r in run_rows),
            "each_seed_mean_intervention_fraction_below_shield_only": all(
                r["mean_shield_intervention_fraction"] < baseline_summary["mean_shield_intervention_fraction"]
                for r in run_rows
            ),
            "role": "DESCRIPTIVE ONLY; NOT USED TO TUNE, SELECT, RETRY OR RESEED",
        },
        "claim_boundary": "DEVELOPMENT characterization only; no final-seed or formal-held-out performance claim",
    }

    protocol_manifest = {
        "stage": STAGE,
        "mode": "formal",
        "declaration": DECLARATION,
        "protocol": frozen.protocol_payload(p),
        "selected_config": config.name,
        "observation_mode": "Point",
        "development_seeds": list(p.seeds.development),
        "development_budget_per_seed": p.development_budget,
        "final_budget_unused": True,
        "full_development_specs": [[year, int(tid)] for year, tid in specs],
        "deterministic_evaluation": p.evaluation.deterministic,
        "shield": "frozen q50-only set-membership predictive shield",
        "shield_inputs": ["q50_float32_history", "executed_action_history"],
        "policy_action_semantics": "PPO proposal -> frozen shield -> executed action -> frozen reward/physics",
        "shield_only_baseline": "always-WAIT proposal under the identical frozen shield; descriptive only",
        "performance_gates": None,
        "performance_diagnostics_role": "descriptive development readiness only; no tuning/reselection",
        "PPO_hyperparameter_reselection": False,
        "FINAL_seed_training_runs": 0,
        "formal_RL_access_count": 0,
        "formal_perception_seed_access_count": 0,
        "RANDOM_TEST_access_count": 0,
        "SEALED_DATES_access_count": 0,
    }

    source_manifest = {
        "HEAD": head,
        "predecessor_checkpoint": CHECKPOINT,
        "source_sha256": sources,
        "new_source_git_blob": self_blob,
        "pinned_predecessor_blobs": PINNED_BLOBS,
        "dependencies": versions,
        "core_assets_provenance": assets.provenance,
        "smoke_output_sha256": smoke_audit["output_sha256"],
    }

    documents = {
        "protocol_manifest.json": protocol_manifest,
        "development_summary.json": development_summary,
        "source_artifact_hashes.json": source_manifest,
    }
    content = {name: encode(document) for name, document in documents.items()}
    content["development_run_summary.csv"] = csv_bytes(run_rows)
    content["development_episode_metrics.csv"] = csv_bytes(episode_rows)
    content["shield_only_development_metrics.csv"] = csv_bytes(baseline_rows)

    for name, data in content.items():
        write_exclusive(OUTPUT / name, data)

    hashes = {**evidence_hashes, **{name: sha_bytes(data) for name, data in content.items()}}
    output_hashes = encode({
        "hashes": hashes,
        "exclusions": {
            "output_hashes.json": "SHA256 pinned in audit_summary; avoid self-reference",
            "audit_summary.json": "Published last PASS marker",
        },
    })
    write_exclusive(OUTPUT / "output_hashes.json", output_hashes)

    expected_inventory = set(content) | set(evidence_hashes) | {"output_hashes.json"}
    actual_inventory = {item.relative_to(OUTPUT).as_posix() for item in OUTPUT.rglob("*") if item.is_file()}
    gate("output_inventory_complete", actual_inventory == expected_inventory)
    gate("output_hashes_exact",
         json.loads(output_hashes)["hashes"]
         == {name: runner.file_sha256(OUTPUT / name) for name in hashes})
    gate("source_HEAD_and_worktree_unchanged",
         git("rev-parse", "HEAD") == head
         and git("status", "--porcelain", "--untracked-files=all") == ""
         and all(sha_file(ROOT / name) == digest for name, digest in sources.items()))
    gate("smoke_evidence_preserved",
         verify_smoke_predecessor(head)["output_sha256"] == smoke_audit["output_sha256"])
    gate("audit_summary_published_last", not (OUTPUT / "audit_summary.json").exists())

    require(all(gates.values()), "All development structural gates required")

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
        "PPO_training_runs": 2,
        "Point_training_runs": 2,
        "UA_training_runs": 0,
        "development_seed_count": 2,
        "development_evaluation_episode_count": 1200,
        "shield_only_baseline_episode_count": 600,
        "scientific_budget_used": True,
        "final_budget_used": False,
        "final_seed_training_runs": 0,
        "heldout_ppo_trajectory_access_count": 0,
        "formal_perception_seed_access_count": 0,
        "RANDOM_TEST_access_count": 0,
        "SEALED_DATES_access_count": 0,
        "two_seed_mean_of_seed_mean_J_total": development_summary["two_seed_mean_of_seed_mean_J_total"],
        "two_seed_mean_of_seed_intervention_fraction": development_summary["two_seed_mean_of_seed_intervention_fraction"],
        "readiness_diagnostics_descriptive_only": development_summary["readiness_diagnostics_descriptive_only"],
        "output_sha256": {**hashes, "output_hashes.json": sha_bytes(output_hashes)},
        "audit_summary_published_last": True,
    }
    write_exclusive(OUTPUT / "audit_summary.json", encode(summary))
    require(
        {item.relative_to(OUTPUT).as_posix() for item in OUTPUT.rglob("*") if item.is_file()}
        == expected_inventory | {"audit_summary.json"},
        "Final output inventory",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
