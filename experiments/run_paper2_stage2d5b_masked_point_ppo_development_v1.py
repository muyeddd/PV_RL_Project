
"""P2-2D-5B-v1: full Masked Point-PPO DEVELOPMENT characterization.

Runs exactly the frozen two DEVELOPMENT seeds (510001, 510002), each for the
frozen 491,520-step DEVELOPMENT budget, then evaluates each deterministic final
checkpoint on all 600 DEVELOPMENT episodes.

Only the action-selection mechanism differs from the frozen post-hoc Point-PPO:
q50-history safety masking is applied before MaskablePPO action selection, while
the unchanged q50 shield remains active as a fail-safe. The mask/shield,
Point observation, BLOCK10 v1 perception, reward, physics, PPO CONFIG_A,
network, seeds and budget are unchanged.

The already-frozen P2-2D-3R post-hoc Point-PPO and shield-only DEVELOPMENT
outputs are read only and paired episode-by-episode for descriptive comparison.

Performance is NEVER a structural PASS gate. No retry/reseed/tuning, no FINAL
seeds, FORMAL HELD-OUT, RANDOM_TEST, SEALED_DATES or UA runtime access.
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
SELF = "experiments/run_paper2_stage2d5b_masked_point_ppo_development_v1.py"
RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1_1.py"
SMOKE_SOURCE = "experiments/run_paper2_stage2d5ar_masked_point_ppo_smoke_recovery_v1.py"
POSTHOC_SOURCE = "experiments/run_paper2_stage2d3r_shielded_point_ppo_development_recovery_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
FAILED_MASKED_RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1.py"
SHIELDED_RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
PERCEPTION_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
MASKED_REQUIREMENTS = "requirements-paper2-masked-v1.txt"
BASE_REQUIREMENTS = "requirements.txt"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
SMOKE = BASE / "p2_2d_5ar_masked_point_ppo_smoke_recovery_v1/smoke"
POSTHOC = BASE / "p2_2d_3r_shielded_point_ppo_development_recovery_v1/formal"
STAGE_DIRECTORY = BASE / "p2_2d_5b_masked_point_ppo_development_v1"
OUTPUT = STAGE_DIRECTORY / "formal"

CHECKPOINT = "fa87664e043302a401206758b05706da6a3e1e62"
POSTHOC_HEAD = "bebde79b87c43b8faf7fda2a399522ad5e8f3855"
DEVELOPMENT_SEEDS = (510001, 510002)
DEVELOPMENT_BUDGET = 491520
N_DEV = 600
ATOL = 1e-12

PINNED_BLOBS = {
    RUNNER: "9b1777ce4262efb95c677432ea52997352d76750",
    SMOKE_SOURCE: "b88504d151992d00b96a9db4139bc6118958c078",
    POSTHOC_SOURCE: "b7571d873c35474bec4ffcd61b0621f0c76f5f94",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    FAILED_MASKED_RUNNER: "be86a5e8a8a162c0d6be720c5340914ccc1ed82e",
    SHIELDED_RUNNER: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    PERCEPTION_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
    MASKED_REQUIREMENTS: "8bb5ad1e5b6d84eb31b7a207aefde57cc0e3b88c",
    BASE_REQUIREMENTS: "430357153fbdc6b2b77d92d8040333d2de30135b",
}

EXPECTED_5AR_HASHES = {
    "protocol_manifest.json": "ba804a2760b6a9e956ea12aa775dc2ac121db25366db5de312eab5450dd8e5bd",
    "recovery_provenance.json": "f6551b48e0e17c744f43aa263deddca426b8d02211e90520e820ea5fd56ef515",
    "masked_smoke_training_summary.json": "46e0bd5d2725f410775fadc0af6f8e36a6c57587d419a7f2c6515f9c33596bb8",
    "masked_smoke_evaluation_metrics.csv": "954279ff06a2780ef17548fb45da023055fea2fcbcda4eaa1075a2863a2662a1",
    "training_episode_assignments.csv": "764b1f7d6d51bfca411b321af54a48a3d3d2c8d32710d91ec6a655e0c5c658d1",
    "recovered_masked_ppo_construction.json": "8ba26f5448c76e8337d9b2645cfa8ced55f2dd107276fd57e3af4d4a5fe9f489",
    "final_masked_smoke_model.zip": "1fadce0b5e92c2465319affb1efe51c25b02c622563d5b713791b91e7410e626",
    "output_hashes.json": "d916bcc59dfdb86c44051646477c313b37cfa17df938a987fec7821185b70369",
}

STAGE = "P2-2D-5B-v1"
STATUS = "MASKED_POINT_PPO_DEVELOPMENT_CHARACTERIZATION_PASS"
SCHEDULE_FIELDS = (
    "episode_ordinal", "cycle_index", "position_in_cycle", "year",
    "trajectory_id", "environment_root", "population_size",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_bytes(rows):
    require(rows and all(set(r) == set(rows[0]) for r in rows), "CSV schema")
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode()


def write_exclusive(path, data):
    path = Path(path)
    with path.open("xb") as f:
        f.write(data)
    require(path.read_bytes() == data, "Exact output readback: " + path.name)


def verify_sources(head):
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned source changed: " + name,
        )


def verify_5ar(head):
    audit_path = SMOKE / "audit_summary.json"
    require(audit_path.exists(), "P2-2D-5AR audit exists")
    audit = json.loads(audit_path.read_bytes())
    require(
        audit["stage"] == "P2-2D-5AR-v1"
        and audit["mode"] == "smoke"
        and audit["stage_pass"] is True
        and audit["scientific_status"] == "MASKED_POINT_PPO_SMOKE_API_RECOVERY_PASS"
        and audit["HEAD"] == CHECKPOINT
        and audit["failed_gates"] == []
        and all(audit["gates"].values())
        and audit["runtime_algorithm"] == "MaskablePPO"
        and audit["training_timesteps"] == 4096
        and audit["training_shield_interventions"] == 0
        and audit["evaluation_shield_interventions"] == 0
        and audit["UA_training_runs"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-2D-5AR",
    )
    require(audit["output_sha256"] == EXPECTED_5AR_HASHES, "P2-2D-5AR hash registry")
    for name, digest in EXPECTED_5AR_HASHES.items():
        require(sha_file(SMOKE / name) == digest, "P2-2D-5AR hash: " + name)
    verify_sources(head)
    return sha_file(audit_path)


def verify_posthoc(head):
    audit_path = POSTHOC / "audit_summary.json"
    require(audit_path.exists(), "P2-2D-3R audit exists")
    audit = json.loads(audit_path.read_bytes())
    require(
        audit["stage"] == "P2-2D-3R-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"] == "SHIELDED_POINT_PPO_DEVELOPMENT_RECOVERY_PASS"
        and audit["HEAD"] == POSTHOC_HEAD
        and audit["cumulative_development_model_count"] == 2
        and audit["development_evaluation_episode_count"] == 1200
        and audit["shield_only_baseline_episode_count"] == 600
        and audit["final_seed_training_runs"] == 0
        and audit["heldout_ppo_trajectory_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0
        and audit["failed_gates"] == []
        and all(audit["gates"].values()),
        "Accepted P2-2D-3R",
    )
    for name, digest in audit["output_sha256"].items():
        require(sha_file(POSTHOC / name) == digest, "P2-2D-3R hash: " + name)
    require(
        git("rev-parse", f"{POSTHOC_HEAD}:{POSTHOC_SOURCE}") == PINNED_BLOBS[POSTHOC_SOURCE]
        and git("rev-parse", f"{head}:{POSTHOC_SOURCE}") == PINNED_BLOBS[POSTHOC_SOURCE],
        "P2-2D-3R source unchanged",
    )

    posthoc_rows = read_csv(POSTHOC / "development_episode_metrics.csv")
    baseline_rows = read_csv(POSTHOC / "shield_only_development_metrics.csv")
    require(len(posthoc_rows) == 1200 and len(baseline_rows) == 600, "Predecessor populations")

    posthoc = {}
    for r in posthoc_rows:
        key = (int(r["parent_rl_seed"]), r["year"], int(r["trajectory_id"]))
        require(key not in posthoc, "Unique post-hoc key")
        posthoc[key] = r
    baseline = {}
    for r in baseline_rows:
        key = (r["year"], int(r["trajectory_id"]))
        require(key not in baseline, "Unique baseline key")
        baseline[key] = r

    for seed in DEVELOPMENT_SEEDS:
        for key, b in baseline.items():
            p = posthoc[(seed, key[0], key[1])]
            require(
                int(p["N_clean_proposed"]) == 0
                and int(p["N_clean"]) == int(p["shield_interventions"])
                and abs(float(p["J_total"]) - float(b["J_total"])) <= ATOL
                and int(p["N_clean"]) == int(b["N_clean"]),
                "Frozen post-hoc collapse equals shield-only",
            )
    return sha_file(audit_path), posthoc, baseline


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    trains = [
        n for n in calls
        if isinstance(n.func, ast.Attribute)
        and n.func.attr == "train_masked_final_checkpoint"
    ]
    require(len(trains) == 1, "One DEVELOPMENT training call site")
    call = trains[0]
    require(
        len(call.keywords) == 1
        and call.keywords[0].arg == "purpose"
        and isinstance(call.keywords[0].value, ast.Constant)
        and call.keywords[0].value.value == "development",
        "DEVELOPMENT budget only",
    )
    require(
        "build_masked_scheduled_perception_env" in attrs
        and "evaluate_masked_perception_deterministic" in attrs,
        "Masked runtime only",
    )
    require(
        "build_shielded_scheduled_perception_env" not in attrs
        and "evaluate_shielded_perception_deterministic" not in attrs
        and "build_perception_ppo_model" not in attrs,
        "No post-hoc/unmasked runtime bypass",
    )
    require("learn" not in attrs and "save" not in attrs and "load" not in attrs,
            "No direct learn/save/load bypass")
    protected = []
    for node in calls:
        values = list(node.args) + [kw.value for kw in node.keywords]
        for value in values:
            if (
                isinstance(value, ast.Constant)
                and value.value in (
                    "FORMAL HELD-OUT",
                    "RANDOM_TEST",
                    "SEALED_DATES",
                    "UA",
                )
            ):
                protected.append(value.value)
    require(not protected, "No protected/UA runtime constructor literal")

    require(
        "performance_diagnostics_not_pass_gates" in source
        and "paired_comparison_descriptive_only" in source
        and "zero_training_shield_interventions" in source
        and "zero_evaluation_shield_interventions" in source,
        "Scientific/structural separation present",
    )
    return {
        "single_development_training_call_site": True,
        "masked_runtime_only": True,
        "no_posthoc_or_unmasked_runtime_bypass": True,
        "no_direct_learn_save_load": True,
        "Point_only_no_protected_runtime": True,
        "zero_intervention_contract_present": True,
        "performance_diagnostics_not_pass_gates": True,
        "paired_comparison_descriptive_only": True,
    }


def corrected_schedule_audit(runner, env, seed, config_id):
    schedule = runner.build_training_schedule(seed, config_id)
    expected = [schedule.next_assignment() for _ in env.assignments]
    actual_latent = [{f: r[f] for f in SCHEDULE_FIELDS} for r in env.assignments]
    expected_latent = [{f: r[f] for f in SCHEDULE_FIELDS} for r in expected]
    perception_seed = runner.get_perception_contract().development_seed
    require(actual_latent == expected_latent, "Exact latent training schedule")
    require(all(r["perception_seed"] == perception_seed for r in env.assignments),
            "Frozen injected DEV perception seed")
    require(all(r["perception_seed"] is None for r in expected),
            "Raw True-State schedule perception_seed remains None")
    return {
        "assignment_count": len(env.assignments),
        "latent_schedule_exact": True,
        "runtime_perception_seed_exact": True,
        "runtime_injected_perception_seed": perception_seed,
        "compared_latent_fields": list(SCHEDULE_FIELDS),
    }


def summarize(rows):
    costs = [float(r["J_total"]) for r in rows]
    forced = [int(r["N_forced_clean"]) for r in rows]
    voluntary = [int(r["N_voluntary_clean"]) for r in rows]
    free = [int(r["N_free_choice"]) for r in rows]
    ratios = [float(r["voluntary_clean_fraction_given_free"]) for r in rows]
    interventions = [int(r["N_shield_interventions"]) for r in rows]
    ordered = sorted(costs)
    k = math.ceil(0.05 * len(ordered))
    return {
        "episode_count": len(rows),
        "mean_J_total": float(statistics.fmean(costs)),
        "std_J_total": float(statistics.stdev(costs)),
        "median_J_total": float(statistics.median(costs)),
        "P95_J_total": float(np.quantile(costs, 0.95, method="linear")),
        "CVaR95_J_total": float(statistics.fmean(ordered[-k:])),
        "mean_N_forced_clean": float(statistics.fmean(forced)),
        "mean_N_voluntary_clean": float(statistics.fmean(voluntary)),
        "mean_N_free_choice": float(statistics.fmean(free)),
        "mean_voluntary_clean_fraction_given_free": float(statistics.fmean(ratios)),
        "total_N_forced_clean": int(sum(forced)),
        "total_N_voluntary_clean": int(sum(voluntary)),
        "total_N_free_choice": int(sum(free)),
        "total_shield_interventions": int(sum(interventions)),
        "episodes_with_any_voluntary_clean": int(sum(v > 0 for v in voluntary)),
        "episodes_with_zero_voluntary_clean": int(sum(v == 0 for v in voluntary)),
    }


def evaluate_full(runner, model, assets, specs, seed):
    ledger = runner.AccessLedger()
    rows = runner.evaluate_masked_perception_deterministic(
        model, assets, specs, ledger, "Point"
    )
    p = runner.get_protocol()
    require(
        len(rows) == N_DEV
        and all(
            r["step_count"] == p.decision_reward_steps
            and r["natural_transition_count"] == p.natural_transitions
            and r["N_shield_interventions"] == 0
            and r["N_forced_clean"] + r["N_free_choice"] == p.natural_transitions
            and r["N_free_choice"] == r["N_voluntary_clean"] + r["N_free_wait"]
            and r["N_clean"] == r["N_forced_clean"] + r["N_voluntary_clean"] + r["N_terminal_clean"]
            and r["terminal_mask"] == [True, True]
            and r["all_observations_finite"]
            and r["all_rewards_finite"]
            and r["all_states_finite"]
            and r["reward_sign_regression_diff"] <= p.selection.atol
            for r in rows
        ),
        f"seed_{seed}: full zero-intervention DEVELOPMENT evaluation",
    )
    require(
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0,
        f"seed_{seed}: DEVELOPMENT access only",
    )
    return rows, summarize(rows), ledger


def build_pairs(seed, masked_rows, posthoc, baseline):
    rows = []
    for m in masked_rows:
        year, tid = m["year"], int(m["trajectory_id"])
        p = posthoc[(seed, year, tid)]
        b = baseline[(year, tid)]
        delta = float(m["J_total"]) - float(b["J_total"])
        require(abs(float(p["J_total"]) - float(b["J_total"])) <= ATOL,
                "Post-hoc/baseline cost identity")
        relation = (
            "MASKED_LOWER_COST" if delta < -ATOL
            else "MASKED_HIGHER_COST" if delta > ATOL
            else "COST_TIE_WITHIN_ATOL"
        )
        rows.append({
            "parent_rl_seed": seed,
            "year": year,
            "trajectory_id": tid,
            "J_masked": float(m["J_total"]),
            "J_posthoc_point": float(p["J_total"]),
            "J_shield_only": float(b["J_total"]),
            "delta_J_masked_minus_posthoc": float(m["J_total"]) - float(p["J_total"]),
            "delta_J_masked_minus_shield_only": delta,
            "paired_cost_relation_vs_shield_only": relation,
            "N_forced_clean_masked": int(m["N_forced_clean"]),
            "N_voluntary_clean_masked": int(m["N_voluntary_clean"]),
            "N_free_choice_masked": int(m["N_free_choice"]),
            "voluntary_clean_fraction_given_free":
                float(m["voluntary_clean_fraction_given_free"]),
            "N_shield_interventions_masked": int(m["N_shield_interventions"]),
            "N_clean_posthoc": int(p["N_clean"]),
            "shield_interventions_posthoc": int(p["shield_interventions"]),
            "N_clean_shield_only": int(b["N_clean"]),
        })
    require(len(rows) == N_DEV, "Complete paired population")
    return rows


def summarize_pairs(rows):
    d = [float(r["delta_J_masked_minus_shield_only"]) for r in rows]
    lower = sum(v < -ATOL for v in d)
    higher = sum(v > ATOL for v in d)
    tie = len(d) - lower - higher
    return {
        "episode_count": len(rows),
        "mean_delta_J_masked_minus_shield_only": float(statistics.fmean(d)),
        "median_delta_J_masked_minus_shield_only": float(statistics.median(d)),
        "masked_lower_cost_episode_count": int(lower),
        "masked_higher_cost_episode_count": int(higher),
        "tie_episode_count": int(tie),
        "masked_lower_cost_episode_fraction": float(lower / len(d)),
        "masked_higher_cost_episode_fraction": float(higher / len(d)),
        "tie_episode_fraction": float(tie / len(d)),
    }


def run_formal():
    require(not STAGE_DIRECTORY.exists(), "Exclusive stage directory; no overwrite/resume/retry")
    require(git("status", "--porcelain", "--untracked-files=all") == "",
            "Committed unchanged clean worktree required")

    head = git("rev-parse", "HEAD")
    verify_sources(head)
    smoke_sha = verify_5ar(head)
    posthoc_sha, posthoc, baseline = verify_posthoc(head)

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(
        self_blob == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "New source committed unchanged",
    )
    static = static_contract()
    source_sha256 = {
        name: sha_file(ROOT / name)
        for name in (SELF, *PINNED_BLOBS.keys())
    }

    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as version_authority
    import paper2_ppo_protocol_v1 as frozen
    import paper2_masked_perception_ppo_runner_v1_1 as runner

    p = runner.get_protocol()
    config = runner.candidate("CONFIG_A")
    base_versions = version_authority.dependency_versions()
    masked_versions = runner.masked_dependency_versions()
    api_contract = runner.verify_maskable_api_contract()
    require(base_versions == dict(p.dependencies),
            "Frozen base dependencies")
    require(masked_versions == {
        "stable-baselines3": "2.7.1", "sb3-contrib": "2.7.1"
    }, "Frozen masked dependencies")
    require(
        api_contract["frozen_use_sde"] is False
        and api_contract["frozen_sde_sample_freq"] == -1
        and api_contract["MaskablePPO_init_rejects_use_sde"] is True
        and api_contract["MaskablePPO_init_rejects_sde_sample_freq"] is True,
        "Frozen MaskablePPO API recovery contract",
    )
    require(frozen.protocol_payload(p) == frozen.protocol_payload(frozen.Protocol()),
            "Exact frozen protocol")
    require(
        tuple(p.seeds.development) == DEVELOPMENT_SEEDS
        and p.development_budget == DEVELOPMENT_BUDGET
        and p.development_budget == p.selection.development_checkpoint
        and config.name == runner.inherited_config_id() == "CONFIG_A"
        and dict(p.observations)["Point"] == ("q50", "sin_DOY", "cos_DOY"),
        "Frozen Point CONFIG_A DEVELOPMENT contract",
    )

    assets = runner.load_assets()
    assets.ensure_perception()
    dev = runner.partition("DEVELOPMENT")
    specs = tuple((year, tid) for year in dev.years for tid in dev.trajectory_ids)
    require(len(specs) == len(set(specs)) == N_DEV, "Full DEVELOPMENT population")

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        return execute(
            runner, frozen, p, config, assets, specs, head, self_blob,
            static, smoke_sha, posthoc_sha, posthoc, baseline,
            source_sha256, base_versions, masked_versions, api_contract
        )
    except Exception as exc:
        if not (OUTPUT / "execution_failure.json").exists():
            write_exclusive(OUTPUT / "execution_failure.json", encode({
                "stage": STAGE,
                "stage_pass": False,
                "HEAD": head,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "action":
                    "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/MASK/SHIELD/PERCEPTION/PPO TUNING",
            }))
        raise


def execute(runner, frozen, p, config, assets, specs, head, self_blob,
            static, smoke_sha, posthoc_sha, posthoc, baseline,
            source_sha256, base_versions, masked_versions, api_contract):
    gates = {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate: " + name)
        gates[name] = bool(condition)
        require(condition, name)

    gate("P2_2D_5AR_smoke_predecessor_exact", bool(smoke_sha))
    gate("P2_2D_3R_posthoc_predecessor_exact", bool(posthoc_sha))
    gate("static_development_contract", all(static.values()))
    gate("frozen_CONFIG_A_seed_budget_contract",
         tuple(p.seeds.development) == DEVELOPMENT_SEEDS
         and p.development_budget == DEVELOPMENT_BUDGET
         and config.learning_rate == 3e-4 and config.n_steps == 2048)

    run_rows, episode_rows, paired_rows = [], [], []
    model_paths, completion_paths = {}, {}
    access_ledgers = []

    for seed in DEVELOPMENT_SEEDS:
        label = f"seed_{seed}"
        run_dir = OUTPUT / "runs" / label
        run_dir.mkdir(parents=True, exist_ok=False)
        model_path = run_dir / "final_model.zip"
        completion_path = run_dir / "training_completion.json"

        ledger = runner.AccessLedger()
        env = runner.build_masked_scheduled_perception_env(
            assets=assets, rl_seed=seed, ledger=ledger,
            config_id=config.name, observation_mode="Point",
        )
        loaded = None
        try:
            gate(f"{label}:fresh_training_environment",
                 env.observation_space.shape == (3,)
                 and env.schedule.parent_seed == seed
                 and not env.started and not env.failed and not env.assignments)

            bundle = runner.build_masked_perception_ppo_model(env, config.name, seed)
            gate(f"{label}:frozen_MaskablePPO_contract",
                 bundle.model.num_timesteps == 0
                 and runner.verify_masked_model_contract(bundle.model, config.name, seed))

            training = runner.train_masked_final_checkpoint(
                bundle, model_path, purpose="development"
            )
            gate(f"{label}:exact_DEVELOPMENT_budget",
                 training["total_timesteps"] == env.total_steps == DEVELOPMENT_BUDGET
                 and training["scientific_budget_used"] is True
                 and training["rollout_updates"] == DEVELOPMENT_BUDGET // config.n_steps
                 and training["final_checkpoint_only"] is True
                 and runner.parameters_finite(bundle.model))

            ma = env._mask_audit_snapshot()
            gate(f"{label}:zero_training_shield_interventions",
                 ma["total_shield_interventions"] == 0
                 and ma["mask_shield_intervention_invariant"] is True
                 and ma["total_proposed_clean"] == ma["total_executed_clean"])
            gate(f"{label}:training_mask_accounting_exact",
                 ma["total_masked_safety_decisions"] + ma["total_terminal_passthrough"] == env.total_steps
                 and ma["total_masked_safety_decisions"] == ma["total_forced_clean"] + ma["total_free_choice_decisions"]
                 and ma["total_free_choice_decisions"] == ma["total_voluntary_clean"] + ma["total_mask_free_wait"])

            schedule_audit = corrected_schedule_audit(runner, env, seed, config.name)
            training_records = [r for r in ledger.records if r["role"] == "TRAINING"]
            gate(f"{label}:TRAINING_roots_only",
                 len(training_records) == len(env.assignments)
                 and all(r["environment_root"] in runner.partition("TRAINING").roots
                         and r["observation_mode"] == "Point" for r in training_records))

            completion = {
                "stage": STAGE,
                "parent_rl_seed": seed,
                "training": training,
                "model_sha256": sha_file(model_path),
                "model_manifest": runner.masked_model_manifest(bundle, training),
                "mask_training_audit": ma,
                "corrected_schedule_audit": schedule_audit,
                "training_access_log": training_records,
                "performance_diagnostics_not_pass_gates": {
                    "training_forced_clean": ma["total_forced_clean"],
                    "training_voluntary_clean": ma["total_voluntary_clean"],
                    "training_free_choice": ma["total_free_choice_decisions"],
                    "role": "DESCRIPTIVE ONLY; NOT A PASS GATE",
                },
            }
            write_exclusive(completion_path, encode(completion))

            loaded = runner.reload_masked_final_checkpoint(model_path, bundle)
            gate(f"{label}:save_reload_exact",
                 loaded is not bundle.model
                 and runner.parameters_finite(loaded)
                 and sha_file(model_path) == training["model_sha256"])
        finally:
            env.close()

        require(loaded is not None, f"{label}: checkpoint reloaded")
        rows, summary, eval_ledger = evaluate_full(runner, loaded, assets, specs, seed)
        gate(f"{label}:full_600_DEVELOPMENT_evaluation", summary["episode_count"] == N_DEV)
        gate(f"{label}:zero_evaluation_shield_interventions",
             summary["total_shield_interventions"] == 0)

        pairs = build_pairs(seed, rows, posthoc, baseline)
        ps = summarize_pairs(pairs)
        run_rows.append({
            "parent_rl_seed": seed,
            "config_id": config.name,
            "development_timesteps": DEVELOPMENT_BUDGET,
            "mean_J_total": summary["mean_J_total"],
            "std_J_total": summary["std_J_total"],
            "median_J_total": summary["median_J_total"],
            "P95_J_total": summary["P95_J_total"],
            "CVaR95_J_total": summary["CVaR95_J_total"],
            "mean_N_forced_clean": summary["mean_N_forced_clean"],
            "mean_N_voluntary_clean": summary["mean_N_voluntary_clean"],
            "mean_N_free_choice": summary["mean_N_free_choice"],
            "mean_voluntary_clean_fraction_given_free":
                summary["mean_voluntary_clean_fraction_given_free"],
            "episodes_with_any_voluntary_clean":
                summary["episodes_with_any_voluntary_clean"],
            "total_shield_interventions": summary["total_shield_interventions"],
            "mean_delta_J_masked_minus_shield_only":
                ps["mean_delta_J_masked_minus_shield_only"],
            "masked_lower_cost_episode_fraction":
                ps["masked_lower_cost_episode_fraction"],
            "masked_higher_cost_episode_fraction":
                ps["masked_higher_cost_episode_fraction"],
        })
        episode_rows.extend({"parent_rl_seed": seed, **r} for r in rows)
        paired_rows.extend(pairs)
        model_paths[seed] = model_path
        completion_paths[seed] = completion_path
        access_ledgers.extend((ledger, eval_ledger))

    gate("all_runtime_access_TRAINING_or_DEVELOPMENT_only",
         all(l.heldout_ppo_trajectory_access_count == 0
             and l.formal_perception_seed_access_count == 0
             and l.denied_accesses == 0 for l in access_ledgers))
    gate("paired_comparison_population_complete",
         len(episode_rows) == 1200 and len(paired_rows) == 1200)

    seed_summaries = {
        seed: summarize([r for r in episode_rows if int(r["parent_rl_seed"]) == seed])
        for seed in DEVELOPMENT_SEEDS
    }
    seed_pairs = {
        seed: summarize_pairs([r for r in paired_rows if int(r["parent_rl_seed"]) == seed])
        for seed in DEVELOPMENT_SEEDS
    }

    performance_diagnostics_not_pass_gates = {
        "each_seed_has_any_deterministic_voluntary_clean":
            all(seed_summaries[s]["episodes_with_any_voluntary_clean"] > 0
                for s in DEVELOPMENT_SEEDS),
        "any_deterministic_voluntary_clean_across_both_seeds":
            any(seed_summaries[s]["episodes_with_any_voluntary_clean"] > 0
                for s in DEVELOPMENT_SEEDS),
        "each_seed_mean_cost_below_frozen_shield_only":
            all(seed_pairs[s]["mean_delta_J_masked_minus_shield_only"] < -ATOL
                for s in DEVELOPMENT_SEEDS),
        "two_seed_mean_of_seed_mean_J_total":
            float(statistics.fmean(seed_summaries[s]["mean_J_total"]
                                   for s in DEVELOPMENT_SEEDS)),
        "two_seed_mean_of_seed_mean_voluntary_clean":
            float(statistics.fmean(seed_summaries[s]["mean_N_voluntary_clean"]
                                   for s in DEVELOPMENT_SEEDS)),
        "role":
            "DESCRIPTIVE DEVELOPMENT SCIENCE ONLY; NOT USED TO PASS, RETRY, RESEED, TUNE OR SELECT",
    }
    paired_comparison_descriptive_only = {
        "posthoc_point_predecessor": "P2-2D-3R-v1",
        "posthoc_point_equals_shield_only_verified": True,
        "seed_level": {str(s): seed_pairs[s] for s in DEVELOPMENT_SEEDS},
        "performance_gate": False,
    }

    development_summary = {
        "stage": STAGE,
        "runtime_algorithm": "MaskablePPO",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "development_budget_per_seed": DEVELOPMENT_BUDGET,
        "evaluation_episodes_per_seed": N_DEV,
        "seed_level": run_rows,
        "performance_diagnostics_not_pass_gates":
            performance_diagnostics_not_pass_gates,
        "paired_comparison_descriptive_only":
            paired_comparison_descriptive_only,
        "claim_boundary":
            "DEVELOPMENT characterization only; no FINAL/formal-heldout/RANDOM_TEST/SEALED_DATES/UA claim",
    }
    protocol_manifest = {
        "stage": STAGE,
        "mode": "formal",
        "protocol": frozen.protocol_payload(p),
        "selected_config": config.name,
        "runtime_algorithm": "MaskablePPO",
        "observation_mode": "Point",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "PPO_training_runs": 2,
        "Masked_Point_training_runs": 2,
        "UA_training_runs": 0,
        "development_budget_per_seed": DEVELOPMENT_BUDGET,
        "full_development_episode_count_per_model": N_DEV,
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
        "source_sha256": source_sha256,
        "pinned_predecessor_blobs": PINNED_BLOBS,
        "base_dependencies": base_versions,
        "masked_dependencies": masked_versions,
        "MaskablePPO_API_contract": api_contract,
        "core_assets_provenance": assets.provenance,
        "P2_2D_5AR_audit_summary_sha256": smoke_sha,
        "P2_2D_3R_audit_summary_sha256": posthoc_sha,
    }

    content = {
        "protocol_manifest.json": encode(protocol_manifest),
        "development_summary.json": encode(development_summary),
        "source_artifact_hashes.json": encode(provenance),
        "masked_development_run_summary.csv": csv_bytes(run_rows),
        "masked_development_episode_metrics.csv": csv_bytes(episode_rows),
        "paired_development_comparison.csv": csv_bytes(paired_rows),
    }
    for name, data in content.items():
        write_exclusive(OUTPUT / name, data)

    extras = {}
    for seed in DEVELOPMENT_SEEDS:
        extras[f"runs/seed_{seed}/training_completion.json"] = completion_paths[seed]
        extras[f"runs/seed_{seed}/final_model.zip"] = model_paths[seed]

    hashes = {**{n: sha_bytes(d) for n, d in content.items()},
              **{n: sha_file(pth) for n, pth in extras.items()}}
    output_hashes = encode({
        "hashes": hashes,
        "read_only_predecessor_evidence": {
            "P2_2D_5AR_audit_summary_sha256": smoke_sha,
            "P2_2D_3R_audit_summary_sha256": posthoc_sha,
        },
        "exclusions": {
            "output_hashes.json": "SHA256 pinned in audit_summary",
            "audit_summary.json": "Published last PASS marker",
        },
    })
    write_exclusive(OUTPUT / "output_hashes.json", output_hashes)

    expected = set(content) | set(extras) | {"output_hashes.json"}
    actual = {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob("*") if p.is_file()}
    gate("output_inventory_complete", actual == expected)
    gate("output_hashes_exact",
         json.loads(output_hashes)["hashes"]
         == {n: sha_file(OUTPUT / n) for n in hashes})
    gate("predecessor_evidence_preserved",
         verify_5ar(head) == smoke_sha and verify_posthoc(head)[0] == posthoc_sha)
    gate("source_HEAD_and_worktree_unchanged",
         git("rev-parse", "HEAD") == head
         and git("status", "--porcelain", "--untracked-files=all") == "")
    gate("audit_summary_published_last", not (OUTPUT / "audit_summary.json").exists())
    require(all(gates.values()), "All P2-2D-5B structural gates")

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
        "PPO_training_runs": 2,
        "Masked_Point_training_runs": 2,
        "UA_training_runs": 0,
        "development_budget_per_seed": DEVELOPMENT_BUDGET,
        "development_evaluation_episode_count": 1200,
        "training_shield_interventions": {
            str(s): json.loads(completion_paths[s].read_bytes())
            ["mask_training_audit"]["total_shield_interventions"]
            for s in DEVELOPMENT_SEEDS
        },
        "evaluation_shield_interventions": {
            str(s): seed_summaries[s]["total_shield_interventions"]
            for s in DEVELOPMENT_SEEDS
        },
        "final_seed_training_runs": 0,
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
    write_exclusive(OUTPUT / "audit_summary.json", encode(audit))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
