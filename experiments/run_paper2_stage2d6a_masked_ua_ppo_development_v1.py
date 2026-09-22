
"""P2-2D-6A-v1: full Masked UA-PPO DEVELOPMENT characterization.

Scientific purpose
------------------
P2-2D-5B established a complete masked Point-PPO DEVELOPMENT baseline:
two frozen DEVELOPMENT seeds, 491,520 steps per seed, 600 deterministic
DEVELOPMENT episodes per final checkpoint, zero training/evaluation shield
interventions, but deterministic voluntary CLEAN = 0 and exact equality with
the frozen shield-only baseline.

P2-2D-5C/5D then showed that Point observations contain economic-action
ambiguity and that uncertainty width carries additional but limited
discriminative information.

P2-2D-6A now performs the fair policy-level comparison.  It changes exactly
one policy input relative to P2-2D-5B:

    Point: [q50, sin_DOY, cos_DOY]
    UA:    [q50, width, sin_DOY, cos_DOY]

Everything else remains frozen:
CONFIG_A, DEVELOPMENT seeds 510001/510002, 491,520 steps/seed, training and
DEVELOPMENT trajectory partitions, BLOCK10-v1 perception, q50-history safety
mask, frozen q50 shield fail-safe, reward, physics, network architecture,
optimizer defaults and deterministic evaluation.

Crucially, width is POLICY INFORMATION ONLY.  The safety mask/shield remains
q50-history-only and is mechanically checked to have no width dependency.

Every UA DEVELOPMENT episode is paired with:
  1) the same-seed frozen Masked Point episode from P2-2D-5B;
  2) the frozen shield-only cost carried by the P2-2D-5B paired comparison.

Performance outcomes are descriptive only and never structural PASS gates.
No hyperparameter reselection, retry/reseed, FINAL-seed training, formal
held-out access, RANDOM_TEST or SEALED_DATES access is allowed.

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
import statistics
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SELF = "experiments/run_paper2_stage2d6a_masked_ua_ppo_development_v1.py"
RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1_1.py"
MASK_RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1.py"
STAGE5B_SOURCE = (
    "experiments/run_paper2_stage2d5b_masked_point_ppo_development_v1.py"
)
STAGE5D_SOURCE = (
    "experiments/run_paper2_stage2d5d_ua_width_observation_disambiguation_audit_v1.py"
)
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
PERCEPTION_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
MASKED_REQUIREMENTS = "requirements-paper2-masked-v1.txt"
BASE_REQUIREMENTS = "requirements.txt"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
POINT5B = BASE / "p2_2d_5b_masked_point_ppo_development_v1/formal"

STAGE_DIRECTORY = BASE / "p2_2d_6a_masked_ua_ppo_development_v1"
OUTPUT = STAGE_DIRECTORY / "formal"

CHECKPOINT = "a5b7830f760685a97995ea6dc24845fe17c89cd0"
POINT5B_HEAD = "c0605bae1f4e0eeb32341c5f078bb94dfe321e96"

DEVELOPMENT_SEEDS = (510001, 510002)
DEVELOPMENT_BUDGET = 491520
DEVELOPMENT_EPISODES = 600
ATOL = 1e-12

PINNED_BLOBS = {
    RUNNER: "9b1777ce4262efb95c677432ea52997352d76750",
    MASK_RUNNER: "be86a5e8a8a162c0d6be720c5340914ccc1ed82e",
    STAGE5B_SOURCE: "66e55ee5c4e6bd8c3663ea9aa707d78f79c2a690",
    STAGE5D_SOURCE: "58a62eff427f6a266165440a14ab239ed9da5ad5",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    PERCEPTION_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
    MASKED_REQUIREMENTS: "8bb5ad1e5b6d84eb31b7a207aefde57cc0e3b88c",
    BASE_REQUIREMENTS: "430357153fbdc6b2b77d92d8040333d2de30135b",
}

EXPECTED_POINT5B_HASHES = {
    "protocol_manifest.json":
        "341948b218ba642610a1f9deed3f71ef7c8184ad88ca852f60ec749ec274a836",
    "development_summary.json":
        "9c39ed20b27317d408ec1ad9392376eb36ee1174c44790f8ce0e20866a29bd88",
    "source_artifact_hashes.json":
        "e9442f844f71bdba95569f7ac3a0701a67c86e106d3762972b24443b5a44a0e0",
    "masked_development_run_summary.csv":
        "d45e17b279ca9f505515490255a8d8b6fcc9be3f4e3e96733ff174f94c1fe7da",
    "masked_development_episode_metrics.csv":
        "40232df1ddc57efb6c0a9fffc580772d0620ed803e66bb28883a69b27e6582e9",
    "paired_development_comparison.csv":
        "d01e7c0c0c6fef953c56896e866b1d1f1f3b4a9d54a05c635cd72990c892ff1a",
    "runs/seed_510001/training_completion.json":
        "c22906de966c678ff8233b7303d23698cb25f5f7c9ad4e0a0d3396b977d67841",
    "runs/seed_510001/final_model.zip":
        "43cbe8bb83d2f41ea5e9fabbd93bdf9b5d7f5296f7e2bc640096d8cd1203e7aa",
    "runs/seed_510002/training_completion.json":
        "084b6ee81cc9e144061d1f7952a388afe5942557de0863080a1cf523cd0706d1",
    "runs/seed_510002/final_model.zip":
        "de7f63173eee2e4eb671c2dae4127bed712aebf092eec9f4f8e1c74ef183677c",
    "output_hashes.json":
        "92c6ef51d0a01a2ce15a7df6a425bfa17ebc7159185b97660987bf84a1d16fb5",
}

STAGE = "P2-2D-6A-v1"
STATUS = "MASKED_UA_PPO_DEVELOPMENT_CHARACTERIZATION_PASS"

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
            "Pinned source changed: " + name,
        )


def verify_point5b(head):
    audit_path = POINT5B / "audit_summary.json"
    require(audit_path.exists(), "P2-2D-5B audit exists")
    audit = json.loads(audit_path.read_bytes())

    require(
        audit["stage"] == "P2-2D-5B-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "MASKED_POINT_PPO_DEVELOPMENT_CHARACTERIZATION_PASS"
        and audit["HEAD"] == POINT5B_HEAD
        and audit["failed_gates"] == []
        and all(audit["gates"].values())
        and audit["runtime_algorithm"] == "MaskablePPO"
        and audit["selected_config"] == "CONFIG_A"
        and audit["PPO_training_runs"] == 2
        and audit["Masked_Point_training_runs"] == 2
        and audit["UA_training_runs"] == 0
        and audit["development_budget_per_seed"] == DEVELOPMENT_BUDGET
        and audit["development_evaluation_episode_count"] == 1200
        and audit["training_shield_interventions"]
        == {"510001": 0, "510002": 0}
        and audit["evaluation_shield_interventions"]
        == {"510001": 0, "510002": 0}
        and audit["final_seed_training_runs"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted frozen Masked Point DEVELOPMENT authority",
    )

    require(
        audit["output_sha256"] == EXPECTED_POINT5B_HASHES,
        "P2-2D-5B output hash registry exact",
    )
    for name, digest in EXPECTED_POINT5B_HASHES.items():
        require(
            sha_file(POINT5B / name) == digest,
            "P2-2D-5B output hash: " + name,
        )

    point_rows = read_csv(
        POINT5B / "masked_development_episode_metrics.csv"
    )
    pair_rows = read_csv(
        POINT5B / "paired_development_comparison.csv"
    )
    require(
        len(point_rows) == 1200
        and len(pair_rows) == 1200,
        "Exact frozen Point comparison populations",
    )

    point = {}
    for row in point_rows:
        key = (
            int(row["parent_rl_seed"]),
            row["year"],
            int(row["trajectory_id"]),
        )
        require(key not in point, "Unique Point episode key")
        point[key] = row

    reference = {}
    for row in pair_rows:
        key = (
            int(row["parent_rl_seed"]),
            row["year"],
            int(row["trajectory_id"]),
        )
        require(key not in reference, "Unique Point paired key")
        require(
            abs(float(row["J_masked"]) - float(point[key]["J_total"]))
            <= ATOL,
            "P2-2D-5B Point cost identity",
        )
        reference[key] = row

    require(
        len(point) == len(reference) == 1200,
        "Complete frozen Point/reference maps",
    )

    # Freeze the observed Point collapse as a comparison fact, not a new gate
    # for UA performance.
    require(
        all(int(row["N_voluntary_clean"]) == 0 for row in point.values()),
        "Frozen Point deterministic voluntary-CLEAN collapse",
    )

    verify_pinned_sources(head)
    return {
        "audit_summary_sha256": sha_file(audit_path),
        "point": point,
        "reference": reference,
        "read_only": True,
    }


def _method_ast(source, class_name, method_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return item
    raise RuntimeError(f"Missing {class_name}.{method_name}")


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

    training_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "train_masked_final_checkpoint"
    ]
    require(
        len(training_calls) == 1,
        "Exactly one UA DEVELOPMENT training call site",
    )
    call = training_calls[0]
    require(
        len(call.keywords) == 1
        and call.keywords[0].arg == "purpose"
        and isinstance(call.keywords[0].value, ast.Constant)
        and call.keywords[0].value.value == "development",
        "Frozen DEVELOPMENT budget only",
    )

    require(
        "build_masked_scheduled_perception_env" in attrs
        and "evaluate_masked_perception_deterministic" in attrs
        and "build_masked_perception_ppo_model" in attrs,
        "Masked UA runtime surface present",
    )
    require(
        "learn" not in attrs
        and "save" not in attrs
        and "load" not in attrs,
        "No direct learn/save/load bypass",
    )
    require(
        "performance_diagnostics_not_pass_gates" in source
        and "paired_comparison_descriptive_only" in source
        and "zero_training_shield_interventions" in source
        and "zero_evaluation_shield_interventions" in source,
        "Scientific outcomes separated from structural gates",
    )

    # Mechanical proof that the inherited action mask never reads width.
    mask_source = (ROOT / MASK_RUNNER).read_text(encoding="utf-8")
    method = _method_ast(
        mask_source,
        "Q50MaskedShieldedPerceptionEnv",
        "_compute_action_mask",
    )
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
        and "belief_lower" in tokens
        and "belief_upper" in tokens
        and "action_safe" in tokens,
        "Safety mask must be q50-belief-only and width-free",
    )

    return {
        "single_UA_development_training_call_site": True,
        "masked_runtime_only": True,
        "no_direct_learn_save_load": True,
        "UA_observation_runtime_present": True,
        "q50_only_width_free_mask_mechanically_verified": True,
        "zero_intervention_contract_present": True,
        "paired_comparison_descriptive_only": True,
        "performance_diagnostics_not_pass_gates": True,
    }


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


def evaluate_full(runner, model, assets, specs, seed):
    ledger = runner.AccessLedger()
    rows = runner.evaluate_masked_perception_deterministic(
        model,
        assets,
        specs,
        ledger,
        "UA",
    )
    p = runner.get_protocol()

    require(
        len(rows) == DEVELOPMENT_EPISODES
        and all(
            row["step_count"] == p.decision_reward_steps
            and row["natural_transition_count"] == p.natural_transitions
            and row["N_shield_interventions"] == 0
            and row["N_forced_clean"] + row["N_free_choice"]
            == p.natural_transitions
            and row["N_free_choice"]
            == row["N_voluntary_clean"] + row["N_free_wait"]
            and row["N_clean"]
            == row["N_forced_clean"]
            + row["N_voluntary_clean"]
            + row["N_terminal_clean"]
            and row["terminal_mask"] == [True, True]
            and row["all_observations_finite"]
            and row["all_rewards_finite"]
            and row["all_states_finite"]
            and row["reward_sign_regression_diff"] <= p.selection.atol
            for row in rows
        ),
        f"seed_{seed}: complete zero-intervention UA DEVELOPMENT evaluation",
    )
    require(
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0
        and all(row["observation_mode"] == "UA" for row in ledger.records),
        f"seed_{seed}: DEVELOPMENT UA access only",
    )
    return rows, summarize(rows), ledger


def build_pairs(seed, ua_rows, point, reference):
    rows = []
    seen = set()

    for ua in ua_rows:
        year = ua["year"]
        tid = int(ua["trajectory_id"])
        key = (seed, year, tid)
        require(key not in seen, "Unique UA paired key")
        seen.add(key)

        point_row = point[key]
        ref = reference[key]

        j_ua = float(ua["J_total"])
        j_point = float(point_row["J_total"])
        j_shield = float(ref["J_shield_only"])

        delta_point = j_ua - j_point
        delta_shield = j_ua - j_shield

        def relation(delta):
            if delta < -ATOL:
                return "UA_LOWER_COST"
            if delta > ATOL:
                return "UA_HIGHER_COST"
            return "COST_TIE_WITHIN_ATOL"

        rows.append(
            {
                "parent_rl_seed": seed,
                "year": year,
                "trajectory_id": tid,
                "J_UA": j_ua,
                "J_Point": j_point,
                "J_shield_only": j_shield,
                "delta_J_UA_minus_Point": delta_point,
                "delta_J_UA_minus_shield_only": delta_shield,
                "relation_UA_vs_Point": relation(delta_point),
                "relation_UA_vs_shield_only": relation(delta_shield),
                "N_clean_UA": int(ua["N_clean"]),
                "N_forced_clean_UA":
                    int(ua["N_forced_clean"]),
                "N_voluntary_clean_UA":
                    int(ua["N_voluntary_clean"]),
                "N_free_choice_UA":
                    int(ua["N_free_choice"]),
                "voluntary_clean_fraction_given_free_UA":
                    float(
                        ua[
                            "voluntary_clean_fraction_given_free"
                        ]
                    ),
                "N_shield_interventions_UA":
                    int(ua["N_shield_interventions"]),
                "N_clean_Point":
                    int(point_row["N_clean"]),
                "N_forced_clean_Point":
                    int(point_row["N_forced_clean"]),
                "N_voluntary_clean_Point":
                    int(point_row["N_voluntary_clean"]),
                "N_free_choice_Point":
                    int(point_row["N_free_choice"]),
            }
        )

    require(
        len(rows) == DEVELOPMENT_EPISODES,
        "Complete 600-episode UA paired population",
    )
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
        "UA_lower_cost_episode_count": int(lower),
        "UA_higher_cost_episode_count": int(higher),
        "tie_episode_count": int(tie),
        "UA_lower_cost_episode_fraction":
            float(lower / len(values)),
        "UA_higher_cost_episode_fraction":
            float(higher / len(values)),
        "tie_episode_fraction":
            float(tie / len(values)),
    }


def run_formal():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive P2-2D-6A directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean worktree required",
    )

    head = git("rev-parse", "HEAD")
    verify_pinned_sources(head)
    point5b = verify_point5b(head)

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(
        self_blob
        == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "P2-2D-6A source committed unchanged",
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
        and p.development_budget == p.selection.development_checkpoint
        and p.development_budget % config.n_steps == 0
        and config.name == runner.inherited_config_id() == "CONFIG_A",
        "Frozen CONFIG_A DEVELOPMENT seed/budget registry",
    )
    require(
        dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY")
        and dict(p.observations)["UA"]
        == ("q50", "width", "sin_DOY", "cos_DOY")
        and dict(p.observation_dimensions)["Point"] == 3
        and dict(p.observation_dimensions)["UA"] == 4,
        "Frozen Point/UA observation contract",
    )
    require(
        api_contract["frozen_use_sde"] is False
        and api_contract["frozen_sde_sample_freq"] == -1
        and api_contract["MaskablePPO_init_rejects_use_sde"] is True
        and api_contract["MaskablePPO_init_rejects_sde_sample_freq"] is True,
        "Frozen MaskablePPO API compatibility rule",
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
        == p.evaluation.episodes_per_model
        == DEVELOPMENT_EPISODES,
        "Full frozen 600-episode DEVELOPMENT population",
    )

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        return execute_formal(
            runner=runner,
            frozen=frozen,
            p=p,
            config=config,
            assets=assets,
            specs=specs,
            head=head,
            self_blob=self_blob,
            static=static,
            point5b=point5b,
            source_sha256=source_sha256,
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
                            "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/UA-WIDTH/MASK/SHIELD/REWARD/PERCEPTION/PPO TUNING",
                    }
                ),
            )
        raise


def execute_formal(
    *,
    runner,
    frozen,
    p,
    config,
    assets,
    specs,
    head,
    self_blob,
    static,
    point5b,
    source_sha256,
    base_versions,
    masked_versions,
    api_contract,
):
    gates = {}

    def gate(name, condition):
        require(name not in gates, "Duplicate gate: " + name)
        gates[name] = bool(condition)
        require(condition, name)

    gate("P2_2D_5B_Point_reference_exact", bool(point5b))
    gate("static_UA_development_contract", all(static.values()))
    gate(
        "frozen_CONFIG_A_seed_budget_contract",
        config.name == "CONFIG_A"
        and tuple(p.seeds.development) == DEVELOPMENT_SEEDS
        and p.development_budget == DEVELOPMENT_BUDGET
        and config.learning_rate == 3e-4
        and config.n_steps == 2048,
    )
    gate(
        "Point_UA_only_policy_input_difference_contract",
        dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY")
        and dict(p.observations)["UA"]
        == ("q50", "width", "sin_DOY", "cos_DOY")
        and all(static.values()),
    )

    run_rows = []
    ua_episode_rows = []
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
        env = runner.build_masked_scheduled_perception_env(
            assets=assets,
            rl_seed=seed,
            ledger=ledger,
            config_id=config.name,
            observation_mode="UA",
        )
        loaded = None

        try:
            gate(
                f"{label}:fresh_UA_training_environment",
                env.observation_mode == "UA"
                and env.observation_space.shape == (4,)
                and env.action_space.n == 2
                and env.action_space.start == 0
                and env.schedule.parent_seed == seed
                and not env.started
                and not env.failed
                and not env.assignments,
            )

            bundle = runner.build_masked_perception_ppo_model(
                env,
                config.name,
                seed,
            )
            gate(
                f"{label}:frozen_MaskablePPO_contract",
                bundle.model.num_timesteps == 0
                and runner.verify_masked_model_contract(
                    bundle.model,
                    config.name,
                    seed,
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
                runner,
                env,
                seed,
                config.name,
            )

            training_records = [
                row
                for row in ledger.records
                if row["role"] == "TRAINING"
            ]
            gate(
                f"{label}:TRAINING_roots_UA_only",
                len(training_records) == len(env.assignments)
                and all(
                    row["environment_root"] in training_roots
                    and row["observation_mode"] == "UA"
                    for row in training_records
                ),
            )

            completion = {
                "stage": STAGE,
                "parent_rl_seed": seed,
                "observation_mode": "UA",
                "policy_observation":
                    ["q50", "width", "sin_DOY", "cos_DOY"],
                "safety_mask_inputs":
                    ["q50-history belief lower/upper only"],
                "training": training,
                "model_sha256": sha_file(model_path),
                "model_manifest":
                    runner.masked_model_manifest(bundle, training),
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
                model_path,
                bundle,
            )
            gate(
                f"{label}:save_reload_exact",
                loaded is not bundle.model
                and runner.parameters_finite(loaded)
                and sha_file(model_path)
                == training["model_sha256"],
            )
        finally:
            env.close()

        require(loaded is not None, f"{label}: checkpoint reloaded")

        eval_rows, eval_summary, eval_ledger = evaluate_full(
            runner,
            loaded,
            assets,
            specs,
            seed,
        )
        gate(
            f"{label}:full_600_UA_DEVELOPMENT_evaluation",
            eval_summary["episode_count"] == DEVELOPMENT_EPISODES,
        )
        gate(
            f"{label}:zero_evaluation_shield_interventions",
            eval_summary["total_shield_interventions"] == 0,
        )

        pairs = build_pairs(
            seed,
            eval_rows,
            point5b["point"],
            point5b["reference"],
        )
        point_summary = paired_summary(
            pairs,
            "delta_J_UA_minus_Point",
        )
        shield_summary = paired_summary(
            pairs,
            "delta_J_UA_minus_shield_only",
        )

        run_rows.append(
            {
                "parent_rl_seed": seed,
                "config_id": config.name,
                "observation_mode": "UA",
                "development_timesteps": DEVELOPMENT_BUDGET,
                "mean_J_total": eval_summary["mean_J_total"],
                "std_J_total": eval_summary["std_J_total"],
                "median_J_total":
                    eval_summary["median_J_total"],
                "P95_J_total": eval_summary["P95_J_total"],
                "CVaR95_J_total":
                    eval_summary["CVaR95_J_total"],
                "mean_N_clean":
                    eval_summary["mean_N_clean"],
                "mean_N_forced_clean":
                    eval_summary["mean_N_forced_clean"],
                "mean_N_voluntary_clean":
                    eval_summary["mean_N_voluntary_clean"],
                "mean_N_free_choice":
                    eval_summary["mean_N_free_choice"],
                "mean_voluntary_clean_fraction_given_free":
                    eval_summary[
                        "mean_voluntary_clean_fraction_given_free"
                    ],
                "episodes_with_any_voluntary_clean":
                    eval_summary[
                        "episodes_with_any_voluntary_clean"
                    ],
                "total_shield_interventions":
                    eval_summary["total_shield_interventions"],
                "mean_delta_J_UA_minus_Point":
                    point_summary["mean_delta"],
                "UA_lower_cost_vs_Point_fraction":
                    point_summary[
                        "UA_lower_cost_episode_fraction"
                    ],
                "UA_higher_cost_vs_Point_fraction":
                    point_summary[
                        "UA_higher_cost_episode_fraction"
                    ],
                "mean_delta_J_UA_minus_shield_only":
                    shield_summary["mean_delta"],
                "UA_lower_cost_vs_shield_fraction":
                    shield_summary[
                        "UA_lower_cost_episode_fraction"
                    ],
            }
        )

        ua_episode_rows.extend(
            [
                {"parent_rl_seed": seed, **row}
                for row in eval_rows
            ]
        )
        paired_rows.extend(pairs)
        model_paths[seed] = model_path
        completion_paths[seed] = completion_path
        access_ledgers.extend((ledger, eval_ledger))

    gate(
        "all_runtime_access_TRAINING_or_DEVELOPMENT_UA_only",
        all(
            ledger.heldout_ppo_trajectory_access_count == 0
            and ledger.formal_perception_seed_access_count == 0
            and ledger.denied_accesses == 0
            and all(
                row["observation_mode"] == "UA"
                for row in ledger.records
            )
            for ledger in access_ledgers
        ),
    )
    gate(
        "paired_comparison_population_complete",
        len(ua_episode_rows) == 1200
        and len(paired_rows) == 1200
        and all(
            row["N_shield_interventions_UA"] == 0
            for row in paired_rows
        ),
    )

    seed_summaries = {
        seed: summarize(
            [
                row
                for row in ua_episode_rows
                if int(row["parent_rl_seed"]) == seed
            ]
        )
        for seed in DEVELOPMENT_SEEDS
    }
    point_pairs = {
        seed: paired_summary(
            [
                row
                for row in paired_rows
                if int(row["parent_rl_seed"]) == seed
            ],
            "delta_J_UA_minus_Point",
        )
        for seed in DEVELOPMENT_SEEDS
    }
    shield_pairs = {
        seed: paired_summary(
            [
                row
                for row in paired_rows
                if int(row["parent_rl_seed"]) == seed
            ],
            "delta_J_UA_minus_shield_only",
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
        "each_seed_mean_cost_below_Point":
            all(
                point_pairs[seed]["mean_delta"] < -ATOL
                for seed in DEVELOPMENT_SEEDS
            ),
        "each_seed_mean_cost_below_shield_only":
            all(
                shield_pairs[seed]["mean_delta"] < -ATOL
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
        "Point_reference": "P2-2D-5B-v1",
        "shield_only_reference":
            "frozen shield-only costs carried by P2-2D-5B paired comparison",
        "paired_episode_unit":
            "(parent_rl_seed, year, trajectory_id)",
        "UA_vs_Point_by_seed": {
            str(seed): point_pairs[seed]
            for seed in DEVELOPMENT_SEEDS
        },
        "UA_vs_shield_only_by_seed": {
            str(seed): shield_pairs[seed]
            for seed in DEVELOPMENT_SEEDS
        },
        "performance_gate": False,
    }

    development_summary = {
        "stage": STAGE,
        "runtime_algorithm": "MaskablePPO",
        "observation_mode": "UA",
        "policy_observation":
            ["q50", "width", "sin_DOY", "cos_DOY"],
        "Point_reference_observation":
            ["q50", "sin_DOY", "cos_DOY"],
        "only_policy_input_added_relative_to_Point": "width",
        "safety_mask_width_used": False,
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
            "DEVELOPMENT characterization only; no FINAL-seed, formal-held-out, RANDOM_TEST or SEALED_DATES claim",
    }

    protocol_manifest = {
        "stage": STAGE,
        "mode": "formal",
        "protocol": frozen.protocol_payload(p),
        "selected_config": config.name,
        "runtime_algorithm": "MaskablePPO",
        "observation_mode": "UA",
        "policy_observation":
            ["q50", "width", "sin_DOY", "cos_DOY"],
        "safety_mask_inputs":
            ["q50-history belief lower/upper only"],
        "safety_mask_width_used": False,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "PPO_training_runs": 2,
        "Masked_UA_training_runs": 2,
        "Masked_Point_training_runs": 0,
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
        "source_sha256": source_sha256,
        "pinned_predecessor_blobs": PINNED_BLOBS,
        "base_dependencies": base_versions,
        "masked_dependencies": masked_versions,
        "MaskablePPO_API_contract": api_contract,
        "core_assets_provenance": assets.provenance,
        "P2_2D_5B_Point_reference": {
            "audit_summary_sha256":
                point5b["audit_summary_sha256"],
            "read_only": True,
        },
    }

    content = {
        "protocol_manifest.json":
            encode(protocol_manifest),
        "development_summary.json":
            encode(development_summary),
        "source_artifact_hashes.json":
            encode(provenance),
        "masked_ua_development_run_summary.csv":
            csv_bytes(run_rows),
        "masked_ua_development_episode_metrics.csv":
            csv_bytes(ua_episode_rows),
        "paired_ua_point_shield_comparison.csv":
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
                "P2_2D_5B_audit_summary_sha256":
                    point5b["audit_summary_sha256"],
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

    point5b_after = verify_point5b(head)
    gate(
        "Point_reference_evidence_preserved",
        point5b_after["audit_summary_sha256"]
        == point5b["audit_summary_sha256"],
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

    require(all(gates.values()), "All P2-2D-6A structural gates")

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
        "observation_mode": "UA",
        "policy_observation":
            ["q50", "width", "sin_DOY", "cos_DOY"],
        "safety_mask_width_used": False,
        "PPO_training_runs": 2,
        "Masked_UA_training_runs": 2,
        "Masked_Point_training_runs": 0,
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
