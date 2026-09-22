
"""P2-2D-5C-v1: frozen Masked Point policy probability + observation ambiguity audit.

Purpose
-------
P2-2D-5B completed two full DEVELOPMENT MaskablePPO runs and reproduced the
shield-only baseline exactly under deterministic evaluation:
    voluntary CLEAN = 0 for both seeds,
    1200/1200 episode costs tied with the frozen shield-only baseline.

This stage does NOT train or tune anything.  It loads the two frozen 5B final
checkpoints read-only and replays the exact 600 DEVELOPMENT episodes per seed.
It diagnoses:

1) Free-choice policy confidence:
   P(CLEAN | Point observation, mask=[WAIT,CLEAN]) over every free-choice
   nonterminal decision, including quantiles and deterministic action margin.

2) Economic-label response on the 163 frozen P2-2D-4AR safe-action anchors:
   CLEAN_BENEFICIAL / WAIT_BENEFICIAL / COST_TIE_WITHIN_ATOL are linked by the
   exact (year, trajectory_id, anchor_day) key.  No new counterfactual is run.

3) Point-observation ambiguity:
   exact duplicate Point observations with conflicting economic labels, plus a
   threshold-free nearest-neighbor label-mixing diagnostic in the actual
   3-dimensional Point observation coordinates [q50, sin_DOY, cos_DOY].

Scientific diagnostics are descriptive only.  They are never structural PASS
gates and cannot trigger retry, reseed, threshold adjustment, reward shaping,
mask/shield changes or extra training.

No FINAL seed, FORMAL HELD-OUT, RANDOM_TEST, SEALED_DATES or UA runtime access.
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

SELF = (
    "experiments/"
    "run_paper2_stage2d5c_masked_point_policy_probability_aliasing_audit_v1.py"
)
RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1_1.py"
STAGE5B_SOURCE = "experiments/run_paper2_stage2d5b_masked_point_ppo_development_v1.py"
STAGE4B_SOURCE = (
    "experiments/"
    "run_paper2_stage2d4b_safe_action_distinction_decomposition_audit_v1.py"
)
STAGE4AR_SOURCE = (
    "experiments/"
    "run_paper2_stage2d4ar_shield_delegation_action_identifiability_recovery_v1.py"
)
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
STAGE5B = BASE / "p2_2d_5b_masked_point_ppo_development_v1/formal"
STAGE4B = BASE / "p2_2d_4b_safe_action_distinction_decomposition_audit_v1/formal"
STAGE4AR = (
    BASE
    / "p2_2d_4ar_shield_delegation_action_identifiability_recovery_v1/formal"
)

STAGE_DIRECTORY = BASE / "p2_2d_5c_masked_point_policy_probability_aliasing_audit_v1"
OUTPUT = STAGE_DIRECTORY / "formal"

CHECKPOINT = "c0605bae1f4e0eeb32341c5f078bb94dfe321e96"
DEVELOPMENT_SEEDS = (510001, 510002)
N_DEV = 600
N_ANCHORS = 163
ATOL = 1e-12

PINNED_BLOBS = {
    RUNNER: "9b1777ce4262efb95c677432ea52997352d76750",
    STAGE5B_SOURCE: "66e55ee5c4e6bd8c3663ea9aa707d78f79c2a690",
    STAGE4B_SOURCE: "e5b4006c9ddb1866e78df2335ccc855de2253b98",
    STAGE4AR_SOURCE: "a778fa61aa7559353ed7e655d04298c4c8724f86",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
}

EXPECTED_5B_HASHES = {
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

EXPECTED_4AR_SAFE_ACTION_SHA256 = (
    "5a147ca25d979088ce1ad4bc3af8d4bde0f87cf4cd0f543164452673dfc5c8f8"
)

CLASSIFICATIONS = (
    "CLEAN_BENEFICIAL",
    "WAIT_BENEFICIAL",
    "COST_TIE_WITHIN_ATOL",
)

STAGE = "P2-2D-5C-v1"
STATUS = "MASKED_POINT_POLICY_PROBABILITY_ALIASING_AUDIT_COMPLETE"


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


def csv_bytes(rows, fields=None):
    if fields is None:
        require(rows, "Nonempty CSV rows")
        fields = list(rows[0])
    require(
        all(set(row) == set(fields) for row in rows),
        "Consistent CSV schema",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fields),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_exclusive(path, data):
    path = Path(path)
    require(
        path.parent.resolve() == OUTPUT.resolve(),
        "Top-level writes restricted to P2-2D-5C output",
    )
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, "Exact readback: " + path.name)


def verify_sources(head):
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned source changed: " + name,
        )


def verify_5b(head):
    audit_path = STAGE5B / "audit_summary.json"
    require(audit_path.exists(), "P2-2D-5B audit exists")
    audit = json.loads(audit_path.read_bytes())
    require(
        audit["stage"] == "P2-2D-5B-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "MASKED_POINT_PPO_DEVELOPMENT_CHARACTERIZATION_PASS"
        and audit["HEAD"] == CHECKPOINT
        and audit["failed_gates"] == []
        and all(audit["gates"].values())
        and audit["runtime_algorithm"] == "MaskablePPO"
        and audit["PPO_training_runs"] == 2
        and audit["Masked_Point_training_runs"] == 2
        and audit["UA_training_runs"] == 0
        and audit["development_budget_per_seed"] == 491520
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
        "Accepted P2-2D-5B authority",
    )
    require(
        audit["performance_diagnostics_not_pass_gates"][
            "any_deterministic_voluntary_clean_across_both_seeds"
        ]
        is False
        and audit["performance_diagnostics_not_pass_gates"][
            "two_seed_mean_of_seed_mean_voluntary_clean"
        ]
        == 0.0,
        "Frozen deterministic voluntary-CLEAN collapse",
    )
    require(
        audit["output_sha256"] == EXPECTED_5B_HASHES,
        "P2-2D-5B output hash registry exact",
    )
    for name, digest in EXPECTED_5B_HASHES.items():
        require(
            sha_file(STAGE5B / name) == digest,
            "P2-2D-5B output hash: " + name,
        )
    verify_sources(head)
    return {
        "audit_summary_sha256": sha_file(audit_path),
        "output_sha256": EXPECTED_5B_HASHES,
        "read_only": True,
    }


def verify_4b_and_4ar(head):
    audit_path = STAGE4B / "audit_summary.json"
    safe_path = STAGE4AR / "safe_action_counterfactuals.csv"
    require(
        audit_path.exists() and safe_path.exists(),
        "P2-2D-4B/4AR evidence exists",
    )
    audit = json.loads(audit_path.read_bytes())
    require(
        audit["stage"] == "P2-2D-4B-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "SAFE_ACTION_DISTINCTION_DECOMPOSITION_AUDIT_COMPLETE"
        and audit["counterfactual_anchor_count"] == N_ANCHORS
        and audit["clean_beneficial_count"] == 63
        and audit["wait_beneficial_count"] == 90
        and audit["tie_count"] == 10
        and audit["PPO_training_runs"] == 0
        and audit["PPO_model_loads"] == 0
        and audit["environment_runtime_calls"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0
        and audit["failed_gates"] == [],
        "Accepted P2-2D-4B authority",
    )
    for name, digest in audit["output_sha256"].items():
        require(
            sha_file(STAGE4B / name) == digest,
            "P2-2D-4B output hash: " + name,
        )
    require(
        sha_file(safe_path) == EXPECTED_4AR_SAFE_ACTION_SHA256,
        "Exact frozen P2-2D-4AR safe-action counterfactuals",
    )
    rows = read_csv(safe_path)
    require(len(rows) == N_ANCHORS, "Exact 163 safe-action anchors")
    require(
        sum(r["continuation_classification"] == "CLEAN_BENEFICIAL" for r in rows)
        == 63
        and sum(r["continuation_classification"] == "WAIT_BENEFICIAL" for r in rows)
        == 90
        and sum(
            r["continuation_classification"] == "COST_TIE_WITHIN_ATOL"
            for r in rows
        )
        == 10,
        "Frozen anchor economic labels",
    )
    require(
        all(r["continuation_classification"] in CLASSIFICATIONS for r in rows),
        "Known anchor classifications",
    )
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    return {
        "P2_2D_4B_audit_summary_sha256": sha_file(audit_path),
        "P2_2D_4AR_safe_action_sha256": sha_file(safe_path),
        "rows": rows,
        "read_only": True,
    }


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    require(
        "learn" not in attrs
        and "save" not in attrs
        and "train_masked_final_checkpoint" not in attrs
        and "build_masked_scheduled_perception_env" not in attrs,
        "No training/save/training-environment runtime",
    )
    load_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
    ]
    require(len(load_calls) == 1, "Exactly one model-load call site")
    require(
        "build_masked_perception_eval_env" in attrs,
        "Evaluation environment only",
    )
    require(
        "get_distribution" in attrs
        and "obs_to_tensor" in attrs
        and "probs" in attrs,
        "Policy-probability extraction present",
    )
    require(
        "performance_diagnostics_not_pass_gates" in source
        and "exact_duplicate_conflicting_label_groups" in source
        and "nearest_neighbor_label_mixing" in source
        and "anchor_label_probability_separation" in source,
        "Descriptive probability/ambiguity diagnostics present",
    )
    return {
        "zero_training_call": True,
        "exactly_one_model_load_call_site": True,
        "evaluation_runtime_only": True,
        "policy_probability_extraction_present": True,
        "full_DEVELOPMENT_replay_contract_present": True,
        "frozen_anchor_linkage_present": True,
        "observation_ambiguity_diagnostics_present": True,
        "performance_diagnostics_not_pass_gates": True,
    }


def parse_5b_episode_reference():
    rows = read_csv(STAGE5B / "masked_development_episode_metrics.csv")
    require(len(rows) == 1200, "Exact 1200 frozen P2-2D-5B eval rows")
    out = {}
    for row in rows:
        key = (
            int(row["parent_rl_seed"]),
            row["year"],
            int(row["trajectory_id"]),
        )
        require(key not in out, "Unique P2-2D-5B episode key")
        out[key] = row
    return out


def parse_anchor_index(rows):
    index = {}
    for row in rows:
        key = (
            row["year"],
            int(row["trajectory_id"]),
            int(row["anchor_day"]),
        )
        index.setdefault(key, []).append(row)
    require(
        sum(len(items) for items in index.values()) == N_ANCHORS,
        "Exact anchor index cardinality",
    )
    return index


def policy_probabilities(model, obs, mask):
    import torch

    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(
            obs_tensor,
            action_masks=np.asarray(mask, dtype=np.bool_).reshape(1, -1),
        )
        require(
            distribution.distribution is not None,
            "Categorical distribution exists",
        )
        probs = (
            distribution.distribution.probs.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    require(
        probs.shape == (2,)
        and np.isfinite(probs).all()
        and probs[0] >= 0.0
        and probs[1] >= 0.0
        and abs(float(probs.sum()) - 1.0) <= 1e-6,
        "Valid two-action probability simplex",
    )
    return float(probs[0]), float(probs[1])


def probability_summary(values):
    require(values, "Nonempty probability population")
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(
        arr,
        [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        method="linear",
    )
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "p01": float(qs[0]),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
        "p95": float(qs[6]),
        "p99": float(qs[7]),
        "max": float(np.max(arr)),
        "count_gt_0p5": int(np.sum(arr > 0.5)),
        "count_eq_0p5": int(np.sum(arr == 0.5)),
        "count_lt_0p5": int(np.sum(arr < 0.5)),
        "fraction_gt_0p5": float(np.mean(arr > 0.5)),
    }


def pairwise_auc(clean_values, wait_values):
    require(clean_values and wait_values, "Both economic classes required")
    wins = 0.0
    total = 0
    for clean in clean_values:
        for wait in wait_values:
            total += 1
            if clean > wait:
                wins += 1.0
            elif clean == wait:
                wins += 0.5
    return float(wins / total)


def replay_seed(runner, model, assets, seed, specs, ref, anchor_index):
    p = runner.get_protocol()
    ledger = runner.AccessLedger()
    free_probs = []
    episode_rows = []
    anchor_rows = []

    for year, tid in specs:
        env = runner.build_masked_perception_eval_env(
            assets,
            year,
            tid,
            ledger,
            "Point",
        )
        try:
            obs, info = env.reset()
            runner.audit_perception_observation(env.inner, obs, info)
            costs = []
            rewards = []
            episode_free_probs = []

            for day in range(p.decision_reward_steps):
                mask = env.action_masks()
                require(
                    mask.shape == (2,)
                    and mask.dtype == np.bool_
                    and bool(mask[1]),
                    "Valid q50 action mask",
                )
                p_wait, p_clean = policy_probabilities(model, obs, mask)

                action_array, _ = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=mask,
                )
                action = int(np.asarray(action_array).item())
                require(
                    action in (0, 1) and bool(mask[action]),
                    "Deterministic masked action valid",
                )

                if day < p.natural_transitions and bool(mask[0]):
                    free_probs.append(p_clean)
                    episode_free_probs.append(p_clean)

                    key = (year, int(tid), day)
                    if key in anchor_index:
                        for anchor in anchor_index[key]:
                            ref_q50 = np.float32(
                                float(anchor["q50_float32"])
                            )
                            obs_q50 = np.float32(obs[0])
                            require(
                                obs_q50 == ref_q50,
                                "Anchor q50_float32 exact replay",
                            )
                            anchor_rows.append(
                                {
                                    "parent_rl_seed": seed,
                                    "year": year,
                                    "trajectory_id": int(tid),
                                    "anchor_label": anchor["anchor_label"],
                                    "anchor_day": day,
                                    "continuation_classification":
                                        anchor[
                                            "continuation_classification"
                                        ],
                                    "remaining_horizon_gap_CLEANminusWAIT":
                                        float(
                                            anchor[
                                                "remaining_horizon_gap_CLEANminusWAIT"
                                            ]
                                        ),
                                    "q50_float32": float(obs_q50),
                                    "sin_DOY_float32":
                                        float(np.float32(obs[1])),
                                    "cos_DOY_float32":
                                        float(np.float32(obs[2])),
                                    "p_WAIT": p_wait,
                                    "p_CLEAN": p_clean,
                                    "probability_margin_CLEANminusWAIT":
                                        p_clean - p_wait,
                                    "deterministic_action": action,
                                    "mask_WAIT_valid": True,
                                    "mask_CLEAN_valid": True,
                                }
                            )

                obs, reward, terminated, truncated, info = env.step(action)
                snap = runner.audit_perception_observation(
                    env.inner,
                    obs,
                    info,
                    terminal=terminated,
                )
                decision = env.last_shield_decision
                require(
                    decision is not None
                    and decision["intervention"] is False
                    and decision["executed_action"] == action,
                    "Zero post-hoc intervention replay",
                )
                rewards.append(float(reward))
                costs.append(
                    float(snap["last_reward_components"]["total_cost"])
                )
                require(
                    not truncated
                    and terminated
                    == (day == p.decision_reward_steps - 1),
                    "Exact frozen horizon",
                )

            record = env._mask_audit_snapshot()["episode_records"][-1]
            total_cost = float(sum(costs))
            key = (seed, year, int(tid))
            frozen = ref[key]
            require(
                abs(total_cost - float(frozen["J_total"])) <= ATOL
                and record["forced_clean_count"]
                == int(frozen["N_forced_clean"])
                and record["voluntary_clean_count"]
                == int(frozen["N_voluntary_clean"])
                and record["free_choice_decisions"]
                == int(frozen["N_free_choice"])
                and record["interventions"]
                == int(frozen["N_shield_interventions"]),
                "Exact P2-2D-5B episode replay",
            )
            require(
                record["voluntary_clean_count"] == 0,
                "Frozen deterministic no-voluntary-CLEAN replay",
            )
            require(
                len(episode_free_probs) == record["free_choice_decisions"],
                "Free-choice probability accounting",
            )

            ps = probability_summary(episode_free_probs)
            episode_rows.append(
                {
                    "parent_rl_seed": seed,
                    "year": year,
                    "trajectory_id": int(tid),
                    "J_total": total_cost,
                    "N_forced_clean": record["forced_clean_count"],
                    "N_voluntary_clean":
                        record["voluntary_clean_count"],
                    "N_free_choice": record["free_choice_decisions"],
                    "mean_p_CLEAN_free": ps["mean"],
                    "median_p_CLEAN_free": ps["p50"],
                    "p90_p_CLEAN_free": ps["p90"],
                    "p95_p_CLEAN_free": ps["p95"],
                    "max_p_CLEAN_free": ps["max"],
                    "count_p_CLEAN_gt_0p5":
                        ps["count_gt_0p5"],
                    "shield_interventions": record["interventions"],
                }
            )
        finally:
            env.close()

    require(
        len(episode_rows) == N_DEV,
        "Exact 600 DEVELOPMENT episodes per seed",
    )
    require(
        len(anchor_rows) == N_ANCHORS,
        "Every frozen safe-action anchor captured once per seed",
    )
    require(
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0
        and all(
            row["role"] == "DEVELOPMENT"
            and row["observation_mode"] == "Point"
            for row in ledger.records
        ),
        "DEVELOPMENT Point access only",
    )
    return free_probs, episode_rows, anchor_rows, ledger


def label_probability_summary(anchor_rows):
    rows = []
    for seed in DEVELOPMENT_SEEDS:
        seed_rows = [
            row for row in anchor_rows
            if int(row["parent_rl_seed"]) == seed
        ]
        for label in CLASSIFICATIONS:
            vals = [
                float(row["p_CLEAN"])
                for row in seed_rows
                if row["continuation_classification"] == label
            ]
            ps = probability_summary(vals)
            rows.append(
                {
                    "parent_rl_seed": seed,
                    "continuation_classification": label,
                    **ps,
                }
            )
    return rows


def separation_summary(anchor_rows):
    out = {}
    for seed in DEVELOPMENT_SEEDS:
        rows = [
            row for row in anchor_rows
            if int(row["parent_rl_seed"]) == seed
        ]
        clean = [
            float(row["p_CLEAN"]) for row in rows
            if row["continuation_classification"] == "CLEAN_BENEFICIAL"
        ]
        wait = [
            float(row["p_CLEAN"]) for row in rows
            if row["continuation_classification"] == "WAIT_BENEFICIAL"
        ]
        out[str(seed)] = {
            "clean_beneficial_count": len(clean),
            "wait_beneficial_count": len(wait),
            "mean_p_CLEAN_clean_beneficial":
                float(statistics.fmean(clean)),
            "mean_p_CLEAN_wait_beneficial":
                float(statistics.fmean(wait)),
            "mean_difference_clean_minus_wait":
                float(statistics.fmean(clean) - statistics.fmean(wait)),
            "median_p_CLEAN_clean_beneficial":
                float(statistics.median(clean)),
            "median_p_CLEAN_wait_beneficial":
                float(statistics.median(wait)),
            "pairwise_rank_AUC_clean_higher_than_wait":
                pairwise_auc(clean, wait),
        }
    return out


def exact_duplicate_diagnostic(anchor_rows_seed1):
    groups = {}
    for row in anchor_rows_seed1:
        key = (
            float(np.float32(row["q50_float32"])),
            float(np.float32(row["sin_DOY_float32"])),
            float(np.float32(row["cos_DOY_float32"])),
        )
        groups.setdefault(key, []).append(row)

    duplicate_rows = []
    conflicting_groups = 0
    duplicate_groups = 0
    for key, items in groups.items():
        if len(items) <= 1:
            continue
        duplicate_groups += 1
        labels = sorted(
            {item["continuation_classification"] for item in items}
        )
        conflicting = len(labels) > 1
        conflicting_groups += int(conflicting)
        duplicate_rows.append(
            {
                "q50_float32": key[0],
                "sin_DOY_float32": key[1],
                "cos_DOY_float32": key[2],
                "group_size": len(items),
                "classifications": "|".join(labels),
                "conflicting_labels": conflicting,
                "members": "|".join(
                    f'{item["year"]}:{item["trajectory_id"]}:{item["anchor_day"]}'
                    for item in items
                ),
            }
        )
    return {
        "anchor_count": len(anchor_rows_seed1),
        "unique_exact_Point_observation_count": len(groups),
        "exact_duplicate_group_count": duplicate_groups,
        "exact_duplicate_conflicting_label_groups": conflicting_groups,
        "direct_exact_observation_aliasing_proven":
            conflicting_groups > 0,
        "duplicate_rows": duplicate_rows,
    }


def nearest_neighbor_diagnostic(anchor_rows_seed1):
    rows = [
        row for row in anchor_rows_seed1
        if row["continuation_classification"]
        in ("CLEAN_BENEFICIAL", "WAIT_BENEFICIAL")
    ]
    require(len(rows) == 153, "153 non-tie economic anchors")
    x = np.asarray(
        [
            [
                row["q50_float32"],
                row["sin_DOY_float32"],
                row["cos_DOY_float32"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    labels = [row["continuation_classification"] for row in rows]
    out = []

    for i, row in enumerate(rows):
        distances = np.linalg.norm(x - x[i], axis=1)
        distances[i] = np.inf

        any_idx = int(np.argmin(distances))
        same_candidates = [
            j for j in range(len(rows))
            if j != i and labels[j] == labels[i]
        ]
        opp_candidates = [
            j for j in range(len(rows))
            if labels[j] != labels[i]
        ]
        require(
            same_candidates and opp_candidates,
            "Both same/opposite label neighbors exist",
        )
        same_idx = min(same_candidates, key=lambda j: distances[j])
        opp_idx = min(opp_candidates, key=lambda j: distances[j])

        out.append(
            {
                "year": row["year"],
                "trajectory_id": row["trajectory_id"],
                "anchor_day": row["anchor_day"],
                "classification": labels[i],
                "nearest_any_distance": float(distances[any_idx]),
                "nearest_any_classification": labels[any_idx],
                "nearest_any_is_opposite_label":
                    labels[any_idx] != labels[i],
                "nearest_same_label_distance":
                    float(distances[same_idx]),
                "nearest_opposite_label_distance":
                    float(distances[opp_idx]),
                "opposite_closer_than_same":
                    bool(distances[opp_idx] < distances[same_idx]),
            }
        )

    opposite_any = sum(
        row["nearest_any_is_opposite_label"] for row in out
    )
    opposite_closer = sum(
        row["opposite_closer_than_same"] for row in out
    )
    return out, {
        "population": len(out),
        "distance_space":
            "raw frozen Point coordinates [q50, sin_DOY, cos_DOY]",
        "threshold_used": None,
        "nearest_neighbor_label_mixing": {
            "nearest_any_is_opposite_label_count": int(opposite_any),
            "nearest_any_is_opposite_label_fraction":
                float(opposite_any / len(out)),
            "opposite_closer_than_same_count": int(opposite_closer),
            "opposite_closer_than_same_fraction":
                float(opposite_closer / len(out)),
            "median_nearest_same_label_distance":
                float(
                    statistics.median(
                        row["nearest_same_label_distance"]
                        for row in out
                    )
                ),
            "median_nearest_opposite_label_distance":
                float(
                    statistics.median(
                        row["nearest_opposite_label_distance"]
                        for row in out
                    )
                ),
        },
        "interpretation":
            "Threshold-free local label-mixing diagnostic only; nearest-neighbor overlap is not by itself proof of formal POMDP non-identifiability.",
    }


def run_formal():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive P2-2D-5C directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean worktree required",
    )
    head = git("rev-parse", "HEAD")
    verify_sources(head)
    stage5b = verify_5b(head)
    stage4 = verify_4b_and_4ar(head)

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(
        self_blob
        == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "P2-2D-5C source committed unchanged",
    )
    static = static_contract()

    import paper2_masked_perception_ppo_runner_v1_1 as runner
    from sb3_contrib import MaskablePPO

    p = runner.get_protocol()
    require(
        tuple(p.seeds.development) == DEVELOPMENT_SEEDS
        and dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY")
        and p.evaluation.deterministic is True,
        "Frozen Point DEVELOPMENT evaluation contract",
    )

    assets = runner.load_assets()
    assets.ensure_perception()
    dev = runner.partition("DEVELOPMENT")
    specs = tuple(
        (year, tid)
        for year in dev.years
        for tid in dev.trajectory_ids
    )
    require(
        len(specs) == len(set(specs)) == N_DEV,
        "Exact 600 DEVELOPMENT episodes",
    )

    ref = parse_5b_episode_reference()
    anchor_index = parse_anchor_index(stage4["rows"])

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        gates = {}
        def gate(name, condition):
            require(name not in gates, "Duplicate gate: " + name)
            gates[name] = bool(condition)
            require(condition, name)

        gate("P2_2D_5B_predecessor_exact", bool(stage5b))
        gate("P2_2D_4B_4AR_anchor_evidence_exact", bool(stage4))
        gate("static_read_only_probability_contract", all(static.values()))

        all_episode_rows = []
        all_anchor_rows = []
        seed_probability = {}
        model_loads = 0

        for seed in DEVELOPMENT_SEEDS:
            model_path = STAGE5B / f"runs/seed_{seed}/final_model.zip"
            require(
                sha_file(model_path)
                == EXPECTED_5B_HASHES[f"runs/seed_{seed}/final_model.zip"],
                f"seed_{seed}: frozen checkpoint SHA256",
            )
            model = MaskablePPO.load(
                str(model_path),
                device=p.device,
            )
            model_loads += 1
            require(
                runner.verify_masked_model_contract(
                    model,
                    "CONFIG_A",
                    seed,
                )
                and runner.parameters_finite(model),
                f"seed_{seed}: frozen MaskablePPO model contract",
            )

            free_probs, episode_rows, anchor_rows, ledger = replay_seed(
                runner,
                model,
                assets,
                seed,
                specs,
                ref,
                anchor_index,
            )
            gate(
                f"seed_{seed}:full_600_episode_replay_exact",
                len(episode_rows) == N_DEV,
            )
            gate(
                f"seed_{seed}:zero_shield_interventions",
                all(
                    int(row["shield_interventions"]) == 0
                    for row in episode_rows
                ),
            )
            gate(
                f"seed_{seed}:deterministic_voluntary_clean_reproduced_zero",
                all(
                    int(row["N_voluntary_clean"]) == 0
                    for row in episode_rows
                ),
            )
            gate(
                f"seed_{seed}:all_163_anchor_links_exact",
                len(anchor_rows) == N_ANCHORS,
            )
            gate(
                f"seed_{seed}:DEVELOPMENT_Point_access_only",
                ledger.heldout_ppo_trajectory_access_count == 0
                and ledger.formal_perception_seed_access_count == 0
                and ledger.denied_accesses == 0,
            )

            seed_probability[str(seed)] = probability_summary(free_probs)
            all_episode_rows.extend(episode_rows)
            all_anchor_rows.extend(anchor_rows)

        gate("exact_two_model_loads", model_loads == 2)
        gate(
            "full_1200_episode_replay_exact",
            len(all_episode_rows) == 1200,
        )
        gate(
            "full_326_seed_anchor_links_exact",
            len(all_anchor_rows) == 326,
        )

        anchor_seed1 = [
            row for row in all_anchor_rows
            if int(row["parent_rl_seed"]) == DEVELOPMENT_SEEDS[0]
        ]
        anchor_seed2 = [
            row for row in all_anchor_rows
            if int(row["parent_rl_seed"]) == DEVELOPMENT_SEEDS[1]
        ]
        map2 = {
            (
                row["year"],
                int(row["trajectory_id"]),
                row["anchor_label"],
                int(row["anchor_day"]),
            ): row
            for row in anchor_seed2
        }
        cross_seed_absdiff = []
        for row in anchor_seed1:
            key = (
                row["year"],
                int(row["trajectory_id"]),
                row["anchor_label"],
                int(row["anchor_day"]),
            )
            other = map2[key]
            require(
                row["q50_float32"] == other["q50_float32"]
                and row["sin_DOY_float32"] == other["sin_DOY_float32"]
                and row["cos_DOY_float32"] == other["cos_DOY_float32"]
                and row["continuation_classification"]
                == other["continuation_classification"],
                "Cross-seed anchor observation/label identity",
            )
            cross_seed_absdiff.append(
                abs(float(row["p_CLEAN"]) - float(other["p_CLEAN"]))
            )

        label_rows = label_probability_summary(all_anchor_rows)
        separation = separation_summary(all_anchor_rows)
        duplicate = exact_duplicate_diagnostic(anchor_seed1)
        nn_rows, nn_summary = nearest_neighbor_diagnostic(anchor_seed1)

        performance_diagnostics_not_pass_gates = {
            "free_choice_probability_by_seed": seed_probability,
            "anchor_label_probability_separation": separation,
            "exact_duplicate_observation_diagnostic": {
                key: value
                for key, value in duplicate.items()
                if key != "duplicate_rows"
            },
            "nearest_neighbor_observation_diagnostic": nn_summary,
            "cross_seed_anchor_p_CLEAN_mean_abs_difference":
                float(statistics.fmean(cross_seed_absdiff)),
            "cross_seed_anchor_p_CLEAN_max_abs_difference":
                float(max(cross_seed_absdiff)),
            "role":
                "DESCRIPTIVE DEVELOPMENT DIAGNOSTICS ONLY; NOT USED TO PASS, RETRY, RESEED, TUNE, SELECT OR CHANGE THRESHOLDS",
        }

        protocol = {
            "stage": STAGE,
            "mode": "formal",
            "role":
                "FROZEN MASKED POINT POLICY PROBABILITY AND OBSERVATION AMBIGUITY AUDIT",
            "PPO_training_runs": 0,
            "PPO_model_loads": 2,
            "loaded_parent_rl_seeds": list(DEVELOPMENT_SEEDS),
            "evaluation_episode_count": 1200,
            "policy_observation":
                ["q50", "sin_DOY", "cos_DOY"],
            "probability_population":
                "all nonterminal free-choice DEVELOPMENT states only",
            "economic_anchor_population":
                "frozen 163 P2-2D-4AR safe-action anchors",
            "economic_anchor_link":
                "(year, trajectory_id, anchor_day), exact q50_float32 replay required",
            "nearest_neighbor_distance":
                "Euclidean distance in raw frozen Point coordinates; no radius/threshold",
            "performance_or_scientific_result_as_pass_gate": False,
            "FINAL_seed_training_runs": 0,
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
            "UA_training_or_model_loads": 0,
        }

        summary = {
            "stage": STAGE,
            "scientific_status": STATUS,
            "free_choice_probability_by_seed": seed_probability,
            "anchor_label_probability_separation": separation,
            "exact_duplicate_observation_diagnostic": {
                key: value
                for key, value in duplicate.items()
                if key != "duplicate_rows"
            },
            "nearest_neighbor_observation_diagnostic": nn_summary,
            "cross_seed_anchor_probability_difference": {
                "mean_abs_difference":
                    float(statistics.fmean(cross_seed_absdiff)),
                "max_abs_difference":
                    float(max(cross_seed_absdiff)),
            },
            "claim_boundary":
                "DEVELOPMENT-only diagnostic. Exact conflicting duplicates would be direct evidence of Point-observation ambiguity at those anchors; nearest-neighbor label mixing alone is suggestive local overlap, not proof of formal POMDP non-identifiability. Anchor labels are fixed-continuation counterfactual labels, not optimal-Q labels.",
            "performance_diagnostics_not_pass_gates":
                performance_diagnostics_not_pass_gates,
        }

        provenance = {
            "HEAD": head,
            "self_git_blob": self_blob,
            "P2_2D_5B": stage5b,
            "P2_2D_4B_4AR": {
                key: value
                for key, value in stage4.items()
                if key != "rows"
            },
            "pinned_source_blobs": PINNED_BLOBS,
        }

        duplicate_fields = (
            "q50_float32",
            "sin_DOY_float32",
            "cos_DOY_float32",
            "group_size",
            "classifications",
            "conflicting_labels",
            "members",
        )
        duplicate_rows = duplicate["duplicate_rows"]

        content = {
            "protocol_manifest.json": encode(protocol),
            "probability_aliasing_summary.json": encode(summary),
            "source_artifact_provenance.json": encode(provenance),
            "episode_probability_summary.csv":
                csv_bytes(all_episode_rows),
            "anchor_policy_probability.csv":
                csv_bytes(all_anchor_rows),
            "anchor_label_probability_summary.csv":
                csv_bytes(label_rows),
            "nearest_neighbor_anchor_diagnostic.csv":
                csv_bytes(nn_rows),
            "exact_duplicate_point_observations.csv":
                csv_bytes(
                    duplicate_rows,
                    fields=duplicate_fields,
                ),
        }
        for name, data in content.items():
            write_exclusive(OUTPUT / name, data)

        hashes = {
            name: sha_bytes(data)
            for name, data in content.items()
        }
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

        gate(
            "source_HEAD_and_worktree_unchanged",
            git("rev-parse", "HEAD") == head
            and git("status", "--porcelain", "--untracked-files=all") == "",
        )
        gate(
            "predecessor_evidence_preserved",
            verify_5b(head)["audit_summary_sha256"]
            == stage5b["audit_summary_sha256"]
            and verify_4b_and_4ar(head)[
                "P2_2D_4B_audit_summary_sha256"
            ]
            == stage4["P2_2D_4B_audit_summary_sha256"],
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
            "audit_summary_published_last",
            not (OUTPUT / "audit_summary.json").exists(),
        )
        require(all(gates.values()), "All P2-2D-5C structural gates")

        audit = {
            "stage": STAGE,
            "mode": "formal",
            "stage_pass": True,
            "scientific_status": STATUS,
            "HEAD": head,
            "gates": gates,
            "failed_gates": [],
            "PPO_training_runs": 0,
            "PPO_model_loads": 2,
            "evaluation_episode_count": 1200,
            "economic_anchor_links": 326,
            "FINAL_seed_training_runs": 0,
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
            "UA_training_or_model_loads": 0,
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
            encode(audit),
        )
        return audit

    except Exception as exc:
        if OUTPUT.exists() and not (OUTPUT / "execution_failure.json").exists():
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
                            "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/TRAINING/MASK/SHIELD/REWARD/PERCEPTION TUNING",
                    }
                ),
            )
        raise


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
