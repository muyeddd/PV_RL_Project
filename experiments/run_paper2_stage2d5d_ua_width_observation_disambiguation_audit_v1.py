
"""P2-2D-5D-v1: UA-width observation disambiguation audit.

This DEVELOPMENT-only stage asks one narrow question before any UA-PPO
training:

    Does adding frozen uncertainty width to the Point observation
    [q50, sin_DOY, cos_DOY] reduce the economic-label ambiguity documented
    by P2-2D-5C?

No policy is trained or loaded.  The stage reuses the exact 163 frozen
P2-2D-4AR safe-action anchors and the exact P2-2D-5C Point anchor
observations.  For each anchor it replays the same always-WAIT + frozen q50
shield prefix in UA observation mode and records only:

    [q50, width, sin_DOY, cos_DOY]

The q50/season coordinates must match P2-2D-5C exactly.  Width is the only
additional observation coordinate.

Diagnostics:
1. Reproduce the P2-2D-5C Point exact-duplicate baseline:
      95 unique observations, 25 duplicate groups, 20 conflicting-label groups.
2. Recompute exact duplicates/conflicts in 4D UA observation space.
3. For each of the 20 conflicting Point groups, determine whether exact width
   splits it into label-pure UA subgroups.
4. Compare width distributions for CLEAN_BENEFICIAL vs WAIT_BENEFICIAL using
   threshold-free rank AUC.
5. Compare threshold-free nearest-neighbor label mixing in raw 3D Point and
   raw 4D UA coordinates.

Scientific outcomes are descriptive only.  Whether width resolves zero, some,
or all Point conflicts is NOT a structural PASS gate.  No tunable radius,
classifier, threshold, reward shaping, policy training, FINAL/formal held-out,
RANDOM_TEST, SEALED_DATES, or UA policy model is used.

Import is inert. Only --mode formal executes.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SELF = (
    "experiments/"
    "run_paper2_stage2d5d_ua_width_observation_disambiguation_audit_v1.py"
)
RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
BASE_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
STAGE5C_SOURCE = (
    "experiments/"
    "run_paper2_stage2d5c_masked_point_policy_probability_aliasing_audit_v1.py"
)
STAGE4AR_SOURCE = (
    "experiments/"
    "run_paper2_stage2d4ar_shield_delegation_action_identifiability_recovery_v1.py"
)
STAGE4B_SOURCE = (
    "experiments/"
    "run_paper2_stage2d4b_safe_action_distinction_decomposition_audit_v1.py"
)
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
STAGE5C = (
    BASE
    / "p2_2d_5c_masked_point_policy_probability_aliasing_audit_v1/formal"
)
STAGE4AR = (
    BASE
    / "p2_2d_4ar_shield_delegation_action_identifiability_recovery_v1/formal"
)
STAGE4B = (
    BASE
    / "p2_2d_4b_safe_action_distinction_decomposition_audit_v1/formal"
)

STAGE_DIRECTORY = (
    BASE / "p2_2d_5d_ua_width_observation_disambiguation_audit_v1"
)
OUTPUT = STAGE_DIRECTORY / "formal"

CHECKPOINT = "2d9e54ccbbc870cfa1b4923cea0dbacad5bde024"
N_ANCHORS = 163
WAIT = 0

PINNED_BLOBS = {
    RUNNER: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    BASE_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    STAGE5C_SOURCE: "83fc6965e087a5a1bbf5a98588aee6f5dca94a19",
    STAGE4AR_SOURCE: "a778fa61aa7559353ed7e655d04298c4c8724f86",
    STAGE4B_SOURCE: "e5b4006c9ddb1866e78df2335ccc855de2253b98",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
}

EXPECTED_5C_HASHES = {
    "protocol_manifest.json":
        "1a4427687fc73337b7c3c19a5f47bf7497fc0720c812f6dc073e7576b7af49ac",
    "probability_aliasing_summary.json":
        "cf4fc6588f60205430ae0ca002eaed87cb76fbc38dc071c90740174d1996b598",
    "source_artifact_provenance.json":
        "d958edf9b53db14969feac935fcd39629e4023e8d13e3737a36507f7c96a5b22",
    "episode_probability_summary.csv":
        "47f80609dadba688a521f71396b02a342469c326ef5a9c3ed7644050d541279e",
    "anchor_policy_probability.csv":
        "b501f3d245744b92a5c4a55d9aa037b1c0d25f0c7b73ddba35fca4177ec898f4",
    "anchor_label_probability_summary.csv":
        "9a0e32bdf5a49366373536ea73c22530497c955365881d9a33261e1157f3e609",
    "nearest_neighbor_anchor_diagnostic.csv":
        "9c0b20b604a59785f33e04a924014d954a0454aac17d60b45b1d692733e42e4f",
    "exact_duplicate_point_observations.csv":
        "0883405d7079669c25b0cc9886be8e0b655763d2ead84a3c5d98303622c2fb0d",
    "output_hashes.json":
        "7b094659985527ca10f1a82e9b7707d3b3d2cdd229f80e0f4e6d9e6815c505f0",
}

EXPECTED_4AR_SAFE_SHA256 = (
    "5a147ca25d979088ce1ad4bc3af8d4bde0f87cf4cd0f543164452673dfc5c8f8"
)

LABELS = (
    "CLEAN_BENEFICIAL",
    "WAIT_BENEFICIAL",
    "COST_TIE_WITHIN_ATOL",
)

POINT_EXPECTED = {
    "anchor_count": 163,
    "unique_exact_observation_count": 95,
    "exact_duplicate_group_count": 25,
    "exact_conflicting_group_count": 20,
}

STAGE = "P2-2D-5D-v1"
STATUS = "UA_WIDTH_OBSERVATION_DISAMBIGUATION_AUDIT_COMPLETE"


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
        "Writes restricted to new 5D formal directory",
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


def verify_5c(head):
    audit_path = STAGE5C / "audit_summary.json"
    require(audit_path.exists(), "P2-2D-5C audit exists")
    audit = json.loads(audit_path.read_bytes())

    require(
        audit["stage"] == "P2-2D-5C-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "MASKED_POINT_POLICY_PROBABILITY_ALIASING_AUDIT_COMPLETE"
        and audit["HEAD"] == CHECKPOINT
        and audit["failed_gates"] == []
        and all(audit["gates"].values())
        and audit["PPO_training_runs"] == 0
        and audit["PPO_model_loads"] == 2
        and audit["evaluation_episode_count"] == 1200
        and audit["economic_anchor_links"] == 326
        and audit["FINAL_seed_training_runs"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0
        and audit["UA_training_or_model_loads"] == 0,
        "Accepted P2-2D-5C authority",
    )
    diag = audit["performance_diagnostics_not_pass_gates"]
    exact = diag["exact_duplicate_observation_diagnostic"]
    require(
        exact["anchor_count"] == 163
        and exact["unique_exact_Point_observation_count"] == 95
        and exact["exact_duplicate_group_count"] == 25
        and exact["exact_duplicate_conflicting_label_groups"] == 20
        and exact["direct_exact_observation_aliasing_proven"] is True,
        "Frozen P2-2D-5C Point ambiguity baseline",
    )
    require(
        audit["output_sha256"] == EXPECTED_5C_HASHES,
        "P2-2D-5C output hash registry exact",
    )
    for name, digest in EXPECTED_5C_HASHES.items():
        require(
            sha_file(STAGE5C / name) == digest,
            "P2-2D-5C output hash: " + name,
        )
    verify_sources(head)
    return {
        "audit_summary_sha256": sha_file(audit_path),
        "read_only": True,
    }


def verify_4ar_4b(head):
    safe_path = STAGE4AR / "safe_action_counterfactuals.csv"
    audit_path = STAGE4B / "audit_summary.json"
    require(
        safe_path.exists() and audit_path.exists(),
        "P2-2D-4AR/4B evidence exists",
    )
    require(
        sha_file(safe_path) == EXPECTED_4AR_SAFE_SHA256,
        "Exact frozen 4AR safe-action evidence",
    )
    audit = json.loads(audit_path.read_bytes())
    require(
        audit["stage"] == "P2-2D-4B-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "SAFE_ACTION_DISTINCTION_DECOMPOSITION_AUDIT_COMPLETE"
        and audit["counterfactual_anchor_count"] == 163
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
    rows = read_csv(safe_path)
    require(
        len(rows) == N_ANCHORS
        and sum(r["continuation_classification"] == "CLEAN_BENEFICIAL" for r in rows) == 63
        and sum(r["continuation_classification"] == "WAIT_BENEFICIAL" for r in rows) == 90
        and sum(r["continuation_classification"] == "COST_TIE_WITHIN_ATOL" for r in rows) == 10,
        "Exact frozen economic labels",
    )
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
        "train_final_checkpoint" not in attrs
        and "train_masked_final_checkpoint" not in attrs
        and "learn" not in attrs
        and "save" not in attrs
        and "load" not in attrs,
        "No PPO train/load/save",
    )
    require(
        "build_shielded_perception_eval_env" in attrs,
        "Frozen shielded DEVELOPMENT evaluation environment required",
    )
    require(
        "build_shielded_scheduled_perception_env" not in attrs
        and "build_shielded_perception_ppo_model" not in attrs,
        "No training environment/model constructor",
    )
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
                )
            ):
                protected.append(value.value)
    require(not protected, "No protected-role runtime literal")
    require(
        "width_rank_AUC_clean_higher_than_wait" in source
        and "point_conflict_group_resolution" in source
        and "nearest_neighbor_label_mixing" in source
        and "performance_diagnostics_not_pass_gates" in source,
        "Required no-threshold disambiguation diagnostics present",
    )
    return {
        "zero_PPO_training": True,
        "zero_PPO_model_load": True,
        "shielded_DEVELOPMENT_eval_runtime_only": True,
        "UA_observation_replay_present": True,
        "no_training_environment_or_model_constructor": True,
        "no_protected_runtime": True,
        "threshold_free_disambiguation_diagnostics_present": True,
        "performance_diagnostics_not_pass_gates": True,
    }


def point_anchor_rows():
    rows = read_csv(STAGE5C / "anchor_policy_probability.csv")
    rows = [
        row for row in rows
        if int(row["parent_rl_seed"]) == 510001
    ]
    require(len(rows) == N_ANCHORS, "Exactly one frozen Point anchor copy")
    out = {}
    for row in rows:
        key = (
            row["year"],
            int(row["trajectory_id"]),
            row["anchor_label"],
            int(row["anchor_day"]),
        )
        require(key not in out, "Unique Point anchor key")
        out[key] = row
    return out


def economic_anchor_rows(rows):
    out = {}
    for row in rows:
        key = (
            row["year"],
            int(row["trajectory_id"]),
            row["anchor_label"],
            int(row["anchor_day"]),
        )
        require(key not in out, "Unique economic anchor key")
        out[key] = row
    require(len(out) == N_ANCHORS, "163 unique economic anchors")
    return out


def exact_group_diagnostic(rows, dimensions):
    groups = {}
    for row in rows:
        key = tuple(float(np.float32(row[name])) for name in dimensions)
        groups.setdefault(key, []).append(row)

    duplicate = 0
    conflicting = 0
    detail = []
    for key, items in groups.items():
        if len(items) <= 1:
            continue
        duplicate += 1
        labels = sorted(
            {item["continuation_classification"] for item in items}
        )
        conflict = len(labels) > 1
        conflicting += int(conflict)
        detail.append(
            {
                **{
                    dimensions[index]: key[index]
                    for index in range(len(dimensions))
                },
                "group_size": len(items),
                "classifications": "|".join(labels),
                "conflicting_labels": conflict,
                "members": "|".join(
                    f'{item["year"]}:{item["trajectory_id"]}:'
                    f'{item["anchor_label"]}:{item["anchor_day"]}'
                    for item in items
                ),
            }
        )
    return {
        "anchor_count": len(rows),
        "unique_exact_observation_count": len(groups),
        "exact_duplicate_group_count": duplicate,
        "exact_conflicting_group_count": conflicting,
        "groups": groups,
        "detail": detail,
    }


def point_conflict_group_resolution(point_diag):
    rows = []
    resolved = 0
    unresolved = 0
    total_conflict = 0

    for point_key, items in point_diag["groups"].items():
        labels = {item["continuation_classification"] for item in items}
        if len(labels) <= 1:
            continue

        total_conflict += 1
        by_width = {}
        for item in items:
            width = float(np.float32(item["width_float32"]))
            by_width.setdefault(width, []).append(item)

        subgroup_conflicts = 0
        for subgroup in by_width.values():
            if len(
                {
                    item["continuation_classification"]
                    for item in subgroup
                }
            ) > 1:
                subgroup_conflicts += 1

        is_resolved = subgroup_conflicts == 0
        resolved += int(is_resolved)
        unresolved += int(not is_resolved)
        rows.append(
            {
                "q50_float32": point_key[0],
                "sin_DOY_float32": point_key[1],
                "cos_DOY_float32": point_key[2],
                "point_group_size": len(items),
                "point_classifications": "|".join(sorted(labels)),
                "unique_width_count": len(by_width),
                "UA_subgroup_count": len(by_width),
                "UA_conflicting_subgroup_count": subgroup_conflicts,
                "resolved_by_exact_width": is_resolved,
                "members": "|".join(
                    f'{item["year"]}:{item["trajectory_id"]}:'
                    f'{item["anchor_label"]}:{item["anchor_day"]}:'
                    f'{item["width_float32"]}'
                    for item in items
                ),
            }
        )

    require(
        total_conflict == POINT_EXPECTED["exact_conflicting_group_count"],
        "Reproduced 20 Point conflicting groups",
    )
    return rows, {
        "point_conflicting_group_count": total_conflict,
        "fully_resolved_by_exact_width_count": resolved,
        "still_conflicting_after_exact_width_count": unresolved,
        "fully_resolved_fraction": resolved / total_conflict,
    }


def describe(values):
    require(values, "Nonempty numeric population")
    arr = np.asarray(values, dtype=np.float64)
    q = np.quantile(
        arr,
        [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        method="linear",
    )
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "p01": float(q[0]),
        "p10": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "max": float(np.max(arr)),
    }


def rank_auc(a, b):
    require(a and b, "Both classes required")
    wins = 0.0
    total = 0
    for x in a:
        for y in b:
            total += 1
            if x > y:
                wins += 1.0
            elif x == y:
                wins += 0.5
    return float(wins / total)


def width_label_summary(rows):
    clean = [
        float(row["width_float32"])
        for row in rows
        if row["continuation_classification"] == "CLEAN_BENEFICIAL"
    ]
    wait = [
        float(row["width_float32"])
        for row in rows
        if row["continuation_classification"] == "WAIT_BENEFICIAL"
    ]
    tie = [
        float(row["width_float32"])
        for row in rows
        if row["continuation_classification"] == "COST_TIE_WITHIN_ATOL"
    ]
    require(
        len(clean) == 63 and len(wait) == 90 and len(tie) == 10,
        "Frozen economic-label counts",
    )
    auc = rank_auc(clean, wait)
    return {
        "CLEAN_BENEFICIAL": describe(clean),
        "WAIT_BENEFICIAL": describe(wait),
        "COST_TIE_WITHIN_ATOL": describe(tie),
        "mean_width_difference_CLEANminusWAIT":
            float(statistics.fmean(clean) - statistics.fmean(wait)),
        "median_width_difference_CLEANminusWAIT":
            float(statistics.median(clean) - statistics.median(wait)),
        "width_rank_AUC_clean_higher_than_wait": auc,
        "absolute_rank_separation_from_0p5": abs(auc - 0.5),
        "interpretation":
            "Threshold-free descriptive ranking only; not a classifier and not a PASS gate.",
    }


def nearest_neighbor(rows, dimensions):
    use = [
        row for row in rows
        if row["continuation_classification"]
        in ("CLEAN_BENEFICIAL", "WAIT_BENEFICIAL")
    ]
    require(len(use) == 153, "153 non-tie anchors")
    x = np.asarray(
        [[float(row[name]) for name in dimensions] for row in use],
        dtype=np.float64,
    )
    labels = [row["continuation_classification"] for row in use]
    detail = []

    for i, row in enumerate(use):
        dist = np.linalg.norm(x - x[i], axis=1)
        dist[i] = np.inf
        same = [
            j for j in range(len(use))
            if j != i and labels[j] == labels[i]
        ]
        opp = [
            j for j in range(len(use))
            if labels[j] != labels[i]
        ]
        require(same and opp, "Both neighbor classes exist")
        any_idx = int(np.argmin(dist))
        same_idx = min(same, key=lambda j: dist[j])
        opp_idx = min(opp, key=lambda j: dist[j])

        detail.append(
            {
                "year": row["year"],
                "trajectory_id": row["trajectory_id"],
                "anchor_label": row["anchor_label"],
                "anchor_day": row["anchor_day"],
                "classification": labels[i],
                "nearest_any_distance": float(dist[any_idx]),
                "nearest_any_classification": labels[any_idx],
                "nearest_any_is_opposite_label":
                    labels[any_idx] != labels[i],
                "nearest_same_label_distance":
                    float(dist[same_idx]),
                "nearest_opposite_label_distance":
                    float(dist[opp_idx]),
                "opposite_closer_than_same":
                    bool(dist[opp_idx] < dist[same_idx]),
            }
        )

    opposite_any = sum(
        row["nearest_any_is_opposite_label"] for row in detail
    )
    opposite_closer = sum(
        row["opposite_closer_than_same"] for row in detail
    )
    summary = {
        "population": len(detail),
        "dimensions": list(dimensions),
        "threshold_used": None,
        "nearest_neighbor_label_mixing": {
            "nearest_any_is_opposite_label_count":
                int(opposite_any),
            "nearest_any_is_opposite_label_fraction":
                float(opposite_any / len(detail)),
            "opposite_closer_than_same_count":
                int(opposite_closer),
            "opposite_closer_than_same_fraction":
                float(opposite_closer / len(detail)),
            "median_nearest_same_label_distance":
                float(
                    statistics.median(
                        row["nearest_same_label_distance"]
                        for row in detail
                    )
                ),
            "median_nearest_opposite_label_distance":
                float(
                    statistics.median(
                        row["nearest_opposite_label_distance"]
                        for row in detail
                    )
                ),
        },
    }
    return detail, summary


def replay_ua_anchors(runner, assets, economic, point):
    by_episode = {}
    for key, row in economic.items():
        episode_key = (key[0], key[1])
        by_episode.setdefault(episode_key, []).append((key, row))

    ledger = runner.AccessLedger()
    ua_rows = []

    for (year, tid), items in sorted(by_episode.items()):
        env = runner.build_shielded_perception_eval_env(
            assets,
            year,
            tid,
            ledger,
            "UA",
        )
        try:
            obs, info = env.reset()
            runner.audit_perception_observation(env.inner, obs, info)
            require(
                obs.shape == (4,)
                and obs.dtype == np.float32,
                "Exact frozen UA observation shape/dtype",
            )

            anchors_by_day = {}
            max_anchor = -1
            first_forced_values = set()
            for key, econ in items:
                day = int(econ["anchor_day"])
                anchors_by_day.setdefault(day, []).append((key, econ))
                max_anchor = max(max_anchor, day)
                first_forced_values.add(int(econ["first_forced_day"]))
                require(
                    day < int(econ["first_forced_day"]),
                    "Anchor strictly before first forced CLEAN",
                )
            require(
                len(first_forced_values) == 1,
                "One first-forced day per episode",
            )

            for day in range(max_anchor + 1):
                if day in anchors_by_day:
                    for key, econ in anchors_by_day[day]:
                        p_row = point[key]
                        q50 = float(np.float32(obs[0]))
                        width = float(np.float32(obs[1]))
                        sin_doy = float(np.float32(obs[2]))
                        cos_doy = float(np.float32(obs[3]))

                        require(
                            q50 == float(np.float32(float(econ["q50_float32"])))
                            and q50 == float(np.float32(float(p_row["q50_float32"])))
                            and sin_doy
                            == float(np.float32(float(p_row["sin_DOY_float32"])))
                            and cos_doy
                            == float(np.float32(float(p_row["cos_DOY_float32"]))),
                            "UA replay preserves exact Point coordinates",
                        )
                        require(
                            np.isfinite(width)
                            and 0.0 <= width <= 1.0,
                            "Finite nonnegative frozen UA width",
                        )
                        require(
                            p_row["continuation_classification"]
                            == econ["continuation_classification"],
                            "Economic label identity",
                        )

                        ua_rows.append(
                            {
                                "year": year,
                                "trajectory_id": int(tid),
                                "anchor_label": key[2],
                                "anchor_day": day,
                                "first_forced_day":
                                    int(econ["first_forced_day"]),
                                "continuation_classification":
                                    econ["continuation_classification"],
                                "remaining_horizon_gap_CLEANminusWAIT":
                                    float(
                                        econ[
                                            "remaining_horizon_gap_CLEANminusWAIT"
                                        ]
                                    ),
                                "q50_float32": q50,
                                "width_float32": width,
                                "sin_DOY_float32": sin_doy,
                                "cos_DOY_float32": cos_doy,
                            }
                        )

                obs, _, terminated, truncated, info = env.step(WAIT)
                snap = runner.audit_perception_observation(
                    env.inner,
                    obs,
                    info,
                    terminal=terminated,
                )
                decision = env.last_shield_decision
                require(
                    decision is not None
                    and decision["day_index"] == day
                    and decision["proposed_action"] == WAIT
                    and decision["executed_action"] == WAIT
                    and decision["intervention"] is False
                    and decision["proposed_safe"] is True,
                    "All replay-prefix WAIT actions remain safe before first forced day",
                )
                require(
                    not truncated and not terminated,
                    "Anchor replay prefix remains nonterminal",
                )
                require(
                    snap["day_index"] == day + 1,
                    "Exact next-day progression",
                )
        finally:
            env.close()

    require(
        len(ua_rows) == N_ANCHORS,
        "All 163 anchors replayed exactly once in UA mode",
    )
    require(
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0
        and all(
            row["role"] == "DEVELOPMENT"
            and row["observation_mode"] == "UA"
            for row in ledger.records
        ),
        "DEVELOPMENT UA observation access only",
    )
    return ua_rows, ledger


def run_formal():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive 5D directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean worktree required",
    )
    head = git("rev-parse", "HEAD")
    verify_sources(head)
    stage5c = verify_5c(head)
    stage4 = verify_4ar_4b(head)

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(
        self_blob
        == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "5D source committed unchanged",
    )
    static = static_contract()

    import paper2_shielded_perception_ppo_runner_v1 as runner

    p = runner.get_protocol()
    require(
        dict(p.observations)["Point"]
        == ("q50", "sin_DOY", "cos_DOY")
        and dict(p.observations)["UA"]
        == ("q50", "width", "sin_DOY", "cos_DOY")
        and dict(p.observation_dimensions)["Point"] == 3
        and dict(p.observation_dimensions)["UA"] == 4,
        "Frozen Point/UA observation contract",
    )

    assets = runner.load_assets()
    assets.ensure_perception()

    point = point_anchor_rows()
    economic = economic_anchor_rows(stage4["rows"])
    require(
        set(point) == set(economic),
        "Exact Point/economic anchor-key identity",
    )

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        gates = {}

        def gate(name, condition):
            require(name not in gates, "Duplicate gate: " + name)
            gates[name] = bool(condition)
            require(condition, name)

        gate("P2_2D_5C_predecessor_exact", bool(stage5c))
        gate("P2_2D_4AR_4B_anchor_evidence_exact", bool(stage4))
        gate("static_zero_training_model_load_contract", all(static.values()))

        ua_rows, ledger = replay_ua_anchors(
            runner,
            assets,
            economic,
            point,
        )
        gate("all_163_UA_anchor_observations_replayed", len(ua_rows) == 163)
        gate(
            "DEVELOPMENT_UA_observation_access_only",
            ledger.heldout_ppo_trajectory_access_count == 0
            and ledger.formal_perception_seed_access_count == 0
            and ledger.denied_accesses == 0,
        )
        gate(
            "finite_nonnegative_width_all_anchors",
            all(
                np.isfinite(row["width_float32"])
                and 0.0 <= float(row["width_float32"]) <= 1.0
                for row in ua_rows
            ),
        )

        point_diag = exact_group_diagnostic(
            ua_rows,
            ("q50_float32", "sin_DOY_float32", "cos_DOY_float32"),
        )
        gate(
            "Point_exact_duplicate_baseline_reproduced",
            point_diag["anchor_count"] == POINT_EXPECTED["anchor_count"]
            and point_diag["unique_exact_observation_count"]
            == POINT_EXPECTED["unique_exact_observation_count"]
            and point_diag["exact_duplicate_group_count"]
            == POINT_EXPECTED["exact_duplicate_group_count"]
            and point_diag["exact_conflicting_group_count"]
            == POINT_EXPECTED["exact_conflicting_group_count"],
        )

        ua_diag = exact_group_diagnostic(
            ua_rows,
            (
                "q50_float32",
                "width_float32",
                "sin_DOY_float32",
                "cos_DOY_float32",
            ),
        )
        resolution_rows, resolution = point_conflict_group_resolution(
            point_diag
        )
        widths = width_label_summary(ua_rows)

        point_nn_rows, point_nn = nearest_neighbor(
            ua_rows,
            ("q50_float32", "sin_DOY_float32", "cos_DOY_float32"),
        )
        ua_nn_rows, ua_nn = nearest_neighbor(
            ua_rows,
            (
                "q50_float32",
                "width_float32",
                "sin_DOY_float32",
                "cos_DOY_float32",
            ),
        )

        performance_diagnostics_not_pass_gates = {
            "Point_exact_duplicate_baseline": {
                key: value
                for key, value in point_diag.items()
                if key not in ("groups", "detail")
            },
            "UA_exact_duplicate_diagnostic": {
                key: value
                for key, value in ua_diag.items()
                if key not in ("groups", "detail")
            },
            "point_conflict_group_resolution": resolution,
            "width_by_economic_label": widths,
            "Point_nearest_neighbor": point_nn,
            "UA_nearest_neighbor": ua_nn,
            "width_reduces_exact_conflicting_groups":
                ua_diag["exact_conflicting_group_count"]
                < point_diag["exact_conflicting_group_count"],
            "width_fully_resolves_any_Point_conflict_group":
                resolution["fully_resolved_by_exact_width_count"] > 0,
            "role":
                "DESCRIPTIVE DEVELOPMENT OBSERVATION DIAGNOSTICS ONLY; NOT USED TO PASS, TUNE, SELECT, DEFINE A THRESHOLD OR AUTHORIZE UA PERFORMANCE CLAIMS",
        }

        summary = {
            "stage": STAGE,
            "scientific_status": STATUS,
            "Point_observation":
                ["q50", "sin_DOY", "cos_DOY"],
            "UA_observation":
                ["q50", "width", "sin_DOY", "cos_DOY"],
            "added_coordinate_only": "width",
            "Point_exact_duplicate_baseline": {
                key: value
                for key, value in point_diag.items()
                if key not in ("groups", "detail")
            },
            "UA_exact_duplicate_diagnostic": {
                key: value
                for key, value in ua_diag.items()
                if key not in ("groups", "detail")
            },
            "point_conflict_group_resolution": resolution,
            "width_by_economic_label": widths,
            "nearest_neighbor_comparison": {
                "Point": point_nn,
                "UA": ua_nn,
            },
            "claim_boundary":
                "DEVELOPMENT-only fixed-anchor diagnostic. Fewer exact conflicts after adding width is direct evidence that width disambiguates those specific frozen anchors. Nearest-neighbor changes are descriptive local-overlap evidence only. Economic labels are fixed-continuation counterfactual labels, not optimal-Q labels. No UA policy performance claim is made.",
            "performance_diagnostics_not_pass_gates":
                performance_diagnostics_not_pass_gates,
        }

        protocol = {
            "stage": STAGE,
            "mode": "formal",
            "role":
                "PRE-UA-TRAINING FROZEN WIDTH OBSERVATION DISAMBIGUATION AUDIT",
            "anchor_population":
                "exact 163 frozen P2-2D-4AR safe-action anchors",
            "observation_comparison": {
                "Point": ["q50", "sin_DOY", "cos_DOY"],
                "UA": ["q50", "width", "sin_DOY", "cos_DOY"],
                "only_added_coordinate": "width",
            },
            "anchor_replay":
                "same DEVELOPMENT episode and always-WAIT + frozen q50 shield prefix; q50/season must match 5C exactly",
            "PPO_training_runs": 0,
            "PPO_model_loads": 0,
            "UA_policy_training_runs": 0,
            "UA_policy_model_loads": 0,
            "threshold_or_radius": None,
            "classifier_fit": None,
            "scientific_results_as_pass_gates": False,
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
        }

        provenance = {
            "HEAD": head,
            "self_git_blob": self_blob,
            "P2_2D_5C": stage5c,
            "P2_2D_4AR_4B": {
                key: value
                for key, value in stage4.items()
                if key != "rows"
            },
            "pinned_source_blobs": PINNED_BLOBS,
        }

        point_detail_fields = (
            "q50_float32",
            "sin_DOY_float32",
            "cos_DOY_float32",
            "group_size",
            "classifications",
            "conflicting_labels",
            "members",
        )
        ua_detail_fields = (
            "q50_float32",
            "width_float32",
            "sin_DOY_float32",
            "cos_DOY_float32",
            "group_size",
            "classifications",
            "conflicting_labels",
            "members",
        )

        content = {
            "protocol_manifest.json": encode(protocol),
            "ua_width_disambiguation_summary.json": encode(summary),
            "source_artifact_provenance.json": encode(provenance),
            "ua_anchor_observations.csv": csv_bytes(ua_rows),
            "point_exact_duplicate_groups.csv":
                csv_bytes(point_diag["detail"], point_detail_fields),
            "ua_exact_duplicate_groups.csv":
                csv_bytes(ua_diag["detail"], ua_detail_fields),
            "point_conflict_group_resolution.csv":
                csv_bytes(resolution_rows),
            "point_nearest_neighbor_anchor_diagnostic.csv":
                csv_bytes(point_nn_rows),
            "ua_nearest_neighbor_anchor_diagnostic.csv":
                csv_bytes(ua_nn_rows),
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
            "PPO_training_and_model_loads_zero",
            True,
        )
        gate(
            "source_HEAD_and_worktree_unchanged",
            git("rev-parse", "HEAD") == head
            and git("status", "--porcelain", "--untracked-files=all") == "",
        )
        gate(
            "predecessor_evidence_preserved",
            verify_5c(head)["audit_summary_sha256"]
            == stage5c["audit_summary_sha256"]
            and verify_4ar_4b(head)["P2_2D_4AR_safe_action_sha256"]
            == stage4["P2_2D_4AR_safe_action_sha256"],
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
        require(all(gates.values()), "All P2-2D-5D structural gates")

        audit = {
            "stage": STAGE,
            "mode": "formal",
            "stage_pass": True,
            "scientific_status": STATUS,
            "HEAD": head,
            "gates": gates,
            "failed_gates": [],
            "anchor_count": N_ANCHORS,
            "PPO_training_runs": 0,
            "PPO_model_loads": 0,
            "UA_policy_training_runs": 0,
            "UA_policy_model_loads": 0,
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
                            "STOP; PRESERVE EVIDENCE; NO RETRY/UA TRAINING/THRESHOLD/RADIUS/REWARD/MASK/SHIELD TUNING",
                    }
                ),
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
