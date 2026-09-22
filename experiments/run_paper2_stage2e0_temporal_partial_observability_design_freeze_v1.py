
"""P2-2E-0-v1: compact temporal partial-observability recovery design freeze.

This stage freezes the LAST observation-representation candidate family for the
current Paper2 scope after:
- True-State established that the RL cleaning problem is learnable;
- Masked Point-PPO collapsed to WAIT;
- instantaneous width carried some information but did not improve decisions;
- Masked UA-PPO exactly tied Point/shield-only over 1200 DEVELOPMENT episodes.

No PPO is trained or loaded here.

Frozen candidates
-----------------
History-Point:
    [q50_t, delta_q50_t, tau_t, sin_DOY_t, cos_DOY_t]

History-UA:
    [q50_t, delta_q50_t, width_t, delta_width_t,
     tau_t, sin_DOY_t, cos_DOY_t]

tau_t is the normalized number of days since the most recent executed
in-episode CLEAN; before any in-episode CLEAN it is days since episode start.
The unknown pre-episode cleaning age is NOT imputed.

The two candidates are paired by design:
History-UA differs from History-Point only by width_t and delta_width_t.
This is required so any later History-UA minus History-Point difference can be
attributed to uncertainty information rather than to history itself.

No history-window length, LSTM/recurrent model, belief filter, reward change,
PPO retuning, new safety threshold, or extra uncertainty representation is
introduced in this candidate family.

The frozen q50-history mask/shield remains unchanged and width/delta-width/tau
are prohibited from the safety rule.

Formal runtime is only a deterministic construction audit on four fixed
DEVELOPMENT episodes using scripted safe actions; it performs no learning,
model selection, or protected access.

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
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SELF = (
    "experiments/"
    "run_paper2_stage2e0_temporal_partial_observability_design_freeze_v1.py"
)
TEMPORAL = "experiments/paper2_temporal_observation_v1.py"
MASKED_RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1_1.py"
MASK_BASE = "experiments/paper2_masked_perception_ppo_runner_v1.py"
SHIELDED = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
PERCEPTION = "experiments/paper2_perception_ppo_runner_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"
STAGE6A_SOURCE = (
    "experiments/run_paper2_stage2d6a_masked_ua_ppo_development_v1.py"
)

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
STAGE6A = BASE / "p2_2d_6a_masked_ua_ppo_development_v1/formal"
STAGE_DIRECTORY = (
    BASE / "p2_2e_0_temporal_partial_observability_design_freeze_v1"
)
OUTPUT = STAGE_DIRECTORY / "formal"

CHECKPOINT = "99fd3364d1bdcc6687f1ad7cb1e043ee3d901967"

PINNED_BLOBS = {
    MASKED_RUNNER: "9b1777ce4262efb95c677432ea52997352d76750",
    MASK_BASE: "be86a5e8a8a162c0d6be720c5340914ccc1ed82e",
    SHIELDED: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    PERCEPTION: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
    STAGE6A_SOURCE: "9954d5b41444084697ca8fb92c1372f158160f31",
}

EXPECTED_6A_HASHES = {
    "protocol_manifest.json":
        "aff4e49a13645e93a7513527aa06cb52462927dc60791b8e8d85a9246d3e1e06",
    "development_summary.json":
        "5c6f306ef75a4fd957d3b6a9cfa4e8cca2977fb93c17839ab5de475b50c002d1",
    "source_artifact_hashes.json":
        "75250baa01f8464bd10d1dc8df510814413a89fce455a3d2655dac3ac314dc8c",
    "masked_ua_development_run_summary.csv":
        "eff307020fec7471769881712483c9e5d4aa76eebeda4fb6ef8d6ac898d0c28e",
    "masked_ua_development_episode_metrics.csv":
        "40232df1ddc57efb6c0a9fffc580772d0620ed803e66bb28883a69b27e6582e9",
    "paired_ua_point_shield_comparison.csv":
        "e230e9e83ec347bf9000506aee9dc86f9a51d0147d09236fdb9680d6d97b7a30",
    "runs/seed_510001/training_completion.json":
        "8aa4031c7fe59376ad73a7efcf96ddd0544c25c050c77b07ab2a12900dc7f624",
    "runs/seed_510001/final_model.zip":
        "5d279a091227e18d7543dd163faecc23c245d518f2b476e318c8b470164ec82d",
    "runs/seed_510002/training_completion.json":
        "770647b6ad06816b64cac88d2386868da3ecf015917f0f1978dd3252fbc39df3",
    "runs/seed_510002/final_model.zip":
        "4273719afe151769495fec321b38d81d304bcd3942dea3a490c26a12b678b8b2",
    "output_hashes.json":
        "3b3c45eae6adc7fca1b85ff252d1df4a183f098c60faf50e2274d89b8844fe43",
}

FIXED_DEVELOPMENT_SPECS = (
    ("YEAR1", 0),
    ("YEAR1", 137),
    ("YEAR2", 0),
    ("YEAR2", 137),
)
SCRIPTED_CLEAN_DAYS = (0, 30, 120, 240)

STAGE = "P2-2E-0-v1"
STATUS = "TEMPORAL_PARTIAL_OBSERVABILITY_RECOVERY_DESIGN_FROZEN"


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


def csv_bytes(rows):
    require(rows, "Nonempty CSV rows")
    fields = list(rows[0])
    require(
        all(set(row) == set(fields) for row in rows),
        "Consistent CSV schema",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_exclusive(path, data):
    path = Path(path)
    require(
        path.parent.resolve() == OUTPUT.resolve(),
        "Writes restricted to new 2E-0 formal directory",
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


def _method_ast(source, class_name, method_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == method_name
                ):
                    return item
    raise RuntimeError(f"Missing {class_name}.{method_name}")


def verify_sources(head):
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned predecessor source changed: " + name,
        )
    for name in (SELF, TEMPORAL):
        require(
            git("rev-parse", "--verify", f"{head}:{name}")
            == git("hash-object", f"--path={name}", name)
            == git("rev-parse", f":{name}"),
            "New temporal source committed unchanged: " + name,
        )


def verify_6a():
    audit_path = STAGE6A / "audit_summary.json"
    require(audit_path.exists(), "P2-2D-6A audit exists")
    audit = json.loads(audit_path.read_bytes())
    require(
        audit["stage"] == "P2-2D-6A-v1"
        and audit["mode"] == "formal"
        and audit["stage_pass"] is True
        and audit["scientific_status"]
        == "MASKED_UA_PPO_DEVELOPMENT_CHARACTERIZATION_PASS"
        and audit["HEAD"] == CHECKPOINT
        and audit["failed_gates"] == []
        and all(audit["gates"].values())
        and audit["PPO_training_runs"] == 2
        and audit["Masked_UA_training_runs"] == 2
        and audit["development_budget_per_seed"] == 491520
        and audit["development_evaluation_episode_count"] == 1200
        and audit["training_shield_interventions"]
        == {"510001": 0, "510002": 0}
        and audit["evaluation_shield_interventions"]
        == {"510001": 0, "510002": 0}
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-2D-6A authority",
    )
    diag = audit["performance_diagnostics_not_pass_gates"]
    require(
        diag["any_deterministic_voluntary_clean_across_both_seeds"]
        is False
        and diag["each_seed_mean_cost_below_Point"] is False
        and diag["each_seed_mean_cost_below_shield_only"] is False
        and diag["two_seed_mean_of_seed_mean_J_total"]
        == 15.753629258296787
        and diag["two_seed_mean_of_seed_mean_voluntary_clean"] == 0.0,
        "Frozen 6A no-decision-gain result",
    )
    require(
        audit["output_sha256"] == EXPECTED_6A_HASHES,
        "P2-2D-6A output registry exact",
    )
    for name, digest in EXPECTED_6A_HASHES.items():
        require(
            sha_file(STAGE6A / name) == digest,
            "P2-2D-6A output hash: " + name,
        )
    return sha_file(audit_path)


def static_contract():
    helper_source = (ROOT / TEMPORAL).read_text(encoding="utf-8")
    audit_source = (ROOT / SELF).read_text(encoding="utf-8")

    helper_tree = ast.parse(helper_source)
    audit_tree = ast.parse(audit_source)

    # Feature constructor must not touch hidden/private scientific state.
    build = _method_ast(
        helper_source,
        "TemporalMaskedObservationWrapper",
        "_build_temporal_observation",
    )
    build_tokens = set()
    for node in ast.walk(build):
        if isinstance(node, ast.Name):
            build_tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            build_tokens.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            build_tokens.add(node.value)
    forbidden = {
        "L_pre", "latent", "rain", "reward", "energy", "future",
        "indices", "_audit_snapshot", "episode_return",
    }
    require(
        not (build_tokens & forbidden),
        "Temporal policy feature constructor must be public-observation-only",
    )

    # Mask source must stay width/history-free.
    mask_source = (ROOT / MASK_BASE).read_text(encoding="utf-8")
    mask = _method_ast(
        mask_source,
        "Q50MaskedShieldedPerceptionEnv",
        "_compute_action_mask",
    )
    mask_tokens = set()
    for node in ast.walk(mask):
        if isinstance(node, ast.Name):
            mask_tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            mask_tokens.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            mask_tokens.add(node.value)
    require(
        "width" not in mask_tokens
        and "delta_width" not in mask_tokens
        and "tau" not in mask_tokens
        and "belief_lower" in mask_tokens
        and "belief_upper" in mask_tokens
        and "action_safe" in mask_tokens,
        "Frozen safety mask remains q50-belief-only",
    )

    audit_attrs = {
        node.attr
        for node in ast.walk(audit_tree)
        if isinstance(node, ast.Attribute)
    }
    require(
        "learn" not in audit_attrs
        and "save" not in audit_attrs
        and "load" not in audit_attrs
        and "train_masked_final_checkpoint" not in audit_attrs
        and "build_masked_perception_ppo_model" not in audit_attrs,
        "Design freeze cannot train/load PPO",
    )

    import paper2_temporal_observation_v1 as temporal
    contract = temporal.temporal_contract()
    hp = contract["modes"][temporal.HISTORY_POINT]["features"]
    hua = contract["modes"][temporal.HISTORY_UA]["features"]
    require(
        hp
        == [
            "q50",
            "delta_q50",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ]
        and hua
        == [
            "q50",
            "delta_q50",
            "width",
            "delta_width",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ]
        and [x for x in hua if x not in hp]
        == ["width", "delta_width"],
        "History-UA differs from History-Point only by width information",
    )
    require(
        contract["day0"]
        == {
            "delta_q50": 0.0,
            "delta_width": 0.0,
            "tau_since_last_clean_or_episode_start": 0.0,
        }
        and contract["history_window_hyperparameter"] is None
        and contract["pre_episode_clean_age_imputed"] is False
        and contract["future_information_used"] is False
        and contract["latent_state_used"] is False
        and contract["safety_rule_changed"] is False,
        "Frozen no-leakage/no-window temporal design",
    )

    return {
        "zero_PPO_training_or_model_load": True,
        "public_observation_only_feature_constructor": True,
        "q50_only_width_history_free_safety_mask": True,
        "History_Point_feature_contract_exact": True,
        "History_UA_feature_contract_exact": True,
        "History_UA_minus_History_Point_is_width_and_delta_width_only":
            True,
        "day0_and_tau_semantics_pre_registered": True,
        "no_history_window_or_recurrent_model": True,
    }


def scripted_action(day, mask):
    require(
        mask.shape == (2,)
        and mask.dtype == np.bool_
        and bool(mask[1]),
        "Valid frozen WAIT/CLEAN mask",
    )
    if day in SCRIPTED_CLEAN_DAYS:
        return 1
    return 0 if bool(mask[0]) else 1


def run_pair(masked, temporal, assets, year, tid):
    point_ledger = masked.AccessLedger()
    ua_ledger = masked.AccessLedger()

    point_base = masked.build_masked_perception_eval_env(
        assets, year, tid, point_ledger, "Point"
    )
    ua_base = masked.build_masked_perception_eval_env(
        assets, year, tid, ua_ledger, "UA"
    )
    hp = temporal.wrap_temporal_masked_env(
        point_base, temporal.HISTORY_POINT
    )
    hua = temporal.wrap_temporal_masked_env(
        ua_base, temporal.HISTORY_UA
    )

    rows = []
    try:
        hp_obs, hp_info = hp.reset()
        hua_obs, hua_info = hua.reset()

        require(
            hp_obs.shape == (5,)
            and hua_obs.shape == (7,)
            and hp_obs.dtype == hua_obs.dtype == np.float32,
            "Frozen temporal observation dimensions/dtypes",
        )
        require(
            hp_obs[1] == 0.0
            and hp_obs[2] == 0.0
            and hua_obs[1] == 0.0
            and hua_obs[3] == 0.0
            and hua_obs[4] == 0.0,
            "Exact deterministic day-0 delta/tau semantics",
        )

        previous_q50 = float(hp_obs[0])
        previous_width = float(hua_obs[2])
        expected_age = 0

        for day in range(masked.get_protocol().decision_reward_steps):
            require(
                hp_info["day_index"] == hua_info["day_index"] == day,
                "Paired temporal day identity",
            )
            require(
                np.array_equal(
                    hp_obs[[0, 3, 4]],
                    hua_obs[[0, 5, 6]],
                )
                and hp_obs[1] == hua_obs[1]
                and hp_obs[2] == hua_obs[4],
                "History-Point/History-UA shared features exactly aligned",
            )

            if day == 0:
                require(
                    hp_obs[1] == 0.0
                    and hua_obs[3] == 0.0
                    and hp_obs[2] == 0.0,
                    "Day-0 deltas/tau exactly zero",
                )
            else:
                require(
                    hp_obs[1]
                    == np.float32(float(hp_obs[0]) - previous_q50)
                    and hua_obs[3]
                    == np.float32(float(hua_obs[2]) - previous_width)
                    and hp_obs[2]
                    == np.float32(
                        expected_age / temporal.TAU_DENOMINATOR
                    ),
                    "Exact causal delta/tau recurrence",
                )

            point_mask = hp.action_masks()
            ua_mask = hua.action_masks()
            require(
                np.array_equal(point_mask, ua_mask),
                "History Point/UA safety masks identical",
            )
            action = scripted_action(day, point_mask)

            rows.append(
                {
                    "year": year,
                    "trajectory_id": tid,
                    "day_index": day,
                    "action": action,
                    "WAIT_valid": bool(point_mask[0]),
                    "q50_float32": float(hp_obs[0]),
                    "delta_q50_float32": float(hp_obs[1]),
                    "width_float32": float(hua_obs[2]),
                    "delta_width_float32": float(hua_obs[3]),
                    "tau_float32": float(hp_obs[2]),
                    "sin_DOY_float32": float(hp_obs[3]),
                    "cos_DOY_float32": float(hp_obs[4]),
                }
            )

            previous_q50 = float(hp_obs[0])
            previous_width = float(hua_obs[2])

            hp_next = hp.step(action)
            hua_next = hua.step(action)
            hp_obs, _, hp_term, hp_trunc, hp_info = hp_next
            hua_obs, _, hua_term, hua_trunc, hua_info = hua_next

            require(
                hp_term == hua_term
                and hp_trunc == hua_trunc
                and not hp_trunc,
                "Paired temporal termination identity",
            )
            require(
                hp.last_shield_decision
                == hua.last_shield_decision,
                "Point/UA shield decision identity",
            )

            if day < masked.get_protocol().natural_transitions:
                executed = hp.last_shield_decision["executed_action"]
                expected_age = (
                    1 if executed == 1 else expected_age + 1
                )

        require(
            hp_term is True
            and hua_term is True
            and np.array_equal(hp_obs, np.zeros(5, dtype=np.float32))
            and np.array_equal(hua_obs, np.zeros(7, dtype=np.float32)),
            "Exact terminal zero sentinels",
        )
    finally:
        hp.close()
        hua.close()

    for ledger in (point_ledger, ua_ledger):
        require(
            ledger.heldout_ppo_trajectory_access_count == 0
            and ledger.formal_perception_seed_access_count == 0
            and ledger.denied_accesses == 0
            and all(row["role"] == "DEVELOPMENT" for row in ledger.records),
            "DEVELOPMENT-only temporal construction audit",
        )

    return rows


def run_formal():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive P2-2E-0 directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean worktree required",
    )
    head = git("rev-parse", "HEAD")
    verify_sources(head)
    stage6a_sha = verify_6a()
    static = static_contract()

    import paper2_masked_perception_ppo_runner_v1_1 as masked
    import paper2_temporal_observation_v1 as temporal

    assets = masked.load_assets()
    assets.ensure_perception()

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        rows = []
        for year, tid in FIXED_DEVELOPMENT_SPECS:
            rows.extend(
                run_pair(masked, temporal, assets, year, tid)
            )

        require(
            len(rows)
            == len(FIXED_DEVELOPMENT_SPECS)
            * masked.get_protocol().decision_reward_steps,
            "Exact fixed audit trace size",
        )

        contract = temporal.temporal_contract()
        design = {
            "stage": STAGE,
            "scientific_status": STATUS,
            "motivation":
                "Instantaneous q50+width did not improve masked UA decisions in P2-2D-6A; test compact causal temporal state without PPO retuning.",
            "candidate_family_is_last_for_current_scope": True,
            "History_Point":
                contract["modes"][temporal.HISTORY_POINT],
            "History_UA":
                contract["modes"][temporal.HISTORY_UA],
            "only_History_UA_additions_relative_to_History_Point":
                ["width", "delta_width"],
            "day0": contract["day0"],
            "delta_semantics": contract["delta_semantics"],
            "tau_semantics": contract["tau_semantics"],
            "tau_denominator": temporal.TAU_DENOMINATOR,
            "pre_episode_clean_age_imputed": False,
            "history_window_hyperparameter": None,
            "LSTM_or_recurrent_model": None,
            "belief_filter": None,
            "reward_change": None,
            "PPO_hyperparameter_change": None,
            "safety_mask_or_shield_change": None,
            "future_information_used": False,
            "latent_state_used": False,
            "future_experiment_order": [
                "P2-2E-1 History-Point DEVELOPMENT",
                "P2-2E-2 History-UA DEVELOPMENT",
            ],
            "future_fairness_rule":
                "same CONFIG_A, seeds 510001/510002, 491520 steps/seed, TRAINING/DEVELOPMENT populations, BLOCK10-v1 perception, reward, physics and q50-history mask/shield; both candidates run regardless of scientific performance of the first unless structural failure occurs",
            "scientific_comparisons": {
                "History_Point_minus_Static_Point":
                    "marginal value of compact temporal/control history",
                "History_UA_minus_History_Point":
                    "marginal value of uncertainty given the same compact history",
            },
            "stopping_rule":
                "If compact History-UA still fails to recover active decision behavior, stop adding observation candidates in this Paper2 scope; move recurrent/belief-state methods to extension/future work.",
        }

        gates = {
            "P2_2D_6A_negative_result_authority_exact":
                bool(stage6a_sha),
            "static_temporal_contract":
                all(static.values()),
            "fixed_four_DEVELOPMENT_episode_pairs":
                len(FIXED_DEVELOPMENT_SPECS) == 4,
            "full_365_day_construction_trace_each":
                len(rows) == 4 * 365,
            "Point_UA_shared_features_exact_runtime":
                True,
            "delta_and_tau_causal_recurrence_exact_runtime":
                True,
            "q50_only_safety_mask_identical_runtime":
                True,
            "PPO_training_runs_zero": True,
            "PPO_model_loads_zero": True,
            "protected_access_zero": True,
        }
        require(all(gates.values()), "All P2-2E-0 design gates")

        provenance = {
            "HEAD": head,
            "P2_2D_6A_audit_summary_sha256": stage6a_sha,
            "pinned_predecessor_blobs": PINNED_BLOBS,
            "new_source_sha256": {
                SELF: sha_file(ROOT / SELF),
                TEMPORAL: sha_file(ROOT / TEMPORAL),
            },
            "fixed_DEVELOPMENT_specs":
                [list(spec) for spec in FIXED_DEVELOPMENT_SPECS],
            "scripted_clean_days": list(SCRIPTED_CLEAN_DAYS),
        }

        content = {
            "temporal_observation_contract.json": encode(design),
            "construction_trace.csv": csv_bytes(rows),
            "source_artifact_provenance.json": encode(provenance),
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
        write_exclusive(
            OUTPUT / "output_hashes.json",
            output_hashes,
        )

        require(
            git("rev-parse", "HEAD") == head
            and git("status", "--porcelain", "--untracked-files=all") == "",
            "Source HEAD/worktree unchanged",
        )
        require(
            verify_6a() == stage6a_sha,
            "P2-2D-6A evidence preserved",
        )
        require(
            json.loads(output_hashes)["hashes"]
            == {
                name: sha_file(OUTPUT / name)
                for name in hashes
            },
            "Output hashes exact",
        )
        require(
            not (OUTPUT / "audit_summary.json").exists(),
            "Audit summary published last",
        )

        audit = {
            "stage": STAGE,
            "mode": "formal",
            "stage_pass": True,
            "scientific_status": STATUS,
            "HEAD": head,
            "gates": gates,
            "failed_gates": [],
            "PPO_training_runs": 0,
            "PPO_model_loads": 0,
            "development_episode_pairs_audited": 4,
            "construction_trace_rows": len(rows),
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
            "history_window_hyperparameter": None,
            "recurrent_model_used": False,
            "belief_filter_used": False,
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
                            "STOP; PRESERVE EVIDENCE; NO RETRY/OBSERVATION-REDESIGN/PPO-TRAINING/TUNING",
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
