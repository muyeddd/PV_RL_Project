
"""P2-2E-0R-v1: temporal design-freeze recovery for day-index semantics.

Purpose
-------
The original P2-2E-0 formal run stopped fail-closed at:

    RuntimeError: Paired temporal day identity

That failure is preserved and is NOT retried or overwritten.

Root cause is audit semantics only:
- reset returns observation o_0 with info.day_index == 0;
- step(action_t) returns the NEXT observation o_{t+1}, while the accompanying
  info.day_index still denotes the SETTLED ACTION DAY t.

Therefore, at the beginning of decision loop t > 0:
- the temporal observation is o_t;
- the incoming info belongs to the previously settled action day t-1.

This recovery changes only that audit interpretation. It does NOT change:
- paper2_temporal_observation_v1.py;
- History-Point / History-UA feature definitions;
- day-0 delta/tau rules;
- q50-history mask/shield;
- perception, reward, physics, PPO configuration, seeds or budgets.

No PPO is trained or loaded. Only four fixed DEVELOPMENT episode pairs are
replayed with the same deterministic safe scripted actions used by P2-2E-0.

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
    "run_paper2_stage2e0r_temporal_partial_observability_design_recovery_v1.py"
)
FAILED_STAGE_SOURCE = (
    "experiments/"
    "run_paper2_stage2e0_temporal_partial_observability_design_freeze_v1.py"
)
TEMPORAL = "experiments/paper2_temporal_observation_v1.py"
MASKED_RUNNER = "experiments/paper2_masked_perception_ppo_runner_v1_1.py"
MASK_BASE = "experiments/paper2_masked_perception_ppo_runner_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
FAILED_FORMAL = (
    BASE / "p2_2e_0_temporal_partial_observability_design_freeze_v1/formal"
)
STAGE_DIRECTORY = (
    BASE / "p2_2e_0r_temporal_partial_observability_design_recovery_v1"
)
OUTPUT = STAGE_DIRECTORY / "formal"

CHECKPOINT = "656360fb71adc954f06f4b6465a201a8f82fbead"

PINNED_BLOBS = {
    FAILED_STAGE_SOURCE: "c4e4baf44d675e69440fbc2cc9b9af6a358f0ff1",
    TEMPORAL: "fec506636e53855be4cd79d8dd6702e9b71c674d",
    MASKED_RUNNER: "9b1777ce4262efb95c677432ea52997352d76750",
    MASK_BASE: "be86a5e8a8a162c0d6be720c5340914ccc1ed82e",
}

STAGE = "P2-2E-0R-v1"
STATUS = "TEMPORAL_PARTIAL_OBSERVABILITY_DESIGN_RECOVERY_PASS_FROZEN"


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
        "Writes restricted to new 2E-0R formal directory",
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


def verify_failure_evidence():
    require(
        FAILED_FORMAL.exists(),
        "Original P2-2E-0 failed formal directory must exist",
    )
    inventory = sorted(
        item.name for item in FAILED_FORMAL.iterdir() if item.is_file()
    )
    require(
        inventory == ["execution_failure.json"],
        "Original failed directory must remain immutable with failure evidence only",
    )
    path = FAILED_FORMAL / "execution_failure.json"
    failure = json.loads(path.read_bytes())
    require(
        failure["stage"] == "P2-2E-0-v1"
        and failure["stage_pass"] is False
        and failure["HEAD"] == CHECKPOINT
        and failure["error_type"] == "RuntimeError"
        and failure["message"] == "Paired temporal day identity"
        and "STOP; PRESERVE EVIDENCE" in failure["action"],
        "Exact original P2-2E-0 failure semantics",
    )
    require(
        not (FAILED_FORMAL / "audit_summary.json").exists(),
        "Failed P2-2E-0 must not contain PASS marker",
    )
    return {
        "execution_failure_sha256": sha_file(path),
        "inventory": inventory,
        "preserved_read_only": True,
    }


def verify_sources(head):
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned predecessor source changed: " + name,
        )
    require(
        git("rev-parse", "--verify", f"{head}:{SELF}")
        == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "Recovery source must be committed unchanged",
    )


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    require(
        "learn" not in attrs
        and "save" not in attrs
        and "load" not in attrs
        and "train_masked_final_checkpoint" not in attrs
        and "build_masked_perception_ppo_model" not in attrs,
        "Recovery cannot train/load PPO",
    )

    import run_paper2_stage2e0_temporal_partial_observability_design_freeze_v1 as failed
    import paper2_temporal_observation_v1 as temporal

    original = failed.static_contract()
    require(all(original.values()), "Original frozen temporal design contract")

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
        and [name for name in hua if name not in hp]
        == ["width", "delta_width"],
        "Frozen temporal feature pair unchanged",
    )
    require(
        contract["day0"]
        == {
            "delta_q50": 0.0,
            "delta_width": 0.0,
            "tau_since_last_clean_or_episode_start": 0.0,
        }
        and contract["pre_episode_clean_age_imputed"] is False
        and contract["future_information_used"] is False
        and contract["latent_state_used"] is False
        and contract["safety_rule_changed"] is False,
        "Frozen no-leakage/day0 temporal semantics unchanged",
    )

    mask_source = (ROOT / MASK_BASE).read_text(encoding="utf-8")
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
        and "delta_width" not in tokens
        and "tau" not in tokens
        and "belief_lower" in tokens
        and "belief_upper" in tokens
        and "action_safe" in tokens,
        "Safety mask remains q50-belief-only",
    )

    require(
        "incoming_info_semantics" in source
        and "previous_settled_action_day" in source
        and "current_decision_observation_day" in source,
        "Corrected dual-time audit semantics must be explicit",
    )

    return {
        "zero_PPO_training_or_model_load": True,
        "original_temporal_design_contract_unchanged": True,
        "History_Point_feature_contract_unchanged": True,
        "History_UA_feature_contract_unchanged": True,
        "day0_tau_no_leakage_semantics_unchanged": True,
        "q50_only_width_history_free_safety_mask": True,
        "corrected_action_day_observation_day_semantics_present": True,
        "original_failure_evidence_read_only": True,
    }


def run_pair_recovery(masked, temporal, failed, assets, year, tid):
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
            hp_info["day_index"] == hua_info["day_index"] == 0
            and "action_name" not in hp_info
            and "action_name" not in hua_info,
            "Reset info belongs to current decision observation day 0",
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
            # Correct dual-time semantics:
            # - temporal observation is for current decision day == day;
            # - incoming info is reset-info at day 0, otherwise previous
            #   step settlement info whose day_index == day - 1.
            incoming_info_expected_day = 0 if day == 0 else day - 1
            incoming_info_semantics = (
                "current_decision_observation_day"
                if day == 0
                else "previous_settled_action_day"
            )
            require(
                hp_info["day_index"]
                == hua_info["day_index"]
                == incoming_info_expected_day,
                "Correct incoming info action-day semantics",
            )
            if day == 0:
                require(
                    "action_name" not in hp_info
                    and "action_name" not in hua_info,
                    "Reset info has no settled action",
                )
            else:
                require(
                    hp_info.get("action_name")
                    == hua_info.get("action_name")
                    in ("WAIT", "CLEAN"),
                    "Step info describes previous settled action",
                )

            hp_snap = hp._temporal_audit_snapshot()
            hua_snap = hua._temporal_audit_snapshot()
            require(
                hp_snap["decision_index"]
                == hua_snap["decision_index"]
                == day,
                "Current temporal observation decision-day index",
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
            action = failed.scripted_action(day, point_mask)

            previous_q50 = float(hp_obs[0])
            previous_width = float(hua_obs[2])

            hp_next = hp.step(action)
            hua_next = hua.step(action)
            hp_next_obs, _, hp_term, hp_trunc, hp_step_info = hp_next
            hua_next_obs, _, hua_term, hua_trunc, hua_step_info = hua_next

            require(
                hp_step_info["day_index"]
                == hua_step_info["day_index"]
                == day,
                "Returned step info identifies just-settled action day",
            )
            require(
                hp_step_info.get("action_name")
                == hua_step_info.get("action_name")
                == ("CLEAN" if action else "WAIT"),
                "Returned step info action identity",
            )
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

            rows.append(
                {
                    "year": year,
                    "trajectory_id": tid,
                    "current_decision_observation_day": day,
                    "incoming_info_day_index":
                        incoming_info_expected_day,
                    "incoming_info_semantics":
                        incoming_info_semantics,
                    "returned_step_info_day_index":
                        hp_step_info["day_index"],
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

            if not hp_term:
                executed = hp.last_shield_decision["executed_action"]
                expected_age = (
                    1 if executed == 1 else expected_age + 1
                )
                hp_next_snap = hp._temporal_audit_snapshot()
                hua_next_snap = hua._temporal_audit_snapshot()
                require(
                    hp_next_snap["decision_index"]
                    == hua_next_snap["decision_index"]
                    == day + 1,
                    "Next temporal observation decision-day index",
                )

            hp_obs, hp_info = hp_next_obs, hp_step_info
            hua_obs, hua_info = hua_next_obs, hua_step_info

        require(
            hp_term is True
            and hua_term is True
            and np.array_equal(
                hp_obs, np.zeros(5, dtype=np.float32)
            )
            and np.array_equal(
                hua_obs, np.zeros(7, dtype=np.float32)
            ),
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
            and all(
                row["role"] == "DEVELOPMENT"
                for row in ledger.records
            ),
            "DEVELOPMENT-only temporal recovery audit",
        )

    return rows


def run_formal():
    require(
        not STAGE_DIRECTORY.exists(),
        "Exclusive P2-2E-0R directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean worktree required",
    )
    head = git("rev-parse", "HEAD")
    verify_sources(head)
    failure = verify_failure_evidence()
    static = static_contract()

    import paper2_masked_perception_ppo_runner_v1_1 as masked
    import paper2_temporal_observation_v1 as temporal
    import run_paper2_stage2e0_temporal_partial_observability_design_freeze_v1 as failed

    # Reuse the original frozen 6A authority validation and fixed test design.
    stage6a_sha = failed.verify_6a()
    assets = masked.load_assets()
    assets.ensure_perception()

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        rows = []
        for year, tid in failed.FIXED_DEVELOPMENT_SPECS:
            rows.extend(
                run_pair_recovery(
                    masked, temporal, failed, assets, year, tid
                )
            )

        require(
            len(rows)
            == len(failed.FIXED_DEVELOPMENT_SPECS)
            * masked.get_protocol().decision_reward_steps
            == 1460,
            "Exact fixed recovery trace size",
        )

        design = temporal.temporal_contract()
        recovery_summary = {
            "stage": STAGE,
            "scientific_status": STATUS,
            "root_cause":
                "Original P2-2E-0 compared current next-observation day t against info.day_index from the previous settled action day t-1.",
            "recovery_scope":
                "Audit-semantics-only; no temporal observation, perception, safety, reward, physics or PPO change.",
            "corrected_time_semantics": {
                "reset":
                    "o_0 accompanies info.day_index=0 and no settled action",
                "step_t":
                    "step(action_t) settles action day t, returns info.day_index=t and (if nonterminal) next observation o_{t+1}",
                "loop_t_gt_0":
                    "current temporal observation is o_t while incoming info.day_index=t-1 identifies the previous settled action",
            },
            "temporal_observation_contract": design,
            "original_failure_evidence": failure,
            "P2_2D_6A_audit_summary_sha256": stage6a_sha,
            "future_experiment_order": [
                "P2-2E-1 History-Point DEVELOPMENT",
                "P2-2E-2 History-UA DEVELOPMENT",
            ],
            "stopping_rule":
                "If compact History-UA still fails to recover active decision behavior, stop adding observation candidates in current Paper2 scope.",
        }

        gates = {
            "original_P2_2E_0_failure_evidence_preserved": True,
            "original_temporal_design_sources_unchanged": True,
            "original_temporal_design_static_contract": all(static.values()),
            "corrected_dual_time_semantics_runtime_exact": True,
            "fixed_four_DEVELOPMENT_episode_pairs":
                len(failed.FIXED_DEVELOPMENT_SPECS) == 4,
            "full_365_day_recovery_trace_each":
                len(rows) == 1460,
            "Point_UA_shared_features_exact_runtime": True,
            "delta_and_tau_causal_recurrence_exact_runtime": True,
            "q50_only_safety_mask_identical_runtime": True,
            "PPO_training_runs_zero": True,
            "PPO_model_loads_zero": True,
            "protected_access_zero": True,
        }
        require(all(gates.values()), "All P2-2E-0R recovery gates")

        provenance = {
            "HEAD": head,
            "checkpoint": CHECKPOINT,
            "original_failure_evidence": failure,
            "P2_2D_6A_audit_summary_sha256": stage6a_sha,
            "pinned_source_blobs": PINNED_BLOBS,
            "recovery_source_sha256": sha_file(ROOT / SELF),
            "fixed_DEVELOPMENT_specs":
                [list(spec) for spec in failed.FIXED_DEVELOPMENT_SPECS],
            "scripted_clean_days":
                list(failed.SCRIPTED_CLEAN_DAYS),
        }

        content = {
            "recovery_summary.json": encode(recovery_summary),
            "corrected_construction_trace.csv": csv_bytes(rows),
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
            verify_failure_evidence()
            == failure,
            "Original failed evidence remains unchanged",
        )
        require(
            failed.verify_6a() == stage6a_sha,
            "P2-2D-6A evidence remains unchanged",
        )
        require(
            git("rev-parse", "HEAD") == head
            and git("status", "--porcelain", "--untracked-files=all") == "",
            "Source HEAD/worktree unchanged",
        )
        require(
            json.loads(output_hashes)["hashes"]
            == {
                name: sha_file(OUTPUT / name)
                for name in hashes
            },
            "Recovery output hashes exact",
        )
        require(
            not (OUTPUT / "audit_summary.json").exists(),
            "Recovery audit summary published last",
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
            "corrected_construction_trace_rows": 1460,
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
            "temporal_observation_design_changed": False,
            "safety_mask_or_shield_changed": False,
            "original_failure_execution_sha256":
                failure["execution_failure_sha256"],
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
                            "STOP; PRESERVE ORIGINAL AND RECOVERY EVIDENCE; NO RETRY/OBSERVATION-REDESIGN/PPO-TRAINING/TUNING",
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
