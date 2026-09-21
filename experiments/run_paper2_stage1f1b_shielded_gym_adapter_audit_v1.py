
"""P2-1F-1B-v1: frozen shielded Gym-adapter integration audit.

ENGINEERING/SEMANTIC INTEGRATION ONLY. No PPO training is performed.
The stage verifies that the passed q50-only shield can sit between unchanged
Point/UA observations and the frozen Paper2 core without changing PPO
hyperparameters, public info, reward algebra, schedule semantics, perception,
dynamics, or cleaning mechanics.

If PASS, the next authorized stage is a fresh Shielded Point-PPO smoke run.
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
SELF = "experiments/run_paper2_stage1f1b_shielded_gym_adapter_audit_v1.py"
RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
BASE_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"
STAGE1FA = "experiments/run_paper2_stage1f1a_q50_set_membership_shield_feasibility_audit_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
AUTHORITY = BASE / "p2_1f_1a_q50_set_membership_shield_feasibility_audit_v1/formal"
STAGE = BASE / "p2_1f_1b_shielded_gym_adapter_audit_v1"
OUTPUT = STAGE / "formal"

CHECKPOINT = "0823e7688919320c7a6ef550e6de7ca181dc60a2"
PINNED_BLOBS = {
    BASE_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
    STAGE1FA: "a1da2d4e2b455f8dcd60263f7ea89d073b679de1",
}
PAIRING_TIDS = (0, 149, 299)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=30
    ).stdout.strip()


def write_bytes(name, data):
    path = OUTPUT / name
    with path.open("xb") as f:
        f.write(data)
    require(path.read_bytes() == data, "Exact output readback: " + name)


def write_json(name, value):
    write_bytes(
        name,
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )


def write_csv(name, rows):
    require(rows, "Nonempty CSV: " + name)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(name, stream.getvalue().encode())


def verify_predecessor(head):
    audit = json.loads((AUTHORITY / "audit_summary.json").read_bytes())
    summary = json.loads((AUTHORITY / "shield_feasibility_summary.json").read_bytes())
    primary = json.loads((AUTHORITY / "primary_population_summary.json").read_bytes())
    require(
        audit["stage"] == "P2-1F-1A-v1"
        and audit["HEAD"] == CHECKPOINT
        and audit["stage_pass"] is True
        and audit["support_compatible_shield_feasibility_pass"] is True
        and audit["scientific_status"] == "Q50_SET_MEMBERSHIP_SHIELD_FEASIBILITY_PASS"
        and audit["failed_gates"] == []
        and audit["PPO_training_runs"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-1F-1A predecessor",
    )
    require(
        summary["support_compatible_shield_feasibility_pass"] is True
        and summary["failed_gates"] == []
        and all(summary["gates"].values())
        and primary["episodes"] == 1200
        and primary["steps"] == 438000
        and primary["safety_filtered_decisions"] == 436800
        and primary["interventions"] == 82727
        and primary["free_waits"] == 355273,
        "Exact frozen P2-1F-1A scientific regression",
    )
    for name, digest in audit["output_sha256"].items():
        require(sha(AUTHORITY / name) == digest, "P2-1F-1A output hash: " + name)

    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Pinned source unchanged: " + name,
        )
    return audit


def function(tree, name):
    return next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def class_node(tree, name):
    return next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == name
    )


def static_contract():
    runner_src = (ROOT / RUNNER).read_text(encoding="utf-8")
    base_src = (ROOT / BASE_RUNNER).read_text(encoding="utf-8")
    rtree, btree = ast.parse(runner_src), ast.parse(base_src)

    wrapper = class_node(rtree, "Q50ShieldedPerceptionEnv")
    step = next(
        n for n in wrapper.body
        if isinstance(n, ast.FunctionDef) and n.name == "step"
    )
    step_text = ast.unparse(step)
    require("_audit_snapshot" not in step_text and "width" not in step_text,
            "No hidden L/UA width in shield step")
    require("filter_action" in step_text and "predict_after_action" in step_text,
            "Shield path explicit")
    require(step_text.index("filter_action") < step_text.index("self.inner.step"),
            "Shield decides before frozen core")
    require(
        "settlement" not in step_text
        and "dry_next_samples" not in step_text
        and "rain_next_samples" not in step_text,
        "No copied reward/dynamics",
    )

    old = function(btree, "build_perception_ppo_model")
    new = function(rtree, "build_shielded_perception_ppo_model")

    def constructor_body(func):
        start = next(
            i for i, node in enumerate(func.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Tuple)
            and [x.id for x in node.targets[0].elts] == ["common", "architecture"]
        )
        return ast.dump(ast.Module(body=func.body[start:-1], type_ignores=[]))

    require(
        constructor_body(old) == constructor_body(new),
        "Frozen PPO constructor body structurally identical",
    )

    source = (ROOT / SELF).read_text(encoding="utf-8")
    stage_tree = ast.parse(source)
    protected_role_calls = []
    for call in (n for n in ast.walk(stage_tree) if isinstance(n, ast.Call)):
        values = list(call.args) + [kw.value for kw in call.keywords]
        for value in values:
            if (
                isinstance(value, ast.Constant)
                and value.value in ("RANDOM_TEST", "SEALED_DATES")
            ):
                protected_role_calls.append(value.value)
    require(
        not protected_role_calls,
        "No protected-role constructor/call literal",
    )
    banned_attrs = {
        "learn", "train_final_checkpoint", "sample_one",
        "perception_substream_seed", "settlement",
        "dry_next_samples", "rain_next_samples",
    }
    for node in ast.walk(stage_tree):
        if isinstance(node, ast.Attribute):
            require(node.attr not in banned_attrs,
                    "No training/scientific duplication: " + node.attr)

    return {
        "shield_before_core_step": True,
        "no_hidden_L_or_width_in_wrapper_step": True,
        "reward_and_dynamics_delegated": True,
        "PPO_constructor_equivalence": True,
        "no_PPO_training_call_in_stage": True,
        "no_protected_role_code_path": True,
    }


def run_episode(env, proposed_rule):
    obs, info = env.reset()
    public_keys = set(info)
    rows = []
    for day in range(365):
        proposed = int(proposed_rule(day))
        require(proposed in (0, 1), "External proposal")
        obs, reward, terminated, truncated, info = env.step(proposed)
        decision = env.last_shield_decision
        require(
            decision is not None
            and decision["proposed_action"] == proposed
            and np.isfinite(reward)
            and not truncated
            and terminated == (day == 364),
            "Shielded adapter step semantics",
        )
        public_keys.update(info)
        rows.append(
            {
                "day_index": day,
                "proposed_action": proposed,
                "executed_action": decision["executed_action"],
                "intervention": decision["intervention"],
                "safety_constraint_applicable": decision["safety_constraint_applicable"],
                "terminated": terminated,
            }
        )
    record = env._shield_audit_snapshot()["episode_records"][-1]
    require(
        record["steps"] == 365
        and record["safety_filtered_decisions"] == 364
        and record["terminal_passthrough_count"] == 1
        and record["completed"] is True,
        "Complete adapter episode accounting",
    )
    return rows, record, public_keys


def scheduled_integration(runner, assets, ledger):
    p = runner.get_protocol()
    c = runner.candidate(runner.inherited_config_id())
    seed = p.seeds.development[0]
    env = runner.build_shielded_scheduled_perception_env(
        assets, seed, ledger, c.name, "Point"
    )
    patterns = (
        ("WAIT", lambda day: 0),
        ("CLEAN", lambda day: 1),
        ("ALTERNATING", lambda day: day % 2),
    )
    summaries = []
    public_schema = None
    try:
        for name, rule in patterns:
            rows, record, keys = run_episode(env, rule)
            if public_schema is None:
                public_schema = keys
            require(keys == public_schema, "Stable public info across scheduled episodes")
            if name == "CLEAN":
                require(
                    record["proposed_clean_count"] == 365
                    and record["executed_clean_count"] == 365
                    and record["interventions"] == 0,
                    "Proposed CLEAN is never overridden",
                )
            if name == "WAIT":
                require(
                    record["interventions"] > 0
                    and record["free_wait_count"] > 0,
                    "WAIT integration exercises pass/intervene branches",
                )
            summaries.append({"pattern": name, **record})
        require(
            env.total_steps == 3 * 365
            and env.episodes_completed == 3
            and env.total_safety_filtered_decisions == 3 * 364
            and all(r["completed"] for r in env.episode_records),
            "Scheduled wrapper multi-episode accounting",
        )
        return summaries, sorted(public_schema)
    finally:
        env.close()


def ppo_constructor_integration(runner, assets, ledger):
    p = runner.get_protocol()
    c = runner.candidate(runner.inherited_config_id())
    seed = p.seeds.development[0]
    env = runner.build_shielded_scheduled_perception_env(
        assets, seed, ledger, c.name, "Point"
    )
    try:
        bundle = runner.build_shielded_perception_ppo_model(env, c.name, seed)
        require(
            bundle.model.num_timesteps == 0
            and not bundle.training_finished
            and runner.verify_model_contract(bundle.model, c.name, seed)
            and not env.started,
            "PPO construction only; no training/reset",
        )
        return {
            "constructed": True,
            "num_timesteps": int(bundle.model.num_timesteps),
            "training_finished": bool(bundle.training_finished),
            "observation_shape": list(env.observation_space.shape),
            "action_n": int(env.action_space.n),
            "config_id": c.name,
            "parent_seed": int(seed),
            "PPO_training_runs": 0,
        }
    finally:
        env.close()


def pairing_integration(runner, assets, ledger):
    part = runner.partition("DEVELOPMENT")
    rows = []
    for year in part.years:
        for tid in PAIRING_TIDS:
            point = runner.build_shielded_perception_eval_env(
                assets, year, tid, ledger, "Point"
            )
            ua = runner.build_shielded_perception_eval_env(
                assets, year, tid, ledger, "UA"
            )
            try:
                po, pi = point.reset()
                uo, ui = ua.reset()
                require(
                    np.array_equal(po, uo[[0, 2, 3]]) and set(pi) == set(ui),
                    "Paired reset shared q50/season and public info",
                )
                interventions = 0
                for day in range(365):
                    proposed = 0 if day % 3 else 1
                    pr = point.step(proposed)
                    ur = ua.step(proposed)
                    pd, ud = point.last_shield_decision, ua.last_shield_decision
                    require(
                        pd == ud
                        and pr[1:4] == ur[1:4]
                        and np.array_equal(pr[0], ur[0][[0, 2, 3]])
                        and np.isfinite(ur[0]).all(),
                        "Point/UA exact common shield and transition",
                    )
                    interventions += int(pd["intervention"])
                    po, uo = pr[0], ur[0]
                rows.append(
                    {
                        "year": year,
                        "trajectory_id": int(tid),
                        "steps": 365,
                        "interventions": interventions,
                        "point_ua_exact": True,
                        "width_used_by_shield": False,
                    }
                )
            finally:
                point.close()
                ua.close()
    return rows


def final_integrity(head, sources, blobs):
    require(
        git("rev-parse", "HEAD") == head
        and git("status", "--porcelain", "--untracked-files=all") == "",
        "HEAD/tree unchanged",
    )
    require(all(sha(ROOT / p) == h for p, h in sources.items()), "New sources unchanged")
    require(
        all(
            git("rev-parse", f"{head}:{p}")
            == git("rev-parse", ":" + p)
            == git("hash-object", "--path=" + p, p)
            == blob
            for p, blob in blobs.items()
        ),
        "New source blobs unchanged",
    )


def run_formal():
    require(not STAGE.exists(), "Immutable stage; no overwrite/resume/retry")
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean tree required",
    )
    head = git("rev-parse", "HEAD")
    verify_predecessor(head)
    sources = {p: sha(ROOT / p) for p in (SELF, RUNNER)}
    blobs = {p: git("rev-parse", f"{head}:{p}") for p in (SELF, RUNNER)}
    final_integrity(head, sources, blobs)

    import paper2_shielded_perception_ppo_runner_v1 as runner

    static = static_contract()
    authority = runner.read_shield_authority()
    assets = runner.load_assets()
    assets.ensure_perception()
    ledger = runner.AccessLedger()

    STAGE.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    try:
        construction = ppo_constructor_integration(runner, assets, ledger)
        scheduled_rows, public_schema = scheduled_integration(
            runner, assets, ledger
        )
        pairing_rows = pairing_integration(runner, assets, ledger)

        require(
            ledger.heldout_ppo_trajectory_access_count == 0
            and ledger.formal_perception_seed_access_count == 0
            and ledger.denied_accesses == 0,
            "Development/training roles only",
        )

        gates = {
            "predecessor_shield_authority_exact":
                authority.feasibility_audit["support_compatible_shield_feasibility_pass"] is True,
            "static_wrapper_contract": all(static.values()),
            "PPO_constructor_without_training":
                construction["constructed"] and construction["num_timesteps"] == 0,
            "scheduled_multi_episode_adapter":
                len(scheduled_rows) == 3 and all(r["completed"] for r in scheduled_rows),
            "WAIT_pass_and_intervention_branches":
                scheduled_rows[0]["interventions"] > 0
                and scheduled_rows[0]["free_wait_count"] > 0,
            "CLEAN_never_overridden":
                scheduled_rows[1]["interventions"] == 0
                and scheduled_rows[1]["executed_clean_count"] == 365,
            "terminal_day_passthrough":
                all(r["terminal_passthrough_count"] == 1 for r in scheduled_rows),
            "public_info_schema_unchanged":
                set(public_schema) <= set(runner.get_perception_contract().public_info_keys),
            "point_ua_common_shield_exact":
                all(r["point_ua_exact"] and not r["width_used_by_shield"] for r in pairing_rows),
            "zero_formal_access":
                ledger.heldout_ppo_trajectory_access_count == 0
                and ledger.formal_perception_seed_access_count == 0
                and ledger.denied_accesses == 0,
        }
        failed = [name for name, value in gates.items() if not value]

        write_json(
            "protocol_manifest.json",
            {
                "stage": "P2-1F-1B-v1",
                "role": "SHIELDED GYM ADAPTER INTEGRATION; NO PPO TRAINING",
                "policy_action_semantics":
                    "PPO action is a proposal; nonterminal environment transition includes the frozen q50-only shield mapping to executed action",
                "policy_observation_Point": ["q50", "sin_DOY", "cos_DOY"],
                "policy_observation_UA": ["q50", "width", "sin_DOY", "cos_DOY"],
                "shield_inputs": ["q50_float32_history", "executed_action_history"],
                "shield_forbidden": [
                    "current_true_L", "UA_width", "rain_flag", "future_weather"
                ],
                "reward_shaping": False,
                "terminal_day_semantics":
                    "day 364 proposal passes unchanged because no t->t+1 transition exists",
                "PPO_hyperparameter_reselection": False,
                "PPO_training_runs": 0,
                "formal_RL_access_count": 0,
                "formal_perception_seed_access_count": 0,
                "RANDOM_TEST_access_count": 0,
                "SEALED_DATES_access_count": 0,
                "next_if_pass": "Fresh Shielded Point-PPO engineering smoke",
            },
        )
        write_json("static_contract.json", static)
        write_json("ppo_constructor_integration.json", construction)
        write_csv("scheduled_adapter_regression.csv", scheduled_rows)
        write_csv("point_ua_adapter_pairing.csv", pairing_rows)
        write_json(
            "adapter_integration_summary.json",
            {
                "adapter_integration_pass": not failed,
                "gates": gates,
                "failed_gates": failed,
                "public_info_keys": public_schema,
                "scheduled": scheduled_rows,
                "pairing": pairing_rows,
                "PPO_training_runs": 0,
                "claim_boundary":
                    "Engineering/semantic adapter integration only; no Point/UA performance claim",
            },
        )

        final_integrity(head, sources, blobs)
        expected = {
            "protocol_manifest.json",
            "static_contract.json",
            "ppo_constructor_integration.json",
            "scheduled_adapter_regression.csv",
            "point_ua_adapter_pairing.csv",
            "adapter_integration_summary.json",
        }
        require({p.name for p in OUTPUT.iterdir()} == expected, "Exact pre-audit inventory")
        hashes = {name: sha(OUTPUT / name) for name in sorted(expected)}
        write_json(
            "output_hashes.json",
            {
                "hashes": hashes,
                "exclusions": {
                    "output_hashes.json": "Pinned by final audit",
                    "audit_summary.json": "Published last",
                },
            },
        )
        hashes["output_hashes.json"] = sha(OUTPUT / "output_hashes.json")
        write_json(
            "audit_summary.json",
            {
                "stage": "P2-1F-1B-v1",
                "stage_pass": not failed,
                "scientific_status": (
                    "SHIELDED_GYM_ADAPTER_INTEGRATION_PASS_FROZEN"
                    if not failed
                    else "SHIELDED_GYM_ADAPTER_INTEGRATION_FAIL"
                ),
                "declaration": (
                    "Q50-ONLY SET-MEMBERSHIP SHIELD IS INTEGRATED AS A FROZEN "
                    "POINT/UA GYM ACTION FILTER WITH UNCHANGED POLICY OBSERVATIONS, "
                    "PUBLIC INFO, PPO CONFIGURATION, REWARD AND PHYSICS; NO PPO TRAINING"
                    if not failed else
                    "SHIELDED GYM ADAPTER INTEGRATION FAILED; DO NOT START POINT/UA PPO"
                ),
                "failed_gates": failed,
                "HEAD": head,
                "PPO_training_runs": 0,
                "formal_RL_access_count": 0,
                "formal_perception_seed_access_count": 0,
                "RANDOM_TEST_access_count": 0,
                "SEALED_DATES_access_count": 0,
                "output_sha256": hashes,
                "audit_summary_published_last": True,
            },
        )
        return json.loads((OUTPUT / "audit_summary.json").read_bytes())
    except Exception as exc:
        if not (OUTPUT / "audit_summary.json").exists():
            write_json(
                "execution_failure.json",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "action": "STOP; NO PPO TRAINING; PRESERVE ADAPTER EVIDENCE",
                },
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
