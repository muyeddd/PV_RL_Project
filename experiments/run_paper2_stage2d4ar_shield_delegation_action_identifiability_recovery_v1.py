"""P2-2D-4AR-v1: DEVELOPMENT-only mechanism recovery; import is inert.

All 600 episodes receive first-forced-CLEAN delegation alias witnesses.
Safe anchors depend ONLY on first_forced_day d_f: none for 0, PRE_FORCE=0
for 1, otherwise MID_PRE_FORCE=(d_f-1)//2 and PRE_FORCE=d_f-1.
Eligibility is fixed for the full population before any paired outcome is read.
Counterfactuals use always-WAIT + the frozen q50 shield through day 364.
The CLEAN-minus-WAIT remaining cost gap is a paired fixed-continuation
counterfactual, NOT an optimal Q-function or Q(CLEAN)-Q(WAIT).
Economic results and masking-route support are descriptive, never PASS gates.
No PPO training/loading, UA training, protected evaluation, or parameter change.
Only --mode formal executes; an existing stage directory forbids any retry.
The failed 4A evidence is read-only and pinned before and after execution.
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
SELF = "experiments/run_paper2_stage2d4ar_shield_delegation_action_identifiability_recovery_v1.py"
FAILED_SOURCE = "experiments/run_paper2_stage2d4a_shield_delegation_action_identifiability_audit_v1.py"
RECOVERY_SOURCE = "experiments/run_paper2_stage2d3r_shielded_point_ppo_development_recovery_v1.py"
RUNNER = "experiments/paper2_shielded_perception_ppo_runner_v1.py"
BASE_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
SHIELD = "experiments/paper2_q50_set_membership_shield_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
PROTOCOL = "experiments/paper2_ppo_protocol_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
PREDECESSOR = BASE / "p2_2d_3r_shielded_point_ppo_development_recovery_v1/formal"
FAILED_DIRECTORY = BASE / "p2_2d_4a_shield_delegation_action_identifiability_audit_v1/formal"
FAILED_HEAD = "9404415bb03036b6c80297ac19e312928fc21558"
FAILED_SHA256 = "8c58deaaa82cc4fa85c209c9fc2213f42742cf42b44c4e526aad4d81a7c481b5"
FAILED_SOURCE_BLOB = "90c8a7b7c45615601a841d8457c89a2c0a14308b"
STAGE_DIRECTORY = BASE / "p2_2d_4ar_shield_delegation_action_identifiability_recovery_v1"
OUTPUT = STAGE_DIRECTORY / "formal"

PREDECESSOR_HEAD = "bebde79b87c43b8faf7fda2a399522ad5e8f3855"
PINNED_BLOBS = {
    RECOVERY_SOURCE: "b7571d873c35474bec4ffcd61b0621f0c76f5f94",
    RUNNER: "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    BASE_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    SHIELD: "db78b7be651fae6c15231b3d8ac55cae822ab409",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    PROTOCOL: "0dedab12015978a01a41aea02f065a470be39094",
}

STAGE = "P2-2D-4AR-v1"
STATUS = "POSTHOC_SHIELD_DELEGATION_ACTION_IDENTIFIABILITY_RECOVERY_COMPLETE"
ANCHOR_LABELS = ("MID_PRE_FORCE", "PRE_FORCE")
WAIT = 0
CLEAN = 1


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
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def csv_bytes(rows, fieldnames=None):
    require((rows or fieldnames) and all(set(row) == set(rows[0]) for row in rows),
            "Nonempty consistent CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_exclusive(path, data):
    path = Path(path)
    require(path.parent.resolve() == OUTPUT.resolve(), "Writes only to recovery formal directory")
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, "Exact readback: " + path.name)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_predecessor(head):
    audit_path = PREDECESSOR / "audit_summary.json"
    dev_path = PREDECESSOR / "development_summary.json"
    episode_path = PREDECESSOR / "development_episode_metrics.csv"
    baseline_path = PREDECESSOR / "shield_only_development_metrics.csv"
    require(audit_path.exists() and dev_path.exists()
            and episode_path.exists() and baseline_path.exists(),
            "P2-2D-3R predecessor outputs exist")

    audit = json.loads(audit_path.read_bytes())
    dev = json.loads(dev_path.read_bytes())
    require(
        audit["stage"] == "P2-2D-3R-v1"
        and audit["HEAD"] == PREDECESSOR_HEAD
        and audit["stage_pass"] is True
        and audit["scientific_status"] == "SHIELDED_POINT_PPO_DEVELOPMENT_RECOVERY_PASS"
        and audit["failed_gates"] == []
        and audit["new_PPO_training_runs"] == 1
        and audit["recovered_predecessor_training_runs"] == 1
        and audit["cumulative_development_model_count"] == 2
        and audit["Point_models"] == 2
        and audit["UA_training_runs"] == 0
        and audit["recovered_seed_retrained"] is False
        and audit["development_evaluation_episode_count"] == 1200
        and audit["shield_only_baseline_episode_count"] == 600
        and audit["final_seed_training_runs"] == 0
        and audit["heldout_ppo_trajectory_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-2D-3R predecessor",
    )
    for name, digest in audit["output_sha256"].items():
        require(sha_file(PREDECESSOR / name) == digest,
                "P2-2D-3R output hash: " + name)

    require(
        dev["independent_training_seed_count"] == 2
        and dev["evaluation_episodes_per_seed"] == 600
        and dev["shield_only_baseline"]["episode_count"] == 600
        and dev["readiness_diagnostics_descriptive_only"]["each_seed_proposes_clean_somewhere"] is False
        and dev["readiness_diagnostics_descriptive_only"]["each_seed_mean_cost_below_shield_only"] is False
        and dev["readiness_diagnostics_descriptive_only"]["each_seed_mean_intervention_fraction_below_shield_only"] is False,
        "Exact predecessor collapse summary",
    )

    git("merge-base", "--is-ancestor", PREDECESSOR_HEAD, head)
    for name, blob in PINNED_BLOBS.items():
        require(git("rev-parse", f"{PREDECESSOR_HEAD}:{name}") == blob
                and git("rev-parse", f"{head}:{name}") == blob,
                "Pinned predecessor source unchanged: " + name)
    return audit, dev


def verify_failed_predecessor(head):
    require(FAILED_DIRECTORY.is_dir()
            and {p.name for p in FAILED_DIRECTORY.iterdir()} == {"execution_failure.json"},
            "Failed 4A directory contains only original execution_failure.json")
    path = FAILED_DIRECTORY / "execution_failure.json"
    require(path.is_file() and not path.is_symlink(), "Original regular failure evidence")
    evidence = json.loads(path.read_bytes())
    expected = {
        "HEAD": FAILED_HEAD, "stage": "P2-2D-4A-v1", "stage_pass": False,
        "error_type": "RuntimeError",
        "message": "Two distinct safe anchors required before first forced CLEAN",
        "action": "STOP; PRESERVE EVIDENCE; NO PPO TRAINING, NO MASKED-PPO START, NO TUNING/RETRY",
    }
    require(evidence == expected and evidence["stage_pass"] is False,
            "Exact frozen 4A failure content")
    require(sha_file(path) == FAILED_SHA256, "Original 4A failure SHA256 unchanged")
    git("merge-base", "--is-ancestor", FAILED_HEAD, head)
    blobs = {FAILED_SOURCE: FAILED_SOURCE_BLOB, **PINNED_BLOBS}
    for name, blob in blobs.items():
        require(git("rev-parse", f"{FAILED_HEAD}:{name}") == blob
                and git("rev-parse", f"{head}:{name}") == blob
                and git("hash-object", f"--path={name}", name) == blob
                and git("rev-parse", f":{name}") == blob,
                "Frozen predecessor Git/worktree/index blob: " + name)
    return {
        "failure_path": path.relative_to(ROOT).as_posix(),
        "execution_failure_sha256": FAILED_SHA256,
        "failure_evidence": evidence, "recovery_HEAD": head,
        "frozen_source_git_blobs": blobs,
        "read_only": True,
    }


def anchor_eligibility(first_forced_day):
    """Pure structural rule: no cost, reward, latent state or outcome input."""
    require(type(first_forced_day) is int and 0 <= first_forced_day < 364,
            "First forced day is a nonterminal DEVELOPMENT decision")
    if first_forced_day == 0:
        return "ZERO_PRE_FORCE_SAFE_ANCHOR", {}
    if first_forced_day == 1:
        return "ONE_PRE_FORCE_SAFE_ANCHOR", {"PRE_FORCE": 0}
    anchors = {"MID_PRE_FORCE": (first_forced_day - 1) // 2,
               "PRE_FORCE": first_forced_day - 1}
    require(len(set(anchors.values())) == 2
            and all(0 <= d < first_forced_day for d in anchors.values()),
            "Distinct structural pre-force anchors")
    return "TWO_PRE_FORCE_SAFE_ANCHORS", anchors


def eligibility_summary(baseline_rows):
    counts = {i: sum(r["available_safe_action_anchor_count"] == i
                     for r in baseline_rows) for i in range(3)}
    total = len(baseline_rows)
    require(total == 600 and sum(counts.values()) == total, "Full eligibility coverage")
    available = sum(r["available_safe_action_anchor_count"] for r in baseline_rows)
    require(available == counts[1] + 2 * counts[2], "Mechanical anchor count identity")
    return {
        "total_development_episodes": total,
        "zero_anchor_episode_count": counts[0],
        "one_anchor_episode_count": counts[1],
        "two_anchor_episode_count": counts[2],
        "zero_anchor_fraction": counts[0] / total,
        "one_anchor_fraction": counts[1] / total,
        "two_anchor_fraction": counts[2] / total,
        "total_available_safe_action_anchors": available,
        "episodes_with_at_least_one_anchor": counts[1] + counts[2],
        "anchor_eligible_episode_fraction": (counts[1] + counts[2]) / total,
        "eligibility_input_only": "first_forced_day",
    }


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(
        "train_final_checkpoint" not in attrs
        and "build_shielded_perception_ppo_model" not in attrs
        and "learn" not in attrs
        and "save" not in attrs
        and "load" not in attrs,
        "No PPO construction/training/load/save",
    )
    require("build_shielded_perception_eval_env" in attrs,
            "Frozen shielded evaluation environment used")
    require("build_scheduled_perception_env" not in attrs
            and "build_perception_ppo_model" not in attrs,
            "No unshielded/training runtime")

    protected_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        values = list(node.args) + [kw.value for kw in node.keywords]
        for value in values:
            if (isinstance(value, ast.Constant)
                    and value.value in ("FORMAL HELD-OUT", "RANDOM_TEST",
                                        "SEALED_DATES", "UA", "TRAINING")):
                protected_calls.append(value.value)
    require(not protected_calls, "No protected/UA/TRAINING runtime call literal")
    runner_calls = {n.func.attr for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "runner"}
    require(runner_calls == {"AccessLedger", "build_shielded_perception_eval_env",
                             "get_protocol", "load_assets", "partition"},
            "Only explicitly allowlisted frozen runner calls")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "partition":
                require(len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "DEVELOPMENT", "DEVELOPMENT only")
            if node.func.attr == "build_shielded_perception_eval_env":
                require(isinstance(node.args[-1], ast.Constant)
                        and node.args[-1].value == "Point", "Point-only eval runtime")
    for day in range(364):
        category, anchors = anchor_eligibility(day)
        expected = 0 if day == 0 else 1 if day == 1 else 2
        require(len(anchors) == expected and len(set(anchors.values())) == expected
                and all(0 <= d < day for d in anchors.values()), "Adaptive eligibility contract")
        require(category == ("ZERO_PRE_FORCE_SAFE_ANCHOR" if day == 0
                             else "ONE_PRE_FORCE_SAFE_ANCHOR" if day == 1
                             else "TWO_PRE_FORCE_SAFE_ANCHORS"), "Exact eligibility labels")
        if day == 1:
            require(anchors == {"PRE_FORCE": 0}, "Sole day-zero anchor")
        if day >= 2:
            require(anchors == {"MID_PRE_FORCE": (day - 1) // 2, "PRE_FORCE": day - 1},
                    "Exact floor((d_f - 1)/2) rule")
    writer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "write_exclusive")
    require("path.parent.resolve() == OUTPUT.resolve()" in ast.unparse(writer),
            "All output writes restricted to recovery")
    opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "open"]
    require(all(n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value in ("r", "rb", "xb") for n in opens),
            "Read-only inputs and exclusive recovery output")
    require(FAILED_DIRECTORY != OUTPUT and "verify_failed_predecessor" in source,
            "Read-only failed predecessor provenance contract")
    return {
        "no_PPO_training_or_loading": True,
        "shielded_DEVELOPMENT_eval_runtime_only": True,
        "no_unshielded_or_training_runtime": True,
        "no_formal_protected_UA_or_training_call": True,
        "failed_4A_evidence_read_only": True,
        "adaptive_anchor_eligibility_protocol_present": True,
    }


def predecessor_collapse_audit(atol):
    episode_rows = read_csv(PREDECESSOR / "development_episode_metrics.csv")
    baseline_rows = read_csv(PREDECESSOR / "shield_only_development_metrics.csv")
    require(len(episode_rows) == 1200, "1200 predecessor PPO eval rows")
    require(len(baseline_rows) == 600, "600 predecessor baseline rows")
    baseline = {(row["year"], int(row["trajectory_id"])): row for row in baseline_rows}
    require(len(baseline) == 600, "Unique baseline DEVELOPMENT specs")
    seeds = sorted({int(row["parent_rl_seed"]) for row in episode_rows})
    require(seeds == [510001, 510002], "Exact two Point DEVELOPMENT seeds")

    for seed in seeds:
        selected = [r for r in episode_rows if int(r["parent_rl_seed"]) == seed]
        keys = {(r["year"], int(r["trajectory_id"])) for r in selected}
        require(len(selected) == 600 and keys == set(baseline),
                "Each seed has all 600 unique DEVELOPMENT episodes")
    exact_rows = 0
    for row in episode_rows:
        ref = baseline[(row["year"], int(row["trajectory_id"]))]
        require(row["deterministic_action"] == "True"
                and int(row["step_count"]) == 365
                and int(row["natural_transition_count"]) == 364
                and row["environment_root"] == ref["environment_root"],
                "Deterministic complete predecessor evaluation with matching root")
        require(int(row["N_clean_proposed"]) == 0,
                "Predecessor deterministic Point proposes no CLEAN")
        require(int(row["N_clean"]) == int(row["shield_interventions"]),
                "All predecessor executed CLEAN is shield-forced")
        require(
            int(row["N_clean"]) == int(ref["N_clean"])
            and int(row["shield_interventions"]) == int(ref["shield_interventions"])
            and abs(float(row["J_total"]) - float(ref["J_total"])) <= atol,
            "Point predecessor equals shield-only baseline per episode",
        )
        exact_rows += 1
    return {
        "point_eval_rows": len(episode_rows),
        "shield_only_rows": len(baseline_rows),
        "development_seeds": seeds,
        "deterministic_episodes_per_seed": {str(seed): 600 for seed in seeds},
        "rows_exactly_matching_shield_only": exact_rows,
        "all_deterministic_proposed_clean_zero": True,
        "all_executed_clean_is_shield_forced": True,
        "all_Point_rows_match_shield_only": exact_rows == 1200,
    }


def require_development_ledger(ledger):
    require(len(ledger.records) == 1 and ledger.denied_accesses == 0
            and all(r["role"] == "DEVELOPMENT" and r["observation_mode"] == "Point"
                    for r in ledger.records)
            and ledger.heldout_ppo_trajectory_access_count == 0
            and ledger.formal_perception_seed_access_count == 0,
            "Only authorized DEVELOPMENT Point construction; no training runtime")


def core_snapshot(env):
    return env.inner._audit_snapshot()


def last_cost(env):
    parts = core_snapshot(env)["last_reward_components"]
    require(parts is not None, "Last reward components exist")
    return float(parts["total_cost"])


def exact_core_identity(a, b):
    keys = (
        "year", "trajectory_id", "day_index", "date", "L_pre", "L_post_last",
        "environment_root", "perception_seed", "transition_count", "clean_count",
        "episode_return", "terminated", "needs_reset", "perception_query_count",
        "indices_sha256", "full_bank_sha256", "indices_read_only",
    )
    require(all(a[key] == b[key] for key in keys),
            "Exact paired core state identity")
    require(a["last_reward_components"] == b["last_reward_components"],
            "Exact paired last reward/transition identity")


def step_and_audit(env, proposal):
    obs, reward, terminated, truncated, info = env.step(proposal)
    require(np.isfinite(reward) and not truncated,
            "Finite nontruncated paired transition")
    decision = env.last_shield_decision
    require(decision is not None, "Shield decision recorded")
    return obs, float(reward), bool(terminated), info, decision, core_snapshot(env)


def env_pair(runner, assets, year, tid):
    ledger_a = runner.AccessLedger()
    ledger_b = runner.AccessLedger()
    env_a = runner.build_shielded_perception_eval_env(
        assets, year, tid, ledger_a, "Point")
    env_b = runner.build_shielded_perception_eval_env(
        assets, year, tid, ledger_b, "Point")
    require_development_ledger(ledger_a)
    require_development_ledger(ledger_b)
    return env_a, env_b, ledger_a, ledger_b


def replay_prefix_identical(env_a, env_b, anchor_day):
    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()
    require(np.array_equal(obs_a, obs_b) and info_a == info_b,
            "Exact paired reset")
    exact_core_identity(core_snapshot(env_a), core_snapshot(env_b))
    for _ in range(anchor_day):
        ra = step_and_audit(env_a, WAIT)
        rb = step_and_audit(env_b, WAIT)
        require(np.array_equal(ra[0], rb[0])
                and ra[1] == rb[1] and ra[2] == rb[2]
                and ra[3] == rb[3] and ra[4] == rb[4],
                "Exact common-history paired prefix")
        exact_core_identity(ra[5], rb[5])


def scan_baseline_episode(runner, assets, year, tid):
    ledger = runner.AccessLedger()
    env = runner.build_shielded_perception_eval_env(
        assets, year, tid, ledger, "Point")
    require_development_ledger(ledger)
    forced_days, free_wait_days = [], []
    total_cost = 0.0
    try:
        obs, _ = env.reset()
        require(np.isfinite(obs).all(), "Finite baseline reset")
        for day in range(365):
            out = step_and_audit(env, WAIT)
            total_cost += last_cost(env)
            require(out[2] == (day == 364), "Exact 365-day baseline horizon")
            if day < 364:
                if out[4]["intervention"]:
                    forced_days.append(day)
                else:
                    require(out[4]["proposed_safe"] is True, "Baseline WAIT certified safe")
                    free_wait_days.append(day)
        require(len(forced_days) + len(free_wait_days) == 364,
                "Exact nonterminal decision partition")
        require(len(forced_days) > 0, "Every DEVELOPMENT episode exercises forced CLEAN")
        first_forced = forced_days[0]
        require(set(range(first_forced)).issubset(set(free_wait_days)),
                "Every pre-first-force WAIT is certified safe")
        require(ledger.heldout_ppo_trajectory_access_count == 0
                and ledger.formal_perception_seed_access_count == 0
                and ledger.denied_accesses == 0,
                "Baseline DEVELOPMENT-only access")
        record = env._shield_audit_snapshot()["episode_records"][-1]
        require(record["completed"] and record["steps"] == 365
                and record["interventions"] == len(forced_days)
                and record["executed_clean_count"] == len(forced_days)
                and record["free_wait_count"] == len(free_wait_days) + 1,
                "365-day accounting includes reward-only terminal WAIT")
        return {
            "year": year,
            "trajectory_id": int(tid),
            "J_total": total_cost,
            "forced_days": forced_days,
            "free_wait_days": free_wait_days,
            "first_forced_day": int(first_forced),
            "forced_clean_count": len(forced_days),
            "free_wait_count": int(record["free_wait_count"]),
            "nonterminal_free_wait_count": len(free_wait_days),
        }
    finally:
        env.close()


def compatible_alias_accounting(env_w, env_c):
    w = env_w._shield_audit_snapshot()
    c = env_c._shield_audit_snapshot()
    for key in ("observation_mode", "decision_index", "episode_live",
                "total_safety_filtered_decisions", "total_executed_clean",
                "total_free_wait", "total_terminal_passthrough"):
        require(w[key] == c[key], "Equal alias accounting: " + key)
    require(w["total_shield_interventions"] == c["total_shield_interventions"] + 1
            and c["total_proposed_clean"] == w["total_proposed_clean"] + 1,
            "Only intended proposal/intervention accounting differs")
    rw, rc = w["episode_records"][-1], c["episode_records"][-1]
    require(set(rw) == set(rc), "Alias record schema identical")
    for key in rw:
        delta = 1 if key == "interventions" else -1 if key == "proposed_clean_count" else 0
        require(rw[key] == rc[key] + delta if delta else rw[key] == rc[key],
                "Compatible episode accounting: " + key)


def delegation_alias_witness(runner, assets, year, tid, forced_day, atol):
    env_w, env_c, ledger_w, ledger_c = env_pair(runner, assets, year, tid)
    try:
        replay_prefix_identical(env_w, env_c, forced_day)
        pre_w, pre_c = core_snapshot(env_w), core_snapshot(env_c)
        exact_core_identity(pre_w, pre_c)
        q50 = float(env_w.inner._assets._perception_cache[
            (pre_w["perception_seed"], pre_w["year"], pre_w["trajectory_id"],
             pre_w["day_index"], pre_w["L_pre"])
        ][0])
        belief_lower = float(env_w.shield.belief_lower)
        belief_upper = float(env_w.shield.belief_upper)

        out_w = step_and_audit(env_w, WAIT)
        out_c = step_and_audit(env_c, CLEAN)
        dec_w, dec_c = out_w[4], out_c[4]
        require(dec_w["intervention"] is True
                and dec_w["proposed_action"] == WAIT
                and dec_w["executed_action"] == CLEAN
                and dec_w["proposed_safe"] is False,
                "Forced WAIT replaced by CLEAN")
        require(dec_c["intervention"] is False
                and dec_c["proposed_action"] == CLEAN
                and dec_c["executed_action"] == CLEAN
                and dec_c["proposed_safe"] is True,
                "Direct CLEAN passes unchanged")
        require(np.array_equal(out_w[0], out_c[0])
                and out_w[1] == out_c[1]
                and out_w[2] == out_c[2]
                and out_w[3] == out_c[3],
                "Delegated/direct CLEAN immediate public outcome exact")
        exact_core_identity(out_w[5], out_c[5])
        require(abs(last_cost(env_w) - last_cost(env_c)) <= atol,
                "Delegated/direct CLEAN immediate cost identity")

        compatible_alias_accounting(env_w, env_c)
        next_equal = True
        if forced_day < 364:
            nw = step_and_audit(env_w, WAIT)
            nc = step_and_audit(env_c, WAIT)
            next_equal = (np.array_equal(nw[0], nc[0])
                          and nw[1] == nc[1] and nw[2] == nc[2]
                          and nw[3] == nc[3] and nw[4] == nc[4])
            require(next_equal, "One-step future proposal-history independence")
            exact_core_identity(nw[5], nc[5])
            compatible_alias_accounting(env_w, env_c)

        require(ledger_w.heldout_ppo_trajectory_access_count == 0
                and ledger_c.heldout_ppo_trajectory_access_count == 0
                and ledger_w.formal_perception_seed_access_count == 0
                and ledger_c.formal_perception_seed_access_count == 0
                and ledger_w.denied_accesses == 0 and ledger_c.denied_accesses == 0,
                "Delegation DEVELOPMENT-only access")
        return {
            "year": year,
            "trajectory_id": int(tid),
            "forced_day": int(forced_day),
            "q50_float32": q50,
            "L_pre_audit_only": float(pre_w["L_pre"]),
            "belief_lower": belief_lower,
            "belief_upper": belief_upper,
            "WAIT_proposed_safe": bool(dec_w["proposed_safe"]),
            "WAIT_executed_action": int(dec_w["executed_action"]),
            "WAIT_intervention": bool(dec_w["intervention"]),
            "CLEAN_proposed_safe": bool(dec_c["proposed_safe"]),
            "CLEAN_executed_action": int(dec_c["executed_action"]),
            "CLEAN_intervention": bool(dec_c["intervention"]),
            "immediate_reward_exact_equal": out_w[1] == out_c[1],
            "next_observation_exact_equal": bool(np.array_equal(out_w[0], out_c[0])),
            "inner_core_exact_equal": True,
            "L_post_L_next_exact_equal": True,
            "episode_accounting_compatible": True,
            "next_common_WAIT_step_exact_equal": bool(next_equal),
            "delegation_alias_exact": True,
        }
    finally:
        env_w.close()
        env_c.close()


def safe_action_counterfactual(runner, assets, year, tid, anchor_day,
                               anchor_label, first_forced_day, atol):
    env_w, env_c, ledger_w, ledger_c = env_pair(runner, assets, year, tid)
    try:
        replay_prefix_identical(env_w, env_c, anchor_day)
        pre_w, pre_c = core_snapshot(env_w), core_snapshot(env_c)
        exact_core_identity(pre_w, pre_c)
        require(anchor_day < first_forced_day,
                "Safe-action anchor strictly before first forced CLEAN")
        q50 = float(env_w.inner._assets._perception_cache[
            (pre_w["perception_seed"], pre_w["year"], pre_w["trajectory_id"],
             pre_w["day_index"], pre_w["L_pre"])
        ][0])
        belief_lower = float(env_w.shield.belief_lower)
        belief_upper = float(env_w.shield.belief_upper)

        out_w = step_and_audit(env_w, WAIT)
        out_c = step_and_audit(env_c, CLEAN)
        dec_w, dec_c = out_w[4], out_c[4]
        require(dec_w["proposed_safe"] is True and dec_w["intervention"] is False
                and dec_w["executed_action"] == WAIT,
                "Anchor WAIT safe and executed")
        require(dec_c["proposed_safe"] is True and dec_c["intervention"] is False
                and dec_c["executed_action"] == CLEAN,
                "Anchor CLEAN safe and executed")

        cost_w, cost_c = last_cost(env_w), last_cost(env_c)
        immediate_gap = cost_c - cost_w
        post_w = out_w[5]["last_reward_components"]
        post_c = out_c[5]["last_reward_components"]
        immediate_distinguishable = (
            abs(immediate_gap) > atol
            or post_w["L_post"] != post_c["L_post"]
            or post_w["L_next"] != post_c["L_next"]
            or not np.array_equal(out_w[0], out_c[0])
        )

        physical_distinguishable = (post_w["L_post"] != post_c["L_post"]
                                    or post_w["L_next"] != post_c["L_next"])
        economic_distinguishable = abs(immediate_gap) > atol
        remaining_wait, remaining_clean = cost_w, cost_c
        for day in range(anchor_day + 1, 365):
            ow = step_and_audit(env_w, WAIT)
            oc = step_and_audit(env_c, WAIT)
            remaining_wait += last_cost(env_w)
            remaining_clean += last_cost(env_c)
            require(ow[2] == (day == 364) and oc[2] == (day == 364),
                    "Exact paired counterfactual horizon")

        horizon_gap = remaining_clean - remaining_wait
        classification = (
            "CLEAN_BENEFICIAL" if horizon_gap < -atol
            else "WAIT_BENEFICIAL" if horizon_gap > atol
            else "COST_TIE_WITHIN_ATOL"
        )
        rec_w = env_w._shield_audit_snapshot()["episode_records"][-1]
        rec_c = env_c._shield_audit_snapshot()["episode_records"][-1]
        require(ledger_w.heldout_ppo_trajectory_access_count == 0
                and ledger_c.heldout_ppo_trajectory_access_count == 0
                and ledger_w.formal_perception_seed_access_count == 0
                and ledger_c.formal_perception_seed_access_count == 0
                and ledger_w.denied_accesses == 0 and ledger_c.denied_accesses == 0,
                "Counterfactual DEVELOPMENT-only access")

        return {
            "year": year,
            "trajectory_id": int(tid),
            "anchor_label": anchor_label,
            "anchor_day": int(anchor_day),
            "first_forced_day": int(first_forced_day),
            "days_before_first_forced": int(first_forced_day - anchor_day),
            "q50_float32": q50,
            "L_pre_audit_only": float(pre_w["L_pre"]),
            "belief_lower": belief_lower,
            "belief_upper": belief_upper,
            "WAIT_anchor_cost": float(cost_w),
            "CLEAN_anchor_cost": float(cost_c),
            "immediate_cost_gap_CLEANminusWAIT": float(immediate_gap),
            "WAIT_L_post": float(post_w["L_post"]),
            "CLEAN_L_post": float(post_c["L_post"]),
            "WAIT_L_next": None if post_w["L_next"] is None else float(post_w["L_next"]),
            "CLEAN_L_next": None if post_c["L_next"] is None else float(post_c["L_next"]),
            "immediate_action_distinguishable": bool(immediate_distinguishable),
            "physical_action_distinguishable": bool(physical_distinguishable),
            "economic_action_distinguishable": bool(economic_distinguishable),
            "remaining_cost_WAITbranch": float(remaining_wait),
            "remaining_cost_CLEANbranch": float(remaining_clean),
            "remaining_horizon_gap_CLEANminusWAIT": float(horizon_gap),
            "continuation_classification": classification,
            "WAITbranch_total_interventions": int(rec_w["interventions"]),
            "CLEANbranch_total_interventions": int(rec_c["interventions"]),
            "WAITbranch_total_executed_clean": int(rec_w["executed_clean_count"]),
            "CLEANbranch_total_executed_clean": int(rec_c["executed_clean_count"]),
        }
    finally:
        env_w.close()
        env_c.close()


def summarize_counterfactuals(rows, atol):
    def describe(subset):
        gaps = [float(r["remaining_horizon_gap_CLEANminusWAIT"]) for r in subset]
        n = len(gaps)
        beneficial = sum(g < -atol for g in gaps)
        return {
            "count": n, "clean_beneficial_count": beneficial,
            "wait_beneficial_count": sum(g > atol for g in gaps),
            "tie_count": sum(abs(g) <= atol for g in gaps),
            "clean_beneficial_fraction": beneficial / n if n else None,
            "mean_horizon_gap_CLEANminusWAIT": float(statistics.fmean(gaps)) if n else None,
            "median_horizon_gap_CLEANminusWAIT": float(statistics.median(gaps)) if n else None,
            "min_horizon_gap_CLEANminusWAIT": min(gaps) if n else None,
            "max_horizon_gap_CLEANminusWAIT": max(gaps) if n else None,
        }
    result = describe(rows)
    return {
        **result,
        "paired_counterfactual_count": len(rows),
        "all_immediate_actions_distinguishable": bool(rows) and all(
            r["immediate_action_distinguishable"] for r in rows),
        "both_safe_actions_physically_economically_distinguishable": bool(rows) and all(
            r["physical_action_distinguishable"] and r["economic_action_distinguishable"]
            for r in rows),
        "preemptive_clean_opportunity_observed": result["clean_beneficial_count"] > 0,
        "by_anchor_label": {label: describe([r for r in rows if r["anchor_label"] == label])
                            for label in ANCHOR_LABELS},
        "classification_tolerance": atol,
        "empty_population_fraction": "null (undefined), never a structural failure",
        "interpretation": "Paired fixed-continuation counterfactual under always-WAIT + frozen shield; NOT an optimal Q-function or Q(CLEAN)-Q(WAIT)",
    }


def verify_current_baseline_against_predecessor(baseline_rows, atol):
    prior = read_csv(PREDECESSOR / "shield_only_development_metrics.csv")
    ref = {(r["year"], int(r["trajectory_id"])): r for r in prior}
    require(len(ref) == 600 and len(baseline_rows) == 600,
            "Comparable predecessor/current baselines")
    for row in baseline_rows:
        old = ref[(row["year"], int(row["trajectory_id"]))]
        require(
            abs(float(row["J_total"]) - float(old["J_total"])) <= atol
            and int(row["forced_clean_count"]) == int(old["shield_interventions"])
            and int(row["free_wait_count"]) == int(old["free_wait_count"]),
            "Current always-WAIT mechanism scan reproduces predecessor baseline",
        )
    return True


def run_formal():
    require(not STAGE_DIRECTORY.exists(),
            "Exclusive stage directory; no overwrite/resume/retry")
    require(git("status", "--porcelain", "--untracked-files=all") == "",
            "Committed unchanged clean sources required")
    head = git("rev-parse", "HEAD")
    failed_provenance = verify_failed_predecessor(head)
    predecessor_audit, _ = verify_predecessor(head)

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(self_blob == git("hash-object", f"--path={SELF}", SELF)
            == git("rev-parse", f":{SELF}"),
            "Mechanism-audit source committed unchanged")
    static = static_contract()
    failed_provenance["recovery_source_git_blob"] = self_blob
    failed_provenance["recovery_source_sha256"] = sha_file(ROOT / SELF)

    import paper2_shielded_perception_ppo_runner_v1 as runner

    p = runner.get_protocol()
    atol = float(p.selection.atol)
    require(atol == 1e-12
            and dict(p.observations)["Point"] == ("q50", "sin_DOY", "cos_DOY"),
            "Frozen Point protocol/tolerance")
    collapse = predecessor_collapse_audit(atol)

    dev = runner.partition("DEVELOPMENT")
    specs = tuple((year, tid) for year in dev.years for tid in dev.trajectory_ids)
    require(len(specs) == 600 and len(set(specs)) == 600, "Full unique DEVELOPMENT population")

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)

    try:
        assets = runner.load_assets()
        assets.ensure_perception()
        baseline_rows, baseline_map = [], {}
        for year, tid in specs:
            row = scan_baseline_episode(runner, assets, year, tid)
            category, anchors = anchor_eligibility(row["first_forced_day"])
            row["anchors"] = anchors
            baseline_rows.append({
                "year": year,
                "trajectory_id": int(tid),
                "J_total": row["J_total"],
                "first_forced_day": row["first_forced_day"],
                "anchor_eligibility": category,
                "available_safe_action_anchor_count": len(anchors),
                "MID_PRE_FORCE": anchors.get("MID_PRE_FORCE"),
                "PRE_FORCE": anchors.get("PRE_FORCE"),
                "forced_clean_count": row["forced_clean_count"],
                "free_wait_count": row["free_wait_count"],
                "nonterminal_free_wait_count": row["nonterminal_free_wait_count"],
            })
            baseline_map[(year, int(tid))] = row

        require(verify_current_baseline_against_predecessor(baseline_rows, atol),
                "Baseline runtime regression")
        coverage = eligibility_summary(baseline_rows)
        alias_rows, counterfactual_rows = [], []
        for year, tid in specs:
            baseline = baseline_map[(year, int(tid))]
            forced = baseline["first_forced_day"]
            alias_rows.append(
                delegation_alias_witness(runner, assets, year, tid, forced, atol)
            )
            anchors = baseline["anchors"]
            for label, anchor_day in anchors.items():
                counterfactual_rows.append(
                    safe_action_counterfactual(
                        runner, assets, year, tid,
                        anchor_day, label, forced, atol
                    )
                )

        require(len(alias_rows) == 600
                and all(r["delegation_alias_exact"] for r in alias_rows),
                "Exact delegation aliasing across all DEVELOPMENT episodes")
        expected_anchors = {(year, int(tid), label, day)
                            for (year, tid), row in baseline_map.items()
                            for label, day in row["anchors"].items()}
        actual_anchors = {(r["year"], r["trajectory_id"], r["anchor_label"], r["anchor_day"])
                          for r in counterfactual_rows}
        require(len(counterfactual_rows) == coverage["total_available_safe_action_anchors"]
                and actual_anchors == expected_anchors
                and len(actual_anchors) == len(counterfactual_rows),
                "Every structurally available anchor audited exactly once")
        coverage["counterfactual_anchor_count"] = len(counterfactual_rows)

        counterfactual_summary = summarize_counterfactuals(
            counterfactual_rows, atol)
        forced_days = [r["first_forced_day"] for r in baseline_rows]
        baseline_summary = {
            "episode_count": 600,
            "mean_first_forced_day": float(statistics.fmean(forced_days)),
            "median_first_forced_day": float(statistics.median(forced_days)),
            "min_first_forced_day": int(min(forced_days)),
            "max_first_forced_day": int(max(forced_days)),
            "mean_forced_clean_count": float(statistics.fmean(
                r["forced_clean_count"] for r in baseline_rows)),
            "mean_free_wait_count": float(statistics.fmean(
                r["free_wait_count"] for r in baseline_rows)),
        }
        alias_summary = {
            "episode_count": len(alias_rows),
            "exact_alias_count": sum(r["delegation_alias_exact"] for r in alias_rows),
            "all_immediate_reward_equal": all(
                r["immediate_reward_exact_equal"] for r in alias_rows),
            "all_next_observation_equal": all(
                r["next_observation_exact_equal"] for r in alias_rows),
            "all_inner_core_equal": all(r["inner_core_exact_equal"] for r in alias_rows),
            "all_next_common_WAIT_step_equal": all(
                r["next_common_WAIT_step_exact_equal"] for r in alias_rows),
            "delegation_aliasing_confirmed": all(
                r["delegation_alias_exact"] for r in alias_rows),
        }
        route_supported = (
            alias_summary["delegation_aliasing_confirmed"]
            and counterfactual_summary["both_safe_actions_physically_economically_distinguishable"]
            and counterfactual_summary["preemptive_clean_opportunity_observed"]
        )
        mechanism_summary = {
            "stage": STAGE,
            "predecessor_collapse": collapse,
            "baseline_scan": baseline_summary,
            "anchor_eligibility_coverage": coverage,
            "delegation_aliasing": alias_summary,
            "safe_action_counterfactual": counterfactual_summary,
            "masking_route_diagnostic": {
                "posthoc_action_aliasing_confirmed":
                    alias_summary["delegation_aliasing_confirmed"],
                "both_safe_actions_physically_economically_distinguishable":
                    counterfactual_summary["both_safe_actions_physically_economically_distinguishable"],
                "clean_beneficial_counterfactual_fraction": counterfactual_summary["clean_beneficial_fraction"],
                "preemptive_clean_opportunity_observed": counterfactual_summary["preemptive_clean_opportunity_observed"],
                "preemptive_clean_benefit_observed_under_fixed_continuation":
                    counterfactual_summary["preemptive_clean_opportunity_observed"],
                "action_masking_mechanistically_supported": bool(route_supported),
                "route_interpretation": (
                    "ACTION-MASKING ROUTE HAS DIRECT MECHANISTIC SUPPORT"
                    if route_supported
                    else "DO NOT START MASKED PPO YET; REVIEW COUNTERFACTUAL ECONOMIC RESULT"
                ),
                "role": "DESCRIPTIVE DEVELOPMENT MECHANISM DIAGNOSTIC; NO TUNING/SELECTION",
            },
            "claim_boundary": (
                "DEVELOPMENT paired mechanism audit only; no optimal-Q, "
                "final-seed or formal-held-out claim"
            ),
        }

        gates = {
            "predecessor_recovery_exact": predecessor_audit["stage_pass"] is True,
            "static_no_training_contract": all(static.values()),
            "full_600_DEVELOPMENT_baseline_scan": len(baseline_rows) == 600,
            "baseline_reproduces_predecessor": True,
            "predecessor_Point_collapse_reproduced_from_artifacts":
                collapse["all_Point_rows_match_shield_only"],
            "delegation_alias_exact_all_600":
                alias_summary["delegation_aliasing_confirmed"],
            "all_available_safe_action_anchors_audited":
                len(counterfactual_rows) == coverage["total_available_safe_action_anchors"],
            "zero_PPO_training": True,
            "zero_protected_access": True,
        }
        failed = [k for k, v in gates.items() if not v]
        require(not failed, "Structural mechanism-audit gates failed")

        protocol_manifest = {
            "stage": STAGE,
            "mode": "formal",
            "role": "POST-HOC SHIELD DELEGATION / SAFE-ACTION IDENTIFIABILITY DIAGNOSTIC",
            "population": "all 600 DEVELOPMENT episode specifications",
            "observation_mode": "Point",
            "PPO_training_runs": 0,
            "PPO_model_loads": 0,
            "UA_training_runs": 0,
            "anchor_protocol": {
                "delegation_alias": "first forced CLEAN day under always-WAIT + frozen shield",
                "eligibility": "d_f=0: ZERO_PRE_FORCE_SAFE_ANCHOR; d_f=1: ONE_PRE_FORCE_SAFE_ANCHOR (PRE_FORCE=0); d_f>=2: TWO_PRE_FORCE_SAFE_ANCHORS",
                "safe_action_anchor_1": "floor((first_forced_day - 1) / 2), only when d_f>=2",
                "safe_action_anchor_2": "first_forced_day - 1, only when d_f>=1",
                "selection_input_only": "first_forced_day; no outcome-based filtering",
                "downstream_continuation":
                    "always-WAIT proposal under the same frozen shield",
            },
            "horizon_gap_definition": (
                "remaining_cost(CLEAN-now then always-WAIT+shield) - "
                "remaining_cost(WAIT-now then always-WAIT+shield)"
            ),
            "classification_atol": atol,
            "claim_boundary": "Paired fixed-continuation cost gap; not optimal Q-function or Q(CLEAN)-Q(WAIT)",
            "stage_pass_meaning": "Complete frozen DEVELOPMENT mechanism audit executed successfully; economic diagnostics are not gates",
            "terminal_day_accounting": "Day 364 reward-only WAIT included in total cost and free_wait_count; excluded from nonterminal free_wait_count",
            "performance_gate": None,
            "masking_route_decision": (
                "made only after this frozen diagnostic; no reward/shield/PPO "
                "parameter tuned here"
            ),
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
        }

        content = {
            "protocol_manifest.json": encode(protocol_manifest),
            "failed_predecessor_provenance.json": encode(failed_provenance),
            "anchor_eligibility_summary.json": encode(coverage),
            "predecessor_collapse_audit.json": encode(collapse),
            "mechanism_summary.json": encode(mechanism_summary),
            "baseline_scan.csv": csv_bytes(baseline_rows),
            "delegation_alias_witnesses.csv": csv_bytes(alias_rows),
            "safe_action_counterfactuals.csv": csv_bytes(counterfactual_rows,
                [
                    'year',
                    'trajectory_id',
                    'anchor_label',
                    'anchor_day',
                    'first_forced_day',
                    'days_before_first_forced',
                    'q50_float32',
                    'L_pre_audit_only',
                    'belief_lower',
                    'belief_upper',
                    'WAIT_anchor_cost',
                    'CLEAN_anchor_cost',
                    'immediate_cost_gap_CLEANminusWAIT',
                    'WAIT_L_post',
                    'CLEAN_L_post',
                    'WAIT_L_next',
                    'CLEAN_L_next',
                    'immediate_action_distinguishable',
                    'physical_action_distinguishable',
                    'economic_action_distinguishable',
                    'remaining_cost_WAITbranch',
                    'remaining_cost_CLEANbranch',
                    'remaining_horizon_gap_CLEANminusWAIT',
                    'continuation_classification',
                    'WAITbranch_total_interventions',
                    'CLEANbranch_total_interventions',
                    'WAITbranch_total_executed_clean',
                    'CLEANbranch_total_executed_clean',
                ]),
        }
        for name, data in content.items():
            write_exclusive(OUTPUT / name, data)

        hashes = {name: sha_bytes(data) for name, data in content.items()}
        output_hashes = encode({
            "hashes": hashes,
            "exclusions": {
                "output_hashes.json": "SHA256 pinned in audit_summary; avoid self-reference",
                "audit_summary.json": "Published last PASS marker",
            },
        })
        write_exclusive(OUTPUT / "output_hashes.json", output_hashes)

        expected = set(content) | {"output_hashes.json"}
        actual = {p.relative_to(OUTPUT).as_posix()
                  for p in OUTPUT.rglob("*") if p.is_file()}
        require(actual == expected, "Exact pre-audit output inventory")
        require(git("rev-parse", "HEAD") == head
                and git("status", "--porcelain", "--untracked-files=all") == "",
                "HEAD/worktree unchanged")
        verify_predecessor(head)
        require(verify_failed_predecessor(head)["execution_failure_sha256"]
                == failed_provenance["execution_failure_sha256"], "Failed 4A preserved at publication")

        summary = {
            "stage": STAGE,
            "mode": "formal",
            "stage_pass": True,
            "scientific_status": STATUS,
            "HEAD": head,
            "gates": gates,
            "failed_gates": [],
            "PPO_training_runs": 0,
            "PPO_model_loads": 0,
            "UA_training_runs": 0,
            "development_episode_count": 600,
            "anchor_eligibility_coverage": coverage,
            "failed_4A_evidence_sha256": FAILED_SHA256,
            "frozen_source_git_blobs": failed_provenance["frozen_source_git_blobs"],
            "recovery_source_git_blob": self_blob,
            "delegation_alias_witness_count": len(alias_rows),
            "safe_action_counterfactual_count": len(counterfactual_rows),
            "delegation_aliasing_confirmed":
                alias_summary["delegation_aliasing_confirmed"],
            "both_safe_actions_distinguishable":
                counterfactual_summary["all_immediate_actions_distinguishable"],
            "preemptive_clean_opportunity_observed":
                counterfactual_summary["preemptive_clean_opportunity_observed"],
            "clean_beneficial_counterfactual_fraction":
                counterfactual_summary["clean_beneficial_fraction"],
            "action_masking_mechanistically_supported": bool(route_supported),
            "masking_route_interpretation":
                mechanism_summary["masking_route_diagnostic"]["route_interpretation"],
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
            "output_sha256": {
                **hashes,
                "output_hashes.json": sha_bytes(output_hashes),
            },
            "audit_summary_published_last": True,
        }
        write_exclusive(OUTPUT / "audit_summary.json", encode(summary))
        return summary

    except Exception as exc:
        if not (OUTPUT / "execution_failure.json").exists():
            write_exclusive(
                OUTPUT / "execution_failure.json",
                encode({
                    "stage": STAGE,
                    "stage_pass": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "action": (
                        "STOP; PRESERVE EVIDENCE; NO PPO TRAINING, NO MASKED-PPO "
                        "START, NO TUNING/RETRY"
                    ),
                    "HEAD": head,
                }),
            )
        raise
    finally:
        verify_failed_predecessor(head)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
