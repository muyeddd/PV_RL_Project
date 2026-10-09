"""P2-2F-0B0-v1 fixed-continuation counterfactual feasibility diagnostic.

Four DEVELOPMENT episodes are pre-registered before outcomes are observed.
Within each episode, nonterminal days 0..363 are split into five contiguous
strata.  One baseline free-choice day per nonempty stratum is selected by a
fixed SHA-256 modulo rule.  Each selected anchor is replayed from the episode
start in two independent frozen shielded Point environments: WAIT at the
anchor versus CLEAN at the anchor, followed by WAIT proposals plus the frozen
shield through day 364 in both branches.

This is a paired fixed-continuation action-gap diagnostic, not Q* and not an
estimate of the 600-episode DEVELOPMENT opportunity rate.  Import is inert;
only ``--mode formal`` constructs environments or writes output.  Formal mode
must never be launched before this source is reviewed and committed.
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
SELF = "experiments/run_paper2_stage2f0b0_counterfactual_opportunity_feasibility_v1.py"
STAGE = "P2-2F-0B0-v1"
STATUS = "COUNTERFACTUAL_OPPORTUNITY_FEASIBILITY_COMPLETE"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
STAGE_DIRECTORY = BASE / "p2_2f_0b0_counterfactual_opportunity_feasibility_v1"
OUTPUT = STAGE_DIRECTORY / "formal"

WAIT = 0
CLEAN = 1
ATOL = 1e-12
RTOL = 0.0
PERCEPTION_SEED = 20260906
HASH_SALT = "P2-2F-0B0-v1|free-choice-anchor|sha256-modulo|2026-10-08"
EPISODES = (("YEAR1", 0), ("YEAR1", 137), ("YEAR2", 0), ("YEAR2", 137))
STRATA = (
    ("S1", 0, 72),
    ("S2", 73, 145),
    ("S3", 146, 218),
    ("S4", 219, 291),
    ("S5", 292, 363),
)
CLASSIFICATIONS = ("CLEAN_BENEFICIAL", "WAIT_BENEFICIAL", "TIE_WITHIN_ATOL")
PAIR_FIELDS = (
    "year", "trajectory_id", "stratum", "anchor_day", "anchor_date",
    "selection_sha256", "prefix_cumulative_total_cost", "anchor_q50",
    "anchor_width_audit_only", "L_pre_common_audit_only",
    "WAIT_L_post_audit_only", "CLEAN_L_post_audit_only",
    "WAIT_L_next_audit_only", "CLEAN_L_next_audit_only",
    "WAIT_next_q50_audit_only", "CLEAN_next_q50_audit_only",
    "WAIT_next_width_audit_only", "CLEAN_next_width_audit_only",
    "WAIT_immediate_soiling_cost", "WAIT_immediate_cleaning_cost",
    "WAIT_immediate_total_cost", "CLEAN_immediate_soiling_cost",
    "CLEAN_immediate_cleaning_cost", "CLEAN_immediate_total_cost",
    "WAIT_future_soiling_cost", "WAIT_future_cleaning_cost",
    "WAIT_future_total_cost", "CLEAN_future_soiling_cost",
    "CLEAN_future_cleaning_cost", "CLEAN_future_total_cost",
    "WAIT_J_anchor_to_terminal", "CLEAN_J_anchor_to_terminal",
    "WAIT_soiling_cost_anchor_to_terminal",
    "CLEAN_soiling_cost_anchor_to_terminal",
    "WAIT_cleaning_cost_anchor_to_terminal",
    "CLEAN_cleaning_cost_anchor_to_terminal",
    "WAIT_CLEAN_count_anchor_to_terminal",
    "CLEAN_CLEAN_count_anchor_to_terminal",
    "WAIT_forced_CLEAN_count_anchor_to_terminal",
    "CLEAN_forced_CLEAN_count_anchor_to_terminal",
    "WAIT_forced_CLEAN_days_json", "CLEAN_forced_CLEAN_days_json",
    "WAIT_forced_CLEAN_dates_json", "CLEAN_forced_CLEAN_dates_json",
    "WAIT_next_forced_CLEAN_day", "CLEAN_next_forced_CLEAN_day",
    "WAIT_next_forced_CLEAN_date", "CLEAN_next_forced_CLEAN_date",
    "WAIT_days_to_next_forced_CLEAN", "CLEAN_days_to_next_forced_CLEAN",
    "delta_soiling_cost_CLEANminusWAIT",
    "delta_cleaning_cost_CLEANminusWAIT",
    "delta_total_cost_CLEANminusWAIT", "delta_decomposition_residual",
    "classification",
)

SOURCE_SHA256 = {
    "experiments/paper2_gym_pomdp_env_v1.py":
        "f99bc91516af768dc8c9989264f64a8bb0db0360323d72738513bf642149a8b8",
    "experiments/paper2_shielded_perception_ppo_runner_v1.py":
        "804b34cd60f70a93c60b87dd5131f9dd3ca7d78ad2ba4d1c0558f1e92c2c4e7c",
    "experiments/paper2_masked_perception_ppo_runner_v1.py":
        "0e1ca9012218d361a5e83b0e849dd3baae7c94a7fee43c7ca57eec183a3661e0",
    "experiments/paper2_perception_ppo_runner_v1.py":
        "9ffb8cf557b904287ac00f9d3cfad678b5ca0e89228573c7d55222f92256cde0",
    "experiments/run_paper2_stage2d4ar_shield_delegation_action_identifiability_recovery_v1.py":
        "06bc0ffa1a29799f8114b00fdb7c67557f2a48651c5c5f4211fc8523ea1a63f0",
    "experiments/run_paper2_stage2d4b_safe_action_distinction_decomposition_audit_v1.py":
        "74e02333c925eedd728c22092537004ad20a89de17c0ab6873d3308f532f3264",
    "experiments/run_paper2_stage2f0ar_existing_opportunity_evidence_audit_recovery_v1.py":
        "8b0deb619fd2fb35a24d187f90861b63456e899a0f9d67c6744a7521e24559ef",
    "experiments/paper2_ppo_protocol_v1.py":
        "1460f7947bc6f522dee787ac12a1716be73884c9e8efce26f95fb9facf998f1c",
    "experiments/paper2_q50_set_membership_shield_v1.py":
        "1c8e3058f89ab1507185e65e8bf75a567853e92603eedb8b19eb1fb4b5533cf6",
}
SOURCE_GIT_BLOBS = {
    "experiments/paper2_gym_pomdp_env_v1.py": "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    "experiments/paper2_shielded_perception_ppo_runner_v1.py": "b61fc7b0005b64fcb938963b7b8c111567fa4f1a",
    "experiments/paper2_masked_perception_ppo_runner_v1.py": "be86a5e8a8a162c0d6be720c5340914ccc1ed82e",
    "experiments/paper2_perception_ppo_runner_v1.py": "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    "experiments/run_paper2_stage2d4ar_shield_delegation_action_identifiability_recovery_v1.py": "a778fa61aa7559353ed7e655d04298c4c8724f86",
    "experiments/run_paper2_stage2d4b_safe_action_distinction_decomposition_audit_v1.py": "e5b4006c9ddb1866e78df2335ccc855de2253b98",
    "experiments/run_paper2_stage2f0ar_existing_opportunity_evidence_audit_recovery_v1.py": "6dd187e1a05bc915eb823bd7e09ade0852218b37",
    "experiments/paper2_ppo_protocol_v1.py": "0dedab12015978a01a41aea02f065a470be39094",
    "experiments/paper2_q50_set_membership_shield_v1.py": "db78b7be651fae6c15231b3d8ac55cae822ab409",
}

AUTHORITIES = {
    "P2-2D-4AR-v1": {
        "directory": BASE / "p2_2d_4ar_shield_delegation_action_identifiability_recovery_v1/formal",
        "audit_sha256": "4f1afc1e78ed6fdb1d02e12afef2cdde765fe56a008c091bcd85e7e5026293ff",
        "head": "9c333d0b5694c4796dcaf24a43728a0a96d27373",
    },
    "P2-2D-4B-v1": {
        "directory": BASE / "p2_2d_4b_safe_action_distinction_decomposition_audit_v1/formal",
        "audit_sha256": "522737511d1132fd9ea2711a80915d2b90fc955adb18caad6ee439740b113165",
        "head": "2e053879dee9ad1df15aaff3956ea641274022e7",
    },
    "P2-2F-0AR-v1": {
        "directory": BASE / "p2_2f_0ar_existing_opportunity_evidence_audit_recovery_v1/formal",
        "audit_sha256": "5219ed05aab1ae8b16787db0b978c4fea3253bc7aca6e02afd3da4d915db650a",
        "head": "1f1f9b6133440ecabbe7487b23016510e70f333b",
    },
}


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


def csv_bytes(rows, fieldnames):
    require(all(set(row) == set(fieldnames) for row in rows), "Consistent CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def write_exclusive(path, data):
    path = Path(path)
    require(path.parent.resolve() == OUTPUT.resolve(), "Writes restricted to new formal directory")
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, "Exact output readback: " + path.name)


def classify(delta_total_cost, atol=ATOL):
    require(np.isfinite(delta_total_cost), "Finite classification input")
    if delta_total_cost < -atol:
        return "CLEAN_BENEFICIAL"
    if delta_total_cost > atol:
        return "WAIT_BENEFICIAL"
    return "TIE_WITHIN_ATOL"


def paired_feasibility_status(selected_count, paired_count):
    """Pure structural status; economic classifications are deliberately absent."""
    require(type(selected_count) is int and type(paired_count) is int
            and 0 <= paired_count <= selected_count <= 20,
            "Valid selected/paired feasibility counts")
    verified = paired_count > 0 and paired_count == selected_count
    if verified:
        status = "VERIFIED_AT_LEAST_ONE_COMPLETE_LEGAL_PAIR"
    elif paired_count == 0:
        status = "NOT_VERIFIED_NO_COMPLETE_LEGAL_PAIR"
    else:
        status = "NOT_VERIFIED_INCOMPLETE_SELECTED_PAIR_SET"
    return {"technical_feasibility": verified, "technical_feasibility_status": status}


def choose_hashed_candidate(year, trajectory_id, stratum_label, start_day,
                            end_day, free_choice_days, salt=HASH_SALT):
    """Pure, outcome-blind SHA-256 modulo selection from sorted eligible days."""
    require(year in ("YEAR1", "YEAR2"), "Known year")
    require(type(trajectory_id) is int and trajectory_id in (0, 137), "Pre-registered trajectory")
    require(type(start_day) is int and type(end_day) is int
            and 0 <= start_day <= end_day <= 363, "Nonterminal stratum bounds")
    eligible = tuple(sorted({int(day) for day in free_choice_days
                             if start_day <= int(day) <= end_day}))
    require(all(type(day) is int and 0 <= day <= 363 for day in eligible),
            "Sorted nonterminal free-choice candidates")
    payload = "\n".join((
        salt, year, str(trajectory_id), stratum_label,
        str(start_day), str(end_day), ",".join(map(str, eligible)),
    )) + "\n"
    digest = sha_bytes(payload.encode("utf-8"))
    if not eligible:
        return {
            "status": "EMPTY", "eligible_days": [], "eligible_count": 0,
            "hash_payload": payload, "sha256": digest,
            "hash_integer_base16": None, "modulo_index": None,
            "selected_day": None,
        }
    integer = int(digest, 16)
    index = integer % len(eligible)
    return {
        "status": "SELECTED", "eligible_days": list(eligible),
        "eligible_count": len(eligible), "hash_payload": payload,
        "sha256": digest, "hash_integer_base16": format(integer, "x"),
        "modulo_index": index, "selected_day": eligible[index],
    }


def selection_rows(baselines):
    require(set(baselines) == set(EPISODES), "Exactly four pre-registered baselines")
    rows = []
    for year, tid in EPISODES:
        free = baselines[(year, tid)]["free_wait_days"]
        for label, start, end in STRATA:
            selected = choose_hashed_candidate(year, tid, label, start, end, free)
            rows.append({
                "year": year,
                "trajectory_id": tid,
                "stratum": label,
                "start_day_inclusive": start,
                "end_day_inclusive": end,
                "selection_status": selected["status"],
                "eligible_free_choice_count": selected["eligible_count"],
                "eligible_free_choice_days_json": json.dumps(selected["eligible_days"], separators=(",", ":")),
                "hash_salt": HASH_SALT,
                "hash_payload_utf8_json": json.dumps(selected["hash_payload"], ensure_ascii=True),
                "sha256": selected["sha256"],
                "modulo_index_zero_based": selected["modulo_index"],
                "selected_anchor_day": selected["selected_day"],
            })
    require(len(rows) == 20, "Four episodes times five strata")
    return rows


def verify_sources(head):
    require(set(SOURCE_SHA256) == set(SOURCE_GIT_BLOBS), "Source pin registries align")
    records = {}
    for path, digest in SOURCE_SHA256.items():
        blob = SOURCE_GIT_BLOBS[path]
        actual_sha = sha_file(ROOT / path)
        working_blob = git("hash-object", f"--path={path}", path)
        head_blob = git("rev-parse", f"{head}:{path}")
        index_blob = git("rev-parse", f":{path}")
        require(actual_sha == digest and working_blob == blob
                and head_blob == blob and index_blob == blob,
                "Frozen source SHA/blob unchanged: " + path)
        records[path] = {"sha256": actual_sha, "git_blob": blob}
    return records


def verify_authorities(head):
    records = {}
    for stage, spec in AUTHORITIES.items():
        directory = spec["directory"]
        audit_path = directory / "audit_summary.json"
        require(directory.is_dir() and audit_path.is_file(), stage + " authority exists")
        require(sha_file(audit_path) == spec["audit_sha256"], stage + " original audit SHA-256")
        audit = json.loads(audit_path.read_bytes())
        require(audit["stage"] == stage and audit["stage_pass"] is True
                and audit["HEAD"] == spec["head"] and audit["failed_gates"] == [],
                stage + " accepted frozen audit")
        frozen_gates = audit.get("gates")
        require(isinstance(frozen_gates, dict) and bool(frozen_gates)
                and all(value is True for value in frozen_gates.values()),
                stage + " all frozen structural gates explicitly true")
        require(audit["PPO_training_runs"] == 0 and audit["PPO_model_loads"] == 0
                and audit.get("formal_perception_seed_access_count", 0) == 0
                and audit["RANDOM_TEST_access_count"] == 0
                and audit["SEALED_DATES_access_count"] == 0,
                stage + " no training/model/protected-role access")
        if stage == "P2-2D-4AR-v1":
            require(audit["development_episode_count"] == 600
                    and audit["safe_action_counterfactual_count"] == 163,
                    "Frozen 4AR DEVELOPMENT and anchor counts")
        elif stage == "P2-2D-4B-v1":
            require(audit["counterfactual_anchor_count"] == 163
                    and audit["clean_beneficial_count"] == 63
                    and audit["wait_beneficial_count"] == 90
                    and audit["tie_count"] == 10,
                    "Frozen 4B classification counts")
        else:
            diagnostics = audit["performance_diagnostics_not_pass_gates"]
            require(diagnostics["CLEAN_beneficial_selected_anchors"] == 63
                    and diagnostics["CLEAN_beneficial_distinct_episodes"] == 52
                    and diagnostics["selected_anchor_CLEAN_beneficial_fraction_not_population_rate"]
                    == 63 / 163,
                    "Frozen 0AR opportunity evidence and nonpopulation ratio")
            require(all(diagnostics["policies_all_voluntary_CLEAN_zero_in_1200_development_episodes"].values()),
                    "Frozen three-policy zero voluntary CLEAN result")
        hashes = audit["output_sha256"]
        require(bool(hashes), stage + " output registry nonempty")
        for name, digest in hashes.items():
            require(sha_file(directory / name) == digest, stage + " output SHA-256: " + name)
        actual = {p.name for p in directory.iterdir() if p.is_file()}
        require(actual == set(hashes) | {"audit_summary.json"}, stage + " exact formal inventory")
        git("merge-base", "--is-ancestor", spec["head"], head)
        records[stage] = {
            "directory": directory.relative_to(ROOT).as_posix(),
            "HEAD": spec["head"],
            "audit_summary_sha256": spec["audit_sha256"],
            "verified_structural_gates": dict(frozen_gates),
            "verified_output_sha256": hashes,
            "read_only": True,
        }
    return records


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    require(not ({"learn", "predict", "save", "load", "build_perception_ppo_model",
                  "build_shielded_perception_ppo_model", "build_masked_perception_ppo_model",
                  "train_final_checkpoint", "reload_final_checkpoint"} & attrs),
            "No PPO construction, training, loading, prediction, or save")
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    require(not any(name in ("stable_baselines3", "sb3_contrib", "torch") for name in imports),
            "No PPO dependency import")
    require(EPISODES == (("YEAR1", 0), ("YEAR1", 137), ("YEAR2", 0), ("YEAR2", 137)),
            "Exact pre-registered episode list")
    require(STRATA == (("S1", 0, 72), ("S2", 73, 145), ("S3", 146, 218),
                       ("S4", 219, 291), ("S5", 292, 363)),
            "Exact five contiguous strata")
    covered = [day for _, start, end in STRATA for day in range(start, end + 1)]
    require(covered == list(range(364)), "Strata partition all and only days 0..363")
    probe = choose_hashed_candidate("YEAR1", 0, "S1", 0, 72, [9, 2, 9, 5])
    repeat = choose_hashed_candidate("YEAR1", 0, "S1", 0, 72, [5, 9, 2])
    require(probe == repeat and probe["eligible_days"] == [2, 5, 9]
            and probe["selected_day"] == probe["eligible_days"][probe["modulo_index"]],
            "Deterministic sorted-set SHA-256 modulo selection")
    empty = choose_hashed_candidate("YEAR2", 137, "S5", 292, 363, [])
    require(empty["status"] == "EMPTY" and empty["selected_day"] is None
            and empty["modulo_index"] is None, "Empty stratum is retained without replacement")
    require(classify(-2 * ATOL) == "CLEAN_BENEFICIAL"
            and classify(2 * ATOL) == "WAIT_BENEFICIAL"
            and classify(ATOL) == "TIE_WITHIN_ATOL"
            and classify(-ATOL) == "TIE_WITHIN_ATOL", "Classification boundary")
    require(paired_feasibility_status(0, 0) == {
                "technical_feasibility": False,
                "technical_feasibility_status": "NOT_VERIFIED_NO_COMPLETE_LEGAL_PAIR",
            }
            and paired_feasibility_status(3, 3)["technical_feasibility"] is True
            and paired_feasibility_status(3, 2)["technical_feasibility"] is False,
            "At least one complete legal pair required for technical feasibility")
    writer = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "write_exclusive")
    require("path.parent.resolve() == OUTPUT.resolve()" in ast.unparse(writer),
            "Exclusive writes restricted to the new formal directory")
    opens = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "open"]
    require(all(node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in ("rb", "xb") for node in opens),
            "Inputs read-only; outputs exclusive-create only")
    require("if __name__ == \"__main__\"" in source and "--mode" in source,
            "Import-inert explicit formal entrypoint")
    return {
        "import_inert": True,
        "no_PPO_training_loading_prediction_or_save": True,
        "development_Point_shield_runtime_only": True,
        "episodes_pre_registered_exact": True,
        "five_contiguous_nonterminal_strata_exact": True,
        "SHA256_sorted_set_modulo_selection": True,
        "empty_strata_retained_without_replacement": True,
        "classification_tolerance_exact": True,
        "nonempty_complete_pair_required_for_feasibility": True,
        "exclusive_new_output_directory": True,
        "read_only_frozen_inputs": True,
    }


def require_development_ledger(ledger):
    require(len(ledger.records) == 1 and ledger.denied_accesses == 0
            and all(record["role"] == "DEVELOPMENT"
                    and record["observation_mode"] == "Point"
                    for record in ledger.records)
            and ledger.heldout_ppo_trajectory_access_count == 0
            and ledger.formal_perception_seed_access_count == 0,
            "Only authorized DEVELOPMENT Point construction")


def perception_audit_record(env, snapshot):
    require(snapshot["perception_seed"] == PERCEPTION_SEED, "Frozen DEVELOPMENT perception seed")
    key = (snapshot["perception_seed"], snapshot["year"], snapshot["trajectory_id"],
           snapshot["day_index"], snapshot["L_pre"])
    require(key in env.inner._assets._perception_cache, "Expected frozen perception cache record")
    q50, width = env.inner._assets._perception_cache[key]
    return {"q50": float(q50), "width": float(width)}


def exact_episode_inputs(env_w, env_c):
    inner_w, inner_c = env_w.inner, env_c.inner
    require(np.array_equal(inner_w._indices, inner_c._indices)
            and np.array_equal(inner_w._rain, inner_c._rain)
            and np.array_equal(inner_w._energy, inner_c._energy)
            and inner_w._calendar.equals(inner_c._calendar),
            "Exact paired exogenous arrays and calendar")
    return {
        "indices_sha256": sha_bytes(inner_w._indices.tobytes()),
        "rain_sha256": sha_bytes(np.asarray(inner_w._rain).tobytes()),
        "energy_sha256": sha_bytes(inner_w._energy.tobytes()),
        "calendar_date_sha256": sha_bytes(
            "\n".join(inner_w._calendar.date.dt.strftime("%Y-%m-%d")).encode("ascii")
        ),
    }


def add_step(accumulator, output):
    parts = output[5]["last_reward_components"]
    decision = output[4]
    required = {"soiling_cost", "cleaning_cost", "total_cost", "action",
                "date", "day_index", "L_pre", "L_post", "L_next"}
    require(parts is not None and required.issubset(parts), "Actual reward-component schema")
    require(abs(float(parts["total_cost"])
                - (float(parts["soiling_cost"]) + float(parts["cleaning_cost"]))) <= ATOL,
            "Daily total cost decomposition")
    accumulator["soiling_cost"] += float(parts["soiling_cost"])
    accumulator["cleaning_cost"] += float(parts["cleaning_cost"])
    accumulator["total_cost"] += float(parts["total_cost"])
    accumulator["clean_count"] += int(parts["action"] == CLEAN)
    if decision["intervention"]:
        require(decision["proposed_action"] == WAIT
                and decision["executed_action"] == CLEAN
                and parts["action"] == CLEAN, "Forced CLEAN semantics")
        accumulator["forced_clean_days"].append(int(parts["day_index"]))
        accumulator["forced_clean_dates"].append(parts["date"])
    return parts


def new_accumulator():
    return {
        "soiling_cost": 0.0,
        "cleaning_cost": 0.0,
        "total_cost": 0.0,
        "clean_count": 0,
        "forced_clean_days": [],
        "forced_clean_dates": [],
    }


def finalize_accumulator(accumulator, anchor_day):
    require(abs(accumulator["total_cost"]
                - (accumulator["soiling_cost"] + accumulator["cleaning_cost"])) <= ATOL,
            "Branch cumulative cost decomposition")
    forced_days = accumulator["forced_clean_days"]
    next_day = forced_days[0] if forced_days else None
    return {
        **accumulator,
        "forced_clean_count": len(forced_days),
        "next_forced_clean_day": next_day,
        "next_forced_clean_date": accumulator["forced_clean_dates"][0] if forced_days else None,
        "days_to_next_forced_clean": next_day - anchor_day if next_day is not None else None,
    }


def paired_counterfactual(prior4ar, runner, assets, baseline, selection, atol):
    year = selection["year"]
    tid = int(selection["trajectory_id"])
    anchor_day = int(selection["selected_anchor_day"])
    require(anchor_day in baseline["free_wait_days"] and 0 <= anchor_day < 364,
            "Selected anchor is a baseline nonterminal free-choice day")
    env_w, env_c, ledger_w, ledger_c = prior4ar.env_pair(runner, assets, year, tid)
    try:
        input_hashes = exact_episode_inputs(env_w, env_c)
        prior4ar.replay_prefix_identical(env_w, env_c, anchor_day)
        pre_w, pre_c = prior4ar.core_snapshot(env_w), prior4ar.core_snapshot(env_c)
        prior4ar.exact_core_identity(pre_w, pre_c)
        require(env_w.last_shield_decision == env_c.last_shield_decision
                and env_w.shield.belief_lower == env_c.shield.belief_lower
                and env_w.shield.belief_upper == env_c.shield.belief_upper,
                "Exact paired shield/action history before branch")
        prefix_cost = -float(pre_w["episode_return"])
        require(prefix_cost == -float(pre_c["episode_return"]), "Exact prefix cumulative cost")
        pre_perception = perception_audit_record(env_w, pre_w)
        anchor_substream = int(assets.perception.perception_substream_seed(
            PERCEPTION_SEED, year, tid, anchor_day))

        out_w = prior4ar.step_and_audit(env_w, WAIT)
        out_c = prior4ar.step_and_audit(env_c, CLEAN)
        dec_w, dec_c = out_w[4], out_c[4]
        require(dec_w["proposed_action"] == WAIT and dec_w["executed_action"] == WAIT
                and dec_w["proposed_safe"] is True and dec_w["clean_safe"] is True
                and dec_w["intervention"] is False,
                "Anchor WAIT is legal, safe, and unintervened")
        require(dec_c["proposed_action"] == CLEAN and dec_c["executed_action"] == CLEAN
                and dec_c["proposed_safe"] is True and dec_c["clean_safe"] is True
                and dec_c["intervention"] is False,
                "Anchor CLEAN is legal, safe, and unintervened")

        acc_w, acc_c = new_accumulator(), new_accumulator()
        anchor_w = add_step(acc_w, out_w)
        anchor_c = add_step(acc_c, out_c)
        require(anchor_w["day_index"] == anchor_c["day_index"] == anchor_day,
                "Paired anchor day identity")
        immediate = {
            "WAIT": {key: float(anchor_w[key]) for key in
                     ("soiling_cost", "cleaning_cost", "total_cost")},
            "CLEAN": {key: float(anchor_c[key]) for key in
                      ("soiling_cost", "cleaning_cost", "total_cost")},
        }
        immediate["delta_total_cost_CLEANminusWAIT"] = (
            immediate["CLEAN"]["total_cost"] - immediate["WAIT"]["total_cost"])
        post_state = {
            "L_pre_common": float(anchor_w["L_pre"]),
            "WAIT_L_post": float(anchor_w["L_post"]),
            "CLEAN_L_post": float(anchor_c["L_post"]),
            "WAIT_L_next": float(anchor_w["L_next"]),
            "CLEAN_L_next": float(anchor_c["L_next"]),
        }
        next_w, next_c = prior4ar.core_snapshot(env_w), prior4ar.core_snapshot(env_c)
        next_perception_w = perception_audit_record(env_w, next_w)
        next_perception_c = perception_audit_record(env_c, next_c)
        next_substream = int(assets.perception.perception_substream_seed(
            PERCEPTION_SEED, year, tid, anchor_day + 1))

        future_w, future_c = new_accumulator(), new_accumulator()
        for day in range(anchor_day + 1, 365):
            step_w = prior4ar.step_and_audit(env_w, WAIT)
            step_c = prior4ar.step_and_audit(env_c, WAIT)
            add_step(acc_w, step_w)
            add_step(acc_c, step_c)
            add_step(future_w, step_w)
            add_step(future_c, step_c)
            require(step_w[2] == (day == 364) and step_c[2] == (day == 364),
                    "Exact paired continuation horizon")

        branch_w = finalize_accumulator(acc_w, anchor_day)
        branch_c = finalize_accumulator(acc_c, anchor_day)
        future_w = finalize_accumulator(future_w, anchor_day)
        future_c = finalize_accumulator(future_c, anchor_day)
        delta_soiling = branch_c["soiling_cost"] - branch_w["soiling_cost"]
        delta_cleaning = branch_c["cleaning_cost"] - branch_w["cleaning_cost"]
        delta_total = branch_c["total_cost"] - branch_w["total_cost"]
        decomposition_residual = delta_total - (delta_soiling + delta_cleaning)
        require(abs(decomposition_residual) <= atol,
                "Delta total equals delta soiling plus delta cleaning")
        require(classify(delta_total, atol) in CLASSIFICATIONS, "Known scientific class")
        for ledger in (ledger_w, ledger_c):
            require_development_ledger(ledger)
        rec_w = env_w._shield_audit_snapshot()["episode_records"][-1]
        rec_c = env_c._shield_audit_snapshot()["episode_records"][-1]
        require(rec_w["completed"] and rec_c["completed"]
                and rec_w["steps"] == rec_c["steps"] == 365,
                "Both paired branches complete")
        require(branch_w["clean_count"] == len([day for day in branch_w["forced_clean_days"]])
                and branch_c["clean_count"] == 1 + len(branch_c["forced_clean_days"]),
                "Suffix CLEAN partition: forced only plus anchor voluntary CLEAN")

        return {
            "year": year,
            "trajectory_id": tid,
            "stratum": selection["stratum"],
            "anchor_day": anchor_day,
            "anchor_date": anchor_w["date"],
            "selection_sha256": selection["sha256"],
            "selection_modulo_index_zero_based": selection["modulo_index_zero_based"],
            "prefix": {
                "history_rule": "proposal WAIT; frozen shield may force CLEAN",
                "prefix_day_count": anchor_day,
                "prefix_cumulative_total_cost": prefix_cost,
                "exact_core_state_cost_observation_action_history": True,
            },
            "common_random_numbers": {
                **input_hashes,
                "environment_root": int(pre_w["environment_root"]),
                "perception_seed": int(pre_w["perception_seed"]),
                "perception_substream_rule":
                    "SeedSequence([perception_seed, year_code, trajectory_id, day_index, 2202])",
                "anchor_substream_seed": anchor_substream,
                "next_day_substream_seed_both_branches": next_substream,
                "same_substream_after_state_fork": True,
                "q50_width_may_differ_after_state_fork": True,
            },
            "anchor_pre_perception_policy_input": pre_perception,
            "anchor_actions": {
                "WAIT": dec_w,
                "CLEAN": dec_c,
                "both_legal_safe_and_no_intervention": True,
            },
            "immediate_action_cost": immediate,
            "true_state_audit_only": post_state,
            "next_day_perception_audit_only": {
                "WAIT": next_perception_w,
                "CLEAN": next_perception_c,
            },
            "future_after_anchor": {"WAIT": future_w, "CLEAN": future_c},
            "anchor_through_terminal": {"WAIT": branch_w, "CLEAN": branch_c},
            "delta_soiling_cost_CLEANminusWAIT": delta_soiling,
            "delta_cleaning_cost_CLEANminusWAIT": delta_cleaning,
            "delta_total_cost_CLEANminusWAIT": delta_total,
            "delta_decomposition_residual": decomposition_residual,
            "delta_decomposition_within_atol": abs(decomposition_residual) <= atol,
            "classification": classify(delta_total, atol),
            "classification_is_structural_gate": False,
        }
    finally:
        env_w.close()
        env_c.close()


def flatten_counterfactual(row):
    w = row["anchor_through_terminal"]["WAIT"]
    c = row["anchor_through_terminal"]["CLEAN"]
    fw = row["future_after_anchor"]["WAIT"]
    fc = row["future_after_anchor"]["CLEAN"]
    immediate = row["immediate_action_cost"]
    state = row["true_state_audit_only"]
    perception = row["next_day_perception_audit_only"]
    flat = {
        "year": row["year"],
        "trajectory_id": row["trajectory_id"],
        "stratum": row["stratum"],
        "anchor_day": row["anchor_day"],
        "anchor_date": row["anchor_date"],
        "selection_sha256": row["selection_sha256"],
        "prefix_cumulative_total_cost": row["prefix"]["prefix_cumulative_total_cost"],
        "anchor_q50": row["anchor_pre_perception_policy_input"]["q50"],
        "anchor_width_audit_only": row["anchor_pre_perception_policy_input"]["width"],
        "L_pre_common_audit_only": state["L_pre_common"],
        "WAIT_L_post_audit_only": state["WAIT_L_post"],
        "CLEAN_L_post_audit_only": state["CLEAN_L_post"],
        "WAIT_L_next_audit_only": state["WAIT_L_next"],
        "CLEAN_L_next_audit_only": state["CLEAN_L_next"],
        "WAIT_next_q50_audit_only": perception["WAIT"]["q50"],
        "CLEAN_next_q50_audit_only": perception["CLEAN"]["q50"],
        "WAIT_next_width_audit_only": perception["WAIT"]["width"],
        "CLEAN_next_width_audit_only": perception["CLEAN"]["width"],
        "WAIT_immediate_soiling_cost": immediate["WAIT"]["soiling_cost"],
        "WAIT_immediate_cleaning_cost": immediate["WAIT"]["cleaning_cost"],
        "WAIT_immediate_total_cost": immediate["WAIT"]["total_cost"],
        "CLEAN_immediate_soiling_cost": immediate["CLEAN"]["soiling_cost"],
        "CLEAN_immediate_cleaning_cost": immediate["CLEAN"]["cleaning_cost"],
        "CLEAN_immediate_total_cost": immediate["CLEAN"]["total_cost"],
        "WAIT_future_soiling_cost": fw["soiling_cost"],
        "WAIT_future_cleaning_cost": fw["cleaning_cost"],
        "WAIT_future_total_cost": fw["total_cost"],
        "CLEAN_future_soiling_cost": fc["soiling_cost"],
        "CLEAN_future_cleaning_cost": fc["cleaning_cost"],
        "CLEAN_future_total_cost": fc["total_cost"],
        "WAIT_J_anchor_to_terminal": w["total_cost"],
        "CLEAN_J_anchor_to_terminal": c["total_cost"],
        "WAIT_soiling_cost_anchor_to_terminal": w["soiling_cost"],
        "CLEAN_soiling_cost_anchor_to_terminal": c["soiling_cost"],
        "WAIT_cleaning_cost_anchor_to_terminal": w["cleaning_cost"],
        "CLEAN_cleaning_cost_anchor_to_terminal": c["cleaning_cost"],
        "WAIT_CLEAN_count_anchor_to_terminal": w["clean_count"],
        "CLEAN_CLEAN_count_anchor_to_terminal": c["clean_count"],
        "WAIT_forced_CLEAN_count_anchor_to_terminal": w["forced_clean_count"],
        "CLEAN_forced_CLEAN_count_anchor_to_terminal": c["forced_clean_count"],
        "WAIT_forced_CLEAN_days_json": json.dumps(w["forced_clean_days"], separators=(",", ":")),
        "CLEAN_forced_CLEAN_days_json": json.dumps(c["forced_clean_days"], separators=(",", ":")),
        "WAIT_forced_CLEAN_dates_json": json.dumps(w["forced_clean_dates"], separators=(",", ":")),
        "CLEAN_forced_CLEAN_dates_json": json.dumps(c["forced_clean_dates"], separators=(",", ":")),
        "WAIT_next_forced_CLEAN_day": w["next_forced_clean_day"],
        "CLEAN_next_forced_CLEAN_day": c["next_forced_clean_day"],
        "WAIT_next_forced_CLEAN_date": w["next_forced_clean_date"],
        "CLEAN_next_forced_CLEAN_date": c["next_forced_clean_date"],
        "WAIT_days_to_next_forced_CLEAN": w["days_to_next_forced_clean"],
        "CLEAN_days_to_next_forced_CLEAN": c["days_to_next_forced_clean"],
        "delta_soiling_cost_CLEANminusWAIT": row["delta_soiling_cost_CLEANminusWAIT"],
        "delta_cleaning_cost_CLEANminusWAIT": row["delta_cleaning_cost_CLEANminusWAIT"],
        "delta_total_cost_CLEANminusWAIT": row["delta_total_cost_CLEANminusWAIT"],
        "delta_decomposition_residual": row["delta_decomposition_residual"],
        "classification": row["classification"],
    }
    require(tuple(flat) == PAIR_FIELDS, "Stable paired-counterfactual CSV schema")
    return flat


def run_formal():
    require(not STAGE_DIRECTORY.exists(), "Exclusive new stage directory; no overwrite/resume/retry")
    require(git("status", "--porcelain", "--untracked-files=all") == "",
            "Committed unchanged clean worktree required")
    head = git("rev-parse", "HEAD")
    source_records = verify_sources(head)
    authority_records = verify_authorities(head)
    self_blob = git("rev-parse", f"{head}:{SELF}")
    require(self_blob == git("hash-object", f"--path={SELF}", SELF)
            == git("rev-parse", f":{SELF}"), "This source committed unchanged")
    static = static_contract()

    import paper2_shielded_perception_ppo_runner_v1 as runner
    import run_paper2_stage2d4ar_shield_delegation_action_identifiability_recovery_v1 as prior4ar

    require(all(prior4ar.static_contract().values()), "Frozen 4AR static contract")
    protocol = runner.get_protocol()
    atol = float(protocol.selection.atol)
    require(atol == ATOL and float(protocol.selection.rtol) == RTOL
            and protocol.decision_reward_steps == 365
            and protocol.natural_transitions == 364
            and dict(protocol.actions) == {WAIT: "WAIT", CLEAN: "CLEAN"}
            and dict(protocol.observations)["Point"] == ("q50", "sin_DOY", "cos_DOY"),
            "Frozen horizon/action/Point/tolerance protocol")
    development = runner.partition("DEVELOPMENT")
    require(tuple(development.years) == ("YEAR1", "YEAR2")
            and tuple(development.roots) == (1020001, 1030001)
            and tuple(development.trajectory_ids) == tuple(range(300)),
            "Exact 600-episode DEVELOPMENT population")

    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    failure_context = {"phase": "CREATE_FORMAL_OUTPUT_DIRECTORY"}
    try:
        OUTPUT.mkdir(exist_ok=False)
        failure_context = {"phase": "FORMAL_EXECUTION_AFTER_OUTPUT_DIRECTORY_CREATED"}
        assets = runner.load_assets()
        assets.ensure_perception()
        baselines = {}
        for year, tid in EPISODES:
            row = prior4ar.scan_baseline_episode(runner, assets, year, tid)
            require(row["year"] == year and row["trajectory_id"] == tid
                    and set(row["forced_days"]).isdisjoint(row["free_wait_days"])
                    and sorted(row["forced_days"] + row["free_wait_days"]) == list(range(364)),
                    "Exact nonterminal baseline decision partition")
            baselines[(year, tid)] = row

        selections = selection_rows(baselines)
        selected = [row for row in selections if row["selection_status"] == "SELECTED"]
        empty = [row for row in selections if row["selection_status"] == "EMPTY"]
        require(0 <= len(selected) <= 20 and len(selected) + len(empty) == 20,
                "At most twenty anchors; EMPTY retained")
        require(len({(row["year"], row["trajectory_id"], row["stratum"])
                     for row in selections}) == 20, "Unique episode-stratum rows")

        pairs = []
        for selection in selected:
            key = (selection["year"], int(selection["trajectory_id"]))
            pairs.append(paired_counterfactual(
                prior4ar, runner, assets, baselines[key], selection, atol))
        flat_pairs = [flatten_counterfactual(row) for row in pairs]
        require(len(pairs) == len(selected)
                and len({(row["year"], row["trajectory_id"], row["stratum"])
                         for row in pairs}) == len(pairs), "One pair per selected anchor")
        require(all(row["delta_decomposition_within_atol"] for row in pairs),
                "Every pair passes cost decomposition")

        pair_status = paired_feasibility_status(len(selected), len(pairs))
        failure_context.update({
            "selected_anchor_count": len(selected),
            "paired_counterfactual_count": len(pairs),
            **pair_status,
        })
        counts = {label: sum(row["classification"] == label for row in pairs)
                  for label in CLASSIFICATIONS}
        feasibility = {
            "stage": STAGE,
            "pre_registered_episode_count": 4,
            "episode_stratum_count": 20,
            "selected_anchor_count": len(selected),
            "empty_stratum_count": len(empty),
            "paired_counterfactual_count": len(pairs),
            "classification_counts_descriptive_only": counts,
            "all_cost_decompositions_within_atol": bool(pairs) and all(
                row["delta_decomposition_within_atol"] for row in pairs),
            "maximum_absolute_decomposition_residual": (
                max(abs(row["delta_decomposition_residual"]) for row in pairs)
                if pairs else None
            ),
            **pair_status,
            "scientific_results_are_structural_pass_gates": False,
            "claim_boundary": (
                "Four fixed DEVELOPMENT episodes test replay and accounting feasibility only; "
                "they do not estimate the 600-episode opportunity rate."
            ),
        }
        protocol_manifest = {
            "stage": STAGE,
            "mode": "formal",
            "role": "DEVELOPMENT FIXED-CONTINUATION COUNTERFACTUAL FEASIBILITY",
            "episodes_fixed_before_new_counterfactual_outcomes": [
                {"year": year, "trajectory_id": tid} for year, tid in EPISODES],
            "nonterminal_anchor_days": [0, 363],
            "strata": [{"label": label, "start_day_inclusive": start,
                         "end_day_inclusive": end} for label, start, end in STRATA],
            "selection": {
                "candidate_definition":
                    "baseline nonterminal day where WAIT and CLEAN are both shield-safe; baseline WAIT has no intervention",
                "candidate_sort": "unique integer day ascending",
                "salt": HASH_SALT,
                "payload": "salt\\nyear\\ntrajectory_id\\nstratum_label\\nstart\\nend\\ncomma_joined_sorted_candidates\\n",
                "digest": "SHA-256 over UTF-8 payload",
                "integer": "entire 256-bit digest interpreted base-16 big-endian",
                "modulo": "integer modulo eligible_count; zero-based index into sorted candidates",
                "empty_rule": "record EMPTY; no replacement and no rule change",
                "outcome_blind":
                    "selection does not read counterfactual cost, reward, future latent state, or scientific classification",
            },
            "branches": {
                "W": "WAIT at anchor, then WAIT proposals plus frozen shield",
                "C": "CLEAN at anchor, then WAIT proposals plus frozen shield",
                "prefix": "fresh replay from episode start with WAIT proposals plus frozen shield",
                "horizon": "anchor day through reward-only day 364 inclusive",
            },
            "delta_J": "J_CLEANbranch - J_WAITbranch",
            "classification_atol": atol,
            "classification_rtol": RTOL,
            "atol_relation_to_4AR": "identical to frozen protocol.selection.atol used by 4AR",
            "fixed_continuation_not_optimal_Q": True,
            "terminal_rule": "day 364 reward-only; WAIT proposal passes through; never an anchor",
            "true_latent_use": "counterfactual physics and post-hoc audit only; never anchor selection or policy input",
            "postfork_perception":
                "same day-level perception substream; q50/width may differ because latent states differ",
            "multiple_anchors_within_episode_are_correlated": True,
            "future_formal_statistics_must_cluster_by_episode": True,
            "no_episode_anchor_seed_reselection_after_outcomes": True,
            "no_reward_physics_perception_shield_mask_or_PPO_change": True,
            "PPO_training_runs": 0,
            "PPO_model_loads": 0,
            "formal_RL_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
        }
        baseline_payload = {
            "rule": "always propose WAIT; original frozen shield forces CLEAN when WAIT unsafe",
            "episodes": [
                {
                    "year": year,
                    "trajectory_id": tid,
                    "J_total_full_episode": baselines[(year, tid)]["J_total"],
                    "forced_CLEAN_days": baselines[(year, tid)]["forced_days"],
                    "free_choice_nonterminal_days": baselines[(year, tid)]["free_wait_days"],
                    "first_forced_CLEAN_day": baselines[(year, tid)]["first_forced_day"],
                    "forced_CLEAN_count": baselines[(year, tid)]["forced_clean_count"],
                    "free_WAIT_count_including_terminal": baselines[(year, tid)]["free_wait_count"],
                } for year, tid in EPISODES
            ],
        }
        provenance = {
            "HEAD": head,
            "new_source_path": SELF,
            "new_source_git_blob": self_blob,
            "new_source_sha256": sha_file(ROOT / SELF),
            "frozen_source_records": source_records,
            "frozen_authorities": authority_records,
            "development_population": {
                "years": list(development.years),
                "environment_roots": list(development.roots),
                "trajectory_ids": [0, 299],
                "episode_count": 600,
                "perception_seed": PERCEPTION_SEED,
            },
            "protected_roles_used": [],
        }
        selection_fields = list(selections[0])
        pair_fields = list(PAIR_FIELDS)
        content = {
            "protocol_manifest.json": encode(protocol_manifest),
            "source_artifact_provenance.json": encode(provenance),
            "episode_baseline_selection.json": encode(baseline_payload),
            "anchor_selection.csv": csv_bytes(selections, selection_fields),
            "paired_counterfactuals.json": encode(pairs),
            "paired_counterfactuals.csv": csv_bytes(flat_pairs, pair_fields),
            "feasibility_summary.json": encode(feasibility),
        }
        for name, payload in content.items():
            write_exclusive(OUTPUT / name, payload)
        hashes = {name: sha_bytes(payload) for name, payload in content.items()}
        output_hashes = encode({
            "hashes": hashes,
            "exclusions": {
                "output_hashes.json": "SHA-256 pinned in audit_summary; avoid self-reference",
                "audit_summary.json": "published last after every structural validation",
            },
        })
        write_exclusive(OUTPUT / "output_hashes.json", output_hashes)

        expected = set(content) | {"output_hashes.json"}
        actual = {path.name for path in OUTPUT.iterdir() if path.is_file()}
        require(actual == expected, "Exact pre-audit output inventory")
        require(all(sha_file(OUTPUT / name) == digest for name, digest in hashes.items()),
                "Output hashes verified")
        require(git("rev-parse", "HEAD") == head
                and git("status", "--porcelain", "--untracked-files=all") == "",
                "HEAD and worktree unchanged during formal execution")
        verify_sources(head)
        verify_authorities(head)

        gates = {
            "static_contract_pass": all(static.values()),
            "frozen_sources_SHA256_and_git_blobs_exact": True,
            "frozen_4AR_4B_0AR_authorities_and_outputs_exact": True,
            "exact_four_pre_registered_DEVELOPMENT_episodes": True,
            "exact_five_contiguous_strata_per_episode": True,
            "outcome_blind_SHA256_modulo_selection": True,
            "empty_strata_not_replaced": True,
            "at_least_one_complete_legal_counterfactual_pair":
                pair_status["technical_feasibility"],
            "fresh_identical_prefix_replay_for_every_pair":
                pair_status["technical_feasibility"],
            "both_anchor_actions_safe_and_unintervened":
                pair_status["technical_feasibility"],
            "same_exogenous_CRN_and_perception_substream_rule":
                pair_status["technical_feasibility"],
            "complete_anchor_to_terminal_cost_decomposition":
                pair_status["technical_feasibility"],
            "all_delta_decompositions_within_atol": bool(pairs) and all(
                row["delta_decomposition_within_atol"] for row in pairs),
            "scientific_classifications_not_PASS_gates": True,
            "zero_PPO_training_or_loading": True,
            "zero_protected_role_access": True,
            "outputs_hash_verified": True,
        }
        require(all(gates.values()), "All structural feasibility gates pass")
        audit = {
            "stage": STAGE,
            "mode": "formal",
            "stage_pass": True,
            "scientific_status": STATUS,
            "HEAD": head,
            "gates": gates,
            "failed_gates": [],
            "pre_registered_episode_count": 4,
            "episode_stratum_count": 20,
            "selected_anchor_count": len(selected),
            "empty_stratum_count": len(empty),
            "paired_counterfactual_count": len(pairs),
            "classification_counts_descriptive_only": counts,
            "classification_atol": atol,
            "PPO_training_runs": 0,
            "PPO_model_loads": 0,
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
            "claim_boundary": (
                "Paired fixed-continuation action gap under WAIT-proposal-plus-shield; "
                "not Q*, not a 600-episode opportunity-rate estimate."
            ),
            "output_sha256": {
                **hashes,
                "output_hashes.json": sha_bytes(output_hashes),
            },
            "audit_summary_published_last": True,
        }
        write_exclusive(OUTPUT / "audit_summary.json", encode(audit))
        return audit
    except Exception as exc:
        if OUTPUT.is_dir() and not (OUTPUT / "execution_failure.json").exists():
            try:
                write_exclusive(OUTPUT / "execution_failure.json", encode({
                    "stage": STAGE,
                    "stage_pass": False,
                    "HEAD": head,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "failure_context": failure_context,
                    "action": (
                        "STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/EPISODE OR ANCHOR "
                        "REPLACEMENT; NO TRAINING/TUNING"
                    ),
                }))
            except Exception as evidence_exc:
                raise RuntimeError(
                    "Formal execution failed and failure evidence could not be preserved: "
                    + str(evidence_exc)
                ) from exc
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
