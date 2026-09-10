#!/usr/bin/env python
"""P2-1D-3-v1: Final Reward Contract and Numerical Sanity Audit.

Explicit --mode execution only. No tuning, selection, training or perception.
Runtime Git calls are read-only provenance checks; this file must first be
committed unchanged. Failed runs never publish a successful audit or overwrite
an existing output directory. The user-specified checkpoint is literal: a
malformed or unavailable identifier fails closed, without guessing a correction.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STAGE = "P2-1D-3-v1"
TITLE = "Final Reward Contract and Numerical Sanity Audit"
ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage1d3_final_reward_contract_audit_v1.py"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUTPUT = BASE / "p2_1d_3_final_reward_contract_audit_v1"
PRE = BASE / "p2_1d_2b_reward_behaviour_audit_v1/formal"
# Validation must not silently substitute the different HEAD in the manifest.
CHECKPOINT = "0abece7317e8b10c9855803905de9cc7e33ee4c4"
PREDECESSOR = "experiments/run_paper2_stage1d2b_reward_behaviour_audit_v1.py"
PREDECESSOR_BLOB = "f4400a94f5df4c46d975a3944182f084206ef37a"
FORMAL_HASHES = {
    "audit_summary.json": "99aba74d388b11d02a29c85f775b94f523e38d4d3372f864ec57c11e9e4aa0ac",
    "main_lambda_manifest.json": "12ce7a5eac2ef1596a4ad3409b7a446fde71ebfe7d58860a535526503b8226db",
    "trajectory_policy_base_metrics.csv": "baac9aec1108d387d0afaa94e083cc142ac13ea6a901ab94f70c6e554e2d5e47",
    "lambda_policy_summary.csv": "ce5a43f03495ec140f0e83e56ec872c713de20cf26aebcaa5b236a73e39f4300",
    "lambda_classification_summary.csv": "d9ceca392f4054e8a3170241d36d1ccb9e7f69839b259c9aa1ca78f9c10c1841",
    "protocol_manifest.json": "6f0c8f573fc29149f5b0d04615e6ebb66991c39724771945af1eda76e99a8581",
}
INPUTS = {
    "energy": (BASE / "p2_1d_1a_daily_ghi_energy_value_audit_v1/daily_energy_value_proxy.csv",
               "443a023c42e9b0db9cf888c120eba53b5ad092779ea0875cd6de32826414c155"),
    "ledger": (BASE / "p2_1a_timeline_intervention_audit_v1/environment_master_ledger.csv",
               "7906680af03d94f4989efc16787d9f5431bfb53b87bd2eb0f8586c6a340fde04"),
    "transitions": (BASE / "p2_1a_timeline_intervention_audit_v1/transition_audit.csv",
                    "195a23f1bb893b4b92c8f567e17824007c351886d2cadd1600cd5f3022d7d75d"),
}
HELPERS = {
    "predecessor": (PREDECESSOR, PREDECESSOR_BLOB),
    "mechanics": ("experiments/run_paper2_stage1c2a_clean_mechanics_operational_support_audit_v1.py",
                  "3be8472e8963e0ee97b6173d54916a92dbc05277"),
    "natural": ("experiments/run_paper2_stage1c1r_b_rain_repair_candidate_validation_v1.py",
                "c1430b7b4ccac776433294489984b07af0c82c5f"),
}
EXPECTED_MAIN_LAMBDA = 0.19925428989934618  # Integrity assertion only.
ETA = 0.95
REWARD_SCALE = 1.0
ATOL = 1e-12
RTOL = 0.0  # Absolute-only tolerance; never round values for comparisons.
ENV_ROOTS = {"YEAR1": 1020001, "YEAR2": 1030001}
TRACE_POLICIES = ["NEVER_CLEAN", "DAILY_CLEAN", "TRUE_THRESHOLD_0.05"]
KEYS = ["year", "trajectory_id", "policy"]
METRICS = ["J_soiling", "N_clean", "mean_L_pre", "mean_L_post", "max_L_pre"]
TRACE_COLUMNS = ["year", "trajectory_id", "day_index", "date", "policy",
                 "L_pre", "action", "L_post", "E_clean_GHI", "soiling_cost",
                 "cleaning_cost", "total_cost", "raw_reward", "reward_scale",
                 "final_reward", "rain_affected_transition", "transition_executed", "L_next"]
GATES = dict(enumerate([
    "P2_1D_2B_checkpoint_ancestry", "predecessor_script_blob",
    "six_frozen_formal_output_hashes", "frozen_main_lambda_contract",
    "frozen_energy_proxy_integrity", "frozen_environment_provenance",
    "CRN_exact_regeneration", "trace_population_exact", "trace_policy_set_exact",
    "CLEAN_WAIT_L_post_identity", "daily_reward_identity", "annual_cumulative_identity",
    "frozen_formal_base_metric_regression", "multiplier_2_formal_aggregate_regression",
    "latent_continuity_2022_06_29", "information_role_no_GHI_leakage",
    "reward_theoretical_bound", "reward_scale_1_exact", "no_prohibited_retuning",
    "output_completeness", "current_stage_script_committed_unchanged",
], start=1))


def check(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def gate(audit, number, condition, detail):
    audit["gates"][GATES[number]] = bool(condition)
    audit["gate_details"][f"G{number}"] = detail
    check(condition, f"G{number} {GATES[number]}: {detail}")


def close(actual, expected):
    return bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True, timeout=30).stdout.strip()


def script_record(path, head, expected=None):
    committed = git("rev-parse", "--verify", f"{head}:{path}")
    working = git("hash-object", f"--path={path}", path)
    check(committed == working and (expected is None or working == expected),
          f"Script must be committed unchanged with required blob: {path}")
    frozen = git("rev-parse", "--verify", f"{CHECKPOINT}:{path}") if expected else None
    check(expected is None or frozen == expected, f"Checkpoint helper blob: {path}")
    return {"path": path, "HEAD_blob": committed, "working_tree_blob": working,
            "checkpoint_blob": frozen, "sha256": sha((ROOT / path).read_bytes())}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(f"_p2_1d3_{name}", ROOT / path)
    check(spec is not None and spec.loader is not None, f"Module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def csv(data):
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", float_precision="round_trip")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def calendar_check(frame):
    check("date" in frame, "Missing date")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    check(len(frame) == 730 and not frame.date.duplicated().any()
          and pd.DatetimeIndex(frame.date).equals(pd.date_range("2021-08-09", "2023-08-08")),
          "Exact ordered 730-day calendar required")


def inputs(audit, manifest):
    # No output directory or dynamic helper import precedes these checks.
    check(len(CHECKPOINT) == 40 and all(c in "0123456789abcdef" for c in CHECKPOINT),
          "G1: user-specified frozen checkpoint is not a full 40-character Git SHA; no substitution allowed")
    head = git("rev-parse", "HEAD")
    commit = git("rev-parse", "--verify", f"{CHECKPOINT}^{{commit}}")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", CHECKPOINT, head],
                              cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
    gate(audit, 1, commit == CHECKPOINT and ancestry.returncode == 0,
         {"checkpoint": CHECKPOINT, "HEAD": head, "ancestry_returncode": ancestry.returncode})
    record = script_record(PREDECESSOR, head, PREDECESSOR_BLOB)
    gate(audit, 2, record["checkpoint_blob"] == PREDECESSOR_BLOB, record)
    own = script_record(SELF, head)
    gate(audit, 21, own["HEAD_blob"] == own["working_tree_blob"], own)
    manifest.update(current_git_head=head, current_stage_script=own)
    payloads, records = {}, {}
    for name, digest in FORMAL_HASHES.items():
        data = (PRE / name).read_bytes()
        check(sha(data) == digest, f"G3 frozen formal SHA256 mismatch: {name}")
        payloads[name] = data
        records[name] = {"path": str(PRE / name), "sha256": sha(data), "expected_sha256": digest}
    prior = json.loads(payloads["audit_summary.json"])
    gate(audit, 3, prior.get("stage_pass") is True and prior.get("failed_gates") == []
         and prior.get("scientific_status") == "ROBUST_INTERIOR_MAIN_LAMBDA_SELECTED", records)
    manifest["frozen_formal_provenance"] = records
    main = json.loads(payloads["main_lambda_manifest.json"])
    gate(audit, 4, main.get("main_multiplier") == 2
         and main.get("main_lambda_c") == EXPECTED_MAIN_LAMBDA
         and main.get("selection_eligibility") == "support_validated_robust_interior"
         and main.get("near_optimal_tolerance") == 0.005,
         {key: main.get(key) for key in ("main_multiplier", "main_lambda_c",
                                        "selection_eligibility", "near_optimal_tolerance")})
    lambda_main = main["main_lambda_c"]  # Only source of the settlement parameter.
    frozen = json.loads(payloads["protocol_manifest.json"])
    snapshots = {}
    manifest["input_provenance"] = {}
    for name, (path, digest) in INPUTS.items():
        snapshots[name] = path.read_bytes()
        check(sha(snapshots[name]) == digest, f"Frozen {name} SHA256 mismatch")
        check(frozen["input_provenance"][name]["sha256"] == digest, f"Frozen input lineage: {name}")
        manifest["input_provenance"][name] = {"path": str(path), "sha256": digest}
    energy = csv(snapshots["energy"])
    calendar_check(energy)
    e = energy.E_clean_GHI.to_numpy(float)
    gate(audit, 5, np.isfinite(e).all() and (e > 0).all() and close(e.mean(), 1.0),
         {"days": len(e), "mean": float(e.mean()), "role": "REALIZED_REWARD_SETTLEMENT_ONLY"})
    records = {name: script_record(path, head, blob) for name, (path, blob) in HELPERS.items()}
    for name in ("mechanics", "natural"):
        check(records[name]["working_tree_blob"] == frozen["environment_helper_provenance"][name]["working_tree_blob"]
              and records[name]["sha256"] == frozen["environment_helper_provenance"][name]["sha256"],
              f"Frozen helper lineage: {name}")
    modules = {name: load_module(name, path) for name, (path, _) in HELPERS.items()}
    predecessor, mechanics = modules["predecessor"], modules["mechanics"]
    ledger = mechanics.load_ledger(io.BytesIO(snapshots["ledger"]))
    calendar_check(ledger)
    raw_ledger = csv(snapshots["ledger"])
    calendar_check(raw_ledger)
    raw_ledger["bridge_valid"] = mechanics.parse_bool_series(raw_ledger.bridge_valid, "bridge_valid")
    transitions = mechanics.load_transitions(io.BytesIO(snapshots["transitions"]))
    check(len(transitions) == 729 and np.array_equal(transitions.transition_index, np.arange(729))
          and np.array_equal(transitions.source_date, ledger.date.iloc[:-1])
          and np.array_equal(transitions.dest_date, ledger.date.iloc[1:]), "Transition alignment")
    context = mechanics.attach_context(transitions, ledger)
    dry = context.loc[context.transition_class.eq("DRY_NATURAL"), "delta_L_power_proxy"].to_numpy(float)
    rain_rows = context.loc[context.transition_class.eq("RAIN_AFFECTED")]
    r3 = mechanics.fit_r3(rain_rows)
    parameters = {"dry": "D0_GLOBAL", "rain": "R3_OLS_RESIDUAL",
                  "intercept": r3["intercept"], "slope": r3["slope"],
                  "dry_pool_sha256": sha(dry.tobytes()),
                  "residual_pool_sha256": sha(r3["residuals"].tobytes())}
    gate(audit, 6, parameters == frozen["environment_parameters"]
         and frozen["current_git_head"] == CHECKPOINT
         and frozen["environment_roots"] == predecessor.ENV_ROOTS == ENV_ROOTS
         and frozen["ETA"] == predecessor.ETA == ETA
         and frozen["environment_population"] == "DEVELOPMENT_ONLY"
         and frozen["formal_environment_roots_accessed"] is False
         and len(dry) == 494 and len(rain_rows) == 201
         and np.isfinite(dry).all() and np.isfinite(r3["residuals"]).all(), parameters)
    manifest.update(environment_helper_provenance=records, environment_parameters=parameters)
    return modules, ledger, raw_ledger, dry, r3, energy.set_index("date").E_clean_GHI, lambda_main, frozen, payloads


def settlement(E_clean_GHI, L_post, action, lambda_main):
    """Realized reward only; no policy decision or natural transition here."""
    soiling_cost = E_clean_GHI * L_post
    cleaning_cost = lambda_main * action
    total_cost = soiling_cost + cleaning_cost
    raw_reward = -total_cost
    final_reward = raw_reward
    return soiling_cost, cleaning_cost, total_cost, raw_reward, final_reward


def simulate(modules, ledger, dry, r3, energy, lambda_main, frozen, n, manifest, audit):
    mechanics, predecessor = modules["mechanics"], modules["predecessor"]
    rows, crn_records = [], {}
    for year, seed in ENV_ROOTS.items():
        check(seed == mechanics.scenario_seed(420001, year, "HISTORICAL_FIRST_DAY"), "Development root")
        calendar = mechanics.get_year_calendar(ledger, year)
        dates = pd.date_range("2021-08-09" if year == "YEAR1" else "2022-08-09", periods=365)
        check(pd.DatetimeIndex(calendar.date).equals(dates), "Year calendar exact")
        initial = float(calendar.L_power_proxy.iloc[0])
        reference = frozen["CRN"][year]
        check(bool(calendar.state_valid.iloc[0]) and np.isfinite(initial) and 0 <= initial <= 1
              and initial == reference["initial_L"] and seed == reference["root"], "Frozen initial state")
        bank, rain, report = mechanics.build_crn_indices(calendar, len(dry), len(r3["residuals"]), 300, seed)
        full = np.stack(bank)
        full_hash = sha(full.tobytes())
        check(full.shape == (364, 300) and report == reference["full_bank"]
              and report["same_seed_exact_reproduction"] is True
              and full_hash == reference["used_bank_sha256"] == report["index_hash"], "G7 frozen full CRN exact")
        expected_rain = calendar.rain_day.to_numpy(bool)
        check(np.array_equal(rain, expected_rain[:-1] | expected_rain[1:]), "Rain transition calendar")
        indices = full[:, :n].copy()
        indices.setflags(write=False)
        rain.setflags(write=False)
        values = energy.loc[calendar.date].to_numpy(float)
        values.setflags(write=False)
        used_hash, rain_hash = sha(indices.tobytes()), sha(rain.tobytes())
        usage = {}
        for policy in TRACE_POLICIES:
            pre = np.full(n, initial, dtype=float)
            for day in range(365):
                # The ONLY policy call receives policy, day_index and latent L_pre.
                actions = predecessor.action(policy, day, pre)
                post = np.where(actions, (1 - ETA) * pre, pre)
                soiling, cleaning, total, raw, final = settlement(values[day], post, actions, lambda_main)
                executed = day < 364
                following = (predecessor.natural_transition(modules["natural"], post, indices[day],
                             rain[day], dry, r3) if executed else np.full(n, np.nan))
                for trajectory in range(n):
                    rows.append((year, trajectory, day, dates[day].strftime("%Y-%m-%d"), policy,
                                 float(pre[trajectory]), int(actions[trajectory]), float(post[trajectory]),
                                 float(values[day]), float(soiling[trajectory]), float(cleaning[trajectory]),
                                 float(total[trajectory]), float(raw[trajectory]), REWARD_SCALE,
                                 float(final[trajectory]), bool(rain[day]) if executed else False,
                                 executed, float(following[trajectory])))
                if executed:
                    check(np.isfinite(following).all() and ((following >= 0) & (following <= 1)).all(),
                          "Finite projected natural next state")
                    pre = following
            usage[policy] = sha(indices.tobytes())
            check(usage[policy] == used_hash and sha(rain.tobytes()) == rain_hash, "Unchanged shared CRN")
        crn_records[year] = {"root": seed, "initial_L": initial, "full_bank": report,
                             "full_bank_sha256": full_hash, "used_ids": list(range(n)),
                             "used_bank_sha256": used_hash, "policy_bank_hashes": usage,
                             "rain_calendar_sha256": rain_hash}
    gate(audit, 7, len(crn_records) == 2, crn_records)
    manifest["CRN"] = crn_records
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


def validate_trace(trace, energy, lambda_main, n, audit):
    expected = {(year, tid, policy, day) for year in ENV_ROOTS for tid in range(n)
                for policy in TRACE_POLICIES for day in range(365)}
    actual = set(trace[KEYS + ["day_index"]].itertuples(index=False, name=None))
    gate(audit, 8, len(trace) == 2 * n * 3 * 365 and actual == expected,
         {"rows": len(trace), "years": list(ENV_ROOTS), "trajectory_ids": list(range(n)), "days": 365})
    gate(audit, 9, set(trace.policy) == set(TRACE_POLICIES), TRACE_POLICIES)
    finite_columns = ["L_pre", "L_post", "E_clean_GHI", "soiling_cost", "cleaning_cost",
                      "total_cost", "raw_reward", "reward_scale", "final_reward"]
    check(np.isfinite(trace[finite_columns].to_numpy(float)).all(), "All settlements and latent states finite")
    check(trace.action.isin([0, 1]).all(), "Binary action")
    gate(audit, 10, close(trace.loc[trace.action.eq(0), "L_post"], trace.loc[trace.action.eq(0), "L_pre"])
         and close(trace.loc[trace.action.eq(1), "L_post"], 0.05 * trace.loc[trace.action.eq(1), "L_pre"]),
         {"atol": ATOL, "rtol": RTOL, "eta": ETA})
    mapped_energy = energy.loc[pd.to_datetime(trace.date)].to_numpy(float)
    gate(audit, 11, np.array_equal(trace.E_clean_GHI.to_numpy(), mapped_energy)
         and close(trace.soiling_cost, trace.E_clean_GHI * trace.L_post)
         and close(trace.cleaning_cost, lambda_main * trace.action)
         and close(trace.total_cost, trace.soiling_cost + trace.cleaning_cost)
         and close(trace.raw_reward, -trace.total_cost)
         and close(trace.final_reward, trace.raw_reward), "Every row checked without rounding")
    terminal = trace.day_index.eq(364)
    check(np.array_equal(trace.transition_executed, ~terminal)
          and trace.loc[terminal, "L_next"].isna().all()
          and not trace.loc[terminal, "rain_affected_transition"].any()
          and np.isfinite(trace.loc[~terminal, "L_next"]).all(), "365 settlements, 364 transitions")
    for (year, _, _), block in trace.groupby(KEYS, sort=False):
        block = block.sort_values("day_index")
        dates = pd.date_range("2021-08-09" if year == "YEAR1" else "2022-08-09", periods=365)
        check(pd.DatetimeIndex(pd.to_datetime(block.date)).equals(dates)
              and np.array_equal(block.L_next.iloc[:-1], block.L_pre.iloc[1:]), "Unbroken latent episode")
    maximum = float(energy.max())
    wait_bound = maximum
    clean_bound = (1 - ETA) * maximum + lambda_main
    bound = max(wait_bound, clean_bound)
    gate(audit, 17, trace.L_pre.between(0, 1).all() and trace.L_post.between(0, 1).all()
         and trace.total_cost.ge(0).all() and trace.total_cost.le(bound + ATOL).all()
         and trace.final_reward.ge(-bound - ATOL).all() and trace.final_reward.le(0).all(),
         {"E_max": maximum, "WAIT_BOUND": wait_bound, "CLEAN_BOUND": clean_bound,
          "THEORETICAL_STEP_COST_BOUND": bound, "numerical_tolerance": ATOL})
    gate(audit, 18, REWARD_SCALE == 1.0 and trace.reward_scale.eq(1.0).all()
         and np.array_equal(trace.final_reward, trace.raw_reward), "Exact scale and reward equality")
    return {"E_max": maximum, "WAIT_BOUND": wait_bound, "CLEAN_BOUND": clean_bound,
            "THEORETICAL_STEP_COST_BOUND": bound}


def regress(trace, payloads, frozen, lambda_main, formal, audit):
    base = csv(payloads["trajectory_policy_base_metrics.csv"])
    expected = {(year, tid, policy) for year in ENV_ROOTS for tid in range(300)
                for policy in frozen["active_policies"]}
    check(base.columns.tolist() == KEYS + METRICS and len(base) == len(expected)
          and set(base[KEYS].itertuples(index=False, name=None)) == expected, "Frozen full base population")
    check(np.isfinite(base[METRICS].to_numpy(float)).all(), "Finite frozen base metrics")
    indexed = base.set_index(KEYS, verify_integrity=True)
    rows = []
    for key, block in trace.groupby(KEYS, sort=False):
        block = block.sort_values("day_index")
        reference = indexed.loc[key]
        measured = {"J_soiling": float(block.soiling_cost.to_numpy().sum()),
                    "N_clean": int(block.action.sum()), "mean_L_pre": float(block.L_pre.mean()),
                    "mean_L_post": float(block.L_post.mean()), "max_L_pre": float(block.L_pre.max())}
        sums = {"sum_soiling_cost": measured["J_soiling"], "sum_action": measured["N_clean"],
                "sum_total_cost": float(block.total_cost.sum()),
                "sum_raw_reward": float(block.raw_reward.sum()),
                "sum_final_reward": float(block.final_reward.sum())}
        # Frozen reference makes annual identities independent of trace construction.
        annual_pass = (len(block) == 365 and close(sums["sum_soiling_cost"], reference.J_soiling)
                       and sums["sum_action"] == reference.N_clean
                       and close(sums["sum_total_cost"], reference.J_soiling + lambda_main * reference.N_clean)
                       and close(sums["sum_total_cost"], measured["J_soiling"] + lambda_main * measured["N_clean"])
                       and close(sums["sum_raw_reward"], -sums["sum_total_cost"])
                       and close(sums["sum_final_reward"], sums["sum_raw_reward"]))
        row = dict(zip(KEYS, key))
        row.update(measured, **sums, annual_identity_pass=bool(annual_pass))
        for metric in METRICS:
            row[f"frozen_{metric}"] = float(reference[metric])
            row[f"delta_{metric}"] = float(measured[metric] - reference[metric])
            row[f"{metric}_pass"] = (measured[metric] == reference[metric] if metric == "N_clean"
                                     else close(measured[metric], reference[metric]))
        row["base_regression_pass"] = all(row[f"{metric}_pass"] for metric in METRICS)
        rows.append(row)
    regression = pd.DataFrame(rows)
    gate(audit, 12, regression.annual_identity_pass.all(), "All 365-day sums checked against frozen J_soiling/N_clean")
    gate(audit, 13, regression.base_regression_pass.all(),
         {"records": len(regression), "metrics": METRICS, "atol": ATOL, "rtol": RTOL})
    aggregate_rows = []
    if formal:
        # Parse only multiplier=2 rows. Other performance values remain opaque CSV
        # strings and are never converted, ranked, compared or used for selection.
        import csv as csv_module
        reader = csv_module.DictReader(io.StringIO(payloads["lambda_policy_summary.csv"].decode("utf-8-sig")))
        selected = [row for row in reader if float(row["multiplier"]) == 2
                    and row["policy"] in TRACE_POLICIES]
        required = {(scope, policy) for scope in ("YEAR1", "YEAR2", "JOINT") for policy in TRACE_POLICIES}
        check(len(selected) == 9 and {(row["scope"], row["policy"]) for row in selected} == required,
              "Exactly nine multiplier=2 reference records")
        for reference in selected:
            scope, policy = reference["scope"], reference["policy"]
            check(float(reference["lambda_c"]) == lambda_main, "Frozen summary main lambda")
            population = base.loc[base.policy.eq(policy)]
            if scope != "JOINT":
                population = population.loc[population.year.eq(scope)]
            check(len(population) == (600 if scope == "JOINT" else 300), "Full frozen aggregate population")
            means = population.groupby("year")[["J_soiling", "N_clean"]].mean().mean()
            calculated = {"mean_J_total": float(means.J_soiling + lambda_main * means.N_clean),
                          "mean_J_soiling": float(means.J_soiling), "mean_N_clean": float(means.N_clean)}
            for metric, actual in calculated.items():
                target = float(reference[metric])
                aggregate_rows.append({"scope": scope, "policy": policy, "multiplier": 2,
                                       "metric": metric, "actual": actual, "frozen": target,
                                       "delta": actual - target, "pass": close(actual, target),
                                       "population_count": len(population)})
        gate(audit, 14, all(row["pass"] for row in aggregate_rows), aggregate_rows)
    else:
        gate(audit, 14, True, {"applicable": False, "reason": "SMOKE_ONLY; full population aggregate regression is formal-only"})
    return regression, aggregate_rows


def continuity(trace, ledger, raw_ledger, energy, n, audit):
    date = pd.Timestamp("2022-06-29")
    observed = ledger.loc[ledger.date.eq(date)]
    bridge = raw_ledger.loc[raw_ledger.date.eq(date)]
    block = trace.loc[trace.year.eq("YEAR1") & trace.date.eq("2022-06-29")]
    expected = {(tid, policy) for tid in range(n) for policy in TRACE_POLICIES}
    passed = (len(observed) == len(bridge) == 1 and not bool(observed.state_valid.iloc[0])
              and not bool(bridge.bridge_valid.iloc[0]) and not np.isfinite(observed.L_power_proxy.iloc[0])
              and np.isfinite(energy.loc[date]) and energy.loc[date] > 0
              and len(block) == n * 3
              and set(block[["trajectory_id", "policy"]].itertuples(index=False, name=None)) == expected
              and np.isfinite(block[["L_pre", "L_post", "total_cost", "final_reward"]].to_numpy()).all()
              and block.transition_executed.all())
    detail = {"date": "2022-06-29", "trace_rows": len(block), "observed_state_interpolated": False,
              "settlement": "simulator-generated latent L_t + observed realized E_clean_GHI",
              "episode_interrupted": False, "date_deleted": False}
    gate(audit, 15, passed, detail)
    return detail


def static_contract(modules, audit):
    predecessor = modules["predecessor"]
    signature = tuple(inspect.signature(predecessor.action).parameters)
    own_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    simulation = next(node for node in own_tree.body if isinstance(node, ast.FunctionDef) and node.name == "simulate")
    action_calls = [node for node in ast.walk(simulation) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "action"]
    check(len(action_calls) == 1, "One restricted action call site")
    call = action_calls[0]
    call_ok = (isinstance(call.func.value, ast.Name) and call.func.value.id == "predecessor"
               and not call.keywords and len(call.args) == 3
               and all(isinstance(node, ast.Name) for node in call.args)
               and [node.id for node in call.args] == ["policy", "day", "pre"])
    action_tree = ast.parse(inspect.getsource(predecessor.action))
    loaded_names = {node.id for node in ast.walk(action_tree) if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)}
    gate(audit, 16, signature == ("policy", "day_index", "L_pre") and call_ok
         and loaded_names <= {"policy", "day_index", "L_pre", "np", "bool", "int", "float", "ValueError"},
         {"policy_signature": list(signature), "action_loaded_names": sorted(loaded_names),
          "E_clean_GHI_role": "REALIZED_REWARD_SETTLEMENT_ONLY", "lambda_role": "SETTLEMENT_ONLY"})
    helper_calls = {node.func.attr for node in ast.walk(own_tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "predecessor"}
    lambda_writes = [node for node in ast.walk(own_tree) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "lambda_main" for target in node.targets)]
    lambda_source_ok = (len(lambda_writes) == 1
                        and ast.unparse(lambda_writes[0].value) == "main['main_lambda_c']")
    gate(audit, 19, helper_calls == {"action", "natural_transition"} and lambda_source_ok
         and TRACE_POLICIES == ["NEVER_CLEAN", "DAILY_CLEAN", "TRUE_THRESHOLD_0.05"]
         and ENV_ROOTS == {"YEAR1": 1020001, "YEAR2": 1030001}
         and ETA == 0.95 and REWARD_SCALE == 1.0,
         {"predecessor_calls": sorted(helper_calls), "main_lambda_from_frozen_manifest_only": lambda_source_ok,
          "reward_tuning": False, "lambda_selection": False, "policy_selection": False,
          "scale_tuning": False, "training": False, "perception": False,
          "formal_environment_access": False, "currency_claim": False})


def distributions(trace):
    rows = []
    for policy in ["OVERALL"] + TRACE_POLICIES:
        population = trace if policy == "OVERALL" else trace.loc[trace.policy.eq(policy)]
        for action_label in ("ALL", "CLEAN", "WAIT"):
            block = population if action_label == "ALL" else population.loc[population.action.eq(int(action_label == "CLEAN"))]
            for reward in ("raw_reward", "final_reward"):
                values = block[reward].to_numpy(float)
                names = ["min", "P01", "P05", "P25", "median", "P75", "P95", "P99", "max"]
                quantiles = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1]) if len(values) else [np.nan] * 9
                rows.append({"policy": policy, "action_group": action_label, "reward": reward,
                             "count": len(values), **dict(zip(names, quantiles)),
                             "mean": float(values.mean()) if len(values) else np.nan,
                             "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan})
    return pd.DataFrame(rows)


def protocol(mode):
    return {"stage": STAGE, "title": TITLE, "mode": mode, "predecessor_checkpoint": CHECKPOINT,
            "environment_roots": ENV_ROOTS, "environment_population": "DEVELOPMENT_ONLY",
            "trajectory_ids": list(range(10 if mode == "formal" else 2)), "CRN_bank_columns": 300,
            "TRACE_POLICIES": TRACE_POLICIES, "trace_purpose": "REWARD_CONTRACT_PROBES_ONLY",
            "structural_order": ["L_pre", "action", "L_post", "reward/cost settlement", "natural transition", "L_next"],
            "terminal_semantics": "day_index=364: settlement, transition_executed=False, L_next=NaN, rain flag=False",
            "policy_inputs": ["policy", "day_index", "L_pre"],
            "E_clean_GHI_role": "REALIZED_REWARD_SETTLEMENT_ONLY", "pre_action_energy_access": False,
            "energy_recomputed": False, "energy_renormalized": False, "eta": ETA, "reward_scale": REWARD_SCALE,
            "numerical_tolerance": {"atol": ATOL, "rtol": RTOL, "equal_nan": False, "rounding": False},
            "exact_checks": ["lambda", "eta", "reward_scale", "N_clean", "CRN", "dates", "Git blobs", "SHA256"],
            "aggregate_population": "Frozen 300 trajectories/year; multiplier=2 only; equal-year JOINT means",
            "distribution_role": "NUMERICAL_DIAGNOSTIC_ONLY", "distribution_std_ddof": 1,
            "empty_distribution_groups": "count=0; statistics missing (CSV empty), never used for tuning",
            "formal_freeze_requires_all_gates": True, "smoke_freezes_reward": False,
            "required_gates": GATES}


def publish(out, manifest, trace, regression, distribution, contract, audit):
    # A manifest in a partial directory is not an effective freeze. The final
    # audit is the sole publication marker and is written only after readback.
    check(all(value for name, value in audit["gates"].items() if name != GATES[20]),
          "All implementation gates must pass before any outputs")
    frames = {"daily_reward_trace.csv": trace, "trajectory_reward_regression.csv": regression,
              "reward_distribution_summary.csv": distribution}
    payloads = {"protocol_manifest.json": json_bytes(manifest),
                "reward_contract_manifest.json": json_bytes(contract)}
    payloads.update({name: frame.to_csv(index=False).encode("utf-8-sig") for name, frame in frames.items()})
    hash_index = {"hashes": {name: sha(data) for name, data in payloads.items()},
                  "excluded": ["output_hashes.json", "audit_summary.json"],
                  "semantics": "Non-audit expected hashes; actual disk verification certified by final audit"}
    payloads["output_hashes.json"] = json_bytes(hash_index)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (out / name).open("xb") as stream:
            stream.write(data)
    verified = {}
    for name, expected in payloads.items():
        actual = (out / name).read_bytes()
        check(actual == expected and sha(actual) == sha(expected), f"Output bytes/hash: {name}")
        if name in hash_index["hashes"]:
            check(sha(actual) == hash_index["hashes"][name], f"Hash index: {name}")
        if name in frames:
            original, reread = frames[name], csv(actual)
            check(reread.columns.tolist() == original.columns.tolist() and len(reread) == len(original),
                  f"Persisted CSV schema: {name}")
            pd.testing.assert_frame_equal(original.reset_index(drop=True), reread,
                                          check_dtype=False, check_exact=True)
        else:
            check(json.loads(actual) == json.loads(expected), f"Persisted JSON schema/content: {name}")
        verified[name] = sha(actual)
    gate(audit, 20, set(p.name for p in out.iterdir()) == set(payloads),
         {"verified_non_audit_sha256": verified, "verification": "actual write + readback + schema/content + SHA256"})
    # Recheck all input bytes and scripts immediately before publishing success.
    for name, digest in FORMAL_HASHES.items():
        check(sha((PRE / name).read_bytes()) == digest, f"Frozen input changed during run: {name}")
    for name, (path, digest) in INPUTS.items():
        check(sha(path.read_bytes()) == digest, f"Frozen input changed during run: {name}")
    head = git("rev-parse", "HEAD")
    check(head == manifest["current_git_head"], "HEAD changed during run")
    check(script_record(SELF, head) == manifest["current_stage_script"], "Current script changed during run")
    for name, (path, blob) in HELPERS.items():
        check(script_record(path, head, blob) == manifest["environment_helper_provenance"][name],
              f"Helper changed during run: {name}")
    audit["stage_pass"] = all(audit["gates"].values())
    audit["failed_gates"] = [name for name, passed in audit["gates"].items() if not passed]
    audit["scientific_status"] = contract["scientific_status"]
    audit["reward_contract_status"] = contract["reward_contract_status"]
    audit["reward_scale_status"] = contract["reward_scale_status"]
    data = json_bytes(audit)
    with (out / "audit_summary.json").open("xb") as stream:
        stream.write(data)
    check((out / "audit_summary.json").read_bytes() == data, "Final audit readback")


def main():
    parser = argparse.ArgumentParser(description=f"{STAGE}: {TITLE}")
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    audit = {"stage": STAGE, "mode": args.mode, "stage_pass": False,
             "gates": {name: False for name in GATES.values()}, "gate_details": {}}
    manifest = protocol(args.mode)
    out = OUTPUT / args.mode
    try:
        check(not out.exists(), f"Immutable output directory already exists: {out}")
        modules, ledger, raw_ledger, dry, r3, energy, lambda_main, frozen, payloads = inputs(audit, manifest)
        static_contract(modules, audit)
        formal = args.mode == "formal"
        n = 10 if formal else 2
        trace = simulate(modules, ledger, dry, r3, energy, lambda_main, frozen, n, manifest, audit)
        bounds = validate_trace(trace, energy, lambda_main, n, audit)
        regression, aggregate = regress(trace, payloads, frozen, lambda_main, formal, audit)
        latent = continuity(trace, ledger, raw_ledger, energy, n, audit)
        distribution = distributions(trace)
        manifest.update(lambda_c_main=lambda_main, theoretical_bounds=bounds,
                        frozen_aggregate_regression=aggregate, latent_continuity=latent)
        contract = {"stage": STAGE, "mode": args.mode,
                    "scientific_status": "REWARD_CONTRACT_PASS_FROZEN" if formal else "SMOKE_ONLY_NO_FREEZE",
                    "reward_contract_status": "PASS_FROZEN" if formal else "SMOKE_ONLY_NO_FREEZE",
                    "reward_scale_status": "NO_SCALING_REQUIRED", "reward_scale": REWARD_SCALE,
                    "lambda_c_main": lambda_main, "eta": ETA, "theoretical_bounds": bounds,
                    "lambda_source": manifest["frozen_formal_provenance"]["main_lambda_manifest.json"],
                    "WAIT": "L_post = L_pre", "CLEAN": "L_post = (1 - ETA) * L_pre = 0.05 * L_pre",
                    "soiling_cost": "E_clean_GHI * L_post", "cleaning_cost": "lambda_c_main * action",
                    "total_cost": "soiling_cost + cleaning_cost", "raw_reward": "-total_cost",
                    "final_reward": "raw_reward",
                    "frozen_formula": "r_t = -[E_clean_GHI_t * L_post_t + 0.19925428989934618 * action_t]",
                    "declaration": "NO FURTHER REWARD TUNING AFTER THIS STAGE" if formal else "SMOKE ONLY: NO FORMAL REWARD FREEZE",
                    "required_downstream_consumers": ["Gym", "True-State PPO", "Point PPO", "UA-PPO"],
                    "downstream_requirement": "Must reuse this reward contract after successful formal freeze",
                    "freeze_effective_only_with": "Same-directory final audit_summary.json: stage_pass=True, failed_gates=[], scientific_status=REWARD_CONTRACT_PASS_FROZEN",
                    "E_clean_GHI_role": "REALIZED_REWARD_SETTLEMENT_ONLY", "latent_continuity": latent}
        publish(out, manifest, trace, regression, distribution, contract, audit)
    except Exception as exc:
        audit.update(stage_pass=False, scientific_status="FAIL_CLOSED_NO_FREEZE",
                     reward_contract_status="NOT_FROZEN", execution_error=f"{type(exc).__name__}: {exc}")
        audit["failed_gates"] = [name for name, passed in audit["gates"].items() if not passed]
        # Never write a failure audit over immutable artifacts; stdout is the
        # failure report. Partial output directories remain unpublishable.
        print(json_bytes(audit).decode("utf-8"))
        return 1
    print(json_bytes({"stage": STAGE, "stage_pass": audit["stage_pass"],
                      "scientific_status": audit["scientific_status"], "output": str(out)}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
