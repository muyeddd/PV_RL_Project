#!/usr/bin/env python
"""P2-1D-2B-v1: Normalized Reward Behaviour and Degeneracy Audit.

Explicit execution only. Both modes use development environments exclusively.
Policies receive only a day index and current true state. Frozen energy is passed
to the settlement layer, never to the policy. No training or perception imports.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STAGE = "P2-1D-2B-v1"
TITLE = "Normalized Reward Behaviour and Degeneracy Audit"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUTPUT = BASE / "p2_1d_2b_reward_behaviour_audit_v1"
SELF = "experiments/run_paper2_stage1d2b_reward_behaviour_audit_v1.py"
CHECKPOINT = "96a20b34160a8640763a537f26f501dd84f40076"
PREDECESSOR = "experiments/run_paper2_stage1d2a_cleaning_cost_scale_audit_v1.py"
PREDECESSOR_BLOB = "de172e3c9c689c1391f5d13f3b556edac7d9d7d0"
PRE = BASE / "p2_1d_2a_cleaning_cost_scale_audit_v1"
TIMELINE = BASE / "p2_1a_timeline_intervention_audit_v1"
INPUTS = {
    "audit": (PRE / "audit_summary.json", "33003c91b48c363cb230806886841c97bfc346c967496164b6398bca5bf206d8"),
    "scale": (PRE / "cost_scale_summary.json", "8f305064c544619fa657c61b196bcd321a8f4d3d4571e71aa9a609e88e6165c9"),
    "grid": (PRE / "candidate_lambda_grid.csv", "10d5fa090234b07f2dad65c295d44889958518a6b65ed0a070b3c590ca9ac280"),
    "energy": (BASE / "p2_1d_1a_daily_ghi_energy_value_audit_v1/daily_energy_value_proxy.csv",
               "443a023c42e9b0db9cf888c120eba53b5ad092779ea0875cd6de32826414c155"),
    "ledger": (TIMELINE / "environment_master_ledger.csv", "7906680af03d94f4989efc16787d9f5431bfb53b87bd2eb0f8586c6a340fde04"),
    # Pinned from the existing frozen timeline during static source inspection.
    "transitions": (TIMELINE / "transition_audit.csv", "195a23f1bb893b4b92c8f567e17824007c351886d2cadd1600cd5f3022d7d75d"),
}
HELPERS = {
    "mechanics": ("experiments/run_paper2_stage1c2a_clean_mechanics_operational_support_audit_v1.py",
                  "3be8472e8963e0ee97b6173d54916a92dbc05277"),
    "natural": ("experiments/run_paper2_stage1c1r_b_rain_repair_candidate_validation_v1.py",
                "c1430b7b4ccac776433294489984b07af0c82c5f"),
}
ETA = 0.95
EXPECTED_ANCHOR = 0.09962714494967309  # Validation only; never constructs lambdas.
MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEARCH_ORDER = [2, 4, 1, 8, 0.5, 16, 0.25]
RELATIVE_NEAR_OPTIMAL_TOLERANCE = 0.005
ABSOLUTE_TOLERANCE = 1e-12
ENV_ROOTS = {"YEAR1": 1020001, "YEAR2": 1030001}
POLICIES = ["NEVER_CLEAN", "DAILY_CLEAN", "PERIODIC_7", "PERIODIC_14",
            "PERIODIC_30", "PERIODIC_60", "PERIODIC_90", "PERIODIC_180",
            "TRUE_THRESHOLD_0.05", "TRUE_THRESHOLD_0.08", "TRUE_THRESHOLD_0.10",
            "TRUE_THRESHOLD_0.12", "TRUE_THRESHOLD_0.15"]
SMOKE_POLICIES = ["NEVER_CLEAN", "DAILY_CLEAN", "PERIODIC_30", "TRUE_THRESHOLD_0.10"]
SUPPORT_VALIDATED_INTERIOR_POLICIES = [
    "TRUE_THRESHOLD_0.05", "TRUE_THRESHOLD_0.08", "TRUE_THRESHOLD_0.10",
    "TRUE_THRESHOLD_0.12", "TRUE_THRESHOLD_0.15",
]
SCOPES = ["YEAR1", "YEAR2", "JOINT"]
BASE_COLUMNS = ["year", "trajectory_id", "policy", "J_soiling", "N_clean",
                "mean_L_pre", "mean_L_post", "max_L_pre"]
SUMMARY_COLUMNS = ["multiplier", "lambda_c", "scope", "policy", "mean_J_total",
                   "std_J_total", "median_J_total", "mean_J_soiling", "mean_N_clean",
                   "std_N_clean", "min_N_clean", "P05_N_clean", "median_N_clean",
                   "P95_N_clean", "max_N_clean", "near_optimal"]
CLASS_COLUMNS = ["multiplier", "lambda_c", "YEAR1_classification", "YEAR2_classification",
                 "JOINT_classification", "robust_interior", "YEAR1_near_optimal_policies",
                 "YEAR2_near_optimal_policies", "JOINT_near_optimal_policies",
                 "YEAR1_support_validated_interior_present",
                 "YEAR2_support_validated_interior_present",
                 "JOINT_support_validated_interior_present",
                 "support_validated_robust_interior"]
GATES = dict(enumerate([
    "predecessor_checkpoint_ancestry_pass", "predecessor_script_blob_pass",
    "predecessor_frozen_hashes_and_contract_pass", "frozen_energy_proxy_pass",
    "candidate_grid_exact_pass", "environment_provenance_pass", "environment_CRN_pass",
    "trajectory_population_exact_pass", "fixed_policy_library_exact_pass",
    "no_perception_or_formal_seed_access_pass", "clean_eta_exact_pass",
    "action_reward_transition_order_pass", "lambda_linearity_pass", "zero_cost_daily_sanity_pass",
    "near_optimal_identity_pass", "classification_identity_pass", "robust_interior_identity_pass",
    "main_lambda_search_rule_identity_pass", "no_prohibited_tuning_pass",
    "information_role_no_GHI_leakage_pass", "output_completeness_pass",
    "current_stage_script_committed_unchanged_pass"], start=1))


def check(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def gate(audit, number, passed, detail=None):
    audit["gates"][GATES[number]] = bool(passed)
    audit["gate_details"][f"G{number}"] = detail
    check(passed, f"G{number}: {GATES[number]}: {detail}")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def script_record(path, head, expected=None):
    committed = git("rev-parse", "--verify", f"{head}:{path}")
    working = git("hash-object", f"--path={path}", path)
    check(committed == working and (expected is None or working == expected),
          f"Script must be committed unchanged with the required blob: {path}")
    return {"path": path, "HEAD_blob": committed, "working_tree_blob": working,
            "sha256": sha((ROOT / path).read_bytes())}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(f"_p2_1d2b_{name}", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def read_calendar(payload, columns):
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", float_precision="round_trip")
    check(set(columns).issubset(frame.columns), f"Missing columns: {columns}")
    stamp = pd.to_datetime(frame["date"], errors="raise")
    check(stamp.notna().all() and not stamp.duplicated().any()
          and stamp.eq(stamp.dt.normalize()).all(), "Invalid calendar date labels")
    frame["date"] = stamp
    frame = frame.sort_values("date").reset_index(drop=True)
    check(len(frame) == 730 and pd.DatetimeIndex(frame.date).equals(
        pd.date_range("2021-08-09", "2023-08-08")), "Exact 730-day calendar required")
    return frame


def inputs(audit, manifest):
    head = git("rev-parse", "HEAD")
    commit = git("rev-parse", "--verify", f"{CHECKPOINT}^{{commit}}")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", CHECKPOINT, "HEAD"],
                              cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    gate(audit, 1, commit == CHECKPOINT and ancestry.returncode == 0,
         {"checkpoint": commit, "current_HEAD": head, "returncode": ancestry.returncode})
    frozen_blob = git("rev-parse", f"{CHECKPOINT}:{PREDECESSOR}")
    working_blob = git("hash-object", f"--path={PREDECESSOR}", PREDECESSOR)
    gate(audit, 2, frozen_blob == working_blob == PREDECESSOR_BLOB)
    own = script_record(SELF, head)
    gate(audit, 22, own["HEAD_blob"] == own["working_tree_blob"], own)
    manifest.update(current_git_head=head, current_stage_script_sha256=own["sha256"],
                    current_stage_script_git_blob=own["working_tree_blob"])
    payloads, provenance = {}, {}
    for name, (path, digest) in INPUTS.items():
        payloads[name] = path.read_bytes()
        actual = sha(payloads[name])
        provenance[name] = {"path": str(path), "sha256": actual, "expected_sha256": digest}
        check(actual == digest, f"Frozen hash mismatch: {name}")
    manifest["input_provenance"] = provenance
    prior = json.loads(payloads["audit"].decode("utf-8-sig"))
    scale = json.loads(payloads["scale"].decode("utf-8-sig"))
    gate(audit, 3, prior.get("stage_pass") is True and prior.get("failed_gates") == []
         and scale.get("COST_SCALE_ANCHOR") == EXPECTED_ANCHOR
         and scale.get("main_lambda_selected") is False and scale.get("ETA") == ETA)
    energy = read_calendar(payloads["energy"], ["date", "E_clean_GHI"])
    e = pd.to_numeric(energy.E_clean_GHI, errors="raise").to_numpy(float)
    gate(audit, 4, np.isfinite(e).all() and (e > 0).all()
         and np.isclose(e.mean(), 1, rtol=1e-12, atol=1e-12))
    grid = pd.read_csv(io.BytesIO(payloads["grid"]), float_precision="round_trip")
    check(grid.columns.tolist() == ["multiplier", "lambda_c", "role"], "Grid schema")
    gate(audit, 5, grid.multiplier.tolist() == MULTIPLIERS and len(grid) == 8
         and grid.role.tolist() == ["ZERO_COST_SANITY_CONTROL"] + ["BEHAVIOUR_AUDIT_CANDIDATE"] * 7
         and np.isfinite(grid.lambda_c).all() and grid.lambda_c.iloc[0] == 0
         and (grid.lambda_c.iloc[1:] > 0).all()
         and np.array_equal(grid.lambda_c.to_numpy(), scale["COST_SCALE_ANCHOR"] * grid.multiplier.to_numpy()),
         "Hash-exact CSV lambdas are used unchanged; multiplication is integrity checking only")
    records = {name: script_record(path, head, blob) for name, (path, blob) in HELPERS.items()}
    manifest["environment_helper_provenance"] = records
    modules = {name: load_module(name, path) for name, (path, _) in HELPERS.items()}
    mechanics = modules["mechanics"]
    # Parse the hash-verified snapshots through the frozen loaders, not another model.
    ledger = mechanics.load_ledger(io.BytesIO(payloads["ledger"]))
    read_calendar(payloads["ledger"], ["date", "L_power_proxy", "state_valid", "rain_day"])
    transitions = mechanics.load_transitions(io.BytesIO(payloads["transitions"]))
    check(len(transitions) == 729 and np.array_equal(transitions.transition_index, np.arange(729))
          and np.array_equal(transitions.source_date, ledger.date.iloc[:-1])
          and np.array_equal(transitions.dest_date, ledger.date.iloc[1:]), "Transition alignment")
    context = mechanics.attach_context(transitions, ledger)
    dry = context.loc[context.transition_class.eq("DRY_NATURAL"), "delta_L_power_proxy"].to_numpy(float)
    rain_rows = context.loc[context.transition_class.eq("RAIN_AFFECTED")]
    r3 = mechanics.fit_r3(rain_rows)
    gate(audit, 6, len(dry) == 494 and len(rain_rows) == 201 and np.isfinite(dry).all()
         and np.isfinite(r3["residuals"]).all()
         and np.allclose([r3["intercept"], r3["slope"], r3["source_max"], np.std(r3["residuals"], ddof=1)],
                         [0.00090811689781017, -0.48383380267297343, 0.0490334865821413,
                          0.004494874318643408], rtol=0, atol=1e-15), records)
    manifest["environment_parameters"] = {"dry": "D0_GLOBAL", "rain": "R3_OLS_RESIDUAL",
        "intercept": r3["intercept"], "slope": r3["slope"],
        "dry_pool_sha256": sha(dry.tobytes()), "residual_pool_sha256": sha(r3["residuals"].tobytes())}
    return modules, ledger, dry, r3, energy.set_index("date").E_clean_GHI, grid, scale


def action(policy, day_index, L_pre):
    """Restricted policy interface: no energy, weather, reward history or RNG."""
    if policy == "NEVER_CLEAN":
        return np.zeros_like(L_pre, dtype=bool)
    if policy == "DAILY_CLEAN":
        return np.ones_like(L_pre, dtype=bool)
    if policy.startswith("PERIODIC_"):
        return np.full_like(L_pre, (day_index + 1) % int(policy.split("_")[1]) == 0, dtype=bool)
    if policy.startswith("TRUE_THRESHOLD_"):
        return L_pre >= float(policy.removeprefix("TRUE_THRESHOLD_"))
    raise ValueError(f"Unknown fixed policy: {policy}")


def natural_transition(helper, post, idx, rain, dry, r3):
    """Adapter to the actual frozen scalar sampling helpers; no new simulator."""
    result = np.empty_like(post)
    for i, state in enumerate(post):
        if rain:
            model = {"candidate": "R3_OLS_RESIDUAL", "intercept": r3["intercept"],
                     "slope": r3["slope"], "residuals": r3["residuals"][idx[i]:idx[i] + 1]}
            result[i] = helper.rain_next_samples(float(state), model)[0]
        else:
            result[i] = helper.dry_next_samples(float(state), dry[idx[i]:idx[i] + 1])[0]
    return result


def rollout(policy, initial, indices, rain, dry, r3, energy, helper, grid):
    n = indices.shape[1]
    pre = np.empty((n, 365))
    post = np.empty_like(pre)
    actions = np.empty((n, 365), dtype=bool)
    soiling = np.empty_like(pre)
    daily_cost_sum = np.zeros((n, len(grid)))
    lambdas = grid.lambda_c.to_numpy()
    pre[:, 0] = initial
    for day in range(365):
        actions[:, day] = action(policy, day, pre[:, day])
        post[:, day] = np.where(actions[:, day], (1 - ETA) * pre[:, day], pre[:, day])
        check(np.array_equal(post[:, day], pre[:, day] * np.where(actions[:, day], 1 - ETA, 1)), "CLEAN order identity")
        # Realized settlement happens before the natural transition. Lambdas do
        # not enter actions, dynamics or innovation generation.
        soiling[:, day] = energy[day] * post[:, day]
        cleaning = actions[:, day, None] * lambdas[None, :]
        total = soiling[:, day, None] + cleaning
        reward = -total
        check(np.isfinite(total).all() and (total >= 0).all()
              and np.array_equal(-reward, total), "Reward settlement identity")
        daily_cost_sum += total
        if day < 364:
            pre[:, day + 1] = natural_transition(helper, post[:, day], indices[day], rain[day], dry, r3)
            # Independent algebra regression of the frozen helper at POST state.
            if rain[day]:
                raw = post[:, day] + (r3["intercept"] + r3["slope"] * post[:, day]) + r3["residuals"][indices[day]]
            else:
                raw = post[:, day] + dry[indices[day]]
            check(np.array_equal(pre[:, day + 1], np.clip(raw, 0, 1)), "Frozen natural transition/order identity")
    check(np.isfinite(pre).all() and np.isfinite(post).all(), "Nonfinite states")
    j = soiling.sum(axis=1)
    count = actions.sum(axis=1)
    totals = j[:, None] + count[:, None] * lambdas[None, :]
    # Canonical totals are exactly the affine formula. Daily accumulation and
    # intercept/slope regression are independent floating-point cross-checks.
    check(np.array_equal(totals, np.add(j[:, None], np.multiply(count[:, None], lambdas[None, :]))), "Exact lambda identity")
    check(np.allclose(totals, daily_cost_sum, rtol=1e-12, atol=1e-12), "Daily vs affine totals")
    slope, intercept = np.linalg.lstsq(np.column_stack([lambdas, np.ones(len(lambdas))]), totals.T, rcond=None)[0]
    check(np.allclose(slope, count, rtol=1e-12, atol=1e-12)
          and np.allclose(intercept, j, rtol=1e-12, atol=1e-12), "Linearity slope/intercept regression")
    return {"J_soiling": j, "N_clean": count, "mean_L_pre": pre.mean(axis=1),
            "mean_L_post": post.mean(axis=1), "max_L_pre": pre.max(axis=1)}


def simulate(modules, ledger, dry, r3, energy, grid, policies, n, audit, manifest):
    mechanics = modules["mechanics"]
    rows, crn = [], {}
    for year, seed in ENV_ROOTS.items():
        check(seed == mechanics.scenario_seed(420001, year, "HISTORICAL_FIRST_DAY"), "Development root mapping")
        calendar = mechanics.get_year_calendar(ledger, year)
        expected = pd.date_range("2021-08-09" if year == "YEAR1" else "2022-08-09", periods=365)
        check(pd.DatetimeIndex(calendar.date).equals(expected), "Year calendar")
        initial = float(calendar.L_power_proxy.iloc[0])
        check(bool(calendar.state_valid.iloc[0]) and np.isfinite(initial) and 0 <= initial <= 1, "Historical initial state")
        # Always generate the development 300-column bank; smoke slices IDs 0..9.
        bank, rain, report = mechanics.build_crn_indices(calendar, len(dry), len(r3["residuals"]), 300, seed)
        full = np.stack(bank)
        check(full.shape == (364, 300) and report["same_seed_exact_reproduction"], "CRN regeneration")
        indices = full[:, :n].copy()
        indices.setflags(write=False)
        rain.setflags(write=False)
        original = sha(indices.tobytes())
        values = energy.loc[calendar.date].to_numpy(float)
        values.setflags(write=False)
        usage = {}
        for policy in policies:
            result = rollout(policy, initial, indices, rain, dry, r3, values, modules["natural"], grid)
            usage[policy] = sha(indices.tobytes())
            check(usage[policy] == original, "Policy mutated CRN bank")
            for trajectory in range(n):
                rows.append({"year": year, "trajectory_id": trajectory, "policy": policy,
                             **{key: int(value[trajectory]) if key == "N_clean" else float(value[trajectory])
                                for key, value in result.items()}})
        crn[year] = {"root": seed, "initial_L": initial, "full_bank": report,
                     "used_bank_sha256": original, "policy_bank_hashes": usage,
                     "used_ids": list(range(n)), "innovations_read_only": True}
    manifest["CRN"] = crn
    frame = pd.DataFrame(rows, columns=BASE_COLUMNS)
    gate(audit, 7, all(len(set(x["policy_bank_hashes"].values())) == 1 for x in crn.values()), crn)
    expected_keys = {(year, trajectory, policy) for year in ENV_ROOTS for trajectory in range(n) for policy in policies}
    gate(audit, 8, len(frame) == len(expected_keys) and set(frame[["year", "trajectory_id", "policy"]].itertuples(index=False, name=None)) == expected_keys,
         {"mode": manifest["mode"], "trajectories_per_year": n, "formal_requirement": 300,
          "smoke_formal_population_gate_applicable": manifest["mode"] == "formal"})
    gate(audit, 9, policies == (POLICIES if manifest["mode"] == "formal" else SMOKE_POLICIES), policies)
    gate(audit, 10, ENV_ROOTS == {"YEAR1": 1020001, "YEAR2": 1030001}
         and set(modules) == {"mechanics", "natural"}, "Only development RNG banks and non-perception helpers accessed")
    gate(audit, 11, ETA == .95 and mechanics.PRIMARY_ETA == ETA)
    gate(audit, 12, True, "Every day checked CLEAN, settlement, and frozen post-state natural transition identities; 365 settlements and 364 transitions")
    gate(audit, 13, True, "Every fixed rollout: exact affine totals plus independent daily sum and slope/intercept regression")
    return frame


def near_mask(values):
    best = float(np.min(values))
    return np.asarray(values) <= best + max(best * RELATIVE_NEAR_OPTIMAL_TOLERANCE, ABSOLUTE_TOLERANCE)


def classification(names):
    chosen = set(names)
    if chosen == {"DAILY_CLEAN"}:
        return "DAILY_DEGENERATE"
    if chosen == {"NEVER_CLEAN"}:
        return "NEVER_DEGENERATE"
    if chosen and not chosen.intersection({"DAILY_CLEAN", "NEVER_CLEAN"}):
        return "INTERIOR_TRADEOFF"
    return "BOUNDARY_AMBIGUOUS"


def summarize(base, grid, policies, formal, audit):
    rows, class_rows, all_sets = [], [], {}
    for candidate in grid.itertuples(index=False):
        sets, classes = {}, {}
        for scope in SCOPES:
            block = base if scope == "JOINT" else base.loc[base.year.eq(scope)]
            scope_rows = []
            for policy in policies:
                b = block.loc[block.policy.eq(policy)]
                totals = b.J_soiling.to_numpy() + candidate.lambda_c * b.N_clean.to_numpy()
                # Equal-size years give the equal-weight annual mixture for std
                # and median; JOINT means are explicitly averaged over years.
                means = b.groupby("year")[["J_soiling", "N_clean"]].mean().mean()
                mean_total = float(means.J_soiling + candidate.lambda_c * means.N_clean)
                check(np.isclose(mean_total, totals.mean(), rtol=1e-12, atol=1e-12), "Equal-year JOINT identity")
                counts = b.N_clean.to_numpy()
                scope_rows.append({"multiplier": candidate.multiplier, "lambda_c": candidate.lambda_c,
                    "scope": scope, "policy": policy, "mean_J_total": mean_total,
                    "std_J_total": float(totals.std(ddof=1)), "median_J_total": float(np.median(totals)),
                    "mean_J_soiling": float(means.J_soiling), "mean_N_clean": float(means.N_clean),
                    "std_N_clean": float(counts.std(ddof=1)), "min_N_clean": int(counts.min()),
                    "P05_N_clean": float(np.quantile(counts, .05)), "median_N_clean": float(np.median(counts)),
                    "P95_N_clean": float(np.quantile(counts, .95)), "max_N_clean": int(counts.max())})
            values = np.array([r["mean_J_total"] for r in scope_rows])
            mask = near_mask(values)
            limit = max(values.min() * 1.005, values.min() + 1e-12)
            check(np.array_equal(mask, values <= limit), "Near-optimal 0.5% identity")
            for row, chosen in zip(scope_rows, mask):
                row["near_optimal"] = bool(chosen)
            sets[scope] = [r["policy"] for r in scope_rows if r["near_optimal"]]
            classes[scope] = classification(sets[scope])
            rows.extend(scope_rows)
        all_sets[candidate.multiplier] = sets
        if formal:
            robust = all(classes[scope] == "INTERIOR_TRADEOFF" for scope in SCOPES)
            support_present = {s: bool(set(sets[s]).intersection(SUPPORT_VALIDATED_INTERIOR_POLICIES))
                               for s in SCOPES}
            support_robust = robust and all(support_present.values())
            # Verify the category predicates independently of classification().
            for scope in SCOPES:
                selected = set(sets[scope])
                interior = bool(selected.intersection(set(POLICIES) - {"DAILY_CLEAN", "NEVER_CLEAN"})
                                and "DAILY_CLEAN" not in selected and "NEVER_CLEAN" not in selected)
                check((classes[scope] == "INTERIOR_TRADEOFF") == interior, "Interior classification identity")
                check((classes[scope] == "DAILY_DEGENERATE") == (selected == {"DAILY_CLEAN"}), "Daily classification identity")
                check((classes[scope] == "NEVER_DEGENERATE") == (selected == {"NEVER_CLEAN"}), "Never classification identity")
            check(robust == all(not set(sets[s]).intersection({"DAILY_CLEAN", "NEVER_CLEAN"})
                               and bool(sets[s]) for s in SCOPES), "Robust interior identity")
            check(support_robust == all(classes[s] == "INTERIOR_TRADEOFF"
                  and any(p in sets[s] for p in SUPPORT_VALIDATED_INTERIOR_POLICIES)
                  for s in SCOPES), "Support-validated robust interior identity")
            class_rows.append({"multiplier": candidate.multiplier, "lambda_c": candidate.lambda_c,
                **{f"{s}_classification": classes[s] for s in SCOPES}, "robust_interior": robust,
                **{f"{s}_support_validated_interior_present": support_present[s] for s in SCOPES},
                "support_validated_robust_interior": support_robust,
                **{f"{s}_near_optimal_policies": json.dumps(sets[s]) for s in SCOPES}})
    gate(audit, 14, all("DAILY_CLEAN" in all_sets[0.0][scope] for scope in SCOPES), all_sets[0.0])
    gate(audit, 15, True, "0.5% relative rule with 1e-12 floor; every scope/candidate checked")
    gate(audit, 16, True, "All classification predicates checked" if formal else "NOT_APPLICABLE_SMOKE: no classification output")
    gate(audit, 17, True, "Raw and support-validated three-scope conjunctions checked" if formal else "NOT_APPLICABLE_SMOKE")
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS), pd.DataFrame(class_rows, columns=CLASS_COLUMNS), all_sets


def select_main(classes, grid, formal):
    result = {"main_multiplier": None, "main_lambda_c": None,
              "scientific_status": "SMOKE_ONLY_NO_SELECTION"}
    if not formal:
        return result
    robust = set(classes.loc[classes.support_validated_robust_interior & classes.multiplier.gt(0), "multiplier"])
    candidate = next((m for m in SEARCH_ORDER if m in robust), None)
    if candidate is None:
        raw_present = bool((classes.robust_interior & classes.multiplier.gt(0)).any())
        result["scientific_status"] = ("YELLOW_NO_SUPPORT_VALIDATED_ROBUST_INTERIOR"
                                       if raw_present else "YELLOW_NO_ROBUST_INTERIOR")
    elif candidate in {0.25, 16}:
        result["scientific_status"] = "YELLOW_EDGE_ONLY"
    else:
        result.update(main_multiplier=candidate,
                      main_lambda_c=float(grid.loc[grid.multiplier.eq(candidate), "lambda_c"].iloc[0]),
                      scientific_status="ROBUST_INTERIOR_MAIN_LAMBDA_SELECTED")
    return result


def selection_regression(result, classes, grid, formal):
    if not formal:
        return result == {"main_multiplier": None, "main_lambda_c": None, "scientific_status": "SMOKE_ONLY_NO_SELECTION"}
    eligible, raw_present = [], False
    for row in classes.itertuples():
        raw = all(getattr(row, f"{s}_classification") == "INTERIOR_TRADEOFF" for s in SCOPES)
        support = []
        for scope in SCOPES:
            names = json.loads(getattr(row, f"{scope}_near_optimal_policies"))
            present = any(name in SUPPORT_VALIDATED_INTERIOR_POLICIES for name in names)
            if present != getattr(row, f"{scope}_support_validated_interior_present"):
                return False
            support.append(present)
        supported = raw and all(support)
        if raw != row.robust_interior or supported != row.support_validated_robust_interior:
            return False
        if row.multiplier > 0:
            raw_present = raw_present or raw
            if supported:
                eligible.append(row)
    eligible.sort(key=lambda r: SEARCH_ORDER.index(r.multiplier))
    if not eligible:
        status = "YELLOW_NO_SUPPORT_VALIDATED_ROBUST_INTERIOR" if raw_present else "YELLOW_NO_ROBUST_INTERIOR"
        return result["scientific_status"] == status and result["main_lambda_c"] is None and result["main_multiplier"] is None
    first = eligible[0]
    if first.multiplier in (.25, 16):
        return result["scientific_status"] == "YELLOW_EDGE_ONLY" and result["main_lambda_c"] is None and result["main_multiplier"] is None
    return (result["scientific_status"] == "ROBUST_INTERIOR_MAIN_LAMBDA_SELECTED"
            and result["main_multiplier"] == first.multiplier and result["main_lambda_c"] == first.lambda_c)


def protocol(mode):
    return {"stage": STAGE, "title": TITLE, "mode": mode, "environment_population": "DEVELOPMENT_ONLY",
        "predecessor_checkpoint": CHECKPOINT, "checkpoint_semantics": "ancestor, not current HEAD equality",
        "ETA": ETA, "environment_roots": ENV_ROOTS, "trajectory_ids": list(range(300 if mode == "formal" else 10)),
        "CRN_bank_columns": 300, "initialization": "HISTORICAL_FIRST_DAY",
        "policy_library": POLICIES, "active_policies": POLICIES if mode == "formal" else SMOKE_POLICIES,
        "SUPPORT_VALIDATED_INTERIOR_POLICIES": SUPPORT_VALIDATED_INTERIOR_POLICIES,
        "support_reference_source": "Frozen P2-1C-2A operational support audit at eta=.95; no threshold retuning or support reevaluation",
        "main_selection_eligibility": "support_validated_robust_interior: all three scopes INTERIOR_TRADEOFF and each near-optimal set intersects the frozen support reference set",
        "raw_robust_interior_role": "DIAGNOSTIC_ONLY",
        "policy_inputs": ["day_index", "L_pre"], "periodic_rule": "(day_index + 1) % P == 0",
        "threshold_rule": "L_pre >= theta", "day_index_origin": 0,
        "structural_order": ["L_pre", "action", "L_post", "realized settlement", "natural transition", "L_next"],
        "terminal_semantics": "Day 364 has action and settlement; no transition beyond 365-day horizon",
        "soiling_cost_t": "E_clean_GHI_t * L_post_t", "cleaning_cost_t": "lambda_c * action_t",
        "total_cost_t": "soiling_cost_t + cleaning_cost_t", "reward_t": "-total_cost_t",
        "evaluation_primary": "undiscounted 365-day total cost",
        "lambda_evaluation": "J_soiling + lambda_c * N_clean; one fixed rollout per year/trajectory/policy",
        "JOINT_mean": "equal average of YEAR1 and YEAR2 means",
        "JOINT_std_median": "equal-weight mixture of the two annual populations, not two-year sums",
        "std_ddof": 1, "lambda_grid_source": str(INPUTS["grid"][0]),
        "no_GHI_recomputation_or_renormalization": True,
        "E_clean_GHI_role": "REALIZED_REWARD_SETTLEMENT_ONLY", "forbidden_in_pre_action_observation": True,
        "no_future_GHI_provided_to_agent": True, "perception_used": False, "training_implemented": False,
        "formal_environment_roots_accessed": False, "perception_seeds_accessed": False,
        "historical_cleaning_used_for_selection": False, "currency_claim": False,
        "RELATIVE_NEAR_OPTIMAL_TOLERANCE": RELATIVE_NEAR_OPTIMAL_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE, "search_order": SEARCH_ORDER,
        "search_rationale": "Predeclared 2x anchor center; equal log-distance prefers higher cost to avoid excessive-cleaning preference",
        "smoke_selection_forbidden": True, "required_gates": GATES}


def encode_json(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def save_outputs(out, manifest, base, summary, classes, main, audit):
    # No successful audit is serialized until all other artifacts are persisted
    # and verified. A failed partial directory remains immutable and unaudited.
    audit["stage_pass"] = False
    audit["gates"][GATES[21]] = False
    payloads = {
        "protocol_manifest.json": encode_json(manifest),
        "trajectory_policy_base_metrics.csv": base.to_csv(index=False).encode("utf-8-sig"),
        "lambda_policy_summary.csv": summary.to_csv(index=False).encode("utf-8-sig"),
        "lambda_classification_summary.csv": classes.to_csv(index=False).encode("utf-8-sig"),
    }
    main["all_output_hashes"] = {name: sha(data) for name, data in payloads.items()}
    main["hash_coverage"] = (
        "Expected raw SHA256 for protocol and three CSV artifacts above. "
        "output_hashes.json also covers this main manifest. Both omit the final audit, "
        "which is published only after actual non-audit readback succeeds. The index omits "
        "itself; the main manifest omits itself, avoiding self/final-audit hash recursion. "
        "These expected hashes alone do not certify completed readback; the final audit does.")
    payloads["main_lambda_manifest.json"] = encode_json(main)
    hash_index = {"hashes": {name: sha(data) for name, data in payloads.items()},
                  "excluded": ["output_hashes.json", "audit_summary.json"],
                  "coverage_semantics": "Expected raw hashes of five non-audit artifacts; self and last-published audit excluded. Verification is certified only by the final audit."}
    payloads["output_hashes.json"] = encode_json(hash_index)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (out / name).open("xb") as stream:
            stream.write(data)
    schemas = {"trajectory_policy_base_metrics.csv": (BASE_COLUMNS, len(base)),
               "lambda_policy_summary.csv": (SUMMARY_COLUMNS, len(summary)),
               "lambda_classification_summary.csv": (CLASS_COLUMNS, len(classes))}
    verified_hashes = {}
    for name, data in payloads.items():
        actual = (out / name).read_bytes()
        check(actual == data, f"Output readback mismatch: {name}")
        digest = sha(actual)
        check(digest == sha(data), f"Output hash mismatch: {name}")
        if name in hash_index["hashes"]:
            check(digest == hash_index["hashes"][name], f"Hash index mismatch: {name}")
        if name in schemas:
            columns, length = schemas[name]
            reread = pd.read_csv(io.BytesIO(actual), encoding="utf-8-sig")
            check(reread.columns.tolist() == columns and len(reread) == length, f"Persisted schema: {name}")
        else:
            check(json.loads(actual.decode("utf-8")) == json.loads(data.decode("utf-8")),
                  f"Persisted JSON schema/content: {name}")
        verified_hashes[name] = digest
    gate(audit, 21, True, {"verified_non_audit_output_sha256": verified_hashes,
                          "verification": "Actual disk readback bytes, schema/content and SHA256 all passed",
                          "audit_publication": "Exclusive write last, followed immediately by byte readback"})
    audit["stage_pass"] = all(audit["gates"].values())
    audit["failed_gates"] = [name for name, passed in audit["gates"].items() if not passed]
    audit["scientific_status"] = main["scientific_status"]
    audit_bytes = encode_json(audit)
    with (out / "audit_summary.json").open("xb") as stream:
        stream.write(audit_bytes)
    check((out / "audit_summary.json").read_bytes() == audit_bytes, "Final audit byte readback mismatch")


def main():
    parser = argparse.ArgumentParser(description=f"{STAGE}: {TITLE}")
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    out = OUTPUT / args.mode
    if out.exists():
        raise FileExistsError(f"Immutable output directory exists: {out}")
    audit = {"stage": STAGE, "mode": args.mode, "gates": {name: False for name in GATES.values()},
             "gate_details": {}, "stage_pass": False}
    manifest = protocol(args.mode)
    try:
        modules, ledger, dry, r3, energy, grid, scale = inputs(audit, manifest)
        formal = args.mode == "formal"
        policies = POLICIES if formal else SMOKE_POLICIES
        base = simulate(modules, ledger, dry, r3, energy, grid, policies, 300 if formal else 10, audit, manifest)
        summary, classes, sets = summarize(base, grid, policies, formal, audit)
        selected = select_main(classes, grid, formal)
        gate(audit, 18, selection_regression(selected, classes, grid, formal), selected)
        gate(audit, 19, not manifest["historical_cleaning_used_for_selection"]
             and not manifest["perception_used"] and not manifest["training_implemented"],
             "Selection receives only fixed-grid classifications; historical reference is read afterwards")
        gate(audit, 20, action.__code__.co_varnames[:action.__code__.co_argcount] == ("policy", "day_index", "L_pre")
             and not set(action.__code__.co_names).intersection({"energy", "grid", "reward", "rain", "ENV_ROOTS"}),
             "Restricted action signature; realized energy only reaches settlement")
        historical = {"role": "BEHAVIOURAL_PLAUSIBILITY_REFERENCE_ONLY", "expected_count": 26,
                      "years": 2, "expected_approximate_per_year": 13, "used_for_selection": False}
        historical["observed_count"] = int(ledger.modb_manual_cleaning_day.sum())
        historical["observed_per_year"] = historical["observed_count"] / 2
        main_manifest = {"stage": STAGE, "mode": args.mode, "anchor": scale["COST_SCALE_ANCHOR"],
            "grid_provenance": manifest["input_provenance"]["grid"], "search_order": SEARCH_ORDER,
            "SUPPORT_VALIDATED_INTERIOR_POLICIES": SUPPORT_VALIDATED_INTERIOR_POLICIES,
            "selection_eligibility": "support_validated_robust_interior",
            "raw_robust_interior_role": "DIAGNOSTIC_ONLY",
            "near_optimal_tolerance": RELATIVE_NEAR_OPTIMAL_TOLERANCE,
            "absolute_tolerance": ABSOLUTE_TOLERANCE, **selected,
            "historical_clean_reference": historical,
            "high_cost_diagnostic": {s: "NEVER_CLEAN" in sets[16.0][s] for s in SCOPES},
            "high_cost_is_hard_gate": False, "currency_claim": False,
            "lambda_c_unit": "normalized average-clean-day energy-value unit per CLEAN"}
        save_outputs(out, manifest, base, summary, classes, main_manifest, audit)
    except Exception as exc:
        audit["stage_pass"] = False
        audit["execution_error"] = f"{type(exc).__name__}: {exc}"
        audit["failed_gates"] = [name for name, passed in audit["gates"].items() if not passed]
        print(encode_json(audit).decode("utf-8"))
        return 1
    print(encode_json({"stage": STAGE, "stage_pass": audit["stage_pass"],
                       "scientific_status": audit["scientific_status"], "output": str(out)}).decode("utf-8"))
    return 0 if audit["stage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
