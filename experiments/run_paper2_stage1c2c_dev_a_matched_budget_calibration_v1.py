#!/usr/bin/env python
"""DEV-A cleaning-budget calibration only. No work executes on import.

Both CLI modes are opt-in. Hidden arrays exist only inside rollout/regression;
the search receives support and full-year cleaning counts, never state metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUT = BASE / "p2_1c_2c_dev_a_matched_budget_calibration_v1"
STAGE = "P2-1C-2C-DEV-A-v1"
TITLE = "Global Support-Aware Matched-Budget Calibration"
PREDECESSOR = "experiments/run_paper2_stage1c2c_prerl_decision_value_probe_v1_1.py"
PREDECESSOR_BLOB = "867141877056e8808176d9efaba3155ae58ec9e2"
PREDECESSOR_HEAD = "0612d6331e6740b1111a037e9d79e40990bb0798"
TOTAL_DEV_TRAJECTORIES = 300
CALIBRATION_IDS = tuple(range(100))
SELECTION_IDS = tuple(range(100, 300))
DEV_PERCEPTION_SEED = 20260906
DEV_ENV_ROOTS = {"YEAR1": 1020001, "YEAR2": 1030001}
K_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
BUDGETS = [2, 4, 6, 8, 10, 12]
SEARCH_ORDER = [12, 10, 8, 6, 4, 2]
ETA = 0.95
CALIBRATION_RELATIVE_ERROR_TARGET = 0.03
MAX_SUPPORT_BRACKET_EVALS = 12
MAX_REFINEMENT_EVALS_PER_BUDGET = 10
GATES = ["predecessor_blob_pass", "frozen_source_provenance_pass",
         "generic_rollout_regression_pass", "dev_environment_crn_pass",
         "perception_crn_pass", "calibration_selection_split_pass",
         "no_selection_access_pass", "no_formal_seed_access_pass",
         "global_threshold_pass", "support_semantics_pass",
         "cache_consistency_pass", "clean_count_objective_only_pass",
         "decision_value_blind_pass", "output_completeness_pass"]
CACHE_COLUMNS = ["policy", "k", "threshold", "threshold_hex", "support_status",
                 "mean_clean_YEAR1", "mean_clean_YEAR2", "joint_mean_clean_count",
                 "queried_target_budget", "relative_budget_error", "year",
                 "trajectory", "day", "first_unsupported_L", "local_rows", "local_dates"]
CAL_COLUMNS = ["policy", "k", "target_budget", "theta_global", "support_status",
               "mean_clean_YEAR1", "mean_clean_YEAR2", "joint_mean_clean_count",
               "relative_budget_error", "calibration_status", "warm_start_threshold",
               "new_refinement_evaluations", "primary_comparison_eligible"]
FRONTIER_COLUMNS = ["policy", "k", "highest_known_supported_threshold",
                    "upper_unsupported_threshold", "frontier_evaluations",
                    "frontier_status"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predecessor():
    actual = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "hash-object", PREDECESSOR],
        cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()
    require(actual == PREDECESSOR_BLOB, "Predecessor blob mismatch")
    spec = importlib.util.spec_from_file_location("_dev_a_predecessor", ROOT / PREDECESSOR)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    require(module.DEV_PERCEPTION_SEED == DEV_PERCEPTION_SEED
            and module.DEV_ENV_ROOTS == DEV_ENV_ROOTS and module.ETA == ETA
            and module.K_GRID == K_GRID and module.BUDGETS == BUDGETS,
            "Predecessor protocol mismatch")
    return module


class AccessAudit:
    """Guard IDs before slicing arrays or calling any perception substream."""

    def __init__(self, allowed_ids):
        self.allowed = set(allowed_ids)
        self.accessed = set()
        self.selection = set()
        self.perception_seeds = set()
        self.environment_seeds = set()
        self.rejected_accesses = 0
        self.query_calls = 0

    def ids(self, ids):
        ids = tuple(ids)
        self.selection.update(set(ids) & set(SELECTION_IDS))
        if len(set(ids)) != len(ids) or not set(ids) <= self.allowed:
            self.rejected_accesses += 1
            raise RuntimeError("Trajectory access outside permitted calibration IDs")
        self.accessed.update(ids)

    def perception(self, seed, year, trajectory, day):
        self.ids((trajectory,))
        self.perception_seeds.add(seed)
        require(seed == DEV_PERCEPTION_SEED and year in DEV_ENV_ROOTS
                and 0 <= day < 365, "Non-DEV perception access")
        self.query_calls += 1

    def environment(self, year, seed):
        self.environment_seeds.add(seed)
        require(year in DEV_ENV_ROOTS and seed == DEV_ENV_ROOTS[year],
                "Non-DEV environment access")

    def report(self):
        return {"calibration_ids_accessed": sorted(self.accessed),
                "selection_ids_accessed": sorted(self.selection),
                "perception_seeds_accessed": sorted(self.perception_seeds),
                "environment_seeds_accessed": sorted(self.environment_seeds),
                "rejected_accesses": self.rejected_accesses,
                "perception_query_calls": self.query_calls}


class GuardedFrozen:
    """Predecessor query uses this guard; the actual frozen seed function is reused."""

    def __init__(self, frozen, audit):
        self.frozen, self.audit = frozen, audit

    def perception_substream_seed(self, seed, year, trajectory, day):
        self.audit.perception(seed, year, trajectory, day)
        return self.frozen.perception_substream_seed(seed, year, trajectory, day)


def load_inputs(args, predecessor, frozen):
    paths = {name: getattr(args, name).expanduser().resolve() for name in
             ("master_ledger", "transition_audit", "paper1_cqr",
              "wapp_power_bridge", "frozen_trajectory_bank")}
    provenance = {name: {"path": str(path), "sha256": sha256_file(path)}
                  for name, path in paths.items()}
    for name in ("master_ledger", "transition_audit"):
        provenance[name].update(historical_expected_sha256=None,
                                historical_hash_status="no_historical_expected_hash")
    for name in ("paper1_cqr", "wapp_power_bridge", "frozen_trajectory_bank"):
        require(provenance[name]["sha256"] == frozen.EXPECTED_HASHES[name],
                f"Frozen input hash mismatch: {name}")
    ledger = frozen.load_ledger(paths["master_ledger"])
    transitions = frozen.load_transitions(paths["transition_audit"])
    bridge = frozen.load_wapp_power_bridge(paths["wapp_power_bridge"])
    valid = ledger.loc[ledger.state_valid, ["date", "L_power_proxy"]]
    require(len(ledger) == 730 and len(valid) == 729
            and np.array_equal(valid.date.dt.strftime("%Y-%m-%d"), bridge.date)
            and np.array_equal(valid.L_power_proxy.to_numpy(float),
                               bridge.L_power_proxy.to_numpy(float)), "Ledger bridge alignment")
    require(np.array_equal(ledger.date, pd.date_range("2021-08-09", "2023-08-08"))
            and int(ledger.rain_day.sum()) == 143
            and int(ledger.modb_manual_cleaning_day.sum()) == 26
            and len(transitions) == 729
            and np.array_equal(transitions.transition_index, np.arange(729))
            and np.array_equal(transitions.source_date, ledger.date.iloc[:-1])
            and np.array_equal(transitions.dest_date, ledger.date.iloc[1:])
            and transitions.transition_class.value_counts().to_dict() == {
                "DRY_NATURAL": 494, "RAIN_AFFECTED": 201,
                "MANUAL_CLEAN_CONTAMINATED": 26, "MAINTENANCE_ADJACENT": 6,
                "INVALID_GAP": 2}, "Ledger transition structure regression")
    context = frozen.attach_context(transitions, ledger)
    natural = context.transition_class.isin(["DRY_NATURAL", "RAIN_AFFECTED"])
    require(np.allclose(context.loc[natural, "delta_L_power_proxy"],
                        context.loc[natural, "dest_L"] - context.loc[natural, "source_L"],
                        atol=1e-12, rtol=0), "Transition numerical regression")
    dry = context.loc[context.transition_class.eq("DRY_NATURAL"), "delta_L_power_proxy"].to_numpy(float)
    rain = context.loc[context.transition_class.eq("RAIN_AFFECTED")]
    require(len(dry) == 494 and len(rain) == 201 and np.isfinite(dry).all()
            and np.isfinite(rain[["source_L", "delta_L_power_proxy"]].to_numpy(float)).all(),
            "Natural pool regression")
    r3 = frozen.fit_r3(rain)
    actual = [r3["intercept"], r3["slope"], float(rain.source_L.max()),
              float(np.std(r3["residuals"], ddof=1))]
    expected = [predecessor.EXPECTED_R3_INTERCEPT, predecessor.EXPECTED_R3_SLOPE,
                predecessor.EXPECTED_R3_SOURCE_L_MAX, predecessor.EXPECTED_R3_RESIDUAL_STD]
    require(len(r3["residuals"]) == 201 and np.isfinite(r3["residuals"]).all()
            and np.allclose(actual, expected, atol=1e-15, rtol=0), "Frozen R3 exact regression")
    provenance["natural_dynamics"] = {
        "frozen_r3_parameter_regression_pass": True, "atol": 1e-15, "rtol": 0,
        "residual_std_ddof": 1, "dry_pool_current_array_sha256": frozen.sha256_array(dry),
        "residual_pool_current_array_sha256": frozen.sha256_array(r3["residuals"])}
    source = frozen.load_paper1_cqr(paths["paper1_cqr"])
    emulator = frozen.OnlineBlock10Emulator(source)
    return ledger, dry, r3, emulator, provenance


def build_assets(frozen, ledger, dry, r3, audit):
    """One 300-column bank per year plus independent verification, no split RNG."""
    assets, hashes, regeneration = {}, {}, {}
    for year, seed in DEV_ENV_ROOTS.items():
        audit.environment(year, seed)
        require(seed == frozen.scenario_seed(420001, year), "DEV root mapping")
        calendar = frozen.get_year_calendar(ledger, year)
        indices, rain = frozen.build_env_crn_indices(
            calendar, len(dry), len(r3["residuals"]), TOTAL_DEV_TRAJECTORIES, seed)
        indices_2, rain_2 = frozen.build_env_crn_indices(
            calendar, len(dry), len(r3["residuals"]), TOTAL_DEV_TRAJECTORIES, seed)
        hash_1 = frozen.sha256_array(np.stack(indices))
        hash_2 = frozen.sha256_array(np.stack(indices_2))
        regeneration[year] = {
            "rain_exact": bool(np.array_equal(rain, rain_2)),
            "daily_indices_exact": bool(len(indices) == len(indices_2) and all(
                np.array_equal(left, right) for left, right in zip(indices, indices_2))),
            "stacked_sha256_exact": hash_1 == hash_2,
            "first_sha256": hash_1, "independent_sha256": hash_2}
        require(all(regeneration[year][key] for key in
                    ("rain_exact", "daily_indices_exact", "stacked_sha256_exact")),
                f"Independent environment CRN regression failed: {year}")
        require(len(calendar) == 365 and len(indices) == 364 and rain.shape == (364,),
                "Environment dimensions")
        for day, idx in enumerate(indices):
            require(idx.shape == (300,) and np.issubdtype(idx.dtype, np.integer)
                    and np.all(idx >= 0)
                    and np.all(idx < (len(r3["residuals"]) if rain[day] else len(dry))),
                    "Environment index range")
            idx.setflags(write=False)
        rain.setflags(write=False)
        initial = float(calendar.L_power_proxy.iloc[0])
        require(np.isfinite(initial) and 0 <= initial <= 1, "Invalid initial state")
        assets[year] = dict(year=year, seed=seed, calendar=calendar, initial=initial,
                            indices=indices, rain=rain, dry=dry, r3=r3)
        hashes[year] = hash_1
    return assets, hashes, regeneration


def subset_asset(asset, trajectory_ids, audit):
    audit.ids(trajectory_ids)
    selected = dict(asset)
    selected["indices"] = [idx[list(trajectory_ids)] for idx in asset["indices"]]
    return selected


def generic_rollout(predecessor, guarded, emulator, asset, policy, k, threshold,
                    n_trajectories, trajectory_ids, audit):
    ids = tuple(trajectory_ids)
    require(n_trajectories == len(ids) and n_trajectories > 0, "Trajectory dimensions")
    require(policy in ("True-State", "Point", "UA")
            and ((policy == "UA" and k in K_GRID) or (policy != "UA" and k is None)),
            "Policy/grid mismatch")
    require(np.isfinite(threshold) and 0 <= threshold <= (2 if policy == "UA" else 1),
            "Threshold outside fixed range")
    selected = subset_asset(asset, ids, audit)
    n, days = n_trajectories, len(asset["calendar"])
    require(days == 365, "Full-year rollout required")
    pre = np.empty((n, days))
    post = np.empty_like(pre)
    actions = np.zeros((n, days), dtype=bool)
    q50, width = np.full_like(pre, np.nan), np.full_like(pre, np.nan)
    pre[:, 0] = asset["initial"]
    for day in range(days):
        require(np.isfinite(pre[:, day]).all(), "Non-finite rollout")
        if policy == "True-State":
            actions[:, day] = pre[:, day] >= threshold
        else:
            for row, original_id in enumerate(ids):
                rec = predecessor.query(guarded, emulator, pre[row, day],
                                        asset["year"], original_id, day)
                require(all(np.isfinite(rec[key]) for key in predecessor.NUMERIC),
                        "Non-finite frozen perception")
                q50[row, day], width[row, day] = rec["q50"], rec["width"]
                score = rec["q50"] if policy == "Point" else rec["q50"] + k * rec["width"]
                actions[row, day] = score >= threshold
        post[:, day] = pre[:, day]
        clean = actions[:, day]
        post[clean, day] = (1.0 - ETA) * pre[clean, day]
        if day == days - 1:
            break
        idx = selected["indices"][day]
        if asset["rain"][day]:
            r3 = asset["r3"]
            delta = (float(r3["intercept"]) + float(r3["slope"]) * post[:, day]
                     + np.asarray(r3["residuals"], dtype=float)[idx])
        else:
            delta = asset["dry"][idx]
        raw = post[:, day] + delta
        require(np.isfinite(raw).all(), "Non-finite natural transition")
        pre[:, day + 1] = np.clip(raw, 0.0, 1.0)
    require(np.isfinite(pre).all() and np.isfinite(post).all(), "Incomplete rollout")
    return dict(pre=pre, post=post, actions=actions, q50=q50, width=width)


def regression(predecessor, frozen, guarded, emulator, assets, audit, mode):
    # Smoke's strict ID limit takes precedence over the 20-ID DEV regression.
    n = 10 if mode == "smoke" else 20
    ids = tuple(range(n))
    thresholds = {"True-State": (0.0625, 0.125), "Point": (0.0, 0.0625),
                  "UA": (0.0, 0.125)}
    records = []
    old_n = predecessor.SMOKE_N
    try:
        predecessor.SMOKE_N = n  # Private imported module only; source is untouched.
        for year, asset in assets.items():
            sub = subset_asset(asset, ids, audit)
            zero_threshold_results = {}
            for policy, values in thresholds.items():
                for threshold in values:
                    before = audit.query_calls
                    new = generic_rollout(predecessor, guarded, emulator, asset, policy,
                                          1.0 if policy == "UA" else None, threshold, n, ids, audit)
                    old = predecessor.rollout(guarded, emulator, sub, policy, threshold)
                    if threshold == 0.0:
                        zero_threshold_results[policy] = new
                    require(np.array_equal(new["actions"], old["actions"]), "Action regression")
                    keys = ("pre", "post") if policy == "True-State" else ("pre", "post", "q50", "width")
                    require(all(np.isfinite(new[key]).all() and np.isfinite(old[key]).all()
                                and np.max(np.abs(new[key] - old[key])) <= 1e-12
                                for key in keys), "Generic numerical regression")
                    if policy == "True-State":
                        require(audit.query_calls == before, "True-State perception access")
                        ref_pre, ref_actions = frozen.simulate_threshold_hidden_states(
                            sub["calendar"], sub["initial"], threshold, ETA, sub["dry"],
                            sub["r3"], sub["indices"], sub["rain"])
                        require(np.array_equal(new["pre"], ref_pre)
                                and np.array_equal(new["actions"], ref_actions), "Natural CLEAN regression")
                    records.append(dict(year=year, policy=policy, threshold=threshold, passed=True))
            # Same threshold zero forces identical actions and states for Point/UA.
            point, ua = zero_threshold_results["Point"], zero_threshold_results["UA"]
            require(all(np.array_equal(point[key], ua[key]) for key in ("pre", "q50", "width")),
                    "Policy-independent perception CRN regression")
            # Reverse-order replay uses the same original IDs and frozen substream.
            for trajectory in reversed(ids):
                for day in (364, 0):
                    rec = predecessor.query(guarded, emulator, point["pre"][trajectory, day],
                                            year, trajectory, day)
                    require(rec["q50"] == point["q50"][trajectory, day]
                            and rec["width"] == point["width"][trajectory, day], "Perception replay")
    finally:
        predecessor.SMOKE_N = old_n
    return {"n_trajectories": n, "trajectory_ids": list(ids), "checks": records,
            "numeric_tolerance": 1e-12, "actions_exact": True, "perception_crn_pass": True}


def budget_error(candidate, budget):
    return abs(candidate["joint_mean_clean_count"] - budget) / budget


def best_supported(candidates, budget):
    supported = [c for c in candidates if c["support_status"] == "SUPPORTED"]
    return min(supported, key=lambda c: (budget_error(c, budget), c["threshold"]), default=None)


class CandidateCache:
    """Only support and complete cleaning counts cross the rollout boundary."""

    def __init__(self, predecessor, guarded, emulator, assets, ids, audit):
        self.predecessor, self.guarded, self.emulator = predecessor, guarded, emulator
        self.assets, self.ids, self.audit = assets, tuple(ids), audit
        self.entries, self.targets, self.executions = {}, {}, {}
        self.year_threshold_calls = {}
        self.hits = 0
        self.ready = False

    @staticmethod
    def key(policy, k, threshold):
        require((policy == "UA" and k in K_GRID)
                or (policy in ("Point", "True-State") and k is None), "Cache policy key")
        return policy, k, float(threshold).hex()

    def candidates(self, policy, k):
        return [c for key, c in self.entries.items() if key[:2] == (policy, k)]

    def evaluate(self, policy, k, threshold, budget=None):
        require(self.ready, "Calibration forbidden before regression gates")
        key = self.key(policy, k, threshold)
        self.targets.setdefault(key, set()).add(budget)
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        require(key not in self.executions, "Duplicate threshold rollout attempt")
        self.executions[key] = 1
        self.year_threshold_calls[key] = []
        entry = dict(policy=policy, k=k, threshold=float(threshold), threshold_hex=key[2],
                     support_status="SUPPORTED", mean_clean_YEAR1=None, mean_clean_YEAR2=None,
                     joint_mean_clean_count=None, year=None, trajectory=None, day=None,
                     first_unsupported_L=None, local_rows=None, local_dates=None)
        counts = {}
        try:
            for year, asset in self.assets.items():
                self.year_threshold_calls[key].append((year, float(threshold).hex()))
                result = generic_rollout(self.predecessor, self.guarded, self.emulator,
                                         asset, policy, k, threshold, len(self.ids), self.ids, self.audit)
                counts[year] = float(result["actions"].sum(axis=1).mean())
                del result
        except self.predecessor.UnsupportedPerceptionQuery as exc:
            # Discard even a completed first-year count if the other year fails.
            entry["support_status"] = "UNSUPPORTED"
            entry.update(exc.context)
            cand = self.emulator._candidate_indices(entry["first_unsupported_L"])
            entry["local_dates"] = int(len(np.unique(self.emulator.date[cand])))
        else:
            entry.update(mean_clean_YEAR1=counts["YEAR1"], mean_clean_YEAR2=counts["YEAR2"],
                         joint_mean_clean_count=(counts["YEAR1"] + counts["YEAR2"]) / 2)
        self.entries[key] = entry
        return entry

    def rows(self):
        rows = []
        for key, candidate in self.entries.items():
            targets = sorted(self.targets[key], key=lambda b: -1 if b is None else b)
            for budget in targets:
                rows.append({**candidate, "queried_target_budget": budget,
                             "relative_budget_error": budget_error(candidate, budget)
                             if budget is not None and candidate["support_status"] == "SUPPORTED" else None})
        return rows


def frontier(cache, policy, k, limit):
    high = 2.0 if policy == "UA" else 1.0
    if policy == "True-State":
        cache.evaluate(policy, k, 0.0)
        cache.evaluate(policy, k, high)
        return dict(policy=policy, k=k, highest_known_supported_threshold=high,
                    upper_unsupported_threshold=None, frontier_evaluations=0,
                    frontier_status="NOT_APPLICABLE")
    supported, unsupported = None, None
    evaluations = 0
    for threshold in (0.0, high):
        candidate = cache.evaluate(policy, k, threshold)
        evaluations += 1
        if candidate["support_status"] == "SUPPORTED":
            supported = threshold
        else:
            unsupported = threshold
    if supported is not None and unsupported is not None and unsupported <= supported:
        unsupported = None  # A lower unsupported hole is not an upper frontier.
    # A failed zero-threshold probe alone is not proof that every operating
    # threshold fails. Use the remaining fixed bracket allowance to look lower.
    while evaluations < limit and unsupported is not None and (
            supported is None or supported < unsupported):
        threshold = ((0.0 if supported is None else supported) + unsupported) / 2.0
        if cache.key(policy, k, threshold) in cache.entries:
            break
        candidate = cache.evaluate(policy, k, threshold)
        evaluations += 1
        if candidate["support_status"] == "SUPPORTED":
            supported = threshold
        else:
            unsupported = threshold
    return dict(policy=policy, k=k, highest_known_supported_threshold=supported,
                upper_unsupported_threshold=unsupported, frontier_evaluations=evaluations,
                frontier_status="CALIBRATION_FAILURE" if supported is None else "HIGHEST_KNOWN_BRACKET")


def refine_budget(cache, policy, k, budget, frontier_info, warm, limit):
    """Cached count brackets first; deterministic largest-gap refinement otherwise.

    Directional refinement is an engineering search, not a claim of globally
    monotone counts or mathematically exact support. Unsupported holes remain
    in the cache; no proposal above the established supported frontier is made.
    """
    added = 0
    if warm is not None:
        cache.evaluate(policy, k, warm, budget)
    ceiling = frontier_info["highest_known_supported_threshold"]
    while ceiling is not None and added < limit:
        candidates = cache.candidates(policy, k)
        best = best_supported(candidates, budget)
        if best is not None and budget_error(best, budget) <= CALIBRATION_RELATIVE_ERROR_TARGET:
            break
        good = [c for c in candidates if c["support_status"] == "SUPPORTED"]
        bad = [c["threshold"] for c in candidates if c["support_status"] == "UNSUPPORTED"]
        brackets = [(a["threshold"], b["threshold"]) for a in good for b in good
                    if a["threshold"] < b["threshold"]
                    and a["joint_mean_clean_count"] >= budget >= b["joint_mean_clean_count"]
                    and not any(a["threshold"] < t < b["threshold"] for t in bad)]
        brackets.sort(key=lambda pair: (pair[1] - pair[0], pair[0]))
        known = sorted(set([0.0, ceiling] + [c["threshold"] for c in candidates
                                            if c["threshold"] <= ceiling]))
        gaps = [(a, b) for a, b in zip(known[:-1], known[1:])
                if a not in bad and a < b]
        gaps.sort(key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
        proposal = None
        for low, high in brackets + gaps:
            middle = (low + high) / 2.0
            if low < middle < high and cache.key(policy, k, middle) not in cache.entries:
                proposal = middle
                break
        if proposal is None:
            break
        cache.evaluate(policy, k, proposal, budget)
        added += 1
    best = best_supported(cache.candidates(policy, k), budget)
    status = "CALIBRATION_FAILURE" if best is None else "CALIBRATION_TARGET_NOT_MET"
    if best is not None:
        cache.evaluate(policy, k, best["threshold"], budget)  # Explicit cache hit, no rollout.
        if budget_error(best, budget) <= CALIBRATION_RELATIVE_ERROR_TARGET:
            status = "SUPPORTED_MATCHED"
        elif policy != "True-State":
            good = [c for c in cache.candidates(policy, k) if c["support_status"] == "SUPPORTED"]
            highest = max(c["threshold"] for c in good)
            upper = frontier_info["upper_unsupported_threshold"]
            if (upper is not None and upper > highest
                    and min(c["joint_mean_clean_count"] for c in good)
                    > budget * (1 + CALIBRATION_RELATIVE_ERROR_TARGET)):
                status = "SUPPORT_INFEASIBLE_WITHIN_PROTOCOL"
    return dict(policy=policy, k=k, target_budget=budget,
                theta_global=None if best is None else best["threshold"],
                support_status=None if best is None else best["support_status"],
                mean_clean_YEAR1=None if best is None else best["mean_clean_YEAR1"],
                mean_clean_YEAR2=None if best is None else best["mean_clean_YEAR2"],
                joint_mean_clean_count=None if best is None else best["joint_mean_clean_count"],
                relative_budget_error=None if best is None else budget_error(best, budget),
                calibration_status=status, warm_start_threshold=warm,
                new_refinement_evaluations=added,
                primary_comparison_eligible=status == "SUPPORTED_MATCHED")


def common_budget_set(rows, mode):
    if mode == "smoke":
        return {"common_budget_set_status": "SMOKE-ONLY", "common_budget_set": [],
                "eligible_k_values": [], "k_calibration_eligible": []}
    lookup = {(r["policy"], r["k"], r["target_budget"]): r["calibration_status"] for r in rows}
    common, eligible = [], []
    for suffix in (BUDGETS, BUDGETS[1:], BUDGETS[2:], BUDGETS[3:]):
        if all(lookup.get(("Point", None, b)) == "SUPPORTED_MATCHED" for b in suffix):
            eligible = [k for k in K_GRID if all(
                lookup.get(("UA", k, b)) == "SUPPORTED_MATCHED" for b in suffix)]
            if eligible:
                common = list(suffix)
                break
    return {"common_budget_set_status": "SUPPORTED_COMMON_RANGE" if common
            else "YELLOW_INSUFFICIENT_COMMON_RANGE", "common_budget_set": common,
            "eligible_k_values": eligible,
            "k_calibration_eligible": [{"k": k, "eligible": bool(common and k in eligible)} for k in K_GRID]}


def protocol(mode):
    return {"stage": STAGE, "title": TITLE, "mode": mode,
            "scope": "SMOKE-ONLY" if mode == "smoke" else "DEV-A CALIBRATION ONLY",
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "predecessor_stage": "P2-1C-2C-v1.1", "predecessor_status": "PASS — CLOSED",
            "predecessor_head_commit": PREDECESSOR_HEAD, "predecessor_script_blob": PREDECESSOR_BLOB,
            "protocol_constants": {
                "TOTAL_DEV_TRAJECTORIES": TOTAL_DEV_TRAJECTORIES,
                "CALIBRATION_IDS": list(CALIBRATION_IDS), "SELECTION_IDS": list(SELECTION_IDS),
                "DEV_PERCEPTION_SEED": DEV_PERCEPTION_SEED, "DEV_ENV_ROOTS": DEV_ENV_ROOTS,
                # Formal values are documentation only; no runtime consumer exists.
                "FORMAL_PERCEPTION_SEEDS": [20260907, 20260908, 20260909, 20260910, 20260911],
                "FORMAL_ENV_PARENT_SEED": 620001,
                "FORMAL_ENV_ROOTS": {"YEAR1": 1220001, "YEAR2": 1230001},
                "K_GRID": K_GRID, "BUDGETS": BUDGETS, "SEARCH_ORDER": SEARCH_ORDER,
                "CALIBRATION_RELATIVE_ERROR_TARGET": CALIBRATION_RELATIVE_ERROR_TARGET,
                "MAX_SUPPORT_BRACKET_EVALS": MAX_SUPPORT_BRACKET_EVALS,
                "MAX_REFINEMENT_EVALS_PER_BUDGET": MAX_REFINEMENT_EVALS_PER_BUDGET,
                "eta": ETA, "dry": "D0_GLOBAL", "rain": "R3_OLS_RESIDUAL"},
            "runtime_settings": {"trajectory_ids": list(range(10)) if mode == "smoke" else list(CALIBRATION_IDS),
                                 "budgets": [10, 6] if mode == "smoke" else SEARCH_ORDER,
                                 "ua_k": [1.0] if mode == "smoke" else K_GRID,
                                 "support_limit": 6 if mode == "smoke" else 12,
                                 "refinement_limit": 4 if mode == "smoke" else 10},
            "objective": "abs(joint_mean_clean_count - target_budget)",
            "tie_break": "smaller threshold", "threshold_scope": "global, identical across years",
            "cache_key": ["policy", "k_or_null", "float.hex(threshold)"],
            "trajectory_day_indexing": "zero-based original IDs",
            "generic_regression_setting": {
                "n_trajectories": 10 if mode == "smoke" else 20,
                "k": 1.0, "perception_seed": DEV_PERCEPTION_SEED,
                "environment": "prefix of the same once-generated 300-trajectory CRN",
                "smoke_id_limit": "SMOKE-ONLY: 0..9 including regression; private predecessor SMOKE_N restored afterward"},
            "search": "Cached Support-Aware Coarse-to-Fine Search",
            "support_frontier_interpretation": "highest-known supported bracket, not an exact boundary",
            "support_infeasibility_interpretation": (
                "This is protocol-scoped support infeasibility, "
                "not a proof of mathematical or physical impossibility.")}


def leakage_check(value):
    """Recursively reject performance fields; support failure L is diagnostic only."""
    forbidden = {"j_post", "j_pre", "reward", "profit", "cost-benefit", "cost", "k_star",
                 "mean_hidden_state", "tail_hidden_state", "p95_hidden_state", "max_hidden_state",
                 "mean_l", "tail_l", "max_l_pre", "p95_l_pre", "ua_improvement", "point_improvement",
                 "pre", "post", "actions", "q50", "width", "hidden_states"}
    if isinstance(value, dict):
        require(not any(str(key).lower() in forbidden or "improvement" in str(key).lower()
                        for key in value), "Decision-value output leakage")
        for item in value.values():
            leakage_check(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            leakage_check(item)


def dump(path, value):
    leakage_check(value)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    require(json.loads(path.read_text(encoding="utf-8")) == value, "JSON readback")


def write_csv(path, rows, columns):
    leakage_check(dict.fromkeys(columns))
    require(all(set(row) == set(columns) for row in rows), "CSV schema mismatch")
    with path.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows, columns=columns).to_csv(stream, index=False)
    loaded = pd.read_csv(path)
    require(list(loaded.columns) == columns and len(loaded) == len(rows), "CSV readback")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "dev-a"], required=True)
    timeline = BASE / "p2_1a_timeline_intervention_audit_v1"
    parser.add_argument("--master-ledger", type=Path, default=timeline / "environment_master_ledger.csv")
    parser.add_argument("--transition-audit", type=Path, default=timeline / "transition_audit.csv")
    parser.add_argument("--paper1-cqr", type=Path, required=True)
    parser.add_argument("--wapp-power-bridge", type=Path, required=True)
    parser.add_argument("--frozen-trajectory-bank", type=Path,
                        default=BASE / "p2_0c_3c_perception_trajectory_bank_v1/trajectory_bank.csv")
    args = parser.parse_args()
    out = OUT / ("smoke" if args.mode == "smoke" else "dev_a")
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Refusing existing output directory: {out}", file=sys.stderr)
        return 2
    manifest = protocol(args.mode)
    manifest["script_sha256"] = sha256_file(Path(__file__))
    settings = manifest["runtime_settings"]
    audit = AccessAudit(settings["trajectory_ids"])
    gates = dict.fromkeys(GATES, False)
    summary = {"stage": STAGE, "mode": args.mode, "scope": manifest["scope"],
               "gates": gates, "stage_pass": False, "errors": []}
    rows, frontiers, cache_rows = [], [], []
    cache = None
    try:
        predecessor = load_predecessor()
        gates[GATES[0]] = True
        frozen, sources = predecessor.load_frozen()
        manifest["frozen_sources"] = sources
        ledger, dry, r3, emulator, inputs = load_inputs(args, predecessor, frozen)
        manifest["input_provenance"] = inputs
        gates[GATES[1]] = True
        require(frozen.MIN_LOCAL_SAMPLES == 20 and frozen.MIN_LOCAL_DATES == 3,
                "Frozen support constants")
        manifest["perception_support"] = {"local_radius": frozen.LOCAL_RADIUS,
            "min_local_samples": frozen.MIN_LOCAL_SAMPLES, "min_local_dates": frozen.MIN_LOCAL_DATES,
            "kernel_bandwidth": frozen.KERNEL_BANDWIDTH, "block_minutes": frozen.BLOCK_MINUTES,
            "fallback": False, "radius_widening": False, "original_csv_order": True,
            "implementation": "predecessor.query / frozen OnlineBlock10Emulator.sample_one"}
        assets, original_hashes, regeneration = build_assets(frozen, ledger, dry, r3, audit)
        manifest["environment_crn_array_sha256"] = original_hashes
        summary["environment_crn_regression"] = regeneration
        gates[GATES[3]] = all(regeneration[year][key] for year in DEV_ENV_ROOTS for key in
                              ("rain_exact", "daily_indices_exact", "stacked_sha256_exact"))
        guarded = GuardedFrozen(frozen, audit)
        summary["generic_regression"] = regression(predecessor, frozen, guarded, emulator,
                                                     assets, audit, args.mode)
        gates[GATES[2]] = True
        gates[GATES[4]] = summary["generic_regression"]["perception_crn_pass"]
        cache = CandidateCache(predecessor, guarded, emulator, assets,
                               settings["trajectory_ids"], audit)
        cache.ready = all(gates[name] for name in (GATES[0], GATES[1], GATES[2], GATES[4]))
        policies = [("True-State", None), ("Point", None)] + [("UA", k) for k in settings["ua_k"]]
        for policy, k in policies:
            info = frontier(cache, policy, k, settings["support_limit"])
            frontiers.append(info)
            warm = None
            for budget in settings["budgets"]:
                row = refine_budget(cache, policy, k, budget, info, warm, settings["refinement_limit"])
                rows.append(row)
                warm = row["theta_global"]
        # New candidates discovered by later budgets can improve earlier budgets.
        # Re-select mechanically from the final shared cache without any rollout.
        for i, row in enumerate(rows):
            info = next(f for f in frontiers if (f["policy"], f["k"]) == (row["policy"], row["k"]))
            final = refine_budget(cache, row["policy"], row["k"], row["target_budget"], info, None, 0)
            final["warm_start_threshold"] = row["warm_start_threshold"]
            final["new_refinement_evaluations"] = row["new_refinement_evaluations"]
            rows[i] = final
        gates[GATES[8]] = all(
            cache.year_threshold_calls[key] == [(year, key[2]) for year in DEV_ENV_ROOTS]
            for key, candidate in cache.entries.items() if candidate["support_status"] == "SUPPORTED")
        gates[GATES[9]] = all(c["support_status"] in ("SUPPORTED", "UNSUPPORTED")
                              and (c["support_status"] != "UNSUPPORTED" or all(c[key] is None for key in
                                   ("mean_clean_YEAR1", "mean_clean_YEAR2", "joint_mean_clean_count")))
                              for c in cache.entries.values())
        gates[GATES[10]] = (cache.hits > 0 and len(cache.executions) == len(cache.entries)
                            and all(n == 1 for n in cache.executions.values())
                            and all(key == cache.key(c["policy"], c["k"], c["threshold"])
                                    for key, c in cache.entries.items()))
        gates[GATES[11]] = all(r["theta_global"] is None or r["theta_global"] == best_supported(
            cache.candidates(r["policy"], r["k"]), r["target_budget"])["threshold"] for r in rows)
    except Exception as exc:
        # Preserve failures in the immutable audit; calibration catches only Unsupported queries.
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    gates[GATES[5]] = (set(CALIBRATION_IDS).isdisjoint(SELECTION_IDS)
                       and set(CALIBRATION_IDS) | set(SELECTION_IDS) == set(range(300))
                       and audit.accessed <= set(settings["trajectory_ids"]))
    gates[GATES[6]] = not audit.selection and audit.rejected_accesses == 0
    gates[GATES[7]] = (audit.perception_seeds == {DEV_PERCEPTION_SEED}
                       and audit.environment_seeds == set(DEV_ENV_ROOTS.values()))
    summary["access_audit"] = audit.report()
    if cache is not None:
        cache_rows = cache.rows()
        summary["cache_audit"] = {"hits": cache.hits, "unique_candidates": len(cache.entries),
                                  "rollout_attempts": len(cache.executions)}
    common = common_budget_set(rows, args.mode)
    summary["common_budget_set_status"] = common["common_budget_set_status"]
    prefix = "smoke_" if args.mode == "smoke" else ""
    try:
        for value in (manifest, summary, rows, cache_rows, frontiers, common):
            leakage_check(value)
        gates[GATES[12]] = True
        dump(out / "protocol_manifest.json", manifest)
        write_csv(out / f"{prefix}candidate_cache.csv", cache_rows, CACHE_COLUMNS)
        write_csv(out / f"{prefix}calibration_summary.csv", rows, CAL_COLUMNS)
        write_csv(out / "support_frontier_summary.csv", frontiers, FRONTIER_COLUMNS)
        dump(out / "common_budget_set.json", common)
        required = ["protocol_manifest.json", f"{prefix}candidate_cache.csv",
                    f"{prefix}calibration_summary.csv", "support_frontier_summary.csv", "common_budget_set.json"]
        expected_rows = (2 + len(settings["ua_k"])) * len(settings["budgets"])
        audit_name = "smoke_audit_summary.json" if args.mode == "smoke" else "dev_a_audit_summary.json"
        complete = len(rows) == expected_rows and all((out / name).is_file() for name in required)
        if args.mode == "dev-a":
            # Prepare the intended audit in memory. G14 and stage_pass in the
            # live summary remain False until all seven files are verified.
            final_summary = json.loads(json.dumps(summary, allow_nan=False))
            final_summary["gates"][GATES[13]] = complete
            final_summary["stage_pass"] = bool(
                all(final_summary["gates"].values()) and not final_summary["errors"])
            final_summary["numbered_gates"] = {
                f"G{i}": {"name": name, "pass": final_summary["gates"][name]}
                for i, name in enumerate(GATES, 1)}
            audit_bytes = json.dumps(final_summary, ensure_ascii=False, indent=2,
                                    allow_nan=False).encode("utf-8")
            hashes = {name: sha256_file(out / name) for name in required}
            hashes[audit_name] = hashlib.sha256(audit_bytes).hexdigest()
            calibration_manifest = {"stage": STAGE, "final_stage_status_source": audit_name,
                "selected_global_thresholds": rows, **common,
                "calibration_status": common["common_budget_set_status"] if final_summary["stage_pass"] else "AUDIT_FAIL",
                "protocol_constants": manifest["protocol_constants"],
                "input_source_provenance": {"predecessor_script_blob": PREDECESSOR_BLOB,
                    "predecessor_head_commit": PREDECESSOR_HEAD,
                    "inputs": manifest.get("input_provenance", {}), "sources": manifest.get("frozen_sources", [])},
                "output_file_sha256": hashes,
                "hash_scope": "All other output files; this manifest cannot hash itself."}
            dump(out / "dev_a_calibration_manifest.json", calibration_manifest)
            required.extend(["dev_a_calibration_manifest.json", audit_name])
            # Stage the audit only after the calibration manifest succeeds.
            # Binary output makes the predeclared audit hash independent of
            # Windows newline translation. Never publish a PASS audit early.
            pending = out / (audit_name + ".pending")
            leakage_check(final_summary)
            with pending.open("xb") as stream:
                stream.write(audit_bytes)
            require(json.loads(pending.read_text(encoding="utf-8")) == final_summary,
                    "Pending audit JSON readback")
            reread_manifest = json.loads((out / "dev_a_calibration_manifest.json").read_text(encoding="utf-8"))
            require(reread_manifest == calibration_manifest, "Calibration manifest JSON readback")
            require(all(sha256_file(pending if name == audit_name else out / name) == digest
                        for name, digest in reread_manifest["output_file_sha256"].items()),
                    "Calibration manifest output SHA256 verification")
            require(all((pending if name == audit_name else out / name).is_file() for name in required),
                    "Required DEV-A outputs incomplete")
            require(not (out / audit_name).exists(), "Refusing existing final audit")
            pending.rename(out / audit_name)  # Final status is published last.
            require(all((out / name).is_file() for name in required), "Final output completeness")
            summary.update(final_summary)
            gates.update(final_summary["gates"])
        else:
            gates[GATES[13]] = complete
            summary["stage_pass"] = bool(all(gates.values()) and not summary["errors"])
            summary["numbered_gates"] = {f"G{i}": {"name": name, "pass": gates[name]}
                                           for i, name in enumerate(GATES, 1)}
            dump(out / audit_name, summary)
    except Exception as exc:
        print(f"Output failure: {exc}", file=sys.stderr)
        return 2
    print(f"{STAGE} stage_pass={summary['stage_pass']}")
    return 0 if summary["stage_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
