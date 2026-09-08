#!/usr/bin/env python
"""Sealed DEV-B evaluation. Import is inert; neither CLI mode calibrates thresholds."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUT = BASE / "p2_1c_2c_dev_b_decision_value_selection_v1_1"
STAGE = "P2-1C-2C-DEV-B-v1.1"
TITLE = "Sealed-Selection Matched-Budget Decision-Value Evaluation"
DEV_A_SCRIPT = "experiments/run_paper2_stage1c2c_dev_a_matched_budget_calibration_v1.py"
DEV_A_BLOB = "4931c16f81c25bf2c560fdced92fb2fb464e4500"
DEV_A_HEAD = "e992c732a9f7ec802cd79b4a45294863c0b03217"
DEV_A_MANIFEST = (BASE / "p2_1c_2c_dev_a_matched_budget_calibration_v1"
                  / "dev_a/dev_a_calibration_manifest.json")
DEV_A_MANIFEST_SHA256 = "c5530cd0e12a8ac8421f565f3a8403af717175a3e4c2b1f715988b80f1d4b1ac"
DEV_PERCEPTION_SEED = 20260906
DEV_ENV_ROOTS = {"YEAR1": 1020001, "YEAR2": 1030001}
TOTAL_DEV_TRAJECTORIES = 300
SELECTION_IDS = tuple(range(100, 300))
COMMON_BUDGETS = [8, 10, 12]
K_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
ETA = 0.95
FAIRNESS_TOLERANCE = 0.05
TIE_PERCENTAGE_POINTS = 0.25
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 76020917  # Statistical-only namespace; never supplied to the environment/sampler.
GATES = ["dev_a_script_blob_pass", "dev_a_manifest_sha256_pass",
         "dev_a_referenced_output_hashes_pass", "dev_a_audit_pass",
         "environment_crn_exact_regeneration_pass", "generic_rollout_regression_pass",
         "selection_calibration_access_separation_pass", "no_formal_seed_access_pass",
         "frozen_threshold_read_only_pass", "point_ua_perception_crn_pass",
         "fairness_computation_identity_pass", "metric_identity_pass",
         "no_threshold_recalibration_pass", "output_completeness_pass",
         "selection_support_classification_pass"]
SUPPORTED = "SELECTION_SUPPORTED"
UNSUPPORTED = "SELECTION_UNSUPPORTED_WITHIN_FROZEN_PERCEPTION_SUPPORT"
TRUE_STATE_SUPPORT = "NOT_APPLICABLE_TRUE_STATE"
FIRST_FAILURE = {"year": "YEAR1", "trajectory": 180, "day": 192,
                 "policy": "UA", "k": 0.75, "budget": 8, "threshold": 0.3671875,
                 "first_unsupported_L": 0.21705866981982186, "local_rows": 17}
SUPPORT_COLUMNS = ["policy", "k", "budget", "year", "theta_global", "selection_support_status",
                   "trajectory", "day", "first_unsupported_L", "local_rows", "local_dates",
                   "k_support_eligible"]
FAIR_COLUMNS = ["policy", "k", "budget", "year", "mean_clean_count",
                "joint_mean_clean_count", "joint_target_error", "joint_target_pass",
                "pairwise_point_error", "pairwise_point_pass", "k_selection_eligible"]
SECONDARY = ["J_pre", "P_L_pre_gt_0.05", "P_L_pre_gt_0.10", "P_L_pre_gt_0.15",
             "P95_L_pre", "max_L_pre", "clean_count", "consecutive_clean_pairs",
             "max_consecutive_clean_run"]
TRAJECTORY_COLUMNS = ["policy", "k", "budget", "year", "trajectory", "theta_global",
                      "J_post", *SECONDARY, "paired_point_J_post", "paired_difference"]
CELL_COLUMNS = ["policy", "k", "budget", "year", "theta_global", "n_trajectories",
                "mean_J_post", *["mean_" + name for name in SECONDARY],
                "Delta_percent", "paired_mean_difference", "paired_median_difference",
                "win_fraction", "paired_bootstrap_ci_low", "paired_bootstrap_ci_high",
                "bootstrap_cell_seed", "primary_matched_budget_eligible"]
K_COLUMNS = ["k", "k_support_eligible", "k_fairness_eligible",
             "k_selection_eligible", "Score_percent", "selected"]
REFERENCE_COLUMNS = ["k", "budget", "year", "mean_J_True", "mean_J_Point",
                     "true_point_clean_count_error", "GapClosure", "invalid_reason"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dev_a(gates):
    blob = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "hash-object", DEV_A_SCRIPT],
        cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()
    require(blob == DEV_A_BLOB, "DEV-A script blob mismatch")
    gates[GATES[0]] = True
    require(sha256_file(DEV_A_MANIFEST) == DEV_A_MANIFEST_SHA256, "DEV-A manifest SHA256 mismatch")
    gates[GATES[1]] = True
    calibration = json.loads(DEV_A_MANIFEST.read_text(encoding="utf-8"))
    expected_files = {"protocol_manifest.json", "candidate_cache.csv", "calibration_summary.csv",
                      "support_frontier_summary.csv", "common_budget_set.json", "dev_a_audit_summary.json"}
    require(set(calibration["output_file_sha256"]) == expected_files, "DEV-A referenced file set mismatch")
    for name, expected in calibration["output_file_sha256"].items():
        require(Path(name).name == name, "Invalid DEV-A referenced filename")
        require(sha256_file(DEV_A_MANIFEST.parent / name) == expected, f"DEV-A output SHA256 mismatch: {name}")
    gates[GATES[2]] = True
    audit = json.loads((DEV_A_MANIFEST.parent / "dev_a_audit_summary.json").read_text(encoding="utf-8"))
    require(audit["stage_pass"] is True and len(audit["gates"]) == 14
            and all(v is True for v in audit["gates"].values())
            and all(audit["numbered_gates"][f"G{i}"]["pass"] is True for i in range(1, 15)),
            "DEV-A audit is not 14/14 PASS")
    require(calibration["calibration_status"] == "SUPPORTED_COMMON_RANGE"
            and calibration["common_budget_set"] == COMMON_BUDGETS
            and calibration["eligible_k_values"] == K_GRID, "Frozen DEV-A result mismatch")
    gates[GATES[3]] = True
    protocol = json.loads((DEV_A_MANIFEST.parent / "protocol_manifest.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("_sealed_dev_a", ROOT / DEV_A_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    require(module.DEV_PERCEPTION_SEED == DEV_PERCEPTION_SEED and module.DEV_ENV_ROOTS == DEV_ENV_ROOTS
            and module.TOTAL_DEV_TRAJECTORIES == TOTAL_DEV_TRAJECTORIES
            and tuple(module.SELECTION_IDS) == SELECTION_IDS and module.K_GRID == K_GRID
            and module.ETA == ETA, "DEV-A runtime protocol drift")
    return module, calibration, protocol


def frozen_thresholds(calibration):
    thresholds = {}
    policies = [("True-State", None), ("Point", None)] + [("UA", k) for k in K_GRID]
    for policy, k in policies:
        for budget in COMMON_BUDGETS:
            matches = [row for row in calibration["selected_global_thresholds"]
                       if (row["policy"], row["k"], row["target_budget"]) == (policy, k, budget)]
            require(len(matches) == 1, "Missing or duplicate frozen threshold")
            row = matches[0]
            require(row["calibration_status"] == "SUPPORTED_MATCHED"
                    and row["support_status"] == "SUPPORTED"
                    and np.isfinite(row["theta_global"]), "Ineligible frozen threshold")
            thresholds[policy, k, budget] = row["theta_global"]
    return MappingProxyType(thresholds)


class PhaseAudit:
    """Separate objects for engineering regression and the evaluation population."""

    def __init__(self, phase, allowed_ids):
        self.phase, self.allowed = phase, set(allowed_ids)
        self.accessed, self.perception_seeds, self.environment_seeds = set(), set(), set()
        self.query_calls = 0
        self.rejected = 0

    def ids(self, ids):
        ids = tuple(ids)
        if len(ids) != len(set(ids)) or not set(ids) <= self.allowed:
            self.rejected += 1
            raise RuntimeError(f"Forbidden {self.phase} trajectory access")
        self.accessed.update(ids)

    def perception(self, seed, year, trajectory, day):
        self.ids((trajectory,))
        self.perception_seeds.add(seed)
        require(seed == DEV_PERCEPTION_SEED and year in DEV_ENV_ROOTS and 0 <= day < 365,
                "Non-DEV perception access")
        self.query_calls += 1

    def environment(self, year, seed):
        self.environment_seeds.add(seed)
        require(year in DEV_ENV_ROOTS and seed == DEV_ENV_ROOTS[year], "Non-DEV environment access")

    def report(self):
        return {"phase": self.phase, "trajectory_ids_accessed": sorted(self.accessed),
                "perception_seeds_accessed": sorted(self.perception_seeds),
                "environment_seeds_accessed": sorted(self.environment_seeds),
                "query_calls": self.query_calls, "rejected_accesses": self.rejected}


def load_models(dev_a, calibration):
    predecessor = dev_a.load_predecessor()
    frozen, sources = predecessor.load_frozen()
    provenance = calibration["input_source_provenance"]
    paths = {name: Path(provenance["inputs"][name]["path"]) for name in
             ("master_ledger", "transition_audit", "paper1_cqr", "wapp_power_bridge", "frozen_trajectory_bank")}
    for name, path in paths.items():
        require(sha256_file(path) == provenance["inputs"][name]["sha256"], f"Frozen DEV-A input drift: {name}")
    require(sources == provenance["sources"], "Frozen source provenance drift")
    ledger, dry, r3, emulator, current = dev_a.load_inputs(SimpleNamespace(**paths), predecessor, frozen)
    for key in ("dry_pool_current_array_sha256", "residual_pool_current_array_sha256"):
        require(current["natural_dynamics"][key] == provenance["inputs"]["natural_dynamics"][key],
                "DEV-A dynamics array hash drift")
    return predecessor, frozen, ledger, dry, r3, emulator


def build_assets(frozen, ledger, dry, r3, protocol, audit):
    assets, records = {}, {}
    for year, seed in DEV_ENV_ROOTS.items():
        audit.environment(year, seed)
        calendar = frozen.get_year_calendar(ledger, year)
        # Independently recreate the full DEV-A bank once; only later slice IDs.
        indices, rain = frozen.build_env_crn_indices(
            calendar, len(dry), len(r3["residuals"]), TOTAL_DEV_TRAJECTORIES, seed)
        digest = frozen.sha256_array(np.stack(indices))
        require(digest == protocol["environment_crn_array_sha256"][year], "DEV-A CRN hash mismatch")
        expected_rain = calendar.rain_day.to_numpy(bool)
        require(len(calendar) == 365 and len(indices) == 364
                and all(idx.shape == (300,) for idx in indices)
                and np.array_equal(rain, expected_rain[:-1] | expected_rain[1:]), "CRN shape/rain mismatch")
        for idx in indices:
            idx.setflags(write=False)
        rain.setflags(write=False)
        assets[year] = dict(year=year, seed=seed, calendar=calendar,
                            initial=float(calendar.L_power_proxy.iloc[0]),
                            indices=indices, rain=rain, dry=dry, r3=r3)
        records[year] = {"sha256": digest, "frozen_sha256": protocol["environment_crn_array_sha256"][year],
                         "exact_match": True, "generated_trajectories": 300}
    return assets, records


def classify_cell(predecessor, emulator, policy, k, budget, year, threshold, rollout_call):
    """The sole recoverable exception is the frozen typed unsupported query."""
    row = dict(policy=policy, k=k, budget=budget, year=year, theta_global=threshold,
               selection_support_status=TRUE_STATE_SUPPORT if policy == "True-State" else SUPPORTED,
               trajectory=None, day=None, first_unsupported_L=None, local_rows=None,
               local_dates=None, k_support_eligible=None)
    try:
        result = rollout_call()
    except predecessor.UnsupportedPerceptionQuery as exc:
        if policy == "True-State":
            raise
        require(exc.context["year"] == year, "Unsupported query year mismatch")
        row["selection_support_status"] = UNSUPPORTED
        for key in ("trajectory", "day", "first_unsupported_L", "local_rows"):
            row[key] = exc.context[key]
        candidates = emulator._candidate_indices(row["first_unsupported_L"])
        row["local_dates"] = int(len(np.unique(emulator.date[candidates])))
        return None, row  # No partial arrays or counts cross this boundary.
    return result, row


def support_screening(support_rows, results, budgets, ks):
    expected = {(p, k, b, y) for p, k in [("True-State", None), ("Point", None)] + [("UA", k) for k in ks]
                for b in budgets for y in DEV_ENV_ROOTS}
    by_key = {(r["policy"], r["k"], r["budget"], r["year"]): r for r in support_rows}
    require(len(by_key) == len(support_rows) and set(by_key) == expected, "Support cell completeness")
    require(all(row["selection_support_status"] in (SUPPORTED, UNSUPPORTED, TRUE_STATE_SUPPORT)
                and (row["selection_support_status"] == TRUE_STATE_SUPPORT) == (row["policy"] == "True-State")
                for row in support_rows), "Support classification semantics")
    require(set(results) == {key for key, row in by_key.items() if row["selection_support_status"] != UNSUPPORTED},
            "Partial/unsupported rollout entered results")
    point_support = all(by_key["Point", None, b, y]["selection_support_status"] == SUPPORTED
                        for b in budgets for y in DEV_ENV_ROOTS)
    k_support = {k: all(by_key["UA", k, b, y]["selection_support_status"] == SUPPORTED
                       for b in budgets for y in DEV_ENV_ROOTS) for k in ks}
    for row in support_rows:
        if row["policy"] == "UA":
            row["k_support_eligible"] = k_support[row["k"]]
    # Remove the entire incomplete k, including its successfully completed cells.
    screened = {key: result for key, result in results.items()
                if point_support and (key[0] != "UA" or k_support[key[1]])}
    return point_support, k_support, screened


def repair_regression(predecessor, frozen, emulator):
    """Direct fixed-L support probe; historical ID 180 is metadata, never accessed.

    The frozen sampler rejects this query before using randomness. A sentinel
    forbids stochastic sampling, so this is also safe in smoke's 0..9 isolation.
    """
    names = ("LOCAL_RADIUS", "MIN_LOCAL_SAMPLES", "MIN_LOCAL_DATES", "KERNEL_BANDWIDTH", "BLOCK_MINUTES")
    before = {name: getattr(frozen, name) for name in names}

    class NoSamplingRNG:
        def choice(self, *args, **kwargs):
            raise RuntimeError("Known unsupported probe unexpectedly attempted random sampling")

    def probe():
        try:
            emulator.sample_one(FIRST_FAILURE["first_unsupported_L"], NoSamplingRNG())
        except RuntimeError as exc:
            if not str(exc).startswith("Unsupported query "):
                raise
            candidates = emulator._candidate_indices(FIRST_FAILURE["first_unsupported_L"])
            context = {**FIRST_FAILURE, "local_rows": int(len(candidates))}
            raise predecessor.UnsupportedPerceptionQuery(str(exc), context) from exc
        raise RuntimeError("Known frozen unsupported query unexpectedly succeeded")

    result, row = classify_cell(predecessor, emulator, FIRST_FAILURE["policy"], FIRST_FAILURE["k"],
                                FIRST_FAILURE["budget"], FIRST_FAILURE["year"], FIRST_FAILURE["threshold"], probe)
    require(result is None and row["selection_support_status"] == UNSUPPORTED
            and all(row[key] == FIRST_FAILURE[key] for key in
                    ("policy", "k", "budget", "year", "trajectory", "day", "first_unsupported_L", "local_rows"))
            and row["theta_global"] == FIRST_FAILURE["threshold"], "Repair classification regression")
    sentinel = RuntimeError("Non-support exception propagation sentinel")

    def other_error():
        raise sentinel

    try:
        classify_cell(predecessor, emulator, "UA", FIRST_FAILURE["k"], FIRST_FAILURE["budget"],
                      FIRST_FAILURE["year"], FIRST_FAILURE["threshold"], other_error)
    except RuntimeError as exc:
        require(exc is sentinel, "Other RuntimeError was changed")
    else:
        raise RuntimeError("Other RuntimeError was swallowed")
    require(before == {name: getattr(frozen, name) for name in names}, "Support constants changed")
    return {"known_failure_classified": True, "other_runtime_error_propagated": True,
            "support_constants_unchanged": True, "no_rng_consumption": True,
            "historical_context_only_no_trajectory_access": True, "classified_context": row}


def evaluate_frozen(dev_a, predecessor, frozen, emulator, assets, thresholds, ids, budgets, ks, audit):
    guarded = dev_a.GuardedFrozen(frozen, audit)
    results, uses, support_rows = {}, [], []
    for policy, k in [("True-State", None), ("Point", None)] + [("UA", k) for k in ks]:
        for budget in budgets:
            threshold = thresholds[policy, k, budget]
            for year, asset in assets.items():
                before = audit.query_calls
                result, support_row = classify_cell(
                    predecessor, emulator, policy, k, budget, year, threshold,
                    lambda: dev_a.generic_rollout(predecessor, guarded, emulator, asset, policy, k,
                                                 threshold, len(ids), ids, audit))
                if policy == "True-State":
                    require(audit.query_calls == before, "True-State perception access")
                if result is not None:
                    results[policy, k, budget, year] = result
                support_rows.append(support_row)
                uses.append({"policy": policy, "k": k, "budget": budget, "year": year,
                             "theta_global": threshold, "threshold_hex": float(threshold).hex()})
    return results, uses, support_rows


def pairwise_error(left, right):
    denominator = (left + right) / 2.0
    return 0.0 if denominator == 0 else abs(left - right) / denominator


def fairness(results, budgets, ks):
    """Counts only. This must finish before any evaluation population J aggregation."""
    counts = {}
    for key, result in results.items():
        actions = result["actions"]
        count = float(actions.sum(axis=1).mean())
        require(abs(count - sum(np.count_nonzero(row) for row in actions) / len(actions)) <= 1e-12,
                "Clean-count identity")
        counts[key] = count
    rows = []
    target_pass = {}
    pair_pass = {}
    for policy, k in [("True-State", None), ("Point", None)] + [("UA", k) for k in ks]:
        for budget in budgets:
            joint = (counts[policy, k, budget, "YEAR1"] + counts[policy, k, budget, "YEAR2"]) / 2
            error = abs(joint - budget) / budget
            require(abs(error - abs(joint / budget - 1)) <= 1e-12, "Target fairness identity")
            target_pass[policy, k, budget] = error <= FAIRNESS_TOLERANCE
            for year in DEV_ENV_ROOTS:
                count = counts[policy, k, budget, year]
                point = counts["Point", None, budget, year]
                pair = pairwise_error(count, point)
                require(abs(pair - (0 if count + point == 0 else 2 * abs(count - point) / (count + point))) <= 1e-12,
                        "Pairwise fairness identity")
                pair_pass[policy, k, budget, year] = pair <= FAIRNESS_TOLERANCE
                rows.append(dict(policy=policy, k=k, budget=budget, year=year, mean_clean_count=count,
                                 joint_mean_clean_count=joint, joint_target_error=error,
                                 joint_target_pass=target_pass[policy, k, budget], pairwise_point_error=pair,
                                 pairwise_point_pass=pair_pass[policy, k, budget, year], k_selection_eligible=None))
    point_pass = all(target_pass["Point", None, b] for b in budgets)
    eligible = {k: bool(point_pass and all(target_pass["UA", k, b] for b in budgets)
                        and all(pair_pass["UA", k, b, y] for b in budgets for y in DEV_ENV_ROOTS)) for k in ks}
    for row in rows:
        if row["policy"] == "UA":
            row["k_selection_eligible"] = eligible[row["k"]]
    for k in ks:
        ua_rows = [row for row in rows if row["policy"] == "UA" and row["k"] == k]
        require(len(ua_rows) == 2 * len(budgets)
                and eligible[k] == bool(point_pass and all(
                    row["joint_target_error"] <= FAIRNESS_TOLERANCE
                    and row["pairwise_point_error"] <= FAIRNESS_TOLERANCE for row in ua_rows)),
                "Mechanical fairness eligibility identity")
    return rows, counts, point_pass, eligible


def trajectory_metrics(result, ids):
    require(result["pre"].shape == (len(ids), 365), "Metric population dimensions")
    rows = []
    for index, original_id in enumerate(ids):
        pre, post, clean = result["pre"][index], result["post"][index], result["actions"][index]
        require(np.isfinite(pre).all() and np.isfinite(post).all(), "Non-finite metric population")
        require(np.array_equal(post[clean], (1 - ETA) * pre[clean])
                and np.array_equal(post[~clean], pre[~clean]), "CLEAN metric order identity")
        j_post, j_pre = float(post.sum()), float(pre.sum())
        require(abs(j_post - math.fsum(map(float, post))) <= 1e-10
                and abs(j_pre - math.fsum(map(float, pre))) <= 1e-10
                and abs(j_pre - j_post - ETA * float(pre[clean].sum())) <= 1e-10, "J metric identity")
        longest = run = 0
        for action in clean:
            run = run + 1 if action else 0
            longest = max(longest, run)
        rows.append({"trajectory": original_id, "J_post": j_post, "J_pre": j_pre,
                     **{f"P_L_pre_gt_{t:.2f}": float(np.mean(pre > t)) for t in (0.05, 0.10, 0.15)},
                     "P95_L_pre": float(np.quantile(pre, 0.95)), "max_L_pre": float(pre.max()),
                     "clean_count": int(clean.sum()),
                     "consecutive_clean_pairs": int(np.sum(clean[:-1] & clean[1:])),
                     "max_consecutive_clean_run": longest})
    return rows


def paired_bootstrap(differences, k, budget, year):
    namespace = f"statistics-only|{BOOTSTRAP_SEED}|{float(k).hex()}|{budget}|{year}"
    cell_seed = int.from_bytes(hashlib.sha256(namespace.encode("ascii")).digest()[:8], "big")
    rng = np.random.default_rng(cell_seed)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences)))
    means = differences[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi), cell_seed


def select_k(delta_cells, eligible):
    """Only six Delta values and fairness eligibility enter selection."""
    expected_cells = {(b, y) for b in COMMON_BUDGETS for y in DEV_ENV_ROOTS}
    scores = {}
    for k in K_GRID:
        if eligible[k]:
            require(set(delta_cells[k]) == expected_cells, "Score must equally weight six cells")
            scores[k] = math.fsum(delta_cells[k].values()) / 6
    if not scores:
        return None, None, [dict(k=k, k_selection_eligible=eligible[k], Score_percent=None, selected=False)
                            for k in K_GRID]
    maximum = max(scores.values())
    chosen = min(k for k, score in scores.items() if maximum - score < TIE_PERCENTAGE_POINTS)
    return chosen, scores[chosen], [dict(k=k, k_selection_eligible=eligible[k],
                                        Score_percent=scores.get(k), selected=k == chosen) for k in K_GRID]


def dev_signal(chosen, score, delta_cells, point_pass):
    if not point_pass:
        return {"dev_signal_category": "YELLOW_DEV_SIGNAL",
                "dev_signal": "YELLOW_POINT_BUDGET_GENERALIZATION_FAIL"}
    if chosen is None:
        return {"dev_signal_category": "YELLOW_DEV_SIGNAL", "dev_signal": "YELLOW_NO_FAIR_K"}
    cells = delta_cells[chosen]
    by_year = {year: math.fsum(cells[b, year] for b in COMMON_BUDGETS) / 3 for year in DEV_ENV_ROOTS}
    conflict = by_year["YEAR1"] * by_year["YEAR2"] < 0
    positive = sum(v > 0 for v in cells.values())
    negative = sum(v < 0 for v in cells.values())
    category = "YELLOW_DEV_SIGNAL"
    if not conflict:
        if score >= 2.0 and positive >= 5 and all(v > 0 for v in by_year.values()):
            category = "STRONG_POSITIVE_DEV_SIGNAL"
        elif 0.5 <= score < 2.0 and positive >= 4:
            category = "POSITIVE_DEV_SIGNAL"
        elif score <= -0.5 and negative >= 4:
            category = "NEGATIVE_DEV_SIGNAL"
    return {"dev_signal_category": category, "dev_signal": category,
            "year_aggregate_Delta_percent": by_year, "year_direction_conflict": conflict,
            "positive_cells": positive, "negative_cells": negative}


def analyze_selection(results, ids, thresholds, counts, point_pass, eligible, point_support, k_support):
    require(tuple(ids) == SELECTION_IDS, "Only sealed selection IDs may enter performance comparison")
    expected = {(p, k, b, y) for p, k in [("True-State", None), ("Point", None)]
                + [("UA", k) for k in K_GRID if k_support[k]]
                for b in COMMON_BUDGETS for y in DEV_ENV_ROOTS} if point_support else set()
    require(set(results) == expected, "Performance requires complete support-screened populations")
    require(all(not eligible[k] or (point_support and k_support[k]) for k in K_GRID),
            "Support-ineligible k entered selection")
    metrics = {key: trajectory_metrics(result, ids) for key, result in results.items()}
    require(all(tuple(row["trajectory"] for row in rows) == SELECTION_IDS for rows in metrics.values()),
            "Performance pairing population drift")
    cells, trajectories, references = [], [], []
    delta_cells = {k: {} for k in K_GRID}
    for (policy, k, budget, year), rows in metrics.items():
        point_rows = metrics["Point", None, budget, year]
        values = np.array([row["J_post"] for row in rows])
        point_values = np.array([row["J_post"] for row in point_rows])
        cell = dict(policy=policy, k=k, budget=budget, year=year, theta_global=thresholds[policy, k, budget],
                    n_trajectories=len(ids), mean_J_post=float(values.mean()),
                    **{"mean_" + name: float(np.mean([row[name] for row in rows])) for name in SECONDARY},
                    Delta_percent=None, paired_mean_difference=None, paired_median_difference=None,
                    win_fraction=None, paired_bootstrap_ci_low=None, paired_bootstrap_ci_high=None,
                    bootstrap_cell_seed=None,
                    primary_matched_budget_eligible=bool(point_pass and (eligible[k] if policy == "UA" else policy == "Point")))
        if policy == "UA":
            require(float(point_values.mean()) > 0, "Undefined relative effect: non-positive Point mean")
            differences = values - point_values
            delta = (float(point_values.mean()) - float(values.mean())) / float(point_values.mean()) * 100
            delta_cells[k][budget, year] = delta
            # Reporting only: these paired statistics never reach select_k.
            lo, hi, cell_seed = paired_bootstrap(differences, k, budget, year)
            cell.update(Delta_percent=delta, paired_mean_difference=float(differences.mean()),
                        paired_median_difference=float(np.median(differences)), win_fraction=float(np.mean(values < point_values)),
                        paired_bootstrap_ci_low=lo, paired_bootstrap_ci_high=hi, bootstrap_cell_seed=cell_seed)
            true_mean = float(np.mean([row["J_post"] for row in metrics["True-State", None, budget, year]]))
            point_mean = float(point_values.mean())
            count_error = pairwise_error(counts["True-State", None, budget, year], counts["Point", None, budget, year])
            reasons = []
            if not true_mean < point_mean:
                reasons.append("TRUE_STATE_NOT_LOWER_THAN_POINT")
            if count_error > FAIRNESS_TOLERANCE:
                reasons.append("TRUE_STATE_POINT_CLEAN_COUNT_NOT_MATCHED")
            references.append(dict(k=k, budget=budget, year=year, mean_J_True=true_mean,
                                   mean_J_Point=point_mean, true_point_clean_count_error=count_error,
                                   GapClosure=None if reasons else (point_mean - float(values.mean())) / (point_mean - true_mean),
                                   invalid_reason=";".join(reasons) if reasons else None))
        cells.append(cell)
        for index, row in enumerate(rows):
            trajectories.append({"policy": policy, "k": k, "budget": budget, "year": year,
                                 "theta_global": thresholds[policy, k, budget], **row,
                                 "paired_point_J_post": float(point_values[index]) if policy == "UA" else None,
                                 "paired_difference": float(values[index] - point_values[index]) if policy == "UA" else None})
    chosen, score, k_rows = select_k(delta_cells, eligible)
    require(all(not delta_cells[k] for k in K_GRID if not k_support[k])
            and (point_support or all(not cells for cells in delta_cells.values())),
            "Support-ineligible k entered decision-value analysis")
    for row in k_rows:
        row["k_support_eligible"] = k_support[row["k"]]
        row["k_fairness_eligible"] = eligible[row["k"]]
    # Independent direct formula check uses only J_post-derived Delta and fairness.
    direct = {k: math.fsum(delta_cells[k].values()) / 6 for k in K_GRID if eligible[k]}
    expected = None if not direct else min(k for k in direct if max(direct.values()) - direct[k] < TIE_PERCENTAGE_POINTS)
    require(chosen == expected and (chosen is None or abs(score - direct[chosen]) <= 1e-12), "Selection rule identity")
    selection = {"k_star": chosen, "Score_k_star_percent": score,
                 "primary_matched_budget_analysis": "MATCHED_BUDGET_FAIRNESS_PASS" if point_pass
                 else "YELLOW_POINT_BUDGET_GENERALIZATION_FAIL",
                 **dev_signal(chosen, score, delta_cells, point_pass),
                 "score_equal_weight_six_cells_pass": True, "tie_rule_pass": True,
                 "bootstrap_excluded_from_selection_pass": True}
    if not point_support:
        yellow = "YELLOW_POINT_SELECTION_SUPPORT_FAIL"
    elif not any(k_support.values()):
        yellow = "YELLOW_NO_SUPPORTED_UA_K"
    elif not point_pass:
        yellow = "YELLOW_POINT_BUDGET_GENERALIZATION_FAIL"
    elif chosen is None:
        yellow = "YELLOW_NO_FAIR_K"
    else:
        yellow = None
    if yellow is not None:
        require(chosen is None, "Scientific support/fairness failure selected a k")
        selection.update(primary_matched_budget_analysis=yellow,
                         dev_signal_category="YELLOW_DEV_SIGNAL", dev_signal=yellow)
    return cells, trajectories, k_rows, references, selection


def protocol(mode):
    manifest = {"stage": STAGE, "title": TITLE, "mode": mode,
                "predecessor_dev_b_stage": "P2-1C-2C-DEV-B-v1",
                "predecessor_dev_b_status": "FAIL",
                "repair_reason": "sealed selection rollout encountered frozen perception support failure before fairness or decision-value aggregation",
                "predecessor_first_failure_context": dict(FIRST_FAILURE),
                "predecessor_observation_statement": "No J_post, Delta, Score, k_star or performance ranking was observed from DEV-B-v1 before this repair.",
                "scope": "SMOKE-ONLY engineering checks" if mode == "smoke" else "SEALED SELECTION",
                "utc_timestamp": datetime.now(timezone.utc).isoformat(),
                "dev_a_checkpoint_head": DEV_A_HEAD, "dev_a_script_blob": DEV_A_BLOB,
                "frozen_dev_a_manifest_sha256": DEV_A_MANIFEST_SHA256,
                "threshold_source": str(DEV_A_MANIFEST), "threshold_recalibration": False,
                "common_budgets": COMMON_BUDGETS, "eligible_calibration_k": K_GRID,
                "TOTAL_DEV_TRAJECTORIES": TOTAL_DEV_TRAJECTORIES, "DEV_PERCEPTION_SEED": DEV_PERCEPTION_SEED,
                "DEV_ENV_ROOTS": DEV_ENV_ROOTS,
                # Documentation only; runtime never consumes these values.
                "formal_seed_documentation": {"perception": [20260907, 20260908, 20260909, 20260910, 20260911],
                                              "environment": {"YEAR1": 1220001, "YEAR2": 1230001}},
                "evaluation_ids": list(range(10)) if mode == "smoke" else list(SELECTION_IDS),
                "evaluated_budgets": [10] if mode == "smoke" else COMMON_BUDGETS,
                "evaluated_ua_k": [1.0] if mode == "smoke" else K_GRID,
                "regression_ids": list(range(10 if mode == "smoke" else 20)),
                "daily_order": ["L_pre", "online perception (Point/UA only)", "action", "L_post",
                                "metric settlement", "natural transition", "next L"],
                "eta": ETA, "dry": "D0_GLOBAL", "rain": "R3_OLS_RESIDUAL",
                "fairness_tolerance": FAIRNESS_TOLERANCE,
                "support": "unchanged frozen sampler; no fallback or retry; unsupported cells classified, other exceptions fail closed",
                "selection_support_screening": "all six Point cells and all six cells per UA k; no budget deletion",
                "repair_regression_scope": "fixed historical L/context only; no trajectory 180 access or performance evaluation"}
    if mode == "dev-b":
        manifest["selection_protocol"] = {
            "primary_metric": "J_post = sum(L_post)", "score": "equally weighted mean of six Delta percent cells",
            "tie": "smallest eligible k with maximum Score minus its Score strictly less than 0.25 percentage point",
            "year_aggregate": "equally weighted mean of three budget Delta cells",
            "year_direction_conflict": "strictly opposite signs; conservatively also defines clearly opposing directions",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_statistical_only_seed": BOOTSTRAP_SEED,
            "bootstrap_ci": "percentile 95% interval of paired mean difference, linear quantile",
            "no_fair_k": "dev_signal=YELLOW_NO_FAIR_K; broad category=YELLOW_DEV_SIGNAL"}
    return manifest


def dump(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    require(json.loads(path.read_text(encoding="utf-8")) == value, "JSON readback mismatch")


def write_csv(path, rows, columns):
    require(all(set(row) == set(columns) for row in rows), "CSV schema mismatch")
    with path.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows, columns=columns).to_csv(stream, index=False)
    readback = pd.read_csv(path)
    require(list(readback.columns) == columns and len(readback) == len(rows), "CSV readback mismatch")


def finalize(out, mode, manifest, summary, csv_outputs, selection_manifest, expected_complete):
    """Validate every scientific output before publishing the final audit."""
    dump(out / "protocol_manifest.json", manifest)
    required = ["protocol_manifest.json"]
    for name, (rows, columns) in csv_outputs.items():
        write_csv(out / name, rows, columns)
        required.append(name)
    audit_name = "smoke_audit_summary.json" if mode == "smoke" else "dev_b_audit_summary.json"
    final = json.loads(json.dumps(summary, allow_nan=False))
    final["gates"][GATES[13]] = bool(expected_complete and all((out / name).is_file() for name in required))
    final["stage_pass"] = bool(all(final["gates"].values()) and not final["errors"])
    final["numbered_gates"] = {f"G{i}": {"name": name, "pass": final["gates"][name]}
                               for i, name in enumerate(GATES, 1)}
    audit_bytes = json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    hashes = {name: sha256_file(out / name) for name in required}
    hashes[audit_name] = hashlib.sha256(audit_bytes).hexdigest()
    if mode == "dev-b":
        sealed = {**selection_manifest, "output_file_sha256": hashes, "final_stage_status_source": audit_name,
                  "hash_scope": "All other output files; this manifest cannot hash itself."}
        dump(out / "dev_b_selection_manifest.json", sealed)
        required.append("dev_b_selection_manifest.json")
    pending = out / (audit_name + ".pending")
    with pending.open("xb") as stream:
        stream.write(audit_bytes)
    require(json.loads(pending.read_text(encoding="utf-8")) == final, "Pending audit readback")
    require(all(sha256_file(pending if name == audit_name else out / name) == digest
                for name, digest in hashes.items()), "Output SHA256 verification")
    require(all((out / name).is_file() for name in required), "Missing scientific output")
    require(not (out / audit_name).exists(), "Refusing final audit overwrite")
    pending.rename(out / audit_name)
    require(all((out / name).is_file() for name in required + [audit_name]), "Output completeness")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "dev-b"], required=True)
    args = parser.parse_args()
    out = OUT / ("smoke" if args.mode == "smoke" else "dev_b")
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Refusing existing output directory: {out}", file=sys.stderr)
        return 2
    manifest = protocol(args.mode)
    manifest["script_sha256"] = sha256_file(Path(__file__))
    gates = dict.fromkeys(GATES, False)
    summary = {"stage": STAGE, "mode": args.mode, "gates": gates, "stage_pass": False, "errors": []}
    regression_audit = PhaseAudit("REGRESSION_ONLY", range(10 if args.mode == "smoke" else 20))
    ids = tuple(range(10)) if args.mode == "smoke" else SELECTION_IDS
    evaluation_audit = PhaseAudit("SMOKE_ONLY" if args.mode == "smoke" else "SELECTION_PERFORMANCE", ids)
    budgets = [10] if args.mode == "smoke" else COMMON_BUDGETS
    ks = [1.0] if args.mode == "smoke" else K_GRID
    fair_rows, cells, trajectories, k_rows, references = [], [], [], [], []
    support_rows = []
    point_support = False
    k_support = dict.fromkeys(ks, False)
    selection = {}
    selection_manifest = {"stage": STAGE, "frozen_dev_a_manifest_sha256": DEV_A_MANIFEST_SHA256,
                          "dev_a_script_blob": DEV_A_BLOB, "common_budgets": COMMON_BUDGETS,
                          "eligible_calibration_k": K_GRID, "selection_status": "AUDIT_FAIL"}
    try:
        dev_a, calibration, frozen_protocol = load_dev_a(gates)
        thresholds = frozen_thresholds(calibration)
        predecessor, frozen, ledger, dry, r3, emulator = load_models(dev_a, calibration)
        assets, crn = build_assets(frozen, ledger, dry, r3, frozen_protocol, regression_audit)
        summary["environment_crn"] = crn
        gates[GATES[4]] = all(row["exact_match"] for row in crn.values())
        regression_result = dev_a.regression(predecessor, frozen, dev_a.GuardedFrozen(frozen, regression_audit),
                                             emulator, assets, regression_audit, args.mode)
        summary["generic_regression"] = regression_result
        gates[GATES[5]] = all(row["passed"] for row in regression_result["checks"])
        gates[GATES[9]] = regression_result["perception_crn_pass"]
        require(all(gates[name] for name in GATES[:6]), "Pre-evaluation gates failed")
        require(thresholds["UA", FIRST_FAILURE["k"], FIRST_FAILURE["budget"]] == FIRST_FAILURE["threshold"],
                "Historical failure threshold provenance mismatch")
        summary["repair_regression"] = repair_regression(predecessor, frozen, emulator)
        support_parameters = {name: getattr(frozen, name) for name in
                              ("LOCAL_RADIUS", "MIN_LOCAL_SAMPLES", "MIN_LOCAL_DATES", "KERNEL_BANDWIDTH", "BLOCK_MINUTES")}
        manifest["frozen_support_parameters"] = support_parameters
        # Metric identities are independently checked on the allowed regression
        # prefix even if no selection population survives support screening.
        regression_ids = tuple(sorted(regression_audit.allowed))
        for asset in assets.values():
            reference = dev_a.generic_rollout(
                predecessor, dev_a.GuardedFrozen(frozen, regression_audit), emulator, asset,
                "True-State", None, thresholds["True-State", None, 10],
                len(regression_ids), regression_ids, regression_audit)
            trajectory_metrics(reference, regression_ids)
        del reference
        summary["regression_metric_identity_pass"] = True
        results, uses, support_rows = evaluate_frozen(dev_a, predecessor, frozen, emulator, assets, thresholds,
                                                     ids, budgets, ks, evaluation_audit)
        manifest["frozen_threshold_uses"] = uses
        gates[GATES[8]] = all(use["threshold_hex"] == float(thresholds[use["policy"], use["k"], use["budget"]]).hex()
                              for use in uses)
        point_support, k_support, results = support_screening(support_rows, results, budgets, ks)
        support_details = {
            "point_selection_support_pass": point_support,
            "k_support_eligible": [{"k": k, "eligible": k_support[k]} for k in ks],
            "support_failure_count": sum(row["selection_support_status"] == UNSUPPORTED for row in support_rows),
            "support_failure_contexts": [row for row in support_rows if row["selection_support_status"] == UNSUPPORTED]}
        summary["selection_support"] = support_details
        selection_manifest.update(support_details)
        # Fairness sees only complete supported populations, and precedes J.
        eligible = dict.fromkeys(ks, False)
        if point_support:
            supported_ks = [k for k in ks if k_support[k]]
            fair_rows, counts, point_pass, fair_eligible = fairness(results, budgets, supported_ks)
            eligible.update(fair_eligible)
        else:
            counts, point_pass = {}, False
            summary["fairness_not_evaluated_reason"] = "YELLOW_POINT_SELECTION_SUPPORT_FAIL"
        selection_manifest["fairness_eligibility"] = [
            {"k": k, "k_fairness_eligible": eligible[k],
             "status": "EVALUATED" if point_support and k_support[k] else "NOT_EVALUATED_SUPPORT_SCREENING"}
            for k in ks]
        gates[GATES[10]] = True
        if args.mode == "smoke":
            for result in results.values():
                trajectory_metrics(result, ids)  # Identity checks only; immediately discard.
            # Smoke never emits eligibility, performance comparisons or selection.
            for row in fair_rows:
                row["k_selection_eligible"] = None
        else:
            cells, trajectories, k_rows, references, selection = analyze_selection(
                results, ids, thresholds, counts, point_pass, eligible, point_support, k_support)
            selection_manifest.update(selection_status="EVALUATED", selection_fairness_results=fair_rows,
                                      k_selection_eligible=k_rows, **selection,
                                      threshold_source_provenance={"manifest": str(DEV_A_MANIFEST),
                                          "sha256": DEV_A_MANIFEST_SHA256, "threshold_uses": uses},
                                      bootstrap_statistical_only_seed=BOOTSTRAP_SEED,
                                      bootstrap_resamples=BOOTSTRAP_RESAMPLES)
            summary["selection_implementation_checks"] = {
                "only_selection_ids_in_performance": all(row["trajectory"] in SELECTION_IDS for row in trajectories),
                "fairness_precedes_performance": True, "mechanical_eligibility": True,
                "six_equal_weight_cells": selection["score_equal_weight_six_cells_pass"],
                "exact_tie_rule": selection["tie_rule_pass"],
                "bootstrap_excluded_from_selection": selection["bootstrap_excluded_from_selection_pass"]}
            require(all(summary["selection_implementation_checks"].values()), "Selection implementation gate")
        gates[GATES[11]] = True
        gates[GATES[12]] = (len(uses) == len(support_rows) == (2 + len(ks)) * len(budgets) * 2
                            and gates[GATES[8]])
        gates[GATES[14]] = bool(
            all(summary["repair_regression"][name] for name in
                ("known_failure_classified", "other_runtime_error_propagated", "support_constants_unchanged"))
            and support_parameters == {name: getattr(frozen, name) for name in support_parameters}
            and all(key[0] != "UA" or k_support[key[1]] for key in results)
            and all(row["Score_percent"] is None for row in k_rows if not row["k_support_eligible"])
            and all(row["k_selection_eligible"] == (row["k_support_eligible"] and row["k_fairness_eligible"])
                    for row in k_rows))
    except Exception as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "context"):
            summary["unsupported_context"] = exc.context
    gates[GATES[6]] = (regression_audit.accessed <= set(range(10 if args.mode == "smoke" else 20))
                       and evaluation_audit.accessed == set(ids)
                       and not regression_audit.rejected and not evaluation_audit.rejected
                       and (args.mode == "smoke" or regression_audit.accessed.isdisjoint(evaluation_audit.accessed)))
    gates[GATES[7]] = (regression_audit.perception_seeds == {DEV_PERCEPTION_SEED}
                       and evaluation_audit.perception_seeds == {DEV_PERCEPTION_SEED}
                       and regression_audit.environment_seeds == set(DEV_ENV_ROOTS.values())
                       and evaluation_audit.environment_seeds <= set(DEV_ENV_ROOTS.values()))
    summary["access_audit"] = {"regression": regression_audit.report(), "evaluation": evaluation_audit.report()}
    expected_cells = (2 + sum(k_support.values())) * len(budgets) * 2 if point_support else 0
    if args.mode == "smoke":
        # Explicit smoke schema omits all selection/performance fields.
        smoke_columns = [name for name in FAIR_COLUMNS if name != "k_selection_eligible"]
        smoke_rows = [{key: row[key] for key in smoke_columns} for row in fair_rows]
        outputs = {"smoke_fairness_summary.csv": (smoke_rows, smoke_columns),
                   "selection_support_summary.csv": (support_rows, SUPPORT_COLUMNS)}
        complete = len(smoke_rows) == expected_cells and len(support_rows) == 6
    else:
        outputs = {"fairness_summary.csv": (fair_rows, FAIR_COLUMNS), "policy_cell_metrics.csv": (cells, CELL_COLUMNS),
                   "paired_trajectory_metrics.csv": (trajectories, TRAJECTORY_COLUMNS),
                   "k_selection_summary.csv": (k_rows, K_COLUMNS),
                   "true_state_reference_summary.csv": (references, REFERENCE_COLUMNS),
                   "selection_support_summary.csv": (support_rows, SUPPORT_COLUMNS)}
        complete = (len(support_rows) == 48 and len(fair_rows) == expected_cells
                    and len(cells) == expected_cells and len(trajectories) == expected_cells * len(SELECTION_IDS)
                    and len(k_rows) == 6 and len(references) == (sum(k_support.values()) * 6 if point_support else 0))
    try:
        final = finalize(out, args.mode, manifest, summary, outputs, selection_manifest, complete)
    except Exception as exc:
        print(f"Output failure: {exc}", file=sys.stderr)
        return 2
    print(f"{STAGE} stage_pass={final['stage_pass']}")
    return 0 if final["stage_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
