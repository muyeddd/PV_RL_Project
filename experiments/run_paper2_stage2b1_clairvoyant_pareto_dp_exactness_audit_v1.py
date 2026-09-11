#!/usr/bin/env python
"""P2-2B-1-v1: development-only solver exactness and contract audit.

Both modes are development audits. Held-out artifacts are hashed for provenance
only; no held-out cost table is parsed or passed to a solver. Explicit execution
after review/checkpoint only. No import-time scientific construction or writes.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import io
import itertools
import json
import csv
import math
from pathlib import Path
import subprocess
import sys


STAGE = "P2-2B-1-v1"
TITLE = "Clairvoyant Pareto-DP Solver Exactness and Contract Audit"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUTPUT = BASE / "p2_2b_1_clairvoyant_pareto_dp_exactness_audit_v1"
PRE = BASE / "p2_2a_2_heldout_nonrl_baseline_evaluation_v1/formal"
SELF = "experiments/run_paper2_stage2b1_clairvoyant_pareto_dp_exactness_audit_v1.py"
SOLVER = "experiments/paper2_clairvoyant_pareto_dp_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
POLICY = "experiments/run_paper2_stage1d2b_reward_behaviour_audit_v1.py"
PRE_SCRIPT = "experiments/run_paper2_stage2a2_heldout_nonrl_baseline_evaluation_v1.py"
CHECKPOINT = "7ed0fe3826a4bc6a9aeaf47e8551d329388c6c61"
PINS = {PRE_SCRIPT: "a5d0bcd2091b46be9cb55968fd24bd0d9efb5c4b",
        CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
        POLICY: "f4400a94f5df4c46d975a3944182f084206ef37a"}
PRE_HASHES = {
    "audit_summary.json": "b6899bd680e3c14e37af7a6f8b15ea4c3df9c3626a4a829b3aa39df4f9d71725",
    "protocol_manifest.json": "d40efcd74094a48d6721638780d4ff989f6065d0138765271e6d27c9c54f5e26",
    "policy_episode_metrics.csv": "6cb9b45b427f42edd15c27fbde697c7786062a15b21bfd324814257ba1a85776",
    "policy_aggregate_summary.csv": "377f88aba16967fc3f7e05c1344deebfc52909a57ff4bec7a442cb68f18305e9",
    "paired_comparison_summary.csv": "6eb39f7b5cda715dc14b66ec4006efe313a78f82bb09491fb670e4cfd33ab4ca",
    "representative_step_trace.csv": "b01f355cb6bf57ec641fa85769019ff96d9de02084debdcc4bff57796c44256c",
    "source_artifact_hashes.json": "3af5a5b3e73798ec07a5a6dee5939a11ff4a15c2b020eb3162cda40b5df2bbcf",
    "output_hashes.json": "e10de3ccb9ec4a8dcb91720a29b9bf7d3af12b50ea442fe70c9ac55770b4d9f8",
}
DEVELOPMENT_ROOTS = {"YEAR1": 1020001, "YEAR2": 1030001}
BASELINES = ("NEVER_CLEAN", "DAILY_CLEAN", "PERIODIC_14", "TRUE_THRESHOLD_0.05")
SMOKE_STATUS = "SMOKE_ONLY_NO_SOLVER_FREEZE"
FORMAL_STATUS = "CLAIRVOYANT_PARETO_DP_SOLVER_EXACTNESS_PASS_FROZEN"
DECLARATION = "SOLVER EXACTNESS ONLY; NO HELD-OUT ORACLE PERFORMANCE CLAIM"
GATES = dict(enumerate([
    "predecessor_provenance", "predecessor_formal_frozen", "core_blob", "reward_contract",
    "development_roots_only", "no_heldout_access", "no_perception", "no_training",
    "solver_no_import_time_experiment_write", "solver_no_approximation", "eta_lambda_energy_conditions",
    "actual_R3_positive_slope", "dry_monotonicity", "rain_monotonicity", "exact_dominance",
    "exhaustive_population", "exhaustive_objective", "rain_window", "dry_window_or_reason",
    "365_settlements", "364_transitions", "365_traceback_actions", "annual_replay", "daily_replay",
    "indices_exact", "full_bank_exact", "oracle_le_never", "oracle_le_daily", "oracle_le_periodic",
    "oracle_le_threshold", "frontier_cap", "no_approximate_fallback", "finite_outputs",
    "sources_unchanged", "HEAD_unchanged", "both_scripts_committed", "exclusive_output_completeness",
    "status_declaration",
], 1))


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def gate(audit, number, condition, detail):
    audit["gates"][f"G{number}"] = bool(condition)
    audit["gate_details"][f"G{number}"] = detail
    require(condition, f"G{number} {GATES[number]}: {detail}")


def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
                          check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def script_record(path, head, expected=None):
    blob = git("rev-parse", "--verify", f"{head}:{path}")
    require(blob == git("hash-object", f"--path={path}", path), f"Working tree differs: {path}")
    require(expected is None or blob == expected, f"Frozen blob mismatch: {path}")
    return {"git_blob": blob, "sha256": sha((ROOT / path).read_bytes())}


def pinned(path, digest, sources):
    payload = path.read_bytes()
    require(sha(payload) == digest, f"Frozen SHA256 mismatch: {path}")
    sources[str(path)] = {"sha256": digest}
    return payload


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    require(spec is not None and spec.loader is not None, "Explicit module path")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(value)
    finally:
        sys.dont_write_bytecode = previous
    return value


def static_audit():
    """AST inspection only; safe to extract and run before any Git checkpoint."""
    solver = ast.parse((ROOT / SOLVER).read_text(encoding="utf-8"))
    harness = ast.parse((ROOT / SELF).read_text(encoding="utf-8"))
    imports = [n.module for n in ast.walk(solver) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(solver) if isinstance(n, ast.Import) for a in n.names]
    require(set(imports) <= {"__future__", "dataclasses", "math", "typing"}, "Solver pure standard-library imports")
    for node in solver.body:
        require(isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.ClassDef, ast.FunctionDef)), "Solver module-level statements")
        if isinstance(node, ast.Expr):
            require(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), "Only docstring expression at module level")
        if isinstance(node, ast.Assign):
            require(isinstance(node.value, ast.Constant) and node.value.value == 1_000_000, "Only protective cap global")
    forbidden = {"open", "write", "write_bytes", "write_text", "round", "isclose", "allclose", "linspace",
                 "arange", "interp", "quantile", "default_rng", "random", "PPO", "learn", "backward", "optimizer"}
    calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(solver)
             if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    require(not calls.intersection(forbidden), "Solver no I/O, RNG, approximation or training")
    prune = next(n for n in solver.body if isinstance(n, ast.FunctionDef) and n.name == "pareto_prune")
    require(not any(isinstance(n, ast.Slice) for n in ast.walk(prune)), "No truncated frontier")
    require(all(n.value >= 1 for n in ast.walk(prune) if isinstance(n, ast.Constant) and isinstance(n.value, float)), "No epsilon in pruning")
    require(any(isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.Lt)
                and ast.unparse(n) == "candidate.cumulative_cost < best_cost" for n in ast.walk(prune)), "Exact frontier sweep")
    roots = next(n for n in harness.body if isinstance(n, ast.Assign)
                 and any(isinstance(x, ast.Name) and x.id == "DEVELOPMENT_ROOTS" for x in n.targets))
    require(ast.literal_eval(roots.value) == {"YEAR1": 1020001, "YEAR2": 1030001}, "Only development roots")
    spec_calls = [n for n in ast.walk(harness) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "EpisodeSpec"]
    require(len(spec_calls) == 1, "Single guarded EpisodeSpec provider")
    kw = {k.arg: ast.unparse(k.value) for k in spec_calls[0].keywords}
    require(kw == {"year": "year", "environment_root": "DEVELOPMENT_ROOTS[year]", "trajectory_id": "tid",
                   "population_size": "300", "perception_seed": "None"}, "No scientific root/seed overrides")
    prohibited = {"PPO", "learn", "backward", "optimizer", "sample_one", "OnlineBlock10Emulator",
                  "ensure_perception", "perception_record", "default_rng", "build_crn_indices"}
    harness_calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(harness)
                     if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    require(not harness_calls.intersection(prohibited), "No training/perception/new CRN implementation")
    return {"solver_imports": imports, "exact_pruning": True, "approximation_calls": [],
            "single_development_spec_provider": True, "perception_seed": None,
            "heldout_artifact_use": "RAW_SHA256_ONLY; only audit/hash index JSON parsed"}


def load_inputs(audit, protocol, sources):
    head = git("rev-parse", "HEAD")
    require(git("rev-parse", "--verify", f"{CHECKPOINT}^{{commit}}") == CHECKPOINT, "Checkpoint exists")
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    records = {path: script_record(path, head, digest) for path, digest in PINS.items()}
    require(all(git("rev-parse", f"{CHECKPOINT}:{p}") == digest for p, digest in PINS.items()), "Historical blobs exact")
    own = {p: script_record(p, head) for p in (SELF, SOLVER)}
    gate(audit, 36, True, {"committed_unchanged": own, "required_in_both_modes": True})
    # All CSVs (including held-out episode costs) remain uninterpreted bytes.
    prior_audit = prior_index = None
    for name, digest in PRE_HASHES.items():
        payload = pinned(PRE / name, digest, sources)
        if name == "audit_summary.json":
            prior_audit = json.loads(payload)
        elif name == "output_hashes.json":
            prior_index = json.loads(payload)
    require(set(prior_index["hashes"]) == set(PRE_HASHES) - {"audit_summary.json", "output_hashes.json"}
            and all(PRE_HASHES[n] == d for n, d in prior_index["hashes"].items()), "Predecessor hash index")
    gate(audit, 1, True, {"checkpoint": CHECKPOINT, "ancestor": True, "hashes": PRE_HASHES})
    gate(audit, 2, prior_audit.get("stage_pass") is True and prior_audit.get("failed_gates") == []
         and prior_audit.get("scientific_status") == "HELDOUT_NONRL_BASELINE_EVALUATION_PASS_FROZEN"
         and prior_audit.get("declaration") == "NO BASELINE RESELECTION FROM HELD-OUT RESULTS", "Frozen formal status/declaration exact")
    gate(audit, 3, records[CORE]["git_blob"] == PINS[CORE], records[CORE])
    core = module("_p2_2b1_core", CORE)
    solver = module("_p2_2b1_solver", SOLVER)
    policy = module("_p2_2b1_policy", POLICY)
    for path, digest in core.ASSET_HASHES.values():
        pinned(path, digest, sources)
    for name, digest in core.REWARD_HASHES.items():
        pinned(core.REWARD_DIRECTORY / name, digest, sources)
    for path, blob in [*core.HELPERS.values(), (core.REWARD_SCRIPT, core.REWARD_BLOB)]:
        records[path] = script_record(path, head, blob)
    assets = core.Paper2EnvAssets()
    gate(audit, 4, assets.eta == core.ETA == .95 and assets.lambda_main == core.LAMBDA_MAIN == .19925428989934618
         and assets.reward_scale == core.REWARD_SCALE == 1., assets.reward_contract)
    protocol.update(current_HEAD=head, script_provenance=records, current_files=own,
                    source_environment_provenance=assets.provenance,
                    policy_semantics={"file": POLICY, "git_blob": PINS[POLICY], "function": "action",
                                      "periodic_rule": "(day_index+1)%14 == 0; first day_index=13",
                                      "threshold_rule": "pre-action float64 L_pre >= 0.05"})
    return core, solver, assets, policy.action


def development_spec(core, year, tid):
    require(year in DEVELOPMENT_ROOTS and type(tid) is int and tid in (0, 1, 2), "Only declared development cases")
    return core.EpisodeSpec(year=year, environment_root=DEVELOPMENT_ROOTS[year], trajectory_id=tid,
                            population_size=300, perception_seed=None)


class Adapter:
    """Read-only audit channel, never an agent observation or ordinary info."""

    def __init__(self, core, assets, year, tid):
        self.core, self.assets = core, assets
        self.year, self.tid = year, tid
        self.spec = development_spec(core, year, tid)
        env = core.Paper2CleaningEnv(assets, self.spec, observation_mode="True-State")
        try:
            env.reset()
            self.identity = env._audit_snapshot()
            self.initial = float(env._initial)
            self.dates = tuple(env._calendar.date.dt.strftime("%Y-%m-%d"))
            self.energy = env._energy
            self.rain = env._rain
            self.indices = env._indices
            self.check_identity()
        finally:
            env.close()

    def check_identity(self):
        require(sha(self.indices.tobytes()) == self.identity["indices_sha256"]
                and not self.indices.flags.writeable, "Read-only exact trajectory CRN")
        bank = self.assets._banks[(self.year, self.spec.environment_root, 300)][2]
        require(sha(bank.tobytes()) == self.identity["full_bank_sha256"] and not bank.flags.writeable, "Read-only full bank unchanged")
        require(self.identity["perception_query_count"] == 0 and self.spec.perception_seed is None, "No perception")

    def components(self, day, pre, action):
        post = (1 - self.assets.eta) * pre if action else pre
        soiling, cleaning, total, raw, final = self.assets.reward_helper.settlement(
            float(self.energy[day]), post, action, self.assets.lambda_main)
        require(all(math.isfinite(float(x)) for x in (post, soiling, cleaning, total, raw, final)), "Finite frozen settlement")
        return float(post), float(soiling), float(cleaning), float(total)

    def settlement(self, day, pre, action):
        post, _, _, total = self.components(day, pre, action)
        return post, total

    def transition(self, day, post):
        require(0 <= day < 364, "No terminal natural transition")
        index = int(self.indices[day])
        if self.rain[day]:
            r3 = self.assets.r3
            model = {"candidate": "R3_OLS_RESIDUAL", "intercept": r3["intercept"], "slope": r3["slope"],
                     "residuals": r3["residuals"][index:index + 1]}
            following = self.assets.natural.rain_next_samples(post, model)[0]
        else:
            following = self.assets.natural.dry_next_samples(post, self.assets.dry[index:index + 1])[0]
        require(math.isfinite(float(following)) and 0 <= following <= 1, "Frozen projected next state")
        return float(following)


def monotonicity(adapters, assets, audit, protocol):
    import numpy as np
    gate(audit, 11, 0 < assets.eta <= 1 and assets.lambda_main >= 0
         and all(np.isfinite(a.energy).all() and (a.energy > 0).all() for a in adapters), "eta/lambda/positive finite realized energy")
    beta = float(assets.r3["slope"])
    gate(audit, 12, math.isfinite(beta) and 1 + beta > 0, {"actual_R3_slope": beta, "one_plus_slope": 1 + beta})
    # Probe resolution is only an implementation check, never a solver input.
    probe_states = np.linspace(0., 1., 1001)
    records = []
    for adapter in adapters:
        for rain in (False, True):
            days = [day for day in range(364) if bool(adapter.rain[day]) == rain]
            require(days, "Both frozen transition types must be available")
            chosen = sorted({days[0], days[len(days) // 2], days[-1]})
            for day in chosen:
                values = [adapter.transition(day, float(state)) for state in probe_states]
                passed = all(left <= right for left, right in zip(values, values[1:]))
                require(passed, f"Nonmonotone frozen transition: {adapter.year}/{adapter.tid}/{day}")
                records.append({"year": adapter.year, "trajectory_id": adapter.tid, "day_index": day,
                                "rain": rain, "innovation_index": int(adapter.indices[day]), "pass": passed})
    gate(audit, 13, all(r["pass"] for r in records if not r["rain"]), "1001 dense states; first/middle/last dry days per trajectory")
    gate(audit, 14, all(r["pass"] for r in records if r["rain"]), "1001 dense states; first/middle/last rain days per trajectory")
    protocol.update(analytic_monotonicity_conditions_pass=True, numeric_transition_monotonicity_probe_pass=True,
                    actual_R3_slope=beta, numeric_monotonicity_probes=records)


def short_windows(adapters, protocol):
    cases = []
    coverage = []
    for adapter in adapters:
        if adapter.tid != 0:
            continue
        counts = {start: sum(bool(x) for x in adapter.rain[start:start + 11]) for start in range(354)}
        rain = [start for start, count in counts.items() if count > 0]
        dry = [start for start, count in counts.items() if count == 0]
        require(rain, "Rain-containing 12-day case must exist")
        selected = sorted({0, rain[0]} | ({dry[0]} if dry else set()))
        coverage.append({"year": adapter.year, "trajectory_id": 0, "first_rain_start": rain[0],
                         "first_dry_start": dry[0] if dry else None,
                         "dry_status": "AVAILABLE" if dry else "NOT_AVAILABLE_WITH_REASON: no 11 consecutive dry transitions in 354 eligible calendar windows"})
        for start in selected:
            cases.append((adapter, start, adapter.initial if start == 0 else .08, counts[start]))
    protocol["short_window_coverage"] = coverage
    return cases


def brute_force(adapter, start, initial, horizon):
    """Independent complete day-by-day loop, no DP/pruning/shared rollout."""
    best_cost = math.inf
    best_actions = None
    optimum_count = count = 0
    for actions in itertools.product((0, 1), repeat=horizon):
        state, cost = initial, 0.0
        for offset, action in enumerate(actions):
            post, step_cost = adapter.settlement(start + offset, state, action)
            cost += step_cost
            if offset < horizon - 1:
                state = adapter.transition(start + offset, post)
        count += 1
        require(math.isfinite(cost), "Finite exhaustive objective")
        if cost < best_cost:
            best_cost, best_actions, optimum_count = cost, actions, 1
        elif cost == best_cost:
            optimum_count += 1
    return best_cost, best_actions, optimum_count, count


def short_exactness(mode, cases, solver, audit):
    rows = []
    for adapter, start, initial, rain_count in cases:
        solution = solver.solve_pareto_dp(initial, 12,
            lambda day, pre, action: adapter.settlement(start + day, pre, action),
            lambda day, post: adapter.transition(start + day, post))
        best, best_actions, optimum_count, count = brute_force(adapter, start, initial, 12)
        # Independently evaluate the recovered DP sequence, without re-solving.
        state, sequence_cost = initial, 0.0
        for offset, action in enumerate(solution.actions):
            post, cost = adapter.settlement(start + offset, state, action)
            sequence_cost += cost
            if offset < 11:
                state = adapter.transition(start + offset, post)
        passed = len(solution.actions) == 12 and close(solution.total_cost, best) and close(sequence_cost, best)
        require(passed and count == 4096, "Short exhaustive objective/traceback exactness")
        exact_multiple_optima = optimum_count > 1
        sequence_differs = solution.actions != best_actions
        objective_equivalent_within_tolerance = sequence_differs and close(solution.total_cost, best)
        # The existing objective acceptance above guarantees tolerance equivalence
        # for differing sequences; it does not establish exact multiple optima.
        sequence_status = ("SAME_OPTIMAL_SEQUENCE" if not sequence_differs else
                           "DIFFERENT_EXACT_OPTIMAL_SEQUENCE" if exact_multiple_optima else
                           "DIFFERENT_SEQUENCE_OBJECTIVE_EQUIVALENT_WITHIN_TOLERANCE")
        rows.append({"mode": mode, "case_id": f"{adapter.year}_t0_d{start}_H12", "year": adapter.year,
                     "trajectory_id": 0, "start_day": start, "end_day": start + 11, "horizon": 12,
                     "initial_state": initial, "rain_transition_count": rain_count, "enumerated_sequences": count,
                     "dp_cost": solution.total_cost, "bruteforce_min_cost": best, "abs_diff": abs(solution.total_cost - best),
                     "dp_sequence_recomputed_cost": sequence_cost, "dp_clean_count": solution.clean_count,
                     "bruteforce_optimal_sequence_count": optimum_count, "multiple_optima": exact_multiple_optima,
                     "sequence_differs": sequence_differs,
                     "objective_equivalent_within_tolerance": objective_equivalent_within_tolerance,
                     "sequence_status": sequence_status,
                     "optimal_sequence_count_definition": "EXACT_COST_EQUALITY",
                     "dp_tie_count": solution.tie_count, "pass": passed})
    gate(audit, 16, len(rows) >= 2 and all(r["enumerated_sequences"] == 4096 for r in rows), "All 2^12 sequences, no exhaustive pruning")
    gate(audit, 17, all(r["pass"] for r in rows), "atol=1e-12 rtol=0; recovered action sequence independently evaluated")
    gate(audit, 18, {r["year"] for r in rows} == set(DEVELOPMENT_ROOTS)
         and any(r["rain_transition_count"] > 0 for r in rows) and any(r["start_day"] == 0 for r in rows), "Both years; day0 and rain windows")
    return rows


def gym_episode(adapter, actions=None, policy=None, action_rule=None):
    """Fresh frozen Gym replay; feedback baselines use their own current state."""
    env = adapter.core.Paper2CleaningEnv(adapter.assets, development_spec(adapter.core, adapter.year, adapter.tid),
                                       observation_mode="True-State")
    rows = []
    try:
        env.reset()
        for day in range(365):
            before = env._audit_snapshot()
            action = actions[day] if actions is not None else int(action_rule(policy, day, before["L_pre"]))
            _, reward, terminal, truncated, info = env.step(action)
            snapshot = env._audit_snapshot()
            row = snapshot["last_reward_components"]
            require(not truncated and terminal == (day == 364) and row["day_index"] == day
                    and row["date"] == adapter.dates[day] and row["action"] == action, "Gym calendar/action/terminal exact")
            require(row["transition_executed"] == (day < 364) and reward == -row["total_cost"], "Frozen reward/transition semantics")
            require(snapshot["perception_query_count"] == 0, "No Gym perception query")
            rows.append(row)
        require(snapshot["transition_count"] == 364 and len(rows) == 365, "Full Gym horizon")
        require(snapshot["clean_count"] == sum(r["action"] for r in rows) == info["clean_count"], "Gym clean count")
        for name in ("environment_root", "trajectory_id", "indices_sha256", "full_bank_sha256"):
            require(snapshot[name] == adapter.identity[name], "Exact extraction/replay CRN identity")
        adapter.check_identity()
        total = sum(r["total_cost"] for r in rows)
        require(close(total, -snapshot["episode_return"]), "Annual Gym cost/return regression tolerance")
        return total, snapshot, rows
    finally:
        env.close()


def full_exactness(adapters, solver, action_rule, audit):
    full, frontier, sanity, traces = [], [], [], []
    for adapter in adapters:
        solution = solver.solve_pareto_dp(adapter.initial, 365, adapter.settlement, adapter.transition)
        require(len(solution.actions) == len(solution.trace) == len(solution.statistics) == 365, "Full solver horizon/traceback")
        gym_cost, snapshot, gym_trace = gym_episode(adapter, actions=solution.actions)
        differences = {name: 0.0 for name in ("L_pre", "L_post", "stage_cost", "L_next")}
        for dp, gym_row, stats in zip(solution.trace, gym_trace, solution.statistics):
            require(dp.action == gym_row["action"] and dp.day_index == gym_row["day_index"], "Traceback categorical exactness")
            for name, actual in (("L_pre", gym_row["L_pre"]), ("L_post", gym_row["L_post"]), ("stage_cost", gym_row["total_cost"])):
                differences[name] = max(differences[name], abs(getattr(dp, name) - actual))
            if dp.L_next is None:
                require(gym_row["L_next"] is None and dp.day_index == 364, "Terminal has no next state")
            else:
                differences["L_next"] = max(differences["L_next"], abs(dp.L_next - gym_row["L_next"]))
            post, soiling, cleaning, cost = adapter.components(dp.day_index, dp.L_pre, dp.action)
            require(close(post, dp.L_post) and close(cost, dp.stage_cost), "Stored DP settlement trace")
            traces.append({"year": adapter.year, "trajectory_id": adapter.tid, "day_index": dp.day_index,
                           "date": adapter.dates[dp.day_index], "L_pre": dp.L_pre, "action": dp.action, "L_post": dp.L_post,
                           "E_clean_GHI": float(adapter.energy[dp.day_index]), "stage_soiling_cost": soiling,
                           "stage_cleaning_cost": cleaning, "stage_total_cost": dp.stage_cost, "L_next": dp.L_next,
                           "cumulative_cost": dp.cumulative_cost, "frontier_size_after_prune": stats.frontier_out})
            frontier.append({"year": adapter.year, "trajectory_id": adapter.tid, **asdict(stats)})
        replay_pass = close(solution.total_cost, gym_cost) and max(differences.values()) <= 1e-12
        require(replay_pass and solution.clean_count == snapshot["clean_count"], "Full DP/Gym replay")
        full.append({"year": adapter.year, "trajectory_id": adapter.tid, "environment_root": adapter.spec.environment_root,
                     "dp_J_total": solution.total_cost, "gym_J_total": gym_cost, "abs_J_diff": abs(solution.total_cost - gym_cost),
                     "dp_clean_count": solution.clean_count, "gym_clean_count": snapshot["clean_count"],
                     **{f"max_abs_{k}_diff": v for k, v in differences.items()},
                     "indices_sha256": adapter.identity["indices_sha256"], "gym_indices_sha256": snapshot["indices_sha256"],
                     "full_bank_sha256": adapter.identity["full_bank_sha256"], "gym_full_bank_sha256": snapshot["full_bank_sha256"],
                     "frontier_max_size": max(s.frontier_out for s in solution.statistics),
                     "frontier_final_size": solution.statistics[-1].frontier_out,
                     "total_candidates_generated": sum(s.candidates_generated for s in solution.statistics),
                     "total_dominated_pruned": sum(s.dominated_pruned for s in solution.statistics),
                     "exact_duplicates_pruned": sum(s.exact_duplicates_pruned for s in solution.statistics),
                     "settlements": len(gym_trace), "natural_transitions": snapshot["transition_count"],
                     "traceback_actions": len(solution.actions), "tie_detected": solution.tie_detected,
                     "tie_count": solution.tie_count, "replay_pass": replay_pass})
        for policy in BASELINES:
            baseline_cost, _, _ = gym_episode(adapter, policy=policy, action_rule=action_rule)
            passed = solution.total_cost <= baseline_cost + 1e-12
            require(passed, f"Oracle feasibility bound: {policy}")
            sanity.append({"year": adapter.year, "trajectory_id": adapter.tid, "oracle_J": solution.total_cost,
                           "policy": policy, "policy_J": baseline_cost, "oracle_minus_policy": solution.total_cost - baseline_cost,
                           "oracle_not_worse_pass": passed})
    for number, name, value in ((20, "settlements", 365), (21, "natural_transitions", 364), (22, "traceback_actions", 365)):
        gate(audit, number, all(r[name] == value for r in full), {name: value})
    gate(audit, 23, all(r["abs_J_diff"] <= 1e-12 for r in full), "Annual absolute tolerance 1e-12; rtol=0")
    gate(audit, 24, all(r[f"max_abs_{k}_diff"] <= 1e-12 for r in full for k in ("L_pre", "L_post", "stage_cost", "L_next")), "Daily states/cost; dates/actions exact")
    gate(audit, 25, all(r["indices_sha256"] == r["gym_indices_sha256"] for r in full), "Exact indexed innovations")
    gate(audit, 26, all(r["full_bank_sha256"] == r["gym_full_bank_sha256"] for r in full), "Exact full banks")
    for number, policy in enumerate(BASELINES, 27):
        block = [r for r in sanity if r["policy"] == policy]
        gate(audit, number, len(block) == len(adapters) and all(r["oracle_not_worse_pass"] for r in block), policy)
    gate(audit, 31, all(r["frontier_out"] <= solver.MAX_FRONTIER_LABELS for r in frontier), "Fail closed at 1,000,000 labels; no truncation")
    return full, frontier, sanity, traces


def contract(assets):
    return {"solver": "EXACT_REACHABLE_LABEL_PARETO_DP", "name": "Finite-Horizon Clairvoyant Pareto-DP Oracle",
            "role": "CLAIRVOYANT_NON_CAUSAL_ORACLE", "state_grid": "NONE", "interpolation": "NONE",
            "epsilon_dominance": "NONE", "beam_search": "NONE", "top_K": "NONE", "frontier_quantization": "NONE",
            "future_information": "FULL_REALIZED_EXOGENOUS_TRAJECTORY", "causal": False, "deployment_role": "NOT_DEPLOYABLE",
            "objective": "MINIMUM_365_DAY_TOTAL_COST", "eta": assets.eta, "lambda_c": assets.lambda_main,
            "reward_scale": assets.reward_scale, "horizon": 365, "actions": {"0": "WAIT", "1": "CLEAN"},
            "terminal_semantics": "365 settlements / 364 transitions; final CLEAN and settlement, no terminal transition",
            "label_semantics": "(pre-action state, cumulative cost before current day); terminal label state denotes L_post",
            "dominance": "Same day: L1<=L2 and C1<=C2 with at least one strict; exact float64 ordering only",
            "dominance_theorem": [
                "WAIT L_post=L and CLEAN L_post=(1-eta)*L are nondecreasing for 0<eta<=1.",
                "Frozen finite E_t>0 makes fixed-action stage cost nondecreasing in L_post; lambda>=0.",
                "Dry next=clip(L_post+innovation,0,1) is nondecreasing.",
                "Rain next=clip((1+beta)*L_post+alpha+residual,0,1); actual frozen 1+beta>0 is checked.",
                "Clipping is nondecreasing. For a common future action sequence a smaller state and no higher cost cannot be worse.",
                "Analytic conditions justify pruning; dense probes only regress the frozen implementation. Tolerance never chooses frontier membership."],
            "actual_R3_slope": float(assets.r3["slope"]), "action_order": "WAIT before CLEAN",
            "frontier_sort": "stable (state ascending, cost ascending, creation_order ascending)",
            "duplicate_tie": "exact (state,cost) pair: retain earliest creation_order; clean_count not a pruning criterion",
            "terminal_tie_rule": "(cumulative_cost, terminal_L_post, clean_count, creation_order)",
            "terminal_tie_count_scope": "Exact minimum-cost ties among surviving terminal candidates after earlier exact dominance/duplicate removal",
            "terminal_frontier_accounting": "Winner-only frontier_out=1; other terminal candidates counted as terminal_unselected, not dominated pruning",
            "MAX_FRONTIER_LABELS": 1_000_000, "cap_response": "EXACT_SOLVER_FRONTIER_EXPLOSION; fail closed, no approximation",
            "traceback": "Saved per-day retained labels, parent_index/action; no re-solving",
            "numerical_comparison": "float64; atol=1e-12, rtol=0 only for objective/replay regression",
            "scientific_interpretation": "Model-trajectory cost lower bound / performance upper benchmark; full future information, not PPO-information-fair",
            "PPO_reference": "True-State PPO need not beat Oracle; core sanity benchmark remains TRUE_THRESHOLD_0.05"}


def parse_csv(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "Unique CSV columns")
    rows = list(reader)
    require(all(set(row) == set(reader.fieldnames) and None not in row.values() for row in rows), "CSV schema")
    return reader.fieldnames, rows


def publish(out, protocol, solver_contract, tables, sources, audit):
    payloads = {"protocol_manifest.json": encode(protocol), "solver_contract.json": encode(solver_contract),
                "source_artifact_hashes.json": encode(sources)}
    schemas = {}
    for name, rows in tables.items():
        require(rows, f"Nonempty output required: {name}")
        columns = list(rows[0])
        schemas[name] = columns
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        payloads[name] = stream.getvalue().encode("utf-8-sig")
    index = {"hashes": {n: sha(p) for n, p in payloads.items()},
             "excluded": ["output_hashes.json", "audit_summary.json"]}
    payloads["output_hashes.json"] = encode(index)
    require(all(v for k, v in audit["gates"].items() if k not in {"G34", "G35", "G37"}), "All scientific gates before publication")
    out.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (out / name).open("xb") as file:
            file.write(data)
    for name, expected in payloads.items():
        actual = (out / name).read_bytes()
        require(actual == expected and sha(actual) == sha(expected), "Disk readback/hash")
        if name in tables:
            columns, rows = parse_csv(actual)
            require(columns == schemas[name] and rows == [{k: "" if r[k] is None else str(r[k]) for k in columns}
                                                          for r in tables[name]], "CSV schema/content roundtrip")
        else:
            require(json.loads(actual) == json.loads(expected), "JSON schema/content roundtrip")
        if name in index["hashes"]:
            require(sha(actual) == index["hashes"][name], "Output hash index coverage")
    gate(audit, 34, all(sha(Path(p).read_bytes()) == r["sha256"] for p, r in sources.items())
         and all(script_record(p, protocol["current_HEAD"], r["git_blob"]) == r for p, r in protocol["script_provenance"].items()), "Frozen files unchanged during run")
    gate(audit, 35, git("rev-parse", "HEAD") == protocol["current_HEAD"], "HEAD unchanged")
    gate(audit, 36, all(script_record(p, protocol["current_HEAD"]) == r for p, r in protocol["current_files"].items()), "Both new files committed unchanged throughout run")
    gate(audit, 37, {p.name for p in out.iterdir()} == set(payloads),
         {"exclusive_write": True, "schema_readback": schemas, "hashes": {n: sha(p) for n, p in payloads.items()}, "audit_last": True})
    require(len(audit["gates"]) == 38 and all(audit["gates"].values()), "All G1-G38 required")
    audit.update(stage_pass=True, failed_gates=[], scientific_status=protocol["scientific_status"], declaration=DECLARATION)
    payload = encode(audit)
    with (out / "audit_summary.json").open("xb") as file:
        file.write(payload)
    require((out / "audit_summary.json").read_bytes() == payload
            and {p.name for p in out.iterdir()} == set(payloads) | {"audit_summary.json"}, "Final audit readback/completeness")


def main():
    parser = argparse.ArgumentParser(description=f"{STAGE}: {TITLE}; both modes DEVELOPMENT ONLY")
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    mode = parser.parse_args().mode
    audit = {"stage": STAGE, "stage_pass": False, "gates": {f"G{i}": False for i in GATES}, "gate_details": {}}
    protocol = {"stage": STAGE, "title": TITLE, "mode": mode, "required_gates": GATES,
                "scientific_status": SMOKE_STATUS if mode == "smoke" else FORMAL_STATUS, "declaration": DECLARATION,
                "environment_population": "DEVELOPMENT_ONLY", "formal_mode_meaning": "FORMAL SOLVER EXACTNESS AUDIT",
                "environment_roots": dict(DEVELOPMENT_ROOTS), "population_size": 300,
                "heldout_roots_accessed": False, "predecessor_artifact_role": "PROVENANCE_ONLY; raw CSV SHA256, no metric parsing",
                "baseline_role": "FIXED_FEASIBILITY_SANITY_ONLY; NO_RESELECTION", "baselines": list(BASELINES),
                "perception_seed": None, "observation_mode": "True-State", "perception_construction_count": 0, "perception_query_count": 0,
                "short_horizon": 12, "short_window_rule": "For each year trajectory 0: day0, earliest rain-containing and earliest all-dry 12-day window; deduplicate start days. Calendar/rain-mask only.",
                "short_initial_rule": "day0 uses frozen true initial state; all other windows fixed audit initial_state=0.08",
                "settlement_source": "frozen assets.reward_helper.settlement; adapter computes CLEAN post-state exactly as core.step",
                "transition_source": "frozen assets.natural.dry_next_samples/rain_next_samples; identical frozen day/index/single-innovation slice",
                "adapter_source": "Private read-only env._initial/_calendar/_energy/_rain/_indices and _audit_snapshot; never ordinary observation/info",
                "claim_boundary": "Solver exactness only. No held-out Oracle performance claim; no field or deployment validation.",
                "PPO_sanity_benchmark": "TRUE_THRESHOLD_0.05; Oracle is non-causal and need not be beaten"}
    sources = {}
    try:
        out = OUTPUT / mode
        require(not out.exists(), f"Immutable output exists: {out}")
        static = static_audit()
        for number in (8, 9, 10, 15, 32):
            gate(audit, number, True, static)
        core, solver, assets, action_rule = load_inputs(audit, protocol, sources)
        tids = (0,) if mode == "smoke" else (0, 1, 2)
        protocol["trajectory_ids"] = list(tids)
        adapters = [Adapter(core, assets, year, tid) for year in DEVELOPMENT_ROOTS for tid in tids]
        gate(audit, 5, {(a.year, a.tid) for a in adapters} == {(y, t) for y in DEVELOPMENT_ROOTS for t in tids}
             and all(a.spec.environment_root == DEVELOPMENT_ROOTS[a.year] for a in adapters), "Fixed 2 smoke / 6 formal development cases")
        # Preconditions and probes MUST finish before any Pareto solver call.
        monotonicity(adapters, assets, audit, protocol)
        cases = short_windows(adapters, protocol)
        brute = short_exactness(mode, cases, solver, audit)
        gate(audit, 19, all(any(r["year"] == c["year"] and r["rain_transition_count"] == 0 for r in brute)
             if c["first_dry_start"] is not None else c["dry_status"].startswith("NOT_AVAILABLE_WITH_REASON")
             for c in protocol["short_window_coverage"]), protocol["short_window_coverage"])
        full, frontier, sanity, traces = full_exactness(adapters, solver, action_rule, audit)
        require(len(full) == len(adapters) and len(frontier) == len(traces) == len(adapters) * 365
                and len(sanity) == len(adapters) * 4, "Complete output case populations")
        gate(audit, 6, set(assets._banks) == {(y, root, 300) for y, root in DEVELOPMENT_ROOTS.items()}
             and assets.bank_generation_count == 2, "Only the two development banks constructed")
        gate(audit, 7, assets.perception_construction_count == assets.perception_sample_count == 0
             and assets.emulator is None and not assets._perception_cache, "Zero construction/sample/query; seed=None on every EpisodeSpec")
        tables = {"bruteforce_exactness.csv": brute, "full_horizon_exactness.csv": full,
                  "frontier_size_by_day.csv": frontier, "dominance_sanity.csv": sanity, "oracle_action_trace.csv": traces}
        gate(audit, 33, all(math.isfinite(v) for rows in tables.values() for r in rows for v in r.values()
                            if isinstance(v, (float, int))), "Every numeric solver/audit output finite; terminal L_next is null")
        gate(audit, 38, protocol["scientific_status"] == (SMOKE_STATUS if mode == "smoke" else FORMAL_STATUS)
             and protocol["declaration"] == DECLARATION, "Exact mode status/declaration")
        solver_contract = contract(assets)
        protocol["dominance_theorem"] = solver_contract["dominance_theorem"]
        for adapter in adapters:
            adapter.check_identity()
        publish(out, protocol, solver_contract, tables, sources, audit)
    except Exception as exc:
        status = getattr(exc, "status", "EXACT_SOLVER_FRONTIER_EXPLOSION" if isinstance(exc, MemoryError) else "FAIL_CLOSED_NO_SOLVER_FREEZE")
        audit.update(stage_pass=False, scientific_status=status, execution_error=f"{type(exc).__name__}: {exc}",
                     failed_gates=[k for k, value in audit["gates"].items() if not value])
        if hasattr(exc, "statistics"):
            audit["frontier_failure"] = {"day_index": exc.day, "size_at_failure": exc.size,
                                         "completed_days": [asdict(s) for s in exc.statistics]}
        print(encode(audit).decode("utf-8"))
        return 1
    print(encode({"stage": STAGE, "scientific_status": audit["scientific_status"], "output": str(out)}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
