#!/usr/bin/env python
"""P2-2B-2-v1: frozen solver evaluation, never solver development.

Explicit execution only, after human review and Git checkpoint. Importing this
file constructs no environment and reads no scientific inputs. Smoke parses only
development comparator metrics; formal predecessor files are hashed in both modes.
Oracle has full knowledge of the realized future exogenous sequence and the exact
simulator. Action-conditioned future states are recursively generated, not given.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


STAGE = "P2-2B-2-v1"
TITLE = "Held-Out Clairvoyant Oracle Evaluation"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUTPUT = BASE / "p2_2b_2_heldout_clairvoyant_oracle_evaluation_v1"
B1 = BASE / "p2_2b_1_clairvoyant_pareto_dp_exactness_audit_v1/formal"
A2 = BASE / "p2_2a_2_heldout_nonrl_baseline_evaluation_v1"
SELF = "experiments/run_paper2_stage2b2_heldout_clairvoyant_oracle_evaluation_v1.py"
SOLVER = "experiments/paper2_clairvoyant_pareto_dp_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
CHECKPOINT = "eeb079c0cb24da8ecc61d2068e7b3db6a2130b47"
PINS = {
    SOLVER: "b9ff864426f2dabab430ac85672d58a9dd3eebd7",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    "experiments/run_paper2_stage2b1_clairvoyant_pareto_dp_exactness_audit_v1.py": "91062267a6596ce7417983d796720510b3a26b7a",
    "experiments/run_paper2_stage2a2_heldout_nonrl_baseline_evaluation_v1.py": "a5d0bcd2091b46be9cb55968fd24bd0d9efb5c4b",
}
B1_HASHES = {
    "audit_summary.json": "c7af605f8b3e31667d36c402969b33ab29b08a41dcf841b24130a95cc79b791c",
    "solver_contract.json": "72a479a1664cd8f1446b6805adbcd4ec96e817b91068772d773b2784a839cc0a",
    "protocol_manifest.json": "3791d48114b101ff7f22eed9a8b61051c18ca2681fda5d44ed73284de5e2c0bd",
    "bruteforce_exactness.csv": "76aef99f0d9967d62b0ed8d02dabb21e1b29953a97f789a5184d9a8d9a70d034",
    "full_horizon_exactness.csv": "8bd9e48a75aab70426ab26634960243abcd3493de2e225d7dee6ea26f73aa077",
    "frontier_size_by_day.csv": "db6647287015af88977e7a5122222d747cdb8d6f854a6ec66b41b716cae15125",
    "dominance_sanity.csv": "346714039d2718c496bb51b8e47f552475c3abe885d2b53ce8994adf4c4e0626",
    "oracle_action_trace.csv": "733fa0afcf9d12e135b44ca8719b83b01e35aef6d74d70586c9ecb1e92205c46",
    "source_artifact_hashes.json": "61e513e1a1a4bb1fdb96f664d43f6f73ccf1c5ba07b96507218ea2e0cdae3650",
    "output_hashes.json": "c99514a2077e415d3af2b28ad78a5abdc2c5fea2f8f952cf1806de175ad42158",
}
A2_HASHES = {
    "audit_summary.json": "b6899bd680e3c14e37af7a6f8b15ea4c3df9c3626a4a829b3aa39df4f9d71725",
    "protocol_manifest.json": "d40efcd74094a48d6721638780d4ff989f6065d0138765271e6d27c9c54f5e26",
    "policy_episode_metrics.csv": "6cb9b45b427f42edd15c27fbde697c7786062a15b21bfd324814257ba1a85776",
    "policy_aggregate_summary.csv": "377f88aba16967fc3f7e05c1344deebfc52909a57ff4bec7a442cb68f18305e9",
    "paired_comparison_summary.csv": "6eb39f7b5cda715dc14b66ec4006efe313a78f82bb09491fb670e4cfd33ab4ca",
    "representative_step_trace.csv": "b01f355cb6bf57ec641fa85769019ff96d9de02084debdcc4bff57796c44256c",
    "source_artifact_hashes.json": "3af5a5b3e73798ec07a5a6dee5939a11ff4a15c2b020eb3162cda40b5df2bbcf",
    "output_hashes.json": "e10de3ccb9ec4a8dcb91720a29b9bf7d3af12b50ea442fe70c9ac55770b4d9f8",
}
BASELINES = ("NEVER_CLEAN", "DAILY_CLEAN", "PERIODIC_14", "TRUE_THRESHOLD_0.05")
SMOKE_STATUS = "SMOKE_ONLY_NO_HELDOUT_ORACLE_CLAIM"
FORMAL_STATUS = "HELDOUT_CLAIRVOYANT_ORACLE_EVALUATION_PASS_FROZEN"
DECLARATION = "CLAIRVOYANT NON-CAUSAL ORACLE; NO DEPLOYMENT OR CAUSAL POLICY CLAIM; NO BASELINE RESELECTION"
GATES = dict(enumerate([
    "predecessor_checkpoint_blobs", "B1_artifact_hashes", "B1_frozen_status", "solver_contract",
    "core_blob", "A2_artifact_hashes", "A2_frozen_status", "mode_roots", "oracle_population",
    "trajectory_ids", "actions_settlements", "natural_transitions", "zero_perception",
    "no_training", "frozen_solver_import_only", "no_approximate_fallback", "annual_replay",
    "daily_replay", "clean_count_replay", "gym_indices", "gym_full_bank", "baseline_population",
    "four_baselines", "paired_keys", "paired_roots", "paired_indices", "paired_full_bank",
    "baseline_aggregate_regression", "oracle_dominance", "oracle_aggregate_counts", "CVaR_counts",
    "gap_summary_counts", "positive_baseline_costs", "frontier_cap", "finite_outputs",
    "representative_trace", "sources_unchanged", "HEAD_unchanged", "self_committed_unchanged",
    "exclusive_publication", "status_declaration",
], 1))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gate(audit, number, condition, detail):
    audit["gates"][f"G{number}"] = bool(condition)
    audit["gate_details"][f"G{number}"] = detail
    require(condition, f"G{number} {GATES[number]}: {detail}")


def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=0)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
                          cwd=ROOT, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def script_record(path, head, expected=None):
    blob = git("rev-parse", "--verify", f"{head}:{path}")
    require(blob == git("hash-object", f"--path={path}", path), f"Working tree differs: {path}")
    require(expected is None or blob == expected, f"Frozen blob mismatch: {path}")
    return {"git_blob": blob, "sha256": sha((ROOT / path).read_bytes())}


def read_source(path, sources, expected=None):
    payload = path.read_bytes()
    digest = sha(payload)
    require(expected is None or digest == expected, f"Frozen SHA256 mismatch: {path}")
    key = str(path)
    require(key not in sources or sources[key]["sha256"] == digest, f"Source changed: {path}")
    sources[key] = {"sha256": digest}
    return payload


def frozen_artifacts(directory, pins, sources):
    # CSV contents are never parsed here, including held-out episode metrics.
    for name, digest in pins.items():
        read_source(directory / name, sources, digest)
    index = json.loads(read_source(directory / "output_hashes.json", sources))
    require(index["hashes"] == {n: h for n, h in pins.items()
                                if n not in ("output_hashes.json", "audit_summary.json")}, "Frozen hash index")
    return json.loads(read_source(directory / "audit_summary.json", sources))


def frozen_status(summary, status, declaration):
    return (summary.get("stage_pass") is True and summary.get("failed_gates") == []
            and summary.get("scientific_status") == status and summary.get("declaration") == declaration)


def verify_contract(contract):
    # Assertions against the predecessor contract, never a replacement contract.
    expected = {"solver": "EXACT_REACHABLE_LABEL_PARETO_DP", "state_grid": "NONE",
                "interpolation": "NONE", "epsilon_dominance": "NONE", "beam_search": "NONE",
                "top_K": "NONE", "frontier_quantization": "NONE", "causal": False,
                "deployment_role": "NOT_DEPLOYABLE", "future_information": "FULL_REALIZED_EXOGENOUS_TRAJECTORY",
                "eta": .95, "lambda_c": .19925428989934618, "horizon": 365,
                "MAX_FRONTIER_LABELS": 1000000, "reward_scale": 1.0,
                "name": "Finite-Horizon Clairvoyant Pareto-DP Oracle", "role": "CLAIRVOYANT_NON_CAUSAL_ORACLE"}
    require(all(type(contract.get(k)) is type(v) and contract[k] == v for k, v in expected.items()),
            "Frozen solver contract exact types/values")
    return True


def mode_population(mode):
    if mode == "smoke":
        return {"YEAR1": 1020001, "YEAR2": 1030001}, (0,)
    if mode == "formal":
        return {"YEAR1": 1220001, "YEAR2": 1230001}, tuple(range(300))
    raise ValueError("Explicit smoke/formal mode required")


def comparator_directory(mode):
    if mode == "smoke":
        return A2 / "smoke"
    if mode == "formal":
        return A2 / "formal"
    raise ValueError("Explicit comparator mode required")


def parse_csv(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    columns = reader.fieldnames
    require(columns and len(columns) == len(set(columns)), "Unique CSV columns")
    rows = list(reader)
    require(rows and all(set(r) == set(columns) and None not in r.values() for r in rows), "Complete CSV schema")
    return columns, rows


def load_comparator(mode, roots, sources, audit):
    directory = comparator_directory(mode)
    # This is the sole baseline episode CSV parser; its directory is mode-guarded.
    columns, rows = parse_csv(read_source(directory / "policy_episode_metrics.csv", sources))
    required = {"year", "trajectory_id", "environment_root", "policy", "J_total", "N_clean",
                "indices_sha256", "full_bank_sha256"}
    require(required <= set(columns), f"Comparator schema: actual={columns}; missing={sorted(required - set(columns))}")
    n = 3 if mode == "smoke" else 300
    expected = {(y, t, roots[y], p) for y in roots for t in range(n) for p in BASELINES}
    index = {}
    for row in rows:
        for field in ("trajectory_id", "environment_root", "N_clean"):
            row[field] = int(row[field])
        row["J_total"] = float(row["J_total"])
        require(math.isfinite(row["J_total"]) and 0 <= row["N_clean"] <= 365, "Finite valid comparator")
        for field in ("indices_sha256", "full_bank_sha256"):
            require(len(row[field]) == 64 and all(c in "0123456789abcdef" for c in row[field]), "CRN SHA256 schema")
        key = (row["year"], row["trajectory_id"], row["environment_root"], row["policy"])
        require(key not in index, f"Duplicate comparator key: {key}")
        index[key] = row
    gate(audit, 22, set(index) == expected and len(rows) == 2 * n * 4,
         {"mode": mode, "baseline_episodes": len(rows), "per_year_per_policy": n, "columns": columns})
    gate(audit, 33, all(r["J_total"] > 0 for r in rows), "Positive costs before relative division")
    _, aggregates = parse_csv(read_source(directory / "policy_aggregate_summary.csv", sources))
    aggregate_index = {(r["scope"], r["policy"]): r for r in aggregates}
    require(len(aggregate_index) == len(aggregates) == 12
            and set(aggregate_index) == {(s, p) for s in ("YEAR1", "YEAR2", "JOINT") for p in BASELINES},
            "Frozen aggregate population")
    for (scope, policy), frozen in aggregate_index.items():
        block = [r for r in rows if r["policy"] == policy and (scope == "JOINT" or r["year"] == scope)]
        calculated = cost_summary(block)
        require(int(frozen["N_episodes"]) == len(block), "Frozen aggregate count")
        for field in ("mean_J_total", "median_J_total", "P05_J_total", "P95_J_total", "CVaR95_J_total", "mean_N_clean"):
            require(close(calculated[field], frozen[field]), f"Frozen aggregate regression: {scope}/{policy}/{field}")
    gate(audit, 28, True, "All 12 frozen aggregates; atol=1e-12, rtol=0; no baseline rollout")
    return index


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    require(spec is not None and spec.loader is not None, "Explicit frozen module path")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def preflight(audit, protocol, sources):
    head = git("rev-parse", "HEAD")
    require(git("rev-parse", "--verify", f"{CHECKPOINT}^{{commit}}") == CHECKPOINT, "Checkpoint exists")
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    records = {p: script_record(p, head, h) for p, h in PINS.items()}
    require(all(git("rev-parse", f"{CHECKPOINT}:{p}") == h for p, h in PINS.items()), "Historical blobs exact")
    gate(audit, 1, True, {"checkpoint": CHECKPOINT, "ancestor": True, "blobs": records})
    gate(audit, 5, records[CORE]["git_blob"] == PINS[CORE], records[CORE])
    b1 = frozen_artifacts(B1, B1_HASHES, sources)
    gate(audit, 2, True, B1_HASHES)
    gate(audit, 3, frozen_status(b1, "CLAIRVOYANT_PARETO_DP_SOLVER_EXACTNESS_PASS_FROZEN",
                              "SOLVER EXACTNESS ONLY; NO HELD-OUT ORACLE PERFORMANCE CLAIM"), "Exact B1 PASS/FROZEN")
    contract = json.loads(read_source(B1 / "solver_contract.json", sources))
    gate(audit, 4, verify_contract(contract), "Rules consumed from SHA-pinned B1 contract")
    a2 = frozen_artifacts(A2 / "formal", A2_HASHES, sources)
    gate(audit, 6, True, A2_HASHES)
    gate(audit, 7, frozen_status(a2, "HELDOUT_NONRL_BASELINE_EVALUATION_PASS_FROZEN",
                              "NO BASELINE RESELECTION FROM HELD-OUT RESULTS"), "Exact A2 PASS/FROZEN")
    records[SELF] = script_record(SELF, head)
    gate(audit, 39, True, records[SELF])
    protocol.update(current_HEAD=head, script_provenance=records, frozen_solver_contract=contract,
                    solver_contract_source=str(B1 / "solver_contract.json"))
    return contract


def environment_spec(core, mode, year, tid):
    roots, tids = mode_population(mode)
    require(year in roots and tid in tids, "Exact mode trajectory population")
    return core.EpisodeSpec(year=year, environment_root=roots[year], trajectory_id=tid,
                            population_size=300, perception_seed=None)


class Adapter:
    """Frozen Gym read-only exogenous extraction and model callback bridge.

    Same settlement/transition calls as frozen B1 adapter; no Pareto logic here.
    """

    def __init__(self, core, assets, mode, year, tid):
        self.core, self.assets = core, assets
        self.year, self.tid = year, tid
        self.spec = environment_spec(core, mode, year, tid)
        env = core.Paper2CleaningEnv(assets, self.spec, observation_mode="True-State")
        try:
            env.reset()
            self.identity = env._audit_snapshot()
            self.initial = float(env._initial)
            self.dates = tuple(env._calendar.date.dt.strftime("%Y-%m-%d"))
            self.energy, self.rain, self.indices = env._energy, env._rain, env._indices
            self.check_identity()
        finally:
            env.close()

    def check_identity(self):
        require(sha(self.indices.tobytes()) == self.identity["indices_sha256"]
                and not self.indices.flags.writeable, "Read-only exact trajectory CRN")
        bank = self.assets._banks[(self.year, self.spec.environment_root, 300)][2]
        require(sha(bank.tobytes()) == self.identity["full_bank_sha256"] and not bank.flags.writeable,
                "Read-only exact full bank")
        require(self.identity["perception_query_count"] == 0 and self.spec.perception_seed is None, "No perception")

    def settlement(self, day, pre, action):
        post = (1 - self.assets.eta) * pre if action else pre
        _, _, total, _, _ = self.assets.reward_helper.settlement(
            float(self.energy[day]), post, action, self.assets.lambda_main)
        return float(post), float(total)

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
        return float(following)


def evaluate_episode(adapter, solver, contract):
    import numpy as np
    solution = solver.solve_pareto_dp(adapter.initial, contract["horizon"], adapter.settlement, adapter.transition)
    require(len(solution.actions) == len(solution.trace) == len(solution.statistics) == 365
            and all(type(a) is int and a in (0, 1) for a in solution.actions), "365 binary Oracle actions")
    env = adapter.core.Paper2CleaningEnv(adapter.assets, adapter.spec, observation_mode="True-State")
    differences = {k: 0.0 for k in ("L_pre", "L_post", "stage_cost", "L_next")}
    gym_rows, traces = [], []
    try:
        env.reset()
        for day, (action, dp, stats) in enumerate(zip(solution.actions, solution.trace, solution.statistics)):
            _, reward, terminal, truncated, info = env.step(action)
            snapshot = env._audit_snapshot()
            row = snapshot["last_reward_components"]
            require(row["year"] == adapter.year and row["trajectory_id"] == adapter.tid
                    and row["date"] == info["date"] == adapter.dates[day]
                    and row["day_index"] == info["day_index"] == dp.day_index == stats.day_index == day
                    and row["action"] == dp.action == action and info["action_name"] == ("CLEAN" if action else "WAIT")
                    and terminal == (day == 364) and not truncated, "Exact replay categories/date/actions")
            require(row["transition_executed"] == (day < 364)
                    and row["rain_affected_transition"] == (bool(adapter.rain[day]) if day < 364 else False)
                    and row["E_clean_GHI"] == float(adapter.energy[day]), "Exogenous replay categories exact")
            require(reward == row["final_reward"] == -row["total_cost"]
                    and snapshot["perception_query_count"] == 0, "Reward and no perception")
            for name, actual in (("L_pre", row["L_pre"]), ("L_post", row["L_post"]), ("stage_cost", row["total_cost"])):
                differences[name] = max(differences[name], abs(getattr(dp, name) - actual))
            if day == 364:
                require(dp.L_next is None and row["L_next"] is None, "Terminal null next state")
            else:
                require(dp.L_next is not None and row["L_next"] is not None, "Nonterminal next state")
                differences["L_next"] = max(differences["L_next"], abs(dp.L_next - row["L_next"]))
            gym_rows.append(row)
            require(close(dp.cumulative_cost, -snapshot["episode_return"]), "Cumulative cost replay")
            if adapter.tid == 0:
                traces.append({"year": adapter.year, "trajectory_id": adapter.tid, "date": row["date"],
                               "day_index": day, "L_pre": row["L_pre"], "action": action, "L_post": row["L_post"],
                               "E_clean_GHI": row["E_clean_GHI"], "J_soiling_step": row["soiling_cost"],
                               "J_cleaning_step": row["cleaning_cost"], "J_total_step": row["total_cost"],
                               "reward": reward, "L_next": row["L_next"], "cumulative_cost": -snapshot["episode_return"],
                               **{k: getattr(stats, k) for k in ("frontier_in", "candidates_generated", "dominated_pruned",
                                                               "exact_duplicates_pruned", "frontier_out")}})
        require(snapshot["transition_count"] == 364 and snapshot["terminated"]
                and snapshot["clean_count"] == info["clean_count"] == solution.clean_count == sum(solution.actions), "Annual counts exact")
        for name in ("year", "trajectory_id", "environment_root", "indices_sha256", "full_bank_sha256"):
            require(snapshot[name] == adapter.identity[name], f"Gym/extraction identity: {name}")
        adapter.check_identity()
    finally:
        env.close()
    total = sum(r["total_cost"] for r in gym_rows)
    require(close(total, solution.total_cost) and close(total, -snapshot["episode_return"])
            and max(differences.values()) <= 1e-12, "Annual/daily DP/Gym replay atol=1e-12 rtol=0")
    sizes = [s.frontier_out for s in solution.statistics]
    maximum = max(sizes)
    cap = contract["MAX_FRONTIER_LABELS"]
    require(maximum <= cap and all(s.frontier_in <= cap for s in solution.statistics), "Frozen protective cap")
    engineering = {"year": adapter.year, "trajectory_id": adapter.tid, "frontier_max_size": maximum,
                   "frontier_max_day": sizes.index(maximum), "frontier_mean_size": float(np.mean(sizes)),
                   "total_candidates_generated": sum(s.candidates_generated for s in solution.statistics),
                   "total_dominated_pruned": sum(s.dominated_pruned for s in solution.statistics),
                   "exact_duplicates_pruned": sum(s.exact_duplicates_pruned for s in solution.statistics),
                   "cap": cap, "cap_fraction_used": maximum / cap, "frontier_cap_pass": True}
    episode = {"year": adapter.year, "trajectory_id": adapter.tid, "environment_root": adapter.spec.environment_root,
               "J_total": total, "J_soiling": sum(r["soiling_cost"] for r in gym_rows),
               "J_cleaning": sum(r["cleaning_cost"] for r in gym_rows), "N_clean": snapshot["clean_count"],
               "episode_return": snapshot["episode_return"], "mean_L_pre": float(np.mean([r["L_pre"] for r in gym_rows])),
               "max_L_pre": max(r["L_pre"] for r in gym_rows), "settlements": len(gym_rows),
               "natural_transitions": snapshot["transition_count"], "dp_clean_count": solution.clean_count,
               "gym_clean_count": snapshot["clean_count"], "dp_J_total": solution.total_cost, "gym_J_total": total,
               "abs_replay_J_diff": abs(solution.total_cost - total),
               **{f"max_abs_{k}_diff": v for k, v in differences.items()},
               "indices_sha256": adapter.identity["indices_sha256"], "full_bank_sha256": adapter.identity["full_bank_sha256"],
               "gym_indices_sha256": snapshot["indices_sha256"], "gym_full_bank_sha256": snapshot["full_bank_sha256"],
               "action_sequence_sha256": sha(np.asarray(solution.actions, dtype=np.uint8).tobytes()),
               "frontier_max_size": maximum, "frontier_final_size": sizes[-1],
               **{k: engineering[k] for k in ("total_candidates_generated", "total_dominated_pruned", "exact_duplicates_pruned")},
               "perception_query_count": snapshot["perception_query_count"]}
    require(close(episode["J_total"], episode["J_soiling"] + episode["J_cleaning"]), "Cost decomposition")
    return episode, engineering, traces


def cost_summary(rows):
    import numpy as np
    costs = np.asarray([r["J_total"] for r in rows], dtype=float)
    cleans = np.asarray([r["N_clean"] for r in rows], dtype=float)
    n = len(rows)
    require(n > 0, "Nonempty summary")
    worst = math.ceil(.05 * n)
    return {"N_episodes": n, "mean_J_total": float(costs.mean()),
            "std_J_total": float(costs.std(ddof=1)) if n > 1 else 0.0,
            "median_J_total": float(np.median(costs)),
            **{f"P{q:02d}_J_total": float(np.percentile(costs, q, method="linear")) for q in (5, 25, 75, 95)},
            "CVaR95_J_total": float(np.sort(costs)[::-1][:worst].mean()), "CVaR95_episode_count": worst,
            "mean_N_clean": float(cleans.mean()), "std_N_clean": float(cleans.std(ddof=1)) if n > 1 else 0.0,
            "median_N_clean": float(np.median(cleans)),
            **{f"P{q:02d}_N_clean": float(np.percentile(cleans, q, method="linear")) for q in (5, 95)}}


def oracle_aggregates(episodes):
    rows = []
    for scope in ("YEAR1", "YEAR2", "JOINT"):
        block = [r for r in episodes if scope == "JOINT" or r["year"] == scope]
        row = {"scope": scope, **cost_summary(block)}
        for field in ("J_soiling", "J_cleaning", "frontier_max_size", "total_candidates_generated"):
            row[f"mean_{field}"] = math.fsum(r[field] for r in block) / len(block)
        for field in ("frontier_max_size", "total_candidates_generated"):
            row[f"max_{field}"] = max(r[field] for r in block)
        rows.append(row)
    return rows


def pair_episode(episode, comparator):
    rows = []
    for policy in BASELINES:
        key = (episode["year"], episode["trajectory_id"], episode["environment_root"], policy)
        require(key in comparator, f"Missing exact comparator key: {key}")
        baseline = comparator[key]
        for field in ("environment_root", "indices_sha256", "full_bank_sha256"):
            require(episode[field] == baseline[field], f"Exact paired identity: {key}/{field}")
        oracle, base = episode["J_total"], baseline["J_total"]
        require(base > 0 and oracle <= base + 1e-12, f"Oracle feasible dominance failed: {key}")
        gap = base - oracle
        rows.append({"year": episode["year"], "trajectory_id": episode["trajectory_id"],
                     "environment_root": episode["environment_root"], "baseline_policy": policy,
                     "oracle_J_total": oracle, "baseline_J_total": base, "gap_absolute": gap,
                     "relative_headroom_vs_baseline": gap / base, "oracle_cost_ratio": oracle / base,
                     "oracle_N_clean": episode["N_clean"], "baseline_N_clean": baseline["N_clean"],
                     "clean_count_difference": episode["N_clean"] - baseline["N_clean"],
                     "indices_sha256": episode["indices_sha256"], "full_bank_sha256": episode["full_bank_sha256"],
                     "oracle_strictly_better": gap > 1e-12, "oracle_tie_within_tolerance": abs(gap) <= 1e-12,
                     "oracle_not_worse_pass": oracle <= base + 1e-12})
    return rows


def gap_summaries(pairs):
    import numpy as np
    summaries = []
    for scope in ("YEAR1", "YEAR2", "JOINT"):
        for policy in BASELINES:
            block = [r for r in pairs if r["baseline_policy"] == policy and (scope == "JOINT" or r["year"] == scope)]
            n = len(block)
            require(n > 0, "Nonempty paired summary")
            row = {"scope": scope, "baseline_policy": policy, "N_pairs": n}
            for field, label in (("gap_absolute", "gap_absolute"), ("relative_headroom_vs_baseline", "relative_headroom"),
                                 ("oracle_cost_ratio", "oracle_cost_ratio")):
                values = np.asarray([r[field] for r in block])
                row[f"mean_{label}"] = float(values.mean())
                row[f"median_{label}"] = float(np.median(values))
                if field != "oracle_cost_ratio":
                    for q in (5, 95):
                        row[f"P{q:02d}_{label}"] = float(np.percentile(values, q, method="linear"))
                if field == "gap_absolute":
                    row["std_gap_absolute"] = float(values.std(ddof=1)) if n > 1 else 0.0
                    row["min_gap_absolute"], row["max_gap_absolute"] = float(values.min()), float(values.max())
            row.update(oracle_strict_better_fraction=sum(r["oracle_strictly_better"] for r in block) / n,
                       oracle_tie_fraction=sum(r["oracle_tie_within_tolerance"] for r in block) / n,
                       mean_clean_count_difference=math.fsum(r["clean_count_difference"] for r in block) / n)
            summaries.append(row)
    return summaries


def static_audit():
    """AST only; safe before checkpoint, without importing core or solver."""
    tree = ast.parse((ROOT / SELF).read_text(encoding="utf-8"))
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports.update(a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names)
    require(imports <= {"__future__", "argparse", "ast", "csv", "hashlib", "importlib.util", "io", "json",
                        "math", "pathlib", "subprocess", "sys", "tempfile", "numpy"}, "No training or parallel runtime imports")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
             for n in calls if isinstance(n.func, (ast.Name, ast.Attribute))}
    forbidden = {"PPO", "learn", "backward", "optimizer", "OnlineBlock10Emulator", "ensure_perception",
                 "perception_record", "sample_one", "default_rng", "build_crn_indices", "pareto_prune",
                 "linspace", "interp", "multiprocessing", "short_exactness", "gym_episode"}
    require(not names.intersection(forbidden), "No training/perception/baseline rollout/solver implementation")
    require(not any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in
                    {"solve_pareto_dp", "pareto_prune", "Label", "Solution"} for n in ast.walk(tree)), "No solver copy")
    solve_calls = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "solve_pareto_dp"]
    require(len(solve_calls) == 1 and ast.unparse(solve_calls[0].func) == "solver.solve_pareto_dp", "Single frozen solve call")
    spec_calls = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "EpisodeSpec"]
    require(len(spec_calls) == 1 and {k.arg: ast.unparse(k.value) for k in spec_calls[0].keywords} ==
            {"year": "year", "environment_root": "roots[year]", "trajectory_id": "tid",
             "population_size": "300", "perception_seed": "None"}, "Single mode-guarded spec; no perception seed")
    module_calls = [ast.unparse(n) for n in calls if isinstance(n.func, ast.Name) and n.func.id == "module"]
    require(sorted(module_calls) == sorted(["module('_p2_2b2_core', CORE)", "module('_p2_2b2_solver', SOLVER)"]),
            "Only frozen core and solver imported; baseline/audit scripts never executed")
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    roots = functions["mode_population"]
    require(ast.unparse(roots.body[1].test) == "mode == 'formal'"
            and ast.literal_eval(roots.body[1].body[0].value.elts[0]) == {"YEAR1": 1220001, "YEAR2": 1230001}, "Formal-only heldout roots")
    require(ast.literal_eval(roots.body[0].body[0].value.elts[0]) == {"YEAR1": 1020001, "YEAR2": 1030001}, "Smoke roots")
    # Exclude the audit's own assertion literals from executable root-use checks.
    runtime_nodes = [node for top in tree.body
                     if not (isinstance(top, ast.FunctionDef) and top.name == "static_audit")
                     for node in ast.walk(top)]
    for node in runtime_nodes:
        if isinstance(node, ast.Constant) and node.value in (1220001, 1230001):
            require(any(node is child for child in ast.walk(roots.body[1])), "Heldout root literal outside formal guard")
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            require(node.value not in range(20260907, 20260912), "No formal perception seeds")
    directory = functions["comparator_directory"]
    require(ast.unparse(directory.body[0].test) == "mode == 'smoke'"
            and ast.unparse(directory.body[0].body[0].value) == "A2 / 'smoke'"
            and ast.unparse(directory.body[1].test) == "mode == 'formal'"
            and ast.unparse(directory.body[1].body[0].value) == "A2 / 'formal'", "Comparator path isolation")
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "parse_csv"
                    for n in ast.walk(functions["frozen_artifacts"])), "Formal provenance never parses CSV")
    parser_owners = {name for name, function in functions.items()
                     if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "parse_csv"
                            for n in ast.walk(function))}
    require(parser_owners == {"load_comparator", "publish"}, "Only mode comparator and output readback parse CSV")
    require(ast.unparse(functions["load_comparator"].body[0]) == "directory = comparator_directory(mode)",
            "Comparator directory comes only from mode guard")
    require(sum(isinstance(n, ast.Try) for n in ast.walk(functions["main"])) == 1,
            "Single fail-closed execution boundary; no trajectory retry")
    return {"mode_roots_guarded": True, "smoke_comparator": "A2/smoke only",
            "formal_provenance_CSV_use": "SHA256_BYTES_ONLY", "baseline_execution": "NONE",
            "solver_calls": len(solve_calls), "solver_implementation": "FROZEN_IMPORT_ONLY",
            "perception_seed": None, "parallel_execution": False}


def final_gates(audit, mode, roots, tids, assets, episodes, pairs, aggregates, gaps, frontier, traces):
    expected = {(y, t, roots[y]) for y in roots for t in tids}
    keys = [(r["year"], r["trajectory_id"], r["environment_root"]) for r in episodes]
    gate(audit, 8, set(assets._banks) == {(y, root, 300) for y, root in roots.items()}
         and assets.bank_generation_count == 2, {"mode": mode, "roots": roots})
    gate(audit, 9, len(keys) == len(set(keys)) == len(expected) and set(keys) == expected, "Exact Oracle population")
    gate(audit, 10, all({r["trajectory_id"] for r in episodes if r["year"] == y} == set(tids) for y in roots), list(tids))
    gate(audit, 11, all(r["settlements"] == 365 for r in episodes), "365 actions verified before each replay")
    gate(audit, 12, all(r["natural_transitions"] == 364 for r in episodes), "No terminal transition")
    gate(audit, 13, assets.perception_construction_count == assets.perception_sample_count == 0
         and assets.emulator is None and not assets._perception_cache
         and all(r["perception_query_count"] == 0 for r in episodes), "Zero construction/query/sample; seed=None")
    gate(audit, 17, all(r["abs_replay_J_diff"] <= 1e-12 for r in episodes), "atol=1e-12; rtol=0")
    gate(audit, 18, all(r[f"max_abs_{k}_diff"] <= 1e-12 for r in episodes
                      for k in ("L_pre", "L_post", "stage_cost", "L_next")), "Daily numeric and categorical replay checked")
    gate(audit, 19, all(r["dp_clean_count"] == r["gym_clean_count"] == r["N_clean"] for r in episodes), "Exact clean count")
    gate(audit, 20, all(r["indices_sha256"] == r["gym_indices_sha256"] for r in episodes), "Exact SHA256")
    gate(audit, 21, all(r["full_bank_sha256"] == r["gym_full_bank_sha256"] for r in episodes), "Exact SHA256")
    pairkeys = [(r["year"], r["trajectory_id"], r["environment_root"], r["baseline_policy"]) for r in pairs]
    expected_pairs = {(*k, p) for k in expected for p in BASELINES}
    gate(audit, 23, all(sum(k[:3] == key for k in pairkeys) == 4 for key in expected), "Four frozen comparators per Oracle")
    gate(audit, 24, len(pairkeys) == len(set(pairkeys)) == len(expected_pairs) and set(pairkeys) == expected_pairs, "No duplicate/missing pairs")
    for number, field in ((25, "environment_root"), (26, "indices_sha256"), (27, "full_bank_sha256")):
        gate(audit, number, True, f"{field} exact checked for every comparator before pair creation")
    gate(audit, 29, all(r["oracle_not_worse_pass"] for r in pairs), "Every pair: Oracle <= baseline + 1e-12; ties allowed")
    counts = [len(tids), len(tids), 2 * len(tids)]
    gate(audit, 30, [r["N_episodes"] for r in aggregates] == counts, counts)
    gate(audit, 31, [r["CVaR95_episode_count"] for r in aggregates] == [math.ceil(.05 * n) for n in counts],
         "Upper cost tail; fixed worst ceil(.05*N), independent of quantile ties")
    gate(audit, 32, len(gaps) == 12 and [r["N_pairs"] for r in gaps] == [n for n in counts for _ in BASELINES], "12 exact gap summary rows")
    gate(audit, 34, len(frontier) == len(episodes) and all(r["frontier_cap_pass"] for r in frontier), "Frozen cap; no fallback")
    gate(audit, 36, len(traces) == 730 and {(r["year"], r["trajectory_id"], r["day_index"]) for r in traces}
         == {(y, 0, d) for y in roots for d in range(365)}, "Fixed YEAR1/2 tid0, no outcome selection")


def unchanged(audit, protocol, sources):
    gate(audit, 37, all(sha(Path(p).read_bytes()) == r["sha256"] for p, r in sources.items())
         and all(script_record(p, protocol["current_HEAD"], r["git_blob"]) == r
                 for p, r in protocol["script_provenance"].items()), "All sources and scripts unchanged")
    gate(audit, 38, git("rev-parse", "HEAD") == protocol["current_HEAD"], "HEAD unchanged")
    gate(audit, 39, script_record(SELF, protocol["current_HEAD"]) == protocol["script_provenance"][SELF], "Self committed unchanged")


def publish(out, protocol, tables, sources, audit):
    """Stage complete results privately; audit last; atomic Windows rename.

    No output directory is visible until all evaluations and readbacks pass.
    TemporaryDirectory cleans failed staging only; existing outputs are untouched.
    """
    require(not out.exists(), f"Exclusive output already exists: {out}")
    payloads = {"protocol_manifest.json": encode(protocol), "source_artifact_hashes.json": encode(sources)}
    schemas = {}
    for name, rows in tables.items():
        require(rows, f"Empty table: {name}")
        columns = list(rows[0])
        require(all(set(r) == set(columns) for r in rows), f"Uniform schema: {name}")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        schemas[name] = columns
        payloads[name] = stream.getvalue().encode("utf-8-sig")
    payloads["output_hashes.json"] = encode({"hashes": {n: sha(p) for n, p in payloads.items()},
                                            "excluded": ["output_hashes.json", "audit_summary.json"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    # Same-volume rename; Path.rename on Windows fails if destination exists.
    require(sys.platform == "win32", "Publication v1 requires Windows exclusive rename semantics")
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-staging-", dir=out.parent) as temporary:
        staged = Path(temporary) / "complete"
        staged.mkdir()
        for name, payload in payloads.items():
            with (staged / name).open("xb") as file:
                file.write(payload)
            actual = (staged / name).read_bytes()
            require(actual == payload and sha(actual) == sha(payload), "Output byte/hash readback")
            if name in tables:
                columns, rows = parse_csv(actual)
                require(columns == schemas[name] and rows == [{k: "" if r[k] is None else str(r[k]) for k in columns}
                                                              for r in tables[name]], "CSV schema/content roundtrip")
            else:
                require(json.loads(actual) == json.loads(payload), "JSON roundtrip")
        unchanged(audit, protocol, sources)
        gate(audit, 40, {p.name for p in staged.iterdir()} == set(payloads),
             {"exclusive": True, "atomic_directory_publication": True, "schemas": schemas,
              "hashes": {n: sha(p) for n, p in payloads.items()}, "audit_last": True})
        require(set(audit["gates"]) == {f"G{i}" for i in range(1, 42)} and all(audit["gates"].values()), "All G1-G41 pass")
        audit.update(stage_pass=True, failed_gates=[], scientific_status=protocol["scientific_status"], declaration=DECLARATION)
        payload = encode(audit)
        with (staged / "audit_summary.json").open("xb") as file:
            file.write(payload)
        require((staged / "audit_summary.json").read_bytes() == payload
                and {p.name for p in staged.iterdir()} == set(payloads) | {"audit_summary.json"}, "Final audit readback/completeness")
        require(not out.exists(), "Exclusive destination still absent")
        staged.rename(out)


def main():
    parser = argparse.ArgumentParser(description=f"{STAGE}: {TITLE}")
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    mode = parser.parse_args().mode
    audit = {"stage": STAGE, "stage_pass": False, "gates": {f"G{i}": False for i in GATES}, "gate_details": {}}
    sources = {}
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        out = OUTPUT / mode
        require(not out.exists(), f"Immutable output exists: {out}")
        roots, tids = mode_population(mode)
        protocol = {"stage": STAGE, "title": TITLE, "mode": mode, "required_gates": GATES,
                    "scientific_status": SMOKE_STATUS if mode == "smoke" else FORMAL_STATUS, "declaration": DECLARATION,
                    "environment_roots": roots, "trajectory_ids": list(tids), "population_size": 300,
                    "N_oracle_episodes": 2 * len(tids), "N_pairs": 8 * len(tids), "baselines": list(BASELINES),
                    "baseline_role": "READ_ONLY_FIXED_COMPARATOR; NO ROLLOUT/SELECTION/TUNING",
                    "comparator_directory": str(comparator_directory(mode)),
                    "heldout_trajectories_accessed": mode == "formal", "perception_seed": None,
                    "perception_construction_count": 0, "perception_query_count": 0,
                    "observation_mode": "True-State", "execution": "DETERMINISTIC_SINGLE_PROCESS",
                    "oracle_information": "Oracle has full knowledge of the realized future exogenous sequence and the exact simulator.",
                    "state_semantics": "Future states recur from current state, chosen action, frozen transition and realized exogenous sequence.",
                    "scientific_role": "CLAIRVOYANT_NON_CAUSAL_ORACLE",
                    "claim_boundary": "Model-internal non-causal cost lower bound; not deployable, causal, PPO-information-fair or field validation.",
                    "headroom_terminology": "normalized-cost headroom / model-internal clairvoyant headroom",
                    "gap_absolute": "J_baseline - J_oracle", "relative_headroom_vs_baseline": "gap_absolute / J_baseline",
                    "oracle_cost_ratio": "J_oracle / J_baseline", "clean_count_difference": "N_oracle - N_baseline",
                    "tie_definition": "abs(gap_absolute) <= 1e-12", "std_ddof": 1,
                    "singleton_std": "0.0 descriptive convention for smoke N=1; no sample-variance inference",
                    "quantile_method": "linear", "CVaR95_definition": "Descending cost; mean worst ceil(0.05*N) episodes",
                    "action_hash": "SHA256(np.asarray(actions, dtype=np.uint8).tobytes()); 365 binary actions",
                    "representatives": "YEAR1 trajectory 0 and YEAR2 trajectory 0, all 365 days",
                    "frontier_summary": "frontier_out after each day; earliest zero-based max day; includes terminal winner",
                    "PPO_sanity_benchmark": "TRUE_THRESHOLD_0.05; no requirement to beat non-causal Oracle"}
        static = static_audit()
        for number in (14, 15, 16):
            gate(audit, number, True, static)
        contract = preflight(audit, protocol, sources)
        comparator = load_comparator(mode, roots, sources, audit)
        # All provenance and comparator checks precede the first environment bank.
        core = module("_p2_2b2_core", CORE)
        solver = module("_p2_2b2_solver", SOLVER)
        require(solver.MAX_FRONTIER_LABELS == contract["MAX_FRONTIER_LABELS"], "Frozen solver cap")
        for path, digest in core.ASSET_HASHES.values():
            read_source(path, sources, digest)
        for name, digest in core.REWARD_HASHES.items():
            read_source(core.REWARD_DIRECTORY / name, sources, digest)
        for path, blob in [*core.HELPERS.values(), (core.REWARD_SCRIPT, core.REWARD_BLOB)]:
            protocol["script_provenance"][path] = script_record(path, protocol["current_HEAD"], blob)
        assets = core.Paper2EnvAssets()
        require(assets.eta == contract["eta"] and assets.lambda_main == contract["lambda_c"]
                and assets.reward_scale == contract["reward_scale"], "Frozen model/contract agreement")
        protocol["source_environment_provenance"] = assets.provenance
        episodes, pairs, frontier, traces = [], [], [], []
        for year in roots:
            for tid in tids:
                adapter = Adapter(core, assets, mode, year, tid)
                episode, engineering, representative = evaluate_episode(adapter, solver, contract)
                pairs.extend(pair_episode(episode, comparator))
                episodes.append(episode)
                frontier.append(engineering)
                traces.extend(representative)
                if (tid + 1) % 25 == 0:
                    print(f"{year} {tid + 1}/{len(tids)} current_max_frontier={max(r['frontier_max_size'] for r in frontier)}", flush=True)
        aggregates, gaps = oracle_aggregates(episodes), gap_summaries(pairs)
        final_gates(audit, mode, roots, tids, assets, episodes, pairs, aggregates, gaps, frontier, traces)
        tables = {"oracle_episode_metrics.csv": episodes, "oracle_aggregate_summary.csv": aggregates,
                  "paired_episode_gaps.csv": pairs, "paired_gap_summary.csv": gaps,
                  "frontier_engineering_summary.csv": frontier, "representative_oracle_trace.csv": traces}
        gate(audit, 35, all(math.isfinite(v) for rows in tables.values() for r in rows for v in r.values()
                            if isinstance(v, (int, float))), "All numeric outputs finite; terminal L_next null")
        gate(audit, 41, protocol["scientific_status"] == (SMOKE_STATUS if mode == "smoke" else FORMAL_STATUS)
             and protocol["declaration"] == DECLARATION, "Exact mode-specific status/declaration")
        publish(out, protocol, tables, sources, audit)
    except Exception as exc:
        audit.update(stage_pass=False, scientific_status="FAIL_CLOSED_NO_HELDOUT_ORACLE_CLAIM",
                     execution_error=f"{type(exc).__name__}: {exc}",
                     failed_gates=[k for k, passed in audit["gates"].items() if not passed])
        print(encode(audit).decode("utf-8"))
        return 1
    finally:
        sys.dont_write_bytecode = previous_bytecode
    print(encode({"stage": STAGE, "scientific_status": audit["scientific_status"], "output": str(out)}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
