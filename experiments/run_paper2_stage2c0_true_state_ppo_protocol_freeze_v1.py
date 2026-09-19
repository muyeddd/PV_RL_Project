"""Metadata/provenance/static audit only. Neither mode runs a scientific episode.

Human source review -> Git checkpoint -> smoke audit/acceptance -> formal audit.
Imports perform no file reads, experiment, output, or random generator operations.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess

import paper2_ppo_protocol_v1 as protocol


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage2c0_true_state_ppo_protocol_freeze_v1.py"
MODULE = "experiments/paper2_ppo_protocol_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
PREDECESSOR = "experiments/run_paper2_stage2b2_heldout_clairvoyant_oracle_evaluation_v1.py"
CHECKPOINT = "689ddce3bbe6fcd77a760c93e6f98e9d4670d7a0"
CORE_BLOB = "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4"
PREDECESSOR_BLOB = "7995fcc20a15b6c208701a62aca0ec721e7338ce"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
FROZEN = BASE / "p2_2b_2_heldout_clairvoyant_oracle_evaluation_v1/formal"
OUTPUT = BASE / "p2_2c_0_true_state_ppo_protocol_freeze_v1"
FROZEN_HASHES = {
    "audit_summary.json": "536abd4613d92b082a03c989453a963d96a0db28f8ba9e0943a891d1e5c678f0",
    "protocol_manifest.json": "d522ff3b79a1f7cbcad789410aa9bfa3600817260cf13708741db41d97a05f49",
    "oracle_episode_metrics.csv": "fc0f67b450a23e89850d6867f400aa9bb0411470ae2dc52d1f99758daf4358ec",
    "oracle_aggregate_summary.csv": "0b105e738eef71ba2ee1d2724e83acf00f29cf247bc9946a89c073039508314d",
    "paired_episode_gaps.csv": "8b2b19a1869012492bda845345fcf20d696c186ae97e3005ee38d929e4ccece1",
    "paired_gap_summary.csv": "8b64507bd7075dbdc7a20275d2ad12507e1511b8599535bd930264e8fc6167d3",
    "frontier_engineering_summary.csv": "702644cf77a749c1c2502d3ef3699487697b88ed08621dfdcd7305c38cc29dfc",
    "representative_oracle_trace.csv": "37a440db55d66cb0defc64f2201a9c37af485e1c4264e8c6011544016fcc085b",
    "source_artifact_hashes.json": "1b547236a8a5f4c97d7fb8e89daba045d492161a8c26e460f07a56182c4a10a0",
    "output_hashes.json": "4b942b882684dd6ed0f398cc2077c632903865b6a5fa2820911bfbbfdd372b9a",
}
FORMAL_STATUS = "TRUE_STATE_PPO_PROTOCOL_PASS_FROZEN"
SMOKE_STATUS = "SMOKE_ONLY_NO_PPO_PROTOCOL_FREEZE"
DECLARATION = ("TRUE-STATE PPO PROTOCOL PRE-REGISTERED BEFORE PPO TRAINING; "
               "NO FORMAL PPO PERFORMANCE ACCESSED; NO POINT/UA HYPERPARAMETER RESELECTION")
# Canonical full metadata pin includes prose obligations as well as numeric fields.
# Updating this v1 pin after Formal Freeze is itself a protocol-version violation.
PROTOCOL_SHA256 = "f561dc9989c5ee7e70b88e8d08adb07ff5f5323a9c18d379b58ad39b77c5cb5e"
GATE_NAMES = (
    "predecessor_checkpoint", "predecessor_blob", "predecessor_artifact_hashes", "predecessor_status",
    "core_historical_blob", "dependencies", "observation", "root_partitions", "roots_disjoint",
    "trajectory_ids", "development_seeds", "final_seeds", "seed_namespace", "seed_substreams",
    "canonical_population", "balanced_schedule", "architecture", "common_hyperparameters", "candidates",
    "development_budget", "final_budget", "selection", "tie_rules", "final_checkpoint_only",
    "no_early_stop_or_best_model", "sanity_gate", "formal_access", "statistical_unit", "inheritance",
    "width_ablation", "no_perception", "static_no_execution", "no_trajectory_access", "normalization",
    "gamma_rationale", "sources_unchanged", "HEAD_unchanged", "formal_committed_sources",
    "exclusive_readback_schema_hash_completeness", "status_declaration",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
                          cwd=ROOT, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def dependency_versions():
    # Distribution metadata only; do not import torch, SB3, Gym or environment.
    return {"Python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("gymnasium", "stable_baselines3", "torch", "numpy", "pandas")}}


def static_review(sources):
    """Fail closed on scientific imports/calls, dynamic dispatch and forbidden aliases.

    Deliberately scans attribute references as well as calls, so assigning a banned
    method to an alias is rejected. Source strings used to describe bans are allowed.
    This complements mandatory human source review, not a general Python sandbox.
    """
    allowed = {"__future__", "dataclasses", "json", "numpy", "argparse", "ast", "csv",
               "hashlib", "importlib.metadata", "io", "pathlib", "platform", "subprocess",
               "paper2_ppo_protocol_v1"}
    forbidden = {"learn", "step", "backward", "Paper2CleaningEnv", "Paper2EnvAssets",
                 "EpisodeSpec", "make_vec_env", "reset", "rollout", "collect_rollouts",
                 "episode_bank", "PPO", "VecNormalize", "eval", "exec", "compile",
                 "__import__", "getattr", "setattr", "import_module", "exec_module"}
    safe_initializers = {"tuple", "range", "RootPartition", "PPOCandidate", "SeedRegistry",
                         "SelectionContract", "EvaluationContract", "Protocol", "dataclass",
                         "Path", "resolve"}

    def check_initializers(nodes):
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = child.func.id if isinstance(child.func, ast.Name) else (
                        child.func.attr if isinstance(child.func, ast.Attribute) else "")
                    require(name in safe_initializers, f"Import-time call forbidden: {name}")

    for path, source in sources.items():
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                require(all(n in allowed for n in names), f"Unapproved import in {path}: {names}")
            if isinstance(node, ast.Attribute):
                require(node.attr not in forbidden, f"Forbidden attribute {path}:{node.lineno}")
            if isinstance(node, ast.Name):
                require(node.id not in forbidden, f"Forbidden name {path}:{node.lineno}")
        # Module scope is declarations only (plus explicit CLI main guard).
        for node in tree.body:
            require(isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
                                     ast.FunctionDef, ast.ClassDef, ast.Expr, ast.If)), "Unexpected module statement")
            if isinstance(node, ast.Expr):
                require(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), "Import-time expression")
            if isinstance(node, ast.If):
                require(path == SELF and ast.dump(node.test) == ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body), "Import-time branch")
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                check_initializers([node])
            if isinstance(node, ast.FunctionDef):
                check_initializers(node.decorator_list + node.args.defaults + [x for x in node.args.kw_defaults if x is not None])
            if isinstance(node, ast.ClassDef):
                check_initializers(node.decorator_list + node.bases)
                require(all(isinstance(x, (ast.Assign, ast.AnnAssign, ast.Expr)) for x in node.body), "Protocol classes must contain data only")
                check_initializers(node.body)
    return True


def contract_checks(p):
    """Independent pre-registration literals, mapped exactly to G7-G35."""
    q = protocol.protocol_payload(p)
    require(sha(json.dumps(q, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            == PROTOCOL_SHA256, "Full v1 metadata fingerprint mismatch; new version required")
    s, e = p.selection, p.evaluation
    roots = [set(part.roots) for part in p.partitions]
    seeds = p.seeds.development + p.seeds.final
    common = {"algorithm": "PPO", "policy": "MlpPolicy", "n_envs": 1, "batch_size": 256,
              "n_epochs": 10, "gamma": 1.0, "gae_lambda": .95, "clip_range": .2,
              "clip_range_vf": None, "normalize_advantage": True, "ent_coef": 0.0,
              "vf_coef": .5, "max_grad_norm": .5, "use_sde": False, "sde_sample_freq": -1,
              "target_kl": None, "learning_rate_schedule": "NONE", "clip_schedule": "NONE", "device": "cpu"}
    expected_specs = tuple((year, root, tid, 300, None) for year, root in
                           (("YEAR1", 1320001), ("YEAR2", 1330001)) for tid in range(300))
    # Local RNG metadata verification only: no environment is imported/constructed.
    import numpy as np
    substreams = []
    for seed in seeds:
        record = protocol.derive_rl_subseeds(seed)
        children = np.random.SeedSequence(seed).spawn(2)
        substreams.append(record == {"parent_seed": seed,
            "conversion": "int(child.generate_state(1, dtype=np.uint32)[0])",
            "children": [{"role": role, "spawn_key": [i], "integer_seed": int(child.generate_state(1, dtype=np.uint32)[0])}
                         for i, (role, child) in enumerate(zip(("learner", "training_schedule"), children))]})
    checks = {
        7: p.observations[0] == ("True-State", ("L_true", "sin_DOY", "cos_DOY")) and p.observation_dimensions == (("True-State", 3), ("Point", 3), ("UA", 4)) and p.role == "CAUSAL_FULL_STATE_RL_BENCHMARK",
        8: tuple((x.role, x.roots) for x in p.partitions) == (("TRAINING", (1320001, 1330001)), ("DEVELOPMENT", (1020001, 1030001)), ("FORMAL HELD-OUT", (1220001, 1230001))),
        9: all(not roots[i] & roots[j] for i in range(3) for j in range(i)),
        10: all(x.years == ("YEAR1", "YEAR2") and x.trajectory_ids == tuple(range(300)) and x.population_size == 600 for x in p.partitions),
        11: p.seeds.development == (510001, 510002),
        12: p.seeds.final == (520001, 520002, 520003, 520004, 520005),
        13: len(set(seeds)) == 7 and not set(seeds) & set.union(*roots),
        14: all(substreams) and p.seeds.derivation == "np.random.SeedSequence(rl_seed).spawn(2)" and p.seeds.child_order == ("learner", "training_schedule"),
        15: protocol.canonical_training_specs() == expected_specs and len(set(expected_specs)) == 600,
        16: dict(p.schedule) == dict(protocol.Protocol().schedule),
        17: dict(p.architecture) == {"policy": "MlpPolicy", "actor": (64, 64), "critic": (64, 64), "activation": "Tanh", "ortho_init": True},
        18: dict(p.common) == common and p.device == "cpu" and p.torch_num_threads == 1,
        19: [(c.name, c.learning_rate, c.n_steps) for c in p.candidates] == [("CONFIG_A", 3e-4, 2048), ("CONFIG_B", 1e-4, 2048), ("CONFIG_C", 3e-4, 4096), ("CONFIG_D", 1e-4, 4096)],
        20: p.development_budget == 491520 and all(491520 % c.n_steps == 0 for c in p.candidates),
        21: p.final_budget == 983040 and all(983040 % c.n_steps == 0 for c in p.candidates),
        22: s.primary_metric == "mean of the two seed-level development mean_J_total values; minimize" and s.evaluation == "final checkpoint of each run; full 600 DEVELOPMENT specs; deterministic=True",
        23: s.atol == 1e-12 and s.rtol == 0 and s.tie_breaks == ("lower worst_seed_dev_mean_J_total", "CONFIG_A", "CONFIG_B", "CONFIG_C", "CONFIG_D"),
        24: s.checkpoint == "FINAL TRAINING CHECKPOINT ONLY" and (s.development_checkpoint, s.final_checkpoint) == (491520, 983040),
        25: s.early_stopping is False and s.best_model_selection is False,
        26: (e.threshold_baseline, e.threshold_mean, e.tolerance_multiplier, e.gate_a_max, e.periodic_baseline, e.periodic_mean) == ("TRUE_THRESHOLD_0.05", 7.926837776021219, 1.03, 8.164642909301856, "PERIODIC_14", 10.126875738879782) and abs(e.threshold_mean * 1.03 - e.gate_a_max) < 1e-14 and e.gate_a == "mean of all 5 final seeds' DEVELOPMENT mean_J_total <= gate_a_max" and e.gate_b == "5/5 final seeds' DEVELOPMENT mean_J_total < periodic_mean",
        27: e.formal_prerequisites == ("selected config frozen", "all 5 final training runs complete", "final model SHA256 frozen", "development sanity gate passed", "training/development artifacts Git checkpointed") and e.formal_access == "each fixed model once on all 600 FORMAL HELD-OUT specs; deterministic=True",
        28: e.replication_unit == "RL TRAINING SEED" and e.replication_n == 5 and e.primary_report == ("each seed mean_J_total over 600 held-out trajectories", "5-seed mean", "5-seed std", "5-seed median", "5-seed range"),
        29: p.inheritance == ("training roots", "development roots", "formal roots", "RL seed sets", "schedule algorithm", "network architecture", "selected learning rate", "selected n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "entropy", "value coefficient", "max grad norm", "training budgets", "final-checkpoint-only rule", "deterministic evaluation rule"),
        30: p.width_ablation == "mandatory after UA main results: within-q50-bin width shuffle ablation; destroy width/sample-error pairing while preserving q50 regime and width marginal distribution as far as possible; not run in P2-2C-0",
        31: p.seeds.perception_seed is None and all(spec[-1] is None for spec in expected_specs),
        34: p.observation_normalization is False and p.reward_normalization is False and p.reward_scale == 1.0,
        35: p.gamma_rationale == "gamma=1.0 aligns PPO with finite 365-day UNDISCOUNTED annual total cost; not SB3 default gamma=0.99",
    }
    require(p.actions == ((0, "WAIT"), (1, "CLEAN")) and p.eta == .95 and p.lambda_c == .19925428989934618
            and (p.decision_reward_steps, p.natural_transitions) == (365, 364), "Frozen action/reward/episode contract")
    protocol.validate_protocol(p)
    return checks, q


def run(mode):
    require(mode in ("smoke", "formal"), "Explicit mode required")
    # Capture before validation; no frozen modules are imported or CSV data parsed.
    head = git("rev-parse", "HEAD")
    snapshots = {name: (ROOT / name).read_bytes() for name in (SELF, MODULE, CORE, PREDECESSOR)}
    snapshots.update({str((FROZEN / n).relative_to(ROOT)): (FROZEN / n).read_bytes() for n in FROZEN_HASHES})
    gates = {}

    def gate(number, condition):
        gates[f"G{number}"] = bool(condition)
        require(condition, f"G{number}: {GATE_NAMES[number - 1]}")

    gate(1, git("rev-parse", "--verify", f"{CHECKPOINT}^{{commit}}") == CHECKPOINT)
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    gate(2, git("rev-parse", f"{head}:{PREDECESSOR}") == git("hash-object", f"--path={PREDECESSOR}", PREDECESSOR) == PREDECESSOR_BLOB)
    gate(3, all(sha(snapshots[str((FROZEN / n).relative_to(ROOT))]) == digest for n, digest in FROZEN_HASHES.items()))
    previous = json.loads(snapshots[str((FROZEN / "audit_summary.json").relative_to(ROOT))])
    gate(4, previous.get("stage_pass") is True and previous.get("failed_gates") == [] and previous.get("scientific_status") == "HELDOUT_CLAIRVOYANT_ORACLE_EVALUATION_PASS_FROZEN" and previous.get("declaration") == "CLAIRVOYANT NON-CAUSAL ORACLE; NO DEPLOYMENT OR CAUSAL POLICY CLAIM; NO BASELINE RESELECTION")
    gate(5, git("rev-parse", f"{CHECKPOINT}:{CORE}") == git("rev-parse", f"{head}:{CORE}") == git("hash-object", f"--path={CORE}", CORE) == CORE_BLOB)
    versions = dependency_versions()
    gate(6, versions == {"Python": "3.11.15", "gymnasium": "1.2.3", "stable_baselines3": "2.7.1", "torch": "2.10.0+cu130", "numpy": "2.3.5", "pandas": "2.3.3"})
    gate(32, static_review({n: snapshots[n].decode("utf-8") for n in (SELF, MODULE)}))
    gate(33, gates["G32"] and all("paper2_gym" not in a.name for n in (SELF, MODULE)
         for node in ast.walk(ast.parse(snapshots[n])) if isinstance(node, ast.Import) for a in node.names))
    p = protocol.Protocol()
    checks, payload = contract_checks(p)
    for number, passed in checks.items():
        gate(number, passed)
    committed = {}
    for name in (SELF, MODULE):
        try:
            committed[name] = git("rev-parse", "--verify", f"{head}:{name}") == git("hash-object", f"--path={name}", name)
        except subprocess.CalledProcessError:
            committed[name] = False
    gate(38, mode != "formal" or all(committed.values()))
    status = FORMAL_STATUS if mode == "formal" else SMOKE_STATUS
    declaration = DECLARATION if mode == "formal" else "SMOKE ONLY; NO PPO PROTOCOL FREEZE; NO PPO TRAINING OR PERFORMANCE ACCESS"
    gate(40, (mode == "formal" and status == "TRUE_STATE_PPO_PROTOCOL_PASS_FROZEN" and declaration == "TRUE-STATE PPO PROTOCOL PRE-REGISTERED BEFORE PPO TRAINING; NO FORMAL PPO PERFORMANCE ACCESSED; NO POINT/UA HYPERPARAMETER RESELECTION") or (mode == "smoke" and status == "SMOKE_ONLY_NO_PPO_PROTOCOL_FREEZE"))
    seed_records = [protocol.derive_rl_subseeds(s) for s in p.seeds.development + p.seeds.final]
    documents = {
        "ppo_protocol.json": payload,
        "environment_role_partition.json": {"partitions": payload["partitions"], "uses": payload["partition_uses"]},
        "rl_seed_registry.json": {"contract": payload["seeds"], "substreams": seed_records},
        "training_schedule_contract.json": {"contract": dict(p.schedule), "n_envs": 1, "canonical_specs": protocol.canonical_training_specs()},
        "selection_contract.json": payload["selection"],
        "evaluation_contract.json": payload["evaluation"],
        "source_artifact_hashes.json": {"HEAD": head, "predecessor_checkpoint": CHECKPOINT,
            "dependencies": versions, "current_files_committed": committed,
            "sha256": {name: sha(data) for name, data in snapshots.items()}},
    }
    content = {name: encode(doc) for name, doc in documents.items()}
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("config", "learning_rate", "n_steps"))
    writer.writerows((c.name, c.learning_rate, c.n_steps) for c in p.candidates)
    content["ppo_candidate_configs.csv"] = buffer.getvalue().encode()
    index = {"hashes": {name: sha(data) for name, data in content.items()},
             "exclusions": {"output_hashes.json": "self-reference avoided; SHA256 in audit_summary",
                            "audit_summary.json": "published last; externally pin after acceptance"}}
    content["output_hashes.json"] = encode(index)
    destination = OUTPUT / mode
    # Exclusive directory claims prevent concurrent runs and overwriting results.
    # On failure retain incomplete evidence; no PASS summary is published.
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in content.items():
        with (destination / name).open("xb") as stream:
            stream.write(data)
    require({x.name for x in destination.iterdir()} == set(content), "Output inventory")
    for name, expected in content.items():
        actual = (destination / name).read_bytes()
        require(actual == expected, f"Readback SHA/schema mismatch: {name}")
        if name.endswith(".json"):
            require(json.loads(actual) == json.loads(expected), f"JSON schema/value mismatch: {name}")
    rows = list(csv.DictReader(io.StringIO((destination / "ppo_candidate_configs.csv").read_text())))
    require(rows == [{"config": c.name, "learning_rate": str(c.learning_rate), "n_steps": str(c.n_steps)} for c in p.candidates], "CSV schema/value mismatch")
    require(json.loads((destination / "output_hashes.json").read_bytes())["hashes"] == {n: sha((destination / n).read_bytes()) for n in content if n != "output_hashes.json"}, "Hash completeness")
    gate(39, len(content) == 9)
    gate(36, all((ROOT / name).read_bytes() == data for name, data in snapshots.items()))
    gate(37, git("rev-parse", "HEAD") == head)
    require(set(gates) == {f"G{i}" for i in range(1, 41)} and all(gates.values()), "All 40 gates required")
    summary = {"stage": "P2-2C-0-v1", "title": p.title, "mode": mode, "stage_pass": True,
               "scientific_status": status, "declaration": declaration, "gates": gates,
               "gate_names": {f"G{i}": name for i, name in enumerate(GATE_NAMES, 1)},
               "failed_gates": [], "HEAD": head, "current_files_committed": committed,
               "G38_scope": "required in formal; not required in smoke",
               "output_sha256": {n: sha(data) for n, data in content.items()},
               "execution": "metadata/provenance/static validation only; zero training, rollout, perception or optimizer steps"}
    summary_bytes = encode(summary)
    with (destination / "audit_summary.json").open("xb") as stream:
        stream.write(summary_bytes)
    require((destination / "audit_summary.json").read_bytes() == summary_bytes, "Final summary readback")
    require({x.name for x in destination.iterdir()} == set(content) | {"audit_summary.json"}, "Final inventory")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    args = parser.parse_args()
    print(json.dumps(run(args.mode), indent=2))


if __name__ == "__main__":
    main()
