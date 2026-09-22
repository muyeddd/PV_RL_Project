"""P2-2D-4B-v1: read-only safe-action distinction decomposition audit.

Reads only frozen P2-2D-4AR formal outputs. It does not run environments,
load/train PPO, or access protected roles. Scientific outcomes are descriptive
and never structural PASS gates. Import is inert; only --mode formal executes.
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

ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage2d4b_safe_action_distinction_decomposition_audit_v1.py"
PREDECESSOR_SOURCE = "experiments/run_paper2_stage2d4ar_shield_delegation_action_identifiability_recovery_v1.py"
PREDECESSOR_HEAD = "9c333d0b5694c4796dcaf24a43728a0a96d27373"
PREDECESSOR_SOURCE_BLOB = "a778fa61aa7559353ed7e655d04298c4c8724f86"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
PREDECESSOR = BASE / "p2_2d_4ar_shield_delegation_action_identifiability_recovery_v1/formal"
STAGE_DIRECTORY = BASE / "p2_2d_4b_safe_action_distinction_decomposition_audit_v1"
OUTPUT = STAGE_DIRECTORY / "formal"
STAGE = "P2-2D-4B-v1"
STATUS = "SAFE_ACTION_DISTINCTION_DECOMPOSITION_AUDIT_COMPLETE"
ATOL = 1e-12
N = 163

EXPECTED_OUTPUT_SHA256 = {
    "protocol_manifest.json": "33394579f8bf3e2f12aeb3456d0e0145d11413550e862c41fcf59bb1c398831a",
    "failed_predecessor_provenance.json": "0baee36e359d3d0fe07d4ca2b5125e1939cc2618d758ed18d2bd75f1640a5bd8",
    "anchor_eligibility_summary.json": "05ddc410db2524cbc21394f14e692b0014cf743be665b2c01121fcd279c7b6e5",
    "predecessor_collapse_audit.json": "138e51935da250ff1c0a9cb09049fcff5c7cf3ffba0c049f876b596d719f9181",
    "mechanism_summary.json": "13d8208bac22f42ed0f26aad077713f01989acf2d72c92647d64a7282772e85e",
    "baseline_scan.csv": "01277a02280bc5b35a67e2d573f9fd55cf9beda564026d98434eb7840e429164",
    "delegation_alias_witnesses.csv": "56ca80d612a85da7d53f499032626e426f52ab77929c0af3ad86467ecc34f7f3",
    "safe_action_counterfactuals.csv": "5a147ca25d979088ce1ad4bc3af8d4bde0f87cf4cd0f543164452673dfc5c8f8",
    "output_hashes.json": "efd7dcabb21fee9bec7a1270d37920fe7e98b251925f3bea7ee76b4b7e8ad0ac",
}

EXPECTED_COLUMNS = (
    "year", "trajectory_id", "anchor_label", "anchor_day", "first_forced_day",
    "days_before_first_forced", "q50_float32", "L_pre_audit_only",
    "belief_lower", "belief_upper", "WAIT_anchor_cost", "CLEAN_anchor_cost",
    "immediate_cost_gap_CLEANminusWAIT", "WAIT_L_post", "CLEAN_L_post",
    "WAIT_L_next", "CLEAN_L_next", "immediate_action_distinguishable",
    "physical_action_distinguishable", "economic_action_distinguishable",
    "remaining_cost_WAITbranch", "remaining_cost_CLEANbranch",
    "remaining_horizon_gap_CLEANminusWAIT", "continuation_classification",
    "WAITbranch_total_interventions", "CLEANbranch_total_interventions",
    "WAITbranch_total_executed_clean", "CLEANbranch_total_executed_clean",
)
CLASSIFICATIONS = ("CLEAN_BENEFICIAL", "WAIT_BENEFICIAL", "COST_TIE_WITHIN_ATOL")
CATEGORIES = ("PHYSICAL_AND_ECONOMIC", "PHYSICAL_ONLY", "ECONOMIC_ONLY", "NEITHER_PHYSICAL_NOR_ECONOMIC")
ANCHORS = ("MID_PRE_FORCE", "PRE_FORCE")


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


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def csv_bytes(rows, fields):
    require(all(set(r) == set(fields) for r in rows), "Consistent CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def write_exclusive(path, data):
    path = Path(path)
    require(path.parent.resolve() == OUTPUT.resolve(), "Writes only to P2-2D-4B formal")
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, "Exact readback: " + path.name)


def parse_bool(value, name):
    require(value in ("True", "False"), "Exact bool: " + name)
    return value == "True"


def f(value, name):
    try:
        x = float(value)
    except Exception as exc:
        raise RuntimeError("Float parse: " + name) from exc
    require(x == x and abs(x) != float("inf"), "Finite float: " + name)
    return x


def nf(value, name):
    return None if value in ("", "None", "null", "NULL") else f(value, name)


def classify(gap):
    if gap < -ATOL: return "CLEAN_BENEFICIAL"
    if gap > ATOL: return "WAIT_BENEFICIAL"
    return "COST_TIE_WITHIN_ATOL"


def category(physical, economic):
    if physical and economic: return "PHYSICAL_AND_ECONOMIC"
    if physical: return "PHYSICAL_ONLY"
    if economic: return "ECONOMIC_ONLY"
    return "NEITHER_PHYSICAL_NOR_ECONOMIC"


def verify_predecessor(head):
    require(PREDECESSOR.is_dir(), "P2-2D-4AR formal exists")
    require({p.name for p in PREDECESSOR.iterdir() if p.is_file()} == set(EXPECTED_OUTPUT_SHA256) | {"audit_summary.json"}, "Exact predecessor inventory")
    audit = read_json(PREDECESSOR / "audit_summary.json")
    mech = read_json(PREDECESSOR / "mechanism_summary.json")
    coverage = read_json(PREDECESSOR / "anchor_eligibility_summary.json")
    require(
        audit["stage"] == "P2-2D-4AR-v1" and audit["mode"] == "formal"
        and audit["stage_pass"] is True and audit["HEAD"] == PREDECESSOR_HEAD
        and audit["PPO_training_runs"] == 0 and audit["PPO_model_loads"] == 0
        and audit["UA_training_runs"] == 0 and audit["development_episode_count"] == 600
        and audit["safe_action_counterfactual_count"] == N
        and audit["delegation_aliasing_confirmed"] is True
        and audit["preemptive_clean_opportunity_observed"] is True
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0 and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-2D-4AR audit",
    )
    require(
        coverage["total_development_episodes"] == 600
        and coverage["zero_anchor_episode_count"] == 484
        and coverage["one_anchor_episode_count"] == 69
        and coverage["two_anchor_episode_count"] == 47
        and coverage["episodes_with_at_least_one_anchor"] == 116
        and coverage["counterfactual_anchor_count"] == N,
        "Accepted anchor coverage",
    )
    scf = mech["safe_action_counterfactual"]
    require(scf["count"] == N and scf["clean_beneficial_count"] == 63 and scf["wait_beneficial_count"] == 90 and scf["tie_count"] == 10 and scf["both_safe_actions_physically_economically_distinguishable"] is False, "Accepted mechanism summary")
    require(audit["output_sha256"] == EXPECTED_OUTPUT_SHA256, "Exact predecessor hash registry")
    for name, digest in EXPECTED_OUTPUT_SHA256.items():
        require(sha_file(PREDECESSOR / name) == digest, "Predecessor hash: " + name)
    git("merge-base", "--is-ancestor", PREDECESSOR_HEAD, head)
    require(git("rev-parse", f"{PREDECESSOR_HEAD}:{PREDECESSOR_SOURCE}") == PREDECESSOR_SOURCE_BLOB and git("rev-parse", f"{head}:{PREDECESSOR_SOURCE}") == PREDECESSOR_SOURCE_BLOB, "P2-2D-4AR source unchanged")
    return {"HEAD": PREDECESSOR_HEAD, "source_git_blob": PREDECESSOR_SOURCE_BLOB, "output_sha256": EXPECTED_OUTPUT_SHA256, "read_only": True}


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom): imported.add(node.module or "")
    banned = ("stable_baselines3", "sb3_contrib", "torch", "gym", "gymnasium", "paper2_", "run_paper2_")
    require(not any(name.startswith(prefix) for name in imported for prefix in banned), "No PPO/environment/project-runtime imports")
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    require(not ({"learn", "predict", "step", "reset", "load", "save"} & attrs), "No model/environment runtime calls")
    require("safe_action_counterfactuals.csv" in source and "physical_action_distinguishable" in source and "economic_action_distinguishable" in source and "remaining_horizon_gap_CLEANminusWAIT" in source, "Required decomposition fields")
    require("scientific_results_are_not_pass_gates" in source, "Diagnostic-not-gate contract")
    return {
        "read_only_predecessor_outputs_only": True,
        "no_PPO_or_environment_runtime": True,
        "no_project_runner_imports": True,
        "no_protected_data_access": True,
        "distinction_decomposition_fields_present": True,
        "scientific_results_are_not_pass_gates": True,
    }


def audit_rows():
    fields, rows = read_csv(PREDECESSOR / "safe_action_counterfactuals.csv")
    require(fields == EXPECTED_COLUMNS and len(rows) == N, "Exact counterfactual schema/population")
    seen, out = set(), []
    for row in rows:
        key = (row["year"], int(row["trajectory_id"]), row["anchor_label"], int(row["anchor_day"]))
        require(key not in seen, "Unique anchor key"); seen.add(key)
        anchor = row["anchor_label"]
        require(anchor in ANCHORS, "Known anchor")
        ad, fd = int(row["anchor_day"]), int(row["first_forced_day"])
        require(0 <= ad < fd < 364, "Anchor before forced day")
        immediate = parse_bool(row["immediate_action_distinguishable"], "immediate")
        physical = parse_bool(row["physical_action_distinguishable"], "physical")
        economic = parse_bool(row["economic_action_distinguishable"], "economic")
        igap = f(row["immediate_cost_gap_CLEANminusWAIT"], "immediate_gap")
        require(economic == (abs(igap) > ATOL), "Economic flag recomputed")
        wp, cp = f(row["WAIT_L_post"], "WAIT_L_post"), f(row["CLEAN_L_post"], "CLEAN_L_post")
        wn, cn = nf(row["WAIT_L_next"], "WAIT_L_next"), nf(row["CLEAN_L_next"], "CLEAN_L_next")
        require(physical == (wp != cp or wn != cn), "Physical flag recomputed")
        require(not (physical or economic) or immediate, "Physical/economic implies immediate")
        hgap = f(row["remaining_horizon_gap_CLEANminusWAIT"], "horizon_gap")
        cls = row["continuation_classification"]
        require(cls in CLASSIFICATIONS and cls == classify(hgap), "Classification recomputed")
        horizon = abs(hgap) > ATOL
        require(horizon == (cls != "COST_TIE_WITHIN_ATOL"), "Horizon distinction identity")
        out.append({
            "year": row["year"], "trajectory_id": int(row["trajectory_id"]),
            "anchor_label": anchor, "anchor_day": ad, "first_forced_day": fd,
            "days_before_first_forced": int(row["days_before_first_forced"]),
            "immediate_action_distinguishable": immediate,
            "physical_action_distinguishable": physical,
            "economic_action_distinguishable": economic,
            "joint_physical_and_economic_distinguishable": physical and economic,
            "distinction_category": category(physical, economic),
            "immediate_cost_gap_CLEANminusWAIT": igap,
            "horizon_cost_distinguishable": horizon,
            "remaining_horizon_gap_CLEANminusWAIT": hgap,
            "continuation_classification": cls,
        })
    require(sum(r["anchor_label"] == "MID_PRE_FORCE" for r in out) == 47 and sum(r["anchor_label"] == "PRE_FORCE" for r in out) == 116, "Anchor populations")
    require(sum(r["continuation_classification"] == "CLEAN_BENEFICIAL" for r in out) == 63 and sum(r["continuation_classification"] == "WAIT_BENEFICIAL" for r in out) == 90 and sum(r["continuation_classification"] == "COST_TIE_WITHIN_ATOL" for r in out) == 10, "Economic class counts")
    return out


def stats(rows):
    n = len(rows)
    c = lambda pred: sum(bool(pred(r)) for r in rows)
    return {
        "count": n,
        "immediate_distinguishable_count": c(lambda r: r["immediate_action_distinguishable"]),
        "physical_distinguishable_count": c(lambda r: r["physical_action_distinguishable"]),
        "economic_distinguishable_count": c(lambda r: r["economic_action_distinguishable"]),
        "joint_physical_and_economic_count": c(lambda r: r["joint_physical_and_economic_distinguishable"]),
        "physical_only_count": c(lambda r: r["distinction_category"] == "PHYSICAL_ONLY"),
        "economic_only_count": c(lambda r: r["distinction_category"] == "ECONOMIC_ONLY"),
        "neither_count": c(lambda r: r["distinction_category"] == "NEITHER_PHYSICAL_NOR_ECONOMIC"),
        "horizon_cost_distinguishable_count": c(lambda r: r["horizon_cost_distinguishable"]),
        "clean_beneficial_count": c(lambda r: r["continuation_classification"] == "CLEAN_BENEFICIAL"),
        "wait_beneficial_count": c(lambda r: r["continuation_classification"] == "WAIT_BENEFICIAL"),
        "tie_count": c(lambda r: r["continuation_classification"] == "COST_TIE_WITHIN_ATOL"),
    }


def with_fractions(d):
    n = d["count"]
    out = dict(d)
    for key, value in list(d.items()):
        if key.endswith("_count") and key != "count":
            out[key.replace("_count", "_fraction")] = value / n if n else None
    return out


def crosstab(rows):
    table = []
    for anchor in ("ALL",) + ANCHORS:
        ar = rows if anchor == "ALL" else [r for r in rows if r["anchor_label"] == anchor]
        for cat in CATEGORIES:
            cr = [r for r in ar if r["distinction_category"] == cat]
            for cls in CLASSIFICATIONS:
                count = sum(r["continuation_classification"] == cls for r in cr)
                table.append({
                    "anchor_label": anchor, "distinction_category": cat,
                    "continuation_classification": cls, "count": count,
                    "category_population": len(cr),
                    "fraction_within_category": count / len(cr) if cr else None,
                })
    return table


def run_formal():
    require(not STAGE_DIRECTORY.exists(), "Exclusive P2-2D-4B directory; no overwrite/resume/retry")
    require(git("status", "--porcelain", "--untracked-files=all") == "", "Committed unchanged clean worktree required")
    head = git("rev-parse", "HEAD")
    predecessor = verify_predecessor(head)
    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")
    require(self_blob == git("hash-object", f"--path={SELF}", SELF) == git("rev-parse", f":{SELF}"), "P2-2D-4B source committed unchanged")
    static = static_contract()
    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False); OUTPUT.mkdir(exist_ok=False)
    try:
        rows = audit_rows()
        overall = with_fractions(stats(rows))
        by_anchor = {a: with_fractions(stats([r for r in rows if r["anchor_label"] == a])) for a in ANCHORS}
        by_class = {c: with_fractions(stats([r for r in rows if r["continuation_classification"] == c])) for c in CLASSIFICATIONS}
        by_cat = {c: with_fractions(stats([r for r in rows if r["distinction_category"] == c])) for c in CATEGORIES}
        table = crosstab(rows)
        prior_all_joint = overall["joint_physical_and_economic_count"] == N
        require(prior_all_joint is False, "Must explain predecessor false universal joint predicate")

        summary = {
            "stage": STAGE,
            "population": {"counterfactual_anchor_count": N, "anchor_eligible_episode_count": 116, "total_development_episode_count": 600, "MID_PRE_FORCE_count": 47, "PRE_FORCE_count": 116},
            "overall": overall,
            "by_anchor_label": by_anchor,
            "by_continuation_classification": by_class,
            "by_distinction_category": by_cat,
            "predecessor_route_predicate_decomposition": {
                "predecessor_all_rows_joint_physical_and_economic_requirement": True,
                "prior_all_rows_joint_requirement_pass": prior_all_joint,
                "joint_failure_row_count": N - overall["joint_physical_and_economic_count"],
                "physical_failure_row_count": N - overall["physical_distinguishable_count"],
                "economic_failure_row_count": N - overall["economic_distinguishable_count"],
                "neither_failure_row_count": overall["neither_count"],
                "interpretation": "Universal all-row predicate decomposition only; P2-2D-4B makes no masked-PPO launch decision.",
            },
            "scientific_diagnostic": {
                "any_clean_beneficial_anchor": overall["clean_beneficial_count"] > 0,
                "any_horizon_cost_distinguishable_anchor": overall["horizon_cost_distinguishable_count"] > 0,
                "all_immediate_actions_distinguishable": overall["immediate_distinguishable_count"] == N,
                "all_physical_actions_distinguishable": overall["physical_distinguishable_count"] == N,
                "all_immediate_economic_actions_distinguishable": overall["economic_distinguishable_count"] == N,
                "all_joint_physical_and_economic": prior_all_joint,
                "role": "DESCRIPTIVE READ-ONLY DEVELOPMENT DECOMPOSITION; NOT A PASS GATE AND NOT AN ALGORITHM-SELECTION RULE",
            },
            "claim_boundary": "Read-only decomposition of frozen P2-2D-4AR paired fixed-continuation counterfactuals; no optimal-Q, final-seed or formal-held-out claim.",
        }

        gates = {
            "predecessor_4AR_exact_and_hash_pinned": True,
            "static_read_only_contract": all(static.values()),
            "exact_163_counterfactual_rows": len(rows) == N,
            "classification_recomputed_exact": True,
            "physical_flag_recomputed_exact": True,
            "economic_flag_recomputed_exact": True,
            "anchor_population_exact": True,
            "scientific_results_are_not_pass_gates": True,
            "zero_PPO_training_or_loading": True,
            "zero_environment_runtime": True,
            "zero_protected_access": True,
        }
        require(all(gates.values()), "All structural gates required")

        protocol = {
            "stage": STAGE, "mode": "formal", "role": "SAFE-ACTION DISTINCTION DECOMPOSITION",
            "source_stage": "P2-2D-4AR-v1", "input_population": N,
            "classification_atol": ATOL,
            "runtime": {"PPO_training_runs": 0, "PPO_model_loads": 0, "environment_resets": 0, "environment_steps": 0, "TRAINING_access_count": 0, "FORMAL_HELDOUT_access_count": 0, "RANDOM_TEST_access_count": 0, "SEALED_DATES_access_count": 0},
            "scientific_outcome_as_structural_gate": False,
            "masked_PPO_start_decision_in_this_stage": False,
        }
        provenance = {"HEAD": head, "self_git_blob": self_blob, "predecessor": predecessor, "predecessor_read_only": True}
        row_fields = tuple(rows[0].keys())
        table_fields = ("anchor_label", "distinction_category", "continuation_classification", "count", "category_population", "fraction_within_category")
        content = {
            "protocol_manifest.json": encode(protocol),
            "predecessor_provenance.json": encode(provenance),
            "distinction_summary.json": encode(summary),
            "distinction_row_audit.csv": csv_bytes(rows, row_fields),
            "distinction_crosstab.csv": csv_bytes(table, table_fields),
        }
        for name, data in content.items(): write_exclusive(OUTPUT / name, data)
        hashes = {name: sha_bytes(data) for name, data in content.items()}
        output_hashes = encode({"hashes": hashes, "exclusions": {"output_hashes.json": "SHA256 pinned in audit_summary; avoid self-reference", "audit_summary.json": "Published last PASS marker"}})
        write_exclusive(OUTPUT / "output_hashes.json", output_hashes)
        require({p.name for p in OUTPUT.iterdir() if p.is_file()} == set(content) | {"output_hashes.json"}, "Exact pre-audit inventory")
        require(git("rev-parse", "HEAD") == head and git("status", "--porcelain", "--untracked-files=all") == "", "HEAD/worktree unchanged")
        verify_predecessor(head)
        audit = {
            "stage": STAGE, "mode": "formal", "stage_pass": True, "scientific_status": STATUS,
            "HEAD": head, "gates": gates, "failed_gates": [], "counterfactual_anchor_count": N,
            "immediate_distinguishable_count": overall["immediate_distinguishable_count"],
            "physical_distinguishable_count": overall["physical_distinguishable_count"],
            "economic_distinguishable_count": overall["economic_distinguishable_count"],
            "joint_physical_and_economic_count": overall["joint_physical_and_economic_count"],
            "horizon_cost_distinguishable_count": overall["horizon_cost_distinguishable_count"],
            "clean_beneficial_count": overall["clean_beneficial_count"], "wait_beneficial_count": overall["wait_beneficial_count"], "tie_count": overall["tie_count"],
            "prior_all_rows_joint_requirement_pass": prior_all_joint,
            "PPO_training_runs": 0, "PPO_model_loads": 0, "environment_runtime_calls": 0,
            "formal_RL_access_count": 0, "formal_perception_seed_access_count": 0, "RANDOM_TEST_access_count": 0, "SEALED_DATES_access_count": 0,
            "output_sha256": {**hashes, "output_hashes.json": sha_bytes(output_hashes)},
            "audit_summary_published_last": True,
        }
        write_exclusive(OUTPUT / "audit_summary.json", encode(audit))
        return audit
    except Exception as exc:
        if OUTPUT.exists() and not (OUTPUT / "execution_failure.json").exists():
            write_exclusive(OUTPUT / "execution_failure.json", encode({"stage": STAGE, "stage_pass": False, "error_type": type(exc).__name__, "message": str(exc), "action": "STOP; PRESERVE EVIDENCE; NO MASKED-PPO START, NO TUNING/RETRY", "HEAD": head}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
