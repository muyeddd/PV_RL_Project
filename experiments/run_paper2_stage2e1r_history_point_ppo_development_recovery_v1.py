
"""P2-2E-1R-v1: History-Point DEVELOPMENT recovery after empty-dir abort.

The predecessor P2-2E-1 directory contains zero files and only:
formal/, formal/runs/, formal/runs/seed_510001/.  It produced no model,
completion record, audit/failure marker, episode metrics, or performance result.

This recovery never modifies or reuses that directory.  It runs the exact
committed P2-2E-1 scientific implementation in a new immutable output directory
with the same History-Point observation, CONFIG_A, seeds, budget, populations,
Point perception, reward, physics, q50-history mask/shield and PPO settings.

Import is inert. Only --mode formal executes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

SELF = "experiments/run_paper2_stage2e1r_history_point_ppo_development_recovery_v1.py"
ORIGINAL = "experiments/run_paper2_stage2e1_history_point_ppo_development_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OLD_STAGE = BASE / "p2_2e_1_history_point_ppo_development_v1"
NEW_STAGE = BASE / "p2_2e_1r_history_point_ppo_development_recovery_v1"
NEW_OUTPUT = NEW_STAGE / "formal"

CHECKPOINT = "f5f2a92c4f635eca2cf0e2e74e3de83909416e30"
ORIGINAL_BLOB = "50db2943bffdae9729b7fd8e60a52cdea2faeaed"

STAGE = "P2-2E-1R-v1"
STATUS = "HISTORY_POINT_PPO_DEVELOPMENT_RECOVERY_CHARACTERIZATION_PASS"

EXPECTED_DIRS = (
    "formal",
    "formal/runs",
    "formal/runs/seed_510001",
)
DEVELOPMENT_SEEDS = (510001, 510002)
DEVELOPMENT_BUDGET = 491520
DEVELOPMENT_EPISODES = 600


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def encode(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def write_exclusive(path, data):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(data)
    require(path.read_bytes() == data, "Exact readback: " + path.name)


def old_inventory():
    require(OLD_STAGE.is_dir(), "Original P2-2E-1 directory must exist")
    dirs = sorted(
        x.relative_to(OLD_STAGE).as_posix()
        for x in OLD_STAGE.rglob("*")
        if x.is_dir()
    )
    files = sorted(
        x.relative_to(OLD_STAGE).as_posix()
        for x in OLD_STAGE.rglob("*")
        if x.is_file()
    )
    require(tuple(dirs) == EXPECTED_DIRS, "Original empty directory tree changed")
    require(files == [], "Original P2-2E-1 must contain zero files")

    payload = {
        "classification": "ABORTED_NON_EVALUABLE_ENGINEERING_RUN",
        "directories": dirs,
        "files": files,
        "file_count": 0,
        "scientific_performance_observed": False,
        "checkpoint_exists": False,
        "audit_summary_exists": False,
        "execution_failure_exists": False,
        "predecessor_reused_or_resumed": False,
    }
    payload["inventory_sha256"] = hashlib.sha256(
        encode({"directories": dirs, "files": files})
    ).hexdigest()
    return payload


def verify_sources(head):
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    require(
        git("rev-parse", f"{CHECKPOINT}:{ORIGINAL}") == ORIGINAL_BLOB
        and git("rev-parse", f"{head}:{ORIGINAL}") == ORIGINAL_BLOB,
        "Original P2-2E-1 implementation changed",
    )
    require(
        git("rev-parse", "--verify", f"{head}:{SELF}")
        == git("hash-object", f"--path={SELF}", SELF)
        == git("rev-parse", f":{SELF}"),
        "Recovery source must be committed unchanged",
    )


def static_contract():
    source = (ROOT / SELF).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attrs = {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    require(
        "train_masked_final_checkpoint" not in attrs
        and "build_history_point_model" not in attrs
        and "evaluate_history_point" not in attrs
        and "learn" not in attrs
        and "save" not in attrs
        and "load" not in attrs,
        "Recovery cannot locally reimplement or tune scientific runtime",
    )
    require("execute_formal" in attrs, "Must delegate to original execute_formal")

    import run_paper2_stage2e1_history_point_ppo_development_v1 as original
    import paper2_temporal_observation_v1 as temporal

    require(all(original.static_contract().values()), "Original contract")
    require(
        original.DEVELOPMENT_SEEDS == DEVELOPMENT_SEEDS
        and original.DEVELOPMENT_BUDGET == DEVELOPMENT_BUDGET
        and original.DEVELOPMENT_EPISODES == DEVELOPMENT_EPISODES,
        "Exact original seeds/budget/population",
    )
    c = temporal.temporal_contract()
    require(
        c["modes"][temporal.HISTORY_POINT]["features"]
        == [
            "q50",
            "delta_q50",
            "tau_since_last_clean_or_episode_start",
            "sin_DOY",
            "cos_DOY",
        ]
        and c["modes"][temporal.HISTORY_POINT]["dimension"] == 5
        and c["pre_episode_clean_age_imputed"] is False
        and c["future_information_used"] is False
        and c["latent_state_used"] is False
        and c["safety_rule_changed"] is False,
        "Frozen History-Point contract",
    )
    require(OLD_STAGE != NEW_STAGE, "Recovery output must be distinct")

    return {
        "delegates_exact_original_P2_2E_1_scientific_runtime": True,
        "no_local_scientific_reimplementation_or_tuning": True,
        "original_P2_2E_1_static_contract_all_true": True,
        "History_Point_feature_contract_unchanged": True,
        "CONFIG_A_seeds_budget_population_unchanged": True,
        "distinct_immutable_recovery_output": True,
        "original_empty_stage_read_only_contract": True,
        "performance_remains_descriptive_not_a_pass_gate": True,
    }


def configure_original(original, predecessor):
    """Change only stage/output/provenance metadata; not scientific logic."""
    original.STAGE_DIRECTORY = NEW_STAGE
    original.OUTPUT = NEW_OUTPUT
    original.STAGE = STAGE
    original.STATUS = STATUS
    original.SELF = SELF
    original.PINNED_BLOBS = {
        **dict(original.PINNED_BLOBS),
        ORIGINAL: ORIGINAL_BLOB,
    }

    original_encode = original.encode
    original_write = original.write_exclusive

    def recovery_encode(value):
        if isinstance(value, dict):
            x = dict(value)
            if (
                "new_source_git_blob" in x
                and "P2_2E_0R_audit_summary_sha256" in x
            ):
                x["recovery_predecessor_evidence"] = predecessor
                x["recovery_execution"] = {
                    "scientific_runtime_authority": ORIGINAL,
                    "scientific_runtime_authority_blob": ORIGINAL_BLOB,
                    "predecessor_output_reused": False,
                    "predecessor_checkpoint_reused": False,
                    "seeds_changed": False,
                    "budget_changed": False,
                    "History_Point_observation_changed": False,
                    "mask_or_shield_changed": False,
                    "reward_or_physics_changed": False,
                    "PPO_hyperparameters_changed": False,
                }
            if (
                x.get("stage") == STAGE
                and x.get("stage_pass") is True
                and "output_sha256" in x
            ):
                x["recovery_predecessor_evidence"] = predecessor
                x["recovery_classification"] = (
                    "NEW_IMMUTABLE_EXECUTION_AFTER_EMPTY_NON_EVALUABLE_ABORT"
                )
                x["original_P2_2E_1_output_reused"] = False
                x["original_P2_2E_1_checkpoint_reused"] = False
            value = x
        return original_encode(value)

    def recovery_write(path, data):
        # The old empty predecessor must still be unchanged immediately before
        # the final PASS marker is published.
        if Path(path).name == "audit_summary.json":
            require(
                old_inventory() == predecessor,
                "Original empty predecessor changed during recovery",
            )
        return original_write(path, data)

    original.encode = recovery_encode
    original.write_exclusive = recovery_write


def run_formal():
    require(
        not NEW_STAGE.exists(),
        "Exclusive P2-2E-1R directory; no overwrite/resume/retry",
    )
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean worktree required",
    )

    head = git("rev-parse", "HEAD")
    verify_sources(head)
    predecessor = old_inventory()
    static = static_contract()

    import run_paper2_stage2e1_history_point_ppo_development_v1 as original
    import run_paper2_stage2d6a_masked_ua_ppo_development_v1 as stage6a
    import run_paper2_stage2e0_temporal_partial_observability_design_freeze_v1 as stage0
    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as versions
    import paper2_ppo_protocol_v1 as frozen
    import paper2_masked_perception_ppo_runner_v1_1 as runner
    import paper2_temporal_observation_v1 as temporal

    stage0r_sha = original.verify_0r_authority()
    point_ref = stage6a.verify_point5b(head)
    stage6a_sha = stage0.verify_6a()

    ua_rows = original.read_csv(
        original.UA6A / "masked_ua_development_episode_metrics.csv"
    )
    require(len(ua_rows) == 1200, "Frozen static-UA population")
    static_ua = {}
    for row in ua_rows:
        key = (
            int(row["parent_rl_seed"]),
            row["year"],
            int(row["trajectory_id"]),
        )
        require(key not in static_ua, "Unique static-UA key")
        static_ua[key] = row
    require(len(static_ua) == 1200, "Complete static-UA map")

    p = runner.get_protocol()
    config = runner.candidate("CONFIG_A")
    base_versions = versions.dependency_versions()
    masked_versions = runner.masked_dependency_versions()
    api_contract = runner.verify_maskable_api_contract()

    require(
        tuple(p.seeds.development) == DEVELOPMENT_SEEDS
        and p.development_budget == DEVELOPMENT_BUDGET
        and config.name == runner.inherited_config_id() == "CONFIG_A",
        "Frozen CONFIG_A DEVELOPMENT contract",
    )

    assets = runner.load_assets()
    assets.ensure_perception()
    development = runner.partition("DEVELOPMENT")
    specs = tuple(
        (year, tid)
        for year in development.years
        for tid in development.trajectory_ids
    )
    require(
        len(specs)
        == len(set(specs))
        == development.population_size
        == DEVELOPMENT_EPISODES,
        "Full 600-episode DEVELOPMENT population",
    )

    self_blob = git("rev-parse", "--verify", f"{head}:{SELF}")

    NEW_STAGE.mkdir(parents=True, exist_ok=False)
    NEW_OUTPUT.mkdir(exist_ok=False)

    configure_original(original, predecessor)

    try:
        return original.execute_formal(
            runner=runner,
            temporal=temporal,
            frozen=frozen,
            p=p,
            config=config,
            assets=assets,
            specs=specs,
            head=head,
            self_blob=self_blob,
            static=static,
            stage0r_sha=stage0r_sha,
            stage6a_sha=stage6a_sha,
            point_ref=point_ref,
            static_ua=static_ua,
            base_versions=base_versions,
            masked_versions=masked_versions,
            api_contract=api_contract,
        )
    except Exception as exc:
        failure = NEW_OUTPUT / "execution_failure.json"
        if NEW_OUTPUT.exists() and not failure.exists():
            write_exclusive(
                failure,
                encode(
                    {
                        "stage": STAGE,
                        "stage_pass": False,
                        "HEAD": head,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "predecessor_evidence": predecessor,
                        "action":
                            "STOP; PRESERVE OLD EMPTY PREDECESSOR AND RECOVERY EVIDENCE; NO RETRY/RESEED/HISTORY-REDESIGN/MASK/SHIELD/REWARD/PERCEPTION/PPO TUNING",
                    }
                ),
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
