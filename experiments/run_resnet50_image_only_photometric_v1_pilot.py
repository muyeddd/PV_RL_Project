"""Run the isolated photometric_v1 pilot for Fold 3 or Fold 4 only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_resnet50_image_only_date_grouped_cv as baseline_runner
from experiments import train_resnet50_image_only_photometric_v1_date_grouped as training


PILOT_OUTPUT_ROOT = training.PILOT_OUTPUT_ROOT
ALLOWED_FOLDS = training.ALLOWED_PILOT_FOLDS
PROTECTED_ROLES = training.PROTECTED_ROLES
SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 8
AMP = True
PRETRAINED = True
REQUIRED_FOLD_FILES = baseline_runner.REQUIRED_FOLD_FILES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run photometric_v1 pilot; protected Fold 1/2 are unavailable"
    )
    parser.add_argument("--fold", type=int, choices=ALLOWED_FOLDS, required=True)
    return parser.parse_args(argv)


def validate_pilot_fold(fold: int) -> None:
    if isinstance(fold, bool) or fold not in ALLOWED_FOLDS:
        raise ValueError("photometric_v1 pilot permits only Fold 3 and Fold 4")


def validate_output_root_isolation() -> None:
    expected = (
        PROJECT_ROOT
        / "outputs"
        / "date_grouped_v1"
        / "resnet50_image_only_photometric_v1"
        / "pilot"
    ).resolve()
    actual = PILOT_OUTPUT_ROOT.resolve()
    baseline_root = baseline_runner.FORMAL_OUTPUT_ROOT.resolve()
    if actual != expected:
        raise RuntimeError(f"Unexpected photometric_v1 output root: {actual}")
    if actual == baseline_root or baseline_root in actual.parents or actual in baseline_root.parents:
        raise RuntimeError(
            "photometric_v1 pilot output is not isolated from baseline formal_cv"
        )


def validate_frozen_config() -> None:
    validate_output_root_isolation()
    training.validate_single_factor_contract()
    config = training.load_training_config()
    expected = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "amp": AMP,
        "pretrained": PRETRAINED,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"photometric_v1 frozen config mismatch: {mismatches}")
    if tuple(config["allowed_pilot_folds"]) != ALLOWED_FOLDS:
        raise RuntimeError("photometric_v1 config does not restrict pilot to Fold 3/4")
    if tuple(config["protected_roles"]) != PROTECTED_ROLES:
        raise RuntimeError("photometric_v1 protected-role list changed")


def preflight_pilot_fold(fold: int) -> dict[str, Any]:
    validate_pilot_fold(fold)
    audit = training.preflight_pilot_fold(
        training.DEFAULT_MANIFEST,
        training.DEFAULT_IMAGE_ROOT,
        fold,
    )
    if audit["forbidden_role_counts"] != training.EXPECTED_FORBIDDEN_ROLE_COUNTS:
        raise RuntimeError("photometric_v1 protected-role preflight failed")
    return audit


def ensure_new_fold_directory(output_dir: Path) -> None:
    expected_parent = PILOT_OUTPUT_ROOT.resolve()
    actual = Path(output_dir).resolve()
    if actual.parent != expected_parent:
        raise RuntimeError("photometric_v1 fold output escaped the isolated pilot root")
    if actual.exists():
        raise FileExistsError(f"Refusing to overwrite photometric_v1 pilot: {actual}")


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def validate_pilot_outputs(fold: int, output_dir: Path) -> None:
    missing = [name for name in REQUIRED_FOLD_FILES if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"photometric_v1 Fold {fold} outputs incomplete: {missing}")
    config = _read_json(output_dir / "training_config_snapshot.json")
    final_metrics = _read_json(output_dir / "final_metrics.json")
    metadata = _read_json(output_dir / "run_metadata.json")
    for name, payload in (
        ("config", config),
        ("final_metrics", final_metrics),
        ("metadata", metadata),
    ):
        if payload.get("experiment_version") != training.TRAINING_VERSION:
            raise RuntimeError(f"{name} has wrong experiment version")
        if payload.get("final_test_accessed") is not False:
            raise RuntimeError(f"{name} does not prove final-test isolation")
        if payload.get("forbidden_roles_accessed") != []:
            raise RuntimeError(f"{name} records protected-role access")
    if config.get("PILOT_RUN") is not True or config.get("NOT_FOR_RESEARCH_METRICS") is not True:
        raise RuntimeError("photometric_v1 output is not marked as pilot-only")
    if config.get("image_preprocessing") != training.PREPROCESSING_DESCRIPTION:
        raise RuntimeError("photometric_v1 output recorded unexpected transforms")


def run_one_fold(fold: int) -> None:
    validate_frozen_config()
    audit = preflight_pilot_fold(fold)
    output_dir = training.expected_output_dir(fold)
    ensure_new_fold_directory(output_dir)
    print(
        f"photometric_v1 Fold {fold} preflight PASS: "
        f"train={len(audit['train_records'])}, "
        f"validation={len(audit['validation_records'])}, "
        f"forbidden_roles={audit['forbidden_role_counts']}, "
        f"output={output_dir}",
        flush=True,
    )
    args = training.parse_args(["--fold", str(fold)])
    training.run_training(args)
    validate_pilot_outputs(fold, output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_one_fold(args.fold)
    print(
        "photometric_v1 pilot complete; "
        "final_test_accessed=false; forbidden_roles_accessed=[]",
        flush=True,
    )


if __name__ == "__main__":
    main()
