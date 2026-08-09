"""Run formal four-fold Image-only sqrt-L-balanced date-grouped CV."""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_resnet50_image_only_date_grouped_cv as baseline_runner
from experiments import train_resnet50_image_only_lbalanced_date_grouped as training


FORMAL_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "resnet50_image_only_lbalanced"
    / "formal_cv"
)
BASELINE_OUTPUT_ROOT = baseline_runner.FORMAL_OUTPUT_ROOT
FOLDS = (1, 2, 3, 4)
SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 8
AMP = True
PRETRAINED = True
EXPECTED_MANIFEST_SHA256 = baseline_runner.EXPECTED_MANIFEST_SHA256
REQUIRED_FOLD_FILES = baseline_runner.REQUIRED_FOLD_FILES
SUMMARY_NUMERIC_FIELDS = baseline_runner.SUMMARY_NUMERIC_FIELDS
CSV_FIELDS = baseline_runner.CSV_FIELDS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal date_grouped_v1 Image-only sqrt-L-balanced CV"
    )
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=4)
    args = parser.parse_args(argv)
    if args.start_fold not in FOLDS or args.end_fold not in FOLDS:
        parser.error("--start-fold and --end-fold must be between 1 and 4")
    if args.start_fold > args.end_fold:
        parser.error("--start-fold must not exceed --end-fold")
    return args


def validate_output_root_isolation() -> None:
    expected = (
        PROJECT_ROOT
        / "outputs"
        / "date_grouped_v1"
        / "resnet50_image_only_lbalanced"
        / "formal_cv"
    ).resolve()
    actual = FORMAL_OUTPUT_ROOT.resolve()
    if actual != expected:
        raise RuntimeError(f"Unexpected L-balanced output root: {actual}")
    if actual == BASELINE_OUTPUT_ROOT.resolve() or actual.is_relative_to(
        BASELINE_OUTPUT_ROOT.resolve()
    ):
        raise RuntimeError("L-balanced output overlaps the Image-only baseline")


def validate_frozen_config() -> None:
    validate_output_root_isolation()
    training.validate_single_factor_contract()
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
    actual = training.load_training_config()
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen L-balanced config mismatch: {mismatches}")
    source = inspect.getsource(training)
    forbidden_tokens = ("torch" + ".load", "load_state_dict(" + "torch" + ".load")
    if any(token in source for token in forbidden_tokens):
        raise RuntimeError("Formal L-balanced source can read a project checkpoint")


def preflight_formal_fold(fold: int) -> dict[str, Any]:
    audit = training.preflight_fold(
        training.DEFAULT_MANIFEST, training.DEFAULT_IMAGE_ROOT, fold
    )
    expected_train, expected_validation = training.EXPECTED_FOLD_COUNTS[fold]
    actual_counts = (len(audit["train_records"]), len(audit["validation_records"]))
    if actual_counts != (expected_train, expected_validation):
        raise ValueError(
            f"Fold {fold} count mismatch: expected "
            f"{expected_train}/{expected_validation}, got {actual_counts}"
        )
    if audit["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Fold {fold} manifest SHA256 mismatch")
    expected_forbidden = {
        "cp_calibration": 0,
        "decision_development": 0,
        "final_test": 0,
    }
    if audit["forbidden_role_counts"] != expected_forbidden:
        raise ValueError(f"Fold {fold} forbidden-role audit failed")
    if set(audit["train_dates"]) & set(audit["validation_dates"]):
        raise ValueError(f"Fold {fold} train/validation dates overlap")
    pretrained = training.baseline._pretrained_provenance(True)
    if not pretrained["pretrained_cache_present_before_init"]:
        raise FileNotFoundError("ImageNet ResNet50 weights are absent from the cache")
    return audit


def ensure_new_fold_directory(output_dir: Path) -> None:
    baseline_runner.ensure_new_fold_directory(output_dir)


def _validate_sampling_provenance(fold: int, config: dict[str, Any]) -> None:
    provenance = config.get("sampling_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"Fold {fold} lacks sampling provenance")
    expected_counts = dict(
        zip(training.BIN_LABELS, training.FROZEN_TRAIN_BIN_COUNTS[fold])
    )
    expected = {
        "fold": fold,
        "train_bin_counts": expected_counts,
        "weight_formula": "n_bin^-0.5",
        "alpha": 0.5,
        "weights_normalized": False,
        "num_samples": training.EXPECTED_FOLD_COUNTS[fold][0],
        "replacement": True,
        "sampler_seed": SEED,
        "worker_generator_seed": SEED,
        "sanity_simulation_seed": SEED,
        "sanity_simulation_draws": 100000,
        "forced_twenty_percent_per_bin": False,
    }
    mismatches = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Fold {fold} sampling provenance mismatch: {mismatches}")
    if not provenance["sanity_simulation"]["passed"]:
        raise RuntimeError(f"Fold {fold} sampler sanity simulation failed")
    if provenance["sampled_total_variation_from_uniform"] >= provenance[
        "baseline_total_variation_from_uniform"
    ]:
        raise RuntimeError(f"Fold {fold} sampler is not more balanced than baseline")


def validate_fold_outputs(fold: int, output_dir: Path) -> dict[str, Any]:
    result = baseline_runner.validate_fold_outputs(fold, output_dir)
    config = baseline_runner.read_json(output_dir / "training_config_snapshot.json")
    metadata = baseline_runner.read_json(output_dir / "run_metadata.json")
    final_metrics = baseline_runner.read_json(output_dir / "final_metrics.json")
    if config.get("training_version") != training.TRAINING_VERSION:
        raise RuntimeError(f"Fold {fold} training_version mismatch")
    for record in (config, metadata, final_metrics):
        _validate_sampling_provenance(fold, record)
    return result


def run_one_fold(fold: int, audit: dict[str, Any]) -> dict[str, Any]:
    output_dir = FORMAL_OUTPUT_ROOT / f"fold_{fold}_seed_{SEED}"
    ensure_new_fold_directory(output_dir)
    provenance = audit["sampling_provenance"]
    print(
        f"Fold {fold} preflight PASS: train={len(audit['train_records'])}, "
        f"validation={len(audit['validation_records'])}, "
        f"bins={provenance['train_bin_counts']}, "
        f"theory={provenance['expected_sampling_probability_per_bin']}",
        flush=True,
    )
    args = Namespace(
        manifest=training.DEFAULT_MANIFEST,
        image_root=training.DEFAULT_IMAGE_ROOT,
        fold=fold,
        output_dir=output_dir,
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        amp=AMP,
        pretrained=PRETRAINED,
        overwrite=False,
        pilot_run=False,
    )
    training.run_training(args)
    return validate_fold_outputs(fold, output_dir)


def write_cv_summary(started_at_utc: str) -> dict[str, Any]:
    fold_rows = [
        validate_fold_outputs(fold, FORMAL_OUTPUT_ROOT / f"fold_{fold}_seed_{SEED}")
        for fold in FOLDS
    ]
    aggregate = baseline_runner.aggregate_rows(fold_rows)
    csv_path = FORMAL_OUTPUT_ROOT / "cv_summary.csv"
    json_path = FORMAL_OUTPUT_ROOT / "cv_summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Refusing to overwrite existing L-balanced CV summary")
    csv_rows = [baseline_runner.csv_ready(row) for row in fold_rows]
    for statistic_name in ("mean", "std", "min", "max"):
        row = {field: "" for field in CSV_FIELDS}
        row["row_type"] = "aggregate"
        row["aggregate"] = statistic_name
        for field in SUMMARY_NUMERIC_FIELDS:
            row[field] = aggregate[field][statistic_name]
        csv_rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "schema_version": 1,
        "training_version": training.TRAINING_VERSION,
        "split_version": "date_grouped_v1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "model": training.MODEL_NAME,
        "formal_model_development_results": True,
        "final_test_accessed": False,
        "forbidden_roles_accessed": [],
        "single_experimental_change": (
            "Uniform shuffled training changed to sqrt inverse-frequency "
            "L-bin weighted replacement sampling."
        ),
        "configuration": training.load_training_config(),
        "folds": fold_rows,
        "aggregate": aggregate,
        "std_ddof": 1,
        "headline": {
            "validation_mae_mean": aggregate["validation_mae"]["mean"],
            "validation_mae_std": aggregate["validation_mae"]["std"],
            "validation_rmse_mean": aggregate["validation_rmse"]["mean"],
            "validation_rmse_std": aggregate["validation_rmse"]["std"],
            "validation_r2_mean": aggregate["validation_r2"]["mean"],
            "validation_r2_std": aggregate["validation_r2"]["std"],
        },
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_note": training.TEMPORAL_RISK_NOTE,
    }
    baseline_runner.write_json(json_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_frozen_config()
    FORMAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    for fold in range(args.start_fold, args.end_fold + 1):
        audit = preflight_formal_fold(fold)
        result = run_one_fold(fold, audit)
        if result["sustained_throughput_decline"] and fold < args.end_fold:
            raise RuntimeError(f"Fold {fold} throughput declined; stopping later folds")
    all_complete = all(
        all(
            (FORMAL_OUTPUT_ROOT / f"fold_{fold}_seed_{SEED}" / name).is_file()
            for name in REQUIRED_FOLD_FILES
        )
        for fold in FOLDS
    )
    if all_complete:
        write_cv_summary(started_at_utc)


if __name__ == "__main__":
    main()
