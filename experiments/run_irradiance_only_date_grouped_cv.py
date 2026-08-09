"""Run leakage-free formal irradiance-only cross-validation for date_grouped_v1."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import statistics
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_irradiance_only_date_grouped as training


FORMAL_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "date_grouped_v1" / "irradiance_only" / "formal_cv"
)
FOLDS = (1, 2, 3, 4)
SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 8
AMP = True
PRETRAINED = False
EXPECTED_MANIFEST_SHA256 = (
    "a354afc2b691719bf0cc3c3982033da833795006e3e3b0122cae07810bd83e02"
)


def validate_output_root_isolation() -> None:
    expected = (
        PROJECT_ROOT
        / "outputs"
        / "date_grouped_v1"
        / "irradiance_only"
        / "formal_cv"
    ).resolve()
    actual = FORMAL_OUTPUT_ROOT.resolve()
    forbidden_roots = {"resnet50_with_i", "resnet50_image_only"}
    if actual != expected or forbidden_roots.intersection(
        part.lower() for part in actual.parts
    ):
        raise RuntimeError(
            "Irradiance-only formal output root is not isolated: "
            f"expected {expected}, got {actual}"
        )


REQUIRED_FOLD_FILES = (
    "best_model.pth",
    "final_model.pth",
    "metrics_history.csv",
    "final_metrics.json",
    "run_metadata.json",
    "training_config_snapshot.json",
)
SUMMARY_NUMERIC_FIELDS = (
    "validation_mae",
    "validation_rmse",
    "validation_r2",
    "best_epoch",
    "training_time_seconds",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "gpu_max_temperature_c",
    "steady_train_samples_per_second",
    "completed_epochs",
)
CSV_FIELDS = (
    "row_type",
    "fold",
    "aggregate",
    "train_dates",
    "validation_dates",
    "train_count",
    "validation_count",
    "best_epoch",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
    "completed_epochs",
    "early_stopped",
    "training_time_seconds",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "gpu_max_temperature_c",
    "thermal_slowdown_observed",
    "steady_train_samples_per_second",
    "sustained_throughput_decline",
    "oom_occurred",
    "non_finite_detected",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal date_grouped_v1 irradiance-only cross-validation"
    )
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=4)
    args = parser.parse_args(argv)
    if args.start_fold not in FOLDS or args.end_fold not in FOLDS:
        parser.error("--start-fold and --end-fold must be between 1 and 4")
    if args.start_fold > args.end_fold:
        parser.error("--start-fold must not exceed --end-fold")
    return args


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_history(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_frozen_config() -> None:
    validate_output_root_isolation()
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
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=2)",
        "loss": "MSELoss(reduction=mean)",
        "early_stopping_metric": "validation_rmse",
        "checkpoint_selection": "minimum_validation_rmse",
        "input_preprocessing": training.INPUT_PREPROCESSING,
    }
    actual = training.load_training_config()
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen formal training config mismatch: {mismatches}")
    source = inspect.getsource(training)
    forbidden_initialization_tokens = (
        "torch" + ".load",
        "load_state_dict(" + "torch" + ".load",
    )
    if any(token in source for token in forbidden_initialization_tokens):
        raise RuntimeError("Formal training source can read a project checkpoint")


def preflight_formal_fold(fold: int) -> dict[str, Any]:
    audit = training.preflight_fold(
        training.DEFAULT_MANIFEST,
        fold,
    )
    expected_train, expected_validation = training.EXPECTED_FOLD_COUNTS[fold]
    actual_counts = (len(audit["train_records"]), len(audit["validation_records"]))
    if actual_counts != (expected_train, expected_validation):
        raise ValueError(
            f"Fold {fold} count mismatch: expected "
            f"{expected_train}/{expected_validation}, got {actual_counts}"
        )
    if audit["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Fold {fold} manifest SHA256 mismatch: {audit['manifest_sha256']}"
        )
    expected_forbidden = {
        "cp_calibration": 0,
        "decision_development": 0,
        "final_test": 0,
    }
    if audit["forbidden_role_counts"] != expected_forbidden:
        raise ValueError(
            f"Fold {fold} forbidden-role audit failed: "
            f"{audit['forbidden_role_counts']}"
        )
    if set(audit["train_dates"]) & set(audit["validation_dates"]):
        raise ValueError(f"Fold {fold} train/validation dates overlap")
    return audit


def ensure_new_fold_directory(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    present = [name for name in REQUIRED_FOLD_FILES if (output_dir / name).is_file()]
    status = "complete" if len(present) == len(REQUIRED_FOLD_FILES) else "incomplete"
    raise FileExistsError(
        f"Refusing to overwrite {status} fold directory: {output_dir}; "
        f"present required files={present}"
    )


def throughput_stability(history: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [float(row["train_samples_per_second"]) for row in history]
    if not rates or not all(math.isfinite(rate) and rate > 0 for rate in rates):
        raise ValueError("Training history has invalid throughput values")
    steady_rates = rates[1:] if len(rates) > 1 else rates
    steady_samples = [
        int(row["train_sample_count"])
        for row in (history[1:] if len(history) > 1 else history)
    ]
    steady_rate = sum(
        rate * count for rate, count in zip(steady_rates, steady_samples)
    ) / sum(steady_samples)

    sustained_decline = False
    late_vs_reference_ratio = None
    if len(steady_rates) >= 6:
        reference = statistics.median(steady_rates[:3])
        late = statistics.median(steady_rates[-3:])
        late_vs_reference_ratio = late / reference
        sustained_decline = (
            late_vs_reference_ratio < 0.80
            and all(rate < reference * 0.80 for rate in steady_rates[-3:])
        )
    return {
        "steady_train_samples_per_second": steady_rate,
        "sustained_throughput_decline": sustained_decline,
        "late_vs_reference_throughput_ratio": late_vs_reference_ratio,
        "decline_rule": (
            "After excluding epoch 1, all final three epoch throughputs are below "
            "80% of the median of the first three steady-state epochs."
        ),
    }


def validate_fold_outputs(fold: int, output_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FOLD_FILES if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Fold {fold} outputs are incomplete: missing={missing}")
    empty = [name for name in REQUIRED_FOLD_FILES if (output_dir / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"Fold {fold} outputs are empty: {empty}")

    final_metrics = read_json(output_dir / "final_metrics.json")
    metadata = read_json(output_dir / "run_metadata.json")
    config = read_json(output_dir / "training_config_snapshot.json")
    history = read_history(output_dir / "metrics_history.csv")
    completed_epochs = int(final_metrics["completed_epochs"])
    if len(history) != completed_epochs or completed_epochs < 1 or completed_epochs > EPOCHS:
        raise RuntimeError(
            f"Fold {fold} history/completed-epoch mismatch: "
            f"history={len(history)}, completed={completed_epochs}"
        )

    expected_config = {
        "fold": fold,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "amp_enabled": AMP,
        "pretrained_used": PRETRAINED,
        "PILOT_RUN": False,
        "NOT_FOR_RESEARCH_METRICS": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Fold {fold} output config mismatch: {mismatches}")
    if config.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Fold {fold} output manifest hash mismatch")
    if any(int(value) != 0 for value in config["forbidden_role_counts"].values()):
        raise RuntimeError(f"Fold {fold} output records a forbidden role")
    if set(config["train_dates"]) & set(config["validation_dates"]):
        raise RuntimeError(f"Fold {fold} output records overlapping dates")
    pretrained = metadata["pretrained"]
    if (
        pretrained["pretrained_requested"]
        or pretrained["pretrained_used"]
        or not pretrained["pretrained_not_applicable"]
    ):
        raise RuntimeError(f"Fold {fold} incorrectly records pretrained weights")

    stability = final_metrics["stability_summary"]
    if stability["oom_occurred"]:
        raise RuntimeError(f"Fold {fold} records CUDA OOM")
    if stability["non_finite_detected"]:
        raise RuntimeError(f"Fold {fold} records NaN/Inf")
    finite_fields = (
        "train_loss",
        "train_mae",
        "train_rmse",
        "train_r2",
        "validation_loss",
        "validation_mae",
        "validation_rmse",
        "validation_r2",
        "train_samples_per_second",
    )
    if not all(
        math.isfinite(float(row[field])) for row in history for field in finite_fields
    ):
        raise RuntimeError(f"Fold {fold} history contains NaN/Inf")

    best = final_metrics["best_validation_metrics"]
    throughput = throughput_stability(history)
    result = {
        "row_type": "fold",
        "fold": fold,
        "aggregate": "",
        "train_dates": list(config["train_dates"]),
        "validation_dates": list(config["validation_dates"]),
        "train_count": int(config["full_train_count"]),
        "validation_count": int(config["full_validation_count"]),
        "best_epoch": int(final_metrics["best_epoch"]),
        "validation_mae": float(best["mae"]),
        "validation_rmse": float(best["rmse"]),
        "validation_r2": float(best["r2"]),
        "completed_epochs": completed_epochs,
        "early_stopped": completed_epochs < EPOCHS,
        "training_time_seconds": float(final_metrics["duration_seconds"]),
        "gpu_peak_allocated_bytes": int(metadata["peak_memory_allocated_bytes"]),
        "gpu_peak_reserved_bytes": int(metadata["peak_memory_reserved_bytes"]),
        "gpu_max_temperature_c": stability["gpu_temperature_max_c"],
        "thermal_slowdown_observed": bool(
            stability["gpu_thermal_slowdown_observed"]
        ),
        "steady_train_samples_per_second": throughput[
            "steady_train_samples_per_second"
        ],
        "sustained_throughput_decline": throughput[
            "sustained_throughput_decline"
        ],
        "late_vs_reference_throughput_ratio": throughput[
            "late_vs_reference_throughput_ratio"
        ],
        "oom_occurred": False,
        "non_finite_detected": False,
        "output_dir": str(output_dir.resolve()),
    }
    return result


def run_one_fold(fold: int, audit: dict[str, Any]) -> dict[str, Any]:
    output_dir = FORMAL_OUTPUT_ROOT / f"fold_{fold}_seed_{SEED}"
    ensure_new_fold_directory(output_dir)
    print(
        f"Fold {fold} preflight PASS: "
        f"train={len(audit['train_records'])}, "
        f"validation={len(audit['validation_records'])}, "
        f"train_dates={audit['train_dates']}, "
        f"validation_dates={audit['validation_dates']}, "
        f"forbidden={audit['forbidden_role_counts']}, "
        f"manifest_sha256={audit['manifest_sha256']}",
        flush=True,
    )
    args = Namespace(
        manifest=training.DEFAULT_MANIFEST,
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
        pilot_run=False,
    )
    training.run_training(args)
    result = validate_fold_outputs(fold, output_dir)
    print(
        f"Fold {fold} verified: best_epoch={result['best_epoch']}, "
        f"validation_rmse={result['validation_rmse']:.8f}, "
        f"steady_samples/s={result['steady_train_samples_per_second']:.3f}, "
        f"max_temperature={result['gpu_max_temperature_c']}, "
        f"thermal_slowdown={result['thermal_slowdown_observed']}, "
        f"sustained_throughput_decline={result['sustained_throughput_decline']}",
        flush=True,
    )
    return result


def aggregate_rows(fold_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if len(fold_rows) != len(FOLDS):
        raise ValueError("Four completed folds are required for aggregation")
    aggregate: dict[str, dict[str, float]] = {}
    for field in SUMMARY_NUMERIC_FIELDS:
        values = [float(row[field]) for row in fold_rows]
        aggregate[field] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }
    return aggregate


def csv_ready(row: dict[str, Any]) -> dict[str, Any]:
    result = {field: row.get(field, "") for field in CSV_FIELDS}
    for field in ("train_dates", "validation_dates"):
        if isinstance(result[field], list):
            result[field] = ";".join(result[field])
    return result


def write_cv_summary(started_at_utc: str) -> dict[str, Any]:
    fold_rows = [
        validate_fold_outputs(
            fold, FORMAL_OUTPUT_ROOT / f"fold_{fold}_seed_{SEED}"
        )
        for fold in FOLDS
    ]
    aggregate = aggregate_rows(fold_rows)
    csv_path = FORMAL_OUTPUT_ROOT / "cv_summary.csv"
    json_path = FORMAL_OUTPUT_ROOT / "cv_summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Refusing to overwrite existing formal CV summary")

    csv_rows = [csv_ready(row) for row in fold_rows]
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
        "training_version": "irradiance_only_date_grouped_v1",
        "split_version": "date_grouped_v1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "model": "IrradianceOnlyMLP",
        "formal_model_development_results": True,
        "final_test_accessed": False,
        "forbidden_roles_accessed": [],
        "configuration": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "amp": AMP,
            "pretrained": PRETRAINED,
            "early_stopping_metric": "validation_rmse",
        },
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
        "total_fold_training_time_seconds": sum(
            row["training_time_seconds"] for row in fold_rows
        ),
        "oom_occurred": any(row["oom_occurred"] for row in fold_rows),
        "non_finite_detected": any(
            row["non_finite_detected"] for row in fold_rows
        ),
        "sustained_throughput_decline_detected": any(
            row["sustained_throughput_decline"] for row in fold_rows
        ),
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_files": [csv_path.name, json_path.name],
    }
    write_json(json_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_frozen_config()
    FORMAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    print(
        f"Formal CV requested folds {args.start_fold}-{args.end_fold}; "
        f"output={FORMAL_OUTPUT_ROOT}",
        flush=True,
    )

    for fold in range(args.start_fold, args.end_fold + 1):
        audit = preflight_formal_fold(fold)
        result = run_one_fold(fold, audit)
        if result["sustained_throughput_decline"] and fold < args.end_fold:
            raise RuntimeError(
                f"Fold {fold} has sustained throughput decline; stopping later folds"
            )

    all_complete = all(
        all((FORMAL_OUTPUT_ROOT / f"fold_{fold}_seed_{SEED}" / name).is_file()
            for name in REQUIRED_FOLD_FILES)
        for fold in FOLDS
    )
    if all_complete:
        summary = write_cv_summary(started_at_utc)
        print(
            "Formal CV complete: "
            f"MAE={summary['headline']['validation_mae_mean']:.8f} ± "
            f"{summary['headline']['validation_mae_std']:.8f}; "
            f"RMSE={summary['headline']['validation_rmse_mean']:.8f} ± "
            f"{summary['headline']['validation_rmse_std']:.8f}; "
            f"R2={summary['headline']['validation_r2_mean']:.8f} ± "
            f"{summary['headline']['validation_r2_std']:.8f}",
            flush=True,
        )
    else:
        print(
            "Requested fold range complete; four-fold summary deferred until all "
            "fold directories are complete.",
            flush=True,
        )
    print(f"Runner elapsed seconds={time.perf_counter() - run_started:.1f}", flush=True)


if __name__ == "__main__":
    main()
