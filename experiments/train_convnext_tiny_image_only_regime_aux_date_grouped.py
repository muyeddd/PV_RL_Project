"""Fold3 ConvNeXt-Tiny image-only training with one regime auxiliary task.

Importing this module never starts training. The data split, transforms, ordinary
shuffled training loader, validation loader, optimizer, scheduler, AMP policy,
and regression-RMSE model selection are inherited from the frozen baseline.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from experiments import train_convnext_tiny_image_only_date_grouped as baseline
from experiments import train_resnet50_image_only_date_grouped as resnet_baseline
from models.convnext_tiny_image_only_regime_aux import (
    OFFICIAL_CHECKPOINT_FILENAME,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_WEIGHTS,
    SolarConvNeXtTinyImageOnlyRegimeAux,
    verify_official_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = baseline.DEFAULT_MANIFEST
DEFAULT_IMAGE_ROOT = baseline.DEFAULT_IMAGE_ROOT
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "convnext_tiny_image_only_regime_aux_v1_date_grouped.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "convnext_tiny_image_only_regime_aux_v1"
    / "pilot"
)
BASELINE_OUTPUT_ROOT = baseline.OUTPUT_ROOT
TRAINING_VERSION = "convnext_tiny_image_only_regime_aux_v1_date_grouped"
MODEL_NAME = "SolarConvNeXtTinyImageOnlyRegimeAux"
ALLOWED_PILOT_FOLDS = (3,)
REGIME_BOUNDARIES = (0.1, 0.5)
REGIME_CLASS_NAMES = ("low", "medium", "high")
LAMBDA_REGIME = 0.01
EXPECTED_FOLD3_REGIME_COUNTS = (6_549, 8_261, 4_406)
REQUIRED_OUTPUT_FILES = baseline.REQUIRED_OUTPUT_FILES
PREDICTION_FIELDS = baseline.PREDICTION_FIELDS
build_transforms = baseline.build_transforms
PREPROCESSING_DESCRIPTION = baseline.PREPROCESSING_DESCRIPTION


def load_training_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "training_version",
        "split_version",
        "manifest_sha256",
        "model_name",
        "weights_enum",
        "weights_filename",
        "weights_sha256",
        "seed",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "loss",
        "auxiliary_loss",
        "lambda_regime",
        "regime_boundaries",
        "regime_classes",
        "optimizer",
        "scheduler",
        "scheduler_factor",
        "scheduler_patience",
        "early_stopping_patience",
        "selection_metric",
        "amp",
        "pretrained",
        "dropout",
        "allowed_pilot_folds",
        "model_development_sample_count",
        "fold1_train_count",
        "fold1_validation_count",
        "fold2_train_count",
        "fold2_validation_count",
        "fold3_train_count",
        "fold3_validation_count",
        "fold4_train_count",
        "fold4_validation_count",
        "transform_source",
        "training_sampler",
        "validation_weighting",
        "pilot_output_namespace",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Regime auxiliary config missing fields: {missing}")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported regime auxiliary config schema_version")
    return config


def validate_single_variable_contract(
    config: dict[str, Any] | None = None,
    baseline_config: dict[str, Any] | None = None,
) -> None:
    """Prove all baseline protocol fields except the auxiliary task are frozen."""

    config = load_training_config() if config is None else config
    baseline_config = (
        baseline.load_training_config() if baseline_config is None else baseline_config
    )
    shared_keys = (
        "schema_version",
        "split_version",
        "manifest_sha256",
        "weights_enum",
        "weights_filename",
        "weights_sha256",
        "seed",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "loss",
        "optimizer",
        "scheduler",
        "scheduler_factor",
        "scheduler_patience",
        "early_stopping_patience",
        "selection_metric",
        "amp",
        "pretrained",
        "dropout",
        "model_development_sample_count",
        "fold1_train_count",
        "fold1_validation_count",
        "fold2_train_count",
        "fold2_validation_count",
        "fold3_train_count",
        "fold3_validation_count",
        "fold4_train_count",
        "fold4_validation_count",
        "transform_source",
    )
    mismatches = {
        key: {"baseline": baseline_config.get(key), "regime_aux": config.get(key)}
        for key in shared_keys
        if config.get(key) != baseline_config.get(key)
    }
    if mismatches:
        raise ValueError(f"Baseline protocol mismatch: {mismatches}")
    expected_experiment = {
        "training_version": TRAINING_VERSION,
        "model_name": MODEL_NAME,
        "auxiliary_loss": "CrossEntropyLoss",
        "lambda_regime": LAMBDA_REGIME,
        "regime_boundaries": list(REGIME_BOUNDARIES),
        "regime_classes": list(REGIME_CLASS_NAMES),
        "allowed_pilot_folds": [3],
        "training_sampler": "uniform_shuffle",
        "validation_weighting": "none",
        "pilot_output_namespace": (
            "outputs/date_grouped_v1/"
            "convnext_tiny_image_only_regime_aux_v1/pilot"
        ),
    }
    experiment_mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_experiment.items()
        if config.get(key) != value
    }
    if experiment_mismatches:
        raise ValueError(f"Regime auxiliary contract mismatch: {experiment_mismatches}")
    if config["pilot_output_namespace"] == baseline_config["pilot_output_namespace"]:
        raise ValueError("Regime auxiliary output namespace overlaps baseline")


def validate_pilot_fold(fold: int) -> None:
    if isinstance(fold, bool) or fold not in ALLOWED_PILOT_FOLDS:
        raise ValueError("ConvNeXt regime-aware auxiliary v1 permits only Fold3")


def expected_output_dir(fold: int, seed: int = 42) -> Path:
    validate_pilot_fold(fold)
    if seed != 42:
        raise ValueError("The regime-aware v1 protocol requires seed=42")
    return OUTPUT_ROOT / f"fold_{fold}_seed_{seed}"


def _prepare_output_directory(output_dir: Path, fold: int, seed: int) -> Path:
    expected = expected_output_dir(fold, seed).resolve()
    actual = Path(output_dir).resolve()
    if actual != expected:
        raise RuntimeError(f"Output directory escaped regime auxiliary namespace: {actual}")
    if actual.exists():
        raise FileExistsError(f"Refusing to overwrite regime auxiliary output: {actual}")
    actual.mkdir(parents=True, exist_ok=False)
    return actual


def assign_regime_ids(labels: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Regime labels must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("Regime labels must be finite")
    ids = np.searchsorted(REGIME_BOUNDARIES, values, side="right").astype(np.int64)
    if np.any((ids < 0) | (ids > 2)):
        raise RuntimeError("Every label must belong to exactly one regime")
    return ids


def regime_targets_from_tensor(target_l: torch.Tensor) -> torch.Tensor:
    boundaries = torch.tensor(
        REGIME_BOUNDARIES, dtype=target_l.dtype, device=target_l.device
    )
    return torch.bucketize(target_l.reshape(-1), boundaries, right=True).long()


def labels_from_train_records(train_records) -> np.ndarray:
    if train_records.empty:
        raise ValueError("Fold training records must not be empty")
    roles = set(train_records["top_level_role"].astype(str))
    if roles != {baseline.MODEL_DEVELOPMENT_ROLE}:
        raise ValueError(f"Training records contain disallowed roles: {sorted(roles)}")
    return np.asarray(
        [baseline.parse_loss_label(name) for name in train_records["filename"]],
        dtype=np.float64,
    )


def compute_train_regime_audit(train_records) -> dict[str, Any]:
    labels = labels_from_train_records(train_records)
    regime_ids = assign_regime_ids(labels)
    counts = np.bincount(regime_ids, minlength=3).astype(np.int64)
    if int(counts.sum()) != len(train_records):
        raise RuntimeError("Each Fold3 training sample must belong to one regime")
    return {
        "regime_counts": counts.tolist(),
        "regime_labels": list(REGIME_CLASS_NAMES),
        "train_count": len(train_records),
        "weights_source": "none",
        "validation_participated": False,
    }


def preflight_fold(fold: int) -> dict[str, Any]:
    validate_pilot_fold(fold)
    audit = baseline.preflight_fold(fold)
    regime_audit = compute_train_regime_audit(audit["train_records"])
    if tuple(regime_audit["regime_counts"]) != EXPECTED_FOLD3_REGIME_COUNTS:
        raise ValueError(
            "Fold3 train regime count mismatch: expected "
            f"{EXPECTED_FOLD3_REGIME_COUNTS}, got "
            f"{tuple(regime_audit['regime_counts'])}"
        )
    return {**audit, "train_regime_audit": regime_audit}


def build_fold_datasets(
    fold: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    image_root: Path = DEFAULT_IMAGE_ROOT,
):
    validate_pilot_fold(fold)
    return baseline.build_fold_datasets(fold, manifest_path, image_root)


def build_data_loaders(
    train_dataset,
    validation_dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build the same ordinary shuffled/sequential loaders as the baseline."""

    common_loader = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": resnet_baseline.seed_data_loader_worker,
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common_loader
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common_loader)
    return train_loader, validation_loader


def _unpack_model_outputs(outputs):
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise RuntimeError("Regime auxiliary model must return regression and logits")
    predictions, regime_logits = outputs
    if predictions.ndim != 2 or predictions.shape[1] != 1:
        raise RuntimeError("Regression head must return shape [N, 1]")
    if regime_logits.ndim != 2 or regime_logits.shape[1] != 3:
        raise RuntimeError("Regime head must return shape [N, 3]")
    return predictions, regime_logits


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    regression_criterion: nn.Module,
    auxiliary_criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_accumulation_steps: int = 1,
    lambda_regime: float = LAMBDA_REGIME,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)
    regression_metrics = baseline.MetricAccumulator()
    regression_loss_sum = 0.0
    auxiliary_loss_sum = 0.0
    total_loss_sum = 0.0
    sample_count = 0
    correct = 0
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    started = time.perf_counter()
    if is_training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader, start=1):
        images, target_l, _, _ = baseline._unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        target_l = target_l.to(device, non_blocking=True).float().unsqueeze(1)
        regime_targets = regime_targets_from_tensor(target_l)
        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(
                device_type="cuda", enabled=amp_enabled and device.type == "cuda"
            ):
                predictions, regime_logits = _unpack_model_outputs(model(images))
                regression_mse = regression_criterion(predictions, target_l)
                regime_ce = auxiliary_criterion(regime_logits, regime_targets)
                total_loss = regression_mse + lambda_regime * regime_ce
            if is_training:
                scaled_loss = total_loss / gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = (
                    batch_index % gradient_accumulation_steps == 0
                    or batch_index == len(loader)
                )
                if should_step:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        if not all(
            torch.isfinite(value) for value in (regression_mse, regime_ce, total_loss)
        ):
            raise FloatingPointError("Non-finite regime auxiliary loss detected")

        batch_size = int(target_l.shape[0])
        regression_loss_sum += float(regression_mse.detach()) * batch_size
        auxiliary_loss_sum += float(regime_ce.detach()) * batch_size
        total_loss_sum += float(total_loss.detach()) * batch_size
        sample_count += batch_size
        regression_metrics.update(predictions, target_l)
        predicted_regimes = regime_logits.detach().argmax(dim=1)
        correct += int((predicted_regimes == regime_targets).sum().cpu())
        for true_id, predicted_id in zip(
            regime_targets.detach().cpu().tolist(), predicted_regimes.cpu().tolist()
        ):
            confusion[true_id, predicted_id] += 1

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    metrics = regression_metrics.compute()
    metrics.update(
        {
            "regression_mse": regression_loss_sum / sample_count,
            "regime_ce": auxiliary_loss_sum / sample_count,
            "total_loss": total_loss_sum / sample_count,
            "regime_accuracy": correct / sample_count,
            "regime_confusion_matrix": confusion.tolist(),
            "duration_seconds": time.perf_counter() - started,
        }
    )
    return metrics


@torch.no_grad()
def predict_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    fold: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images, target_l, filenames, dates = baseline._unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", enabled=amp_enabled and device.type == "cuda"
        ):
            predictions, _ = _unpack_model_outputs(model(images))
        predicted_values = predictions.float().cpu().numpy().reshape(-1)
        true_values = target_l.float().numpy().reshape(-1)
        for filename, date, true_l, pred_l in zip(
            filenames, dates, true_values, predicted_values
        ):
            error = float(pred_l - true_l)
            rows.append(
                {
                    "filename": filename,
                    "date": date,
                    "fold": fold,
                    "true_L": float(true_l),
                    "pred_L": float(pred_l),
                    "error": error,
                    "abs_error": abs(error),
                }
            )
    return rows, baseline._prediction_summary(rows)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run the Fold3 pilot; this function is not invoked by unit tests."""

    validate_pilot_fold(args.fold)
    config = load_training_config(Path(args.config))
    validate_single_variable_contract(config)
    for argument, key in (
        (args.seed, "seed"),
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.gradient_accumulation_steps, "gradient_accumulation_steps"),
    ):
        if argument != config[key]:
            raise ValueError(f"{key} override is prohibited")

    checkpoint_provenance = verify_official_checkpoint()
    audit = baseline.preflight_fold(
        args.fold, args.manifest, args.image_root, verify_selected_paths=False
    )
    regime_audit = compute_train_regime_audit(audit["train_records"])
    if tuple(regime_audit["regime_counts"]) != EXPECTED_FOLD3_REGIME_COUNTS:
        raise ValueError("Fold3 training regime counts differ from the frozen audit")
    output_dir = _prepare_output_directory(args.output_dir, args.fold, args.seed)

    baseline.set_random_seeds(args.seed)
    train_dataset, validation_dataset, _ = build_fold_datasets(
        args.fold, args.manifest, args.image_root
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = build_data_loaders(
        train_dataset,
        validation_dataset,
        args.batch_size,
        args.num_workers,
        device,
        args.seed,
    )

    model = SolarConvNeXtTinyImageOnlyRegimeAux(
        dropout=config["dropout"], use_pretrained=config["pretrained"]
    ).to(device)
    regression_criterion = nn.MSELoss(reduction="mean")
    auxiliary_criterion = nn.CrossEntropyLoss(reduction="mean")
    if auxiliary_criterion.weight is not None:
        raise RuntimeError("Class-weighted auxiliary loss is prohibited")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    config_snapshot = {
        **config,
        "fold": args.fold,
        "manifest": str(Path(args.manifest).resolve()),
        "image_root": str(Path(args.image_root).resolve()),
        "output_dir": str(output_dir),
        "full_train_count": audit["train_count"],
        "full_validation_count": audit["validation_count"],
        "train_dates": audit["train_dates"],
        "validation_dates": audit["validation_dates"],
        "train_regime_audit": regime_audit,
        "pretrained_provenance": checkpoint_provenance,
        "protected_roles_accessed": False,
        "cp_calibration_accessed": False,
        "decision_development_accessed": False,
        "final_test_accessed": False,
        "forbidden_roles_accessed": [],
    }
    baseline._write_json(output_dir / "config_snapshot.json", config_snapshot)

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            regression_criterion,
            auxiliary_criterion,
            device,
            amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lambda_regime=config["lambda_regime"],
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            regression_criterion,
            auxiliary_criterion,
            device,
            amp_enabled,
            lambda_regime=config["lambda_regime"],
        )
        validation_regression_rmse = validation_metrics["rmse"]
        scheduler.step(validation_regression_rmse)
        improved = validation_regression_rmse < best_rmse
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_regression_mse": train_metrics["regression_mse"],
            "train_regime_ce": train_metrics["regime_ce"],
            "train_total_loss": train_metrics["total_loss"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "train_regime_accuracy": train_metrics["regime_accuracy"],
            "train_regime_confusion_matrix": json.dumps(
                train_metrics["regime_confusion_matrix"], separators=(",", ":")
            ),
            "validation_regression_mse": validation_metrics["regression_mse"],
            "validation_regime_ce": validation_metrics["regime_ce"],
            "validation_total_loss": validation_metrics["total_loss"],
            "validation_rmse": validation_regression_rmse,
            "validation_mae": validation_metrics["mae"],
            "validation_r2": validation_metrics["r2"],
            "validation_regime_accuracy": validation_metrics["regime_accuracy"],
            "validation_regime_confusion_matrix": json.dumps(
                validation_metrics["regime_confusion_matrix"], separators=(",", ":")
            ),
            "improved": improved,
        }
        history.append(row)
        baseline._write_rows(output_dir / "history.csv", list(row), history)
        if improved:
            best_rmse = float(validation_regression_rmse)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "fold": args.fold,
                    "seed": args.seed,
                    "model_name": MODEL_NAME,
                    "weights_enum": str(OFFICIAL_WEIGHTS),
                    "selection_metric": "validation_rmse",
                    "validation_regression_metrics": validation_metrics,
                    "protected_roles_accessed": False,
                    "final_test_accessed": False,
                    "forbidden_roles_accessed": [],
                },
                output_dir / "best_model.pth",
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config["early_stopping_patience"]:
            break

    best_payload = torch.load(
        output_dir / "best_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    prediction_rows, metrics = predict_validation(
        model, validation_loader, device, amp_enabled, args.fold
    )
    baseline._write_rows(output_dir / "predictions.csv", PREDICTION_FIELDS, prediction_rows)
    duration = time.perf_counter() - started
    final_metrics = {
        **metrics,
        "fold": args.fold,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "selection_metric": "validation_rmse",
        "duration_seconds": duration,
    }
    baseline._write_json(output_dir / "final_metrics.json", final_metrics)
    run_metadata = {
        "training_version": config["training_version"],
        "model_name": MODEL_NAME,
        "fold": args.fold,
        "formal_training_run": True,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else 0
        ),
        "peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "pretrained_provenance": checkpoint_provenance,
        "output_files": list(REQUIRED_OUTPUT_FILES),
        "protected_roles_accessed": False,
        "cp_calibration_accessed": False,
        "decision_development_accessed": False,
        "final_test_accessed": False,
        "forbidden_roles_accessed": [],
    }
    baseline._write_json(output_dir / "run_metadata.json", run_metadata)
    return final_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config = load_training_config()
    parser = argparse.ArgumentParser(
        description="Run Fold3 ConvNeXt-Tiny image-only regime auxiliary v1"
    )
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--epochs", type=int, default=config["epochs"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=config["gradient_accumulation_steps"],
    )
    parser.add_argument("--num-workers", type=int, default=config["num_workers"])
    args = parser.parse_args(argv)
    validate_pilot_fold(args.fold)
    if args.output_dir is None:
        args.output_dir = expected_output_dir(args.fold, args.seed)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
