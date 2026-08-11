"""Fold4-only regression-head diagnostics on the frozen DINOv2 feature cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_dinov2_vits14_frozen_regression_date_grouped as base
from models.dinov2_regression_diagnostic_head import DINOv2DiagnosticRegressionHead


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "dinov2_vits14_frozen_regression_diagnostic_v1_fold4.json"
)
EXPECTED_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "dinov2_vits14_frozen_regression_diagnostic_v1"
    / "fold_4"
)
ALLOWED_VARIANTS = ("A", "B", "C")
LOCKED_FOLD = 4
EXPECTED_COUNTS = (19603, 6113)
REQUIRED_OUTPUT_FILES = (
    "best_model.pth",
    "final_metrics.json",
    "history.csv",
    "predictions.csv",
    "training_predictions.csv",
    "run_metadata.json",
    "config_snapshot.json",
    "standardization.npz",
)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "diagnostic_version",
        "split_version",
        "feature_version",
        "feature_cache_dir",
        "output_root",
        "fold",
        "seed",
        "input_dimension",
        "hidden_dimension",
        "dropout",
        "feature_standardization",
        "standard_deviation_epsilon",
        "batch_size",
        "num_workers",
        "loss",
        "optimizer",
        "weight_decay",
        "max_epochs",
        "scheduler",
        "scheduler_factor",
        "scheduler_patience",
        "early_stopping_patience",
        "selection_metric",
        "prediction_clipping",
        "amp",
        "variants",
        "expected_sample_count",
        "expected_feature_shape",
        "expected_feature_dtype",
        "expected_fold_validation_counts",
        "expected_fold_validation_dates",
        "feature_cache_sha256",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Diagnostic config is missing fields: {missing}")
    expected = {
        "schema_version": 1,
        "diagnostic_version": "dinov2_vits14_frozen_regression_diagnostic_v1",
        "split_version": "date_grouped_v1",
        "feature_version": "dinov2_vits14_frozen_v1",
        "fold": 4,
        "seed": 42,
        "input_dimension": 384,
        "hidden_dimension": 128,
        "dropout": 0.3,
        "feature_standardization": "training_fold_per_dimension_zscore",
        "standard_deviation_epsilon": 1e-6,
        "batch_size": 256,
        "num_workers": 0,
        "loss": "MSELoss",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "max_epochs": 50,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "early_stopping_patience": 8,
        "selection_metric": "validation_rmse",
        "prediction_clipping": False,
        "amp": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen Fold4 diagnostic config mismatch: {mismatches}")
    expected_variants = {
        "A": ("standardized_sigmoid_lr1e4", "variant_A", 1e-4, "Sigmoid"),
        "B": ("standardized_sigmoid_lr1e3", "variant_B", 1e-3, "Sigmoid"),
        "C": ("standardized_linear_lr1e3", "variant_C", 1e-3, "Linear"),
    }
    if set(config["variants"]) != set(ALLOWED_VARIANTS):
        raise ValueError("Diagnostic must contain exactly variants A, B, and C")
    for key, values in expected_variants.items():
        variant = config["variants"][key]
        actual = (
            variant.get("name"),
            variant.get("output_directory"),
            variant.get("learning_rate"),
            variant.get("output_activation"),
        )
        if actual != values:
            raise ValueError(f"Variant {key} configuration changed: {actual}")
    if base.resolve_project_path(config["output_root"]).resolve() != EXPECTED_OUTPUT_ROOT.resolve():
        raise ValueError("Diagnostic output root is not isolated")
    return config


def validate_variant(variant: str) -> None:
    if variant not in ALLOWED_VARIANTS:
        raise ValueError("Only diagnostic variants A, B, and C are allowed")


def reject_non_fold4(fold: int) -> None:
    if fold != LOCKED_FOLD:
        raise ValueError("This diagnostic is strictly limited to Fold 4")


def variant_output_dir(config: dict[str, Any], variant: str) -> Path:
    validate_variant(variant)
    root = base.resolve_project_path(config["output_root"]).resolve()
    output = root / config["variants"][variant]["output_directory"]
    if output.resolve().parent != root or output.resolve() == root:
        raise ValueError("Variant output escaped the isolated Fold4 diagnostic root")
    return output


@dataclass(frozen=True)
class TrainingStandardization:
    mean: np.ndarray
    std: np.ndarray
    safe_std: np.ndarray
    near_zero_mask: np.ndarray
    epsilon: float
    fitted_row_count: int


def compute_training_standardization(
    features: np.ndarray,
    training_indices: np.ndarray,
    epsilon: float,
) -> TrainingStandardization:
    """Fit per-dimension moments using training indices and nothing else."""

    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("Training indices must be a non-empty unique 1D array")
    if indices.min() < 0 or indices.max() >= len(features):
        raise ValueError("Training standardization index is out of bounds")
    if epsilon <= 0:
        raise ValueError("Standard-deviation epsilon must be positive")
    training_features = np.asarray(features[indices], dtype=np.float32)
    if training_features.shape[1] != 384 or not np.isfinite(training_features).all():
        raise ValueError("Training features must be finite with dimension 384")
    mean = training_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = training_features.std(axis=0, dtype=np.float64).astype(np.float32)
    near_zero_mask = (~np.isfinite(std)) | (std < epsilon)
    safe_std = std.copy()
    safe_std[near_zero_mask] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(safe_std).all():
        raise ValueError("Training-only standardization produced non-finite values")
    return TrainingStandardization(
        mean=mean,
        std=std,
        safe_std=safe_std,
        near_zero_mask=near_zero_mask,
        epsilon=float(epsilon),
        fitted_row_count=int(indices.size),
    )


def apply_standardization(
    features: np.ndarray, statistics: TrainingStandardization
) -> np.ndarray:
    standardized = (
        np.asarray(features, dtype=np.float32) - statistics.mean[None, :]
    ) / statistics.safe_std[None, :]
    standardized = np.asarray(standardized, dtype=np.float32)
    if standardized.shape[1] != 384 or not np.isfinite(standardized).all():
        raise ValueError("Standardized features contain NaN/Inf or wrong dimension")
    return standardized


@dataclass
class DiagnosticData:
    cache: base.FeatureCache
    standardized_features: np.ndarray
    training_indices: np.ndarray
    validation_indices: np.ndarray
    standardization: TrainingStandardization


def prepare_diagnostic_data(config: dict[str, Any]) -> DiagnosticData:
    reject_non_fold4(config["fold"])
    cache = base.load_and_validate_feature_cache(config)
    training_indices, validation_indices = base.fold_indices(cache, LOCKED_FOLD)
    if (len(training_indices), len(validation_indices)) != EXPECTED_COUNTS:
        raise ValueError("Fold4 diagnostic counts changed")
    statistics = compute_training_standardization(
        cache.features,
        training_indices,
        config["standard_deviation_epsilon"],
    )
    standardized = apply_standardization(cache.features, statistics)
    return DiagnosticData(
        cache=cache,
        standardized_features=standardized,
        training_indices=training_indices,
        validation_indices=validation_indices,
        standardization=statistics,
    )


class StandardizedFeatureDataset(Dataset):
    def __init__(self, data: DiagnosticData, indices: np.ndarray):
        self.features = data.standardized_features
        self.labels = data.cache.metadata["L"].to_numpy(dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        return (
            torch.from_numpy(self.features[row]),
            torch.tensor(self.labels[row], dtype=torch.float32),
        )


def build_model(config: dict[str, Any], variant: str) -> DINOv2DiagnosticRegressionHead:
    validate_variant(variant)
    return DINOv2DiagnosticRegressionHead(
        output_activation=config["variants"][variant]["output_activation"]
    )


def build_loss(config: dict[str, Any]) -> nn.Module:
    if config["loss"] != "MSELoss":
        raise ValueError("Diagnostic requires MSELoss")
    return nn.MSELoss(reduction="mean")


def build_optimizer(
    model: nn.Module, config: dict[str, Any], variant: str
) -> torch.optim.Optimizer:
    validate_variant(variant)
    if config["optimizer"] != "AdamW":
        raise ValueError("Diagnostic requires AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["variants"][variant]["learning_rate"],
        weight_decay=config["weight_decay"],
    )


def build_scheduler(optimizer, config: dict[str, Any]):
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_count
        count += batch_count
    if count == 0:
        raise RuntimeError("Empty training loader")
    return loss_sum / count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    count = 0
    true_chunks: list[np.ndarray] = []
    prediction_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            predictions = model(features)
            loss = criterion(predictions, labels)
            batch_count = labels.shape[0]
            loss_sum += float(loss.cpu()) * batch_count
            count += batch_count
            true_chunks.append(labels.float().cpu().numpy().reshape(-1))
            prediction_chunks.append(predictions.float().cpu().numpy().reshape(-1))
    if count == 0:
        raise RuntimeError("Empty evaluation loader")
    true_values = np.concatenate(true_chunks)
    predictions = np.concatenate(prediction_chunks)
    # Intentionally no clipping: these are the model's raw outputs.
    metrics = base.compute_metrics(true_values, predictions)
    return loss_sum / count, true_values, predictions, metrics


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_predictions(
    path: Path,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    true_values: np.ndarray,
    predictions: np.ndarray,
    split: str,
) -> None:
    rows = metadata.iloc[indices].reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "filename": rows["filename"].astype(str),
            "date": rows["date"].astype(str),
            "fold": LOCKED_FOLD,
            "split": split,
            "true_L": true_values,
            "pred_L": predictions,
        }
    )
    frame["error"] = frame["pred_L"] - frame["true_L"]
    frame["abs_error"] = frame["error"].abs()
    frame.to_csv(path, index=False, lineterminator="\n")


def prepare_output_directory(config: dict[str, Any], variant: str) -> Path:
    output = variant_output_dir(config, variant)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_variant(
    config: dict[str, Any],
    data: DiagnosticData,
    variant: str,
) -> dict[str, Any]:
    """Train one authorized Fold4 variant; called only by the explicit runner."""

    validate_variant(variant)
    reject_non_fold4(config["fold"])
    base.set_random_seeds(config["seed"])
    output_dir = prepare_output_directory(config, variant)
    variant_config = {**config, "selected_variant": variant, "variant": config["variants"][variant]}
    _write_json(output_dir / "config_snapshot.json", variant_config)
    np.savez(
        output_dir / "standardization.npz",
        mean=data.standardization.mean,
        std=data.standardization.std,
        safe_std=data.standardization.safe_std,
        near_zero_mask=data.standardization.near_zero_mask,
        epsilon=np.array(data.standardization.epsilon, dtype=np.float64),
        fitted_row_count=np.array(data.standardization.fitted_row_count, dtype=np.int64),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(config["seed"])
    training_dataset = StandardizedFeatureDataset(data, data.training_indices)
    validation_dataset = StandardizedFeatureDataset(data, data.validation_indices)
    train_loader = DataLoader(
        training_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    train_evaluation_loader = DataLoader(
        training_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config, variant).to(device)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config, variant)
    scheduler = build_scheduler(optimizer, config)

    history: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        optimization_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_loss, _, _, train_metrics = evaluate(
            model, train_evaluation_loader, criterion, device
        )
        validation_loss, _, _, validation_metrics = evaluate(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation_metrics["rmse"])
        history.append(
            {
                "epoch": epoch,
                "optimization_train_loss": optimization_loss,
                "evaluation_train_loss": train_loss,
                "train_r2": train_metrics["r2"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "validation_loss": validation_loss,
                "validation_r2": validation_metrics["r2"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_mae": validation_metrics["mae"],
                "validation_prediction_std": validation_metrics["prediction_std"],
                "validation_true_std": validation_metrics["true_std"],
                "validation_bias": validation_metrics["bias"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_metrics["rmse"] < best_rmse:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "fold": LOCKED_FOLD,
                    "seed": config["seed"],
                    "variant": variant,
                    "variant_config": config["variants"][variant],
                    "training_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                    "feature_cache_sha256": data.cache.file_sha256,
                    "standardization_mean": torch.from_numpy(data.standardization.mean.copy()),
                    "standardization_safe_std": torch.from_numpy(
                        data.standardization.safe_std.copy()
                    ),
                },
                output_dir / "best_model.pth",
            )
        else:
            epochs_without_improvement += 1
        _write_history(output_dir / "history.csv", history)
        if epochs_without_improvement >= config["early_stopping_patience"]:
            break

    checkpoint = torch.load(
        output_dir / "best_model.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _, train_true, train_predictions, train_metrics = evaluate(
        model, train_evaluation_loader, criterion, device
    )
    _, validation_true, validation_predictions, validation_metrics = evaluate(
        model, validation_loader, criterion, device
    )
    _write_predictions(
        output_dir / "training_predictions.csv",
        data.cache.metadata,
        data.training_indices,
        train_true,
        train_predictions,
        "training",
    )
    _write_predictions(
        output_dir / "predictions.csv",
        data.cache.metadata,
        data.validation_indices,
        validation_true,
        validation_predictions,
        "validation",
    )
    final_metrics = {
        "variant": variant,
        "variant_name": config["variants"][variant]["name"],
        "fold": LOCKED_FOLD,
        "seed": config["seed"],
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "early_stopped": len(history) < config["max_epochs"],
        "selection_metric": "validation_rmse",
        "prediction_clipping_applied": False,
        "training_best_epoch_metrics": train_metrics,
        "validation_best_epoch_metrics": validation_metrics,
        "train_count": len(data.training_indices),
        "validation_count": len(data.validation_indices),
        "duration_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "final_metrics.json", final_metrics)
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_version": config["diagnostic_version"],
        "variant": variant,
        "fold": LOCKED_FOLD,
        "top_level_role": "model_development",
        "feature_cache_sha256": data.cache.file_sha256,
        "standardization_fitted_rows": data.standardization.fitted_row_count,
        "standardization_source": "Fold4 training rows only",
        "near_zero_dimension_count": int(data.standardization.near_zero_mask.sum()),
        "raw_images_accessed": False,
        "irradiance_used_as_model_input": False,
        "time_feature_used_as_model_input": False,
        "prediction_clipping_applied": False,
        "device": str(device),
        "output_files": list(REQUIRED_OUTPUT_FILES),
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)
    return final_metrics


def run_variants(variants: Sequence[str], config_path: Path = DEFAULT_CONFIG_PATH):
    requested = list(variants)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Variants must be a non-empty unique sequence")
    for variant in requested:
        validate_variant(variant)
    config = load_config(config_path)
    reject_non_fold4(config["fold"])
    data = prepare_diagnostic_data(config)
    return [run_variant(config, data, variant) for variant in requested]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen-DINOv2 regression-head diagnostics on Fold 4 only"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--variant", choices=ALLOWED_VARIANTS)
    group.add_argument("--all", action="store_true", help="Run A, then B, then C")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    variants = ALLOWED_VARIANTS if args.all else (args.variant,)
    results = run_variants(variants, args.config)
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
