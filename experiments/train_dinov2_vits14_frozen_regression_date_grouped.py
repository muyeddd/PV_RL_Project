"""Train an isolated date-grouped regressor on frozen DINOv2-S/14 features.

This module reads only the formal feature cache. It contains no image-loading or
feature-extraction path. Pilot training is restricted to Fold 3 and Fold 4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
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

from models.dinov2_frozen_feature_regressor import DINOv2FrozenFeatureRegressor


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "dinov2_vits14_frozen_regression_v1_date_grouped.json"
)
EXPECTED_CACHE_DIR = PROJECT_ROOT / "features" / "dinov2_vits14_frozen_v1"
PILOT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "dinov2_vits14_frozen_regression_v1"
    / "pilot"
)
ALLOWED_PILOT_FOLDS = (3, 4)
METADATA_COLUMNS = ("row_index", "filename", "date", "cv_validation_fold", "L")
REQUIRED_OUTPUT_FILES = (
    "best_model.pth",
    "final_metrics.json",
    "history.csv",
    "predictions.csv",
    "run_metadata.json",
    "config_snapshot.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ordered_strings(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "training_version",
        "split_version",
        "feature_version",
        "model_name",
        "feature_cache_dir",
        "pilot_output_root",
        "allowed_pilot_folds",
        "seed",
        "input_dimension",
        "hidden_dimension",
        "dropout",
        "output_dimension",
        "output_activation",
        "feature_standardization",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "num_workers",
        "max_epochs",
        "scheduler",
        "scheduler_factor",
        "scheduler_patience",
        "early_stopping_patience",
        "selection_metric",
        "amp",
        "expected_sample_count",
        "expected_feature_shape",
        "expected_feature_dtype",
        "expected_fold_validation_counts",
        "expected_fold_validation_dates",
        "feature_cache_sha256",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Regression config is missing fields: {missing}")
    expected_values = {
        "schema_version": 1,
        "training_version": "dinov2_vits14_frozen_regression_v1_date_grouped",
        "split_version": "date_grouped_v1",
        "feature_version": "dinov2_vits14_frozen_v1",
        "model_name": "DINOv2FrozenFeatureRegressor",
        "allowed_pilot_folds": [3, 4],
        "seed": 42,
        "input_dimension": 384,
        "hidden_dimension": 128,
        "dropout": 0.3,
        "output_dimension": 1,
        "output_activation": "Sigmoid",
        "feature_standardization": "none",
        "loss": "MSELoss",
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "num_workers": 0,
        "max_epochs": 50,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "early_stopping_patience": 8,
        "selection_metric": "validation_rmse",
        "amp": False,
        "expected_sample_count": 25716,
        "expected_feature_shape": [25716, 384],
        "expected_feature_dtype": "float32",
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_values.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Frozen regression v1 config mismatch: {mismatches}")
    cache_dir = resolve_project_path(config["feature_cache_dir"]).resolve()
    output_root = resolve_project_path(config["pilot_output_root"]).resolve()
    if cache_dir != EXPECTED_CACHE_DIR.resolve():
        raise ValueError("Feature cache path is not the frozen DINOv2 v1 cache")
    if output_root != PILOT_OUTPUT_ROOT.resolve():
        raise ValueError("Pilot output root is not isolated")
    return config


def validate_pilot_fold(fold: int) -> None:
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in ALLOWED_PILOT_FOLDS:
        raise ValueError("DINOv2 frozen regression pilot allows only Fold 3 and Fold 4")


def expected_output_dir(fold: int, seed: int = 42) -> Path:
    validate_pilot_fold(fold)
    if seed != 42:
        raise ValueError("Frozen pilot requires seed=42")
    output = PILOT_OUTPUT_ROOT / f"fold_{fold}_seed_{seed}"
    if output.resolve().parent != PILOT_OUTPUT_ROOT.resolve():
        raise ValueError("Pilot output escaped its isolated root")
    return output


@dataclass
class FeatureCache:
    features: np.ndarray
    metadata: pd.DataFrame
    feature_manifest: dict[str, Any]
    dataset_fingerprint: dict[str, Any]
    file_sha256: dict[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_and_validate_feature_cache(config: dict[str, Any]) -> FeatureCache:
    cache_dir = resolve_project_path(config["feature_cache_dir"]).resolve()
    paths = {
        "features.npy": cache_dir / "features.npy",
        "metadata.csv": cache_dir / "metadata.csv",
        "feature_manifest.json": cache_dir / "feature_manifest.json",
        "dataset_fingerprint.json": cache_dir / "dataset_fingerprint.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Formal feature cache file is missing: {name}")

    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected_hash in config["feature_cache_sha256"].items():
        if actual_hashes.get(name, "").lower() != expected_hash.lower():
            raise ValueError(
                f"Formal feature cache SHA256 mismatch for {name}: "
                f"expected={expected_hash}, actual={actual_hashes.get(name)}"
            )

    feature_manifest = _read_json(paths["feature_manifest.json"])
    dataset_fingerprint = _read_json(paths["dataset_fingerprint.json"])
    manifest_expected = {
        "sample_count": 25716,
        "feature_dimension": 384,
        "feature_dtype": "float32",
        "feature_version": "dinov2_vits14_frozen_v1",
        "split_version": "date_grouped_v1",
        "top_level_role": "model_development",
        "irradiance_used_as_model_input": False,
    }
    for key, expected in manifest_expected.items():
        if feature_manifest.get(key) != expected:
            raise ValueError(f"Feature manifest mismatch for {key}")
    if feature_manifest.get("features_sha256", "").lower() != actual_hashes[
        "features.npy"
    ].lower():
        raise ValueError("features.npy does not match feature_manifest.json")
    if feature_manifest.get("metadata_sha256", "").lower() != actual_hashes[
        "metadata.csv"
    ].lower():
        raise ValueError("metadata.csv does not match feature_manifest.json")
    if feature_manifest.get("dataset_fingerprint_sha256", "").lower() != actual_hashes[
        "dataset_fingerprint.json"
    ].lower():
        raise ValueError("dataset_fingerprint.json does not match feature_manifest.json")

    fingerprint_expected = {
        "feature_version": "dinov2_vits14_frozen_v1",
        "split_version": "date_grouped_v1",
        "top_level_role": "model_development",
        "sample_count": 25716,
        "ordering": "filename_ascending",
    }
    for key, expected in fingerprint_expected.items():
        if dataset_fingerprint.get(key) != expected:
            raise ValueError(f"Dataset fingerprint mismatch for {key}")

    features = np.load(paths["features.npy"], mmap_mode="r", allow_pickle=False)
    if list(features.shape) != config["expected_feature_shape"]:
        raise ValueError(f"Unexpected formal feature shape: {features.shape}")
    if str(features.dtype) != config["expected_feature_dtype"]:
        raise ValueError(f"Unexpected formal feature dtype: {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("Formal feature cache contains NaN or Inf")

    metadata = pd.read_csv(paths["metadata.csv"])
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise ValueError(f"Unexpected metadata columns: {metadata.columns.tolist()}")
    if len(metadata) != config["expected_sample_count"]:
        raise ValueError("Unexpected metadata row count")
    if not np.array_equal(metadata["row_index"].to_numpy(), np.arange(len(metadata))):
        raise ValueError("metadata row_index is not exactly aligned to features.npy")
    if metadata["filename"].isna().any() or not metadata["filename"].is_unique:
        raise ValueError("metadata filenames must be non-null and unique")
    filenames = metadata["filename"].astype(str).tolist()
    if filenames != sorted(filenames):
        raise ValueError("metadata is not in frozen filename order")
    if sha256_ordered_strings(filenames) != dataset_fingerprint.get(
        "ordered_filenames_sha256"
    ):
        raise ValueError("metadata filename order does not match dataset fingerprint")
    labels = pd.to_numeric(metadata["L"], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(labels).all():
        raise ValueError("metadata L contains NaN or Inf")
    metadata["L"] = labels
    folds = pd.to_numeric(metadata["cv_validation_fold"], errors="raise")
    fold_array = folds.to_numpy(dtype=np.float64)
    if not np.isfinite(fold_array).all() or not np.equal(
        fold_array, np.floor(fold_array)
    ).all():
        raise ValueError("cv_validation_fold must contain integers")
    metadata["cv_validation_fold"] = folds.astype(int)
    if set(metadata["cv_validation_fold"]) != {1, 2, 3, 4}:
        raise ValueError("metadata must preserve exactly date_grouped_v1 folds 1-4")
    for fold in (1, 2, 3, 4):
        fold_rows = metadata[metadata["cv_validation_fold"].eq(fold)]
        expected_count = config["expected_fold_validation_counts"][str(fold)]
        expected_dates = config["expected_fold_validation_dates"][str(fold)]
        if len(fold_rows) != expected_count:
            raise ValueError(f"Fold {fold} validation count mismatch")
        if sorted(fold_rows["date"].astype(str).unique().tolist()) != expected_dates:
            raise ValueError(f"Fold {fold} validation dates mismatch")

    return FeatureCache(
        features=features,
        metadata=metadata,
        feature_manifest=feature_manifest,
        dataset_fingerprint=dataset_fingerprint,
        file_sha256=actual_hashes,
    )


def fold_indices(cache: FeatureCache, fold: int) -> tuple[np.ndarray, np.ndarray]:
    validate_pilot_fold(fold)
    fold_values = cache.metadata["cv_validation_fold"].to_numpy(dtype=np.int64)
    validation_indices = np.flatnonzero(fold_values == fold)
    training_indices = np.flatnonzero(fold_values != fold)
    if np.intersect1d(training_indices, validation_indices).size:
        raise RuntimeError("Training and validation feature rows overlap")
    if len(training_indices) + len(validation_indices) != 25716:
        raise RuntimeError("Fold does not cover all model-development feature rows")
    return training_indices, validation_indices


class FeatureRegressionDataset(Dataset):
    def __init__(self, cache: FeatureCache, indices: np.ndarray):
        self.features = cache.features
        self.labels = cache.metadata["L"].to_numpy(dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        feature = np.asarray(self.features[row], dtype=np.float32).copy()
        return torch.from_numpy(feature), torch.tensor(self.labels[row], dtype=torch.float32)


def set_random_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any]) -> DINOv2FrozenFeatureRegressor:
    return DINOv2FrozenFeatureRegressor(
        input_dim=config["input_dimension"],
        hidden_dim=config["hidden_dimension"],
        dropout=config["dropout"],
    )


def build_loss(config: dict[str, Any]) -> nn.Module:
    if config["loss"] != "MSELoss":
        raise ValueError("Frozen regression v1 requires MSELoss")
    return nn.MSELoss(reduction="mean")


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    if config["optimizer"] != "AdamW":
        raise ValueError("Frozen regression v1 requires AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


def build_scheduler(optimizer, config: dict[str, Any]):
    if config["scheduler"] != "ReduceLROnPlateau":
        raise ValueError("Frozen regression v1 requires ReduceLROnPlateau")
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )


def compute_metrics(true_values: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    true_values = np.asarray(true_values, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if true_values.shape != predictions.shape or true_values.size == 0:
        raise ValueError("Metric arrays must have equal non-zero shape")
    if not np.isfinite(true_values).all() or not np.isfinite(predictions).all():
        raise ValueError("Metrics received NaN or Inf")
    errors = predictions - true_values
    squared_error = float(np.dot(errors, errors))
    centered = true_values - true_values.mean()
    total_variance = float(np.dot(centered, centered))
    return {
        "r2": float(1.0 - squared_error / total_variance) if total_variance > 0 else 0.0,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
        "true_mean": float(true_values.mean()),
        "true_std": float(true_values.std()),
        "bias": float(errors.mean()),
        "pred_min": float(predictions.min()),
        "pred_max": float(predictions.max()),
        "true_min": float(true_values.min()),
        "true_max": float(true_values.max()),
        "sample_count": int(true_values.size),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    count = 0
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(features)
            loss = criterion(predictions, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_count = labels.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_count
        count += batch_count
        all_predictions.append(predictions.detach().float().cpu().numpy().reshape(-1))
        all_labels.append(labels.detach().float().cpu().numpy().reshape(-1))
    if count == 0:
        raise RuntimeError("Empty feature DataLoader")
    return loss_sum / count, np.concatenate(all_labels), np.concatenate(all_predictions)


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


def prepare_output_directory(fold: int, seed: int) -> Path:
    output_dir = expected_output_dir(fold, seed)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing pilot output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_training(fold: int, config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Explicit training entry point. Tests must not call this function."""

    validate_pilot_fold(fold)
    config = load_config(config_path)
    cache = load_and_validate_feature_cache(config)
    train_indices, validation_indices = fold_indices(cache, fold)
    expected_validation = config["expected_fold_validation_counts"][str(fold)]
    if len(validation_indices) != expected_validation:
        raise ValueError("Pilot validation count mismatch")
    set_random_seeds(config["seed"])
    output_dir = prepare_output_directory(fold, config["seed"])
    _write_json(output_dir / "config_snapshot.json", config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        FeatureRegressionDataset(cache, train_indices),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        FeatureRegressionDataset(cache, validation_indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    history: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, true_values, predictions = run_epoch(
            model, validation_loader, criterion, device
        )
        validation_metrics = compute_metrics(true_values, predictions)
        scheduler.step(validation_metrics["rmse"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
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
                    "fold": fold,
                    "seed": config["seed"],
                    "validation_metrics": validation_metrics,
                    "training_version": config["training_version"],
                    "feature_cache_sha256": cache.file_sha256,
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
    _, true_values, predictions = run_epoch(model, validation_loader, criterion, device)
    best_metrics = compute_metrics(true_values, predictions)
    validation_rows = cache.metadata.iloc[validation_indices].reset_index(drop=True)
    prediction_frame = pd.DataFrame(
        {
            "filename": validation_rows["filename"].astype(str),
            "date": validation_rows["date"].astype(str),
            "fold": fold,
            "true_L": true_values,
            "pred_L": predictions,
        }
    )
    prediction_frame["error"] = prediction_frame["pred_L"] - prediction_frame["true_L"]
    prediction_frame["abs_error"] = prediction_frame["error"].abs()
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False, lineterminator="\n")

    final_metrics = {
        "fold": fold,
        "seed": config["seed"],
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "early_stopped": len(history) < config["max_epochs"],
        "selection_metric": "validation_rmse",
        "validation_metrics": best_metrics,
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "duration_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "final_metrics.json", final_metrics)
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_run": True,
        "training_version": config["training_version"],
        "feature_version": config["feature_version"],
        "split_version": "date_grouped_v1",
        "top_level_role": "model_development",
        "fold": fold,
        "train_dates": sorted(
            cache.metadata.iloc[train_indices]["date"].astype(str).unique().tolist()
        ),
        "validation_dates": sorted(validation_rows["date"].unique().tolist()),
        "feature_cache_sha256": cache.file_sha256,
        "raw_images_accessed": False,
        "irradiance_used_as_model_input": False,
        "time_feature_used_as_model_input": False,
        "feature_standardization": "none",
        "device": str(device),
        "output_files": list(REQUIRED_OUTPUT_FILES),
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)
    return final_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen DINOv2-S/14 feature regression pilot"
    )
    parser.add_argument("--fold", type=int, choices=ALLOWED_PILOT_FOLDS, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run_training(args.fold, args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
