"""Isolated Fold1-Fold4 ConvNeXt-Tiny image-only challenger training.

Importing this module never starts training. The memory smoke helper performs
exactly one optimizer step and writes no result files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import re
import socket
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
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_resnet50_image_only_date_grouped as resnet_baseline
from models.convnext_tiny_image_only import (
    OFFICIAL_CHECKPOINT_FILENAME,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_WEIGHTS,
    SolarConvNeXtTinyImageOnly,
    verify_official_checkpoint,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "splits" / "date_grouped_v1" / "split_manifest.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "PanelImages"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "convnext_tiny_image_only_v1_date_grouped.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "convnext_tiny_image_only_v1"
    / "pilot"
)
MODEL_NAME = "SolarConvNeXtTinyImageOnly"
MODEL_DEVELOPMENT_ROLE = "model_development"
ALLOWED_PILOT_FOLDS = (1, 2, 3, 4)
EXPECTED_MODEL_DEVELOPMENT_COUNT = 25_716
EXPECTED_MANIFEST_SHA256 = (
    "a354afc2b691719bf0cc3c3982033da833795006e3e3b0122cae07810bd83e02"
)
EXPECTED_FOLD_COUNTS = {
    1: (19_805, 5_911),
    2: (18_524, 7_192),
    3: (19_216, 6_500),
    4: (19_603, 6_113),
}
REQUIRED_MANIFEST_COLUMNS = (
    "filename",
    "date",
    "top_level_role",
    "cv_validation_fold",
)
REQUIRED_OUTPUT_FILES = (
    "best_model.pth",
    "final_metrics.json",
    "history.csv",
    "predictions.csv",
    "run_metadata.json",
    "config_snapshot.json",
)
PREDICTION_FIELDS = (
    "filename",
    "date",
    "fold",
    "true_L",
    "pred_L",
    "error",
    "abs_error",
)

# The transform function object itself is reused from the frozen formal
# ResNet50 Image-only baseline. No ConvNeXt-specific transform is substituted.
build_transforms = resnet_baseline.build_transforms
PREPROCESSING_DESCRIPTION = resnet_baseline.PREPROCESSING_DESCRIPTION

_LOSS_LABEL_PATTERN = re.compile(
    r"^solar_.+?_L_([0-9eE+\-.]+)_I_[0-9eE+\-.]+\.jpg$",
    flags=re.IGNORECASE,
)


def parse_loss_label(filename: str) -> float:
    """Parse only target L; irradiance and time are never returned as features."""

    match = _LOSS_LABEL_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Cannot parse normalized loss L from filename: {filename}")
    return float(match.group(1))


class ModelDevelopmentImageOnlyDataset(Dataset):
    """Dataset restricted to prefiltered model-development manifest records."""

    def __init__(self, records: pd.DataFrame, image_root: Path, transform=None):
        if records.empty:
            raise ValueError("Model-development records must not be empty")
        roles = set(records["top_level_role"].astype(str))
        if roles != {MODEL_DEVELOPMENT_ROLE}:
            raise ValueError(f"Dataset received disallowed roles: {sorted(roles)}")
        self.records = records.reset_index(drop=True).copy()
        self.image_root = Path(image_root).resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records.iloc[index]
        filename = str(record["filename"])
        image_path = self.image_root / filename
        # Only the selected model-development path is opened. The image root is
        # never enumerated by this challenger dataset.
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target_l = torch.tensor(parse_loss_label(filename), dtype=torch.float32)
        return image, target_l, filename, str(record["date"])


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
        "pilot_output_namespace",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Challenger config missing fields: {missing}")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported challenger config schema_version")
    return config


def validate_frozen_protocol(config: dict[str, Any]) -> None:
    expected = {
        "training_version": "convnext_tiny_image_only_v1_date_grouped",
        "split_version": "date_grouped_v1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "model_name": MODEL_NAME,
        "weights_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "weights_filename": OFFICIAL_CHECKPOINT_FILENAME,
        "weights_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "seed": 42,
        "epochs": 50,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "loss": "MSELoss",
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "early_stopping_patience": 8,
        "selection_metric": "validation_rmse",
        "amp": True,
        "pretrained": True,
        "dropout": 0.3,
        "allowed_pilot_folds": [1, 2, 3, 4],
        "model_development_sample_count": EXPECTED_MODEL_DEVELOPMENT_COUNT,
        "fold1_train_count": EXPECTED_FOLD_COUNTS[1][0],
        "fold1_validation_count": EXPECTED_FOLD_COUNTS[1][1],
        "fold2_train_count": EXPECTED_FOLD_COUNTS[2][0],
        "fold2_validation_count": EXPECTED_FOLD_COUNTS[2][1],
        "fold3_train_count": EXPECTED_FOLD_COUNTS[3][0],
        "fold3_validation_count": EXPECTED_FOLD_COUNTS[3][1],
        "fold4_train_count": EXPECTED_FOLD_COUNTS[4][0],
        "fold4_validation_count": EXPECTED_FOLD_COUNTS[4][1],
        "pilot_output_namespace": (
            "outputs/date_grouped_v1/convnext_tiny_image_only_v1/pilot"
        ),
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen ConvNeXt challenger protocol mismatch: {mismatches}")
    batch_policy = (config["batch_size"], config["gradient_accumulation_steps"])
    if batch_policy != (32, 1):
        raise ValueError(f"Unsupported smoke-derived batch policy: {batch_policy}")


def validate_pilot_fold(fold: int) -> None:
    if isinstance(fold, bool) or fold not in ALLOWED_PILOT_FOLDS:
        raise ValueError("ConvNeXt-Tiny image-only v1 permits only Fold1-Fold4")


def expected_output_dir(fold: int, seed: int = 42) -> Path:
    validate_pilot_fold(fold)
    if seed != 42:
        raise ValueError("The isolated v1 protocol requires seed=42")
    return OUTPUT_ROOT / f"fold_{fold}_seed_{seed}"


def _read_model_development_manifest(manifest_path: Path) -> pd.DataFrame:
    resolved_manifest = Path(manifest_path).resolve()
    if resolved_manifest != DEFAULT_MANIFEST.resolve():
        raise ValueError(
            "The challenger must use the frozen date_grouped_v1 split_manifest.csv"
        )
    actual_manifest_sha256 = resnet_baseline.sha256_file(resolved_manifest)
    if actual_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen split manifest SHA256 mismatch: "
            f"expected {EXPECTED_MANIFEST_SHA256}, got {actual_manifest_sha256}"
        )
    manifest = pd.read_csv(
        resolved_manifest, usecols=list(REQUIRED_MANIFEST_COLUMNS)
    )
    if manifest["filename"].isna().any() or manifest["filename"].duplicated().any():
        raise ValueError("Manifest filenames must be non-null and unique")
    records = manifest.loc[
        manifest["top_level_role"].eq(MODEL_DEVELOPMENT_ROLE)
    ].copy()
    if len(records) != EXPECTED_MODEL_DEVELOPMENT_COUNT:
        raise ValueError(
            "model_development count mismatch: "
            f"expected {EXPECTED_MODEL_DEVELOPMENT_COUNT}, got {len(records)}"
        )
    fold_values = pd.to_numeric(records["cv_validation_fold"], errors="raise")
    if not fold_values.map(float.is_integer).all():
        raise ValueError("model_development fold identifiers must be integers")
    records["cv_validation_fold"] = fold_values.astype(int)
    return records.reset_index(drop=True)


def preflight_fold(
    fold: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    verify_selected_paths: bool = False,
) -> dict[str, Any]:
    validate_pilot_fold(fold)
    if Path(image_root).resolve() != DEFAULT_IMAGE_ROOT.resolve():
        raise ValueError("The challenger must use the project PanelImages root")
    records = _read_model_development_manifest(Path(manifest_path))
    validation = records.loc[records["cv_validation_fold"].eq(fold)].copy()
    training = records.loc[records["cv_validation_fold"].ne(fold)].copy()
    expected_train, expected_validation = EXPECTED_FOLD_COUNTS[fold]
    if (len(training), len(validation)) != (expected_train, expected_validation):
        raise ValueError(
            f"Fold{fold} count mismatch: expected "
            f"{expected_train}/{expected_validation}, got "
            f"{len(training)}/{len(validation)}"
        )
    if set(training["date"]) & set(validation["date"]):
        raise ValueError("Fold training and validation dates overlap")
    if set(training["filename"]) & set(validation["filename"]):
        raise ValueError("Fold training and validation filenames overlap")
    if verify_selected_paths:
        root = Path(image_root).resolve()
        missing = [
            filename
            for filename in records["filename"].astype(str)
            if not (root / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} selected model-development images are missing"
            )
    return {
        "train_records": training.reset_index(drop=True),
        "validation_records": validation.reset_index(drop=True),
        "train_count": len(training),
        "validation_count": len(validation),
        "train_dates": sorted(training["date"].astype(str).unique().tolist()),
        "validation_dates": sorted(
            validation["date"].astype(str).unique().tolist()
        ),
        "selected_role": MODEL_DEVELOPMENT_ROLE,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "forbidden_roles_accessed": [],
        "final_test_accessed": False,
    }


def build_fold_datasets(
    fold: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    image_root: Path = DEFAULT_IMAGE_ROOT,
):
    audit = preflight_fold(fold, manifest_path, image_root)
    train_transform, validation_transform = build_transforms()
    train_dataset = ModelDevelopmentImageOnlyDataset(
        audit["train_records"], image_root, train_transform
    )
    validation_dataset = ModelDevelopmentImageOnlyDataset(
        audit["validation_records"], image_root, validation_transform
    )
    return train_dataset, validation_dataset, audit


def set_random_seeds(seed: int) -> None:
    # Reuse the formal baseline's exact Python/NumPy/Torch/CUDA/cuDNN settings.
    resnet_baseline.set_random_seeds(seed)


@dataclass
class MetricAccumulator:
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    target_sum: float = 0.0
    target_squared_sum: float = 0.0
    prediction_sum: float = 0.0
    sample_count: int = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred = predictions.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        true = targets.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
        errors = pred - true
        self.squared_error_sum += float(np.dot(errors, errors))
        self.absolute_error_sum += float(np.abs(errors).sum())
        self.target_sum += float(true.sum())
        self.target_squared_sum += float(np.dot(true, true))
        self.prediction_sum += float(pred.sum())
        self.sample_count += len(true)

    def compute(self) -> dict[str, float | int]:
        if self.sample_count == 0:
            raise ValueError("No samples accumulated")
        mse = self.squared_error_sum / self.sample_count
        target_sst = (
            self.target_squared_sum - self.target_sum**2 / self.sample_count
        )
        return {
            "loss": mse,
            "rmse": math.sqrt(mse),
            "mae": self.absolute_error_sum / self.sample_count,
            "r2": 1.0 - self.squared_error_sum / target_sst
            if target_sst > 0
            else 0.0,
            "sample_count": self.sample_count,
        }


def _unpack_batch(batch):
    images, target_l, filenames, dates = batch
    return images, target_l, filenames, dates


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_accumulation_steps: int = 1,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)
    accumulator = MetricAccumulator()
    started = time.perf_counter()
    if is_training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader, start=1):
        images, target_l, _, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        target_l = target_l.to(device, non_blocking=True).float().unsqueeze(1)
        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(
                device_type="cuda", enabled=amp_enabled and device.type == "cuda"
            ):
                predictions = model(images)
                loss = criterion(predictions, target_l)
            if is_training:
                scaled_loss = loss / gradient_accumulation_steps
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
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss detected")
        accumulator.update(predictions, target_l)
    metrics = accumulator.compute()
    metrics["duration_seconds"] = time.perf_counter() - started
    return metrics


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_rows(path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    true = np.asarray([row["true_L"] for row in rows], dtype=np.float64)
    pred = np.asarray([row["pred_L"] for row in rows], dtype=np.float64)
    errors = pred - true
    squared_error = float(np.dot(errors, errors))
    target_sst = float(np.dot(true - true.mean(), true - true.mean()))
    return {
        "r2": 1.0 - squared_error / target_sst if target_sst > 0 else 0.0,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std(ddof=0)),
        "true_mean": float(true.mean()),
        "true_std": float(true.std(ddof=0)),
        "bias": float(errors.mean()),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
        "true_min": float(true.min()),
        "true_max": float(true.max()),
    }


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
        images, target_l, filenames, dates = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", enabled=amp_enabled and device.type == "cuda"
        ):
            predictions = model(images).float().cpu().numpy().reshape(-1)
        true_values = target_l.float().numpy().reshape(-1)
        for filename, date, true_l, pred_l in zip(
            filenames, dates, true_values, predictions
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
    return rows, _prediction_summary(rows)


def _prepare_output_directory(output_dir: Path, fold: int, seed: int) -> Path:
    expected = expected_output_dir(fold, seed).resolve()
    actual = Path(output_dir).resolve()
    if actual != expected:
        raise RuntimeError(f"Output directory escaped isolated namespace: {actual}")
    if actual.exists():
        raise FileExistsError(f"Refusing to overwrite challenger output: {actual}")
    actual.mkdir(parents=True, exist_ok=False)
    return actual


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run one allowed challenger fold. This is never called by smoke tests."""

    validate_pilot_fold(args.fold)
    config = load_training_config(Path(args.config))
    validate_frozen_protocol(config)
    if args.seed != config["seed"]:
        raise ValueError("Seed override is prohibited")
    if args.epochs != config["epochs"]:
        raise ValueError("Epoch override is prohibited")
    if args.batch_size != config["batch_size"]:
        raise ValueError("Batch override must be recorded in the frozen config")
    if args.gradient_accumulation_steps != config["gradient_accumulation_steps"]:
        raise ValueError("Gradient accumulation override is prohibited")

    checkpoint_provenance = verify_official_checkpoint()
    audit = preflight_fold(args.fold, args.manifest, args.image_root)
    output_dir = _prepare_output_directory(args.output_dir, args.fold, args.seed)
    set_random_seeds(args.seed)
    train_dataset, validation_dataset, _ = build_fold_datasets(
        args.fold, args.manifest, args.image_root
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common_loader = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": resnet_baseline.seed_data_loader_worker,
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common_loader
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common_loader)
    model = SolarConvNeXtTinyImageOnly(dropout=0.3, use_pretrained=True).to(device)
    criterion = nn.MSELoss(reduction="mean")
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
        "pretrained_provenance": checkpoint_provenance,
        "final_test_accessed": False,
        "forbidden_roles_accessed": [],
    }
    _write_json(output_dir / "config_snapshot.json", config_snapshot)

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
            criterion,
            device,
            amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, amp_enabled
        )
        scheduler.step(validation_metrics["rmse"])
        improved = validation_metrics["rmse"] < best_rmse
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "validation_loss": validation_metrics["loss"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_mae": validation_metrics["mae"],
            "validation_r2": validation_metrics["r2"],
            "improved": improved,
        }
        history.append(row)
        _write_rows(output_dir / "history.csv", list(row), history)
        if improved:
            best_rmse = float(validation_metrics["rmse"])
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
                    "validation_metrics": validation_metrics,
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
    _write_rows(output_dir / "predictions.csv", PREDICTION_FIELDS, prediction_rows)
    duration = time.perf_counter() - started
    final_metrics = {
        **metrics,
        "fold": args.fold,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "selection_metric": "validation_rmse",
        "duration_seconds": duration,
    }
    _write_json(output_dir / "final_metrics.json", final_metrics)
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
        "final_test_accessed": False,
        "forbidden_roles_accessed": [],
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)
    return final_metrics


def run_single_batch_memory_smoke(
    batch_size: int,
    fold: int = 4,
    manifest_path: Path = DEFAULT_MANIFEST,
    image_root: Path = DEFAULT_IMAGE_ROOT,
) -> dict[str, Any]:
    """Run one fixed real-image AMP optimizer step and write no output files."""

    if batch_size not in {16, 32}:
        raise ValueError("Memory smoke permits only batch_size 16 or 32")
    validate_pilot_fold(fold)
    if not torch.cuda.is_available():
        return {
            "batch": batch_size,
            "peak_allocated_GB": None,
            "peak_reserved_GB": None,
            "STATUS": "SKIP_NO_CUDA",
        }
    config = load_training_config()
    validate_frozen_protocol(config)
    verify_official_checkpoint()
    set_random_seeds(config["seed"])
    train_dataset, _, _ = build_fold_datasets(fold, manifest_path, image_root)
    fixed_dataset = Subset(train_dataset, range(batch_size))
    loader = DataLoader(fixed_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = torch.device("cuda:0")
    status = "PASS"
    model = None
    optimizer = None
    scaler = None
    loss = None
    predictions = None
    try:
        torch.cuda.empty_cache()
        model = SolarConvNeXtTinyImageOnly(dropout=0.3, use_pretrained=True).to(device)
        criterion = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        images, target_l, _, _ = _unpack_batch(batch)
        images = images.to(device)
        target_l = target_l.to(device).float().unsqueeze(1)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            predictions = model(images)
            loss = criterion(predictions, target_l)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    except torch.OutOfMemoryError:
        status = "OOM"
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    finally:
        del predictions, loss, scaler, optimizer, model, batch, loader, fixed_dataset
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    return {
        "batch": batch_size,
        "peak_allocated_GB": peak_allocated / 1_000_000_000,
        "peak_reserved_GB": peak_reserved / 1_000_000_000,
        "STATUS": status,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config = load_training_config()
    parser = argparse.ArgumentParser(
        description="Run the isolated Fold1-Fold4 ConvNeXt-Tiny image-only challenger"
    )
    parser.add_argument("--fold", type=int, default=4)
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
