"""Leakage-free date-grouped ResNet50 Image-only training entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet50_image_only import SolarResNet50ImageOnly
from utils.date_grouped_training_data import (
    FORBIDDEN_MODEL_ROLES,
    build_fold_datasets,
    load_fold_records,
    validate_fold_isolation,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "splits" / "date_grouped_v1" / "split_manifest.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "PanelImages"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "resnet50_image_only_date_grouped_v1.json"
)
EXPECTED_FOLD_COUNTS = {
    1: (19805, 5911),
    2: (18524, 7192),
    3: (19216, 6500),
    4: (19603, 6113),
}
MODEL_NAME = "SolarResNet50ImageOnly"
PREPROCESSING_DESCRIPTION = {
    "train": (
        "Resize(256,256); RandomResizedCrop(224, scale=(0.85,1.0)); "
        "RandomHorizontalFlip(0.5); RandomRotation(7); "
        "ColorJitter(brightness=0.08,contrast=0.08,saturation=0.05,hue=0.02); "
        "ToTensor; ImageNet normalization"
    ),
    "validation": "Resize(224,224); ToTensor; ImageNet normalization",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
}
REQUIRED_CHECKPOINT_FIELDS = {
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "fold",
    "seed",
    "split_version",
    "manifest_sha256",
    "dataset_fingerprint",
    "train_dates",
    "validation_dates",
    "pretrained_used",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "best_validation_metrics",
    "model_name",
    "image_preprocessing",
}


@dataclass
class MetricAccumulator:
    loss_sum: float = 0.0
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    target_sum: float = 0.0
    target_squared_sum: float = 0.0
    sample_count: int = 0

    def update(
        self,
        mean_batch_loss: float,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        predictions_np = predictions.detach().float().cpu().numpy().reshape(-1)
        targets_np = targets.detach().float().cpu().numpy().reshape(-1)
        if len(predictions_np) != len(targets_np):
            raise ValueError("Predictions and targets have different batch sizes")
        count = len(targets_np)
        errors = predictions_np.astype(np.float64) - targets_np.astype(np.float64)
        targets64 = targets_np.astype(np.float64)
        self.loss_sum += float(mean_batch_loss) * count
        self.squared_error_sum += float(np.dot(errors, errors))
        self.absolute_error_sum += float(np.abs(errors).sum())
        self.target_sum += float(targets64.sum())
        self.target_squared_sum += float(np.dot(targets64, targets64))
        self.sample_count += count

    def compute(self) -> dict[str, Any]:
        if self.sample_count == 0:
            raise ValueError("Cannot compute metrics without samples")
        count = self.sample_count
        mse = self.squared_error_sum / count
        target_sst = self.target_squared_sum - (self.target_sum**2) / count
        r2 = 1.0 - self.squared_error_sum / target_sst if target_sst > 0 else 0.0
        return {
            "loss": self.loss_sum / count,
            "mae": self.absolute_error_sum / count,
            "rmse": float(np.sqrt(mse)),
            "r2": r2,
            "sample_count": count,
        }


def sample_weighted_average(values: Sequence[float], sample_counts: Sequence[int]) -> float:
    if len(values) != len(sample_counts) or not values:
        raise ValueError("values and sample_counts must have equal non-zero length")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("sample_counts must be positive")
    return sum(value * count for value, count in zip(values, sample_counts)) / sum(
        sample_counts
    )


def load_training_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "training_version",
        "split_version",
        "model_name",
        "seed",
        "epochs",
        "batch_size",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "patience",
        "amp",
        "pretrained",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Training config is missing fields: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported training config schema_version")
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = load_training_config()
    parser = argparse.ArgumentParser(
        description="Train ResNet50 Image-only using the frozen date_grouped_v1 manifest"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument(
        "--learning-rate", type=float, default=defaults["learning_rate"]
    )
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--patience", type=int, default=defaults["patience"])
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=defaults["amp"]
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=defaults["pretrained"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot-run", action="store_true")
    return parser.parse_args(argv)


def validate_fold_number(fold: int) -> None:
    if isinstance(fold, bool) or fold not in EXPECTED_FOLD_COUNTS:
        raise ValueError("fold must be an integer from 1 to 4")


def validate_training_arguments(args: argparse.Namespace) -> None:
    validate_fold_number(args.fold)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if args.patience <= 0:
        raise ValueError("patience must be positive")


def prepare_output_directory(output_dir: Path, overwrite: bool) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Pass --overwrite explicitly to replace named run artifacts."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_sha256(path: Path) -> str:
    """Hash manifest content with platform-independent LF line endings.

    The frozen fingerprint was produced from the generator's LF CSV bytes.  Git
    may materialize the tracked CSV with CRLF on Windows, so hashing canonical
    CSV bytes avoids treating an end-of-line checkout conversion as a changed
    split while still detecting any content or ordering change.
    """
    content = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def load_and_validate_dataset_fingerprint(manifest_path: Path) -> dict[str, Any]:
    fingerprint_path = Path(manifest_path).resolve().parent / "dataset_fingerprint.json"
    if not fingerprint_path.is_file():
        raise FileNotFoundError(
            f"Dataset fingerprint is missing beside manifest: {fingerprint_path}"
        )
    with fingerprint_path.open("r", encoding="utf-8") as handle:
        fingerprint = json.load(handle)
    actual_manifest_sha256 = canonical_manifest_sha256(manifest_path)
    expected_manifest_sha256 = fingerprint.get("manifest_sha256")
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    if fingerprint.get("split_version") != "date_grouped_v1":
        raise ValueError("Manifest fingerprint is not date_grouped_v1")
    return fingerprint


def preflight_fold(
    manifest_path: Path,
    image_root: Path,
    fold: int,
) -> dict[str, Any]:
    validate_fold_number(fold)
    fingerprint = load_and_validate_dataset_fingerprint(manifest_path)
    train_records, validation_records = load_fold_records(
        manifest_path, image_root, fold
    )
    validate_fold_isolation(train_records, validation_records)

    expected_train, expected_validation = EXPECTED_FOLD_COUNTS[fold]
    if len(train_records) != expected_train or len(validation_records) != expected_validation:
        raise ValueError(
            f"Fold {fold} count mismatch: expected "
            f"{expected_train}/{expected_validation}, got "
            f"{len(train_records)}/{len(validation_records)}"
        )

    manifest = pd.read_csv(manifest_path, usecols=["filename", "top_level_role"])
    selected_filenames = set(train_records["filename"]) | set(
        validation_records["filename"]
    )
    forbidden_counts = {
        role: int(
            manifest[
                manifest["top_level_role"].eq(role)
                & manifest["filename"].isin(selected_filenames)
            ].shape[0]
        )
        for role in FORBIDDEN_MODEL_ROLES
    }
    if any(forbidden_counts.values()):
        raise ValueError(f"Forbidden roles entered model fold: {forbidden_counts}")

    return {
        "fingerprint": fingerprint,
        "manifest_sha256": fingerprint["manifest_sha256"],
        "train_records": train_records,
        "validation_records": validation_records,
        "train_dates": sorted(train_records["date"].unique().tolist()),
        "validation_dates": sorted(validation_records["date"].unique().tolist()),
        "forbidden_role_counts": forbidden_counts,
    }


def set_random_seeds(seed: int) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Preserve GPU throughput while recording that strict determinism is disabled.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "cuda_seed": seed if torch.cuda.is_available() else None,
        "pythonhashseed": str(seed),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "strict_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def seed_data_loader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def build_transforms():
    normalization = transforms.Normalize(
        PREPROCESSING_DESCRIPTION["normalization_mean"],
        PREPROCESSING_DESCRIPTION["normalization_std"],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, validation_transform


def unpack_batch(batch):
    if len(batch) < 2:
        raise ValueError("Expected image and loss label")
    return batch[0], batch[1]


def run_epoch(
    model,
    loader,
    criterion,
    device,
    amp_enabled: bool,
    optimizer=None,
    scaler=None,
    phase: str = "train",
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)
    accumulator = MetricAccumulator()
    total_batches = len(loader)
    phase_start = time.perf_counter()

    for batch_index, batch in enumerate(loader, start=1):
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp_enabled and device.type == "cuda",
            ):
                predictions = model(images)
                loss = criterion(predictions, labels)
            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        accumulator.update(float(loss.detach().cpu()), predictions, labels)
        if batch_index == 1 or batch_index % 200 == 0 or batch_index == total_batches:
            elapsed = time.perf_counter() - phase_start
            print(
                f"{phase} batch {batch_index}/{total_batches} "
                f"samples={accumulator.sample_count} elapsed={elapsed:.1f}s",
                flush=True,
            )

    metrics = accumulator.compute()
    metrics["duration_seconds"] = time.perf_counter() - phase_start
    metrics["samples_per_second"] = (
        metrics["sample_count"] / metrics["duration_seconds"]
    )
    metrics["non_finite_detected"] = not all(
        math.isfinite(float(metrics[name]))
        for name in ("loss", "mae", "rmse", "r2")
    )
    return metrics


def query_gpu_telemetry() -> dict[str, Any]:
    """Collect lightweight stability telemetry without adding a dependency."""

    observed_at = datetime.now(timezone.utc).isoformat()
    if not torch.cuda.is_available():
        return {
            "observed_at_utc": observed_at,
            "query_success": False,
            "error": "CUDA is unavailable",
        }
    query_fields = (
        "temperature.gpu",
        "clocks.current.sm",
        "clocks.max.sm",
        "utilization.gpu",
        "clocks_event_reasons.sw_thermal_slowdown",
        "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.hw_power_brake_slowdown",
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [item.strip() for item in completed.stdout.strip().split(",")]
        if len(values) != len(query_fields):
            raise ValueError(f"Expected {len(query_fields)} values, got {len(values)}")
        return {
            "observed_at_utc": observed_at,
            "query_success": True,
            "temperature_c": float(values[0]),
            "sm_clock_mhz": float(values[1]),
            "sm_clock_max_mhz": float(values[2]),
            "gpu_utilization_percent": float(values[3]),
            "software_thermal_slowdown": values[4],
            "hardware_thermal_slowdown": values[5],
            "hardware_power_brake_slowdown": values[6],
        }
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {
            "observed_at_utc": observed_at,
            "query_success": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _pretrained_provenance(requested: bool) -> dict[str, Any]:
    if not requested:
        return {
            "pretrained_requested": False,
            "pretrained_used": False,
            "pretrained_source": None,
            "pretrained_cache_path": None,
            "pretrained_cache_present_before_init": False,
            "pretrained_load_success": False,
        }
    weights = ResNet50_Weights.DEFAULT
    checkpoint_name = Path(urlparse(weights.url).path).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    return {
        "pretrained_requested": True,
        "pretrained_used": True,
        "pretrained_source": f"torchvision:{weights}",
        "pretrained_cache_path": str(cache_path),
        "pretrained_cache_present_before_init": cache_path.is_file(),
        "pretrained_load_success": False,
    }


def build_checkpoint_payload(
    model,
    optimizer,
    epoch: int,
    run_context: dict[str, Any],
    best_validation_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    scheduler=None,
    checkpoint_kind: str = "best",
) -> dict[str, Any]:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "fold": run_context["fold"],
        "seed": run_context["seed"],
        "split_version": run_context["split_version"],
        "manifest_sha256": run_context["manifest_sha256"],
        "dataset_fingerprint": run_context["dataset_fingerprint"],
        "train_dates": run_context["train_dates"],
        "validation_dates": run_context["validation_dates"],
        "pretrained_used": run_context["pretrained_used"],
        "pretrained_source": run_context["pretrained_source"],
        "pretrained_load_success": run_context["pretrained_load_success"],
        "batch_size": run_context["batch_size"],
        "learning_rate": run_context["learning_rate"],
        "weight_decay": run_context["weight_decay"],
        "best_validation_metrics": best_validation_metrics,
        "current_metrics": current_metrics,
        "model_name": MODEL_NAME,
        "image_preprocessing": PREPROCESSING_DESCRIPTION,
        "checkpoint_kind": checkpoint_kind,
        "training_version": run_context["training_version"],
        "pilot_run": run_context["pilot_run"],
        "NOT_FOR_RESEARCH_METRICS": run_context["pilot_run"],
    }
    missing = sorted(REQUIRED_CHECKPOINT_FIELDS - set(payload))
    if missing:
        raise RuntimeError(f"Checkpoint payload is missing fields: {missing}")
    return payload


def _write_json(path: Path, value: MappingLike) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


MappingLike = dict[str, Any]


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        raise ValueError("Cannot write empty metrics history")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    validate_training_arguments(args)
    preflight = preflight_fold(args.manifest, args.image_root, args.fold)
    output_dir = prepare_output_directory(args.output_dir, args.overwrite)
    seed_settings = set_random_seeds(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_transform, validation_transform = build_transforms()
    train_dataset, validation_dataset, train_records, validation_records = (
        build_fold_datasets(
            args.manifest,
            args.image_root,
            args.fold,
            train_transform,
            validation_transform,
        )
    )

    loader_generator = make_data_loader_generator(args.seed)
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_data_loader_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **common_loader_args,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_args,
    )

    pretrained = _pretrained_provenance(args.pretrained)
    model = SolarResNet50ImageOnly(
        dropout=0.3,
        use_pretrained=args.pretrained,
    ).to(device)
    pretrained["pretrained_load_success"] = True if args.pretrained else False

    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    run_context = {
        "training_version": load_training_config()["training_version"],
        "split_version": preflight["fingerprint"]["split_version"],
        "fold": args.fold,
        "seed": args.seed,
        "manifest_sha256": preflight["manifest_sha256"],
        "dataset_fingerprint": preflight["fingerprint"],
        "train_dates": preflight["train_dates"],
        "validation_dates": preflight["validation_dates"],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pilot_run": args.pilot_run,
        **pretrained,
    }
    config_snapshot = {
        **run_context,
        "manifest": str(Path(args.manifest).resolve()),
        "image_root": str(Path(args.image_root).resolve()),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "patience": args.patience,
        "amp_requested": args.amp,
        "amp_enabled": amp_enabled,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=2)",
        "early_stopping_metric": "validation_rmse",
        "random_seed_settings": seed_settings,
        "image_preprocessing": PREPROCESSING_DESCRIPTION,
        "forbidden_role_counts": preflight["forbidden_role_counts"],
        "full_train_count": len(train_records),
        "full_validation_count": len(validation_records),
        "overwrite": args.overwrite,
        "PILOT_RUN": args.pilot_run,
        "NOT_FOR_RESEARCH_METRICS": args.pilot_run,
    }
    _write_json(output_dir / "training_config_snapshot.json", config_snapshot)

    run_started_utc = datetime.now(timezone.utc)
    run_started_perf = time.perf_counter()
    print(f"Fold {args.fold}: train={len(train_dataset)}, validation={len(validation_dataset)}")
    print(f"Train dates: {preflight['train_dates']}")
    print(f"Validation dates: {preflight['validation_dates']}")
    print(f"Forbidden role counts: {preflight['forbidden_role_counts']}")
    print(f"Device: {device}; AMP enabled: {amp_enabled}")
    print(f"Pretrained: {pretrained}")
    print(f"PILOT_RUN: {str(args.pilot_run).lower()}")
    print(f"NOT_FOR_RESEARCH_METRICS: {str(args.pilot_run).lower()}", flush=True)

    history = []
    best_validation_rmse = float("inf")
    best_validation_metrics: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    final_train_metrics = None
    final_validation_metrics = None
    gpu_telemetry_samples: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        telemetry_epoch_start = query_gpu_telemetry()
        telemetry_epoch_start["epoch"] = epoch
        telemetry_epoch_start["stage"] = "epoch_start"
        gpu_telemetry_samples.append(telemetry_epoch_start)
        print(f"Epoch {epoch}/{args.epochs} starting", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            phase="train",
        )
        telemetry_after_train = query_gpu_telemetry()
        telemetry_after_train["epoch"] = epoch
        telemetry_after_train["stage"] = "after_train"
        gpu_telemetry_samples.append(telemetry_after_train)
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            amp_enabled,
            optimizer=None,
            scaler=None,
            phase="validation",
        )
        telemetry_after_validation = query_gpu_telemetry()
        telemetry_after_validation["epoch"] = epoch
        telemetry_after_validation["stage"] = "after_validation"
        gpu_telemetry_samples.append(telemetry_after_validation)
        scheduler.step(validation_metrics["rmse"])

        epoch_peak_allocated = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        epoch_peak_reserved = (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        )
        epoch_non_finite = bool(
            train_metrics["non_finite_detected"]
            or validation_metrics["non_finite_detected"]
        )
        successful_telemetry = [
            sample
            for sample in (
                telemetry_epoch_start,
                telemetry_after_train,
                telemetry_after_validation,
            )
            if sample.get("query_success")
        ]
        epoch_temperatures = [
            sample["temperature_c"] for sample in successful_telemetry
        ]
        epoch_sm_clocks = [sample["sm_clock_mhz"] for sample in successful_telemetry]
        epoch_thermal_slowdown = any(
            sample["software_thermal_slowdown"] == "Active"
            or sample["hardware_thermal_slowdown"] == "Active"
            for sample in successful_telemetry
        )
        epoch_power_brake_slowdown = any(
            sample["hardware_power_brake_slowdown"] == "Active"
            for sample in successful_telemetry
        )

        improved = validation_metrics["rmse"] < best_validation_rmse
        if improved:
            best_validation_rmse = validation_metrics["rmse"]
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_duration_seconds = time.perf_counter() - epoch_start
        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "train_sample_count": train_metrics["sample_count"],
            "train_duration_seconds": train_metrics["duration_seconds"],
            "train_samples_per_second": train_metrics["samples_per_second"],
            "validation_loss": validation_metrics["loss"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_r2": validation_metrics["r2"],
            "validation_sample_count": validation_metrics["sample_count"],
            "validation_duration_seconds": validation_metrics["duration_seconds"],
            "validation_samples_per_second": validation_metrics[
                "samples_per_second"
            ],
            "epoch_duration_seconds": epoch_duration_seconds,
            "epoch_samples_per_second": (
                (train_metrics["sample_count"] + validation_metrics["sample_count"])
                / epoch_duration_seconds
            ),
            "gpu_peak_allocated_bytes": epoch_peak_allocated,
            "gpu_peak_reserved_bytes": epoch_peak_reserved,
            "non_finite_detected": epoch_non_finite,
            "gpu_temperature_min_c": min(epoch_temperatures)
            if epoch_temperatures
            else None,
            "gpu_temperature_max_c": max(epoch_temperatures)
            if epoch_temperatures
            else None,
            "gpu_sm_clock_min_mhz": min(epoch_sm_clocks)
            if epoch_sm_clocks
            else None,
            "gpu_sm_clock_max_mhz": max(epoch_sm_clocks)
            if epoch_sm_clocks
            else None,
            "gpu_thermal_slowdown_observed": epoch_thermal_slowdown,
            "gpu_power_brake_slowdown_observed": epoch_power_brake_slowdown,
            "improved": improved,
        }
        history.append(epoch_metrics)
        _write_history(output_dir / "metrics_history.csv", history)

        if improved:
            best_payload = build_checkpoint_payload(
                model,
                optimizer,
                epoch,
                run_context,
                best_validation_metrics,
                {"train": train_metrics, "validation": validation_metrics},
                scheduler=scheduler,
                checkpoint_kind="best",
            )
            torch.save(best_payload, output_dir / "best_model.pth")

        final_train_metrics = train_metrics
        final_validation_metrics = validation_metrics
        print(
            f"Epoch {epoch}: train RMSE={train_metrics['rmse']:.6f}, "
            f"validation RMSE={validation_metrics['rmse']:.6f}, "
            f"validation MAE={validation_metrics['mae']:.6f}, improved={improved}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {epoch} epochs based on validation RMSE",
                flush=True,
            )
            break

    if best_validation_metrics is None or final_validation_metrics is None:
        raise RuntimeError("Training finished without validation metrics")

    completed_epoch = history[-1]["epoch"]
    final_payload = build_checkpoint_payload(
        model,
        optimizer,
        completed_epoch,
        run_context,
        best_validation_metrics,
        {"train": final_train_metrics, "validation": final_validation_metrics},
        scheduler=scheduler,
        checkpoint_kind="final",
    )
    torch.save(final_payload, output_dir / "final_model.pth")

    duration_seconds = time.perf_counter() - run_started_perf
    peak_allocated = max(row["gpu_peak_allocated_bytes"] for row in history)
    peak_reserved = max(row["gpu_peak_reserved_bytes"] for row in history)
    non_finite_detected = any(row["non_finite_detected"] for row in history)
    successful_telemetry = [
        sample for sample in gpu_telemetry_samples if sample.get("query_success")
    ]
    temperatures = [sample["temperature_c"] for sample in successful_telemetry]
    sm_clocks = [sample["sm_clock_mhz"] for sample in successful_telemetry]
    thermal_slowdown_observed = any(
        sample["software_thermal_slowdown"] == "Active"
        or sample["hardware_thermal_slowdown"] == "Active"
        for sample in successful_telemetry
    )
    power_brake_slowdown_observed = any(
        sample["hardware_power_brake_slowdown"] == "Active"
        for sample in successful_telemetry
    )
    total_train_samples = sum(row["train_sample_count"] for row in history)
    total_train_duration = sum(row["train_duration_seconds"] for row in history)
    total_processed_samples = sum(
        row["train_sample_count"] + row["validation_sample_count"]
        for row in history
    )
    total_phase_duration = sum(
        row["train_duration_seconds"] + row["validation_duration_seconds"]
        for row in history
    )
    steady_state_rows = history[1:] if len(history) > 1 else history
    stability_summary = {
        "oom_occurred": False,
        "non_finite_detected": non_finite_detected,
        "gpu_temperature_min_c": min(temperatures) if temperatures else None,
        "gpu_temperature_max_c": max(temperatures) if temperatures else None,
        "gpu_sm_clock_min_mhz": min(sm_clocks) if sm_clocks else None,
        "gpu_sm_clock_max_mhz": max(sm_clocks) if sm_clocks else None,
        "gpu_thermal_slowdown_observed": thermal_slowdown_observed,
        "gpu_power_brake_slowdown_observed": power_brake_slowdown_observed,
        "obvious_clock_throttling_detected": (
            thermal_slowdown_observed or power_brake_slowdown_observed
        ),
        "train_samples_per_second_overall": (
            total_train_samples / total_train_duration
        ),
        "train_samples_per_second_steady_state": sample_weighted_average(
            [row["train_samples_per_second"] for row in steady_state_rows],
            [row["train_sample_count"] for row in steady_state_rows],
        ),
        "train_and_validation_samples_per_second_overall": (
            total_processed_samples / total_phase_duration
        ),
        "gpu_telemetry_samples_attempted": len(gpu_telemetry_samples),
        "gpu_telemetry_samples_successful": len(successful_telemetry),
    }
    final_metrics = {
        "split_version": run_context["split_version"],
        "fold": args.fold,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epoch,
        "best_validation_metrics": best_validation_metrics,
        "final_train_metrics": final_train_metrics,
        "final_validation_metrics": final_validation_metrics,
        "epoch_metrics": history,
        "stability_summary": stability_summary,
        "duration_seconds": duration_seconds,
        "PILOT_RUN": args.pilot_run,
        "NOT_FOR_RESEARCH_METRICS": args.pilot_run,
    }
    _write_json(output_dir / "final_metrics.json", final_metrics)

    run_metadata = {
        "training_version": run_context["training_version"],
        "split_version": run_context["split_version"],
        "fold": args.fold,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else 0
        ),
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "stability_summary": stability_summary,
        "gpu_telemetry_samples": gpu_telemetry_samples,
        "started_at_utc": run_started_utc.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "manifest_sha256": run_context["manifest_sha256"],
        "dataset_fingerprint": run_context["dataset_fingerprint"],
        "random_seed_settings": seed_settings,
        "pretrained": pretrained,
        "output_files": [
            "best_model.pth",
            "final_model.pth",
            "training_config_snapshot.json",
            "metrics_history.csv",
            "final_metrics.json",
            "run_metadata.json",
        ],
        "PILOT_RUN": args.pilot_run,
        "NOT_FOR_RESEARCH_METRICS": args.pilot_run,
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)
    print(f"Training complete in {duration_seconds:.1f}s; output={output_dir}", flush=True)
    return final_metrics


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
