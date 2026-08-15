"""Train ordered CQR ResNet50+irradiance on clean development roles only.

This module defines the frozen Paper1 CQR-v1 training protocol.  Importing it
does not read manifests, checkpoints, images, or create output directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from experiments import train_paper1_resnet50_with_i_v1 as point_train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"
STAGE = "cqr_resnet50_with_i_v1"
TRAIN_ROLE = "TRAIN"
VALIDATION_ROLE = "MODEL_VALIDATION"
EXPECTED_N = {TRAIN_ROLE: 25830, VALIDATION_ROLE: 3692}
SOURCE_POINT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / PROTOCOL
    / "resnet50_with_i_v1"
    / "seed_42"
    / "best_model.pth"
)
SOURCE_POINT_STATS = SOURCE_POINT_CHECKPOINT.parent / "train_irradiance_stats.json"
SOURCE_POINT_CHECKPOINT_SHA256 = (
    "97f3ec016cf99f83a78e28e2b4aca24787203f105243447d908da739c295de23"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE / "seed_42"

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
NUM_WORKERS = 0
QUANTILE_LEVELS = (0.05, 0.50, 0.95)
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
EXPECTED_IRRADIANCE_MEAN = 0.35047130969460816
EXPECTED_IRRADIANCE_STD_DDOF0 = 0.208878529631844
IRRADIANCE_STATS_ABS_TOL = 1e-12
SHARED_CHECKPOINT_PREFIXES = ("backbone.", "i_branch.", "regressor.0.")
POINT_OUTPUT_KEYS = ("regressor.3.weight", "regressor.3.bias")
REQUIRED_CHECKPOINT_FIELDS = (
    "model_state_dict",
    "epoch",
    "validation_rmse",
    "config",
)
REQUIRED_POINT_CONFIG_FIELDS = (
    "protocol",
    "architecture",
    "dropout",
    "initialization",
    "pretrained_source",
    "pretrained_weight_enum",
    "image_preprocessing",
    "irradiance_normalization",
    "legacy_checkpoint_loaded",
)
REQUIRED_IRRADIANCE_STATS_FIELDS = (
    "N",
    "mean",
    "std_ddof0",
    "min",
    "max",
    "normalization",
    "source_role",
)


class Paper1CQRResNet50WithI(nn.Module):
    """Ordered CQR model with the frozen clean point-model shared geometry."""

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        # Deliberately never request torchvision weights.  All shared weights
        # come from the SHA-verified clean point checkpoint.
        backbone = resnet50(weights=None)
        image_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.i_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )
        # Preserve regressor.0 so the shared point-checkpoint key is exact.
        self.regressor = nn.Sequential(
            nn.Linear(image_features + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.center = nn.Linear(128, 1)
        self.lower_distance = nn.Linear(128, 1)
        self.upper_distance = nn.Linear(128, 1)
        self.reset_distance_heads()

    def reset_distance_heads(self) -> None:
        nn.init.zeros_(self.lower_distance.weight)
        nn.init.zeros_(self.lower_distance.bias)
        nn.init.zeros_(self.upper_distance.weight)
        nn.init.zeros_(self.upper_distance.bias)

    def forward_features(
        self, image: torch.Tensor, irradiance: torch.Tensor
    ) -> torch.Tensor:
        image_feature = self.backbone(image)
        irradiance_feature = self.i_branch(irradiance.unsqueeze(1))
        return self.regressor(torch.cat([image_feature, irradiance_feature], dim=1))

    def forward(self, image: torch.Tensor, irradiance: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(image, irradiance)
        z50 = self.center(features)
        d_low = F.softplus(self.lower_distance(features))
        d_high = F.softplus(self.upper_distance(features))
        q05 = torch.sigmoid(z50 - d_low)
        q50 = torch.sigmoid(z50)
        q95 = torch.sigmoid(z50 + d_high)
        return torch.cat((q05, q50, q95), dim=1)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_point_preprocessing() -> dict[str, str]:
    train_transform, validation_transform = point_train.build_transforms()
    return {
        "train": repr(train_transform),
        "validation": repr(validation_transform),
    }


def validate_point_checkpoint_schema(checkpoint: Mapping[str, Any]) -> None:
    missing_top = set(REQUIRED_CHECKPOINT_FIELDS) - set(checkpoint)
    if missing_top:
        raise ValueError(f"Point checkpoint missing top-level fields: {sorted(missing_top)}")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError("Point checkpoint config must be a mapping")
    missing_config = set(REQUIRED_POINT_CONFIG_FIELDS) - set(config)
    if missing_config:
        raise ValueError(f"Point checkpoint config missing fields: {sorted(missing_config)}")
    if config["protocol"] != PROTOCOL:
        raise ValueError("Point checkpoint protocol mismatch")
    if config["architecture"] != "ResNet50+irradiance":
        raise ValueError("Point checkpoint architecture mismatch")
    if not math.isclose(float(config["dropout"]), DROPOUT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Point checkpoint dropout mismatch")
    if config["initialization"] != "ImageNet pretrained":
        raise ValueError("Point checkpoint initialization mismatch")
    if config["pretrained_source"] != "torchvision":
        raise ValueError("Point checkpoint pretrained source mismatch")
    if config["pretrained_weight_enum"] != "ResNet50_Weights.IMAGENET1K_V2":
        raise ValueError("Point checkpoint pretrained weight enum mismatch")
    if config["image_preprocessing"] != expected_point_preprocessing():
        raise ValueError("Point checkpoint image preprocessing mismatch")
    if config["irradiance_normalization"] != "TRAIN-only z-score":
        raise ValueError("Point checkpoint irradiance normalization mismatch")
    if config["legacy_checkpoint_loaded"] is not False:
        raise ValueError("Legacy point checkpoint is forbidden")
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("model_state_dict must be a mapping")
    if int(checkpoint["epoch"]) < 1:
        raise ValueError("Point checkpoint epoch must be positive")
    if not math.isfinite(float(checkpoint["validation_rmse"])):
        raise ValueError("Point checkpoint validation_rmse must be finite")


def load_verified_point_checkpoint(
    path: Path = SOURCE_POINT_CHECKPOINT,
    expected_sha256: str = SOURCE_POINT_CHECKPOINT_SHA256,
) -> Mapping[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed.lower() != expected_sha256.lower():
        raise ValueError(
            f"Point checkpoint SHA256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Point checkpoint must contain a mapping")
    validate_point_checkpoint_schema(checkpoint)
    return checkpoint


def _strict_load_prefixed_module(
    module: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    prefix: str,
) -> None:
    selected = {
        key[len(prefix) :]: value
        for key, value in source_state.items()
        if key.startswith(prefix)
    }
    expected = module.state_dict()
    missing = set(expected) - set(selected)
    unexpected = set(selected) - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"Shared {prefix} key mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    shape_mismatches = {
        key: (tuple(selected[key].shape), tuple(expected[key].shape))
        for key in expected
        if tuple(selected[key].shape) != tuple(expected[key].shape)
    }
    if shape_mismatches:
        raise ValueError(f"Shared {prefix} shape mismatch: {shape_mismatches}")
    module.load_state_dict(selected, strict=True)


def initialize_from_point_checkpoint(
    model: Paper1CQRResNet50WithI,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    validate_point_checkpoint_schema(checkpoint)
    source_state = checkpoint["model_state_dict"]
    _strict_load_prefixed_module(model.backbone, source_state, "backbone.")
    _strict_load_prefixed_module(model.i_branch, source_state, "i_branch.")
    _strict_load_prefixed_module(model.regressor[0], source_state, "regressor.0.")

    present_point_keys = {key for key in source_state if key.startswith("regressor.3.")}
    if present_point_keys != set(POINT_OUTPUT_KEYS):
        raise ValueError(
            "Point output key mismatch; "
            f"expected={list(POINT_OUTPUT_KEYS)}, observed={sorted(present_point_keys)}"
        )
    expected_shapes = {
        "regressor.3.weight": tuple(model.center.weight.shape),
        "regressor.3.bias": tuple(model.center.bias.shape),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(source_state[key].shape) != expected_shape:
            raise ValueError(
                f"Point output shape mismatch for {key}: "
                f"{tuple(source_state[key].shape)} != {expected_shape}"
            )
    with torch.no_grad():
        model.center.weight.copy_(source_state["regressor.3.weight"])
        model.center.bias.copy_(source_state["regressor.3.bias"])
    model.reset_distance_heads()
    return {
        "shared_prefixes": list(SHARED_CHECKPOINT_PREFIXES),
        "center_source_keys": list(POINT_OUTPUT_KEYS),
        "distance_heads_zero_initialized": True,
    }


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error)


def quantile_loss_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if predictions.ndim != 2 or predictions.shape[1] != len(QUANTILE_LEVELS):
        raise ValueError("CQR predictions must have shape [N, 3]")
    if targets.ndim != 1 or len(targets) != len(predictions):
        raise ValueError("CQR targets must have shape [N]")
    losses = {
        "pinball_q05": pinball_loss(predictions[:, 0], targets, 0.05).mean(),
        "pinball_q50": pinball_loss(predictions[:, 1], targets, 0.50).mean(),
        "pinball_q95": pinball_loss(predictions[:, 2], targets, 0.95).mean(),
    }
    mean_pinball = torch.stack(tuple(losses.values())).mean()
    return mean_pinball, losses


def quantile_crossing_count(predictions: torch.Tensor) -> int:
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("CQR predictions must have shape [N, 3]")
    crossing = (predictions[:, 0] > predictions[:, 1]) | (
        predictions[:, 1] > predictions[:, 2]
    )
    return int(crossing.sum().item())


def validate_quantile_outputs(predictions: torch.Tensor) -> None:
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("CQR predictions must have shape [N, 3]")
    if not torch.isfinite(predictions).all():
        raise ValueError("CQR predictions contain NaN or infinity")
    if torch.any(predictions < 0.0) or torch.any(predictions > 1.0):
        raise ValueError("CQR predictions fall outside [0, 1]")
    if quantile_crossing_count(predictions) != 0:
        raise ValueError("Ordered CQR produced a quantile crossing")


def validate_training_roles(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    point_train.validate_role_isolation(train, validation)
    if set(train["role"]) != {TRAIN_ROLE}:
        raise PermissionError("Only TRAIN may update model parameters")
    if set(validation["role"]) != {VALIDATION_ROLE}:
        raise PermissionError("Only MODEL_VALIDATION may select checkpoints")


def load_and_validate_frozen_irradiance_stats(
    path: Path,
    train: pd.DataFrame,
) -> dict[str, float | int | str]:
    if set(train["role"]) != {TRAIN_ROLE}:
        raise PermissionError("Irradiance statistics may use TRAIN only")
    with Path(path).open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    missing = set(REQUIRED_IRRADIANCE_STATS_FIELDS) - set(stats)
    if missing:
        raise ValueError(f"Frozen irradiance stats missing fields: {sorted(missing)}")
    if stats["source_role"] != TRAIN_ROLE or stats["normalization"] != "z_score":
        raise ValueError("Frozen irradiance normalization schema mismatch")
    if int(stats["N"]) != EXPECTED_N[TRAIN_ROLE] or len(train) != EXPECTED_N[TRAIN_ROLE]:
        raise ValueError("Frozen irradiance TRAIN N mismatch")
    if not math.isclose(
        float(stats["mean"]),
        EXPECTED_IRRADIANCE_MEAN,
        rel_tol=0.0,
        abs_tol=IRRADIANCE_STATS_ABS_TOL,
    ):
        raise ValueError("Frozen irradiance mean mismatch")
    if not math.isclose(
        float(stats["std_ddof0"]),
        EXPECTED_IRRADIANCE_STD_DDOF0,
        rel_tol=0.0,
        abs_tol=IRRADIANCE_STATS_ABS_TOL,
    ):
        raise ValueError("Frozen irradiance population std mismatch")
    computed = point_train.compute_train_irradiance_stats(train)
    for key in ("N", "mean", "std_ddof0"):
        if not math.isclose(
            float(stats[key]),
            float(computed[key]),
            rel_tol=0.0,
            abs_tol=IRRADIANCE_STATS_ABS_TOL,
        ):
            raise ValueError(f"Frozen irradiance stats disagree with TRAIN for {key}")
    return stats


def build_optimizer_and_scheduler(
    model: nn.Module,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )
    return optimizer, scheduler


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    validation_metrics: Mapping[str, float | int],
) -> None:
    scheduler.step(float(validation_metrics["mean_pinball"]))


def validation_improved(
    validation_metrics: Mapping[str, float | int], best_mean_pinball: float
) -> bool:
    return float(validation_metrics["mean_pinball"]) < best_mean_pinball


def build_checkpoint_payload(
    model: nn.Module,
    epoch: int,
    validation_metrics: Mapping[str, float | int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "validation_mean_pinball": float(validation_metrics["mean_pinball"]),
        "validation_pinball_q05": float(validation_metrics["pinball_q05"]),
        "validation_pinball_q50": float(validation_metrics["pinball_q50"]),
        "validation_pinball_q95": float(validation_metrics["pinball_q95"]),
        "config": dict(config),
    }


def _autocast_context(device: torch.device):
    return torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    count = 0
    sums = {"mean_pinball": 0.0, "pinball_q05": 0.0, "pinball_q50": 0.0, "pinball_q95": 0.0}
    truths: list[np.ndarray] = []
    q50_predictions: list[np.ndarray] = []
    crossing_count = 0
    with torch.inference_mode():
        for images, irradiance, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            irradiance = irradiance.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast_context(device):
                predictions = model(images, irradiance)
                mean_loss, components = quantile_loss_components(predictions, targets)
            validate_quantile_outputs(predictions.float())
            batch_n = len(targets)
            sums["mean_pinball"] += float(mean_loss.float().item()) * batch_n
            for name, value in components.items():
                sums[name] += float(value.float().item()) * batch_n
            count += batch_n
            crossing_count += quantile_crossing_count(predictions.float())
            truths.append(targets.float().cpu().numpy())
            q50_predictions.append(predictions[:, 1].float().cpu().numpy())
    if count == 0:
        raise ValueError("MODEL_VALIDATION loader is empty")
    y_true = np.concatenate(truths)
    y_q50 = np.concatenate(q50_predictions)
    metrics: dict[str, float | int] = {name: value / count for name, value in sums.items()}
    metrics.update(
        {
            "N": count,
            "q50_MAE": float(mean_absolute_error(y_true, y_q50)),
            "q50_RMSE": float(mean_squared_error(y_true, y_q50) ** 0.5),
            "q50_R2": float(r2_score(y_true, y_q50)),
            "quantile_crossing_count": crossing_count,
        }
    )
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("MODEL_VALIDATION metrics contain NaN or infinity")
    if crossing_count != 0:
        raise ValueError("MODEL_VALIDATION quantile crossing detected")
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, float | int]:
    model.train()
    count = 0
    sums = {"mean_pinball": 0.0, "pinball_q05": 0.0, "pinball_q50": 0.0, "pinball_q95": 0.0}
    crossing_count = 0
    for images, irradiance, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        irradiance = irradiance.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            predictions = model(images, irradiance)
            mean_loss, components = quantile_loss_components(predictions, targets)
        if not torch.isfinite(mean_loss):
            raise ValueError("TRAIN loss is NaN or infinity")
        validate_quantile_outputs(predictions.detach().float())
        scaler.scale(mean_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_n = len(targets)
        sums["mean_pinball"] += float(mean_loss.detach().float().item()) * batch_n
        for name, value in components.items():
            sums[name] += float(value.detach().float().item()) * batch_n
        count += batch_n
        crossing_count += quantile_crossing_count(predictions.detach().float())
    if count == 0:
        raise ValueError("TRAIN loader is empty")
    metrics: dict[str, float | int] = {name: value / count for name, value in sums.items()}
    metrics.update({"N": count, "quantile_crossing_count": crossing_count})
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("TRAIN metrics contain NaN or infinity")
    return metrics


def build_config() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "model": STAGE,
        "seed": SEED,
        "architecture": "ResNet50+irradiance ordered CQR",
        "source_point_checkpoint": str(SOURCE_POINT_CHECKPOINT.relative_to(PROJECT_ROOT)).replace(os.sep, "/"),
        "source_point_checkpoint_sha256": SOURCE_POINT_CHECKPOINT_SHA256,
        "shared_initialization": "backbone+i_branch+fusion from clean point checkpoint",
        "shared_checkpoint_prefixes": list(SHARED_CHECKPOINT_PREFIXES),
        "center_initialization": "old point scalar output layer",
        "distance_initialization": "zero weight and zero bias; softplus(0)=ln2",
        "ordered_quantile_parameterization": True,
        "quantile_levels": list(QUANTILE_LEVELS),
        "loss": "equal-weight mean pinball",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_mode": "min",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_metric": "MODEL_VALIDATION mean pinball",
        "checkpoint_selection_metric": "MODEL_VALIDATION mean pinball",
        "training_roles": [TRAIN_ROLE, VALIDATION_ROLE],
        "train_N": EXPECTED_N[TRAIN_ROLE],
        "model_validation_N": EXPECTED_N[VALIDATION_ROLE],
        "amp": "torch.amp.autocast(cuda)+torch.amp.GradScaler(cuda)",
        "imagenet_weights_at_construction": None,
        "imagenet_download_performed": False,
        "irradiance_normalization": "frozen clean TRAIN-only z-score; population std ddof=0",
        "image_preprocessing": expected_point_preprocessing(),
        "warmup": False,
        "gradient_clipping": False,
        "ema": False,
        "layerwise_learning_rates": False,
        "discriminative_learning_rates": False,
        "resume_training": False,
    }


def build_provenance(duration: float, best_epoch: int) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "model": STAGE,
        "source_point_checkpoint": str(SOURCE_POINT_CHECKPOINT.relative_to(PROJECT_ROOT)).replace(os.sep, "/"),
        "source_point_checkpoint_sha256": SOURCE_POINT_CHECKPOINT_SHA256,
        "shared_initialization": "backbone+i_branch+fusion from clean point checkpoint",
        "center_initialization": "old point scalar output layer",
        "distance_initialization": "zero weight and zero bias; softplus(0)=ln2",
        "ordered_quantile_parameterization": True,
        "quantile_levels": list(QUANTILE_LEVELS),
        "loss": "equal-weight mean pinball",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "checkpoint_selection_metric": "MODEL_VALIDATION mean pinball",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_metric": "MODEL_VALIDATION mean pinball",
        "imagenet_download_performed": False,
        "training_roles": [TRAIN_ROLE, VALIDATION_ROLE],
        "cp_calibration_accessed": False,
        "decision_development_accessed": False,
        "random_test_accessed": False,
        "sealed_final_dates_accessed": False,
        "cqr_conformal_calibration_performed": False,
        "risk_evaluation_performed": False,
        "cleaning_decision_performed": False,
        "economic_decision_performed": False,
        "training_performed": True,
        "training_duration_seconds": duration,
        "best_epoch": best_epoch,
    }


def _save_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run the formal CQR training protocol when explicitly invoked."""
    point_train.ensure_output_available(output_dir)
    point_train.set_seed(SEED)
    train_manifest = point_train.load_role_manifest(point_train.TRAIN_MANIFEST, TRAIN_ROLE)
    validation_manifest = point_train.load_role_manifest(
        point_train.VALIDATION_MANIFEST, VALIDATION_ROLE
    )
    validate_training_roles(train_manifest, validation_manifest)
    train = point_train.attach_development_values(train_manifest)
    validation = point_train.attach_development_values(validation_manifest)
    stats = load_and_validate_frozen_irradiance_stats(SOURCE_POINT_STATS, train)
    train = point_train.normalize_irradiance(train, stats)
    validation = point_train.normalize_irradiance(validation, stats)
    train_transform, validation_transform = point_train.build_transforms()
    if not point_train.validation_transform_is_deterministic(validation_transform):
        raise ValueError("MODEL_VALIDATION preprocessing is not deterministic")

    checkpoint = load_verified_point_checkpoint()
    model = Paper1CQRResNet50WithI(dropout=DROPOUT)
    initialization_audit = initialize_from_point_checkpoint(model, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this formal training run")
    model = model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        point_train.ManifestDataset(train, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        point_train.ManifestDataset(validation, validation_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config()
    _save_json_exclusive(output_dir / "config.json", config)
    _save_json_exclusive(output_dir / "train_irradiance_stats.json", stats)
    _save_json_exclusive(output_dir / "initialization_audit.json", initialization_audit)

    started = time.perf_counter()
    best_mean_pinball = float("inf")
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    best_path = output_dir / "best_model.pth"
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device)
        validation_metrics = evaluate(model, validation_loader, device)
        step_scheduler(scheduler, validation_metrics)
        improved = validation_improved(validation_metrics, best_mean_pinball)
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selected": improved,
            }
        )
        print(
            f"epoch={epoch:02d} train_pinball={train_metrics['mean_pinball']:.8f} "
            f"val_pinball={validation_metrics['mean_pinball']:.8f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if improved:
            best_mean_pinball = float(validation_metrics["mean_pinball"])
            best_epoch = epoch
            no_improve = 0
            torch.save(
                build_checkpoint_payload(model, epoch, validation_metrics, config), best_path
            )
        else:
            no_improve += 1
        if no_improve >= EARLY_STOPPING_PATIENCE:
            break

    if best_epoch < 1 or not best_path.is_file():
        raise ValueError("No valid best CQR checkpoint was generated")
    duration = time.perf_counter() - started
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    required_best = {
        "model_state_dict",
        "epoch",
        "validation_mean_pinball",
        "validation_pinball_q05",
        "validation_pinball_q50",
        "validation_pinball_q95",
        "config",
    }
    if not required_best.issubset(best_checkpoint):
        raise ValueError("Generated best CQR checkpoint schema verification failed")
    if int(best_checkpoint["epoch"]) != best_epoch:
        raise ValueError("Generated best CQR checkpoint epoch verification failed")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    final_validation = evaluate(model, validation_loader, device)
    _save_json_exclusive(
        output_dir / "model_validation_metrics.json",
        {
            "selection_metric": "MODEL_VALIDATION mean pinball",
            "best_epoch": best_epoch,
            "best_validation_mean_pinball": best_mean_pinball,
            "reloaded_best_checkpoint_metrics": final_validation,
        },
    )
    provenance = build_provenance(duration, best_epoch)
    _save_json_exclusive(output_dir / "provenance.json", provenance)
    return {
        "best_epoch": best_epoch,
        "best_validation_mean_pinball": best_mean_pinball,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
