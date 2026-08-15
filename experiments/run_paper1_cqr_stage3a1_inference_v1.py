"""Paper1 CQR Stage 3A1: deterministic quantile inference only.

The formal entry point is restricted to CP_CALIBRATION and
DECISION_DEVELOPMENT.  It reads locator metadata and irradiance only; it does
not parse, load, derive, or save truth.  It performs one deterministic CQR
forward pass per sample and performs no conformal calibration or evaluation.
Importing this module does not read manifests, checkpoints, images, or create
output directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from experiments import train_paper1_cqr_resnet50_with_i_v1 as cqr_train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"
STAGE = "cqr_stage3a1_inference_v1"
CQR_STAGE = "cqr_resnet50_with_i_v1"
ARCHITECTURE = "ResNet50+irradiance ordered CQR"
SEED = 42
DROPOUT = 0.3
QUANTILE_LEVELS = (0.05, 0.50, 0.95)
BATCH_SIZE = 32
NUM_WORKERS = 0

CP_CALIBRATION_ROLE = "CP_CALIBRATION"
DECISION_DEVELOPMENT_ROLE = "DECISION_DEVELOPMENT"
ROLES = (CP_CALIBRATION_ROLE, DECISION_DEVELOPMENT_ROLE)
EXPECTED_N = {
    CP_CALIBRATION_ROLE: 2951,
    DECISION_DEVELOPMENT_ROLE: 1844,
}
SEALED_FINAL_DATES = frozenset({"2017-06-15", "2017-06-24", "2017-06-30"})

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / PROTOCOL
CP_CALIBRATION_MANIFEST = SPLIT_ROOT / "cp_calibration.csv"
DECISION_DEVELOPMENT_MANIFEST = SPLIT_ROOT / "decision_development.csv"
AUTHORIZED_MANIFESTS = {
    CP_CALIBRATION_ROLE: CP_CALIBRATION_MANIFEST,
    DECISION_DEVELOPMENT_ROLE: DECISION_DEVELOPMENT_MANIFEST,
}

SOURCE_CQR_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / PROTOCOL / CQR_STAGE / "seed_42" / "best_model.pth"
)
SOURCE_CQR_CHECKPOINT_SHA256 = (
    "fd5deea62c867fcffe3791f768752da9dc3a39a1c146244b1e225d6b40b0da80"
)
SOURCE_CQR_BEST_EPOCH = 15
SOURCE_CQR_VALIDATION_MEAN_PINBALL = 0.01066008722409606

TRAIN_IRRADIANCE_STATS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / PROTOCOL
    / "resnet50_with_i_v1"
    / "seed_42"
    / "train_irradiance_stats.json"
)
EXPECTED_TRAIN_N = 25830
EXPECTED_IRRADIANCE_MEAN = 0.35047130969460816
EXPECTED_IRRADIANCE_STD_DDOF0 = 0.208878529631844
FLOAT_GUARD_ABS_TOL = 1e-15
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE
METHOD = "cqr_deterministic"
MANIFEST_COLUMNS = ("sample_id", "date", "timestamp", "image_path", "role")
RECORD_COLUMNS = MANIFEST_COLUMNS + ("irradiance", "irradiance_normalized")
PREDICTION_COLUMNS = MANIFEST_COLUMNS + ("irradiance", "q05", "q50", "q95")
FORBIDDEN_PREDICTION_COLUMNS = frozenset(
    {
        "true_L",
        "label",
        "target",
        "absolute_error",
        "abs_error",
        "abs_error_point",
        "abs_error_mc_mean",
        "residual",
    }
)
REQUIRED_CHECKPOINT_FIELDS = frozenset(
    {
        "model_state_dict",
        "epoch",
        "validation_mean_pinball",
        "validation_pinball_q05",
        "validation_pinball_q50",
        "validation_pinball_q95",
        "config",
    }
)
REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "protocol",
        "stage",
        "model",
        "seed",
        "architecture",
        "dropout",
        "quantile_levels",
        "ordered_quantile_parameterization",
        "image_preprocessing",
        "irradiance_normalization",
        "imagenet_weights_at_construction",
        "imagenet_download_performed",
    }
)
REQUIRED_IRRADIANCE_STATS_FIELDS = frozenset(
    {"N", "mean", "std_ddof0", "min", "max", "normalization", "source_role"}
)
VALIDATION_METRIC_FIELDS = (
    "validation_mean_pinball",
    "validation_pinball_q05",
    "validation_pinball_q50",
    "validation_pinball_q95",
)
IRRADIANCE_PATTERN = re.compile(r"_I_([0-9eE+.-]+)$")


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_protocol(protocol: str) -> None:
    if protocol != PROTOCOL:
        raise PermissionError(f"Unauthorized protocol: {protocol!r}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_path(path: Path) -> Path:
    candidate = _resolved(path)
    if candidate != _resolved(SOURCE_CQR_CHECKPOINT):
        raise PermissionError(f"Unauthorized CQR checkpoint: {candidate}")
    return candidate


def validate_checkpoint_schema(checkpoint: Mapping[str, Any]) -> None:
    missing = REQUIRED_CHECKPOINT_FIELDS - set(checkpoint)
    if missing:
        raise ValueError(f"CQR checkpoint missing fields: {sorted(missing)}")
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("CQR model_state_dict must be a mapping")

    epoch = int(checkpoint["epoch"])
    if epoch != SOURCE_CQR_BEST_EPOCH:
        raise ValueError(
            f"CQR best epoch mismatch: expected {SOURCE_CQR_BEST_EPOCH}, got {epoch}"
        )
    for field in VALIDATION_METRIC_FIELDS:
        if not math.isfinite(float(checkpoint[field])):
            raise ValueError(f"CQR checkpoint {field} must be finite")
    if not math.isclose(
        float(checkpoint["validation_mean_pinball"]),
        SOURCE_CQR_VALIDATION_MEAN_PINBALL,
        rel_tol=0.0,
        abs_tol=FLOAT_GUARD_ABS_TOL,
    ):
        raise ValueError("CQR validation_mean_pinball mismatch")

    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError("CQR checkpoint config must be a mapping")
    missing_config = REQUIRED_CONFIG_FIELDS - set(config)
    if missing_config:
        raise ValueError(f"CQR checkpoint config missing fields: {sorted(missing_config)}")
    expected_exact = {
        "protocol": PROTOCOL,
        "stage": CQR_STAGE,
        "model": CQR_STAGE,
        "seed": SEED,
        "architecture": ARCHITECTURE,
        "ordered_quantile_parameterization": True,
        "irradiance_normalization": (
            "frozen clean TRAIN-only z-score; population std ddof=0"
        ),
        "imagenet_weights_at_construction": None,
        "imagenet_download_performed": False,
    }
    mismatches = {
        key: (config[key], expected)
        for key, expected in expected_exact.items()
        if config[key] != expected
    }
    if mismatches:
        raise ValueError(f"CQR checkpoint config mismatch: {mismatches}")
    if not math.isclose(
        float(config["dropout"]),
        DROPOUT,
        rel_tol=0.0,
        abs_tol=FLOAT_GUARD_ABS_TOL,
    ):
        raise ValueError("CQR checkpoint dropout mismatch")
    try:
        observed_levels = tuple(float(value) for value in config["quantile_levels"])
    except (TypeError, ValueError) as error:
        raise ValueError("CQR checkpoint quantile_levels must be numeric") from error
    if observed_levels != QUANTILE_LEVELS:
        raise ValueError("CQR checkpoint quantile_levels mismatch")
    if config["image_preprocessing"] != cqr_train.expected_point_preprocessing():
        raise ValueError("CQR checkpoint image preprocessing mismatch")


def load_verified_cqr_checkpoint(
    path: Path = SOURCE_CQR_CHECKPOINT,
    expected_sha256: str = SOURCE_CQR_CHECKPOINT_SHA256,
) -> Mapping[str, Any]:
    checkpoint_path = validate_checkpoint_path(path)
    observed_sha256 = sha256_file(checkpoint_path)
    if observed_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "CQR checkpoint SHA256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("CQR checkpoint payload must be a mapping")
    validate_checkpoint_schema(checkpoint)
    return checkpoint


def validate_manifest_authorization(path: Path, expected_role: str) -> Path:
    if expected_role not in AUTHORIZED_MANIFESTS:
        raise PermissionError(f"Forbidden Stage 3A1 role: {expected_role}")
    candidate = _resolved(path)
    lowered = str(candidate).lower()
    if "random_test" in lowered:
        raise PermissionError("RANDOM_TEST manifest is forbidden")
    authorized = _resolved(AUTHORIZED_MANIFESTS[expected_role])
    if candidate != authorized:
        raise PermissionError(
            f"Manifest is not authorized for {expected_role}: {candidate}"
        )
    return candidate


def _validate_locator_metadata(
    frame: pd.DataFrame,
    expected_role: str,
    *,
    enforce_expected_n: bool,
) -> pd.DataFrame:
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{expected_role} manifest is empty")
    if expected_role not in EXPECTED_N:
        raise PermissionError(f"Forbidden Stage 3A1 role: {expected_role}")
    if set(frame["role"].astype(str)) != {expected_role}:
        raise PermissionError(f"Role guard failed for {expected_role}")
    if enforce_expected_n and len(frame) != EXPECTED_N[expected_role]:
        raise ValueError(
            f"{expected_role} N guard failed: expected {EXPECTED_N[expected_role]}, "
            f"got {len(frame)}"
        )

    normalized_dates = pd.to_datetime(frame["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    sealed = set(normalized_dates) & SEALED_FINAL_DATES
    if sealed:
        raise PermissionError(f"Sealed final date rejected: {sorted(sealed)}")
    locators = frame["image_path"].astype(str)
    if locators.str.lower().str.contains("random_test", regex=False).any():
        raise PermissionError("RANDOM_TEST locator rejected")
    for sealed_date in SEALED_FINAL_DATES:
        if locators.str.contains(sealed_date, regex=False).any():
            raise PermissionError(f"Sealed final date locator rejected: {sealed_date}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-null and unique")
    if frame["image_path"].isna().any() or frame["image_path"].duplicated().any():
        raise ValueError("image_path must be non-null and unique")
    return frame.loc[:, MANIFEST_COLUMNS].copy()


def validate_manifest_frame(
    frame: pd.DataFrame,
    expected_role: str,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    return _validate_locator_metadata(
        frame, expected_role, enforce_expected_n=enforce_expected_n
    )


def load_role_manifest(path: Path, expected_role: str) -> pd.DataFrame:
    authorized = validate_manifest_authorization(path, expected_role)
    # Reading only locator metadata is the Stage 3A1 truth-isolation boundary.
    frame = pd.read_csv(authorized, usecols=list(MANIFEST_COLUMNS))
    return validate_manifest_frame(frame, expected_role)


def validate_role_isolation(cp: pd.DataFrame, decision: pd.DataFrame) -> None:
    if set(cp["role"].astype(str)) != {CP_CALIBRATION_ROLE}:
        raise PermissionError("First frame must contain CP_CALIBRATION only")
    if set(decision["role"].astype(str)) != {DECISION_DEVELOPMENT_ROLE}:
        raise PermissionError("Second frame must contain DECISION_DEVELOPMENT only")
    if set(cp["sample_id"]) & set(decision["sample_id"]):
        raise ValueError("CP_CALIBRATION/DECISION_DEVELOPMENT sample_id overlap")
    if set(cp["image_path"]) & set(decision["image_path"]):
        raise ValueError("CP_CALIBRATION/DECISION_DEVELOPMENT image_path overlap")


def validate_irradiance_stats_path(path: Path) -> Path:
    candidate = _resolved(path)
    if candidate != _resolved(TRAIN_IRRADIANCE_STATS_PATH):
        raise PermissionError(f"Unauthorized irradiance statistics artifact: {candidate}")
    return candidate


def validate_irradiance_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_IRRADIANCE_STATS_FIELDS - set(stats)
    if missing:
        raise ValueError(f"Frozen irradiance stats missing fields: {sorted(missing)}")
    if stats["source_role"] != "TRAIN":
        raise PermissionError("Irradiance normalization must use TRAIN statistics only")
    if stats["normalization"] != "z_score":
        raise ValueError("Frozen irradiance normalization must be z_score")
    if int(stats["N"]) != EXPECTED_TRAIN_N:
        raise ValueError(f"Frozen irradiance stats N must equal {EXPECTED_TRAIN_N}")
    for field in ("mean", "std_ddof0", "min", "max"):
        if not math.isfinite(float(stats[field])):
            raise ValueError(f"Frozen irradiance stats {field} must be finite")
    if not math.isclose(
        float(stats["mean"]),
        EXPECTED_IRRADIANCE_MEAN,
        rel_tol=0.0,
        abs_tol=FLOAT_GUARD_ABS_TOL,
    ):
        raise ValueError("Frozen irradiance mean mismatch")
    if not math.isclose(
        float(stats["std_ddof0"]),
        EXPECTED_IRRADIANCE_STD_DDOF0,
        rel_tol=0.0,
        abs_tol=FLOAT_GUARD_ABS_TOL,
    ):
        raise ValueError("Frozen irradiance population std mismatch")
    if float(stats["std_ddof0"]) <= 0.0:
        raise ValueError("Frozen irradiance population std must be positive")
    return dict(stats)


def load_train_irradiance_stats(
    path: Path = TRAIN_IRRADIANCE_STATS_PATH,
) -> dict[str, Any]:
    authorized = validate_irradiance_stats_path(path)
    with authorized.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if not isinstance(stats, Mapping):
        raise ValueError("Frozen irradiance statistics must be a mapping")
    return validate_irradiance_stats(stats)


def parse_irradiance_only(image_path: str) -> float:
    match = IRRADIANCE_PATTERN.search(Path(image_path).stem)
    if match is None:
        raise ValueError(f"Cannot parse irradiance from locator: {image_path}")
    irradiance = float(match.group(1))
    if not math.isfinite(irradiance):
        raise ValueError("Parsed irradiance must be finite")
    return irradiance


def prepare_inference_records(
    frame: pd.DataFrame,
    expected_role: str,
    stats: Mapping[str, Any],
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    guarded = validate_manifest_frame(
        frame, expected_role, enforce_expected_n=enforce_expected_n
    )
    frozen = validate_irradiance_stats(stats)
    result = guarded.copy()
    result["irradiance"] = [parse_irradiance_only(value) for value in result["image_path"]]
    result["irradiance_normalized"] = (
        result["irradiance"] - float(frozen["mean"])
    ) / float(frozen["std_ddof0"])
    numeric = result.loc[:, ["irradiance", "irradiance_normalized"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Irradiance inputs must be finite")
    return result.loc[:, RECORD_COLUMNS]


def validate_inference_transform(transform: Any) -> None:
    transform_types = cqr_train.point_train.transforms
    if not isinstance(transform, transform_types.Compose):
        raise ValueError("Inference transform must be torchvision.transforms.Compose")
    items = transform.transforms
    if len(items) != 3:
        raise ValueError("Inference transform must contain exactly three operations")
    if not isinstance(items[0], transform_types.Resize):
        raise ValueError("Inference transform must start with Resize")
    resize_size = tuple(items[0].size) if isinstance(items[0].size, Sequence) else (items[0].size,)
    if resize_size != (224, 224):
        raise ValueError("Inference Resize must be exactly (224, 224)")
    if not isinstance(items[1], transform_types.ToTensor):
        raise ValueError("Inference transform must use ToTensor after Resize")
    if not isinstance(items[2], transform_types.Normalize):
        raise ValueError("Inference transform must end with Normalize")
    if tuple(float(value) for value in items[2].mean) != IMAGENET_MEAN:
        raise ValueError("Inference normalization mean mismatch")
    if tuple(float(value) for value in items[2].std) != IMAGENET_STD:
        raise ValueError("Inference normalization std mismatch")
    if not cqr_train.point_train.validation_transform_is_deterministic(transform):
        raise ValueError("Inference image transform must be deterministic")


def build_inference_transform():
    _, validation_transform = cqr_train.point_train.build_transforms()
    validate_inference_transform(validation_transform)
    return validation_transform


class Stage3A1InferenceDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: Any):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image_path = PROJECT_ROOT / Path(row["image_path"])
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        return (
            self.transform(image),
            torch.tensor(float(row["irradiance_normalized"]), dtype=torch.float32),
            index,
        )


def set_deterministic_inference_mode(model: nn.Module) -> None:
    model.eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("Deterministic inference requires every module in eval mode")
    if any(
        module.training
        for module in model.modules()
        if isinstance(module, nn.Dropout)
    ):
        raise RuntimeError("Dropout must remain disabled during Stage 3A1")


def build_inference_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> cqr_train.Paper1CQRResNet50WithI:
    validate_checkpoint_schema(checkpoint)
    model = cqr_train.Paper1CQRResNet50WithI(dropout=DROPOUT)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model = model.to(device)
    set_deterministic_inference_mode(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Stage 3A1 model parameters must not require gradients")
    return model


def _autocast_context(device: torch.device):
    return torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()


def predict_deterministic(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    set_deterministic_inference_mode(model)
    predictions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for images, irradiance, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            irradiance = irradiance.to(device, non_blocking=True)
            with _autocast_context(device):
                outputs = model(images, irradiance)
            outputs_float = outputs.float()
            cqr_train.validate_quantile_outputs(outputs_float)
            predictions.append(outputs_float.cpu().numpy().astype(np.float64))
            indices.append(np.asarray(batch_indices, dtype=np.int64))
    if not predictions:
        raise ValueError("Stage 3A1 inference loader is empty")
    values = np.concatenate(predictions, axis=0)
    observed_indices = np.concatenate(indices)
    order = np.argsort(observed_indices)
    expected_indices = np.arange(len(loader.dataset))
    if not np.array_equal(observed_indices[order], expected_indices):
        raise ValueError("Stage 3A1 prediction order/index guard failed")
    return values[order]


def _truth_like_columns(columns: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for column in columns:
        lowered = str(column).lower()
        if (
            column in FORBIDDEN_PREDICTION_COLUMNS
            or lowered.startswith("true_")
            or "target" in lowered
            or "label" in lowered
            or "error" in lowered
            or "residual" in lowered
        ):
            result.add(str(column))
    return result


def validate_quantile_array(values: np.ndarray, expected_n: int) -> np.ndarray:
    quantiles = np.asarray(values, dtype=np.float64)
    if quantiles.shape != (expected_n, 3):
        raise ValueError(
            f"CQR predictions must have shape [{expected_n}, 3], got {quantiles.shape}"
        )
    if not np.isfinite(quantiles).all():
        raise ValueError("CQR predictions contain NaN or infinity")
    if np.any(quantiles < 0.0) or np.any(quantiles > 1.0):
        raise ValueError("CQR predictions fall outside [0, 1]")
    crossing = (quantiles[:, 0] > quantiles[:, 1]) | (
        quantiles[:, 1] > quantiles[:, 2]
    )
    if int(crossing.sum()) != 0:
        raise ValueError("CQR quantile crossing detected")
    return quantiles


def build_prediction_frame(
    records: pd.DataFrame,
    quantiles: np.ndarray,
    expected_role: str,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    if set(records["role"].astype(str)) != {expected_role}:
        raise PermissionError(f"Prediction role guard failed for {expected_role}")
    expected_n = EXPECTED_N[expected_role] if enforce_expected_n else len(records)
    values = validate_quantile_array(quantiles, expected_n)
    if len(records) != expected_n:
        raise ValueError(
            f"{expected_role} prediction N guard failed: expected {expected_n}, got {len(records)}"
        )
    result = records.loc[:, MANIFEST_COLUMNS + ("irradiance",)].copy()
    result["q05"] = values[:, 0]
    result["q50"] = values[:, 1]
    result["q95"] = values[:, 2]
    return validate_prediction_frame(
        result, expected_role, enforce_expected_n=enforce_expected_n
    )


def validate_prediction_frame(
    frame: pd.DataFrame,
    expected_role: str,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            "Stage 3A1 prediction schema mismatch: "
            f"expected {PREDICTION_COLUMNS}, got {tuple(frame.columns)}"
        )
    forbidden = _truth_like_columns(frame.columns)
    if forbidden:
        raise ValueError(f"Truth-derived prediction fields are forbidden: {sorted(forbidden)}")
    guarded = _validate_locator_metadata(
        frame, expected_role, enforce_expected_n=enforce_expected_n
    )
    expected_n = EXPECTED_N[expected_role] if enforce_expected_n else len(frame)
    validate_quantile_array(frame.loc[:, ["q05", "q50", "q95"]].to_numpy(), expected_n)
    irradiance = pd.to_numeric(frame["irradiance"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(irradiance).all():
        raise ValueError("Raw irradiance must be finite")
    result = frame.loc[:, PREDICTION_COLUMNS].copy()
    # Keep the metadata validation result live so schema/role checks cannot diverge.
    if not np.array_equal(result["sample_id"].to_numpy(), guarded["sample_id"].to_numpy()):
        raise ValueError("Prediction metadata ordering changed during validation")
    return result


def quantile_qc_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    values = predictions.loc[:, ["q05", "q50", "q95"]].to_numpy(dtype=np.float64)
    summary: dict[str, Any] = {
        "N": int(len(predictions)),
        "quantile_crossing_count": int(
            ((values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])).sum()
        ),
    }
    for index, name in enumerate(("q05", "q50", "q95")):
        column = values[:, index]
        summary[name] = {
            "mean": float(column.mean()),
            "std_ddof0": float(column.std(ddof=0)),
            "min": float(column.min()),
            "max": float(column.max()),
        }
    width = values[:, 2] - values[:, 0]
    summary["raw_interval_width_q95_minus_q05"] = {
        "mean": float(width.mean()),
        "median": float(np.median(width)),
        "p95": float(np.quantile(width, 0.95)),
        "min": float(width.min()),
        "max": float(width.max()),
    }
    return summary


def image_preprocessing_schema() -> dict[str, Any]:
    return {
        "resize": [224, 224],
        "to_tensor": True,
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
        "random_augmentation": False,
    }


def make_config(
    checkpoint: Mapping[str, Any], device: torch.device, batch_size: int
) -> dict[str, Any]:
    validate_checkpoint_schema(checkpoint)
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "source_cqr_checkpoint": project_relative(SOURCE_CQR_CHECKPOINT),
        "source_cqr_checkpoint_sha256": SOURCE_CQR_CHECKPOINT_SHA256,
        "source_cqr_best_epoch": int(checkpoint["epoch"]),
        "source_cqr_validation_mean_pinball": float(
            checkpoint["validation_mean_pinball"]
        ),
        "quantile_levels": list(QUANTILE_LEVELS),
        "inference_mode": "deterministic",
        "model_forward_passes_per_sample": 1,
        "mc_dropout_performed": False,
        "repeated_forward_passes": False,
        "roles": list(ROLES),
        "expected_N": dict(EXPECTED_N),
        "prediction_columns": list(PREDICTION_COLUMNS),
        "batch_size": int(batch_size),
        "device": str(device),
        "cuda_autocast": device.type == "cuda",
        "irradiance_normalization": {
            "source": project_relative(TRAIN_IRRADIANCE_STATS_PATH),
            "source_role": "TRAIN",
            "N": EXPECTED_TRAIN_N,
            "mean": EXPECTED_IRRADIANCE_MEAN,
            "std_ddof0": EXPECTED_IRRADIANCE_STD_DDOF0,
            "method": "z_score",
        },
        "image_preprocessing": image_preprocessing_schema(),
    }


def make_provenance(
    checkpoint: Mapping[str, Any], prediction_qc: Mapping[str, Any]
) -> dict[str, Any]:
    validate_checkpoint_schema(checkpoint)
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "source_cqr_checkpoint": project_relative(SOURCE_CQR_CHECKPOINT),
        "source_cqr_checkpoint_sha256": SOURCE_CQR_CHECKPOINT_SHA256,
        "source_cqr_best_epoch": int(checkpoint["epoch"]),
        "source_cqr_validation_mean_pinball": float(
            checkpoint["validation_mean_pinball"]
        ),
        "quantile_levels": list(QUANTILE_LEVELS),
        "inference_mode": "deterministic",
        "model_forward_passes_per_sample": 1,
        "mc_dropout_performed": False,
        "repeated_forward_passes": False,
        "roles": list(ROLES),
        "expected_N": dict(EXPECTED_N),
        "truth_used_for_inference": False,
        "truth_saved_in_predictions": False,
        "cp_calibration_truth_accessed": False,
        "decision_development_truth_accessed": False,
        "conformal_calibration_performed": False,
        "interval_evaluation_performed": False,
        "risk_evaluation_performed": False,
        "cleaning_decision_performed": False,
        "economic_decision_performed": False,
        "random_test_accessed": False,
        "sealed_final_dates_accessed": False,
        "training_performed": False,
        "imagenet_download_performed": False,
        "irradiance_normalization_source": project_relative(
            TRAIN_IRRADIANCE_STATS_PATH
        ),
        "irradiance_normalization_source_role": "TRAIN",
        "irradiance_mean": EXPECTED_IRRADIANCE_MEAN,
        "irradiance_std_ddof0": EXPECTED_IRRADIANCE_STD_DDOF0,
        "image_preprocessing": image_preprocessing_schema(),
        "prediction_qc": dict(prediction_qc),
    }


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def validate_formal_output_path(output_dir: Path) -> Path:
    candidate = _resolved(output_dir)
    if candidate != _resolved(OUTPUT_DIR):
        raise PermissionError(f"Unauthorized Stage 3A1 output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_stage3a1_outputs(
    output_dir: Path,
    cp_predictions: pd.DataFrame,
    decision_predictions: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cp_predictions.to_csv(
        output_dir / "cp_calibration_predictions.csv", index=False, mode="x"
    )
    decision_predictions.to_csv(
        output_dir / "decision_development_predictions.csv", index=False, mode="x"
    )
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "provenance.json", provenance)


def run(
    protocol: str = PROTOCOL,
    checkpoint_path: Path = SOURCE_CQR_CHECKPOINT,
    output_dir: Path = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run the formal Stage 3A1 protocol only when explicitly invoked."""
    validate_protocol(protocol)
    validate_checkpoint_path(checkpoint_path)
    output_dir = validate_formal_output_path(output_dir)
    ensure_output_available(output_dir)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    stats = load_train_irradiance_stats()
    cp_manifest = load_role_manifest(CP_CALIBRATION_MANIFEST, CP_CALIBRATION_ROLE)
    decision_manifest = load_role_manifest(
        DECISION_DEVELOPMENT_MANIFEST, DECISION_DEVELOPMENT_ROLE
    )
    validate_role_isolation(cp_manifest, decision_manifest)
    cp_records = prepare_inference_records(cp_manifest, CP_CALIBRATION_ROLE, stats)
    decision_records = prepare_inference_records(
        decision_manifest, DECISION_DEVELOPMENT_ROLE, stats
    )
    transform = build_inference_transform()

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    cqr_train.point_train.set_seed(SEED)
    checkpoint = load_verified_cqr_checkpoint(checkpoint_path)
    model = build_inference_model(checkpoint, device)

    def infer_role(records: pd.DataFrame, role: str) -> pd.DataFrame:
        loader = DataLoader(
            Stage3A1InferenceDataset(records, transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        quantiles = predict_deterministic(model, loader, device)
        return build_prediction_frame(records, quantiles, role)

    cp_predictions = infer_role(cp_records, CP_CALIBRATION_ROLE)
    decision_predictions = infer_role(decision_records, DECISION_DEVELOPMENT_ROLE)
    validate_role_isolation(cp_predictions, decision_predictions)
    prediction_qc = {
        CP_CALIBRATION_ROLE: quantile_qc_summary(cp_predictions),
        DECISION_DEVELOPMENT_ROLE: quantile_qc_summary(decision_predictions),
    }
    config = make_config(checkpoint, device, batch_size)
    provenance = make_provenance(checkpoint, prediction_qc)
    write_stage3a1_outputs(
        output_dir, cp_predictions, decision_predictions, config, provenance
    )
    return {"config": config, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=SOURCE_CQR_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = run(
        protocol=args.protocol,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
