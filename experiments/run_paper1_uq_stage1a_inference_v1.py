"""Paper1 Clean UQ Stage 1A: deterministic and 50-pass MC Dropout inference.

This entry point is intentionally limited to CP_CALIBRATION and
DECISION_DEVELOPMENT.  It performs no conformal calibration, risk screening,
CQR, cleaning decision, training, or optimizer operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import experiments.train_paper1_resnet50_with_i_v1 as clean_train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"
ARCHITECTURE = "ResNet50+irradiance"
SEED = 42
BEST_EPOCH = 26
MC_PASSES = 50
MC_STD_DDOF = 1

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / PROTOCOL
CP_CALIBRATION_MANIFEST = SPLIT_ROOT / "cp_calibration.csv"
DECISION_DEVELOPMENT_MANIFEST = SPLIT_ROOT / "decision_development.csv"
AUTHORIZED_MANIFESTS = {
    "CP_CALIBRATION": CP_CALIBRATION_MANIFEST,
    "DECISION_DEVELOPMENT": DECISION_DEVELOPMENT_MANIFEST,
}
ROLES_USED = ("CP_CALIBRATION", "DECISION_DEVELOPMENT")
SEALED_FINAL_DATES = frozenset({"2017-06-15", "2017-06-24", "2017-06-30"})

CLEAN_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / PROTOCOL
    / "resnet50_with_i_v1"
    / "seed_42"
)
CLEAN_CHECKPOINT = CLEAN_RUN_DIR / "best_model.pth"
CLEAN_CHECKPOINT_SHA256 = (
    "97f3ec016cf99f83a78e28e2b4aca24787203f105243447d908da739c295de23"
)
TRAIN_IRRADIANCE_STATS_PATH = CLEAN_RUN_DIR / "train_irradiance_stats.json"
EXPECTED_IRRADIANCE_MEAN = 0.35047130969460816
EXPECTED_IRRADIANCE_STD = 0.208878529631844
LEGACY_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "models_ckpt" / "best_resnet50_with_i.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / "uq_stage1a_inference_v1"

PREDICTION_COLUMNS = (
    "sample_id",
    "date",
    "timestamp",
    "image_path",
    "role",
    "true_L",
    "irradiance",
    "point_pred",
    "mc_mean",
    "mc_std",
    "abs_error_point",
    "abs_error_mc_mean",
)


def validate_protocol(protocol: str) -> None:
    if protocol != PROTOCOL:
        raise PermissionError(f"Unauthorized protocol: {protocol!r}")


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_checkpoint_path(path: Path) -> Path:
    candidate = _resolved(path)
    if candidate == _resolved(LEGACY_CHECKPOINT) or candidate.name == LEGACY_CHECKPOINT.name:
        raise PermissionError("Legacy checkpoint is forbidden")
    if candidate != _resolved(CLEAN_CHECKPOINT):
        raise PermissionError(f"Unauthorized checkpoint: {candidate}")
    return candidate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> None:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint is missing embedded config metadata")
    expected = {
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "training_role": "TRAIN",
        "selection_role": "MODEL_VALIDATION",
        "legacy_checkpoint_loaded": False,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if checkpoint.get("epoch") != BEST_EPOCH:
        mismatches["epoch"] = (checkpoint.get("epoch"), BEST_EPOCH)
    if "model_state_dict" not in checkpoint:
        mismatches["model_state_dict"] = ("missing", "required")
    if mismatches:
        raise ValueError(f"Clean checkpoint metadata guard failed: {mismatches}")


def load_clean_checkpoint(path: Path = CLEAN_CHECKPOINT) -> Mapping[str, Any]:
    checkpoint_path = validate_checkpoint_path(path)
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != CLEAN_CHECKPOINT_SHA256:
        raise ValueError(
            "Clean checkpoint SHA256 mismatch: "
            f"expected {CLEAN_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Clean checkpoint payload must be a mapping")
    validate_checkpoint_metadata(checkpoint)
    return checkpoint


def validate_irradiance_stats(stats: Mapping[str, Any]) -> tuple[float, float]:
    if stats.get("source_role") != "TRAIN":
        raise PermissionError("Irradiance statistics must come from TRAIN only")
    if stats.get("normalization") != "z_score":
        raise ValueError("Expected TRAIN-only z-score irradiance normalization")
    mean = float(stats["mean"])
    std = float(stats["std_ddof0"])
    if not math.isclose(mean, EXPECTED_IRRADIANCE_MEAN, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("TRAIN irradiance mean does not match the clean training artifact")
    if not math.isclose(std, EXPECTED_IRRADIANCE_STD, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("TRAIN irradiance std does not match the clean training artifact")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid TRAIN irradiance standard deviation")
    return mean, std


def load_train_irradiance_stats(
    path: Path = TRAIN_IRRADIANCE_STATS_PATH,
) -> tuple[float, float]:
    if _resolved(path) != _resolved(TRAIN_IRRADIANCE_STATS_PATH):
        raise PermissionError("Only the clean TRAIN irradiance statistics artifact is authorized")
    with Path(path).open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return validate_irradiance_stats(stats)


def validate_manifest_authorization(path: Path, expected_role: str) -> Path:
    if expected_role not in AUTHORIZED_MANIFESTS:
        raise PermissionError(f"Forbidden inference role: {expected_role}")
    authorized = _resolved(AUTHORIZED_MANIFESTS[expected_role])
    candidate = _resolved(path)
    if candidate != authorized:
        raise PermissionError(
            f"Manifest is not authorized for {expected_role}: {candidate}"
        )
    return candidate


def validate_manifest_frame(frame: pd.DataFrame, expected_role: str) -> pd.DataFrame:
    required = {"sample_id", "image_path", "date", "timestamp", "role"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{expected_role} manifest is empty")
    if set(frame["role"].astype(str)) != {expected_role}:
        raise PermissionError(f"Role guard failed for {expected_role}")

    normalized_dates = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    sealed = set(normalized_dates) & SEALED_FINAL_DATES
    if sealed:
        raise PermissionError(f"Sealed final date rejected: {sorted(sealed)}")

    locators = frame["image_path"].astype(str)
    lowered_locators = locators.str.lower()
    if lowered_locators.str.contains("random_test", regex=False).any():
        raise PermissionError("RANDOM_TEST locator rejected")
    for sealed_date in SEALED_FINAL_DATES:
        if locators.str.contains(sealed_date, regex=False).any():
            raise PermissionError(f"Sealed final date locator rejected: {sealed_date}")

    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-null and unique")
    if frame["image_path"].isna().any() or frame["image_path"].duplicated().any():
        raise ValueError("image_path must be non-null and unique")
    return frame.loc[:, ["sample_id", "image_path", "date", "timestamp", "role"]].copy()


def load_role_manifest(path: Path, expected_role: str) -> pd.DataFrame:
    authorized_path = validate_manifest_authorization(path, expected_role)
    frame = pd.read_csv(authorized_path)
    return validate_manifest_frame(frame, expected_role)


def validate_role_isolation(cp: pd.DataFrame, decision: pd.DataFrame) -> None:
    if set(cp["role"]) != {"CP_CALIBRATION"}:
        raise PermissionError("First frame must be CP_CALIBRATION only")
    if set(decision["role"]) != {"DECISION_DEVELOPMENT"}:
        raise PermissionError("Second frame must be DECISION_DEVELOPMENT only")
    if set(cp["sample_id"]) & set(decision["sample_id"]):
        raise ValueError("CP_CALIBRATION/DECISION_DEVELOPMENT sample_id overlap")
    if set(cp["image_path"]) & set(decision["image_path"]):
        raise ValueError("CP_CALIBRATION/DECISION_DEVELOPMENT image_path overlap")


def prepare_records(
    frame: pd.DataFrame, expected_role: str, irradiance_mean: float, irradiance_std: float
) -> pd.DataFrame:
    guarded = validate_manifest_frame(frame, expected_role)
    result = clean_train.attach_development_values(guarded)
    result["irradiance"] = result["irradiance_raw"].astype(float)
    result["irradiance_normalized"] = (
        result["irradiance"] - irradiance_mean
    ) / irradiance_std
    return result


class Stage1AInferenceDataset(Dataset):
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


def build_inference_model(checkpoint: Mapping[str, Any], device: torch.device) -> nn.Module:
    validate_checkpoint_metadata(checkpoint)
    model = clean_train.Paper1ResNet50WithI(
        weights=clean_train.ResNet50_Weights.IMAGENET1K_V2,
        dropout=clean_train.DROPOUT,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    return model.to(device)


def set_deterministic_inference_mode(model: nn.Module) -> None:
    model.eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("Deterministic inference requires every module to remain in eval mode")


def enable_mc_dropout_only(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if not module.training:
                raise RuntimeError("Dropout activation failed")
        elif module.training:
            raise RuntimeError(
                f"MC mode must not activate non-Dropout module {type(module).__name__}"
            )
    batch_norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    if any(
        module.training
        for module in model.modules()
        if isinstance(module, batch_norm_types)
    ):
        raise RuntimeError("BatchNorm must remain in eval mode during MC Dropout")


def _predict_once(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for images, irradiance, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            irradiance = irradiance.to(device, non_blocking=True)
            with clean_train.autocast_context(device):
                outputs = model(images, irradiance).squeeze(1)
            predictions.append(outputs.float().cpu().numpy().astype(np.float64))
            indices.append(np.asarray(batch_indices, dtype=np.int64))
    values = np.concatenate(predictions)
    observed_indices = np.concatenate(indices)
    order = np.argsort(observed_indices)
    if not np.array_equal(observed_indices[order], np.arange(len(loader.dataset))):
        raise ValueError("Inference prediction order/index guard failed")
    return values[order]


def predict_deterministic(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    set_deterministic_inference_mode(model)
    return _predict_once(model, loader, device)


class MCSummaryAccumulator:
    """Online mean/sample-std accumulator; individual MC passes are not retained."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Each MC pass must be a one-dimensional prediction array")
        if self.mean is None:
            self.mean = np.zeros_like(values, dtype=np.float64)
            self.m2 = np.zeros_like(values, dtype=np.float64)
        if values.shape != self.mean.shape:
            raise ValueError("MC pass prediction shapes differ")
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count != MC_PASSES:
            raise ValueError(f"Exactly {MC_PASSES} MC passes are required; got {self.count}")
        assert self.mean is not None and self.m2 is not None
        return self.mean.copy(), np.sqrt(self.m2 / (self.count - MC_STD_DDOF))


def predict_mc_dropout(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    enable_mc_dropout_only(model)
    accumulator = MCSummaryAccumulator()
    for _ in range(MC_PASSES):
        accumulator.update(_predict_once(model, loader, device))
    mc_mean, mc_std = accumulator.finalize()
    model.eval()
    return mc_mean, mc_std


def build_prediction_frame(
    records: pd.DataFrame,
    point_pred: np.ndarray,
    mc_mean: np.ndarray,
    mc_std: np.ndarray,
) -> pd.DataFrame:
    n = len(records)
    arrays = {
        "point_pred": np.asarray(point_pred, dtype=np.float64),
        "mc_mean": np.asarray(mc_mean, dtype=np.float64),
        "mc_std": np.asarray(mc_std, dtype=np.float64),
    }
    if any(values.shape != (n,) for values in arrays.values()):
        raise ValueError("Prediction length does not match manifest length")
    result = records.loc[
        :, ["sample_id", "date", "timestamp", "image_path", "role", "true_L", "irradiance"]
    ].copy()
    for name, values in arrays.items():
        result[name] = values
    result["abs_error_point"] = np.abs(result["point_pred"] - result["true_L"])
    result["abs_error_mc_mean"] = np.abs(result["mc_mean"] - result["true_L"])
    return result.loc[:, PREDICTION_COLUMNS]


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_stage1a_outputs(
    output_dir: Path,
    cp_predictions: pd.DataFrame,
    decision_predictions: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cp_predictions.to_csv(output_dir / "cp_calibration_predictions.csv", index=False, mode="x")
    decision_predictions.to_csv(
        output_dir / "decision_development_predictions.csv", index=False, mode="x"
    )
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "provenance.json", provenance)


def make_config(
    checkpoint_path: Path,
    irradiance_mean: float,
    irradiance_std: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "best_epoch": BEST_EPOCH,
        "checkpoint_path": project_relative(checkpoint_path),
        "checkpoint_sha256": CLEAN_CHECKPOINT_SHA256,
        "mc_passes": MC_PASSES,
        "mc_std_ddof": MC_STD_DDOF,
        "irradiance_mean": irradiance_mean,
        "irradiance_std": irradiance_std,
        "irradiance_stats_source": "TRAIN_only",
        "image_preprocessing": "clean training validation transform",
        "batch_size": batch_size,
        "device": str(device),
        "passes_saved": False,
        "prediction_columns": list(PREDICTION_COLUMNS),
    }


def make_provenance(
    checkpoint_path: Path, irradiance_mean: float, irradiance_std: float
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "checkpoint_path": project_relative(checkpoint_path),
        "checkpoint_sha256": CLEAN_CHECKPOINT_SHA256,
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "mc_passes": MC_PASSES,
        "mc_std_ddof": MC_STD_DDOF,
        "irradiance_mean": irradiance_mean,
        "irradiance_std": irradiance_std,
        "irradiance_stats_source": "TRAIN_only",
        "roles_used": list(ROLES_USED),
        "random_test_accessed": False,
        "random_test_truth_accessed": False,
        "random_test_predictions_generated": False,
        "sealed_final_dates_accessed": False,
        "legacy_checkpoint_loaded": False,
        "training_performed": False,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "conformal_calibration_performed": False,
        "risk_screening_performed": False,
        "cqr_performed": False,
        "cleaning_decision_performed": False,
    }


def run(
    protocol: str = PROTOCOL,
    checkpoint_path: Path = CLEAN_CHECKPOINT,
    output_dir: Path = OUTPUT_DIR,
    batch_size: int = 32,
    device_name: str | None = None,
) -> dict[str, Any]:
    validate_protocol(protocol)
    checkpoint_path = validate_checkpoint_path(checkpoint_path)
    ensure_output_available(output_dir)
    irradiance_mean, irradiance_std = load_train_irradiance_stats()

    cp_manifest = load_role_manifest(CP_CALIBRATION_MANIFEST, "CP_CALIBRATION")
    decision_manifest = load_role_manifest(
        DECISION_DEVELOPMENT_MANIFEST, "DECISION_DEVELOPMENT"
    )
    validate_role_isolation(cp_manifest, decision_manifest)
    cp_records = prepare_records(
        cp_manifest, "CP_CALIBRATION", irradiance_mean, irradiance_std
    )
    decision_records = prepare_records(
        decision_manifest, "DECISION_DEVELOPMENT", irradiance_mean, irradiance_std
    )

    _, inference_transform = clean_train.build_transforms()
    if not clean_train.validation_transform_is_deterministic(inference_transform):
        raise RuntimeError("Clean validation image transform must be deterministic")
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    clean_train.set_seed(SEED)
    checkpoint = load_clean_checkpoint(checkpoint_path)
    model = build_inference_model(checkpoint, device)

    def infer_role(records: pd.DataFrame) -> pd.DataFrame:
        loader = DataLoader(
            Stage1AInferenceDataset(records, inference_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        point_pred = predict_deterministic(model, loader, device)
        mc_mean, mc_std = predict_mc_dropout(model, loader, device)
        return build_prediction_frame(records, point_pred, mc_mean, mc_std)

    cp_predictions = infer_role(cp_records)
    decision_predictions = infer_role(decision_records)
    config = make_config(
        checkpoint_path, irradiance_mean, irradiance_std, batch_size, device
    )
    provenance = make_provenance(checkpoint_path, irradiance_mean, irradiance_std)
    write_stage1a_outputs(
        output_dir, cp_predictions, decision_predictions, config, provenance
    )
    return {"config": config, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=CLEAN_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
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
