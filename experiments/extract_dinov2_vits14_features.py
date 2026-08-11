"""Extract deterministic frozen DINOv2-S/14 features from model-development data.

The default command performs no extraction. ``--smoke-test`` processes only the
first configured 32 model-development filenames and writes no feature cache.
``--full`` is the explicit, future-only command for creating the complete cache.
Official DINOv2 code and weights are loaded from the existing torch.hub cache;
network downloads are deliberately blocked.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "features" / "dinov2_vits14_frozen_v1.json"
)
EXPECTED_OUTPUT_DIR = PROJECT_ROOT / "features" / "dinov2_vits14_frozen_v1"
REQUIRED_MANIFEST_COLUMNS = (
    "filename",
    "date",
    "top_level_role",
    "cv_validation_fold",
)
METADATA_COLUMNS = ("row_index", "filename", "date", "cv_validation_fold", "L")
LOSS_PATTERN = re.compile(
    r"(?:^|_)L_([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)_I_"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ordered_strings(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "feature_version",
        "split_version",
        "manifest_path",
        "split_fingerprint_path",
        "image_root",
        "output_dir",
        "allowed_top_level_role",
        "forbidden_top_level_roles",
        "expected_sample_count",
        "ordering",
        "torch_hub_repo",
        "torch_hub_model",
        "torch_hub_source",
        "cached_repository_glob",
        "cached_weight_filename",
        "cached_weight_sha256",
        "feature_dimension",
        "image_size",
        "batch_size",
        "num_workers",
        "amp",
        "normalization",
        "smoke_sample_count",
        "source_artifact_sha256",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Feature config is missing fields: {missing}")
    if config["feature_version"] != "dinov2_vits14_frozen_v1":
        raise ValueError("Unexpected feature_version")
    if config["split_version"] != "date_grouped_v1":
        raise ValueError("Only date_grouped_v1 is allowed")
    if config["allowed_top_level_role"] != "model_development":
        raise ValueError("Only model_development is allowed")
    if set(config["forbidden_top_level_roles"]) != {
        "cp_calibration",
        "decision_development",
        "final_test",
    }:
        raise ValueError("Protected-role policy does not match date_grouped_v1")
    if config["expected_sample_count"] != 25716:
        raise ValueError("Expected model_development count must be 25,716")
    if config["feature_dimension"] != 384:
        raise ValueError("dinov2_vits14 feature dimension must be 384")
    if config["image_size"] != 224 or config["batch_size"] != 32:
        raise ValueError("Frozen v1 requires image_size=224 and batch_size=32")
    if config["ordering"] != "filename_ascending":
        raise ValueError("Frozen v1 ordering must be filename_ascending")
    return config


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_source_artifacts(config: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative_path, expected_hash in config["source_artifact_sha256"].items():
        path = resolve_project_path(relative_path)
        digest = sha256_file(path)
        if digest.lower() != str(expected_hash).lower():
            raise ValueError(
                f"Frozen source artifact changed: {relative_path}; "
                f"expected={expected_hash}, actual={digest}"
            )
        actual[relative_path] = digest
    return actual


def parse_loss_label(filename: str) -> float:
    match = LOSS_PATTERN.search(filename)
    if match is None:
        raise ValueError(f"Cannot parse L from model-development filename: {filename}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite L in filename: {filename}")
    return value


def load_model_development_records(
    manifest_path: Path,
    image_root: Path,
    expected_count: int = 25716,
    limit: int | None = None,
) -> pd.DataFrame:
    """Select only model_development before resolving any image or label."""

    manifest = pd.read_csv(manifest_path, usecols=list(REQUIRED_MANIFEST_COLUMNS))
    if manifest["filename"].isna().any() or manifest["filename"].duplicated().any():
        raise ValueError("Manifest filenames must be non-null and globally unique")

    records = manifest.loc[
        manifest["top_level_role"].eq("model_development"),
        list(REQUIRED_MANIFEST_COLUMNS),
    ].copy()
    if len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} model_development rows, found {len(records)}"
        )
    if set(records["top_level_role"]) != {"model_development"}:
        raise ValueError("A protected role entered feature extraction")

    folds = pd.to_numeric(records["cv_validation_fold"], errors="raise")
    if folds.isna().any() or not folds.map(float.is_integer).all():
        raise ValueError("Every model_development row must have an integer CV fold")
    records["cv_validation_fold"] = folds.astype(int)
    if set(records["cv_validation_fold"]) != {1, 2, 3, 4}:
        raise ValueError("Expected exactly date_grouped_v1 folds 1-4")

    records = records.sort_values("filename", kind="mergesort").reset_index(drop=True)
    if limit is not None:
        if isinstance(limit, bool) or limit <= 0 or limit > 64:
            raise ValueError("Smoke limit must be an integer from 1 to 64")
        records = records.iloc[:limit].copy().reset_index(drop=True)

    # Paths and L are resolved only after the protected-role filter and smoke limit.
    image_root = Path(image_root).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    records["row_index"] = np.arange(len(records), dtype=np.int64)
    records["L"] = [parse_loss_label(name) for name in records["filename"]]
    records["image_path"] = [str(image_root / name) for name in records["filename"]]
    missing = [path for path in records["image_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Selected model-development image is missing: {missing[0]}")
    return records


def build_deterministic_transform(config: dict[str, Any]):
    return transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["normalization"]["mean"],
                std=config["normalization"]["std"],
            ),
        ]
    )


class FrozenFeatureImageDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: Any):
        if set(records["top_level_role"]) != {"model_development"}:
            raise ValueError("Dataset may contain only model_development")
        self.records = records.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), int(row["row_index"]), str(row["filename"])


def torch_hub_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    hub_dir = Path(torch.hub.get_dir())
    repositories = sorted(
        path
        for path in hub_dir.glob(config["cached_repository_glob"])
        if path.is_dir() and (path / "hubconf.py").is_file()
    )
    if not repositories:
        raise FileNotFoundError(
            "Official facebookresearch/dinov2 repository is absent from torch.hub cache"
        )
    repository = repositories[-1]
    weight = hub_dir / "checkpoints" / config["cached_weight_filename"]
    if not weight.is_file():
        raise FileNotFoundError(f"Cached DINOv2 weight is missing: {weight}")
    actual_hash = sha256_file(weight)
    if actual_hash.lower() != config["cached_weight_sha256"].lower():
        raise ValueError(
            f"Cached DINOv2 weight SHA256 mismatch: expected "
            f"{config['cached_weight_sha256']}, got {actual_hash}"
        )
    return repository, weight


@contextlib.contextmanager
def block_torch_hub_downloads() -> Iterator[None]:
    original = torch.hub.download_url_to_file

    def blocked(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("Network downloads are disabled for frozen DINOv2 v1")

    torch.hub.download_url_to_file = blocked
    try:
        yield
    finally:
        torch.hub.download_url_to_file = original


def freeze_backbone(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DINOv2 backbone was not fully frozen")
    return model


def load_cached_official_backbone(
    config: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    repository, weight = torch_hub_paths(config)
    with block_torch_hub_downloads():
        model = torch.hub.load(
            str(repository),
            config["torch_hub_model"],
            source="local",
            pretrained=True,
            verbose=False,
        )
    model = freeze_backbone(model).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    provenance = {
        "torch_hub_repo": config["torch_hub_repo"],
        "torch_hub_model": config["torch_hub_model"],
        "torch_hub_source": config["torch_hub_source"],
        "cached_repository_path": str(repository),
        "cached_weight_path": str(weight),
        "cached_weight_sha256": sha256_file(weight),
        "device": str(device),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    return model, provenance


@dataclass
class ExtractionResult:
    features: np.ndarray
    metadata: pd.DataFrame


def extract_features(
    model: torch.nn.Module,
    records: pd.DataFrame,
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> ExtractionResult:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Feature extraction requires a frozen eval-mode backbone")
    dataset = FrozenFeatureImageDataset(records, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    chunks: list[torch.Tensor] = []
    observed_indices: list[int] = []
    observed_filenames: list[str] = []
    with torch.inference_mode():
        for images, row_indices, filenames in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                output = model(images)
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("DINOv2 backbone must return a 2D feature tensor")
            chunks.append(output.detach().to(dtype=torch.float32, device="cpu"))
            observed_indices.extend(int(value) for value in row_indices.tolist())
            observed_filenames.extend(str(value) for value in filenames)

    feature_tensor = torch.cat(chunks, dim=0)
    features = feature_tensor.numpy()
    expected_indices = records["row_index"].astype(int).tolist()
    expected_filenames = records["filename"].astype(str).tolist()
    if observed_indices != expected_indices or observed_filenames != expected_filenames:
        raise RuntimeError("Feature rows and metadata filenames are misaligned")
    if features.shape != (len(records), 384):
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"Unexpected feature dtype: {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("DINOv2 features contain NaN or Inf")
    metadata = records.loc[:, list(METADATA_COLUMNS)].copy()
    return ExtractionResult(features=features, metadata=metadata)


def feature_statistics(features: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(features.mean(dtype=np.float64)),
        "std": float(features.std(dtype=np.float64)),
        "min": float(features.min()),
        "max": float(features.max()),
        "average_l2_norm": float(
            np.linalg.norm(features.astype(np.float64), axis=1).mean()
        ),
    }


def validate_output_path(config: dict[str, Any]) -> Path:
    configured = resolve_project_path(config["output_dir"]).resolve()
    if configured != EXPECTED_OUTPUT_DIR.resolve():
        raise ValueError(
            f"Frozen v1 output must remain isolated at {EXPECTED_OUTPUT_DIR.resolve()}"
        )
    return configured


def write_full_cache(
    result: ExtractionResult,
    config: dict[str, Any],
    provenance: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if result.features.shape != (25716, 384):
        raise ValueError("Refusing to write an incomplete formal feature cache")
    if len(result.metadata) != 25716:
        raise ValueError("Formal metadata must contain exactly 25,716 rows")
    if result.metadata["row_index"].tolist() != list(range(25716)):
        raise ValueError("Formal metadata row_index is not contiguous")
    if not result.metadata["filename"].is_unique:
        raise ValueError("Formal metadata filenames are not unique")

    output_dir = validate_output_path(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing feature cache: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    metadata_path = output_dir / "metadata.csv"
    manifest_path = output_dir / "feature_manifest.json"
    fingerprint_path = output_dir / "dataset_fingerprint.json"

    np.save(feature_path, result.features, allow_pickle=False)
    result.metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    ordered_filenames = result.metadata["filename"].astype(str).tolist()
    fingerprint = {
        "schema_version": 1,
        "feature_version": config["feature_version"],
        "split_version": config["split_version"],
        "top_level_role": "model_development",
        "sample_count": len(ordered_filenames),
        "ordering": config["ordering"],
        "ordered_filenames_sha256": sha256_ordered_strings(ordered_filenames),
        "source_artifact_sha256": source_hashes,
    }
    fingerprint_path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "feature_version": config["feature_version"],
        "split_version": config["split_version"],
        "top_level_role": "model_development",
        "sample_count": int(result.features.shape[0]),
        "feature_dimension": int(result.features.shape[1]),
        "feature_dtype": str(result.features.dtype),
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "normalization": config["normalization"],
        "deterministic_transform": [
            "Image.open(...).convert('RGB')",
            "Resize((224,224))",
            "ToTensor()",
            "Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])",
        ],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "device": provenance["device"],
        "model_provenance": provenance,
        "feature_statistics": feature_statistics(result.features),
        "features_sha256": sha256_file(feature_path),
        "metadata_sha256": sha256_file(metadata_path),
        "dataset_fingerprint_sha256": sha256_file(fingerprint_path),
        "irradiance_used_as_model_input": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_records(config: dict[str, Any], smoke: bool) -> pd.DataFrame:
    limit = config["smoke_sample_count"] if smoke else None
    return load_model_development_records(
        resolve_project_path(config["manifest_path"]),
        resolve_project_path(config["image_root"]),
        expected_count=config["expected_sample_count"],
        limit=limit,
    )


def run_extraction(config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    source_hashes = verify_source_artifacts(config)
    records = prepare_records(config, smoke=smoke)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, provenance = load_cached_official_backbone(config, device)
    transform = build_deterministic_transform(config)
    result = extract_features(
        model=model,
        records=records,
        transform=transform,
        device=device,
        batch_size=config["batch_size"],
        num_workers=0 if smoke else config["num_workers"],
        amp=config["amp"],
    )
    summary = {
        "mode": "smoke" if smoke else "full",
        "sample_count": len(records),
        "feature_shape": list(result.features.shape),
        "feature_dtype": str(result.features.dtype),
        "finite": bool(np.isfinite(result.features).all()),
        "first_filename": result.metadata.iloc[0]["filename"],
        "last_filename": result.metadata.iloc[-1]["filename"],
        "feature_statistics": feature_statistics(result.features),
        "device": str(device),
        "model_provenance": provenance,
        "inference_mode_used": True,
        "selected_top_level_roles": sorted(records["top_level_role"].unique()),
        "forbidden_role_count": 0,
    }
    if not smoke:
        summary["feature_manifest"] = write_full_cache(
            result, config, provenance, source_hashes
        )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen official DINOv2-S/14 model-development features"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process only the first 32 sorted model-development images; write nothing",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Explicitly create the complete isolated 25,716-row feature cache",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    summary = run_extraction(config, smoke=args.smoke_test)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
