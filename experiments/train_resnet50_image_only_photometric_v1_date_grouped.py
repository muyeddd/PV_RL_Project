"""Isolated photometric_v1 adapter for the frozen Image-only baseline.

Only the training transform changes: saturation is 0.15 and a seeded
RandomGamma is inserted immediately before ToTensor. The baseline remains the
source of truth for data loading, model construction, optimization, metrics,
scheduling, early stopping, checkpointing, and telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import torch
from torchvision import transforms
from torchvision.transforms import functional as transform_functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_resnet50_image_only_date_grouped as baseline


DEFAULT_MANIFEST = baseline.DEFAULT_MANIFEST
DEFAULT_IMAGE_ROOT = baseline.DEFAULT_IMAGE_ROOT
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "resnet50_image_only_photometric_v1_date_grouped.json"
)
PILOT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "resnet50_image_only_photometric_v1"
    / "pilot"
)
TRAINING_VERSION = "resnet50_image_only_photometric_v1_date_grouped"
ALLOWED_PILOT_FOLDS = (3, 4)
PROTECTED_ROLES = (
    "cp_calibration",
    "decision_development",
    "final_test",
)
EXPECTED_FORBIDDEN_ROLE_COUNTS = {role: 0 for role in PROTECTED_ROLES}
EXPECTED_MANIFEST_SHA256 = (
    "a354afc2b691719bf0cc3c3982033da833795006e3e3b0122cae07810bd83e02"
)
SolarResNet50ImageOnly = baseline.SolarResNet50ImageOnly
_BASELINE_BUILD_TRANSFORMS = baseline.build_transforms

PREPROCESSING_DESCRIPTION = {
    "train": (
        "Resize(256,256); RandomResizedCrop(224, scale=(0.85,1.0)); "
        "RandomHorizontalFlip(0.5); RandomRotation(7); "
        "ColorJitter(brightness=0.08,contrast=0.08,saturation=0.15,hue=0.02); "
        "RandomGamma(p=0.5,gamma=(0.85,1.15),gain=1.0); ToTensor; "
        "ImageNet normalization"
    ),
    "validation": baseline.PREPROCESSING_DESCRIPTION["validation"],
    "normalization_mean": list(
        baseline.PREPROCESSING_DESCRIPTION["normalization_mean"]
    ),
    "normalization_std": list(
        baseline.PREPROCESSING_DESCRIPTION["normalization_std"]
    ),
}


class RandomGamma:
    """Apply gamma with a probability using only PyTorch's worker-seeded RNG."""

    def __init__(
        self,
        gamma: tuple[float, float] = (0.85, 1.15),
        probability: float = 0.5,
        gain: float = 1.0,
    ) -> None:
        gamma_min, gamma_max = (float(gamma[0]), float(gamma[1]))
        if gamma_min <= 0 or gamma_max < gamma_min:
            raise ValueError("gamma must be a positive ordered pair")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if gain <= 0:
            raise ValueError("gain must be positive")
        self.gamma = (gamma_min, gamma_max)
        self.probability = float(probability)
        self.gain = float(gain)

    def sample_gamma(self) -> float:
        gamma_min, gamma_max = self.gamma
        return float(
            gamma_min + (gamma_max - gamma_min) * torch.rand((), dtype=torch.float32)
        )

    def __call__(self, image):
        if float(torch.rand((), dtype=torch.float32)) >= self.probability:
            return image
        gamma = self.sample_gamma()
        return transform_functional.adjust_gamma(image, gamma=gamma, gain=self.gain)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(probability={self.probability}, "
            f"gamma={self.gamma}, gain={self.gain})"
        )


def build_transforms():
    baseline_train, baseline_validation = _BASELINE_BUILD_TRANSFORMS()
    components = list(baseline_train.transforms)
    component_names = [type(component).__name__ for component in components]
    expected_names = [
        "Resize",
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomRotation",
        "ColorJitter",
        "ToTensor",
        "Normalize",
    ]
    if component_names != expected_names:
        raise RuntimeError(
            "Frozen baseline train transform changed: "
            f"expected {expected_names}, got {component_names}"
        )

    color_jitter_index = component_names.index("ColorJitter")
    to_tensor_index = component_names.index("ToTensor")
    components[color_jitter_index] = transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
        saturation=0.15,
        hue=0.02,
    )
    components.insert(
        to_tensor_index,
        RandomGamma(gamma=(0.85, 1.15), probability=0.5, gain=1.0),
    )
    return transforms.Compose(components), baseline_validation


def _canonical_lf_sha256(path: Path) -> str:
    content = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


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
        "allowed_pilot_folds",
        "protected_roles",
        "pilot_output_namespace",
        "photometric_augmentation",
        "baseline_source_sha256_lf",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"photometric_v1 config is missing fields: {missing}")
    if config["training_version"] != TRAINING_VERSION:
        raise ValueError("Unexpected photometric_v1 training_version")
    if tuple(config["allowed_pilot_folds"]) != ALLOWED_PILOT_FOLDS:
        raise ValueError("photometric_v1 pilot folds are not frozen to Fold 3/4")
    if tuple(config["protected_roles"]) != PROTECTED_ROLES:
        raise ValueError("photometric_v1 protected roles changed")
    expected_output_namespace = PILOT_OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix()
    if config["pilot_output_namespace"] != expected_output_namespace:
        raise ValueError("photometric_v1 pilot output namespace changed")
    expected_photometric = {
        "brightness": 0.08,
        "contrast": 0.08,
        "saturation": 0.15,
        "hue": 0.02,
        "random_gamma": {
            "probability": 0.5,
            "gamma_min": 0.85,
            "gamma_max": 1.15,
            "gain": 1.0,
            "position": "after_color_jitter_before_to_tensor",
            "random_source": "torch_worker_rng",
        },
    }
    if config["photometric_augmentation"] != expected_photometric:
        raise ValueError("photometric_v1 augmentation configuration is not frozen")
    return config


def validate_baseline_integrity() -> None:
    expected = load_training_config()["baseline_source_sha256_lf"]
    mismatches = {}
    for relative_path, expected_sha256 in expected.items():
        path = PROJECT_ROOT / relative_path
        actual_sha256 = _canonical_lf_sha256(path)
        if actual_sha256 != expected_sha256:
            mismatches[relative_path] = {
                "expected": expected_sha256,
                "actual": actual_sha256,
            }
    if mismatches:
        raise RuntimeError(f"Frozen baseline source mismatch: {mismatches}")


def validate_single_factor_contract() -> None:
    validate_baseline_integrity()
    experiment_config = load_training_config()
    baseline_config = baseline.load_training_config()
    frozen_fields = (
        "schema_version",
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
    )
    mismatches = {
        field: {
            "baseline": baseline_config.get(field),
            "photometric_v1": experiment_config.get(field),
        }
        for field in frozen_fields
        if baseline_config.get(field) != experiment_config.get(field)
    }
    if mismatches:
        raise RuntimeError(f"photometric_v1 differs on controlled fields: {mismatches}")
    if SolarResNet50ImageOnly is not baseline.SolarResNet50ImageOnly:
        raise RuntimeError("photometric_v1 model is not the baseline model class")
    _, experiment_validation = build_transforms()
    _, baseline_validation = _BASELINE_BUILD_TRANSFORMS()
    if repr(experiment_validation) != repr(baseline_validation):
        raise RuntimeError("photometric_v1 validation transform differs from baseline")
    source = inspect.getsource(sys.modules[__name__])
    forbidden_tokens = (
        "WeightedRandom" + "Sampler",
        "torch" + ".load",
        "load_state_dict(" + "torch" + ".load",
    )
    if any(token in source for token in forbidden_tokens):
        raise RuntimeError("photometric_v1 source violates the single-factor contract")


def expected_output_dir(fold: int) -> Path:
    if isinstance(fold, bool) or fold not in ALLOWED_PILOT_FOLDS:
        raise ValueError("photometric_v1 pilot permits only Fold 3 and Fold 4")
    return PILOT_OUTPUT_ROOT / f"fold_{fold}_seed_42"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = load_training_config()
    parser = argparse.ArgumentParser(
        description="Run isolated photometric_v1 pilot training for Fold 3 or Fold 4"
    )
    parser.add_argument("--fold", type=int, choices=ALLOWED_PILOT_FOLDS, required=True)
    parsed = parser.parse_args(argv)
    parsed.manifest = DEFAULT_MANIFEST
    parsed.image_root = DEFAULT_IMAGE_ROOT
    parsed.output_dir = expected_output_dir(parsed.fold)
    parsed.seed = defaults["seed"]
    parsed.epochs = defaults["epochs"]
    parsed.batch_size = defaults["batch_size"]
    parsed.num_workers = defaults["num_workers"]
    parsed.learning_rate = defaults["learning_rate"]
    parsed.weight_decay = defaults["weight_decay"]
    parsed.patience = defaults["patience"]
    parsed.amp = defaults["amp"]
    parsed.pretrained = defaults["pretrained"]
    parsed.overwrite = False
    parsed.pilot_run = True
    return parsed


def preflight_pilot_fold(
    manifest_path: Path,
    image_root: Path,
    fold: int,
) -> dict[str, Any]:
    if fold not in ALLOWED_PILOT_FOLDS:
        raise ValueError("photometric_v1 pilot permits only Fold 3 and Fold 4")
    audit = baseline.preflight_fold(manifest_path, image_root, fold)
    if audit["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("photometric_v1 manifest is not frozen date_grouped_v1")
    if audit["forbidden_role_counts"] != EXPECTED_FORBIDDEN_ROLE_COUNTS:
        raise RuntimeError(
            "photometric_v1 attempted to select a protected role: "
            f"{audit['forbidden_role_counts']}"
        )
    for name in ("train_records", "validation_records"):
        roles = set(audit[name]["top_level_role"])
        if roles != {"model_development"}:
            raise RuntimeError(f"{name} contains protected roles: {sorted(roles)}")
    return audit


def validate_experiment_arguments(args: argparse.Namespace) -> None:
    config = load_training_config()
    if args.fold not in ALLOWED_PILOT_FOLDS:
        raise ValueError("photometric_v1 pilot permits only Fold 3 and Fold 4")
    if Path(args.manifest).resolve() != DEFAULT_MANIFEST.resolve():
        raise ValueError("photometric_v1 must use the frozen date_grouped_v1 manifest")
    if Path(args.image_root).resolve() != DEFAULT_IMAGE_ROOT.resolve():
        raise ValueError("photometric_v1 must use the frozen baseline image root")
    if Path(args.output_dir).resolve() != expected_output_dir(args.fold).resolve():
        raise ValueError("photometric_v1 output directory is not the isolated pilot path")
    expected = {
        "seed": config["seed"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "patience": config["patience"],
        "amp": config["amp"],
        "pretrained": config["pretrained"],
        "overwrite": False,
        "pilot_run": True,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(args, name, None)}
        for name, value in expected.items()
        if getattr(args, name, None) != value
    }
    if mismatches:
        raise ValueError(f"photometric_v1 controlled arguments changed: {mismatches}")
    baseline.validate_training_arguments(args)


def _add_experiment_metadata(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["experiment_version"] = TRAINING_VERSION
    result["photometric_augmentation"] = load_training_config()[
        "photometric_augmentation"
    ]
    result["final_test_accessed"] = False
    result["forbidden_roles_accessed"] = []
    return result


@contextmanager
def photometric_runtime():
    original_transform_builder = baseline.build_transforms
    original_config_loader = baseline.load_training_config
    original_preprocessing = baseline.PREPROCESSING_DESCRIPTION
    original_checkpoint_builder = baseline.build_checkpoint_payload
    original_json_writer = baseline._write_json

    def checkpoint_builder(*args, **kwargs):
        return _add_experiment_metadata(original_checkpoint_builder(*args, **kwargs))

    def json_writer(path, value):
        original_json_writer(path, _add_experiment_metadata(value))

    baseline.build_transforms = build_transforms
    baseline.load_training_config = load_training_config
    baseline.PREPROCESSING_DESCRIPTION = PREPROCESSING_DESCRIPTION
    baseline.build_checkpoint_payload = checkpoint_builder
    baseline._write_json = json_writer
    try:
        yield
    finally:
        baseline.build_transforms = original_transform_builder
        baseline.load_training_config = original_config_loader
        baseline.PREPROCESSING_DESCRIPTION = original_preprocessing
        baseline.build_checkpoint_payload = original_checkpoint_builder
        baseline._write_json = original_json_writer


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    validate_single_factor_contract()
    validate_experiment_arguments(args)
    preflight_pilot_fold(args.manifest, args.image_root, args.fold)
    with photometric_runtime():
        return baseline.run_training(args)


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
