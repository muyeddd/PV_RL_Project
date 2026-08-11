"""ConvNeXt-Tiny image-only training with one L-balanced sampler change.

The frozen ConvNeXt-Tiny v1 implementation remains the source of truth for
the model, transforms, loss, optimizer, scheduler, AMP, early stopping,
checkpoint selection, and validation. This adapter replaces only the training
DataLoader's uniform shuffled sampling with fixed sqrt inverse-frequency L-bin
weighted replacement sampling.

Importing this module never starts training.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_convnext_tiny_image_only_date_grouped as baseline


BASELINE_PROTOCOL_VALIDATOR = baseline.validate_frozen_protocol
DEFAULT_MANIFEST = baseline.DEFAULT_MANIFEST
DEFAULT_IMAGE_ROOT = baseline.DEFAULT_IMAGE_ROOT
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "convnext_tiny_image_only_lbalanced_v1_date_grouped.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "convnext_tiny_image_only_lbalanced_v1"
    / "pilot"
)
BASELINE_OUTPUT_ROOT = baseline.OUTPUT_ROOT
TRAINING_VERSION = "convnext_tiny_image_only_lbalanced_v1_date_grouped"
BASELINE_TRAINING_VERSION = "convnext_tiny_image_only_v1_date_grouped"
MODEL_NAME = baseline.MODEL_NAME
ALLOWED_PILOT_FOLDS = (3, 4)
SEED = 42
ALPHA = 0.5
BIN_EDGES = (-np.inf, 0.1, 0.3, 0.5, 0.7, np.inf)
FINITE_BIN_EDGES = np.asarray(BIN_EDGES[1:-1], dtype=np.float64)
BIN_LABELS = (
    "(-inf,0.1)",
    "[0.1,0.3)",
    "[0.3,0.5)",
    "[0.5,0.7)",
    "[0.7,+inf)",
)
FROZEN_TRAIN_BIN_COUNTS = {
    3: (6_549, 6_321, 1_940, 4_154, 252),
    4: (7_404, 4_981, 994, 5_111, 1_113),
}


def load_training_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "training_version",
        "baseline_training_version",
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
        "fold3_train_count",
        "fold3_validation_count",
        "fold4_train_count",
        "fold4_validation_count",
        "transform_source",
        "pilot_output_namespace",
        "sampling",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"ConvNeXt L-balanced config missing fields: {missing}")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported ConvNeXt L-balanced schema_version")
    return config


def validate_single_factor_contract(config: dict[str, Any] | None = None) -> None:
    current = config or load_training_config()
    reference = baseline.load_training_config()
    BASELINE_PROTOCOL_VALIDATOR(reference)

    frozen_fields = (
        "schema_version",
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
        "model_development_sample_count",
        "fold3_train_count",
        "fold3_validation_count",
        "fold4_train_count",
        "fold4_validation_count",
        "transform_source",
    )
    mismatches = {
        key: {"expected": reference.get(key), "actual": current.get(key)}
        for key in frozen_fields
        if current.get(key) != reference.get(key)
    }
    if mismatches:
        raise RuntimeError(
            f"ConvNeXt L-balanced protocol differs from v1 baseline: {mismatches}"
        )

    expected_sampling = {
        "method": "WeightedRandomSampler",
        "bin_edges": ["-inf", 0.1, 0.3, 0.5, 0.7, "+inf"],
        "bin_interval_rule": "left-closed, right-open",
        "weight_formula": "n_bin^-0.5",
        "alpha": ALPHA,
        "normalize_weights": False,
        "replacement": True,
        "num_samples": "train_dataset_length",
        "sampler_seed": SEED,
        "worker_generator_seed": SEED,
    }
    expected_variant = {
        "training_version": TRAINING_VERSION,
        "baseline_training_version": BASELINE_TRAINING_VERSION,
        "allowed_pilot_folds": [3, 4],
        "pilot_output_namespace": (
            "outputs/date_grouped_v1/"
            "convnext_tiny_image_only_lbalanced_v1/pilot"
        ),
        "sampling": expected_sampling,
    }
    variant_mismatches = {
        key: {"expected": value, "actual": current.get(key)}
        for key, value in expected_variant.items()
        if current.get(key) != value
    }
    if variant_mismatches:
        raise RuntimeError(
            f"ConvNeXt L-balanced variant contract mismatch: {variant_mismatches}"
        )


def validate_pilot_fold(fold: int) -> None:
    if isinstance(fold, bool) or fold not in ALLOWED_PILOT_FOLDS:
        raise ValueError("ConvNeXt-Tiny L-balanced v1 pilot permits only Fold3/Fold4")


def expected_output_dir(fold: int, seed: int = SEED) -> Path:
    validate_pilot_fold(fold)
    if seed != SEED:
        raise ValueError("ConvNeXt-Tiny L-balanced v1 requires seed=42")
    return OUTPUT_ROOT / f"fold_{fold}_seed_{seed}"


def validate_output_namespace() -> None:
    expected = (
        PROJECT_ROOT
        / "outputs"
        / "date_grouped_v1"
        / "convnext_tiny_image_only_lbalanced_v1"
        / "pilot"
    ).resolve()
    actual = OUTPUT_ROOT.resolve()
    baseline_root = BASELINE_OUTPUT_ROOT.resolve()
    if actual != expected:
        raise RuntimeError(f"Unexpected ConvNeXt L-balanced output root: {actual}")
    if actual == baseline_root or actual.is_relative_to(baseline_root):
        raise RuntimeError("L-balanced output overlaps the frozen ConvNeXt baseline")


def ensure_new_output_target(output_dir: Path, fold: int, seed: int) -> Path:
    expected = expected_output_dir(fold, seed).resolve()
    actual = Path(output_dir).resolve()
    if actual != expected:
        raise RuntimeError(f"Output directory escaped L-balanced namespace: {actual}")
    if actual.exists():
        raise FileExistsError(f"Refusing to overwrite L-balanced output: {actual}")
    return actual


def assign_l_bin_ids(labels: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Training labels must be a finite one-dimensional array")
    return np.searchsorted(FINITE_BIN_EDGES, values, side="right").astype(np.int64)


def labels_from_training_records(train_records, fold: int) -> np.ndarray:
    if train_records.empty:
        raise ValueError("Training records are empty")
    if set(train_records["top_level_role"].astype(str)) != {"model_development"}:
        raise ValueError("Sampler labels require model_development training records")
    fold_ids = train_records["cv_validation_fold"].astype(int)
    if fold_ids.eq(fold).any():
        raise ValueError("Validation records must not enter sampler weight calculation")
    return np.asarray(
        [baseline.parse_loss_label(name) for name in train_records["filename"]],
        dtype=np.float64,
    )


def compute_train_bin_counts(bin_ids: np.ndarray) -> np.ndarray:
    counts = np.bincount(bin_ids, minlength=len(BIN_LABELS)).astype(np.int64)
    if len(counts) != len(BIN_LABELS) or np.any(counts <= 0):
        raise ValueError(f"Every fixed L bin must be non-empty, got {counts.tolist()}")
    return counts


def compute_sample_weights(bin_ids: np.ndarray, counts: np.ndarray) -> torch.Tensor:
    count_tensor = torch.as_tensor(counts, dtype=torch.double)
    id_tensor = torch.as_tensor(bin_ids, dtype=torch.long)
    return count_tensor[id_tensor].double().pow(-ALPHA)


def theoretical_sampling_probabilities(counts: np.ndarray) -> np.ndarray:
    roots = np.sqrt(np.asarray(counts, dtype=np.float64))
    return roots / roots.sum()


def build_sampling_plan(train_records, fold: int) -> dict[str, Any]:
    validate_pilot_fold(fold)
    labels = labels_from_training_records(train_records, fold)
    bin_ids = assign_l_bin_ids(labels)
    counts = compute_train_bin_counts(bin_ids)
    expected_counts = np.asarray(FROZEN_TRAIN_BIN_COUNTS[fold], dtype=np.int64)
    expected_train_count = baseline.EXPECTED_FOLD_COUNTS[fold][0]
    if len(labels) != expected_train_count or int(counts.sum()) != expected_train_count:
        raise RuntimeError(
            f"Fold{fold} training count mismatch: labels={len(labels)}, "
            f"bins={int(counts.sum())}, expected={expected_train_count}"
        )
    if not np.array_equal(counts, expected_counts):
        raise RuntimeError(
            f"Fold{fold} frozen L-bin mismatch: expected "
            f"{expected_counts.tolist()}, got {counts.tolist()}"
        )
    weights = compute_sample_weights(bin_ids, counts)
    probabilities = theoretical_sampling_probabilities(counts)
    provenance = {
        "fold": fold,
        "train_dates": sorted(train_records["date"].astype(str).unique().tolist()),
        "bin_edges": ["-inf", 0.1, 0.3, 0.5, 0.7, "+inf"],
        "bin_interval_rule": "left-closed, right-open",
        "train_bin_counts": dict(zip(BIN_LABELS, counts.tolist())),
        "weight_formula": "n_bin^-0.5",
        "alpha": ALPHA,
        "weights_normalized": False,
        "expected_sampling_probability_per_bin": dict(
            zip(BIN_LABELS, probabilities.tolist())
        ),
        "num_samples": len(labels),
        "replacement": True,
        "sampler_seed": SEED,
        "worker_generator_seed": SEED,
        "train_only_weights": True,
        "validation_weighted": False,
        "protected_roles_accessed": False,
        "cp_calibration_accessed": False,
        "decision_development_accessed": False,
        "final_test_accessed": False,
    }
    return {
        "provenance": provenance,
        "bin_ids": bin_ids,
        "weights": weights,
    }


def preflight_fold(fold: int) -> dict[str, Any]:
    validate_pilot_fold(fold)
    audit = baseline.preflight_fold(
        fold,
        manifest_path=DEFAULT_MANIFEST,
        image_root=DEFAULT_IMAGE_ROOT,
        verify_selected_paths=False,
    )
    if audit["selected_role"] != "model_development":
        raise RuntimeError("ConvNeXt L-balanced preflight selected a protected role")
    if audit["forbidden_roles_accessed"] or audit["final_test_accessed"]:
        raise RuntimeError("ConvNeXt L-balanced preflight accessed a protected role")
    plan = build_sampling_plan(audit["train_records"], fold)
    plan["provenance"]["manifest_sha256"] = audit["manifest_sha256"]
    result = dict(audit)
    result["sampling_plan"] = plan
    result["sampling_provenance"] = plan["provenance"]
    return result


class ConvNeXtLBalancedDataLoaderFactory:
    """Replace only the ConvNeXt training loader with the weighted sampler."""

    def __init__(self, train_records, validation_records, sampling_plan):
        self.train_records = train_records.reset_index(drop=True).copy()
        self.validation_records = validation_records.reset_index(drop=True).copy()
        self.plan = sampling_plan
        self.calls = 0
        self.train_sampler: WeightedRandomSampler | None = None
        self.sampler_generator: torch.Generator | None = None
        self.worker_generator: torch.Generator | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        return self.plan["provenance"]

    @staticmethod
    def _dataset_filenames(dataset) -> list[str]:
        if not hasattr(dataset, "records"):
            raise RuntimeError("ConvNeXt dataset must expose dataset.records")
        records = dataset.records
        if "filename" not in records.columns:
            raise RuntimeError("ConvNeXt dataset.records lacks filename")
        return records["filename"].astype(str).tolist()

    def __call__(self, dataset, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            if kwargs.get("shuffle") is not True or "sampler" in kwargs:
                raise RuntimeError("Frozen ConvNeXt training loader signature changed")
            expected_files = self.train_records["filename"].astype(str).tolist()
            actual_files = self._dataset_filenames(dataset)
            if actual_files != expected_files:
                raise RuntimeError(
                    "Training dataset.records filename order differs from sampler weights"
                )
            if len(dataset) != len(self.plan["weights"]):
                raise RuntimeError("Training dataset and sampler weight lengths differ")
            self.sampler_generator = torch.Generator().manual_seed(SEED)
            self.worker_generator = torch.Generator().manual_seed(SEED)
            if self.sampler_generator is self.worker_generator:
                raise RuntimeError("Sampler and worker generators must be independent")
            self.train_sampler = WeightedRandomSampler(
                weights=self.plan["weights"],
                num_samples=len(dataset),
                replacement=True,
                generator=self.sampler_generator,
            )
            kwargs["shuffle"] = False
            kwargs["sampler"] = self.train_sampler
            kwargs["generator"] = self.worker_generator
            return DataLoader(dataset, *args, **kwargs)

        if self.calls == 2:
            if kwargs.get("shuffle") is not False or "sampler" in kwargs:
                raise RuntimeError("Frozen ConvNeXt validation loader signature changed")
            expected_files = self.validation_records["filename"].astype(str).tolist()
            actual_files = self._dataset_filenames(dataset)
            if actual_files != expected_files:
                raise RuntimeError("Validation dataset.records filename order changed")
            if len(dataset) != len(self.validation_records):
                raise RuntimeError("Validation dataset sample count changed")
            return DataLoader(dataset, *args, **kwargs)

        raise RuntimeError("Unexpected additional ConvNeXt DataLoader construction")


@contextmanager
def lbalanced_runtime(factory: ConvNeXtLBalancedDataLoaderFactory):
    original_loader = baseline.DataLoader
    original_output_root = baseline.OUTPUT_ROOT
    original_protocol_validator = baseline.validate_frozen_protocol
    original_json_writer = baseline._write_json

    def json_writer(path: Path, value: dict[str, Any]) -> None:
        enriched = dict(value)
        if Path(path).name in {
            "config_snapshot.json",
            "final_metrics.json",
            "run_metadata.json",
        }:
            enriched["sampling_provenance"] = factory.provenance
        original_json_writer(path, enriched)

    baseline.DataLoader = factory
    baseline.OUTPUT_ROOT = OUTPUT_ROOT
    baseline.validate_frozen_protocol = validate_single_factor_contract
    baseline._write_json = json_writer
    try:
        yield
    finally:
        baseline.DataLoader = original_loader
        baseline.OUTPUT_ROOT = original_output_root
        baseline.validate_frozen_protocol = original_protocol_validator
        baseline._write_json = original_json_writer


def validate_experiment_arguments(args: argparse.Namespace) -> None:
    validate_pilot_fold(args.fold)
    config = load_training_config(Path(args.config))
    validate_single_factor_contract(config)
    if Path(args.config).resolve() != DEFAULT_CONFIG_PATH.resolve():
        raise ValueError("The L-balanced pilot requires its frozen config")
    if Path(args.manifest).resolve() != DEFAULT_MANIFEST.resolve():
        raise ValueError("The L-balanced pilot requires the frozen manifest")
    if Path(args.image_root).resolve() != DEFAULT_IMAGE_ROOT.resolve():
        raise ValueError("The L-balanced pilot requires the frozen image root")
    locked = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
    }
    mismatches = {
        key: {"expected": config[key], "actual": value}
        for key, value in locked.items()
        if value != config[key]
    }
    if mismatches:
        raise ValueError(f"ConvNeXt L-balanced argument override prohibited: {mismatches}")
    validate_output_namespace()
    ensure_new_output_target(args.output_dir, args.fold, args.seed)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run one L-balanced pilot fold; importing and testing never calls this."""

    validate_experiment_arguments(args)
    audit = preflight_fold(args.fold)
    factory = ConvNeXtLBalancedDataLoaderFactory(
        audit["train_records"],
        audit["validation_records"],
        audit["sampling_plan"],
    )
    with lbalanced_runtime(factory):
        result = baseline.run_training(args)
    if factory.calls != 2:
        raise RuntimeError(f"Expected two DataLoaders, observed {factory.calls}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config = load_training_config()
    parser = argparse.ArgumentParser(
        description="Run the Fold3/Fold4 ConvNeXt-Tiny L-balanced v1 pilot"
    )
    parser.add_argument("--fold", type=int, required=True)
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
