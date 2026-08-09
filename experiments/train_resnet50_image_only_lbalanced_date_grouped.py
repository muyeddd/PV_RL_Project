"""Formal ResNet50 Image-only training with one L-balanced sampling change.

The frozen Image-only training implementation remains the source of truth for
the model, transforms, optimizer, loss, scheduler, metrics, early stopping,
checkpointing, and telemetry.  This module injects only a training DataLoader
using sqrt inverse-frequency weighted replacement sampling.
"""

from __future__ import annotations

import argparse
import inspect
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

from experiments import train_resnet50_image_only_date_grouped as baseline
from models.resnet50_image_only import SolarResNet50ImageOnly
from utils.parser import parse_filename


DEFAULT_MANIFEST = baseline.DEFAULT_MANIFEST
DEFAULT_IMAGE_ROOT = baseline.DEFAULT_IMAGE_ROOT
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training"
    / "resnet50_image_only_lbalanced_date_grouped_v1.json"
)
EXPECTED_FOLD_COUNTS = baseline.EXPECTED_FOLD_COUNTS
MODEL_NAME = baseline.MODEL_NAME
PREPROCESSING_DESCRIPTION = baseline.PREPROCESSING_DESCRIPTION
TRAINING_VERSION = "resnet50_image_only_lbalanced_date_grouped_v1"
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
    1: (8388, 5001, 2357, 2820, 1239),
    2: (7923, 3395, 2269, 3710, 1227),
    3: (6549, 6321, 1940, 4154, 252),
    4: (7404, 4981, 994, 5111, 1113),
}
SANITY_SIMULATION_DRAWS = 100_000
PROBABILITY_TOLERANCE = 0.005
TRAINING_METRICS_NOTE = (
    "Training metrics are computed on sampled draws, not on the original "
    "uniformly weighted training distribution. Formal model comparison uses "
    "unweighted validation metrics."
)
TEMPORAL_RISK_NOTE = (
    "DeepSolarEye frames are highly temporally correlated; rare high-L states "
    "are concentrated in a few continuous time blocks. Fold 3 has only 252 "
    "training records in the highest L bin. Sqrt sampling still repeats nearby "
    "frames, and random image augmentation does not turn them into independent "
    "scenes. If high-L training fit improves without held-out high-L validation "
    "improvement, oversampling strength must not simply be increased."
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
        "sampling",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"L-balanced config is missing fields: {missing}")
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
        "sanity_simulation_seed": SEED,
        "sanity_simulation_draws": SANITY_SIMULATION_DRAWS,
    }
    if config["training_version"] != TRAINING_VERSION:
        raise ValueError("Unexpected L-balanced training_version")
    if config["sampling"] != expected_sampling:
        raise ValueError("L-balanced sampling configuration is not frozen")
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = load_training_config()
    parser = argparse.ArgumentParser(
        description="Train formal date_grouped_v1 Image-only sqrt-L-balanced model"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--fold", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(
        seed=defaults["seed"],
        epochs=defaults["epochs"],
        batch_size=defaults["batch_size"],
        num_workers=defaults["num_workers"],
        learning_rate=defaults["learning_rate"],
        weight_decay=defaults["weight_decay"],
        patience=defaults["patience"],
        amp=defaults["amp"],
        pretrained=defaults["pretrained"],
        overwrite=False,
        pilot_run=False,
    )
    return parser.parse_args(argv)


def assign_l_bin_ids(labels: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Training labels must be a finite one-dimensional array")
    # side='right' puts exact 0.1/0.3/0.5/0.7 values in the bin on the right.
    return np.searchsorted(FINITE_BIN_EDGES, values, side="right").astype(np.int64)


def labels_from_training_records(train_records) -> np.ndarray:
    if train_records.empty:
        raise ValueError("Training records are empty")
    if set(train_records["top_level_role"]) != {"model_development"}:
        raise ValueError("Sampler labels may only come from model_development training records")
    return np.asarray(
        [parse_filename(name)[1] for name in train_records["filename"]],
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
    # The sole experimental change: exactly train_bin_counts[ids].double().pow(-0.5).
    return count_tensor[id_tensor].double().pow(-0.5)


def theoretical_sampling_probabilities(counts: np.ndarray) -> np.ndarray:
    roots = np.sqrt(np.asarray(counts, dtype=np.float64))
    return roots / roots.sum()


def total_variation_from_uniform(probabilities: np.ndarray) -> float:
    uniform = np.full(len(BIN_LABELS), 1.0 / len(BIN_LABELS))
    return float(0.5 * np.abs(np.asarray(probabilities) - uniform).sum())


def run_sampler_sanity_simulation(
    sample_weights: torch.Tensor,
    bin_ids: np.ndarray,
    theory: np.ndarray,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=SANITY_SIMULATION_DRAWS,
        replacement=True,
        generator=generator,
    )
    sampled_indices = torch.as_tensor(list(sampler), dtype=torch.long)
    sampled_bins = torch.as_tensor(bin_ids, dtype=torch.long)[sampled_indices]
    simulated_counts = torch.bincount(
        sampled_bins, minlength=len(BIN_LABELS)
    ).numpy()
    simulated_probabilities = simulated_counts / SANITY_SIMULATION_DRAWS
    max_error = float(np.max(np.abs(simulated_probabilities - theory)))
    if max_error >= PROBABILITY_TOLERANCE:
        raise RuntimeError(
            f"Sampler simulation differs from theory by {max_error:.8f}"
        )
    return {
        "seed": SEED,
        "draws": SANITY_SIMULATION_DRAWS,
        "simulated_bin_counts": dict(zip(BIN_LABELS, simulated_counts.tolist())),
        "simulated_bin_probabilities": dict(
            zip(BIN_LABELS, simulated_probabilities.tolist())
        ),
        "maximum_absolute_probability_error": max_error,
        "tolerance": PROBABILITY_TOLERANCE,
        "passed": True,
    }


def build_sampling_provenance(train_records, fold: int) -> dict[str, Any]:
    if fold not in FROZEN_TRAIN_BIN_COUNTS:
        raise ValueError("fold must be 1, 2, 3, or 4")
    labels = labels_from_training_records(train_records)
    bin_ids = assign_l_bin_ids(labels)
    counts = compute_train_bin_counts(bin_ids)
    expected_counts = np.asarray(FROZEN_TRAIN_BIN_COUNTS[fold], dtype=np.int64)
    expected_train_n = EXPECTED_FOLD_COUNTS[fold][0]
    if len(labels) != expected_train_n or int(counts.sum()) != expected_train_n:
        raise RuntimeError(
            f"Fold {fold} training count mismatch: labels={len(labels)}, bins={counts.sum()}"
        )
    if not np.array_equal(counts, expected_counts):
        raise RuntimeError(
            f"Fold {fold} frozen L-bin mismatch: expected "
            f"{expected_counts.tolist()}, got {counts.tolist()}"
        )

    sample_weights = compute_sample_weights(bin_ids, counts)
    theory = theoretical_sampling_probabilities(counts)
    baseline_probabilities = counts / counts.sum()
    baseline_tv = total_variation_from_uniform(baseline_probabilities)
    sampled_tv = total_variation_from_uniform(theory)
    if not sampled_tv < baseline_tv:
        raise RuntimeError("Sqrt sampling did not make the training distribution more balanced")
    if np.allclose(theory, np.full(5, 0.2), rtol=0.0, atol=1e-12):
        raise RuntimeError("Sqrt sampling unexpectedly became forced five-bin balance")

    simulation = run_sampler_sanity_simulation(sample_weights, bin_ids, theory)
    expected_draws = theory * len(labels)
    provenance = {
        "fold": fold,
        "train_dates": sorted(train_records["date"].unique().tolist()),
        "bin_edges": ["-inf", 0.1, 0.3, 0.5, 0.7, "+inf"],
        "bin_interval_rule": "left-closed, right-open (right=False)",
        "train_bin_counts": dict(zip(BIN_LABELS, counts.tolist())),
        "train_bin_proportions": dict(
            zip(BIN_LABELS, baseline_probabilities.tolist())
        ),
        "weight_formula": "n_bin^-0.5",
        "alpha": ALPHA,
        "weights_normalized": False,
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
        "expected_sampling_probability_per_bin": dict(zip(BIN_LABELS, theory.tolist())),
        "expected_draws_per_bin": dict(zip(BIN_LABELS, expected_draws.tolist())),
        "baseline_total_variation_from_uniform": baseline_tv,
        "sampled_total_variation_from_uniform": sampled_tv,
        "forced_twenty_percent_per_bin": False,
        "num_samples": len(labels),
        "replacement": True,
        "sampler_seed": SEED,
        "worker_generator_seed": SEED,
        "sanity_simulation_seed": SEED,
        "sanity_simulation_draws": SANITY_SIMULATION_DRAWS,
        "sanity_simulation": simulation,
        "training_metrics_note": TRAINING_METRICS_NOTE,
        "temporal_correlation_risk": TEMPORAL_RISK_NOTE,
    }
    if fold == 3:
        provenance["fold3_high_l_audit"] = {
            "original_count": 252,
            "train_count": 19216,
            "original_proportion": 252 / 19216,
            "sqrt_sampling_probability": float(theory[4]),
            "expected_draws_per_epoch": float(expected_draws[4]),
            "risk": "Repeated rare temporally adjacent high-L frames.",
        }
    return {"provenance": provenance, "bin_ids": bin_ids, "weights": sample_weights}


class LBalancedDataLoaderFactory:
    """Replace only the first (training) baseline DataLoader construction."""

    def __init__(self, fold: int, train_records, sampling_plan=None):
        self.fold = fold
        self.train_records = train_records
        self.plan = sampling_plan or build_sampling_provenance(train_records, fold)
        self.calls = 0
        self.train_sampler: WeightedRandomSampler | None = None
        self.sampler_generator: torch.Generator | None = None
        self.worker_generator: torch.Generator | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        return self.plan["provenance"]

    def __call__(self, dataset, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            if kwargs.get("shuffle") is not True or "sampler" in kwargs:
                raise RuntimeError("Frozen baseline training loader signature changed")
            if len(dataset) != len(self.train_records):
                raise RuntimeError("Training dataset/record count mismatch")
            expected_files = self.train_records["filename"].tolist()
            if not hasattr(dataset, "files") or list(dataset.files) != expected_files:
                raise RuntimeError(
                    "Training dataset order differs from the records used to build weights"
                )
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
                raise RuntimeError("Frozen baseline validation loader signature changed")
            return DataLoader(dataset, *args, **kwargs)
        raise RuntimeError("Unexpected additional DataLoader construction")


def _metadata_with_sampling(path: Path, value: dict[str, Any], provenance) -> dict[str, Any]:
    result = dict(value)
    if Path(path).name in {
        "training_config_snapshot.json",
        "final_metrics.json",
        "run_metadata.json",
    }:
        result["sampling_provenance"] = provenance
        result["experiment_note"] = TEMPORAL_RISK_NOTE
    return result


@contextmanager
def lbalanced_runtime(factory: LBalancedDataLoaderFactory):
    original_loader = baseline.DataLoader
    original_config_loader = baseline.load_training_config
    original_checkpoint_builder = baseline.build_checkpoint_payload
    original_json_writer = baseline._write_json

    def checkpoint_builder(*args, **kwargs):
        payload = original_checkpoint_builder(*args, **kwargs)
        payload["sampling_provenance"] = factory.provenance
        return payload

    def json_writer(path, value):
        original_json_writer(path, _metadata_with_sampling(path, value, factory.provenance))

    baseline.DataLoader = factory
    baseline.load_training_config = load_training_config
    baseline.build_checkpoint_payload = checkpoint_builder
    baseline._write_json = json_writer
    try:
        yield
    finally:
        baseline.DataLoader = original_loader
        baseline.load_training_config = original_config_loader
        baseline.build_checkpoint_payload = original_checkpoint_builder
        baseline._write_json = original_json_writer


def validate_single_factor_contract() -> None:
    config = load_training_config()
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
        name: (baseline_config.get(name), config.get(name))
        for name in frozen_fields
        if baseline_config.get(name) != config.get(name)
    }
    if mismatches:
        raise RuntimeError(f"L-balanced config differs from baseline: {mismatches}")
    if PREPROCESSING_DESCRIPTION != baseline.PREPROCESSING_DESCRIPTION:
        raise RuntimeError("Transforms or normalization changed from baseline")
    if SolarResNet50ImageOnly is not baseline.SolarResNet50ImageOnly:
        raise RuntimeError("Model class changed from baseline")
    source = inspect.getsource(sys.modules[__name__])
    forbidden_checkpoint_tokens = ("torch" + ".load", "load_state_dict(" + "torch" + ".load")
    if any(token in source for token in forbidden_checkpoint_tokens):
        raise RuntimeError("L-balanced training can read an old checkpoint")


def preflight_fold(manifest_path: Path, image_root: Path, fold: int) -> dict[str, Any]:
    audit = baseline.preflight_fold(manifest_path, image_root, fold)
    plan = build_sampling_provenance(audit["train_records"], fold)
    plan["provenance"]["manifest_sha256"] = audit["manifest_sha256"]
    plan["provenance"]["dataset_fingerprint"] = audit["fingerprint"]
    audit["sampling_plan"] = plan
    audit["sampling_provenance"] = plan["provenance"]
    return audit


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    validate_single_factor_contract()
    baseline.validate_training_arguments(args)
    audit = preflight_fold(args.manifest, args.image_root, args.fold)
    factory = LBalancedDataLoaderFactory(
        args.fold, audit["train_records"], sampling_plan=audit["sampling_plan"]
    )
    with lbalanced_runtime(factory):
        result = baseline.run_training(args)
    if factory.calls != 2:
        raise RuntimeError(f"Expected two DataLoaders, observed {factory.calls}")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
