"""Reconstruct leakage-free Image-only OOF predictions from frozen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_resnet50_image_only_date_grouped as training
from models.resnet50_image_only import SolarResNet50ImageOnly
from utils.dataset import SolarDataset
from utils.parser import parse_filename


ANALYSIS_VERSION = "resnet50_image_only_oof_diagnostics_v1"
MODEL_NAME = "SolarResNet50ImageOnly"
SEED = 42
EXPECTED_MANIFEST_SHA256 = (
    "a354afc2b691719bf0cc3c3982033da833795006e3e3b0122cae07810bd83e02"
)
FORMAL_CV_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "resnet50_image_only"
    / "formal_cv"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "date_grouped_v1"
    / "resnet50_image_only"
    / "diagnostics"
)
EXPECTED_TOTAL_OOF_SAMPLES = 25716
FORBIDDEN_ROLES = ("cp_calibration", "decision_development", "final_test")
DEVELOPMENT_DATES = (
    "2017-06-13",
    "2017-06-14",
    "2017-06-16",
    "2017-06-20",
    "2017-06-21",
    "2017-06-26",
    "2017-06-28",
    "2017-06-29",
)
FOLD_SPECS = {
    1: {
        "validation_dates": ("2017-06-13", "2017-06-28"),
        "validation_count": 5911,
    },
    2: {
        "validation_dates": ("2017-06-14", "2017-06-29"),
        "validation_count": 7192,
    },
    3: {
        "validation_dates": ("2017-06-16", "2017-06-26"),
        "validation_count": 6500,
    },
    4: {
        "validation_dates": ("2017-06-20", "2017-06-21"),
        "validation_count": 6113,
    },
}
EXPECTED_FORMAL_METRICS = {
    1: {
        "mae": 0.040377808234005975,
        "rmse": 0.07055234919884372,
        "r2": 0.9312461405353576,
    },
    2: {
        "mae": 0.052554944578117165,
        "rmse": 0.09850287546629752,
        "r2": 0.8254019662103234,
    },
    3: {
        "mae": 0.17508760724733907,
        "rmse": 0.23575375344444752,
        "r2": 0.47387488068627803,
    },
    4: {
        "mae": 0.09370982157972256,
        "rmse": 0.14576553463703384,
        "r2": 0.3842772273650996,
    },
}
METRIC_ABSOLUTE_TOLERANCE = {
    "mae": 2e-5,
    "rmse": 2e-5,
    "r2": 2e-4,
}
L_BIN_EDGES = (-np.inf, 0.1, 0.3, 0.5, 0.7, np.inf)
L_BIN_LABELS = (
    "(-inf,0.1)",
    "[0.1,0.3)",
    "[0.3,0.5)",
    "[0.5,0.7)",
    "[0.7,+inf)",
)
OOF_COLUMNS = (
    "filename",
    "timestamp",
    "date",
    "fold",
    "L_true",
    "L_pred",
    "error",
    "abs_error",
    "squared_error",
    "irradiance",
)
GENERATED_RELATIVE_FILES = (
    "oof_predictions.csv",
    "per_date_metrics.csv",
    "per_l_bin_metrics.csv",
    "fold_support_diagnostics.csv",
    "diagnostics_summary.json",
    "figures/01_true_vs_pred_by_date.png",
    "figures/02_metrics_by_date.png",
    "figures/03_weak_fold_l_ecdf.png",
    "figures/04_date_l_bin_error_heatmaps.png",
)


class DiagnosticValidationDataset(Dataset):
    """Add frozen routing metadata to the existing SolarDataset output."""

    def __init__(self, records: pd.DataFrame, transform):
        records = records.reset_index(drop=True).copy()
        required = {"filename", "timestamp", "date", "top_level_role"}
        missing = sorted(required - set(records.columns))
        if missing:
            raise ValueError(f"Diagnostic records are missing columns: {missing}")
        if records.empty:
            raise ValueError("Diagnostic records must not be empty")
        roles = set(records["top_level_role"])
        if roles != {"model_development"}:
            raise ValueError(
                f"Only model_development records may be inferred, got {sorted(roles)}"
            )
        if set(records["date"]) - set(DEVELOPMENT_DATES):
            raise ValueError("Diagnostic records contain a non-development date")

        self.records = records
        self.base = SolarDataset(str(training.DEFAULT_IMAGE_ROOT), transform=transform)
        self.base.files = records["filename"].tolist()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, label, irradiance, _ = self.base[index]
        row = self.records.iloc[index]
        return {
            "image": image,
            "label": label,
            "irradiance": irradiance,
            "filename": row["filename"],
            "timestamp": row["timestamp"],
            "date": row["date"],
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct formal Image-only OOF predictions and diagnostics"
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def validate_output_root(output_root: Path = OUTPUT_ROOT) -> Path:
    expected = OUTPUT_ROOT.resolve()
    actual = Path(output_root).resolve()
    forbidden_parts = {
        "formal_cv",
        "resnet50_with_i",
        "irradiance_only",
        "cp_calibration",
        "decision_development",
        "final_test",
    }
    if actual != expected or forbidden_parts.intersection(
        part.lower() for part in actual.parts
    ):
        raise RuntimeError(
            f"OOF diagnostics output root is unsafe: expected {expected}, got {actual}"
        )
    return actual


def expected_checkpoint_path(fold: int) -> Path:
    if fold not in FOLD_SPECS:
        raise ValueError("fold must be 1, 2, 3, or 4")
    return FORMAL_CV_ROOT / f"fold_{fold}_seed_{SEED}" / "best_model.pth"


def load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    fold: int,
    require_formal_path: bool = True,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name != "best_model.pth":
        raise ValueError("Only best_model.pth is permitted for OOF reconstruction")
    if "final_model.pth" in checkpoint_path.as_posix():
        raise ValueError("final_model.pth is forbidden for OOF reconstruction")
    if require_formal_path and checkpoint_path.resolve() != expected_checkpoint_path(
        fold
    ).resolve():
        raise ValueError(
            f"Fold {fold} checkpoint path mismatch: {checkpoint_path}"
        )

    expected = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "fold": fold,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "checkpoint_kind": "best",
        "pilot_run": False,
        "NOT_FOR_RESEARCH_METRICS": False,
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fold {fold} checkpoint metadata mismatch: {mismatches}")
    if checkpoint.get("validation_dates") != list(
        FOLD_SPECS[fold]["validation_dates"]
    ):
        raise ValueError(f"Fold {fold} checkpoint validation dates mismatch")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Fold {fold} checkpoint has no model_state_dict")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_preflight(fold: int) -> dict[str, Any]:
    path = expected_checkpoint_path(fold)
    if not path.is_file():
        raise FileNotFoundError(f"Missing formal best checkpoint: {path}")
    checkpoint = load_trusted_checkpoint(path)
    validate_checkpoint_metadata(checkpoint, path, fold)
    result = {
        "fold": fold,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "validation_dates": list(checkpoint["validation_dates"]),
        "manifest_sha256": checkpoint["manifest_sha256"],
        "model_name": checkpoint["model_name"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
    }
    del checkpoint
    return result


def collect_fold_records(fold: int) -> dict[str, Any]:
    audit = training.preflight_fold(
        training.DEFAULT_MANIFEST,
        training.DEFAULT_IMAGE_ROOT,
        fold,
    )
    if audit["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Fold {fold} manifest SHA256 mismatch")
    expected_count = FOLD_SPECS[fold]["validation_count"]
    if len(audit["validation_records"]) != expected_count:
        raise ValueError(f"Fold {fold} validation count mismatch")
    if tuple(audit["validation_dates"]) != FOLD_SPECS[fold]["validation_dates"]:
        raise ValueError(f"Fold {fold} validation dates mismatch")
    if audit["forbidden_role_counts"] != {role: 0 for role in FORBIDDEN_ROLES}:
        raise ValueError(f"Fold {fold} contains a protected role")
    return audit


def set_deterministic_inference(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_validation_transform():
    _, validation_transform = training.build_transforms()
    return validation_transform


def predict_images(model, images: torch.Tensor) -> torch.Tensor:
    return model(images)


def run_fold_inference(
    fold: int,
    audit: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    checkpoint_path = expected_checkpoint_path(fold)
    checkpoint = load_trusted_checkpoint(checkpoint_path)
    validate_checkpoint_metadata(checkpoint, checkpoint_path, fold)

    model = SolarResNet50ImageOnly(
        dropout=0.3,
        use_pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint

    dataset = DiagnosticValidationDataset(
        audit["validation_records"],
        build_validation_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    rows: list[dict[str, Any]] = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                predictions = predict_images(model, images)
            predictions_np = predictions.detach().float().cpu().numpy().reshape(-1)
            labels_np = batch["label"].detach().float().cpu().numpy().reshape(-1)
            irradiance_np = (
                batch["irradiance"].detach().float().cpu().numpy().reshape(-1)
            )
            if len(predictions_np) != len(labels_np):
                raise RuntimeError("Prediction/label batch size mismatch")
            for index in range(len(labels_np)):
                label = float(labels_np[index])
                prediction = float(predictions_np[index])
                error = prediction - label
                rows.append(
                    {
                        "filename": batch["filename"][index],
                        "timestamp": batch["timestamp"][index],
                        "date": batch["date"][index],
                        "fold": fold,
                        "L_true": label,
                        "L_pred": prediction,
                        "error": error,
                        "abs_error": abs(error),
                        "squared_error": error**2,
                        "irradiance": float(irradiance_np[index]),
                    }
                )
    if len(rows) != FOLD_SPECS[fold]["validation_count"]:
        raise RuntimeError(f"Fold {fold} inference row count mismatch")
    return pd.DataFrame(rows, columns=OOF_COLUMNS)


def compute_regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    true = frame["L_true"].to_numpy(dtype=np.float32).astype(np.float64)
    pred = frame["L_pred"].to_numpy(dtype=np.float32).astype(np.float64)
    errors = pred - true
    sse = float(np.dot(errors, errors))
    count = len(true)
    if count == 0:
        raise ValueError("Cannot compute metrics for an empty frame")
    sst = float(np.dot(true, true) - true.sum() ** 2 / count)
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(sse / count)),
        "r2": float(1.0 - sse / sst) if sst > 0 else 0.0,
        "sse": sse,
        "sst": sst,
    }


def validate_reconstructed_metrics(
    oof: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    comparisons: dict[int, dict[str, Any]] = {}
    failures = []
    for fold in FOLD_SPECS:
        actual = compute_regression_metrics(oof[oof["fold"].eq(fold)])
        expected = EXPECTED_FORMAL_METRICS[fold]
        differences = {
            name: actual[name] - expected[name] for name in ("mae", "rmse", "r2")
        }
        passed = all(
            abs(differences[name]) <= METRIC_ABSOLUTE_TOLERANCE[name]
            for name in differences
        )
        comparisons[fold] = {
            "expected": expected,
            "reconstructed": {
                name: actual[name] for name in ("mae", "rmse", "r2")
            },
            "difference": differences,
            "absolute_tolerance": METRIC_ABSOLUTE_TOLERANCE,
            "passed": passed,
        }
        if not passed:
            failures.append(fold)
    if failures:
        raise RuntimeError(
            "Reconstructed formal metrics mismatch for folds "
            f"{failures}; diagnostics generation stopped"
        )
    return comparisons


def validate_oof_integrity(oof: pd.DataFrame) -> None:
    if list(oof.columns) != list(OOF_COLUMNS):
        raise ValueError("OOF columns do not match the frozen schema")
    if len(oof) != EXPECTED_TOTAL_OOF_SAMPLES:
        raise ValueError(f"OOF row count mismatch: {len(oof)}")
    if oof["filename"].duplicated().any():
        raise ValueError("OOF filenames must be unique")
    if tuple(sorted(oof["date"].unique().tolist())) != DEVELOPMENT_DATES:
        raise ValueError("OOF development dates are incomplete or unexpected")
    for fold, spec in FOLD_SPECS.items():
        fold_frame = oof[oof["fold"].eq(fold)]
        if len(fold_frame) != spec["validation_count"]:
            raise ValueError(f"Fold {fold} OOF count mismatch")
        if tuple(sorted(fold_frame["date"].unique())) != spec["validation_dates"]:
            raise ValueError(f"Fold {fold} OOF dates mismatch")
    errors = oof["L_pred"].to_numpy() - oof["L_true"].to_numpy()
    if not np.allclose(errors, oof["error"].to_numpy(), rtol=0.0, atol=1e-12):
        raise ValueError("OOF error definition mismatch")
    if not np.allclose(
        np.abs(errors), oof["abs_error"].to_numpy(), rtol=0.0, atol=1e-12
    ):
        raise ValueError("OOF absolute error definition mismatch")
    if not np.allclose(
        errors**2, oof["squared_error"].to_numpy(), rtol=0.0, atol=1e-12
    ):
        raise ValueError("OOF squared error definition mismatch")


def assign_l_bins(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        values,
        bins=L_BIN_EDGES,
        labels=L_BIN_LABELS,
        right=False,
        include_lowest=True,
        ordered=True,
    )


def safe_std_ratio(predicted: pd.Series, true: pd.Series) -> float:
    true_std = float(true.std(ddof=1))
    if not math.isfinite(true_std) or true_std == 0.0:
        return float("nan")
    return float(predicted.std(ddof=1) / true_std)


def safe_pearson(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right, method="pearson"))


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    return safe_pearson(left.rank(method="average"), right.rank(method="average"))


def prediction_slope(frame: pd.DataFrame) -> float:
    true = frame["L_true"].to_numpy(dtype=np.float64)
    pred = frame["L_pred"].to_numpy(dtype=np.float64)
    centered = true - true.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(centered, pred - pred.mean()) / denominator)


def build_per_date_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    fold_sse = oof.groupby("fold", observed=True)["squared_error"].sum().to_dict()
    fold_n = oof.groupby("fold", observed=True).size().to_dict()
    rows = []
    for (date, fold), frame in oof.groupby(["date", "fold"], sort=True):
        metrics = compute_regression_metrics(frame)
        rows.append(
            {
                "date": date,
                "fold": int(fold),
                "N": len(frame),
                "sample_share_within_fold": len(frame) / fold_n[fold],
                "L_mean": frame["L_true"].mean(),
                "L_std": frame["L_true"].std(ddof=1),
                "L_min": frame["L_true"].min(),
                "L_max": frame["L_true"].max(),
                "I_mean": frame["irradiance"].mean(),
                "I_std": frame["irradiance"].std(ddof=1),
                "I_min": frame["irradiance"].min(),
                "I_max": frame["irradiance"].max(),
                "pred_mean": frame["L_pred"].mean(),
                "pred_std": frame["L_pred"].std(ddof=1),
                "pred_min": frame["L_pred"].min(),
                "pred_max": frame["L_pred"].max(),
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
                "bias": frame["error"].mean(),
                "SSE": metrics["sse"],
                "SST": metrics["sst"],
                "prediction_std_ratio": safe_std_ratio(
                    frame["L_pred"], frame["L_true"]
                ),
                "pred_vs_true_slope": prediction_slope(frame),
                "pearson_r": safe_pearson(frame["L_true"], frame["L_pred"]),
                "spearman_r": safe_spearman(frame["L_true"], frame["L_pred"]),
                "irradiance_error_pearson_r": safe_pearson(
                    frame["irradiance"], frame["error"]
                ),
                "irradiance_abs_error_pearson_r": safe_pearson(
                    frame["irradiance"], frame["abs_error"]
                ),
                "irradiance_error_spearman_r": safe_spearman(
                    frame["irradiance"], frame["error"]
                ),
                "irradiance_abs_error_spearman_r": safe_spearman(
                    frame["irradiance"], frame["abs_error"]
                ),
                "SSE_share_within_fold": metrics["sse"] / fold_sse[fold],
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_per_l_bin_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    work = oof.copy()
    work["L_bin"] = assign_l_bins(work["L_true"])
    if work["L_bin"].isna().any():
        raise RuntimeError("At least one OOF sample was not assigned to an L bin")
    rows = []
    for (date, fold), date_frame in work.groupby(["date", "fold"], sort=True):
        date_sse = float(date_frame["squared_error"].sum())
        for label in L_BIN_LABELS:
            frame = date_frame[date_frame["L_bin"].eq(label)]
            count = len(frame)
            sse = float(frame["squared_error"].sum()) if count else 0.0
            rows.append(
                {
                    "date": date,
                    "fold": int(fold),
                    "L_bin": label,
                    "N": count,
                    "MAE": float(frame["abs_error"].mean()) if count else np.nan,
                    "RMSE": float(np.sqrt(sse / count)) if count else np.nan,
                    "bias": float(frame["error"].mean()) if count else np.nan,
                    "SSE": sse,
                    "SSE_share": sse / date_sse if date_sse > 0 else np.nan,
                    "sparse_bin": count < 30,
                }
            )
    return pd.DataFrame(rows)


def parse_labels(filenames: Sequence[str]) -> np.ndarray:
    return np.asarray([parse_filename(filename)[1] for filename in filenames])


def build_fold_support_diagnostics(
    audits: dict[int, dict[str, Any]],
    oof: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for fold, audit in audits.items():
        train_labels = parse_labels(audit["train_records"]["filename"].tolist())
        quantiles = np.quantile(
            train_labels,
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
        )
        support = {
            "train_min": float(train_labels.min()),
            "train_max": float(train_labels.max()),
            "train_q01": float(quantiles[0]),
            "train_q05": float(quantiles[1]),
            "train_q25": float(quantiles[2]),
            "train_q50": float(quantiles[3]),
            "train_q75": float(quantiles[4]),
            "train_q95": float(quantiles[5]),
            "train_q99": float(quantiles[6]),
        }
        train_bins = assign_l_bins(train_labels)
        for date in FOLD_SPECS[fold]["validation_dates"]:
            validation = oof[oof["date"].eq(date)]
            values = validation["L_true"].to_numpy()
            validation_bins = assign_l_bins(values)
            base = {
                "fold": fold,
                "validation_date": date,
                "train_N": len(train_labels),
                "validation_N": len(values),
                **support,
                "validation_L_mean": float(values.mean()),
                "validation_L_std": float(values.std(ddof=1)),
                "below_train_min_ratio": float(np.mean(values < support["train_min"])),
                "above_train_max_ratio": float(np.mean(values > support["train_max"])),
                "outside_train_q01_q99_ratio": float(
                    np.mean(
                        (values < support["train_q01"])
                        | (values > support["train_q99"])
                    )
                ),
            }
            for label in L_BIN_LABELS:
                rows.append(
                    {
                        **base,
                        "L_bin": label,
                        "train_bin_N": int(np.sum(train_bins == label)),
                        "train_bin_proportion": float(np.mean(train_bins == label)),
                        "validation_bin_N": int(np.sum(validation_bins == label)),
                        "validation_bin_proportion": float(
                            np.mean(validation_bins == label)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _global_axis_limits(oof: pd.DataFrame) -> tuple[float, float]:
    minimum = float(min(oof["L_true"].min(), oof["L_pred"].min()))
    maximum = float(max(oof["L_true"].max(), oof["L_pred"].max()))
    padding = max((maximum - minimum) * 0.03, 0.01)
    return minimum - padding, maximum + padding


def plot_true_vs_pred_by_date(oof: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True)
    lower, upper = _global_axis_limits(oof)
    for axis, date in zip(axes.flat, DEVELOPMENT_DATES):
        frame = oof[oof["date"].eq(date)]
        axis.scatter(
            frame["L_true"],
            frame["L_pred"],
            s=5,
            alpha=0.16,
            linewidths=0,
            rasterized=True,
        )
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        axis.set_title(f"{date} | Fold {int(frame['fold'].iloc[0])}")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("True L")
    for axis in axes[:, 0]:
        axis.set_ylabel("Predicted L")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_metrics_by_date(per_date: pd.DataFrame, path: Path) -> None:
    frame = per_date.sort_values("date")
    positions = np.arange(len(frame))
    colors = [f"C{int(fold) - 1}" for fold in frame["fold"]]
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, column, label in zip(
        axes,
        ("R2", "RMSE", "bias"),
        ("R²", "RMSE", "Bias (prediction - truth)"),
    ):
        axis.bar(positions, frame[column], color=colors, alpha=0.85)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(positions, frame["date"], rotation=35, ha="right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values))
    probabilities = np.arange(1, len(ordered) + 1) / len(ordered)
    return ordered, probabilities


def plot_weak_fold_l_ecdf(
    audits: dict[int, dict[str, Any]],
    oof: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for axis, fold in zip(axes, (3, 4)):
        train = parse_labels(audits[fold]["train_records"]["filename"].tolist())
        x_values, probabilities = _ecdf(train)
        axis.plot(x_values, probabilities, label="Pooled train", linewidth=2)
        for date in FOLD_SPECS[fold]["validation_dates"]:
            values = oof.loc[oof["date"].eq(date), "L_true"].to_numpy()
            x_values, probabilities = _ecdf(values)
            axis.plot(x_values, probabilities, label=date, linewidth=1.5)
        axis.set_title(f"Fold {fold}")
        axis.set_xlabel("L")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Empirical cumulative probability")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_date_l_bin_error_heatmaps(per_bin: pd.DataFrame, path: Path) -> None:
    dates = list(DEVELOPMENT_DATES)
    mae = per_bin.pivot(index="date", columns="L_bin", values="MAE").reindex(
        index=dates, columns=L_BIN_LABELS
    )
    bias = per_bin.pivot(index="date", columns="L_bin", values="bias").reindex(
        index=dates, columns=L_BIN_LABELS
    )
    counts = per_bin.pivot(index="date", columns="L_bin", values="N").reindex(
        index=dates, columns=L_BIN_LABELS
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    mae_image = axes[0].imshow(mae.to_numpy(), aspect="auto", cmap="viridis")
    bias_limit = float(np.nanmax(np.abs(bias.to_numpy())))
    bias_image = axes[1].imshow(
        bias.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-bias_limit,
        vmax=bias_limit,
    )
    for axis, title, values in (
        (axes[0], "MAE", mae),
        (axes[1], "Bias", bias),
    ):
        axis.set_title(title)
        axis.set_xticks(range(len(L_BIN_LABELS)), L_BIN_LABELS, rotation=30, ha="right")
        axis.set_yticks(range(len(dates)), dates)
        axis.set_xlabel("Fixed L bin")
        for row in range(len(dates)):
            for column in range(len(L_BIN_LABELS)):
                value = values.iloc[row, column]
                count = int(counts.iloc[row, column])
                text = "NA" if pd.isna(value) else f"{value:.3f}\nN={count}"
                axis.text(column, row, text, ha="center", va="center", fontsize=7)
    figure.colorbar(mae_image, ax=axes[0], shrink=0.8)
    figure.colorbar(bias_image, ax=axes[1], shrink=0.8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_summary(
    oof: pd.DataFrame,
    per_date: pd.DataFrame,
    per_bin: pd.DataFrame,
    comparisons: dict[int, dict[str, Any]],
    checkpoint_provenance: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    zero_std_dates = per_date.loc[
        per_date["prediction_std_ratio"].isna(), "date"
    ].tolist()
    weak_dates = per_date[per_date["fold"].isin((3, 4))].to_dict(orient="records")
    return json_ready(
        {
            "analysis_version": ANALYSIS_VERSION,
            "model_name": MODEL_NAME,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "reconstructed_from_frozen_best_checkpoints": True,
            "total_oof_samples": len(oof),
            "unique_filenames": int(oof["filename"].nunique()),
            "development_dates": list(DEVELOPMENT_DATES),
            "fold_counts": {
                str(fold): int(oof["fold"].eq(fold).sum()) for fold in FOLD_SPECS
            },
            "metrics_reconstruction_passed": True,
            "reconstructed_fold_metrics": comparisons,
            "checkpoint_provenance": checkpoint_provenance,
            "inference_device": str(device),
            "amp_enabled": device.type == "cuda",
            "deterministic_inference": {
                "seed": SEED,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "model_eval": True,
                "torch_inference_mode": True,
                "shuffle": False,
            },
            "protected_roles_accessed": [],
            "final_test_accessed": False,
            "cp_calibration_accessed": False,
            "decision_development_accessed": False,
            "fixed_l_bins": list(L_BIN_LABELS),
            "fixed_l_bin_edges": ["-inf", 0.1, 0.3, 0.5, 0.7, "+inf"],
            "error_definition": "L_pred - L_true",
            "sparse_bin_threshold": 30,
            "sparse_bin_count": int(per_bin["sparse_bin"].sum()),
            "prediction_std_ratio_undefined_dates": zero_std_dates,
            "weak_fold_date_statistics": weak_dates,
            "generated_files": list(GENERATED_RELATIVE_FILES),
            "interpretation_policy": (
                "Objective statistics only; daily R2 must be read with RMSE, "
                "L_std, SST, and bias. Irradiance associations are non-causal."
            ),
        }
    )


def generate_outputs(
    oof: pd.DataFrame,
    audits: dict[int, dict[str, Any]],
    comparisons: dict[int, dict[str, Any]],
    checkpoint_provenance: list[dict[str, Any]],
    device: torch.device,
) -> None:
    output_root = validate_output_root()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix="diagnostics_tmp_", dir=str(output_root.parent))
    )
    try:
        figures = temp_root / "figures"
        figures.mkdir()
        per_date = build_per_date_metrics(oof)
        per_bin = build_per_l_bin_metrics(oof)
        support = build_fold_support_diagnostics(audits, oof)

        write_csv(oof, temp_root / "oof_predictions.csv")
        write_csv(per_date, temp_root / "per_date_metrics.csv")
        write_csv(per_bin, temp_root / "per_l_bin_metrics.csv")
        write_csv(support, temp_root / "fold_support_diagnostics.csv")
        plot_true_vs_pred_by_date(
            oof, figures / "01_true_vs_pred_by_date.png"
        )
        plot_metrics_by_date(per_date, figures / "02_metrics_by_date.png")
        plot_weak_fold_l_ecdf(
            audits, oof, figures / "03_weak_fold_l_ecdf.png"
        )
        plot_date_l_bin_error_heatmaps(
            per_bin, figures / "04_date_l_bin_error_heatmaps.png"
        )
        summary = build_summary(
            oof,
            per_date,
            per_bin,
            comparisons,
            checkpoint_provenance,
            device,
        )
        (temp_root / "diagnostics_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        missing = [
            relative
            for relative in GENERATED_RELATIVE_FILES
            if not (temp_root / relative).is_file()
        ]
        if missing:
            raise RuntimeError(f"Diagnostics generation is incomplete: {missing}")
        temp_root.rename(output_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_output_root()
    checkpoint_provenance = [checkpoint_preflight(fold) for fold in FOLD_SPECS]
    audits = {fold: collect_fold_records(fold) for fold in FOLD_SPECS}
    print(json.dumps({"checkpoint_preflight": checkpoint_provenance}, indent=2))
    if args.preflight_only:
        print("CHECKPOINT_PREFLIGHT=PASS")
        return

    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {OUTPUT_ROOT}")
    set_deterministic_inference()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    fold_frames = []
    for fold in FOLD_SPECS:
        print(f"Fold {fold} validation inference starting", flush=True)
        frame = run_fold_inference(fold, audits[fold], device)
        fold_frames.append(frame)
        print(f"Fold {fold} validation inference rows={len(frame)}", flush=True)
    oof = pd.concat(fold_frames, ignore_index=True)
    validate_oof_integrity(oof)
    comparisons = validate_reconstructed_metrics(oof)
    print(json.dumps({"metrics_reconstruction": comparisons}, indent=2))
    print("METRICS_RECONSTRUCTION=PASS", flush=True)
    generate_outputs(oof, audits, comparisons, checkpoint_provenance, device)
    print(f"OOF diagnostics complete: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
