"""Data access helpers for leakage-free date-grouped model training."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from utils.dataset import SolarDataset


REQUIRED_MANIFEST_COLUMNS = (
    "filename",
    "timestamp",
    "date",
    "top_level_role",
    "cv_validation_fold",
)
MODEL_DEVELOPMENT_ROLE = "model_development"
FORBIDDEN_MODEL_ROLES = (
    "cp_calibration",
    "decision_development",
    "final_test",
)


@lru_cache(maxsize=8)
def _available_filenames(image_root_string: str) -> frozenset[str]:
    """Index an immutable image directory once per process."""

    image_root = Path(image_root_string)
    return frozenset(path.name for path in image_root.iterdir() if path.is_file())


def _read_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing = [
        column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns
    ]
    if missing:
        raise ValueError(f"Split manifest is missing columns: {', '.join(missing)}")
    if manifest.empty:
        raise ValueError("Split manifest is empty")
    if manifest["filename"].isna().any() or manifest["filename"].duplicated().any():
        raise ValueError("Split manifest filenames must be non-null and unique")
    if manifest["date"].isna().any() or manifest["top_level_role"].isna().any():
        raise ValueError("Split manifest date and top_level_role must be non-null")

    model_mask = manifest["top_level_role"].eq(MODEL_DEVELOPMENT_ROLE)
    model_folds = pd.to_numeric(
        manifest.loc[model_mask, "cv_validation_fold"], errors="coerce"
    )
    if model_folds.isna().any():
        raise ValueError(
            "Every model_development record must have a cv_validation_fold"
        )
    if not model_folds.map(float.is_integer).all():
        raise ValueError("cv_validation_fold values must be integers")

    manifest = manifest.loc[:, REQUIRED_MANIFEST_COLUMNS].copy()
    manifest.loc[model_mask, "cv_validation_fold"] = model_folds.astype(int)
    return manifest


def _attach_and_validate_paths(
    records: pd.DataFrame,
    image_root: Path,
) -> pd.DataFrame:
    image_root = Path(image_root).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    available_filenames = _available_filenames(str(image_root))
    requested_filenames = records["filename"].tolist()
    missing = sorted(set(requested_filenames) - available_filenames)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} manifest images are missing under {image_root}. "
            f"First missing files: {preview}"
        )

    result = records.copy()
    result["image_path"] = [
        str(image_root / filename) for filename in requested_filenames
    ]
    return result.reset_index(drop=True)


def validate_fold_isolation(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Raise if a fold violates filename, date, or top-level-role isolation."""

    required = {"filename", "date", "top_level_role"}
    for name, frame in (("train", train_df), ("validation", val_df)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} records are missing columns: {', '.join(missing)}")
        if frame.empty:
            raise ValueError(f"{name} records are empty")
        roles = set(frame["top_level_role"])
        if roles != {MODEL_DEVELOPMENT_ROLE}:
            raise ValueError(
                f"{name} contains non-model-development roles: {sorted(roles)}"
            )

    train_filenames = set(train_df["filename"])
    val_filenames = set(val_df["filename"])
    filename_overlap = train_filenames & val_filenames
    if filename_overlap:
        raise ValueError(
            f"Train and validation overlap by {len(filename_overlap)} filenames"
        )

    train_dates = set(train_df["date"])
    val_dates = set(val_df["date"])
    date_overlap = train_dates & val_dates
    if date_overlap:
        raise ValueError(
            f"Train and validation overlap by date: {sorted(date_overlap)}"
        )


def load_fold_records(
    manifest_path: Path,
    image_root: Path,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load deterministic train/validation records for one date-grouped fold.

    Selection uses only ``top_level_role`` and ``cv_validation_fold`` from the
    frozen manifest. Label and irradiance values are not read or consulted.
    """

    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, 5):
        raise ValueError("fold must be an integer from 1 to 4")

    manifest = _read_manifest(Path(manifest_path))
    model_records = manifest[
        manifest["top_level_role"].eq(MODEL_DEVELOPMENT_ROLE)
    ].copy()
    fold_values = pd.to_numeric(
        model_records["cv_validation_fold"], errors="raise"
    ).astype(int)
    available_folds = sorted(fold_values.unique().tolist())
    if fold not in available_folds:
        raise ValueError(
            f"Fold {fold} is absent from the manifest; available folds={available_folds}"
        )

    validation_records = model_records[fold_values.eq(fold)].copy()
    train_records = model_records[fold_values.ne(fold)].copy()

    # Preserve frozen manifest order; do not rank or filter on metadata values.
    train_records = _attach_and_validate_paths(train_records, Path(image_root))
    validation_records = _attach_and_validate_paths(
        validation_records, Path(image_root)
    )
    validate_fold_isolation(train_records, validation_records)
    return train_records, validation_records


def build_fold_datasets(
    manifest_path: Path,
    image_root: Path,
    fold: int,
    train_transform: Any = None,
    validation_transform: Any = None,
) -> tuple[SolarDataset, SolarDataset, pd.DataFrame, pd.DataFrame]:
    """Build existing ``SolarDataset`` instances from frozen fold records."""

    train_records, validation_records = load_fold_records(
        manifest_path=Path(manifest_path),
        image_root=Path(image_root),
        fold=fold,
    )

    train_dataset = SolarDataset(str(Path(image_root)), transform=train_transform)
    train_dataset.files = train_records["filename"].tolist()

    validation_dataset = SolarDataset(
        str(Path(image_root)), transform=validation_transform
    )
    validation_dataset.files = validation_records["filename"].tolist()

    return train_dataset, validation_dataset, train_records, validation_records
