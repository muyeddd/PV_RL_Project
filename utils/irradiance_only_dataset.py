"""Manifest routing and scalar-only data access for irradiance ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.date_grouped_training_data import (
    MODEL_DEVELOPMENT_ROLE,
    REQUIRED_MANIFEST_COLUMNS,
    validate_fold_isolation,
)
from utils.parser import parse_filename


class IrradianceOnlyDataset(Dataset):
    """Return loss labels and raw irradiance parsed from selected filenames."""

    def __init__(self, filenames: Sequence[str]):
        if isinstance(filenames, (str, bytes)):
            raise TypeError("filenames must be a sequence, not a single string")
        self.files = list(filenames)
        if not self.files:
            raise ValueError("filenames must not be empty")
        if any(not isinstance(filename, str) or not filename for filename in self.files):
            raise ValueError("filenames must contain non-empty strings")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        _, label, irradiance = parse_filename(self.files[index])
        return (
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(irradiance, dtype=torch.float32),
        )


def load_irradiance_only_fold_records(
    manifest_path: Path,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one frozen model-development fold without opening image files."""

    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, 5):
        raise ValueError("fold must be an integer from 1 to 4")

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
    manifest = pd.read_csv(
        manifest_path,
        usecols=list(REQUIRED_MANIFEST_COLUMNS),
    )
    if manifest.empty:
        raise ValueError("Split manifest is empty")
    if manifest["filename"].isna().any() or manifest["filename"].duplicated().any():
        raise ValueError("Split manifest filenames must be non-null and unique")
    if manifest["date"].isna().any() or manifest["top_level_role"].isna().any():
        raise ValueError("Split manifest date and top_level_role must be non-null")

    model_records = manifest[
        manifest["top_level_role"].eq(MODEL_DEVELOPMENT_ROLE)
    ].copy()
    fold_values = pd.to_numeric(
        model_records["cv_validation_fold"], errors="coerce"
    )
    if fold_values.isna().any() or not fold_values.map(float.is_integer).all():
        raise ValueError(
            "Every model_development record must have an integer cv_validation_fold"
        )
    fold_values = fold_values.astype(int)
    available_folds = sorted(fold_values.unique().tolist())
    if fold not in available_folds:
        raise ValueError(
            f"Fold {fold} is absent from the manifest; available folds={available_folds}"
        )

    validation_records = model_records[fold_values.eq(fold)].copy()
    train_records = model_records[fold_values.ne(fold)].copy()
    train_records["cv_validation_fold"] = fold_values[fold_values.ne(fold)]
    validation_records["cv_validation_fold"] = fold_values[fold_values.eq(fold)]
    train_records = train_records.reset_index(drop=True)
    validation_records = validation_records.reset_index(drop=True)
    validate_fold_isolation(train_records, validation_records)
    return train_records, validation_records
