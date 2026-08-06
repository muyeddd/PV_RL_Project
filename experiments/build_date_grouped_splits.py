"""Build deterministic, date-grouped dataset split artifacts.

Allocation uses only the timestamp parsed from each filename and the explicit
date mapping in the split configuration. Values following the ``_L_`` marker
are intentionally ignored by the parser and allocation logic.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "PanelImages"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "splits" / "date_grouped_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "splits" / "date_grouped_v1"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TOP_LEVEL_ROLES = (
    "model_development",
    "cp_calibration",
    "decision_development",
    "final_test",
)
ALLOCATION_INPUT_FIELDS = ("filename", "timestamp", "date")

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

TIME_PREFIX_PATTERN = re.compile(
    r"^.+?_(?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)_"
    r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_"
    r"(?P<day>\d{1,2})_(?P<hour>\d{1,2})__(?P<minute>\d{1,2})__"
    r"(?P<second>\d{1,2})_(?P<year>\d{4})$"
)

MANIFEST_FIELDS = (
    "filename",
    "timestamp",
    "date",
    "top_level_role",
    "cv_validation_fold",
)
SUMMARY_FIELDS = (
    "scope",
    "name",
    "top_level_role",
    "cv_fold",
    "subset",
    "date_count",
    "sample_count",
    "dates",
)


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    timestamp: datetime
    date: str


@dataclass(frozen=True)
class SplitArtifacts:
    manifest_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    fingerprint: dict[str, Any]


def parse_filename_timestamp(filename: str) -> datetime:
    """Parse only the timestamp prefix of an image filename."""

    path = Path(filename)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {filename}")

    stem = path.stem
    if "_L_" not in stem:
        raise ValueError(f"Filename has no timestamp/metadata separator: {filename}")
    time_prefix = stem.split("_L_", 1)[0]
    match = TIME_PREFIX_PATTERN.fullmatch(time_prefix)
    if match is None:
        raise ValueError(f"Cannot parse timestamp from filename: {filename}")

    parts = match.groupdict()
    timestamp = datetime(
        year=int(parts["year"]),
        month=MONTHS[parts["month"]],
        day=int(parts["day"]),
        hour=int(parts["hour"]),
        minute=int(parts["minute"]),
        second=int(parts["second"]),
    )
    expected_weekday = WEEKDAYS[timestamp.weekday()]
    if parts["weekday"] != expected_weekday:
        raise ValueError(
            f"Weekday mismatch in {filename}: "
            f"found {parts['weekday']}, expected {expected_weekday}"
        )
    return timestamp


def scan_image_records(image_dir: Path) -> tuple[ImageRecord, ...]:
    """Scan all supported images and return filename-sorted records."""

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {image_dir}")

    records = []
    for path in image_paths:
        timestamp = parse_filename_timestamp(path.name)
        records.append(
            ImageRecord(
                filename=path.name,
                timestamp=timestamp,
                date=timestamp.date().isoformat(),
            )
        )
    return tuple(records)


def load_split_config(config_path: Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Split configuration must contain a JSON object")
    return config


def _validate_date_list(name: str, values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty date list")
    dates = tuple(values)
    if len(dates) != len(set(dates)):
        raise ValueError(f"{name} contains duplicate dates")
    for value in dates:
        if not isinstance(value, str):
            raise ValueError(f"{name} contains a non-string date")
        try:
            parsed = datetime.fromisoformat(value).date().isoformat()
        except ValueError as exc:
            raise ValueError(f"{name} contains an invalid ISO date: {value}") from exc
        if parsed != value:
            raise ValueError(f"{name} date is not canonical YYYY-MM-DD: {value}")
    return dates


def compile_date_mappings(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, int]]:
    """Validate the config and compile date-only allocation mappings."""

    required = {
        "schema_version",
        "split_version",
        "created_for",
        "top_level_splits",
        "cv_folds",
        "final_test_freeze",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing split config fields: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported split schema_version")
    if not isinstance(config["split_version"], str) or not config["split_version"]:
        raise ValueError("split_version must be a non-empty string")
    if not isinstance(config["created_for"], str) or not config["created_for"]:
        raise ValueError("created_for must be a non-empty string")

    top_level = config["top_level_splits"]
    if not isinstance(top_level, Mapping) or set(top_level) != set(TOP_LEVEL_ROLES):
        raise ValueError(
            "top_level_splits must contain exactly: " + ", ".join(TOP_LEVEL_ROLES)
        )

    date_to_role: dict[str, str] = {}
    role_dates: dict[str, tuple[str, ...]] = {}
    for role in TOP_LEVEL_ROLES:
        dates = _validate_date_list(f"top_level_splits.{role}", top_level[role])
        role_dates[role] = dates
        for date_value in dates:
            if date_value in date_to_role:
                raise ValueError(f"Date assigned to multiple top-level roles: {date_value}")
            date_to_role[date_value] = role

    folds = config["cv_folds"]
    if not isinstance(folds, list) or not folds:
        raise ValueError("cv_folds must be a non-empty list")
    model_dates = set(role_dates["model_development"])
    date_to_fold: dict[str, int] = {}
    seen_fold_ids: set[int] = set()
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise ValueError("Each cv_folds entry must be an object")
        if set(fold) != {"fold", "train_dates", "validation_dates"}:
            raise ValueError(
                "Each CV fold must contain fold, train_dates, and validation_dates"
            )
        fold_id = fold["fold"]
        if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id <= 0:
            raise ValueError("CV fold identifiers must be positive integers")
        if fold_id in seen_fold_ids:
            raise ValueError(f"Duplicate CV fold identifier: {fold_id}")
        seen_fold_ids.add(fold_id)

        train_dates = set(_validate_date_list(f"fold {fold_id}.train_dates", fold["train_dates"]))
        validation_dates = set(
            _validate_date_list(
                f"fold {fold_id}.validation_dates", fold["validation_dates"]
            )
        )
        if train_dates & validation_dates:
            raise ValueError(f"Fold {fold_id} train and validation dates overlap")
        if train_dates | validation_dates != model_dates:
            raise ValueError(
                f"Fold {fold_id} train and validation dates must exactly cover "
                "model_development"
            )
        for date_value in validation_dates:
            if date_value in date_to_fold:
                raise ValueError(
                    f"Model-development date is validation in multiple folds: {date_value}"
                )
            date_to_fold[date_value] = fold_id

    if set(date_to_fold) != model_dates:
        raise ValueError(
            "Every model_development date must be validation in exactly one CV fold"
        )

    freeze = config["final_test_freeze"]
    if not isinstance(freeze, Mapping) or freeze.get("frozen") is not True:
        raise ValueError("final_test_freeze.frozen must be true")
    if not freeze.get("statement") or not freeze.get("prohibited_uses"):
        raise ValueError("final_test_freeze must document its statement and prohibited uses")

    return date_to_role, date_to_fold


def _filenames_sha256(filenames: Iterable[str]) -> str:
    canonical = "\n".join(sorted(filenames)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _render_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def render_manifest_csv(artifacts: SplitArtifacts) -> bytes:
    return _render_csv(artifacts.manifest_rows, MANIFEST_FIELDS)


def render_summary_csv(artifacts: SplitArtifacts) -> bytes:
    return _render_csv(artifacts.summary_rows, SUMMARY_FIELDS)


def render_fingerprint_json(artifacts: SplitArtifacts) -> bytes:
    text = json.dumps(
        artifacts.fingerprint,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def build_split_artifacts(image_dir: Path, config_path: Path) -> SplitArtifacts:
    """Build all split artifacts in memory without using label values."""

    config_path = Path(config_path)
    config = load_split_config(config_path)
    date_to_role, date_to_fold = compile_date_mappings(config)
    records = scan_image_records(Path(image_dir))

    filenames = [record.filename for record in records]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Dataset contains duplicate filenames")

    dataset_dates = {record.date for record in records}
    configured_dates = set(date_to_role)
    if dataset_dates != configured_dates:
        missing_from_config = sorted(dataset_dates - configured_dates)
        absent_from_dataset = sorted(configured_dates - dataset_dates)
        raise ValueError(
            "Dataset/config date mismatch. "
            f"Unassigned dataset dates={missing_from_config}; "
            f"configured dates absent from dataset={absent_from_dataset}"
        )

    manifest_rows = []
    for record in records:
        role = date_to_role[record.date]
        manifest_rows.append(
            {
                "filename": record.filename,
                "timestamp": record.timestamp.isoformat(timespec="seconds"),
                "date": record.date,
                "top_level_role": role,
                "cv_validation_fold": date_to_fold[record.date]
                if role == "model_development"
                else "",
            }
        )

    top_level = config["top_level_splits"]
    summary_rows = []
    for role in TOP_LEVEL_ROLES:
        dates = tuple(top_level[role])
        summary_rows.append(
            {
                "scope": "top_level",
                "name": role,
                "top_level_role": role,
                "cv_fold": "",
                "subset": role,
                "date_count": len(dates),
                "sample_count": sum(
                    row["top_level_role"] == role for row in manifest_rows
                ),
                "dates": "|".join(dates),
            }
        )

    for fold in sorted(config["cv_folds"], key=lambda item: item["fold"]):
        fold_id = fold["fold"]
        for subset in ("train", "validation"):
            dates = tuple(fold[f"{subset}_dates"])
            date_set = set(dates)
            summary_rows.append(
                {
                    "scope": "cv",
                    "name": f"fold_{fold_id}_{subset}",
                    "top_level_role": "model_development",
                    "cv_fold": fold_id,
                    "subset": subset,
                    "date_count": len(dates),
                    "sample_count": sum(row["date"] in date_set for row in manifest_rows),
                    "dates": "|".join(dates),
                }
            )

    manifest_bytes = _render_csv(manifest_rows, MANIFEST_FIELDS)
    fingerprint = {
        "schema_version": 1,
        "split_version": config["split_version"],
        "total_files": len(records),
        "sorted_filenames_sha256": _filenames_sha256(filenames),
        "filename_hash_canonicalization": (
            "UTF-8 filenames sorted lexicographically and joined by LF with no trailing LF"
        ),
        "earliest_timestamp": min(record.timestamp for record in records).isoformat(
            timespec="seconds"
        ),
        "latest_timestamp": max(record.timestamp for record in records).isoformat(
            timespec="seconds"
        ),
        "independent_date_count": len(dataset_dates),
        "split_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    return SplitArtifacts(
        manifest_rows=tuple(manifest_rows),
        summary_rows=tuple(summary_rows),
        fingerprint=fingerprint,
    )


def write_split_artifacts(artifacts: SplitArtifacts, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.csv").write_bytes(render_manifest_csv(artifacts))
    (output_dir / "split_summary.csv").write_bytes(render_summary_csv(artifacts))
    (output_dir / "dataset_fingerprint.json").write_bytes(
        render_fingerprint_json(artifacts)
    )


def build_and_write(
    image_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> SplitArtifacts:
    artifacts = build_split_artifacts(image_dir, config_path)
    write_split_artifacts(artifacts, output_dir)
    return artifacts


def main() -> None:
    artifacts = build_and_write(
        DEFAULT_IMAGE_DIR,
        DEFAULT_CONFIG_PATH,
        DEFAULT_OUTPUT_DIR,
    )
    print(f"Split version: {artifacts.fingerprint['split_version']}")
    print(f"Total files: {artifacts.fingerprint['total_files']}")
    for row in artifacts.summary_rows:
        if row["scope"] == "top_level":
            print(f"{row['name']}: {row['sample_count']}")
    print(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    print(
        "Sorted filename SHA256: "
        f"{artifacts.fingerprint['sorted_filenames_sha256']}"
    )


if __name__ == "__main__":
    main()
