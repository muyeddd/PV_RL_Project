"""Build the frozen Paper1 Clean Random v1 split manifest.

Assignment is date-stratified, deterministic, and label blind.  The existing
date metadata is used to discard sealed dates before any locator or embedded
label is retained.  No image, model, checkpoint, or prediction code is used.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATE_MANIFEST = PROJECT_ROOT / "splits" / "date_grouped_v1" / "split_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits" / "paper1_clean_random_v1"
PROTOCOL_NAME = "paper1_clean_random_v1"
BASE_SEED = 42

ALLOWED_DATES = (
    "2017-06-13", "2017-06-14", "2017-06-16", "2017-06-20",
    "2017-06-21", "2017-06-22", "2017-06-23", "2017-06-25",
    "2017-06-26", "2017-06-27", "2017-06-28", "2017-06-29",
)
SEALED_FINAL_DATES = ("2017-06-15", "2017-06-24", "2017-06-30")
ROLE_RATIOS = (
    ("TRAIN", 0.70),
    ("MODEL_VALIDATION", 0.10),
    ("CP_CALIBRATION", 0.08),
    ("DECISION_DEVELOPMENT", 0.05),
    ("RANDOM_TEST", 0.07),
)
ROLES = tuple(role for role, _ in ROLE_RATIOS)
DEVELOPMENT_ROLES = ROLES[:-1]
THRESHOLDS = (5, 10, 30, 60, 120)
MANIFEST_FIELDS = ("sample_id", "image_path", "date", "timestamp", "role")
HASHED_FILES = (
    "all_assignments.csv", "train.csv", "model_validation.csv",
    "cp_calibration.csv", "decision_development.csv", "random_test.csv",
    "protocol.json",
)
ROLE_FILENAMES = {
    "TRAIN": "train.csv",
    "MODEL_VALIDATION": "model_validation.csv",
    "CP_CALIBRATION": "cp_calibration.csv",
    "DECISION_DEVELOPMENT": "decision_development.csv",
    "RANDOM_TEST": "random_test.csv",
}


@dataclass(frozen=True)
class AssignmentKey:
    sample_id: str
    date: str
    timestamp: str


@dataclass(frozen=True)
class LocatedRecord:
    key: AssignmentKey
    image_path: str


def stable_date_seed(date: str, base_seed: int = BASE_SEED) -> int:
    digest = hashlib.sha256(f"{base_seed}|{date}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sample_id_from_timestamp(timestamp: str) -> str:
    digest = hashlib.sha256(f"{PROTOCOL_NAME}|{timestamp}".encode("ascii")).hexdigest()
    return f"p1crv1_{digest[:24]}"


def validate_date(date: str) -> None:
    if date in SEALED_FINAL_DATES:
        raise ValueError(f"Sealed date rejected: {date}")
    if date not in ALLOWED_DATES:
        raise ValueError(f"Non-allowed date rejected: {date}")


def allocate_counts(n: int) -> dict[str, int]:
    if n < len(ROLES):
        raise ValueError("Each date must contain at least one sample per role")
    raw = [(role, n * ratio) for role, ratio in ROLE_RATIOS]
    counts = {role: math.floor(value) for role, value in raw}
    remainder = n - sum(counts.values())
    order = sorted(
        range(len(raw)),
        key=lambda i: (-(raw[i][1] - math.floor(raw[i][1])), i),
    )
    for index in order[:remainder]:
        counts[raw[index][0]] += 1
    if sum(counts.values()) != n or any(counts[role] <= 0 for role in ROLES):
        raise AssertionError("Invalid integer role allocation")
    return counts


def read_allowed_records(source_manifest: Path) -> tuple[LocatedRecord, ...]:
    """Filter on existing date metadata before retaining sample locators.

    Sealed rows are discarded immediately.  Their filename, image, label,
    irradiance, prediction, and per-sample properties are never retained.
    """
    source_manifest = Path(source_manifest)
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    records: list[LocatedRecord] = []
    allowed = set(ALLOWED_DATES)
    recognized = allowed | set(SEALED_FINAL_DATES)
    seen_dates: set[str] = set()
    seen_timestamps: set[str] = set()
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "timestamp", "date"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Source date manifest is missing filename/timestamp/date")
        for row in reader:
            date = row["date"]
            if date not in recognized:
                raise ValueError(f"Unexpected source date: {date}")
            seen_dates.add(date)
            if date not in allowed:
                continue
            validate_date(date)
            timestamp = datetime.fromisoformat(row["timestamp"]).isoformat(timespec="seconds")
            if not timestamp.startswith(date + "T"):
                raise ValueError("Date/timestamp mismatch")
            if timestamp in seen_timestamps:
                raise ValueError(f"Duplicate timestamp prevents label-blind identity: {timestamp}")
            seen_timestamps.add(timestamp)
            sample_id = sample_id_from_timestamp(timestamp)
            locator = (Path("data") / "raw" / "PanelImages" / row["filename"]).as_posix()
            records.append(LocatedRecord(AssignmentKey(sample_id, date, timestamp), locator))
    if seen_dates != recognized:
        raise ValueError(f"Source date set mismatch: {sorted(seen_dates)}")
    if {record.key.date for record in records} != allowed:
        raise ValueError("Allowed date coverage is incomplete")
    return tuple(records)


def assign_keys(keys: Sequence[AssignmentKey]) -> dict[str, str]:
    """Assign roles using only opaque ID, date, timestamp, and deterministic seed."""
    by_date: dict[str, list[AssignmentKey]] = defaultdict(list)
    for key in keys:
        validate_date(key.date)
        if not key.timestamp.startswith(key.date + "T"):
            raise ValueError("Date/timestamp mismatch")
        by_date[key.date].append(key)
    if set(by_date) != set(ALLOWED_DATES):
        raise ValueError("Assignment dates must exactly equal allowed dates")
    if len(keys) != len({key.sample_id for key in keys}):
        raise ValueError("Duplicate sample_id")

    assignments: dict[str, str] = {}
    for date in ALLOWED_DATES:
        group = sorted(by_date[date], key=lambda item: (item.timestamp, item.sample_id))
        random.Random(stable_date_seed(date)).shuffle(group)
        counts = allocate_counts(len(group))
        cursor = 0
        for role in ROLES:
            for key in group[cursor:cursor + counts[role]]:
                assignments[key.sample_id] = role
            cursor += counts[role]
        if cursor != len(group):
            raise AssertionError("Date allocation did not consume all samples")
    return assignments


def materialize_rows(
    records: Sequence[LocatedRecord], assignments: Mapping[str, str]
) -> list[dict[str, str]]:
    rows = []
    for record in sorted(records, key=lambda item: (item.key.date, item.key.timestamp)):
        role = assignments[record.key.sample_id]
        rows.append({
            "sample_id": record.key.sample_id,
            "image_path": record.image_path,
            "date": record.key.date,
            "timestamp": record.key.timestamp,
            "role": role,
        })
    validate_assignments(rows, records)
    return rows


def validate_assignments(rows: Sequence[Mapping[str, str]], records: Sequence[LocatedRecord]) -> None:
    if len(rows) != len(records):
        raise ValueError("Not all legal development samples were assigned")
    ids = [row["sample_id"] for row in rows]
    paths = [row["image_path"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id overlap across roles")
    if len(paths) != len(set(paths)):
        raise ValueError("image_path overlap across roles")
    if set(ids) != {record.key.sample_id for record in records}:
        raise ValueError("Assignment/sample inventory mismatch")
    if set(row["role"] for row in rows) != set(ROLES):
        raise ValueError("Role set mismatch")
    if set(row["date"] for row in rows) != set(ALLOWED_DATES):
        raise ValueError("Output date set mismatch")
    if any(row["date"] in SEALED_FINAL_DATES for row in rows):
        raise ValueError("Sealed date leaked into output")
    for role in ROLES:
        role_ids = {row["sample_id"] for row in rows if row["role"] == role}
        for other in ROLES:
            if role < other and role_ids & {
                row["sample_id"] for row in rows if row["role"] == other
            }:
                raise ValueError("Role sample overlap")


def csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def summary_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    total = len(rows)
    result = []
    for role, target in ROLE_RATIOS:
        group = [row for row in rows if row["role"] == role]
        result.append({
            "role": role, "N": len(group), "ratio": len(group) / total,
            "target_ratio": target,
            "dates": "|".join(sorted({row["date"] for row in group})),
        })
    return result


def date_role_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    result = []
    for date in ALLOWED_DATES:
        date_rows = [row for row in rows if row["date"] == date]
        for role, target in ROLE_RATIOS:
            n = sum(row["role"] == role for row in date_rows)
            result.append({"date": date, "role": role, "N": n,
                           "date_N": len(date_rows), "ratio": n / len(date_rows),
                           "target_ratio": target})
    return result


def temporal_neighbor_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    timestamps: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for row in rows:
        timestamps[(row["date"], row["role"])].append(datetime.fromisoformat(row["timestamp"]))
    for values in timestamps.values():
        values.sort()
    result = []
    for date in (*ALLOWED_DATES, "ALL_DATES"):
        for i, role_1 in enumerate(ROLES):
            for role_2 in ROLES[i + 1:]:
                counts = {limit: 0 for limit in THRESHOLDS}
                dates = ALLOWED_DATES if date == "ALL_DATES" else (date,)
                n1 = n2 = 0
                for one_date in dates:
                    left = timestamps[(one_date, role_1)]
                    right = timestamps[(one_date, role_2)]
                    n1 += len(left); n2 += len(right)
                    for limit in THRESHOLDS:
                        lower = upper = 0
                        for a in left:
                            while lower < len(right) and (a - right[lower]).total_seconds() > limit:
                                lower += 1
                            if upper < lower:
                                upper = lower
                            while upper < len(right) and (right[upper] - a).total_seconds() <= limit:
                                upper += 1
                            counts[limit] += upper - lower
                row: dict[str, object] = {"date": date, "role_1": role_1,
                                         "role_2": role_2, "role_1_N": n1,
                                         "role_2_N": n2}
                row.update({f"pair_N_le_{limit}s": counts[limit] for limit in THRESHOLDS})
                result.append(row)
    return result


TRUE_L_PATTERN = re.compile(r"_L_([0-9eE+.-]+)_I_")


def development_true_l(row: Mapping[str, str]) -> float:
    if row["role"] not in DEVELOPMENT_ROLES:
        raise PermissionError("RANDOM_TEST truth access is locked")
    match = TRUE_L_PATTERN.search(Path(row["image_path"]).name)
    if match is None:
        raise ValueError("Cannot parse development true_L")
    return float(match.group(1))


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index); high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def label_summary_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    result = []
    for role in DEVELOPMENT_ROLES:
        for scope, date in (("role", ""), *(("date_role", d) for d in ALLOWED_DATES)):
            group = [row for row in rows if row["role"] == role and (not date or row["date"] == date)]
            values = [development_true_l(row) for row in group]
            result.append({
                "scope": scope, "date": date, "role": role, "N": len(values),
                "true_mean": fmean(values), "true_std": pstdev(values),
                "q05": quantile(values, .05), "q25": quantile(values, .25),
                "q50": quantile(values, .50), "q75": quantile(values, .75),
                "q95": quantile(values, .95),
                "LOW_fraction": sum(v < .1 for v in values) / len(values),
                "MEDIUM_fraction": sum(.1 <= v < .5 for v in values) / len(values),
                "HIGH_fraction": sum(v >= .5 for v in values) / len(values),
            })
    return result


def protocol_object() -> dict[str, object]:
    return {
        "protocol_name": PROTOCOL_NAME,
        "schema_version": 1,
        "split_type": "date_stratified_random_frame",
        "base_seed": BASE_SEED,
        "date_seed_derivation": "uint64_be(first_8_bytes(SHA256(base_seed|ISO_date)))",
        "integer_allocation": "largest remainder; stable role-order tie break",
        "roles": {role: ratio for role, ratio in ROLE_RATIOS},
        "allowed_dates": list(ALLOWED_DATES),
        "sealed_final_dates": list(SEALED_FINAL_DATES),
        "assignment_inputs": ["opaque sample_id", "date", "timestamp", "derived seed"],
        "assignment_forbidden_inputs": ["true_L", "irradiance", "image content", "image feature", "prediction", "model output", "error", "regime"],
        "random_test_locked": True,
        "random_test_truth_access_allowed": False,
        "random_test_model_selection_allowed": False,
        "random_test_cp_selection_allowed": False,
        "random_test_decision_selection_allowed": False,
        "unlock_condition": "point model architecture and hyperparameters frozen",
        "temporal_neighbor_leakage_possible": True,
        "temporal_independent_validation": False,
        "robustness_validation": "Use a separate block-disjoint protocol",
        "split_frozen": True,
    }


def ensure_output_available(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build(output_dir: Path = OUTPUT_DIR, source_manifest: Path = SOURCE_DATE_MANIFEST) -> dict[str, object]:
    ensure_output_available(output_dir)
    records = read_allowed_records(source_manifest)
    assignments = assign_keys([record.key for record in records])
    rows = materialize_rows(records, assignments)

    core: dict[str, bytes] = {"all_assignments.csv": csv_bytes(rows, MANIFEST_FIELDS)}
    for role in ROLES:
        core[ROLE_FILENAMES[role]] = csv_bytes(
            [row for row in rows if row["role"] == role], MANIFEST_FIELDS
        )
    protocol = protocol_object()
    core["protocol.json"] = json_bytes(protocol)
    core["split_summary.csv"] = csv_bytes(
        summary_rows(rows), ("role", "N", "ratio", "target_ratio", "dates")
    )
    core["date_role_counts.csv"] = csv_bytes(
        date_role_rows(rows), ("date", "role", "N", "date_N", "ratio", "target_ratio")
    )
    temporal_fields = ("date", "role_1", "role_2", "role_1_N", "role_2_N",
                       *(f"pair_N_le_{limit}s" for limit in THRESHOLDS))
    core["temporal_neighbor_audit.csv"] = csv_bytes(temporal_neighbor_rows(rows), temporal_fields)
    hashes = {name: sha256_bytes(core[name]) for name in HASHED_FILES}
    core["hashes.json"] = json_bytes({"algorithm": "SHA256", "files": hashes})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in core.items():
        write_exclusive(output_dir / name, payload)

    # Only after the split and its required hashes are frozen do we parse labels,
    # and only for roles that are not RANDOM_TEST.
    label_fields = ("scope", "date", "role", "N", "true_mean", "true_std",
                    "q05", "q25", "q50", "q75", "q95", "LOW_fraction",
                    "MEDIUM_fraction", "HIGH_fraction")
    write_exclusive(
        output_dir / "development_label_summary.csv",
        csv_bytes(label_summary_rows(rows), label_fields),
    )

    allowed_source_payload = "\n".join(
        f"{record.key.sample_id},{record.key.date},{record.key.timestamp},{record.image_path}"
        for record in records
    ).encode("utf-8")
    provenance = {
        "protocol_name": PROTOCOL_NAME,
        "split_type": "date_stratified_random_frame",
        "base_seed": BASE_SEED,
        "split_frozen": True,
        "label_used_for_assignment": False,
        "prediction_used_for_assignment": False,
        "training_performed": False,
        "checkpoint_loaded": False,
        "random_test_locked": True,
        "random_test_truth_accessed": False,
        "random_test_predictions_generated": False,
        "sealed_final_dates": list(SEALED_FINAL_DATES),
        "sealed_final_dates_accessed": False,
        "sealed_filtering": "date metadata filter applied before retaining locator; no sealed image/label/I/prediction/statistic retained",
        "existing_outputs_modified": False,
        "temporal_neighbor_leakage_possible": True,
        "temporal_independent_validation": False,
        "source_date_manifest": str(Path(source_manifest).relative_to(PROJECT_ROOT)).replace(os.sep, "/") if Path(source_manifest).is_absolute() else str(source_manifest),
        "allowed_source_records_sha256": hashlib.sha256(allowed_source_payload).hexdigest(),
        "total_development_N": len(rows),
        "manifest_sha256": hashes,
    }
    write_exclusive(output_dir / "provenance.json", json_bytes(provenance))
    return {"rows": rows, "summary": summary_rows(rows), "hashes": hashes,
            "provenance": provenance}


def main() -> None:
    result = build()
    print(f"protocol={PROTOCOL_NAME}")
    print(f"total_development_N={len(result['rows'])}")
    for row in result["summary"]:
        print(f"{row['role']}={row['N']} ({row['ratio']:.8f})")
    print(f"output_dir={OUTPUT_DIR}")
    print("random_test_truth_accessed=false")
    print("sealed_final_dates_accessed=false")


if __name__ == "__main__":
    main()
