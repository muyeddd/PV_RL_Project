from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import experiments.build_paper1_clean_random_v1 as split


def synthetic_keys(n_per_date: int = 20):
    keys = []
    for date in split.ALLOWED_DATES:
        start = datetime.fromisoformat(date + "T10:00:00")
        for index in range(n_per_date):
            timestamp = (start + timedelta(seconds=5 * index)).isoformat()
            keys.append(split.AssignmentKey(split.sample_id_from_timestamp(timestamp), date, timestamp))
    return keys


def synthetic_records(n_per_date: int = 20):
    return [split.LocatedRecord(key, f"data/raw/PanelImages/{key.sample_id}.jpg") for key in synthetic_keys(n_per_date)]


def test_allowed_date_guard():
    split.validate_date(split.ALLOWED_DATES[0])
    with pytest.raises(ValueError, match="Non-allowed"):
        split.validate_date("2017-07-01")


def test_sealed_date_rejection():
    with pytest.raises(ValueError, match="Sealed"):
        split.validate_date(split.SEALED_FINAL_DATES[0])


def test_exact_role_names():
    assert split.ROLES == ("TRAIN", "MODEL_VALIDATION", "CP_CALIBRATION", "DECISION_DEVELOPMENT", "RANDOM_TEST")


def test_role_ratios_and_integer_allocation():
    assert sum(value for _, value in split.ROLE_RATIOS) == pytest.approx(1.0)
    counts = split.allocate_counts(101)
    assert sum(counts.values()) == 101
    assert all(abs(counts[role] / 101 - ratio) <= 1 / 101 for role, ratio in split.ROLE_RATIOS)


def test_date_stratified_assignment():
    keys = synthetic_keys()
    assigned = split.assign_keys(keys)
    for date in split.ALLOWED_DATES:
        assert {assigned[k.sample_id] for k in keys if k.date == date} == set(split.ROLES)


def test_sample_and_image_path_disjointness():
    records = synthetic_records()
    rows = split.materialize_rows(records, split.assign_keys([r.key for r in records]))
    assert len(rows) == len({row["sample_id"] for row in rows})
    assert len(rows) == len({row["image_path"] for row in rows})


def test_deterministic_seed_and_assignment():
    keys = synthetic_keys()
    assert split.stable_date_seed("2017-06-13") == split.stable_date_seed("2017-06-13")
    assert split.assign_keys(keys) == split.assign_keys(list(reversed(keys)))


def test_label_blind_assignment_interface():
    fields = set(split.AssignmentKey.__dataclass_fields__)
    forbidden = {"true_L", "irradiance", "image", "prediction", "regime"}
    assert not fields & forbidden
    assert fields == {"sample_id", "date", "timestamp"}


def test_random_test_truth_lock():
    row = {"role": "RANDOM_TEST", "image_path": "x_L_0.9_I_0.3.jpg"}
    with pytest.raises(PermissionError):
        split.development_true_l(row)
    protocol = split.protocol_object()
    assert protocol["random_test_locked"] is True
    assert protocol["random_test_truth_access_allowed"] is False


def test_no_checkpoint_or_training_path():
    protocol = split.protocol_object()
    assert "true_L" in protocol["assignment_forbidden_inputs"]
    assert "prediction" in protocol["assignment_forbidden_inputs"]
    source = Path(split.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "load_state_dict(" not in source
    assert "optimizer." not in source


def test_output_collision_protection(tmp_path):
    target = tmp_path / "out"
    target.mkdir(); (target / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        split.ensure_output_available(target)
    assert (target / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_sha256_generation():
    assert split.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_all_legal_development_samples_assigned_once():
    records = synthetic_records()
    assignments = split.assign_keys([r.key for r in records])
    rows = split.materialize_rows(records, assignments)
    assert len(rows) == len(records) == len(assignments)
    assert set(assignments) == {r.key.sample_id for r in records}


def test_source_filter_excludes_sealed_before_output(tmp_path):
    path = tmp_path / "source.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "timestamp", "date"])
        writer.writeheader()
        for date in (*split.ALLOWED_DATES, *split.SEALED_FINAL_DATES):
            stamp = date + "T10:00:00"
            writer.writerow({"filename": f"sample_{date}_L_0.7_I_0.4.jpg", "timestamp": stamp, "date": date})
    records = split.read_allowed_records(path)
    assert len(records) == len(split.ALLOWED_DATES)
    assert not ({r.key.date for r in records} & set(split.SEALED_FINAL_DATES))


def test_protocol_provenance_flags_are_lockable():
    protocol = split.protocol_object()
    assert protocol["split_frozen"] is True
    assert protocol["random_test_model_selection_allowed"] is False
    assert protocol["random_test_cp_selection_allowed"] is False
    assert protocol["random_test_decision_selection_allowed"] is False
