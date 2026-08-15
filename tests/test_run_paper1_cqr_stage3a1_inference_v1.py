from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import experiments.run_paper1_cqr_stage3a1_inference_v1 as stage3a1


def valid_config() -> dict[str, object]:
    return {
        "protocol": stage3a1.PROTOCOL,
        "stage": stage3a1.CQR_STAGE,
        "model": stage3a1.CQR_STAGE,
        "seed": stage3a1.SEED,
        "architecture": stage3a1.ARCHITECTURE,
        "dropout": stage3a1.DROPOUT,
        "quantile_levels": list(stage3a1.QUANTILE_LEVELS),
        "ordered_quantile_parameterization": True,
        "image_preprocessing": stage3a1.cqr_train.expected_point_preprocessing(),
        "irradiance_normalization": (
            "frozen clean TRAIN-only z-score; population std ddof=0"
        ),
        "imagenet_weights_at_construction": None,
        "imagenet_download_performed": False,
    }


def valid_checkpoint() -> dict[str, object]:
    return {
        "model_state_dict": {},
        "epoch": stage3a1.SOURCE_CQR_BEST_EPOCH,
        "validation_mean_pinball": stage3a1.SOURCE_CQR_VALIDATION_MEAN_PINBALL,
        "validation_pinball_q05": 0.01,
        "validation_pinball_q50": 0.011,
        "validation_pinball_q95": 0.012,
        "config": valid_config(),
    }


def valid_stats() -> dict[str, object]:
    return {
        "N": stage3a1.EXPECTED_TRAIN_N,
        "mean": stage3a1.EXPECTED_IRRADIANCE_MEAN,
        "std_ddof0": stage3a1.EXPECTED_IRRADIANCE_STD_DDOF0,
        "min": 0.0,
        "max": 1.0,
        "normalization": "z_score",
        "source_role": "TRAIN",
    }


def role_frame(
    role: str,
    n: int = 2,
    *,
    date: str = "2017-06-13",
    prefix: str = "sample",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}_{index}" for index in range(n)],
            "date": [date] * n,
            "timestamp": [f"{date}T10:{index:02d}:00" for index in range(n)],
            "image_path": [
                f"data/raw/PanelImages/{prefix}_{index}_L_0.2_I_{0.3 + index * 0.1}.jpg"
                for index in range(n)
            ],
            "role": [role] * n,
        }
    )


def prediction_frame(role: str, prefix: str = "sample") -> pd.DataFrame:
    records = stage3a1.prepare_inference_records(
        role_frame(role, prefix=prefix),
        role,
        valid_stats(),
        enforce_expected_n=False,
    )
    quantiles = np.array([[0.1, 0.2, 0.4], [0.2, 0.3, 0.5]])
    return stage3a1.build_prediction_frame(
        records, quantiles, role, enforce_expected_n=False
    )


def test_fixed_protocol_checkpoint_and_role_constants() -> None:
    assert stage3a1.PROTOCOL == "paper1_clean_random_v1"
    assert stage3a1.STAGE == "cqr_stage3a1_inference_v1"
    assert stage3a1.SOURCE_CQR_CHECKPOINT == (
        stage3a1.PROJECT_ROOT
        / "outputs/paper1_clean_random_v1/cqr_resnet50_with_i_v1/seed_42/best_model.pth"
    )
    assert stage3a1.SOURCE_CQR_CHECKPOINT_SHA256 == (
        "fd5deea62c867fcffe3791f768752da9dc3a39a1c146244b1e225d6b40b0da80"
    )
    assert len(stage3a1.SOURCE_CQR_CHECKPOINT_SHA256) == 64
    assert (
        format(int(stage3a1.SOURCE_CQR_CHECKPOINT_SHA256, 16), "064x")
        == stage3a1.SOURCE_CQR_CHECKPOINT_SHA256
    )
    assert stage3a1.EXPECTED_N == {
        "CP_CALIBRATION": 2951,
        "DECISION_DEVELOPMENT": 1844,
    }


def test_checkpoint_exact_resolved_path_guard(tmp_path: Path, monkeypatch) -> None:
    authorized = tmp_path / "authorized.pth"
    authorized.write_bytes(b"checkpoint")
    monkeypatch.setattr(stage3a1, "SOURCE_CQR_CHECKPOINT", authorized)
    assert stage3a1.validate_checkpoint_path(authorized) == authorized.resolve()
    with pytest.raises(PermissionError, match="Unauthorized CQR checkpoint"):
        stage3a1.validate_checkpoint_path(tmp_path / "other.pth")


def test_checkpoint_sha_checked_before_deserialization(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "best_model.pth"
    path.write_bytes(b"synthetic checkpoint bytes")
    monkeypatch.setattr(stage3a1, "SOURCE_CQR_CHECKPOINT", path)
    load_calls: list[object] = []

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return valid_checkpoint()

    monkeypatch.setattr(stage3a1.torch, "load", fake_load)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        stage3a1.load_verified_cqr_checkpoint(path, "0" * 64)
    assert load_calls == []

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = stage3a1.load_verified_cqr_checkpoint(path, expected)
    assert loaded["epoch"] == 15
    assert load_calls[0][1] == {"map_location": "cpu", "weights_only": True}


def test_frozen_checkpoint_sha_passes_guard_and_wrong_sha_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "best_model.pth"
    path.write_bytes(b"synthetic checkpoint bytes")
    monkeypatch.setattr(stage3a1, "SOURCE_CQR_CHECKPOINT", path)
    monkeypatch.setattr(
        stage3a1,
        "sha256_file",
        lambda candidate: stage3a1.SOURCE_CQR_CHECKPOINT_SHA256,
    )
    monkeypatch.setattr(stage3a1.torch, "load", lambda *args, **kwargs: valid_checkpoint())

    loaded = stage3a1.load_verified_cqr_checkpoint(path)
    assert loaded["epoch"] == stage3a1.SOURCE_CQR_BEST_EPOCH
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        stage3a1.load_verified_cqr_checkpoint(path, "0" * 64)


@pytest.mark.parametrize("missing", sorted(stage3a1.REQUIRED_CHECKPOINT_FIELDS))
def test_required_checkpoint_fields(missing: str) -> None:
    checkpoint = valid_checkpoint()
    checkpoint.pop(missing)
    with pytest.raises(ValueError, match="missing fields"):
        stage3a1.validate_checkpoint_schema(checkpoint)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol", "wrong_protocol"),
        ("stage", "wrong_stage"),
        ("model", "wrong_model"),
        ("seed", 7),
        ("architecture", "wrong_architecture"),
        ("ordered_quantile_parameterization", False),
        ("irradiance_normalization", "CP z-score"),
        ("imagenet_weights_at_construction", "IMAGENET1K_V2"),
        ("imagenet_download_performed", True),
    ],
)
def test_checkpoint_exact_config_guards(field: str, value: object) -> None:
    checkpoint = valid_checkpoint()
    checkpoint["config"][field] = value
    with pytest.raises(ValueError, match="config mismatch"):
        stage3a1.validate_checkpoint_schema(checkpoint)


def test_checkpoint_dropout_and_quantile_level_guards() -> None:
    checkpoint = valid_checkpoint()
    checkpoint["config"]["dropout"] = 0.2
    with pytest.raises(ValueError, match="dropout mismatch"):
        stage3a1.validate_checkpoint_schema(checkpoint)

    checkpoint = valid_checkpoint()
    checkpoint["config"]["quantile_levels"] = [0.1, 0.5, 0.9]
    with pytest.raises(ValueError, match="quantile_levels mismatch"):
        stage3a1.validate_checkpoint_schema(checkpoint)


def test_checkpoint_preprocessing_and_best_epoch_guards() -> None:
    checkpoint = valid_checkpoint()
    checkpoint["config"]["image_preprocessing"] = {"validation": "wrong"}
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        stage3a1.validate_checkpoint_schema(checkpoint)

    checkpoint = valid_checkpoint()
    checkpoint["epoch"] = 14
    with pytest.raises(ValueError, match="best epoch mismatch"):
        stage3a1.validate_checkpoint_schema(checkpoint)


@pytest.mark.parametrize("field", stage3a1.VALIDATION_METRIC_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_all_checkpoint_validation_metrics_must_be_finite(
    field: str, value: float
) -> None:
    checkpoint = valid_checkpoint()
    checkpoint[field] = value
    with pytest.raises(ValueError, match="must be finite"):
        stage3a1.validate_checkpoint_schema(checkpoint)


def test_checkpoint_selected_metric_value_is_frozen() -> None:
    checkpoint = valid_checkpoint()
    checkpoint["validation_mean_pinball"] += 1e-8
    with pytest.raises(ValueError, match="validation_mean_pinball mismatch"):
        stage3a1.validate_checkpoint_schema(checkpoint)


def test_strict_state_load_eval_and_requires_grad_false(monkeypatch) -> None:
    construction: dict[str, object] = {}

    class TinyCQR(nn.Module):
        def __init__(self, dropout: float):
            super().__init__()
            construction["dropout"] = dropout
            self.weight = nn.Parameter(torch.ones(1))
            self.dropout = nn.Dropout(dropout)

        def load_state_dict(self, state_dict, strict: bool = True):
            construction["state"] = state_dict
            construction["strict"] = strict
            return None

    monkeypatch.setattr(stage3a1.cqr_train, "Paper1CQRResNet50WithI", TinyCQR)
    checkpoint = valid_checkpoint()
    checkpoint["model_state_dict"] = {"synthetic": torch.ones(1)}
    model = stage3a1.build_inference_model(checkpoint, torch.device("cpu"))
    assert construction["dropout"] == 0.3
    assert construction["strict"] is True
    assert construction["state"] is checkpoint["model_state_dict"]
    assert model.training is False
    assert model.dropout.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_real_cqr_constructor_requests_no_torchvision_weights(monkeypatch) -> None:
    calls: list[object] = []

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, image):
            return torch.zeros((len(image), 4))

    def fake_resnet50(*, weights):
        calls.append(weights)
        return TinyBackbone()

    monkeypatch.setattr(stage3a1.cqr_train, "resnet50", fake_resnet50)
    stage3a1.cqr_train.Paper1CQRResNet50WithI(dropout=0.3)
    assert calls == [None]


def test_deterministic_inference_is_single_pass_and_maps_quantiles() -> None:
    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.3)
            self.calls = 0
            self.inference_flags: list[bool] = []

        def forward(self, images, irradiance):
            self.calls += 1
            self.inference_flags.append(torch.is_inference_mode_enabled())
            base = irradiance.unsqueeze(1)
            return torch.cat((base, base + 0.1, base + 0.2), dim=1)

    images = torch.zeros((5, 3, 2, 2))
    irradiance = torch.linspace(0.1, 0.5, 5)
    indices = torch.arange(5)
    loader = DataLoader(TensorDataset(images, irradiance, indices), batch_size=2)
    model = CountingModel()
    values = stage3a1.predict_deterministic(model, loader, torch.device("cpu"))
    assert model.calls == len(loader)
    assert model.inference_flags == [True] * len(loader)
    assert model.training is False
    assert model.dropout.training is False
    assert values.shape == (5, 3)
    assert values[:, 0] == pytest.approx(irradiance.numpy())
    assert values[:, 1] == pytest.approx(irradiance.numpy() + 0.1)
    assert values[:, 2] == pytest.approx(irradiance.numpy() + 0.2)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([[0.1, 0.2]]), "shape"),
        (np.array([[0.1, np.nan, 0.3]]), "NaN or infinity"),
        (np.array([[-0.1, 0.2, 0.3]]), "outside"),
        (np.array([[0.1, 0.2, 1.1]]), "outside"),
        (np.array([[0.2, 0.1, 0.3]]), "crossing"),
        (np.array([[0.1, 0.4, 0.3]]), "crossing"),
    ],
)
def test_quantile_output_qc_rejections(values: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        stage3a1.validate_quantile_array(values, 1)


def test_quantile_output_qc_accepts_finite_ordered_unit_interval() -> None:
    values = np.array([[0.0, 0.5, 1.0], [0.2, 0.2, 0.2]])
    observed = stage3a1.validate_quantile_array(values, 2)
    assert np.array_equal(observed, values)


@pytest.mark.parametrize("role", ["TRAIN", "MODEL_VALIDATION", "RANDOM_TEST"])
def test_only_two_inference_roles_are_authorized(role: str, tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="Forbidden|RANDOM_TEST"):
        stage3a1.validate_manifest_authorization(tmp_path / f"{role}.csv", role)


@pytest.mark.parametrize(
    ("role", "constant_name", "filename"),
    [
        ("CP_CALIBRATION", "CP_CALIBRATION_MANIFEST", "cp_calibration.csv"),
        (
            "DECISION_DEVELOPMENT",
            "DECISION_DEVELOPMENT_MANIFEST",
            "decision_development.csv",
        ),
    ],
)
def test_authorized_manifest_paths_are_exact(
    role: str, constant_name: str, filename: str, tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / filename
    monkeypatch.setitem(stage3a1.AUTHORIZED_MANIFESTS, role, path)
    monkeypatch.setattr(stage3a1, constant_name, path)
    assert stage3a1.validate_manifest_authorization(path, role) == path.resolve()
    with pytest.raises(PermissionError, match="not authorized"):
        stage3a1.validate_manifest_authorization(tmp_path / "other.csv", role)


def test_random_test_path_is_explicitly_rejected(tmp_path: Path, monkeypatch) -> None:
    role = stage3a1.CP_CALIBRATION_ROLE
    authorized = tmp_path / "cp_calibration.csv"
    monkeypatch.setitem(stage3a1.AUTHORIZED_MANIFESTS, role, authorized)
    with pytest.raises(PermissionError, match="RANDOM_TEST"):
        stage3a1.validate_manifest_authorization(tmp_path / "random_test.csv", role)


def test_manifest_loader_reads_only_metadata_columns(tmp_path: Path, monkeypatch) -> None:
    role = stage3a1.CP_CALIBRATION_ROLE
    path = tmp_path / "cp_calibration.csv"
    path.write_text("not read by mock", encoding="utf-8")
    monkeypatch.setitem(stage3a1.AUTHORIZED_MANIFESTS, role, path)
    observed: dict[str, object] = {}

    def fake_read_csv(candidate, *, usecols):
        observed["candidate"] = candidate
        observed["usecols"] = usecols
        return role_frame(role, 1)

    monkeypatch.setattr(stage3a1.pd, "read_csv", fake_read_csv)
    monkeypatch.setitem(stage3a1.EXPECTED_N, role, 1)
    stage3a1.load_role_manifest(path, role)
    assert observed["candidate"] == path.resolve()
    assert observed["usecols"] == list(stage3a1.MANIFEST_COLUMNS)
    assert not set(stage3a1.FORBIDDEN_PREDICTION_COLUMNS) & set(observed["usecols"])


def test_expected_n_guards_for_both_roles() -> None:
    for role, expected in (
        (stage3a1.CP_CALIBRATION_ROLE, 2951),
        (stage3a1.DECISION_DEVELOPMENT_ROLE, 1844),
    ):
        assert stage3a1.EXPECTED_N[role] == expected
        with pytest.raises(ValueError, match="N guard failed"):
            stage3a1.validate_manifest_frame(role_frame(role, 1), role)


def test_manifest_role_duplicate_and_locator_guards() -> None:
    frame = role_frame(stage3a1.CP_CALIBRATION_ROLE)
    wrong = frame.copy()
    wrong.loc[0, "role"] = stage3a1.DECISION_DEVELOPMENT_ROLE
    with pytest.raises(PermissionError, match="Role guard"):
        stage3a1.validate_manifest_frame(
            wrong, stage3a1.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )

    duplicate_id = frame.copy()
    duplicate_id.loc[1, "sample_id"] = duplicate_id.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="sample_id"):
        stage3a1.validate_manifest_frame(
            duplicate_id, stage3a1.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )

    duplicate_path = frame.copy()
    duplicate_path.loc[1, "image_path"] = duplicate_path.loc[0, "image_path"]
    with pytest.raises(ValueError, match="image_path"):
        stage3a1.validate_manifest_frame(
            duplicate_path, stage3a1.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )


@pytest.mark.parametrize("sealed_date", sorted(stage3a1.SEALED_FINAL_DATES))
def test_all_sealed_dates_are_rejected(sealed_date: str) -> None:
    frame = role_frame(stage3a1.CP_CALIBRATION_ROLE, 1, date=sealed_date)
    with pytest.raises(PermissionError, match="Sealed final date"):
        stage3a1.validate_manifest_frame(
            frame, stage3a1.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )


def test_random_test_locator_is_rejected() -> None:
    frame = role_frame(stage3a1.CP_CALIBRATION_ROLE, 1)
    frame.loc[0, "image_path"] = "data/random_test/x_L_0.2_I_0.4.jpg"
    with pytest.raises(PermissionError, match="RANDOM_TEST locator"):
        stage3a1.validate_manifest_frame(
            frame, stage3a1.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )


@pytest.mark.parametrize("overlap_field", ["sample_id", "image_path"])
def test_cp_decision_overlap_guard(overlap_field: str) -> None:
    cp = role_frame(stage3a1.CP_CALIBRATION_ROLE, prefix="cp")
    decision = role_frame(stage3a1.DECISION_DEVELOPMENT_ROLE, prefix="decision")
    decision.loc[0, overlap_field] = cp.loc[0, overlap_field]
    with pytest.raises(ValueError, match="overlap"):
        stage3a1.validate_role_isolation(cp, decision)


def test_validation_transform_is_exact_and_has_no_random_augmentation() -> None:
    transform = stage3a1.build_inference_transform()
    stage3a1.validate_inference_transform(transform)
    types = stage3a1.cqr_train.point_train.transforms
    assert [type(item) for item in transform.transforms] == [
        types.Resize,
        types.ToTensor,
        types.Normalize,
    ]
    assert tuple(transform.transforms[0].size) == (224, 224)
    assert tuple(transform.transforms[2].mean) == stage3a1.IMAGENET_MEAN
    assert tuple(transform.transforms[2].std) == stage3a1.IMAGENET_STD


def test_random_transform_is_rejected() -> None:
    types = stage3a1.cqr_train.point_train.transforms
    random_transform = types.Compose(
        [
            types.Resize((224, 224)),
            types.RandomHorizontalFlip(),
            types.ToTensor(),
            types.Normalize(stage3a1.IMAGENET_MEAN, stage3a1.IMAGENET_STD),
        ]
    )
    with pytest.raises(ValueError, match="exactly three"):
        stage3a1.validate_inference_transform(random_transform)


def test_irradiance_stats_path_is_exact(tmp_path: Path, monkeypatch) -> None:
    authorized = tmp_path / "train_irradiance_stats.json"
    monkeypatch.setattr(stage3a1, "TRAIN_IRRADIANCE_STATS_PATH", authorized)
    assert stage3a1.validate_irradiance_stats_path(authorized) == authorized.resolve()
    with pytest.raises(PermissionError, match="Unauthorized irradiance"):
        stage3a1.validate_irradiance_stats_path(tmp_path / "cp_stats.json")


@pytest.mark.parametrize("source_role", ["CP_CALIBRATION", "DECISION_DEVELOPMENT"])
def test_cp_or_decision_stats_cannot_normalize(source_role: str) -> None:
    stats = valid_stats()
    stats["source_role"] = source_role
    with pytest.raises(PermissionError, match="TRAIN statistics only"):
        stage3a1.validate_irradiance_stats(stats)


def test_train_stats_n_mean_std_and_normalization_guards() -> None:
    stats = valid_stats()
    assert stage3a1.validate_irradiance_stats(stats)["N"] == 25830
    for field, value, message in (
        ("N", 25829, "N must equal"),
        ("mean", stage3a1.EXPECTED_IRRADIANCE_MEAN + 1e-10, "mean mismatch"),
        (
            "std_ddof0",
            stage3a1.EXPECTED_IRRADIANCE_STD_DDOF0 + 1e-10,
            "std mismatch",
        ),
        ("normalization", "minmax", "must be z_score"),
    ):
        invalid = valid_stats()
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            stage3a1.validate_irradiance_stats(invalid)


def test_inference_records_parse_only_irradiance_and_preserve_raw_value(monkeypatch) -> None:
    def forbidden_truth_parser(*args, **kwargs):
        raise AssertionError("truth parser must not be called")

    monkeypatch.setattr(
        stage3a1.cqr_train.point_train,
        "attach_development_values",
        forbidden_truth_parser,
    )
    records = stage3a1.prepare_inference_records(
        role_frame(stage3a1.CP_CALIBRATION_ROLE),
        stage3a1.CP_CALIBRATION_ROLE,
        valid_stats(),
        enforce_expected_n=False,
    )
    assert tuple(records.columns) == stage3a1.RECORD_COLUMNS
    assert records["irradiance"].tolist() == pytest.approx([0.3, 0.4])
    assert "true_L" not in records
    assert "label" not in records
    assert "target" not in records


def test_prediction_artifact_schema_excludes_truth_and_maps_q_columns() -> None:
    predictions = prediction_frame(stage3a1.CP_CALIBRATION_ROLE)
    assert tuple(predictions.columns) == stage3a1.PREDICTION_COLUMNS
    assert not stage3a1._truth_like_columns(predictions.columns)
    assert predictions["irradiance"].tolist() == pytest.approx([0.3, 0.4])
    assert predictions["q05"].tolist() == pytest.approx([0.1, 0.2])
    assert predictions["q50"].tolist() == pytest.approx([0.2, 0.3])
    assert predictions["q95"].tolist() == pytest.approx([0.4, 0.5])


@pytest.mark.parametrize(
    "forbidden", ["true_L", "label", "target", "absolute_error", "residual"]
)
def test_truth_or_truth_derived_artifact_columns_are_rejected(forbidden: str) -> None:
    predictions = prediction_frame(stage3a1.CP_CALIBRATION_ROLE)
    predictions[forbidden] = 0.0
    with pytest.raises(ValueError, match="schema mismatch|Truth-derived"):
        stage3a1.validate_prediction_frame(
            predictions,
            stage3a1.CP_CALIBRATION_ROLE,
            enforce_expected_n=False,
        )


def test_quantile_qc_is_descriptive_only() -> None:
    predictions = prediction_frame(stage3a1.CP_CALIBRATION_ROLE)
    summary = stage3a1.quantile_qc_summary(predictions)
    assert summary["N"] == 2
    assert summary["quantile_crossing_count"] == 0
    assert set(summary["q50"]) == {"mean", "std_ddof0", "min", "max"}
    assert set(summary["raw_interval_width_q95_minus_q05"]) == {
        "mean",
        "median",
        "p95",
        "min",
        "max",
    }


def test_output_collision_protection_and_tmp_writer(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage3a1.ensure_output_available(occupied)

    output = tmp_path / "stage3a1"
    cp = prediction_frame(stage3a1.CP_CALIBRATION_ROLE, prefix="cp")
    decision = prediction_frame(
        stage3a1.DECISION_DEVELOPMENT_ROLE, prefix="decision"
    )
    stage3a1.write_stage3a1_outputs(output, cp, decision, {}, {})
    assert {path.name for path in output.iterdir()} == {
        "cp_calibration_predictions.csv",
        "decision_development_predictions.csv",
        "config.json",
        "provenance.json",
    }


def test_formal_output_path_is_fixed(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="Unauthorized Stage 3A1 output"):
        stage3a1.validate_formal_output_path(tmp_path)


def test_config_and_provenance_freeze_non_action_flags() -> None:
    checkpoint = valid_checkpoint()
    config = stage3a1.make_config(checkpoint, torch.device("cpu"), 32)
    provenance = stage3a1.make_provenance(checkpoint, {})
    assert config["source_cqr_best_epoch"] == 15
    assert config["source_cqr_validation_mean_pinball"] == pytest.approx(
        0.01066008722409606
    )
    assert config["quantile_levels"] == [0.05, 0.5, 0.95]
    assert config["model_forward_passes_per_sample"] == 1
    for key in ("mc_dropout_performed", "repeated_forward_passes"):
        assert config[key] is False
        assert provenance[key] is False
    for key in (
        "truth_used_for_inference",
        "truth_saved_in_predictions",
        "cp_calibration_truth_accessed",
        "decision_development_truth_accessed",
        "conformal_calibration_performed",
        "interval_evaluation_performed",
        "risk_evaluation_performed",
        "cleaning_decision_performed",
        "economic_decision_performed",
        "random_test_accessed",
        "sealed_final_dates_accessed",
        "training_performed",
        "imagenet_download_performed",
    ):
        assert provenance[key] is False
    assert provenance["irradiance_normalization_source_role"] == "TRAIN"
    assert provenance["image_preprocessing"]["random_augmentation"] is False


def test_source_has_no_forbidden_pipeline_dependencies_or_formal_invocation() -> None:
    source = Path(stage3a1.__file__).read_text(encoding="utf-8")
    assert "enable_mc_dropout_only" not in source
    assert "model.train(" not in source
    assert "conformal_quantile" not in source
    assert "qhat" not in source.lower()
    assert "compute_interval_metrics" not in source
    assert "risk_stage2" not in source
    assert "attach_development_values(" not in source
    assert "requests" not in source
    assert "urlopen" not in source
    executable_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "run()" not in executable_lines


def test_import_has_no_formal_side_effects() -> None:
    source = inspect.getsource(stage3a1)
    assert "if __name__ == \"__main__\":" in source
    assert "Importing this module does not read manifests" in source
