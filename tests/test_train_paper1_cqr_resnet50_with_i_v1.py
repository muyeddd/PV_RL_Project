from __future__ import annotations

import inspect
import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiments.train_paper1_cqr_resnet50_with_i_v1 as cqr


class TinyBackbone(nn.Module):
    """Small offline stand-in with the torchvision fc contract."""

    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(3, 4)
        self.fc = nn.Linear(4, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(2, 3))
        return self.fc(self.feature(pooled))


@pytest.fixture
def offline_model(monkeypatch) -> cqr.Paper1CQRResNet50WithI:
    calls: list[object] = []

    def fake_resnet50(*, weights):
        calls.append(weights)
        return TinyBackbone()

    monkeypatch.setattr(cqr, "resnet50", fake_resnet50)
    model = cqr.Paper1CQRResNet50WithI()
    assert calls == [None]
    return model


def point_config() -> dict[str, object]:
    return {
        "protocol": cqr.PROTOCOL,
        "architecture": "ResNet50+irradiance",
        "dropout": cqr.DROPOUT,
        "initialization": "ImageNet pretrained",
        "pretrained_source": "torchvision",
        "pretrained_weight_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "image_preprocessing": cqr.expected_point_preprocessing(),
        "irradiance_normalization": "TRAIN-only z-score",
        "legacy_checkpoint_loaded": False,
    }


def synthetic_point_checkpoint(
    model: cqr.Paper1CQRResNet50WithI,
) -> dict[str, object]:
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith(cqr.SHARED_CHECKPOINT_PREFIXES)
    }
    generator = torch.Generator().manual_seed(17)
    state["regressor.3.weight"] = torch.randn(1, 128, generator=generator)
    state["regressor.3.bias"] = torch.randn(1, generator=generator)
    return {
        "model_state_dict": state,
        "epoch": 26,
        "validation_rmse": 0.075,
        "config": point_config(),
    }


def role_frame(role: str, sample: str, date: str = "2017-06-13") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample,
                "image_path": f"data/raw/PanelImages/{sample}_L_0.2_I_0.4.jpg",
                "date": date,
                "timestamp": f"{date}T10:00:00",
                "role": role,
            }
        ]
    )


def test_ordered_quantiles_and_output_range(offline_model):
    offline_model.train()
    output = offline_model(torch.randn(9, 3, 4, 4), torch.randn(9))
    assert output.shape == (9, 3)
    assert torch.all(output[:, 0] <= output[:, 1])
    assert torch.all(output[:, 1] <= output[:, 2])
    assert torch.all((output >= 0.0) & (output <= 1.0))
    cqr.validate_quantile_outputs(output)


def test_ordering_is_not_post_hoc_sort():
    source = inspect.getsource(cqr.Paper1CQRResNet50WithI.forward)
    assert ".sort" not in source
    assert "softplus" in source


def test_distance_heads_are_exactly_zero_initialized(offline_model):
    for layer in (offline_model.lower_distance, offline_model.upper_distance):
        assert torch.count_nonzero(layer.weight) == 0
        assert torch.count_nonzero(layer.bias) == 0


def test_initial_softplus_distance_is_ln2(offline_model):
    features = torch.randn(6, 128)
    expected = torch.full((6, 1), math.log(2.0))
    assert torch.allclose(F.softplus(offline_model.lower_distance(features)), expected)
    assert torch.allclose(F.softplus(offline_model.upper_distance(features)), expected)


@pytest.mark.parametrize(
    ("quantile", "prediction", "target", "expected"),
    [
        (0.05, 0.2, 1.0, 0.04),
        (0.05, 0.8, 0.2, 0.57),
        (0.50, 0.2, 1.0, 0.40),
        (0.50, 0.8, 0.2, 0.30),
        (0.95, 0.2, 1.0, 0.76),
        (0.95, 0.8, 0.2, 0.03),
    ],
)
def test_pinball_formulas(quantile, prediction, target, expected):
    observed = cqr.pinball_loss(
        torch.tensor([prediction]), torch.tensor([target]), quantile
    )
    assert observed.item() == pytest.approx(expected)


def test_equal_weight_mean_pinball():
    predictions = torch.tensor([[0.1, 0.4, 0.8], [0.2, 0.5, 0.9]])
    targets = torch.tensor([0.3, 0.7])
    mean_loss, parts = cqr.quantile_loss_components(predictions, targets)
    expected = sum(value for value in parts.values()) / 3.0
    assert torch.equal(mean_loss, expected)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1])
def test_invalid_pinball_quantile_rejected(quantile):
    with pytest.raises(ValueError):
        cqr.pinball_loss(torch.tensor([0.2]), torch.tensor([0.3]), quantile)


def test_crossing_count_zero_for_model_and_detects_bad_input(offline_model):
    output = offline_model(torch.randn(5, 3, 2, 2), torch.randn(5))
    assert cqr.quantile_crossing_count(output) == 0
    bad = torch.tensor([[0.2, 0.1, 0.9], [0.1, 0.5, 0.4]])
    assert cqr.quantile_crossing_count(bad) == 2


def test_backbone_is_constructed_with_weights_none(monkeypatch):
    observed = []

    def fake_resnet50(*, weights):
        observed.append(weights)
        return TinyBackbone()

    monkeypatch.setattr(cqr, "resnet50", fake_resnet50)
    cqr.Paper1CQRResNet50WithI()
    assert observed == [None]


def test_no_imagenet_download_metadata():
    config = cqr.build_config()
    assert config["imagenet_weights_at_construction"] is None
    assert config["imagenet_download_performed"] is False


def test_shared_whitelist_prefixes_are_frozen():
    assert cqr.SHARED_CHECKPOINT_PREFIXES == (
        "backbone.",
        "i_branch.",
        "regressor.0.",
    )


@pytest.mark.parametrize(
    "missing_key",
    ["backbone.feature.weight", "i_branch.0.weight", "regressor.0.weight"],
)
def test_required_shared_key_missing_fails(offline_model, missing_key):
    checkpoint = synthetic_point_checkpoint(offline_model)
    del checkpoint["model_state_dict"][missing_key]
    with pytest.raises(ValueError, match="key mismatch"):
        cqr.initialize_from_point_checkpoint(offline_model, checkpoint)


@pytest.mark.parametrize(
    "extra_key",
    ["backbone.unexpected", "i_branch.unexpected", "regressor.0.unexpected"],
)
def test_unexpected_shared_key_fails(offline_model, extra_key):
    checkpoint = synthetic_point_checkpoint(offline_model)
    checkpoint["model_state_dict"][extra_key] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        cqr.initialize_from_point_checkpoint(offline_model, checkpoint)


def test_shared_shape_mismatch_fails(offline_model):
    checkpoint = synthetic_point_checkpoint(offline_model)
    checkpoint["model_state_dict"]["regressor.0.weight"] = torch.zeros(127, 20)
    with pytest.raises(ValueError, match="shape mismatch"):
        cqr.initialize_from_point_checkpoint(offline_model, checkpoint)


@pytest.mark.parametrize("missing_key", cqr.POINT_OUTPUT_KEYS)
def test_old_point_output_missing_fails(offline_model, missing_key):
    checkpoint = synthetic_point_checkpoint(offline_model)
    del checkpoint["model_state_dict"][missing_key]
    with pytest.raises(ValueError, match="Point output key mismatch"):
        cqr.initialize_from_point_checkpoint(offline_model, checkpoint)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("regressor.3.weight", torch.zeros(2, 128)),
        ("regressor.3.bias", torch.zeros(2)),
    ],
)
def test_old_point_output_shape_mismatch_fails(offline_model, key, bad_value):
    checkpoint = synthetic_point_checkpoint(offline_model)
    checkpoint["model_state_dict"][key] = bad_value
    with pytest.raises(ValueError, match="Point output shape mismatch"):
        cqr.initialize_from_point_checkpoint(offline_model, checkpoint)


def test_old_point_output_maps_exactly_to_center(offline_model):
    checkpoint = synthetic_point_checkpoint(offline_model)
    cqr.initialize_from_point_checkpoint(offline_model, checkpoint)
    state = checkpoint["model_state_dict"]
    assert torch.equal(offline_model.center.weight, state["regressor.3.weight"])
    assert torch.equal(offline_model.center.bias, state["regressor.3.bias"])


def test_initialized_q50_reproduces_old_point_prediction(offline_model):
    checkpoint = synthetic_point_checkpoint(offline_model)
    cqr.initialize_from_point_checkpoint(offline_model, checkpoint)
    offline_model.eval()
    images = torch.randn(7, 3, 4, 4)
    irradiance = torch.randn(7)
    with torch.inference_mode():
        features = offline_model.forward_features(images, irradiance)
        old_point = torch.sigmoid(
            F.linear(
                features,
                checkpoint["model_state_dict"]["regressor.3.weight"],
                checkpoint["model_state_dict"]["regressor.3.bias"],
            )
        ).squeeze(1)
        q50 = offline_model(images, irradiance)[:, 1]
    assert torch.allclose(q50, old_point, rtol=0.0, atol=1e-7)


def test_distance_heads_are_reset_after_checkpoint_initialization(offline_model):
    nn.init.ones_(offline_model.lower_distance.weight)
    nn.init.ones_(offline_model.upper_distance.bias)
    cqr.initialize_from_point_checkpoint(
        offline_model, synthetic_point_checkpoint(offline_model)
    )
    assert torch.count_nonzero(offline_model.lower_distance.weight) == 0
    assert torch.count_nonzero(offline_model.upper_distance.bias) == 0


def test_source_checkpoint_sha_mismatch_fails_before_load(tmp_path):
    path = tmp_path / "point.pth"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cqr.load_verified_point_checkpoint(path, "0" * 64)


def test_source_checkpoint_sha_match_and_schema_passes(tmp_path, offline_model):
    path = tmp_path / "point.pth"
    torch.save(synthetic_point_checkpoint(offline_model), path)
    loaded = cqr.load_verified_point_checkpoint(path, cqr.sha256_file(path))
    assert loaded["epoch"] == 26


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("protocol", "other_protocol", "protocol mismatch"),
        ("architecture", "OtherNet", "architecture mismatch"),
        ("dropout", 0.2, "dropout mismatch"),
        ("initialization", "random", "initialization mismatch"),
        ("pretrained_source", "other", "pretrained source mismatch"),
        ("pretrained_weight_enum", "V1", "pretrained weight enum mismatch"),
        ("image_preprocessing", {"bad": True}, "preprocessing mismatch"),
        ("irradiance_normalization", "other", "normalization mismatch"),
        ("legacy_checkpoint_loaded", True, "Legacy"),
    ],
)
def test_point_checkpoint_metadata_mismatch_fails(
    offline_model, field, bad_value, message
):
    checkpoint = synthetic_point_checkpoint(offline_model)
    checkpoint["config"][field] = bad_value
    with pytest.raises(ValueError, match=message):
        cqr.validate_point_checkpoint_schema(checkpoint)


@pytest.mark.parametrize("field", cqr.REQUIRED_CHECKPOINT_FIELDS)
def test_required_top_level_checkpoint_field_missing_fails(offline_model, field):
    checkpoint = synthetic_point_checkpoint(offline_model)
    del checkpoint[field]
    with pytest.raises(ValueError, match="top-level"):
        cqr.validate_point_checkpoint_schema(checkpoint)


def test_training_and_validation_roles_only():
    cqr.validate_training_roles(
        role_frame("TRAIN", "train"),
        role_frame("MODEL_VALIDATION", "validation"),
    )
    with pytest.raises(PermissionError):
        cqr.validate_training_roles(
            role_frame("CP_CALIBRATION", "train"),
            role_frame("MODEL_VALIDATION", "validation"),
        )
    with pytest.raises(PermissionError):
        cqr.validate_training_roles(
            role_frame("TRAIN", "train"),
            role_frame("DECISION_DEVELOPMENT", "validation"),
        )


@pytest.mark.parametrize(
    "name",
    ["cp_calibration.csv", "decision_development.csv", "random_test.csv"],
)
def test_forbidden_role_manifest_rejected(tmp_path, name):
    path = tmp_path / name
    role_frame("RANDOM_TEST", "x").to_csv(path, index=False)
    with pytest.raises(PermissionError):
        cqr.point_train.load_role_manifest(path, "TRAIN")


@pytest.mark.parametrize("sealed_date", sorted(cqr.point_train.SEALED_DATES))
def test_sealed_date_rejected(tmp_path, monkeypatch, sealed_date):
    monkeypatch.setitem(cqr.point_train.EXPECTED_N, "TRAIN", 1)
    path = tmp_path / "train.csv"
    role_frame("TRAIN", "x", sealed_date).to_csv(path, index=False)
    with pytest.raises(PermissionError, match="Sealed"):
        cqr.point_train.load_role_manifest(path, "TRAIN")


def test_role_n_guard_logic(tmp_path, monkeypatch):
    monkeypatch.setitem(cqr.point_train.EXPECTED_N, "TRAIN", 2)
    path = tmp_path / "train.csv"
    role_frame("TRAIN", "x").to_csv(path, index=False)
    with pytest.raises(ValueError, match="N guard"):
        cqr.point_train.load_role_manifest(path, "TRAIN")


def test_expected_role_counts_are_frozen():
    assert cqr.EXPECTED_N == {"TRAIN": 25830, "MODEL_VALIDATION": 3692}


def test_train_only_population_irradiance_statistics():
    train = pd.concat(
        [role_frame("TRAIN", "a"), role_frame("TRAIN", "b")], ignore_index=True
    )
    train["irradiance_raw"] = [0.2, 0.6]
    stats = cqr.point_train.compute_train_irradiance_stats(train)
    assert stats["mean"] == pytest.approx(0.4)
    assert stats["std_ddof0"] == pytest.approx(0.2)
    assert stats["std_ddof0"] == pytest.approx(
        train["irradiance_raw"].to_numpy().std(ddof=0)
    )
    assert stats["std_ddof0"] != pytest.approx(
        train["irradiance_raw"].to_numpy().std(ddof=1)
    )


def test_frozen_stats_reused_and_checked_against_train(tmp_path, monkeypatch):
    train = pd.concat(
        [role_frame("TRAIN", "a"), role_frame("TRAIN", "b")], ignore_index=True
    )
    train["irradiance_raw"] = [0.2, 0.6]
    monkeypatch.setitem(cqr.EXPECTED_N, "TRAIN", 2)
    monkeypatch.setattr(cqr, "EXPECTED_IRRADIANCE_MEAN", 0.4)
    monkeypatch.setattr(cqr, "EXPECTED_IRRADIANCE_STD_DDOF0", 0.2)
    path = tmp_path / "stats.json"
    path.write_text(
        '{"N":2,"mean":0.4,"std_ddof0":0.2,"min":0.2,"max":0.6,'
        '"normalization":"z_score","source_role":"TRAIN"}',
        encoding="utf-8",
    )
    stats = cqr.load_and_validate_frozen_irradiance_stats(path, train)
    validation = role_frame("MODEL_VALIDATION", "v")
    validation["irradiance_raw"] = [0.8]
    normalized = cqr.point_train.normalize_irradiance(validation, stats)
    assert normalized["irradiance"].iloc[0] == pytest.approx(2.0)


@pytest.mark.parametrize("bad_role", ["MODEL_VALIDATION", "CP_CALIBRATION", "DECISION_DEVELOPMENT", "RANDOM_TEST"])
def test_non_train_normalization_source_rejected(tmp_path, bad_role):
    data = role_frame(bad_role, "x")
    data["irradiance_raw"] = [0.4]
    with pytest.raises(PermissionError):
        cqr.load_and_validate_frozen_irradiance_stats(tmp_path / "unused.json", data)


def test_frozen_expected_irradiance_values():
    assert cqr.EXPECTED_IRRADIANCE_MEAN == pytest.approx(0.35047130969460816)
    assert cqr.EXPECTED_IRRADIANCE_STD_DDOF0 == pytest.approx(0.208878529631844)


def test_preprocessing_is_exactly_reused_from_point_baseline():
    assert cqr.build_config()["image_preprocessing"] == cqr.expected_point_preprocessing()
    _, validation = cqr.point_train.build_transforms()
    assert cqr.point_train.validation_transform_is_deterministic(validation)


@pytest.mark.parametrize(
    ("name", "observed"),
    [
        ("BATCH_SIZE", 32),
        ("MAX_EPOCHS", 50),
        ("EARLY_STOPPING_PATIENCE", 8),
        ("LEARNING_RATE", 3e-5),
        ("WEIGHT_DECAY", 1e-4),
        ("DROPOUT", 0.3),
        ("SCHEDULER_FACTOR", 0.5),
        ("SCHEDULER_PATIENCE", 2),
    ],
)
def test_fixed_hyperparameters(name, observed):
    assert getattr(cqr, name) == observed


def test_optimizer_is_single_group_adamw(offline_model):
    optimizer, _ = cqr.build_optimizer_and_scheduler(offline_model)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_reduce_on_plateau_parameters(offline_model):
    _, scheduler = cqr.build_optimizer_and_scheduler(offline_model)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "min"
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 2


def test_scheduler_steps_on_validation_mean_pinball_only():
    class SchedulerSpy:
        def __init__(self):
            self.metrics = []

        def step(self, metric):
            self.metrics.append(metric)

    scheduler = SchedulerSpy()
    cqr.step_scheduler(
        scheduler,
        {"mean_pinball": 0.12, "q50_RMSE": 0.001, "q50_R2": 0.99},
    )
    assert scheduler.metrics == [0.12]


def test_best_selection_uses_validation_mean_pinball_only():
    assert cqr.validation_improved(
        {"mean_pinball": 0.09, "q50_RMSE": 999.0, "q50_R2": -99.0}, 0.1
    )
    assert not cqr.validation_improved(
        {"mean_pinball": 0.11, "q50_RMSE": 0.0, "q50_R2": 1.0}, 0.1
    )


def test_best_checkpoint_schema_contains_pinball_fields(offline_model):
    metrics = {
        "mean_pinball": 0.1,
        "pinball_q05": 0.01,
        "pinball_q50": 0.2,
        "pinball_q95": 0.09,
        "q50_RMSE": 999.0,
    }
    payload = cqr.build_checkpoint_payload(offline_model, 3, metrics, cqr.build_config())
    assert payload["epoch"] == 3
    assert payload["validation_mean_pinball"] == pytest.approx(0.1)
    assert "validation_rmse" not in payload
    assert {"validation_pinball_q05", "validation_pinball_q50", "validation_pinball_q95"} <= set(payload)


@pytest.mark.parametrize(
    "flag",
    [
        "warmup",
        "gradient_clipping",
        "ema",
        "layerwise_learning_rates",
        "discriminative_learning_rates",
        "resume_training",
    ],
)
def test_forbidden_training_features_disabled(flag):
    assert cqr.build_config()[flag] is False


def test_protocol_metadata_has_no_forbidden_experiment_paths():
    provenance = cqr.build_provenance(1.0, 1)
    for key in (
        "cp_calibration_accessed",
        "decision_development_accessed",
        "random_test_accessed",
        "sealed_final_dates_accessed",
        "cqr_conformal_calibration_performed",
        "risk_evaluation_performed",
        "cleaning_decision_performed",
        "economic_decision_performed",
    ):
        assert provenance[key] is False
    assert provenance["training_roles"] == ["TRAIN", "MODEL_VALIDATION"]


def test_output_collision_protection(tmp_path):
    output = tmp_path / "formal"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cqr.point_train.ensure_output_available(output)


def test_tests_do_not_call_formal_run():
    source = Path(__file__).read_text(encoding="utf-8")
    executable_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "cqr.run()" not in executable_lines
