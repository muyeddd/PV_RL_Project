from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import experiments.run_paper1_uq_stage1a_inference_v1 as stage1a


def manifest_frame(
    role: str,
    sample: str = "sample-a",
    date: str = "2017-06-13",
    true_l: float = 0.2,
    irradiance: float = 0.4,
) -> pd.DataFrame:
    locator = (
        f"data/raw/PanelImages/{sample}_L_{true_l}_I_{irradiance}.jpg"
    )
    return pd.DataFrame(
        [
            {
                "sample_id": sample,
                "image_path": locator,
                "date": date,
                "timestamp": f"{date}T10:00:00",
                "role": role,
            }
        ]
    )


class ModeSpyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root_train_calls: list[bool] = []
        self.linear = nn.Linear(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(2)

    def train(self, mode: bool = True):
        self.root_train_calls.append(mode)
        return super().train(mode)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(self.dropout(self.linear(values)))


def test_current_protocol_guard() -> None:
    stage1a.validate_protocol("paper1_clean_random_v1")
    with pytest.raises(PermissionError, match="Unauthorized protocol"):
        stage1a.validate_protocol("another_protocol")


def test_random_test_role_is_rejected_before_any_read(tmp_path: Path) -> None:
    forbidden = tmp_path / "random_test.csv"
    with pytest.raises(PermissionError, match="Forbidden inference role"):
        stage1a.validate_manifest_authorization(forbidden, "RANDOM_TEST")
    assert not forbidden.exists()


@pytest.mark.parametrize("sealed_date", sorted(stage1a.SEALED_FINAL_DATES))
def test_sealed_dates_are_rejected(sealed_date: str) -> None:
    with pytest.raises(PermissionError, match="Sealed final date"):
        stage1a.validate_manifest_frame(
            manifest_frame("CP_CALIBRATION", date=sealed_date),
            "CP_CALIBRATION",
        )


def test_random_test_and_sealed_date_locators_are_rejected() -> None:
    random_locator = manifest_frame("CP_CALIBRATION")
    random_locator.loc[0, "image_path"] = "data/RANDOM_TEST/a_L_0.2_I_0.4.jpg"
    with pytest.raises(PermissionError, match="RANDOM_TEST locator"):
        stage1a.validate_manifest_frame(random_locator, "CP_CALIBRATION")

    sealed_locator = manifest_frame("CP_CALIBRATION")
    sealed_locator.loc[0, "image_path"] = (
        "data/raw/2017-06-15/a_L_0.2_I_0.4.jpg"
    )
    with pytest.raises(PermissionError, match="Sealed final date locator"):
        stage1a.validate_manifest_frame(sealed_locator, "CP_CALIBRATION")


def test_legacy_checkpoint_is_rejected() -> None:
    with pytest.raises(PermissionError, match="Legacy checkpoint"):
        stage1a.validate_checkpoint_path(stage1a.LEGACY_CHECKPOINT)
    with pytest.raises(PermissionError, match="Legacy checkpoint"):
        stage1a.validate_checkpoint_path(Path("elsewhere/best_resnet50_with_i.pth"))


def test_checkpoint_metadata_guards_clean_identity() -> None:
    metadata = {
        "epoch": 26,
        "model_state_dict": {},
        "config": {
            "protocol": stage1a.PROTOCOL,
            "architecture": stage1a.ARCHITECTURE,
            "seed": 42,
            "training_role": "TRAIN",
            "selection_role": "MODEL_VALIDATION",
            "legacy_checkpoint_loaded": False,
        },
    }
    stage1a.validate_checkpoint_metadata(metadata)
    metadata["epoch"] = 25
    with pytest.raises(ValueError, match="metadata guard"):
        stage1a.validate_checkpoint_metadata(metadata)


def test_deterministic_mode_keeps_dropout_in_eval() -> None:
    model = ModeSpyNet()
    model.dropout.train()
    stage1a.set_deterministic_inference_mode(model)
    assert not model.training
    assert not model.dropout.training
    assert not model.batch_norm.training
    assert not any(module.training for module in model.modules())


def test_mc_mode_activates_only_dropout_and_preserves_batchnorm() -> None:
    model = ModeSpyNet()
    stage1a.enable_mc_dropout_only(model)
    assert not model.training
    assert model.dropout.training
    assert not model.linear.training
    assert not model.batch_norm.training
    assert all(
        module.training == isinstance(module, nn.Dropout)
        for module in model.modules()
    )


def test_mc_mode_never_calls_whole_model_train_true() -> None:
    model = ModeSpyNet()
    stage1a.enable_mc_dropout_only(model)
    assert model.root_train_calls
    assert True not in model.root_train_calls


def test_mc_passes_are_fixed_at_50_and_summary_uses_sample_std(monkeypatch) -> None:
    model = ModeSpyNet()
    calls = 0

    def fake_predict_once(model_arg, loader_arg, device_arg):
        nonlocal calls
        value = calls
        calls += 1
        return np.array([value, value + 100.0], dtype=np.float64)

    monkeypatch.setattr(stage1a, "_predict_once", fake_predict_once)
    mc_mean, mc_std = stage1a.predict_mc_dropout(
        model, loader=None, device=torch.device("cpu")
    )
    expected = np.arange(50, dtype=np.float64)
    assert stage1a.MC_PASSES == 50
    assert calls == 50
    assert mc_mean == pytest.approx([expected.mean(), expected.mean() + 100.0])
    assert mc_std == pytest.approx([expected.std(ddof=1), expected.std(ddof=1)])
    assert stage1a.MC_STD_DDOF == 1


def test_accumulator_rejects_any_pass_count_other_than_50() -> None:
    accumulator = stage1a.MCSummaryAccumulator()
    for value in range(49):
        accumulator.update(np.array([value], dtype=float))
    with pytest.raises(ValueError, match="Exactly 50"):
        accumulator.finalize()


def test_point_pred_and_mc_mean_are_distinct_output_fields() -> None:
    records = stage1a.prepare_records(
        manifest_frame("CP_CALIBRATION"),
        "CP_CALIBRATION",
        stage1a.EXPECTED_IRRADIANCE_MEAN,
        stage1a.EXPECTED_IRRADIANCE_STD,
    )
    output = stage1a.build_prediction_frame(
        records,
        point_pred=np.array([0.1]),
        mc_mean=np.array([0.3]),
        mc_std=np.array([0.05]),
    )
    assert list(output.columns) == list(stage1a.PREDICTION_COLUMNS)
    assert output.loc[0, "point_pred"] == pytest.approx(0.1)
    assert output.loc[0, "mc_mean"] == pytest.approx(0.3)
    assert output.loc[0, "abs_error_point"] == pytest.approx(0.1)
    assert output.loc[0, "abs_error_mc_mean"] == pytest.approx(0.1)
    assert output.loc[0, "irradiance"] == pytest.approx(0.4)


def test_irradiance_stats_must_be_train_only_and_are_not_reestimated() -> None:
    valid = {
        "source_role": "TRAIN",
        "normalization": "z_score",
        "mean": stage1a.EXPECTED_IRRADIANCE_MEAN,
        "std_ddof0": stage1a.EXPECTED_IRRADIANCE_STD,
    }
    assert stage1a.validate_irradiance_stats(valid) == pytest.approx(
        (stage1a.EXPECTED_IRRADIANCE_MEAN, stage1a.EXPECTED_IRRADIANCE_STD)
    )
    for forbidden_role in ("CP_CALIBRATION", "DECISION_DEVELOPMENT"):
        invalid = dict(valid, source_role=forbidden_role)
        with pytest.raises(PermissionError, match="TRAIN only"):
            stage1a.validate_irradiance_stats(invalid)


def test_cp_and_decision_outputs_are_written_separately(tmp_path: Path) -> None:
    cp = pd.DataFrame({"role": ["CP_CALIBRATION"], "point_pred": [0.1]})
    decision = pd.DataFrame(
        {"role": ["DECISION_DEVELOPMENT"], "point_pred": [0.2]}
    )
    output_dir = tmp_path / "stage1a"
    stage1a.write_stage1a_outputs(output_dir, cp, decision, {}, {})
    cp_written = pd.read_csv(output_dir / "cp_calibration_predictions.csv")
    decision_written = pd.read_csv(
        output_dir / "decision_development_predictions.csv"
    )
    assert set(cp_written["role"]) == {"CP_CALIBRATION"}
    assert set(decision_written["role"]) == {"DECISION_DEVELOPMENT"}
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "config.json",
        "cp_calibration_predictions.csv",
        "decision_development_predictions.csv",
        "provenance.json",
    ]


def test_output_collision_protection(tmp_path: Path) -> None:
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage1a.ensure_output_available(output_dir)


def test_no_optimizer_or_training_operation_code_path() -> None:
    tree = ast.parse(inspect.getsource(stage1a))
    forbidden_calls = {"backward", "zero_grad", "step"}
    call_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (call_names & forbidden_calls)
    assert "optimizer" not in names


def test_no_cp_risk_cqr_or_cleaning_execution_path() -> None:
    function_names = {
        node.name
        for node in ast.walk(ast.parse(inspect.getsource(stage1a)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_terms = ("conformal", "risk", "cqr", "cleaning")
    assert not any(
        term in name.lower() for name in function_names for term in forbidden_terms
    )
    provenance = stage1a.make_provenance(
        stage1a.CLEAN_CHECKPOINT,
        stage1a.EXPECTED_IRRADIANCE_MEAN,
        stage1a.EXPECTED_IRRADIANCE_STD,
    )
    assert provenance["conformal_calibration_performed"] is False
    assert provenance["risk_screening_performed"] is False
    assert provenance["cqr_performed"] is False
    assert provenance["cleaning_decision_performed"] is False
    assert provenance["training_performed"] is False
    assert provenance["optimizer_created"] is False
    assert provenance["model_parameters_updated"] is False


def test_provenance_contains_required_stage1a_audit_fields() -> None:
    provenance = stage1a.make_provenance(
        stage1a.CLEAN_CHECKPOINT,
        stage1a.EXPECTED_IRRADIANCE_MEAN,
        stage1a.EXPECTED_IRRADIANCE_STD,
    )
    assert provenance["roles_used"] == [
        "CP_CALIBRATION",
        "DECISION_DEVELOPMENT",
    ]
    assert provenance["random_test_accessed"] is False
    assert provenance["random_test_truth_accessed"] is False
    assert provenance["random_test_predictions_generated"] is False
    assert provenance["sealed_final_dates_accessed"] is False
    assert provenance["legacy_checkpoint_loaded"] is False
    assert provenance["mc_passes"] == 50
    assert provenance["mc_std_ddof"] == 1
    assert provenance["irradiance_stats_source"] == "TRAIN_only"


def test_default_manifests_are_only_the_two_authorized_roles() -> None:
    assert set(stage1a.AUTHORIZED_MANIFESTS) == {
        "CP_CALIBRATION",
        "DECISION_DEVELOPMENT",
    }
    assert stage1a.CP_CALIBRATION_MANIFEST.name == "cp_calibration.csv"
    assert (
        stage1a.DECISION_DEVELOPMENT_MANIFEST.name
        == "decision_development.csv"
    )
    assert all(path.name != "random_test.csv" for path in stage1a.AUTHORIZED_MANIFESTS.values())
