from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.run_paper1_uq_stage1b_intervals_v1 as stage1b


def synthetic_frame(
    role: str,
    n: int = 1,
    *,
    prefix: str | None = None,
    true_l=0.5,
    irradiance=0.5,
    point_pred=0.5,
    mc_mean=0.5,
    mc_std=0.1,
    date: str = "2017-06-13",
) -> pd.DataFrame:
    prefix = prefix or ("cp" if role == stage1b.CP_CALIBRATION_ROLE else "dd")

    def values(value):
        array = np.asarray(value)
        return np.repeat(array, n) if array.ndim == 0 else array

    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}-{index}" for index in range(n)],
            "date": [date] * n,
            "timestamp": [f"{date}T10:{index % 60:02d}:00" for index in range(n)],
            "image_path": [f"data/mock/{prefix}-{index}.jpg" for index in range(n)],
            "role": [role] * n,
            "true_L": values(true_l),
            "irradiance": values(irradiance),
            "point_pred": values(point_pred),
            "mc_mean": values(mc_mean),
            "mc_std": values(mc_std),
        }
    )


def test_random_test_is_rejected_before_file_read(tmp_path: Path) -> None:
    path = tmp_path / "random_test_predictions.csv"
    with pytest.raises(PermissionError, match="RANDOM_TEST"):
        stage1b.validate_stage1a_input_path(path, "RANDOM_TEST")
    assert not path.exists()


@pytest.mark.parametrize("sealed_date", sorted(stage1b.SEALED_FINAL_DATES))
def test_sealed_dates_are_rejected(sealed_date: str) -> None:
    frame = synthetic_frame(stage1b.CP_CALIBRATION_ROLE, date=sealed_date)
    with pytest.raises(PermissionError, match="Sealed final date"):
        stage1b.validate_stage1a_frame(
            frame, stage1b.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )


def test_role_guard_rejects_other_or_mixed_roles() -> None:
    other = synthetic_frame("MODEL_VALIDATION")
    with pytest.raises(PermissionError, match="Role guard"):
        stage1b.validate_stage1a_frame(
            other, stage1b.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )
    mixed = pd.concat(
        [
            synthetic_frame(stage1b.CP_CALIBRATION_ROLE, prefix="a"),
            synthetic_frame(stage1b.DECISION_DEVELOPMENT_ROLE, prefix="b"),
        ],
        ignore_index=True,
    )
    with pytest.raises(PermissionError, match="Role guard"):
        stage1b.validate_stage1a_frame(
            mixed, stage1b.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )


def test_stage1a_schema_guard() -> None:
    frame = synthetic_frame(stage1b.CP_CALIBRATION_ROLE).drop(columns="mc_std")
    with pytest.raises(ValueError, match="schema missing"):
        stage1b.validate_stage1a_frame(
            frame, stage1b.CP_CALIBRATION_ROLE, enforce_expected_n=False
        )


def test_expected_n_guard_logic() -> None:
    stage1b.validate_expected_n(
        pd.DataFrame(index=range(2951)), stage1b.CP_CALIBRATION_ROLE
    )
    stage1b.validate_expected_n(
        pd.DataFrame(index=range(1844)), stage1b.DECISION_DEVELOPMENT_ROLE
    )
    with pytest.raises(ValueError, match="N guard failed"):
        stage1b.validate_expected_n(
            pd.DataFrame(index=range(2950)), stage1b.CP_CALIBRATION_ROLE
        )


def test_protocol_and_fixed_public_parameters() -> None:
    stage1b.validate_protocol("paper1_clean_random_v1")
    with pytest.raises(PermissionError, match="Unauthorized protocol"):
        stage1b.validate_protocol("wrong_protocol")
    assert stage1b.ALPHA == 0.10
    assert stage1b.TARGET_COVERAGE == 0.90
    assert stage1b.MC_Z_90 == 1.645
    assert stage1b.MIN_CALIB_PER_BIN == 30
    assert stage1b.STD_MC_EPSILON == 1e-8


def test_conformal_quantile_has_finite_sample_correction() -> None:
    scores = np.arange(19, dtype=float)
    rank_level = min(np.ceil((len(scores) + 1) * 0.9) / len(scores), 1.0)
    expected = np.quantile(scores, rank_level, method="higher")
    assert stage1b.conformal_quantile(scores) == expected
    assert rank_level == pytest.approx(18 / 19)


def test_conformal_quantile_uses_method_higher(monkeypatch) -> None:
    original = np.quantile
    observed: dict[str, str] = {}

    def capture(values, q, *, method):
        observed["method"] = method
        return original(values, q, method=method)

    monkeypatch.setattr(stage1b.np, "quantile", capture)
    stage1b.conformal_quantile(np.array([0.1, 0.2, 0.3]))
    assert observed["method"] == "higher"


def test_all_intervals_are_clipped_to_unit_range() -> None:
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE,
        n=2,
        mc_mean=[0.01, 0.99],
        mc_std=[1.0, 1.0],
    )
    predictions, _ = stage1b.raw_mc_intervals(decision)
    assert predictions["lower"].tolist() == [0.0, 0.0]
    assert predictions["upper"].tolist() == [1.0, 1.0]
    assert predictions["width"].between(0.0, 1.0).all()


def test_raw_mc_uses_fixed_z_1p645() -> None:
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE, mc_mean=0.5, mc_std=0.1
    )
    predictions, info = stage1b.raw_mc_intervals(decision)
    assert predictions.loc[0, "lower"] == pytest.approx(0.5 - 1.645 * 0.1)
    assert predictions.loc[0, "upper"] == pytest.approx(0.5 + 1.645 * 0.1)
    assert info["conformal_calibration"] is False


def test_split_cp_uses_mc_mean_not_point_pred() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE,
        n=30,
        true_l=0.4,
        mc_mean=0.5,
        point_pred=0.9,
    )
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE,
        mc_mean=0.5,
        point_pred=0.1,
    )
    predictions, info = stage1b.split_cp_intervals(calibration, decision)
    assert info["base_predictor"] == "mc_mean"
    assert info["quantile_method"] == "higher"
    assert info["global_q"] == pytest.approx(0.1)
    assert predictions.loc[0, "lower"] == pytest.approx(0.4)
    assert predictions.loc[0, "upper"] == pytest.approx(0.6)


def test_decision_truth_cannot_alter_split_cp_quantile_or_bounds() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE, n=30, true_l=0.4, mc_mean=0.5
    )
    decision_a = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE, true_l=0.1, mc_mean=0.5
    )
    decision_b = decision_a.assign(true_L=0.9)
    pred_a, info_a = stage1b.split_cp_intervals(calibration, decision_a)
    pred_b, info_b = stage1b.split_cp_intervals(calibration, decision_b)
    assert info_a["global_q"] == info_b["global_q"]
    assert np.array_equal(pred_a[["lower", "upper"]], pred_b[["lower", "upper"]])


def test_fixed_bin_boundary_semantics_right_true_include_lowest() -> None:
    values = [0.0, 0.2, np.nextafter(0.2, 1.0), 0.4, 0.8, 1.0]
    labels = stage1b.assign_fixed_bins(values, stage1b.PRED_L_BINS).astype(object)
    assert labels.tolist() == [
        "[0.0,0.2]",
        "[0.0,0.2]",
        "(0.2,0.4]",
        "(0.2,0.4]",
        "(0.6,0.8]",
        "(0.8,1.0]",
    ]


def test_irradiance_mondrian_uses_fixed_bins() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE,
        n=30,
        true_l=0.4,
        mc_mean=0.5,
        irradiance=0.2,
    )
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE, irradiance=0.2
    )
    predictions, info = stage1b.irradiance_mondrian_intervals(
        calibration, decision
    )
    assert info["bins"] == list(stage1b.IRRADIANCE_BINS)
    assert info["bin_semantics"] == "pd.cut(include_lowest=True,right=True)"
    assert predictions.loc[0, "bin_label"] == "[0.0,0.2]"
    assert not predictions.loc[0, "used_global_fallback"]


def test_pred_l_is_strictly_mc_mean_and_uses_fixed_bins() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE,
        n=30,
        true_l=0.1,
        mc_mean=0.3,
        point_pred=0.9,
    )
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE, mc_mean=0.3, point_pred=0.9
    )
    predictions, info = stage1b.pred_l_mondrian_intervals(
        calibration, decision
    )
    assert info["binning_variable"] == "pred_L=mc_mean"
    assert info["bins"] == list(stage1b.PRED_L_BINS)
    assert info["calibration_bin_counts"]["(0.2,0.4]"] == 30
    assert predictions.loc[0, "bin_label"] == "(0.2,0.4]"


def test_minimum_30_calibration_samples_per_bin() -> None:
    sparse = stage1b.calibrate_mondrian_quantiles(
        np.linspace(0.0, 0.2, 29),
        np.repeat(0.1, 29),
        binning_variable="pred_L=mc_mean",
        bins=stage1b.PRED_L_BINS,
    )
    eligible = stage1b.calibrate_mondrian_quantiles(
        np.linspace(0.0, 0.2, 30),
        np.repeat(0.1, 30),
        binning_variable="pred_L=mc_mean",
        bins=stage1b.PRED_L_BINS,
    )
    assert sparse["q_by_bin"]["[0.0,0.2]"] is None
    assert eligible["q_by_bin"]["[0.0,0.2]"] is not None


def test_sparse_invalid_and_out_of_range_bins_use_global_fallback() -> None:
    calibration = stage1b.calibrate_mondrian_quantiles(
        np.linspace(0.0, 0.2, 29),
        np.repeat(0.1, 29),
        binning_variable="pred_L=mc_mean",
        bins=stage1b.PRED_L_BINS,
    )
    labels, q_used, fallback = stage1b.apply_mondrian_quantiles(
        [0.1, np.nan, -0.1, 1.1], calibration
    )
    assert labels.isna().tolist() == [False, True, True, True]
    assert fallback.tolist() == [True, True, True, True]
    assert q_used == pytest.approx([calibration["global_q"]] * 4)


def test_mc_interval_nonconformity_formula() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE,
        n=3,
        true_l=[0.2, 0.5, 0.8],
        mc_mean=0.5,
        mc_std=0.1,
    )
    scores = stage1b.mc_interval_nonconformity_scores(calibration)
    edge = 0.5 - 1.645 * 0.1
    assert scores == pytest.approx([edge - 0.2, 0.0, edge - 0.2])


def test_std_mc_nonconformity_formula_and_epsilon() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE,
        n=2,
        true_l=[0.4, 0.8],
        mc_mean=[0.5, 0.5],
        mc_std=[0.1, 0.0],
    )
    scores = stage1b.std_mc_nonconformity_scores(calibration)
    assert stage1b.STD_MC_EPSILON == 1e-8
    assert scores == pytest.approx(
        [0.1 / (0.1 + 1e-8), 0.3 / (0.0 + 1e-8)]
    )


def test_std_mc_final_half_width_formula() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE,
        n=30,
        true_l=0.4,
        mc_mean=0.5,
        mc_std=0.1,
    )
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE, mc_mean=0.5, mc_std=0.2
    )
    predictions, info = stage1b.pred_l_std_mc_intervals(calibration, decision)
    expected_half_width = info["q_by_bin"]["(0.4,0.6]"] * (0.2 + 1e-8)
    assert predictions.loc[0, "lower"] == pytest.approx(0.5 - expected_half_width)
    assert predictions.loc[0, "upper"] == pytest.approx(0.5 + expected_half_width)


def test_picp_mpiw_median_width_and_coverage_error() -> None:
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE,
        n=2,
        true_l=[0.2, 0.8],
    )
    predictions = stage1b._finalize_predictions(
        decision, stage1b.RAW_MC, lower=[0.1, 0.1], upper=[0.3, 0.7]
    )
    metrics = stage1b.compute_interval_metrics(predictions)
    assert metrics["PICP"] == pytest.approx(0.5)
    assert metrics["MPIW"] == pytest.approx(0.4)
    assert metrics["median_width"] == pytest.approx(0.4)
    assert metrics["coverage_error"] == pytest.approx(0.4)


def test_standard_interval_score_alpha_0p10_not_wis() -> None:
    scores = stage1b.standard_interval_score(
        truth=[0.2, 0.8], lower=[0.1, 0.1], upper=[0.3, 0.7]
    )
    assert scores == pytest.approx([0.2, 0.6 + 20.0 * 0.1])
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE,
        n=2,
        true_l=[0.2, 0.8],
    )
    predictions = stage1b._finalize_predictions(
        decision, stage1b.RAW_MC, lower=[0.1, 0.1], upper=[0.3, 0.7]
    )
    metrics = stage1b.compute_interval_metrics(predictions)
    assert metrics["mean_interval_score_alpha_0p10"] == pytest.approx(1.4)
    assert "WIS" not in metrics


def test_per_bin_conditional_diagnostics_report_n_picp_mpiw() -> None:
    decision = synthetic_frame(
        stage1b.DECISION_DEVELOPMENT_ROLE,
        n=2,
        true_l=[0.1, 0.9],
        mc_mean=[0.1, 0.9],
        irradiance=[0.1, 0.9],
    )
    predictions, _ = stage1b.raw_mc_intervals(decision)
    diagnostics = stage1b.conditional_coverage_diagnostics(
        {stage1b.RAW_MC: predictions}
    )
    first = diagnostics[
        (diagnostics["binning_variable"] == "pred_L=mc_mean")
        & (diagnostics["bin_label"] == "[0.0,0.2]")
    ].iloc[0]
    assert first["N"] == 1
    assert first["PICP"] == pytest.approx(1.0)
    assert first["MPIW"] > 0


def test_bin_diagnostics_include_counts_quantiles_and_fallbacks() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE,
        n=30,
        true_l=0.4,
        mc_mean=0.5,
        irradiance=0.5,
    )
    decision = synthetic_frame(stage1b.DECISION_DEVELOPMENT_ROLE)
    predictions, quantiles = stage1b.run_all_methods(calibration, decision)
    diagnostics = stage1b.build_bin_diagnostics(
        quantiles, predictions, decision
    )
    assert {
        "calibration_quantile",
        "decision_bin_count",
        "decision_fallback_summary",
    }.issubset(set(diagnostics["diagnostic_type"]))
    assert "fallback_count" in diagnostics
    assert "q_global" in diagnostics
    assert "q_bin" in diagnostics


def test_six_methods_preserve_stage1a_fields_and_add_interval_schema() -> None:
    calibration = synthetic_frame(
        stage1b.CP_CALIBRATION_ROLE,
        n=30,
        true_l=0.4,
        mc_mean=0.5,
        irradiance=0.5,
    )
    decision = synthetic_frame(stage1b.DECISION_DEVELOPMENT_ROLE)
    predictions, _ = stage1b.run_all_methods(calibration, decision)
    assert set(predictions) == set(stage1b.METHOD_OUTPUTS)
    for method, frame in predictions.items():
        assert set(stage1b.STABLE_STAGE1A_COLUMNS).issubset(frame.columns)
        assert set(stage1b.COMMON_INTERVAL_COLUMNS).issubset(frame.columns)
        assert set(frame["method"]) == {method}
        assert set(frame["role"]) == {stage1b.DECISION_DEVELOPMENT_ROLE}


def test_output_collision_protection(tmp_path: Path) -> None:
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage1b.ensure_output_available(output_dir)


def test_formal_output_path_is_fixed(tmp_path: Path) -> None:
    assert stage1b.validate_formal_output_path(stage1b.OUTPUT_DIR) == stage1b.OUTPUT_DIR.resolve()
    with pytest.raises(PermissionError, match="Unauthorized Stage 1B output"):
        stage1b.validate_formal_output_path(tmp_path / "other_output")


def test_all_required_output_filenames_are_fixed() -> None:
    assert set(stage1b.METHOD_OUTPUTS.values()) == {
        "raw_mc_predictions.csv",
        "split_cp_predictions.csv",
        "irradiance_mondrian_predictions.csv",
        "pred_l_mondrian_predictions.csv",
        "pred_l_mc_interval_predictions.csv",
        "pred_l_std_mc_predictions.csv",
    }


def test_writer_creates_separate_complete_stage1b_artifacts(tmp_path: Path) -> None:
    table = pd.DataFrame({"method": ["synthetic"]})
    predictions = {method: table.copy() for method in stage1b.METHOD_OUTPUTS}
    output_dir = tmp_path / "stage1b"
    stage1b.write_stage1b_outputs(
        output_dir,
        predictions,
        metrics=table,
        quantiles={},
        bin_diagnostics=table,
        conditional_diagnostics=table,
        config={},
        provenance={},
    )
    expected = {
        "all_interval_metrics.csv",
        "conformal_quantiles.json",
        "bin_diagnostics.csv",
        "conditional_coverage_diagnostics.csv",
        "config.json",
        "provenance.json",
        *stage1b.METHOD_OUTPUTS.values(),
    }
    assert {path.name for path in output_dir.iterdir()} == expected


def test_only_two_exact_stage1a_inputs_are_authorized() -> None:
    assert set(stage1b.AUTHORIZED_INPUTS) == {
        stage1b.CP_CALIBRATION_ROLE,
        stage1b.DECISION_DEVELOPMENT_ROLE,
    }
    assert stage1b.CP_CALIBRATION_INPUT.name == "cp_calibration_predictions.csv"
    assert (
        stage1b.DECISION_DEVELOPMENT_INPUT.name
        == "decision_development_predictions.csv"
    )
    with pytest.raises(PermissionError, match="Only the sealed Stage 1A"):
        stage1b.validate_stage1a_input_path(
            Path("legacy_mc_predictions.csv"), stage1b.CP_CALIBRATION_ROLE
        )


def test_no_image_model_mc_dropout_optimizer_or_training_code_path() -> None:
    tree = ast.parse(inspect.getsource(stage1b))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not ({"torch", "torchvision", "PIL"} & imported_roots)
    call_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not ({"backward", "zero_grad", "step", "train"} & call_attributes)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "optimizer" not in names
    assert "model" not in names


def test_no_risk_cqr_or_cleaning_execution_functions() -> None:
    function_names = {
        node.name.lower()
        for node in ast.walk(ast.parse(inspect.getsource(stage1b)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_terms = ("risk", "ause", "cqr", "cleaning", "reject")
    assert not any(
        term in function_name
        for function_name in function_names
        for term in forbidden_terms
    )


def test_provenance_records_all_non_actions_and_frozen_definitions() -> None:
    provenance = stage1b.make_provenance()
    assert provenance["split_cp_base_predictor"] == "mc_mean"
    assert provenance["pred_l_definition"] == "mc_mean"
    assert provenance["cp_calibration_n"] == 2951
    assert provenance["decision_development_n"] == 1844
    false_fields = [
        "random_test_accessed",
        "random_test_truth_accessed",
        "random_test_predictions_generated",
        "sealed_final_dates_accessed",
        "legacy_outputs_used",
        "legacy_checkpoint_loaded",
        "image_inference_performed",
        "mc_dropout_performed",
        "training_performed",
        "optimizer_created",
        "model_parameters_updated",
        "risk_screening_performed",
        "cqr_performed",
        "cleaning_decision_performed",
        "decision_truth_used_for_cp_quantile",
        "decision_truth_used_to_optimize_bins",
        "decision_truth_used_to_optimize_alpha",
        "decision_truth_used_to_optimize_epsilon",
    ]
    assert all(provenance[field] is False for field in false_fields)
