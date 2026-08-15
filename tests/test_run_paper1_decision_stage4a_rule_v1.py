"""Synthetic/unit tests for Paper1 Stage 4A rule decisions.

The suite uses only in-memory frames and pytest temporary directories.  It
does not invoke Stage 4A ``run`` or read any frozen formal output.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import run_paper1_decision_stage4a_rule_v1 as stage4a


def _point(n: int = 12) -> pd.DataFrame:
    true_l = np.linspace(0.03, 0.30, n)
    point_pred = np.clip(true_l + np.resize(np.array([-0.02, 0.0, 0.03]), n), 0, 1)
    mc_mean = np.clip(true_l + 0.01, 0, 1)
    result = pd.DataFrame(
        {
            "sample_id": [f"synthetic-{i:04d}" for i in range(n)],
            "date": ["2017-06-01"] * n,
            "timestamp": [f"2017-06-01T00:{i:02d}:00" for i in range(n)],
            "image_path": [f"synthetic/decision/image_{i:04d}.jpg" for i in range(n)],
            "role": [stage4a.EVALUATION_ROLE] * n,
            "true_L": true_l,
            "irradiance": np.linspace(0.1, 0.9, n),
            "point_pred": point_pred,
            "mc_mean": mc_mean,
            "mc_std": np.linspace(0.01, 0.03, n),
            "abs_error_point": np.abs(point_pred - true_l),
            "abs_error_mc_mean": np.abs(mc_mean - true_l),
        }
    )
    return result.loc[:, stage4a.stage1a.PREDICTION_COLUMNS]


def _cqr(n: int = 12) -> pd.DataFrame:
    point = _point(n)
    q50 = np.linspace(0.04, 0.31, n)
    q05 = np.clip(q50 - 0.04, 0, 1)
    q95 = np.clip(q50 + 0.04, 0, 1)
    lower = np.clip(q50 - 0.06, 0, 1)
    upper = np.clip(q50 + 0.06, 0, 1)
    width = upper - lower
    result = pd.DataFrame(
        {
            "sample_id": point["sample_id"],
            "date": point["date"],
            "timestamp": point["timestamp"],
            "image_path": point["image_path"],
            "role": point["role"],
            "true_L": point["true_L"],
            "irradiance": point["irradiance"],
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "method": [stage4a.stage3a2.METHOD] * n,
            "lower": lower,
            "upper": upper,
            "width": width,
            "covered": (point["true_L"].to_numpy() >= lower)
            & (point["true_L"].to_numpy() <= upper),
            "raw_width": q95 - q05,
            "lower_clipped": lower == 0.0,
            "upper_clipped": upper == 1.0,
        }
    )
    return result.loc[:, stage4a.stage3a2.PREDICTION_OUTPUT_COLUMNS]


def _aligned(n: int = 12) -> pd.DataFrame:
    return stage4a.align_decision_inputs(
        _point(n), _cqr(n), enforce_expected_n=False
    )


def _actions(n: int = 12) -> pd.DataFrame:
    return stage4a.build_decision_actions(_aligned(n), enforce_expected_n=False)


def _metrics(n: int = 12) -> pd.DataFrame:
    return stage4a.build_decision_metrics(
        _actions(n), n_samples=n, enforce_expected_n=False
    )


def test_tau_grid_is_exact_and_ordered() -> None:
    assert stage4a.TAU_GRID == (0.05, 0.10, 0.15, 0.20)
    assert list(stage4a.make_config()["tau_grid"]) == [0.05, 0.10, 0.15, 0.20]


def test_reference_tau_is_exact() -> None:
    assert stage4a.REFERENCE_TAU == 0.15
    assert stage4a.make_config()["reference_tau"] == 0.15
    assert stage4a.make_config()["reference_tau_used_for_tuning"] is False


@pytest.mark.parametrize("tau", [-0.1, 0.0, 0.075, 0.25, 1.0])
def test_nondeclared_tau_is_rejected(tau: float) -> None:
    with pytest.raises(ValueError, match="Unauthorized"):
        stage4a.validate_tau(tau)


def test_no_tau_optimization_or_result_sorting() -> None:
    source = inspect.getsource(stage4a)
    forbidden = ("sort_values(", "np.argmax(", "np.argmin(", "GridSearch", "optimize.")
    for fragment in forbidden:
        assert fragment not in source
    provenance = stage4a.make_provenance()
    assert provenance["thresholds_selected_using_decision_results"] is False
    assert provenance["universal_optimal_threshold_claimed"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.149, stage4a.WAIT), (0.15, stage4a.WAIT), (0.151, stage4a.CLEAN)],
)
def test_oracle_strict_greater_rule(value: float, expected: str) -> None:
    assert stage4a.oracle_actions([value], 0.15).tolist() == [expected]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.149, stage4a.WAIT), (0.15, stage4a.WAIT), (0.151, stage4a.CLEAN)],
)
def test_point_strict_greater_rule(value: float, expected: str) -> None:
    assert stage4a.point_threshold_actions([value], 0.15).tolist() == [expected]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.149, stage4a.WAIT), (0.15, stage4a.WAIT), (0.151, stage4a.CLEAN)],
)
def test_q50_strict_greater_rule(value: float, expected: str) -> None:
    assert stage4a.cqr_q50_threshold_actions([value], 0.15).tolist() == [expected]


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (0.151, 0.20, stage4a.CLEAN),
        (0.10, 0.149, stage4a.WAIT),
        (0.10, 0.20, stage4a.REVIEW),
        (0.15, 0.20, stage4a.REVIEW),
        (0.10, 0.15, stage4a.REVIEW),
        (0.15, 0.15, stage4a.REVIEW),
    ],
)
def test_interval_tristate_boundaries(lower: float, upper: float, expected: str) -> None:
    actual = stage4a.cqr_interval_tristate_actions([lower], [upper], 0.15)
    assert actual.tolist() == [expected]


def test_interval_rule_has_no_width_topk_or_fixed_review_budget() -> None:
    source = inspect.getsource(stage4a.cqr_interval_tristate_actions)
    assert "lower_values > threshold" in source
    assert "upper_values < threshold" in source
    for fragment in ("width", "top", "budget", "q50", "random"):
        assert fragment not in source
    provenance = stage4a.make_provenance()
    assert provenance["review_rate_preselected"] is False
    assert provenance["stage3b_risk_budget_used_as_review_rule"] is False


def test_false_clean_numerator_and_denominator() -> None:
    summary = stage4a.summarize_action_group(
        [stage4a.WAIT, stage4a.WAIT, stage4a.CLEAN, stage4a.CLEAN],
        [stage4a.CLEAN, stage4a.REVIEW, stage4a.WAIT, stage4a.CLEAN],
    )
    assert summary["false_clean_n"] == 1
    assert summary["oracle_wait_n"] == 2
    assert summary["false_clean_rate_oracle_wait"] == 0.5


def test_false_clean_zero_denominator_is_nan() -> None:
    summary = stage4a.summarize_action_group(
        [stage4a.CLEAN, stage4a.CLEAN], [stage4a.CLEAN, stage4a.WAIT]
    )
    assert math.isnan(float(summary["false_clean_rate_oracle_wait"]))


def test_missed_clean_numerator_and_denominator() -> None:
    summary = stage4a.summarize_action_group(
        [stage4a.WAIT, stage4a.WAIT, stage4a.CLEAN, stage4a.CLEAN],
        [stage4a.CLEAN, stage4a.REVIEW, stage4a.WAIT, stage4a.CLEAN],
    )
    assert summary["missed_clean_n"] == 1
    assert summary["oracle_clean_n"] == 2
    assert summary["missed_clean_rate_oracle_clean"] == 0.5


def test_missed_clean_zero_denominator_is_nan() -> None:
    summary = stage4a.summarize_action_group(
        [stage4a.WAIT, stage4a.WAIT], [stage4a.CLEAN, stage4a.WAIT]
    )
    assert math.isnan(float(summary["missed_clean_rate_oracle_clean"]))


@pytest.mark.parametrize("oracle", [stage4a.WAIT, stage4a.CLEAN])
def test_review_is_not_counted_as_binary_error(oracle: str) -> None:
    summary = stage4a.summarize_action_group([oracle], [stage4a.REVIEW])
    assert summary["false_clean_n"] == 0
    assert summary["missed_clean_n"] == 0
    assert summary["auto_decision_error_n"] == 0


def test_review_rate_and_diagnostic_review_counts() -> None:
    summary = stage4a.summarize_action_group(
        [stage4a.WAIT, stage4a.CLEAN, stage4a.CLEAN, stage4a.WAIT],
        [stage4a.REVIEW, stage4a.REVIEW, stage4a.CLEAN, stage4a.WAIT],
    )
    assert summary["review_n"] == 2
    assert summary["pred_review_n"] == 2
    assert summary["review_rate"] == 0.5
    assert summary["review_oracle_clean_n"] == 1
    assert summary["review_oracle_wait_n"] == 1


def test_auto_coverage_is_one_minus_review_rate() -> None:
    summary = stage4a.summarize_action_group(
        [stage4a.WAIT, stage4a.CLEAN, stage4a.CLEAN, stage4a.WAIT],
        [stage4a.REVIEW, stage4a.REVIEW, stage4a.CLEAN, stage4a.WAIT],
    )
    assert summary["auto_decision_n"] == 2
    assert summary["auto_decision_coverage"] == 0.5
    assert summary["auto_decision_coverage"] == 1 - summary["review_rate"]


def test_auto_error_count_and_rate() -> None:
    summary = stage4a.summarize_action_group(
        [stage4a.WAIT, stage4a.WAIT, stage4a.CLEAN, stage4a.CLEAN],
        [stage4a.CLEAN, stage4a.REVIEW, stage4a.WAIT, stage4a.CLEAN],
    )
    assert summary["auto_decision_error_n"] == 2
    assert summary["auto_decision_n"] == 3
    assert summary["auto_decision_error_rate"] == pytest.approx(2 / 3)
    assert summary["auto_correct_n"] == 1


def test_auto_error_zero_denominator_is_nan() -> None:
    summary = stage4a.summarize_action_group(
        [stage4a.WAIT, stage4a.CLEAN], [stage4a.REVIEW, stage4a.REVIEW]
    )
    assert summary["auto_decision_n"] == 0
    assert math.isnan(float(summary["auto_decision_error_rate"]))


@pytest.mark.parametrize("method", [stage4a.POINT_THRESHOLD, stage4a.CQR_Q50_THRESHOLD])
def test_binary_baselines_have_no_review_and_full_auto_coverage(method: str) -> None:
    metrics = _metrics()
    rows = metrics.loc[metrics["method"] == method]
    assert (rows["review_n"] == 0).all()
    assert (rows["review_rate"] == 0.0).all()
    assert (rows["auto_decision_coverage"] == 1.0).all()


def test_expected_n_guard_is_1844() -> None:
    assert stage4a.EXPECTED_N == 1844
    with pytest.raises(ValueError, match="1844"):
        stage4a.align_decision_inputs(_point(), _cqr())


@pytest.mark.parametrize("role", ["CP_CALIBRATION", "TRAIN", "RANDOM_TEST"])
def test_nondecision_point_roles_are_rejected(role: str) -> None:
    point = _point()
    point["role"] = role
    with pytest.raises(PermissionError):
        stage4a.validate_point_predictions(point, enforce_expected_n=False)


def test_cp_cqr_role_is_rejected() -> None:
    cqr = _cqr()
    cqr["role"] = "CP_CALIBRATION"
    with pytest.raises(PermissionError):
        stage4a.validate_cqr_predictions(cqr, enforce_expected_n=False)


@pytest.mark.parametrize("case", ["sealed_date", "random_test", "sealed_locator"])
def test_sealed_dates_and_random_test_are_rejected(case: str) -> None:
    frame = _aligned()
    if case == "sealed_date":
        frame["date"] = "2017-06-15"
    elif case == "random_test":
        frame["image_path"] = [
            f"synthetic/random_test/image_{index}.jpg" for index in range(len(frame))
        ]
    else:
        frame["image_path"] = [
            f"synthetic/2017-06-24/image_{index}.jpg" for index in range(len(frame))
        ]
    with pytest.raises(PermissionError):
        stage4a.validate_aligned_inputs(
            frame, enforce_expected_n=False
        )


def test_duplicate_sample_id_is_rejected() -> None:
    point = _point()
    point.loc[1, "sample_id"] = point.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="sample_id"):
        stage4a.validate_point_predictions(point, enforce_expected_n=False)


def test_alignment_uses_sample_set_not_row_order() -> None:
    cqr = _cqr().iloc[::-1].reset_index(drop=True)
    aligned = stage4a.align_decision_inputs(
        _point(), cqr, enforce_expected_n=False
    )
    assert aligned["sample_id"].tolist() == _point()["sample_id"].tolist()
    cqr.loc[0, "sample_id"] = "synthetic-extra"
    with pytest.raises(ValueError, match="sample_id set mismatch"):
        stage4a.align_decision_inputs(_point(), cqr, enforce_expected_n=False)


@pytest.mark.parametrize("field", ["image_path", "date", "timestamp", "true_L"])
def test_alignment_fields_are_exactly_guarded(field: str) -> None:
    cqr = _cqr()
    if field == "true_L":
        cqr.loc[0, field] += 1e-6
        cqr["covered"] = (cqr["true_L"] >= cqr["lower"]) & (
            cqr["true_L"] <= cqr["upper"]
        )
    elif field == "date":
        cqr.loc[0, field] = "2017-06-02"
    else:
        cqr.loc[0, field] = "synthetic-mismatch"
    with pytest.raises(ValueError, match=field):
        stage4a.align_decision_inputs(_point(), cqr, enforce_expected_n=False)


def test_irradiance_alignment_is_guarded() -> None:
    cqr = _cqr()
    cqr.loc[0, "irradiance"] += 1e-4
    with pytest.raises(ValueError, match="irradiance"):
        stage4a.align_decision_inputs(_point(), cqr, enforce_expected_n=False)


@pytest.mark.parametrize("field", ["q05", "q50", "q95"])
def test_nonfinite_quantiles_are_rejected(field: str) -> None:
    cqr = _cqr()
    cqr.loc[0, field] = np.nan
    with pytest.raises(ValueError):
        stage4a.validate_cqr_predictions(cqr, enforce_expected_n=False)


def test_quantile_order_is_guarded() -> None:
    cqr = _cqr()
    cqr.loc[0, "q05"] = cqr.loc[0, "q50"] + 0.01
    cqr.loc[0, "raw_width"] = cqr.loc[0, "q95"] - cqr.loc[0, "q05"]
    with pytest.raises(ValueError):
        stage4a.validate_cqr_predictions(cqr, enforce_expected_n=False)


@pytest.mark.parametrize(("field", "value"), [("lower", np.nan), ("upper", np.inf)])
def test_nonfinite_interval_bounds_are_rejected(field: str, value: float) -> None:
    cqr = _cqr()
    cqr.loc[0, field] = value
    with pytest.raises(ValueError):
        stage4a.validate_cqr_predictions(cqr, enforce_expected_n=False)


@pytest.mark.parametrize(("field", "value"), [("lower", -0.01), ("upper", 1.01)])
def test_interval_bounds_must_lie_in_unit_range(field: str, value: float) -> None:
    cqr = _cqr()
    cqr.loc[0, field] = value
    cqr.loc[0, "width"] = cqr.loc[0, "upper"] - cqr.loc[0, "lower"]
    with pytest.raises(ValueError):
        stage4a.validate_cqr_predictions(cqr, enforce_expected_n=False)


def test_lower_must_not_exceed_upper() -> None:
    cqr = _cqr()
    cqr.loc[0, ["lower", "upper", "width"]] = [0.2, 0.1, -0.1]
    with pytest.raises(ValueError):
        stage4a.validate_cqr_predictions(cqr, enforce_expected_n=False)


@pytest.mark.parametrize("field", ["point_pred", "true_L", "irradiance"])
def test_deterministic_point_artifact_values_must_be_finite(field: str) -> None:
    point = _point()
    point.loc[0, field] = np.nan
    with pytest.raises(ValueError):
        stage4a.validate_point_predictions(point, enforce_expected_n=False)


def test_point_source_and_column_are_frozen_stage1a_values() -> None:
    assert stage4a.POINT_PREDICTIONS_INPUT == (
        stage4a.stage1a.OUTPUT_DIR / "decision_development_predictions.csv"
    )
    assert "point_pred" in stage4a.stage1a.PREDICTION_COLUMNS
    config = stage4a.make_config()
    assert config["stage1a_deterministic_point_column"] == "point_pred"
    assert config["stage1a_mc_mean_used_as_point_prediction"] is False


def test_cqr_source_is_frozen_stage3a2_predictions() -> None:
    assert stage4a.CQR_PREDICTIONS_INPUT == (
        stage4a.stage3a2.OUTPUT_DIR / "cqr_predictions.csv"
    )
    assert stage4a.make_config()["stage3a2_interval_recomputed"] is False


def test_actions_have_fixed_method_and_tau_order() -> None:
    actions = _actions()
    assert tuple(dict.fromkeys(actions["method"])) == stage4a.METHOD_ORDER
    for method in stage4a.METHOD_ORDER:
        rows = actions.loc[actions["method"] == method]
        assert tuple(dict.fromkeys(rows["tau"])) == stage4a.TAU_GRID


def test_metrics_have_fixed_method_and_tau_order() -> None:
    metrics = _metrics()
    assert tuple(dict.fromkeys(metrics["method"])) == stage4a.METHOD_ORDER
    for method in stage4a.METHOD_ORDER:
        assert tuple(metrics.loc[metrics["method"] == method, "tau"]) == stage4a.TAU_GRID


def test_outputs_have_no_ranking_or_winner_fields() -> None:
    metrics = _metrics()
    lowered = {str(column).lower() for column in metrics.columns}
    assert not (lowered & stage4a.FORBIDDEN_PRESENTATION_FIELDS)
    source = inspect.getsource(stage4a)
    assert "sort_values(" not in source


def test_long_format_synthetic_row_count() -> None:
    n = 12
    actions = _actions(n)
    assert len(actions) == n * 4 * 3
    assert not actions.duplicated(["sample_id", "method", "tau"]).any()


def test_formal_long_format_row_count_is_22128() -> None:
    assert stage4a.EXPECTED_N * len(stage4a.TAU_GRID) * len(stage4a.METHOD_ORDER) == 22128
    assert stage4a.make_config()["expected_action_rows"] == 22128


def test_action_and_metric_schemas_are_exact() -> None:
    assert tuple(_actions().columns) == stage4a.ACTION_COLUMNS
    metrics = _metrics()
    assert tuple(metrics.columns) == stage4a.METRIC_COLUMNS
    assert tuple(stage4a.build_decision_counts(metrics).columns) == stage4a.COUNT_COLUMNS


def test_output_collision_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "stage4a"
    output.mkdir()
    (output / "existing.txt").write_text("synthetic", encoding="utf-8")
    with pytest.raises(FileExistsError):
        stage4a.ensure_output_available(output)


def test_synthetic_writer_creates_only_five_required_files(tmp_path: Path) -> None:
    actions = _actions()
    metrics = _metrics()
    output = tmp_path / "stage4a"
    stage4a.write_outputs(
        output,
        actions,
        metrics,
        stage4a.build_decision_counts(metrics),
        stage4a.make_config(),
        stage4a.make_provenance(),
    )
    assert {path.name for path in output.iterdir()} == {
        "decision_actions.csv",
        "decision_metrics.csv",
        "decision_counts.csv",
        "config.json",
        "provenance.json",
    }


@pytest.mark.parametrize(
    "path",
    [Path("synthetic/random_test.csv"), Path("synthetic/cp_calibration.csv")],
)
def test_forbidden_paths_are_rejected_before_io(path: Path) -> None:
    with pytest.raises(PermissionError):
        stage4a.validate_authorized_input_path(path, "stage1a_point_predictions")


def test_unauthorized_path_is_rejected() -> None:
    with pytest.raises(PermissionError):
        stage4a.validate_authorized_input_path(
            Path("synthetic/decision.csv"), "stage1a_point_predictions"
        )


@pytest.mark.parametrize(
    "field",
    [
        "thresholds_are_scenarios",
        "decision_safety_must_be_interpreted_with_auto_coverage",
    ],
)
def test_required_true_provenance_flags(field: str) -> None:
    assert stage4a.make_provenance()[field] is True


@pytest.mark.parametrize(
    "field",
    [
        "universal_optimal_threshold_claimed",
        "thresholds_selected_using_decision_results",
        "review_rate_preselected",
        "stage3b_risk_budget_used_as_review_rule",
        "nominal_interval_coverage_implies_decision_accuracy",
        "training_performed",
        "image_inference_performed",
        "mc_dropout_performed",
        "conformal_recalibration_performed",
        "risk_score_development_performed",
        "economic_analysis_performed",
        "random_test_accessed",
        "sealed_final_dates_accessed",
        "formal_decision_method_selected",
        "formal_winner_declared",
    ],
)
def test_required_false_provenance_flags(field: str) -> None:
    assert stage4a.make_provenance()[field] is False


def test_provenance_rules_are_explicit() -> None:
    provenance = stage4a.make_provenance()
    assert provenance["oracle_rule"] == "true_L > tau => CLEAN else WAIT"
    assert provenance["point_rule"] == "point_pred > tau => CLEAN else WAIT"
    assert provenance["cqr_q50_rule"] == "q50 > tau => CLEAN else WAIT"
    assert provenance["cqr_interval_rule"] == (
        "lower > tau => CLEAN; upper < tau => WAIT; otherwise REVIEW"
    )


def test_threshold_physical_quantity_boundary_is_explicit() -> None:
    provenance = stage4a.make_provenance()
    assert provenance["threshold_variable"] == "DeepSolarEye relative power loss L"
    assert provenance["literature_reference_variable"] == (
        "temperature-corrected performance ratio"
    )
    assert provenance["threshold_variables_are_physically_equivalent"] is False
    assert provenance["all_tau_reported_at_equal_status"] is True


def test_module_contains_no_training_inference_recalibration_economics_or_rl() -> None:
    source = inspect.getsource(stage4a)
    forbidden = (
        "torch.",
        "model(",
        "predict_deterministic(",
        "predict_mc_dropout(",
        "calibrate_cqr(",
        "conformalize_decision_intervals(",
        "electricity_price",
        "regret",
        "CVaR",
        "reinforcement_learning",
    )
    for fragment in forbidden:
        assert fragment not in source


def test_synthetic_tests_do_not_read_formal_outputs_or_invoke_run() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "outputs/" + "paper1_clean_random_v1" not in source
    assert "pd." + "read_csv(" not in source
    assert "stage4a." + "run(" not in source
