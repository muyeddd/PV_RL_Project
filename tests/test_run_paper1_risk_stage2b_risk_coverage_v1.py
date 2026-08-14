from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.run_paper1_risk_stage2a_score_validity_v1 as stage2a
import experiments.run_paper1_risk_stage2b_risk_coverage_v1 as stage2b


def synthetic_base(n: int = 8, *, role: str = stage2b.PRIMARY_ROLE) -> pd.DataFrame:
    true_l = np.linspace(0.1, 0.8, n)
    mc_mean = np.repeat(0.4, n)
    return pd.DataFrame(
        {
            "sample_id": [f"dd-{index}" for index in range(n)],
            "date": ["2017-06-13"] * n,
            "timestamp": [f"2017-06-13T10:{index:02d}:00" for index in range(n)],
            "image_path": [f"data/mock/dd-{index}.jpg" for index in range(n)],
            "role": [role] * n,
            "true_L": true_l,
            "irradiance": np.linspace(0.1, 0.9, n),
            "point_pred": np.repeat(0.9, n),
            "mc_mean": mc_mean,
            "mc_std": np.linspace(0.01, 0.08, n),
        }
    )


def synthetic_interval(
    base: pd.DataFrame, risk_score: str, widths=None
) -> pd.DataFrame:
    n = len(base)
    widths = np.linspace(0.1, 0.3, n) if widths is None else np.asarray(widths)
    if widths.ndim == 0:
        widths = np.repeat(widths, n)
    lower = 0.5 - widths / 2.0
    upper = 0.5 + widths / 2.0
    result = base.loc[:, stage2a.STAGE1A_REQUIRED_COLUMNS].copy()
    result["method"] = stage2a.STAGE1B_INPUT_SPECS[risk_score][1]
    result["lower"] = lower
    result["upper"] = upper
    result["width"] = upper - lower
    truth = result["true_L"].to_numpy(dtype=float)
    result["covered"] = (truth >= lower) & (truth <= upper)
    return result


def all_intervals(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        risk_score: synthetic_interval(base, risk_score)
        for risk_score in stage2b.RISK_SCORE_ORDER[1:]
    }


def synthetic_evaluation(
    n: int = 20,
    *,
    abs_error=None,
    risk=None,
    role: str = stage2b.PRIMARY_ROLE,
) -> pd.DataFrame:
    errors = (
        np.linspace(0.01, 0.20, n)
        if abs_error is None
        else np.asarray(abs_error, dtype=float)
    )
    base_risk = (
        np.linspace(0.001, 0.100, n)
        if risk is None
        else np.asarray(risk, dtype=float)
    )
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(n)],
            "date": ["2017-06-13"] * n,
            "role": [role] * n,
            "true_L": errors,
            "mc_mean": np.zeros(n),
            "signed_error_mc_mean": errors,
            "abs_error_mc_mean": errors,
            "sq_error_mc_mean": np.square(errors),
        }
    )
    for offset, risk_score in enumerate(stage2b.RISK_SCORE_ORDER):
        frame[risk_score] = base_risk + offset * 0.001
    return frame


def retention(score, errors, target, *, rounded=True):
    errors = np.asarray(errors, dtype=float)
    return stage2b.tie_aware_retention_metrics(
        score,
        errors,
        np.square(errors),
        target,
        use_stage2a_risk_rounding=rounded,
    )


def toy_curves(method_mae, method_rmse, oracle_mae, oracle_rmse) -> pd.DataFrame:
    records = []
    for risk_score in stage2b.CURVE_SCORE_ORDER:
        is_oracle = risk_score == stage2b.ORACLE
        mae_values = oracle_mae if is_oracle else method_mae
        rmse_values = oracle_rmse if is_oracle else method_rmse
        for coverage, mae, rmse in zip(
            stage2b.COVERAGE_GRID, mae_values, rmse_values
        ):
            records.append(
                {
                    "risk_score": risk_score,
                    "coverage_requested": coverage,
                    "MAE_mc_mean": mae,
                    "RMSE_mc_mean": rmse,
                }
            )
    return pd.DataFrame(records)


def test_fixed_coverage_grid_has_exactly_19_points() -> None:
    stage2b.validate_coverage_grid(stage2b.COVERAGE_GRID)
    assert len(stage2b.COVERAGE_GRID) == 19
    assert stage2b.COVERAGE_GRID[0] == 0.10
    assert stage2b.COVERAGE_GRID[-1] == 1.00
    with pytest.raises(ValueError, match="fixed 19-point"):
        stage2b.validate_coverage_grid(stage2b.COVERAGE_GRID[:-1])


def test_floor_retained_count_rule() -> None:
    assert stage2b.target_retained_count(0.10, 1844) == 184
    assert stage2b.target_retained_count(0.15, 1844) == 276
    assert stage2b.target_retained_count(1.00, 1844) == 1844
    assert stage2b.target_retained_count(0.10, 3) == 1


def test_coverage_one_retains_all_samples_and_overall_metrics() -> None:
    errors = np.array([0.1, 0.2, 0.4, 0.8])
    result = retention([4, 3, 2, 1], errors, 4)
    assert result["target_retained_n"] == 4
    assert result["expected_retained_n"] == pytest.approx(4)
    assert result["MAE_mc_mean"] == pytest.approx(errors.mean())
    assert result["RMSE_mc_mean"] == pytest.approx(np.sqrt(np.mean(errors**2)))


def test_low_risk_is_retained_first() -> None:
    result = retention([3, 0, 2, 1], [30, 1, 20, 10], 2)
    assert result["MAE_mc_mean"] == pytest.approx((1 + 10) / 2)


def test_stage2a_12_decimal_risk_tie_helper_is_reused() -> None:
    assert stage2b.RISK_TIE_ROUND_DECIMALS == stage2a.RISK_TIE_ROUND_DECIMALS == 12
    assert stage2b.stage2a.risk_rank_values is stage2a.risk_rank_values


def test_numerical_near_ties_are_merged() -> None:
    risk = [
        0.1771004725719999,
        0.177100472572,
        0.1771004725720001,
    ]
    result = retention(risk, [1, 2, 3], 1)
    assert result["boundary_tie_group_size"] == 3
    assert result["boundary_fraction"] == pytest.approx(1 / 3)
    assert result["MAE_mc_mean"] == pytest.approx(2.0)


def test_meaningful_risk_differences_are_preserved() -> None:
    result = retention(
        [0.177100472572, 0.177100472600], [1.0, 100.0], 1
    )
    assert result["boundary_tie_group_size"] == 1
    assert result["boundary_fraction"] == 1.0
    assert result["MAE_mc_mean"] == pytest.approx(1.0)


def test_boundary_tie_uses_fractional_expected_retention() -> None:
    risk = [0, 0, 1, 1, 1, 2]
    errors = np.array([1, 3, 10, 20, 30, 100], dtype=float)
    result = retention(risk, errors, 4)
    expected_mae = (1 + 3 + (2 / 3) * (10 + 20 + 30)) / 4
    expected_rmse = math.sqrt(
        (1**2 + 3**2 + (2 / 3) * (10**2 + 20**2 + 30**2)) / 4
    )
    assert result["n_fully_retained_lower"] == 2
    assert result["boundary_tie_group_size"] == 3
    assert result["boundary_fraction"] == pytest.approx(2 / 3)
    assert result["fractional_boundary_used"] is True
    assert result["expected_retained_n"] == pytest.approx(4)
    assert result["MAE_mc_mean"] == pytest.approx(expected_mae)
    assert result["RMSE_mc_mean"] == pytest.approx(expected_rmse)


def test_boundary_tie_is_independent_of_row_order() -> None:
    risk = np.array([0, 0, 1, 1, 1, 2], dtype=float)
    errors = np.array([1, 3, 10, 20, 30, 100], dtype=float)
    expected = retention(risk, errors, 4)
    order = np.array([4, 1, 5, 2, 0, 3])
    observed = retention(risk[order], errors[order], 4)
    for field in ("MAE_mc_mean", "RMSE_mc_mean", "boundary_fraction"):
        assert observed[field] == pytest.approx(expected[field])


def test_boundary_tie_does_not_accept_sample_id_as_secondary_key() -> None:
    signature = inspect.signature(stage2b.tie_aware_retention_metrics)
    assert "sample_id" not in signature.parameters
    assert "date" not in signature.parameters
    assert "true_error_sort_key" not in signature.parameters


def test_expected_retained_count_and_boundary_fraction_are_exact() -> None:
    result = retention([0, 0, 0, 1], [1, 2, 3, 4], 2)
    assert result["boundary_fraction"] == pytest.approx(2 / 3)
    assert result["expected_retained_n"] == pytest.approx(2)


def test_all_unique_risk_matches_ordinary_sorted_prefix() -> None:
    risk = np.array([0.4, 0.1, 0.3, 0.2])
    errors = np.array([4.0, 1.0, 3.0, 2.0])
    result = retention(risk, errors, 3)
    selected = errors[np.argsort(risk)[:3]]
    assert result["MAE_mc_mean"] == pytest.approx(selected.mean())
    assert result["RMSE_mc_mean"] == pytest.approx(np.sqrt(np.mean(selected**2)))


def test_all_constant_risk_uses_whole_group_expectation() -> None:
    errors = np.array([1.0, 2.0, 10.0, 20.0])
    result = retention(np.ones(4), errors, 1)
    assert result["boundary_tie_group_size"] == 4
    assert result["boundary_fraction"] == pytest.approx(0.25)
    assert result["MAE_mc_mean"] == pytest.approx(errors.mean())
    assert result["RMSE_mc_mean"] == pytest.approx(np.sqrt(np.mean(errors**2)))


def test_split_cp_like_constant_width_does_not_fabricate_ordering() -> None:
    risk = np.repeat(0.177100472572, 5)
    errors = np.array([0.01, 0.02, 0.05, 0.20, 0.50])
    result = retention(risk, errors, 2)
    assert result["boundary_tie_group_size"] == 5
    assert result["MAE_mc_mean"] == pytest.approx(errors.mean())


def test_oracle_keeps_lowest_absolute_errors() -> None:
    errors = np.array([0.4, 0.1, 0.3, 0.2])
    result = retention(errors, errors, 2, rounded=False)
    assert result["MAE_mc_mean"] == pytest.approx(0.15)
    assert result["RMSE_mc_mean"] == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2))


def test_oracle_exact_ties_use_fractional_boundary() -> None:
    errors = np.array([0.1, 0.2, 0.2, 0.2, 0.5])
    result = retention(errors, errors, 2, rounded=False)
    assert result["n_fully_retained_lower"] == 1
    assert result["boundary_tie_group_size"] == 3
    assert result["boundary_fraction"] == pytest.approx(1 / 3)
    assert result["MAE_mc_mean"] == pytest.approx(0.15)


def test_oracle_does_not_round_distinct_true_errors() -> None:
    errors = np.array(
        [0.1771004725719999, 0.177100472572, 0.1771004725720001]
    )
    result = retention(errors, errors, 1, rounded=False)
    assert result["boundary_tie_group_size"] == 1
    assert result["MAE_mc_mean"] == errors.min()


def test_method_metrics_are_not_better_than_oracle() -> None:
    errors = np.linspace(0.01, 0.20, 20)
    method = stage2b.build_curve_for_score(
        stage2b.MC_STD, errors[::-1], errors, errors**2
    )
    oracle = stage2b.build_curve_for_score(
        stage2b.ORACLE, errors, errors, errors**2, oracle=True
    )
    assert np.all(method["MAE_mc_mean"] >= oracle["MAE_mc_mean"] - 1e-12)
    assert np.all(method["RMSE_mc_mean"] >= oracle["RMSE_mc_mean"] - 1e-12)


def test_coverage_one_method_and_oracle_converge() -> None:
    errors = np.array([0.1, 0.2, 0.4, 0.8])
    method = stage2b.build_curve_for_score(
        stage2b.MC_STD, [4, 3, 2, 1], errors, errors**2
    )
    oracle = stage2b.build_curve_for_score(
        stage2b.ORACLE, errors, errors, errors**2, oracle=True
    )
    assert method.iloc[-1]["MAE_mc_mean"] == pytest.approx(
        oracle.iloc[-1]["MAE_mc_mean"]
    )
    assert method.iloc[-1]["RMSE_mc_mean"] == pytest.approx(
        oracle.iloc[-1]["RMSE_mc_mean"]
    )


def test_oracle_curve_is_nondecreasing() -> None:
    errors = np.linspace(0.01, 0.50, 50)
    oracle = stage2b.build_curve_for_score(
        stage2b.ORACLE, errors, errors, errors**2, oracle=True
    )
    assert np.all(np.diff(oracle["MAE_mc_mean"]) >= -1e-12)
    assert np.all(np.diff(oracle["RMSE_mc_mean"]) >= -1e-12)


def test_ause_perfect_oracle_method_is_zero(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 20)
    values = np.linspace(0.01, 0.20, len(stage2b.COVERAGE_GRID))
    curves = toy_curves(values, values, values, values)
    summary = stage2b.build_ause_summary(curves)
    assert np.allclose(summary["AUSE_MAE"], 0.0)
    assert np.allclose(summary["AUSE_RMSE"], 0.0)


def test_trapezoidal_ause_matches_hand_calculation() -> None:
    assert stage2b.trapezoidal_integral([0.0, 1.0, 1.0], [0.1, 0.5, 1.0]) == pytest.approx(
        0.2 + 0.5
    )


def test_ause_is_not_divided_by_coverage_span(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 20)
    zero = np.zeros(len(stage2b.COVERAGE_GRID))
    one = np.ones(len(stage2b.COVERAGE_GRID))
    curves = toy_curves(one, one, zero, zero)
    summary = stage2b.build_ause_summary(curves)
    assert np.allclose(summary["AUSE_MAE"], 0.9)
    assert np.allclose(summary["AUSE_RMSE"], 0.9)


def test_mae_and_rmse_are_integrated_separately(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 20)
    zero = np.zeros(len(stage2b.COVERAGE_GRID))
    one = np.ones(len(stage2b.COVERAGE_GRID))
    two = np.repeat(2.0, len(stage2b.COVERAGE_GRID))
    summary = stage2b.build_ause_summary(toy_curves(one, two, zero, zero))
    assert np.allclose(summary["AUSE_MAE"], 0.9)
    assert np.allclose(summary["AUSE_RMSE"], 1.8)


def test_curve_and_ause_orders_are_fixed(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 20)
    evaluation = synthetic_evaluation(20)
    curves = stage2b.build_all_curves(evaluation)
    assert tuple(dict.fromkeys(curves["risk_score"])) == stage2b.CURVE_SCORE_ORDER
    summary = stage2b.build_ause_summary(curves)
    assert tuple(summary["risk_score"]) == stage2b.RISK_SCORE_ORDER
    assert stage2b.ORACLE not in set(summary["risk_score"])


@pytest.mark.parametrize("failure", ["missing", "nan", "negative"])
def test_evaluation_qc_rejects_missing_or_invalid_risk_scores(failure: str) -> None:
    evaluation = synthetic_evaluation()
    if failure == "missing":
        evaluation = evaluation.drop(columns=stage2b.PRED_L_STD_MC_WIDTH)
        expected = "missing columns"
    elif failure == "nan":
        evaluation.loc[0, stage2b.MC_STD] = np.nan
        expected = "must all be finite"
    else:
        evaluation.loc[0, stage2b.MC_STD] = -0.1
        expected = "must be non-negative"
    with pytest.raises(ValueError, match=expected):
        stage2b.validate_evaluation_table(evaluation, enforce_expected_n=False)


def test_curve_qc_rejects_expected_retained_count_drift(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 20)
    evaluation = synthetic_evaluation(20)
    curves = stage2b.build_all_curves(evaluation)
    curves.loc[0, "expected_retained_n"] += 0.25
    with pytest.raises(ValueError, match="expected retained count mismatch"):
        stage2b.validate_risk_coverage_curves(
            curves,
            evaluation["abs_error_mc_mean"],
            evaluation["sq_error_mc_mean"],
        )


def test_curve_qc_rejects_method_better_than_oracle(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 20)
    evaluation = synthetic_evaluation(20)
    curves = stage2b.build_all_curves(evaluation)
    oracle_first = curves[
        (curves["risk_score"] == stage2b.ORACLE)
        & (curves["coverage_requested"] == stage2b.COVERAGE_GRID[0])
    ]["MAE_mc_mean"].iloc[0]
    method_mask = (
        (curves["risk_score"] == stage2b.MC_STD)
        & (curves["coverage_requested"] == stage2b.COVERAGE_GRID[0])
    )
    curves.loc[method_mask, "MAE_mc_mean"] = oracle_first - 0.01
    with pytest.raises(ValueError, match="better than Oracle"):
        stage2b.validate_risk_coverage_curves(
            curves,
            evaluation["abs_error_mc_mean"],
            evaluation["sq_error_mc_mean"],
        )


def test_curve_qc_rejects_nonmonotonic_oracle(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 20)
    evaluation = synthetic_evaluation(20)
    curves = stage2b.build_all_curves(evaluation)
    oracle_mask = (
        (curves["risk_score"] == stage2b.ORACLE)
        & (curves["coverage_requested"] == stage2b.COVERAGE_GRID[1])
    )
    curves.loc[oracle_mask, "MAE_mc_mean"] = 0.0
    with pytest.raises(ValueError, match="Oracle MAE must be nondecreasing"):
        stage2b.validate_risk_coverage_curves(
            curves,
            evaluation["abs_error_mc_mean"],
            evaluation["sq_error_mc_mean"],
        )


def test_ause_qc_rejects_significantly_negative_gap(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 20)
    zero = np.zeros(len(stage2b.COVERAGE_GRID))
    one = np.ones(len(stage2b.COVERAGE_GRID))
    curves = toy_curves(zero, zero, one, one)
    with pytest.raises(ValueError, match="significantly negative"):
        stage2b.build_ause_summary(curves)


def test_stage2a_risk_definitions_are_reused_without_drift() -> None:
    base = synthetic_base()
    intervals = all_intervals(base)
    expected = stage2a.build_aligned_risk_table(base, intervals)
    observed = stage2b.build_evaluation_table(base, intervals)
    for risk_score in stage2b.RISK_SCORE_ORDER:
        assert observed[risk_score].tolist() == pytest.approx(expected[risk_score])


def test_evaluation_rejects_cp_calibration_role() -> None:
    evaluation = synthetic_evaluation(role="CP_CALIBRATION")
    with pytest.raises(PermissionError, match="Only DECISION_DEVELOPMENT"):
        stage2b.validate_evaluation_table(evaluation, enforce_expected_n=False)


def test_evaluation_rejects_sealed_dates() -> None:
    evaluation = synthetic_evaluation()
    evaluation["date"] = "2017-06-15"
    with pytest.raises(PermissionError, match="Sealed final date"):
        stage2b.validate_evaluation_table(evaluation, enforce_expected_n=False)


def test_random_test_and_unauthorized_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="RANDOM_TEST"):
        stage2a.validate_authorized_input_path(
            tmp_path / "random_test.csv", "stage1a_decision"
        )
    with pytest.raises(PermissionError, match="Unauthorized Stage 2A input"):
        stage2a.validate_authorized_input_path(
            stage2b.PROJECT_ROOT / "unauthorized" / "legacy_risk.csv",
            "stage1a_decision",
        )


@pytest.mark.parametrize("column", ["true_L", "mc_mean"])
def test_stage2a_alignment_rejects_value_mismatch(column: str) -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2b.RAW_MC_WIDTH)
    interval.loc[0, column] += 0.01
    with pytest.raises(ValueError, match=f"{column} mismatch"):
        stage2a.align_interval_to_base(base, interval, "raw_mc")


def test_stage2a_alignment_rejects_sample_id_mismatch() -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2b.RAW_MC_WIDTH)
    interval.loc[0, "sample_id"] = "wrong-id"
    with pytest.raises(ValueError, match="sample_id set mismatch"):
        stage2a.align_interval_to_base(base, interval, "raw_mc")


def test_output_collision_and_fixed_output_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage2b.ensure_output_available(output_dir)
    with pytest.raises(PermissionError, match="Unauthorized Stage 2B output"):
        stage2b.validate_formal_output_path(tmp_path / "other")


def test_writer_creates_only_required_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage2b"
    table = pd.DataFrame({"synthetic": [1]})
    stage2b.write_outputs(output_dir, table, table, {}, {})
    assert {path.name for path in output_dir.iterdir()} == {
        "risk_coverage_curves.csv",
        "ause_summary.csv",
        "config.json",
        "provenance.json",
    }


def test_no_automatic_ranking_winner_or_threshold_functions() -> None:
    function_names = {
        node.name.lower()
        for node in ast.walk(ast.parse(inspect.getsource(stage2b)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden = (
        "winner",
        "best",
        "select",
        "threshold",
        "reject",
        "capture",
    )
    assert not any(term in name for name in function_names for term in forbidden)


def test_no_fixed_twenty_percent_reject_or_high_error_capture_path() -> None:
    source = inspect.getsource(stage2b)
    tree = ast.parse(source)
    call_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "nlargest" not in call_names
    assert "nsmallest" not in call_names
    assert "sort_values" not in call_names


def test_no_model_image_mc_training_optimizer_cqr_or_cleaning_path() -> None:
    tree = ast.parse(inspect.getsource(stage2b))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not ({"torch", "torchvision", "PIL"} & imported_roots)
    function_names = {
        node.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(
        term in name
        for name in function_names
        for term in ("train", "optimizer", "cqr", "cleaning", "inference")
    )


def test_config_and_provenance_freeze_required_definitions() -> None:
    config = stage2b.make_config()
    provenance = stage2b.make_provenance()
    assert config["coverage_grid"] == list(stage2b.COVERAGE_GRID)
    assert config["risk_tie_round_decimals"] == 12
    assert config["boundary_tie_policy"] == (
        "fractional expected retention within boundary tie group"
    )
    assert config["ause_coverage_span_normalized"] is False
    assert provenance["formal_risk_score_selected"] is False
    assert provenance["formal_risk_threshold_frozen"] is False
    assert provenance["formal_reject_rate_frozen"] is False
    assert provenance["cp_calibration_truth_used_for_risk_evaluation"] is False
    assert provenance["random_test_accessed"] is False
    assert provenance["sealed_final_dates_accessed"] is False
    assert provenance["training_performed"] is False
    assert provenance["cqr_performed"] is False
    assert provenance["cleaning_decision_performed"] is False
