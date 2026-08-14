from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.run_paper1_risk_stage2a_score_validity_v1 as stage2a
import experiments.run_paper1_risk_stage2b_risk_coverage_v1 as stage2b
import experiments.run_paper1_risk_stage2c_high_error_capture_v1 as stage2c


def synthetic_base(n: int = 10, *, role: str = stage2c.PRIMARY_ROLE) -> pd.DataFrame:
    errors = np.linspace(0.01, 0.10, n)
    return pd.DataFrame(
        {
            "sample_id": [f"dd-{index}" for index in range(n)],
            "date": ["2017-06-13"] * n,
            "timestamp": [f"2017-06-13T10:{index:02d}:00" for index in range(n)],
            "image_path": [f"data/mock/dd-{index}.jpg" for index in range(n)],
            "role": [role] * n,
            "true_L": errors,
            "irradiance": np.linspace(0.1, 0.9, n),
            "point_pred": np.repeat(0.9, n),
            "mc_mean": np.zeros(n),
            "mc_std": np.linspace(0.001, 0.010, n),
        }
    )


def synthetic_interval(base: pd.DataFrame, risk_score: str) -> pd.DataFrame:
    n = len(base)
    widths = np.linspace(0.10, 0.30, n)
    lower = 0.5 - widths / 2
    upper = 0.5 + widths / 2
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
        for risk_score in stage2c.RISK_SCORE_ORDER[1:]
    }


def synthetic_evaluation(
    n: int = 10,
    *,
    errors=None,
    risk=None,
    role: str = stage2c.PRIMARY_ROLE,
) -> pd.DataFrame:
    error_values = (
        np.linspace(0.01, 0.10, n)
        if errors is None
        else np.asarray(errors, dtype=float)
    )
    risk_values = (
        np.linspace(0.001, 0.010, n)
        if risk is None
        else np.asarray(risk, dtype=float)
    )
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(n)],
            "date": ["2017-06-13"] * n,
            "role": [role] * n,
            "true_L": error_values,
            "mc_mean": np.zeros(n),
            "signed_error_mc_mean": error_values,
            "abs_error_mc_mean": error_values,
            "sq_error_mc_mean": np.square(error_values),
        }
    )
    for offset, risk_score in enumerate(stage2c.RISK_SCORE_ORDER):
        frame[risk_score] = risk_values + offset * 0.0001
    return frame


def patch_small_n(monkeypatch, n: int) -> None:
    monkeypatch.setattr(stage2c, "EXPECTED_N", n)
    monkeypatch.setattr(stage2b, "EXPECTED_N", n)


def test_only_decision_development_is_allowed() -> None:
    evaluation = synthetic_evaluation(role="CP_CALIBRATION")
    with pytest.raises(PermissionError, match="Only DECISION_DEVELOPMENT"):
        stage2b.validate_evaluation_table(evaluation, enforce_expected_n=False)


def test_n_guard_is_reused_from_stage2b(monkeypatch) -> None:
    monkeypatch.setattr(stage2b, "EXPECTED_N", 10)
    stage2b.validate_evaluation_table(synthetic_evaluation(10))
    with pytest.raises(ValueError, match="N guard failed"):
        stage2b.validate_evaluation_table(synthetic_evaluation(9))


def test_high_error_target_fraction_and_formal_count_are_fixed() -> None:
    assert stage2c.HIGH_ERROR_TARGET_FRACTION == 0.10
    assert stage2c.target_high_error_count(1844) == 184


def test_target_count_uses_floor_rule() -> None:
    assert stage2c.target_high_error_count(19) == 1
    assert stage2c.target_high_error_count(20) == 2


def test_target_uses_abs_true_minus_mc_mean_not_point_pred() -> None:
    base = synthetic_base()
    base["point_pred"] = base["true_L"]
    evaluation = stage2b.build_evaluation_table(base, all_intervals(base))
    assert evaluation["abs_error_mc_mean"].tolist() == pytest.approx(
        np.abs(base["true_L"] - base["mc_mean"])
    )


def test_high_error_target_starts_from_largest_error() -> None:
    weights, audit = stage2c.build_high_error_target(np.arange(1, 11))
    assert weights.tolist() == pytest.approx([0] * 9 + [1])
    assert audit["high_error_threshold_abs_error"] == 10


def test_exact_target_boundary_tie_uses_fractional_membership() -> None:
    errors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 10], dtype=float)
    weights, audit = stage2c.build_high_error_target(errors)
    assert weights[-2:].tolist() == pytest.approx([0.5, 0.5])
    assert audit["target_boundary_tie_group_size"] == 2
    assert audit["target_boundary_fraction"] == pytest.approx(0.5)
    assert audit["target_fractional_boundary_used"] is True


def test_target_ties_do_not_use_12_decimal_risk_rounding() -> None:
    errors = np.array(
        [0.01] * 7
        + [
            0.1771004725719999,
            0.177100472572,
            0.1771004725720001,
        ]
    )
    weights, audit = stage2c.build_high_error_target(errors)
    assert audit["target_boundary_tie_group_size"] == 1
    assert weights[-1] == 1.0
    assert weights[-2] == 0.0


def test_target_weights_are_row_order_independent() -> None:
    errors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 10], dtype=float)
    expected, _ = stage2c.build_high_error_target(errors)
    order = np.array([9, 1, 7, 0, 8, 3, 2, 5, 4, 6])
    shuffled, _ = stage2c.build_high_error_target(errors[order])
    restored = np.empty_like(shuffled)
    restored[order] = shuffled
    assert restored.tolist() == pytest.approx(expected)


def test_target_weight_builder_has_no_sample_id_secondary_key() -> None:
    signature = inspect.signature(stage2c.build_high_error_target)
    assert "sample_id" not in signature.parameters
    assert "date" not in signature.parameters


def test_target_weight_sum_matches_target_count() -> None:
    weights, audit = stage2c.build_high_error_target(np.arange(20))
    assert weights.sum() == pytest.approx(2)
    assert audit["target_weight_sum"] == pytest.approx(2)


def test_risk_budgets_and_formal_counts_are_fixed() -> None:
    stage2c.validate_risk_budget_fractions()
    assert stage2c.RISK_BUDGET_FRACTIONS == (0.10, 0.20, 0.30)
    assert [stage2c.risk_budget_count(value, 1844) for value in stage2c.RISK_BUDGET_FRACTIONS] == [
        184,
        368,
        553,
    ]


def test_unauthorized_risk_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unauthorized risk budget"):
        stage2c.risk_budget_count(0.25, 1844)


def test_risk_selection_is_highest_first() -> None:
    weights, audit = stage2c.build_risk_selection(np.arange(10), 0.20)
    assert weights.tolist() == pytest.approx([0] * 8 + [1, 1])
    assert audit["n_fully_selected_higher_risk"] == 1


def test_risk_selection_reuses_12_decimal_ties() -> None:
    assert stage2c.RISK_TIE_ROUND_DECIMALS == stage2a.RISK_TIE_ROUND_DECIMALS == 12
    assert stage2c.stage2a.risk_rank_values is stage2a.risk_rank_values


def test_numerical_near_risk_ties_are_merged() -> None:
    risk = np.array(
        [0.01] * 7
        + [
            0.1771004725719999,
            0.177100472572,
            0.1771004725720001,
        ]
    )
    weights, audit = stage2c.build_risk_selection(risk, 0.10)
    assert audit["risk_boundary_tie_group_size"] == 3
    assert audit["risk_boundary_fraction"] == pytest.approx(1 / 3)
    assert weights[-3:].tolist() == pytest.approx([1 / 3] * 3)


def test_meaningful_risk_differences_are_preserved() -> None:
    risk = np.array([0.01] * 8 + [0.177100472572, 0.177100472600])
    weights, audit = stage2c.build_risk_selection(risk, 0.10)
    assert audit["risk_boundary_tie_group_size"] == 1
    assert weights[-1] == 1.0
    assert weights[-2] == 0.0


def test_risk_boundary_fractional_selection() -> None:
    risk = np.array([0, 1, 2, 2, 3, 3, 3, 3, 5, 6], dtype=float)
    weights, audit = stage2c.build_risk_selection(risk, 0.30)
    assert audit["n_fully_selected_higher_risk"] == 2
    assert audit["risk_boundary_tie_group_size"] == 4
    assert audit["risk_boundary_fraction"] == pytest.approx(0.25)
    assert weights.sum() == pytest.approx(3)


def test_risk_selection_is_row_order_independent() -> None:
    risk = np.array([0, 1, 2, 3, 3, 3, 3, 4, 5, 6], dtype=float)
    expected, _ = stage2c.build_risk_selection(risk, 0.30)
    order = np.array([8, 3, 1, 9, 5, 0, 6, 2, 7, 4])
    shuffled, _ = stage2c.build_risk_selection(risk[order], 0.30)
    restored = np.empty_like(shuffled)
    restored[order] = shuffled
    assert restored.tolist() == pytest.approx(expected)


def test_risk_selection_builder_has_no_sample_id_secondary_key() -> None:
    signature = inspect.signature(stage2c.build_risk_selection)
    assert "sample_id" not in signature.parameters
    assert "date" not in signature.parameters
    assert "true_error" not in signature.parameters


def test_expected_selected_count_matches_budget() -> None:
    weights, audit = stage2c.build_risk_selection(np.ones(10), 0.30)
    assert weights.sum() == pytest.approx(3)
    assert audit["expected_selected_n"] == pytest.approx(3)


def test_capture_intersection_without_ties_is_exact() -> None:
    target = np.array([0, 0, 0, 1, 1], dtype=float)
    selected = np.array([0, 0, 1, 1, 1], dtype=float)
    metrics = stage2c.capture_metrics(target, selected, target_n=2, budget_n=3)
    assert metrics["expected_captured_high_error"] == pytest.approx(2)


def test_fractional_target_with_unique_risk_capture() -> None:
    target = np.array([0, 0, 0, 0.5, 0.5])
    selected = np.array([0, 0, 0, 1, 0])
    metrics = stage2c.capture_metrics(target, selected, target_n=1, budget_n=1)
    assert metrics["expected_captured_high_error"] == pytest.approx(0.5)


def test_fractional_risk_with_unique_target_capture() -> None:
    target = np.array([0, 0, 0, 1, 0], dtype=float)
    selected = np.array([0, 0, 0, 0.5, 0.5])
    metrics = stage2c.capture_metrics(target, selected, target_n=1, budget_n=1)
    assert metrics["expected_captured_high_error"] == pytest.approx(0.5)


def test_double_fractional_weight_capture() -> None:
    target = np.array([0, 0, 0, 0.5, 0.5])
    selected = np.array([0, 0, 0, 0.5, 0.5])
    metrics = stage2c.capture_metrics(target, selected, target_n=1, budget_n=1)
    assert metrics["expected_captured_high_error"] == pytest.approx(0.5)


def test_capture_rate_precision_random_baseline_and_lift() -> None:
    target = np.array([0, 0, 0, 1, 1], dtype=float)
    selected = np.array([0, 0, 1, 1, 1], dtype=float)
    metrics = stage2c.capture_metrics(target, selected, target_n=2, budget_n=3)
    assert metrics["capture_rate"] == pytest.approx(1.0)
    assert metrics["precision_high_error"] == pytest.approx(2 / 3)
    assert metrics["random_capture_rate"] == pytest.approx(3 / 5)
    assert metrics["random_expected_captured_high_error"] == pytest.approx(6 / 5)
    assert metrics["random_precision_high_error"] == pytest.approx(2 / 5)
    assert metrics["capture_lift_vs_random"] == pytest.approx(1 / (3 / 5))
    assert metrics["capture_lift_vs_random"] == pytest.approx((2 / 3) / (2 / 5))


def test_perfect_risk_ranking_reaches_oracle_ceiling(monkeypatch) -> None:
    patch_small_n(monkeypatch, 10)
    errors = np.arange(1, 11, dtype=float)
    evaluation = synthetic_evaluation(10, errors=errors, risk=errors)
    summary, _ = stage2c.build_capture_summary(evaluation)
    assert np.allclose(summary["capture_rate"], 1.0)
    assert np.allclose(summary["oracle_capture_rate_ceiling"], 1.0)


def test_reversed_risk_ranking_performs_poorly(monkeypatch) -> None:
    patch_small_n(monkeypatch, 10)
    errors = np.arange(1, 11, dtype=float)
    evaluation = synthetic_evaluation(10, errors=errors, risk=errors[::-1])
    summary, _ = stage2c.build_capture_summary(evaluation)
    first = summary[
        (summary["risk_score"] == stage2c.RISK_SCORE_ORDER[0])
        & (summary["risk_budget_fraction"] == 0.10)
    ].iloc[0]
    assert first["capture_rate"] == pytest.approx(0.0)


@pytest.mark.parametrize("risk_score", [stage2c.MC_STD if hasattr(stage2c, "MC_STD") else stage2c.RISK_SCORE_ORDER[0], stage2c.RISK_SCORE_ORDER[2]])
def test_constant_risk_matches_random_expected_capture(risk_score: str) -> None:
    errors = np.arange(1, 11, dtype=float)
    target, audit = stage2c.build_high_error_target(errors)
    selected, selection_audit = stage2c.build_risk_selection(np.ones(10), 0.20)
    metrics = stage2c.capture_metrics(
        target,
        selected,
        audit["target_high_error_n"],
        selection_audit["risk_budget_n"],
    )
    assert risk_score in stage2c.RISK_SCORE_ORDER
    assert metrics["expected_captured_high_error"] == pytest.approx(
        metrics["random_expected_captured_high_error"]
    )
    assert metrics["capture_lift_vs_random"] == pytest.approx(1.0)


def test_oracle_ceiling_is_one_for_all_fixed_formal_budgets() -> None:
    target_n = stage2c.target_high_error_count(1844)
    ceilings = [
        min(1.0, stage2c.risk_budget_count(fraction, 1844) / target_n)
        for fraction in stage2c.RISK_BUDGET_FRACTIONS
    ]
    assert ceilings == [1.0, 1.0, 1.0]


def test_capture_bounds_qc_rejects_impossible_capture(monkeypatch) -> None:
    patch_small_n(monkeypatch, 10)
    summary, audit = stage2c.build_capture_summary(synthetic_evaluation(10))
    summary.loc[0, "expected_captured_high_error"] = audit["target_high_error_n"] + 1
    with pytest.raises(ValueError, match="exceeds target"):
        stage2c.validate_capture_summary(summary, 10, audit["target_high_error_n"])


def test_fixed_risk_and_budget_order(monkeypatch) -> None:
    patch_small_n(monkeypatch, 10)
    summary, _ = stage2c.build_capture_summary(synthetic_evaluation(10))
    assert tuple(dict.fromkeys(summary["risk_score"])) == stage2c.RISK_SCORE_ORDER
    for risk_score in stage2c.RISK_SCORE_ORDER:
        rows = summary[summary["risk_score"] == risk_score]
        assert tuple(rows["risk_budget_fraction"]) == stage2c.RISK_BUDGET_FRACTIONS


def test_stage2a_and_stage2b_helpers_are_reused() -> None:
    assert stage2c.stage2a.build_aligned_risk_table is stage2a.build_aligned_risk_table
    assert stage2c.stage2b.build_evaluation_table is stage2b.build_evaluation_table


def test_random_test_cp_and_unauthorized_paths_are_rejected() -> None:
    with pytest.raises(PermissionError, match="RANDOM_TEST"):
        stage2a.validate_authorized_input_path(
            stage2c.PROJECT_ROOT / "random_test.csv", "stage1a_decision"
        )
    with pytest.raises(PermissionError, match="CP_CALIBRATION"):
        stage2a.validate_authorized_input_path(
            stage2c.PROJECT_ROOT / "cp_calibration.csv", "stage1a_decision"
        )
    with pytest.raises(PermissionError, match="Unauthorized Stage 2A input"):
        stage2a.validate_authorized_input_path(
            stage2c.PROJECT_ROOT / "unauthorized" / "legacy.csv",
            "stage1a_decision",
        )


def test_sealed_dates_are_rejected() -> None:
    evaluation = synthetic_evaluation()
    evaluation["date"] = "2017-06-15"
    with pytest.raises(PermissionError, match="Sealed final date"):
        stage2b.validate_evaluation_table(evaluation, enforce_expected_n=False)


@pytest.mark.parametrize("column", ["true_L", "mc_mean"])
def test_alignment_rejects_truth_or_prediction_mismatch(column: str) -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2a.RAW_MC_WIDTH)
    interval.loc[0, column] += 0.01
    with pytest.raises(ValueError, match=f"{column} mismatch"):
        stage2a.align_interval_to_base(base, interval, "raw_mc")


def test_alignment_rejects_sample_id_mismatch() -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2a.RAW_MC_WIDTH)
    interval.loc[0, "sample_id"] = "wrong-id"
    with pytest.raises(ValueError, match="sample_id set mismatch"):
        stage2a.align_interval_to_base(base, interval, "raw_mc")


def test_output_collision_and_fixed_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage2c.ensure_output_available(output_dir)
    with pytest.raises(PermissionError, match="Unauthorized Stage 2C output"):
        stage2c.validate_formal_output_path(tmp_path / "other")


def test_writer_creates_only_required_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage2c"
    stage2c.write_outputs(output_dir, pd.DataFrame({"x": [1]}), {}, {}, {})
    assert {path.name for path in output_dir.iterdir()} == {
        "high_error_capture_summary.csv",
        "high_error_target_audit.json",
        "config.json",
        "provenance.json",
    }


def test_no_winner_formal_threshold_reject_rate_or_budget_selection_functions() -> None:
    function_names = {
        node.name.lower()
        for node in ast.walk(ast.parse(inspect.getsource(stage2c)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden = ("winner", "best", "optimal", "threshold", "reject", "select_budget")
    assert not any(term in name for name in function_names for term in forbidden)


def test_no_model_image_mc_training_optimizer_cqr_cleaning_or_economic_path() -> None:
    tree = ast.parse(inspect.getsource(stage2c))
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
        for term in ("train", "optimizer", "inference", "cqr", "cleaning", "economic")
    )


def test_config_and_provenance_freeze_scope() -> None:
    config = stage2c.make_config()
    provenance = stage2c.make_provenance()
    assert config["high_error_target_fraction"] == 0.10
    assert config["risk_budget_fractions"] == [0.10, 0.20, 0.30]
    assert config["risk_selection_direction"] == "highest risk first"
    assert config["risk_tie_round_decimals"] == 12
    assert provenance["formal_risk_score_selected"] is False
    assert provenance["formal_risk_threshold_frozen"] is False
    assert provenance["formal_reject_rate_frozen"] is False
    assert provenance["formal_risk_budget_selected"] is False
    assert provenance["stage2b_definition_modified"] is False
    assert provenance["cp_calibration_truth_used_for_risk_evaluation"] is False
    assert provenance["random_test_accessed"] is False
    assert provenance["sealed_final_dates_accessed"] is False
    assert provenance["training_performed"] is False
    assert provenance["cqr_performed"] is False
    assert provenance["cleaning_decision_performed"] is False
    assert provenance["economic_decision_performed"] is False
