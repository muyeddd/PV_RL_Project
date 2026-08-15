"""Synthetic/unit tests for Paper1 Stage 5-A1 normalized economics.

No test reads the frozen Stage 4A output or calls the formal Stage 5-A1 run.
All file writes are restricted to pytest temporary directories.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import run_paper1_economic_stage5a_normalized_v1 as stage5a


def _aligned(n: int = 12) -> pd.DataFrame:
    true_l = np.linspace(0.02, 0.31, n)
    q50 = np.clip(true_l + np.resize(np.array([-0.03, 0.0, 0.025]), n), 0, 1)
    lower = np.clip(q50 - 0.045, 0, 1)
    upper = np.clip(q50 + 0.045, 0, 1)
    point = np.clip(true_l + np.resize(np.array([0.04, -0.025, 0.0]), n), 0, 1)
    return pd.DataFrame(
        {
            "sample_id": [f"synthetic-{index:04d}" for index in range(n)],
            "date": ["2017-06-01"] * n,
            "timestamp": [f"2017-06-01T00:{index:02d}:00" for index in range(n)],
            "image_path": [
                f"synthetic/decision/image_{index:04d}.jpg" for index in range(n)
            ],
            "role": [stage5a.EVALUATION_ROLE] * n,
            "true_L": true_l,
            "irradiance": np.linspace(0.1, 0.9, n),
            "point_pred": point,
            "q50": q50,
            "lower": lower,
            "upper": upper,
        }
    ).loc[:, stage5a.stage4a.ALIGNED_COLUMNS]


def _actions(n: int = 12) -> pd.DataFrame:
    return stage5a.stage4a.build_decision_actions(
        _aligned(n), enforce_expected_n=False
    )


def _sample_regrets(n: int = 12) -> pd.DataFrame:
    return stage5a.build_sample_regrets(_actions(n), enforce_expected_n=False)


def _metrics(n: int = 12) -> pd.DataFrame:
    return stage5a.build_economic_metrics(
        _sample_regrets(n), enforce_expected_n=False
    )


def test_tau_grid_exact_values_and_order() -> None:
    assert stage5a.TAU_GRID == (0.05, 0.10, 0.15, 0.20)
    assert stage5a.TAU_GRID == stage5a.stage4a.TAU_GRID
    assert stage5a.make_config()["tau_grid"] == [0.05, 0.10, 0.15, 0.20]


def test_method_order_exact() -> None:
    assert stage5a.METHOD_ORDER == (
        "point_threshold",
        "cqr_q50_threshold",
        "cqr_interval_tristate",
    )
    assert stage5a.METHOD_ORDER == stage5a.stage4a.METHOD_ORDER


def test_reference_tau_is_predeclared_only() -> None:
    assert stage5a.REFERENCE_TAU == 0.15
    provenance = stage5a.make_provenance()
    assert provenance["reference_tau"] == 0.15
    assert provenance["tau_selected_from_economic_results"] is False


@pytest.mark.parametrize("tau", [-0.1, 0.0, 0.075, 0.25, 1.0])
def test_undeclared_tau_is_rejected(tau: float) -> None:
    with pytest.raises(ValueError, match="Unauthorized"):
        stage5a.validate_tau(tau)


def test_oracle_boundary_true_l_equal_tau_is_wait() -> None:
    assert stage5a.stage4a.oracle_actions([0.15], 0.15).tolist() == [stage5a.WAIT]
    assert stage5a.oracle_cost(0.15, 0.15) == 0.15


def test_normalized_cost_definitions() -> None:
    assert stage5a.NORMALIZED_RECOVERABLE_VALUE == 1.0
    assert stage5a.action_cost_r0(0.08, 0.15, stage5a.CLEAN) == 0.15
    assert stage5a.action_cost_r0(0.08, 0.15, stage5a.WAIT) == 0.08
    assert stage5a.oracle_cost(0.08, 0.15) == 0.08
    assert stage5a.oracle_cost(0.20, 0.15) == 0.15


def test_false_clean_regret() -> None:
    result = stage5a.sample_regret_components(0.08, 0.15, stage5a.CLEAN)
    assert result["false_clean_regret"] == pytest.approx(0.07)
    assert result["missed_clean_regret"] == 0.0
    assert result["regret_r0"] == pytest.approx(0.07)
    assert result["automatic_error_regret"] == pytest.approx(abs(0.08 - 0.15))


def test_missed_clean_regret() -> None:
    result = stage5a.sample_regret_components(0.23, 0.15, stage5a.WAIT)
    assert result["false_clean_regret"] == 0.0
    assert result["missed_clean_regret"] == pytest.approx(0.08)
    assert result["regret_r0"] == pytest.approx(0.08)
    assert result["automatic_error_regret"] == pytest.approx(abs(0.23 - 0.15))


@pytest.mark.parametrize(
    ("true_l", "action"),
    [(0.08, stage5a.WAIT), (0.23, stage5a.CLEAN), (0.15, stage5a.WAIT)],
)
def test_correct_automatic_action_regret_is_zero(true_l: float, action: str) -> None:
    result = stage5a.sample_regret_components(true_l, 0.15, action)
    assert result["regret_r0"] == 0.0
    assert result["automatic_error_regret"] == 0.0


@pytest.mark.parametrize("true_l", [0.08, 0.15, 0.23])
def test_review_at_zero_cost_has_zero_regret(true_l: float) -> None:
    result = stage5a.sample_regret_components(true_l, 0.15, stage5a.REVIEW)
    assert result["regret_r0"] == 0.0
    assert result["review_indicator"] is True
    assert result["base_action_cost_r0"] == result["oracle_cost"]


@pytest.mark.parametrize(
    ("ratio", "tau", "expected"),
    [(0.0, 0.15, 0.0), (0.2, 0.15, 0.03), (0.5, 0.20, 0.10)],
)
def test_review_positive_cost_regret_is_ratio_times_tau(
    ratio: float, tau: float, expected: float
) -> None:
    assert stage5a.review_regret(ratio, tau) == pytest.approx(expected)


def test_review_total_cost_is_oracle_plus_increment() -> None:
    actual = stage5a.review_total_cost(0.08, 0.15, 0.2)
    assert actual == pytest.approx(0.08 + 0.2 * 0.15)
    actual = stage5a.review_total_cost(0.22, 0.15, 0.2)
    assert actual == pytest.approx(0.15 + 0.2 * 0.15)


def test_negative_physical_review_cost_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stage5a.review_regret(-0.1, 0.15)


def test_review_total_cost_inherits_negative_ratio_rejection() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stage5a.review_total_cost(0.08, 0.15, review_cost_ratio=-0.1)


@pytest.mark.parametrize(
    ("true_l", "action"),
    [
        (0.08, stage5a.CLEAN),
        (0.08, stage5a.WAIT),
        (0.08, stage5a.REVIEW),
        (0.23, stage5a.CLEAN),
        (0.23, stage5a.WAIT),
        (0.23, stage5a.REVIEW),
    ],
)
def test_total_cost_equals_oracle_plus_regret(true_l: float, action: str) -> None:
    result = stage5a.sample_regret_components(true_l, 0.15, action)
    assert result["base_action_cost_r0"] == pytest.approx(
        result["oracle_cost"] + result["regret_r0"]
    )
    assert float(result["regret_r0"]) >= 0.0


def test_boundary_false_clean_has_zero_distance_regret() -> None:
    result = stage5a.sample_regret_components(0.15, 0.15, stage5a.CLEAN)
    assert result["false_clean_regret"] == 0.0
    assert result["regret_r0"] == abs(0.15 - 0.15)


def test_illegal_predicted_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="Illegal"):
        stage5a.sample_regret_components(0.1, 0.15, "UNKNOWN")


def test_sample_regret_schema_and_row_count() -> None:
    sample = _sample_regrets()
    assert tuple(sample.columns) == stage5a.SAMPLE_REGRET_COLUMNS
    assert len(sample) == 12 * 4 * 3
    assert (sample["regret_r0"] >= 0.0).all()


def test_metric_definitions_and_cost_identity() -> None:
    sample = _sample_regrets()
    metrics = _metrics()
    row = metrics.iloc[0]
    group = sample.loc[
        (sample["method"] == row["method"]) & (sample["tau"] == row["tau"])
    ]
    assert row["oracle_mean_cost"] == pytest.approx(group["oracle_cost"].mean())
    assert row["mean_regret_r0"] == pytest.approx(group["regret_r0"].mean())
    assert row["mean_total_cost_r0"] == pytest.approx(
        row["oracle_mean_cost"] + row["mean_regret_r0"]
    )


def test_oracle_mean_cost_is_method_independent_per_tau() -> None:
    metrics = _metrics()
    for tau in stage5a.TAU_GRID:
        values = metrics.loc[metrics["tau"] == tau, "oracle_mean_cost"]
        assert values.nunique() == 1


def test_false_and_missed_regret_aggregates() -> None:
    sample = _sample_regrets()
    metrics = _metrics()
    row = metrics.iloc[0]
    group = sample.loc[
        (sample["method"] == row["method"]) & (sample["tau"] == row["tau"])
    ]
    assert row["false_clean_regret_sum"] == pytest.approx(
        group["false_clean_regret"].sum()
    )
    assert row["false_clean_regret_mean_per_N"] == pytest.approx(
        group["false_clean_regret"].sum() / len(group)
    )
    assert row["missed_clean_regret_sum"] == pytest.approx(
        group["missed_clean_regret"].sum()
    )
    assert row["missed_clean_regret_mean_per_N"] == pytest.approx(
        group["missed_clean_regret"].sum() / len(group)
    )
    assert row["automatic_error_regret_sum"] == pytest.approx(
        group["regret_r0"].sum()
    )


@pytest.mark.parametrize("method", [stage5a.POINT_THRESHOLD, stage5a.CQR_Q50_THRESHOLD])
def test_binary_baselines_have_zero_review(method: str) -> None:
    rows = _metrics().loc[lambda frame: frame["method"] == method]
    assert (rows["review_n"] == 0).all()
    assert (rows["review_rate"] == 0.0).all()


def test_metrics_have_fixed_method_and_tau_order() -> None:
    metrics = _metrics()
    assert tuple(dict.fromkeys(metrics["method"])) == stage5a.METHOD_ORDER
    for method in stage5a.METHOD_ORDER:
        assert tuple(metrics.loc[metrics["method"] == method, "tau"]) == stage5a.TAU_GRID


def test_break_even_analytic_formula() -> None:
    actual = stage5a.analytic_break_even_ratio(0.08, 0.02, 0.25, 0.10)
    assert actual == pytest.approx((0.08 - 0.02) / (0.25 * 0.10))


@pytest.mark.parametrize("baseline", [0.08, 0.05])
def test_break_even_substitution_equals_baseline(baseline: float) -> None:
    cqr = 0.02
    review_rate = 0.25
    tau = 0.10
    ratio = stage5a.analytic_break_even_ratio(baseline, cqr, review_rate, tau)
    substituted = stage5a.cqr_mean_regret_at_review_ratio(
        cqr, review_rate, ratio, tau
    )
    assert substituted == pytest.approx(baseline, abs=stage5a.BREAK_EVEN_ABS_TOLERANCE)


def test_negative_break_even_is_not_truncated() -> None:
    ratio = stage5a.analytic_break_even_ratio(0.01, 0.02, 0.25, 0.10)
    assert ratio == pytest.approx(-0.4)
    assert ratio < 0.0
    assert stage5a.make_config()["break_even_negative_values_truncated"] is False


def test_zero_review_rate_returns_nan() -> None:
    ratio = stage5a.analytic_break_even_ratio(0.08, 0.02, 0.0, 0.10)
    assert math.isnan(ratio)


def test_break_even_table_contains_both_fixed_baselines() -> None:
    metrics = _metrics()
    table = stage5a.build_break_even_review_cost(metrics)
    assert tuple(table.columns) == stage5a.BREAK_EVEN_COLUMNS
    assert tuple(table["tau"]) == stage5a.TAU_GRID
    assert "break_even_vs_point" in table
    assert "break_even_vs_q50" in table
    for row in table.itertuples(index=False):
        if row.review_rate > 0:
            assert stage5a.cqr_mean_regret_at_review_ratio(
                row.cqr_mean_regret_r0,
                row.review_rate,
                row.break_even_vs_point,
                row.tau,
            ) == pytest.approx(row.point_mean_regret_r0)
            assert stage5a.cqr_mean_regret_at_review_ratio(
                row.cqr_mean_regret_r0,
                row.review_rate,
                row.break_even_vs_q50,
                row.tau,
            ) == pytest.approx(row.q50_mean_regret_r0)


def test_no_review_cost_ratio_scan() -> None:
    source = inspect.getsource(stage5a)
    assert "review_cost_ratio_scanned" in source
    assert stage5a.make_config()["review_cost_ratio_scanned"] is False
    for fragment in ("np.arange(", "np.linspace(", "GridSearch", "optimize."):
        assert fragment not in source


def test_formal_input_path_is_exact_stage4a_action_artifact() -> None:
    assert stage5a.STAGE4A_ACTIONS_INPUT == (
        stage5a.stage4a.OUTPUT_DIR / "decision_actions.csv"
    )


@pytest.mark.parametrize(
    "path",
    [Path("synthetic/random_test.csv"), Path("synthetic/cp_calibration.csv")],
)
def test_forbidden_formal_input_paths_are_rejected_before_io(path: Path) -> None:
    with pytest.raises(PermissionError):
        stage5a.validate_authorized_input_path(path, "stage4a_decision_actions")


def test_arbitrary_formal_input_path_is_rejected() -> None:
    with pytest.raises(PermissionError):
        stage5a.validate_authorized_input_path(
            Path("synthetic/decision_actions.csv"), "stage4a_decision_actions"
        )


@pytest.mark.parametrize("role", ["CP_CALIBRATION", "RANDOM_TEST", "TRAIN"])
def test_nondecision_roles_are_rejected(role: str) -> None:
    actions = _actions()
    actions["role"] = role
    with pytest.raises(PermissionError):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


@pytest.mark.parametrize("case", ["sealed_date", "random_test", "sealed_locator"])
def test_random_test_and_sealed_dates_are_rejected(case: str) -> None:
    actions = _actions()
    if case == "sealed_date":
        actions["date"] = "2017-06-15"
    elif case == "random_test":
        actions["image_path"] = [
            f"synthetic/random_test/{index}.jpg" for index in range(len(actions))
        ]
    else:
        actions["image_path"] = [
            f"synthetic/2017-06-24/{index}.jpg" for index in range(len(actions))
        ]
    with pytest.raises(PermissionError):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


def test_duplicate_sample_method_tau_is_rejected() -> None:
    actions = _actions()
    duplicate_key = actions.loc[0, ["sample_id", "method", "tau"]]
    same_method_tau = (actions["method"] == duplicate_key["method"]) & (
        actions["tau"] == duplicate_key["tau"]
    )
    target_index = actions.index[same_method_tau][1]
    actions.loc[target_index, "sample_id"] = duplicate_key["sample_id"]
    with pytest.raises(ValueError, match="unique|Duplicate"):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


def test_formal_n_and_row_count_guards() -> None:
    assert stage5a.EXPECTED_N == 1844
    assert stage5a.EXPECTED_FORMAL_ROWS == 1844 * 4 * 3 == 22128
    with pytest.raises(ValueError, match="22128|1844"):
        stage5a.validate_stage4_actions(_actions(), enforce_expected_n=True)


def test_stage4_rule_inconsistency_is_rejected() -> None:
    actions = _actions()
    original = actions.loc[0, "predicted_action"]
    actions.loc[0, "predicted_action"] = (
        stage5a.WAIT if original == stage5a.CLEAN else stage5a.CLEAN
    )
    with pytest.raises(ValueError, match="inconsistent"):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


def test_nonfinite_true_l_is_rejected() -> None:
    actions = _actions()
    actions.loc[0, "true_L"] = np.nan
    with pytest.raises(ValueError):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


def test_point_baseline_review_is_rejected() -> None:
    actions = _actions()
    index = actions.index[actions["method"] == stage5a.POINT_THRESHOLD][0]
    actions.loc[index, "predicted_action"] = stage5a.REVIEW
    with pytest.raises(ValueError):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


def test_q50_baseline_review_is_rejected() -> None:
    actions = _actions()
    index = actions.index[actions["method"] == stage5a.CQR_Q50_THRESHOLD][0]
    actions.loc[index, "predicted_action"] = stage5a.REVIEW
    with pytest.raises(ValueError):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


def test_cqr_illegal_action_is_rejected() -> None:
    actions = _actions()
    index = actions.index[actions["method"] == stage5a.CQR_INTERVAL_TRISTATE][0]
    actions.loc[index, "predicted_action"] = "UNKNOWN"
    with pytest.raises(ValueError):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


@pytest.mark.parametrize("field", ["method", "tau"])
def test_frozen_method_or_tau_drift_is_rejected(field: str) -> None:
    actions = _actions()
    actions.loc[0, field] = "drifted" if field == "method" else 0.075
    with pytest.raises((ValueError, TypeError)):
        stage5a.validate_stage4_actions(actions, enforce_expected_n=False)


def test_output_collision_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "stage5a"
    output.mkdir()
    (output / "existing.txt").write_text("synthetic", encoding="utf-8")
    with pytest.raises(FileExistsError):
        stage5a.ensure_output_available(output)


def test_outputs_have_no_winner_ranking_or_optimal_tau_fields() -> None:
    frames = (
        _sample_regrets(),
        _metrics(),
        stage5a.build_break_even_review_cost(_metrics()),
    )
    for frame in frames:
        assert not ({str(column).lower() for column in frame.columns} & stage5a.FORBIDDEN_OUTPUT_FIELDS)
    assert "sort_values(" not in inspect.getsource(stage5a)


def test_synthetic_writer_creates_only_required_files(tmp_path: Path) -> None:
    sample = _sample_regrets()
    metrics = _metrics()
    break_even = stage5a.build_break_even_review_cost(metrics)
    output = tmp_path / "stage5a"
    stage5a.write_outputs(
        output,
        sample,
        metrics,
        break_even,
        stage5a.make_config(),
        stage5a.make_provenance(),
    )
    assert {path.name for path in output.iterdir()} == {
        "economic_sample_regrets.csv",
        "economic_metrics.csv",
        "break_even_review_cost.csv",
        "config.json",
        "provenance.json",
    }


def test_perfect_review_interpretation_is_bounded_not_realism_claim() -> None:
    config = stage5a.make_config()
    assert "optimistic lower bound" in config["review_benchmark_interpretation"]
    assert "upper bound on benefit" in config["review_benchmark_interpretation"]
    assert config["real_human_review_claimed_perfect"] is False


@pytest.mark.parametrize(
    "field",
    ["normalized_economics", "review_is_perfect_resolution_benchmark"],
)
def test_required_true_provenance_flags(field: str) -> None:
    assert stage5a.make_provenance()[field] is True


@pytest.mark.parametrize(
    "field",
    [
        "currency_used",
        "gansu_price_used_in_core_evaluation",
        "actual_station_scale_claimed",
        "review_cost_ratio_selected_from_results",
        "tau_selected_from_economic_results",
        "random_test_accessed",
        "sealed_final_dates_accessed",
        "training_performed",
        "inference_performed",
        "conformal_recalibration_performed",
        "risk_score_development_performed",
        "threshold_optimization_performed",
        "method_selection_performed",
        "economic_winner_declared",
    ],
)
def test_required_false_provenance_flags(field: str) -> None:
    assert stage5a.make_provenance()[field] is False


def test_module_has_no_model_or_recalibration_execution() -> None:
    source = inspect.getsource(stage5a)
    forbidden = (
        "torch.",
        "model(",
        "predict_deterministic(",
        "predict_mc_dropout(",
        "calibrate_cqr(",
        "conformalize_decision_intervals(",
    )
    for fragment in forbidden:
        assert fragment not in source


def test_tests_do_not_read_formal_stage4_output_or_call_run() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "outputs/" + "paper1_clean_random_v1" not in source
    assert "decision_stage4a_" + "rule_v1/" not in source
    assert "pd." + "read_csv(" not in source
    assert "stage5a." + "run(" not in source
