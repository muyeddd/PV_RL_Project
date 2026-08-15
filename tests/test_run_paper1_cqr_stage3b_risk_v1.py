"""Synthetic/unit tests for Paper1 CQR Stage 3B.

No test in this module reads a formal artifact or invokes the Stage 3B entry
point.  The frozen Stage 2 public helpers are the executable reference for
equivalence checks.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import run_paper1_cqr_stage3b_risk_v1 as stage3b


def _base(n: int = 20) -> pd.DataFrame:
    true_l = np.linspace(0.20, 0.70, n)
    signed = np.resize(np.array([-0.04, 0.01, 0.07, -0.02]), n)
    return pd.DataFrame(
        {
            "sample_id": [f"synthetic-{i:04d}" for i in range(n)],
            "date": ["2017-06-01"] * n,
            "timestamp": [f"2017-06-01T00:{i:02d}:00" for i in range(n)],
            "image_path": [f"synthetic/decision/image_{i:04d}.jpg" for i in range(n)],
            "role": [stage3b.EVALUATION_ROLE] * n,
            "true_L": true_l,
            "irradiance": np.linspace(0.1, 0.9, n),
            "point_pred": true_l - 0.01,
            "mc_mean": true_l - signed,
            "mc_std": np.linspace(0.01, 0.03, n),
        }
    )


def _cqr(n: int = 20) -> pd.DataFrame:
    base = _base(n)
    q50 = np.linspace(0.25, 0.65, n)
    q05 = q50 - 0.08
    q95 = q50 + 0.08
    width = np.linspace(0.10, 0.24, n)
    lower = q50 - width / 2.0
    upper = q50 + width / 2.0
    result = pd.DataFrame(
        {
            "sample_id": base["sample_id"],
            "date": base["date"],
            "timestamp": base["timestamp"],
            "image_path": base["image_path"],
            "role": base["role"],
            "true_L": base["true_L"],
            "irradiance": base["irradiance"],
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "method": [stage3b.stage3a2.METHOD] * n,
            "lower": lower,
            "upper": upper,
            "width": width,
            "covered": (base["true_L"].to_numpy() >= lower)
            & (base["true_L"].to_numpy() <= upper),
            "raw_width": q95 - q05,
            "lower_clipped": [False] * n,
            "upper_clipped": [False] * n,
        }
    )
    return result.loc[:, stage3b.stage3a2.PREDICTION_OUTPUT_COLUMNS]


def _evaluation(n: int = 20) -> pd.DataFrame:
    return stage3b.build_cqr_evaluation_table(
        _base(n), _cqr(n), enforce_expected_n=False
    )


def _provenance() -> dict[str, object]:
    return dict(stage3b.REQUIRED_STAGE3A2_PROVENANCE)


def _baseline_spearman(n: int = 20) -> pd.DataFrame:
    evaluation = _evaluation(n)
    rows = [
        stage3b.stage2a.summarize_risk_score(
            method,
            evaluation[stage3b.CQR_RISK_SCORE].to_numpy() + index / 100.0,
            evaluation["abs_error_mc_mean"].to_numpy(),
        )
        for index, method in enumerate(stage3b.stage2a.RISK_SCORE_ORDER)
    ]
    return pd.DataFrame(rows).loc[:, stage3b.SPEARMAN_COLUMNS]


def _baseline_curves(evaluation: pd.DataFrame) -> pd.DataFrame:
    absolute = evaluation["abs_error_mc_mean"].to_numpy()
    squared = evaluation["sq_error_mc_mean"].to_numpy()
    risk = evaluation[stage3b.CQR_RISK_SCORE].to_numpy()
    curves = [
        stage3b.stage2b.build_curve_for_score(
            method, risk + index / 100.0, absolute, squared
        )
        for index, method in enumerate(stage3b.stage2a.RISK_SCORE_ORDER)
    ]
    curves.append(
        stage3b.stage2b.build_curve_for_score(
            stage3b.stage2b.ORACLE, absolute, absolute, squared, oracle=True
        )
    )
    return pd.concat(curves, ignore_index=True)


def _baseline_capture(cqr_capture: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for method in stage3b.stage2a.RISK_SCORE_ORDER:
        part = cqr_capture.copy()
        part["risk_score"] = method
        parts.append(part)
    return pd.concat(parts, ignore_index=True).loc[:, stage3b.stage2c.SUMMARY_COLUMNS]


def test_common_target_is_abs_true_minus_mc_mean() -> None:
    evaluation = _evaluation()
    expected = np.abs(
        evaluation["true_L"].to_numpy() - evaluation["mc_mean"].to_numpy()
    )
    np.testing.assert_array_equal(evaluation["abs_error_mc_mean"], expected)
    assert stage3b.COMMON_RISK_TARGET == stage3b.stage2a.RISK_TARGET_ERROR


def test_q50_error_is_not_the_target() -> None:
    first = _cqr()
    second = first.copy()
    second["q50"] = second["q50"] + 0.001
    second["q05"] = second["q05"] + 0.001
    second["q95"] = second["q95"] + 0.001
    second["raw_width"] = second["q95"] - second["q05"]
    left = stage3b.build_cqr_evaluation_table(_base(), first, enforce_expected_n=False)
    right = stage3b.build_cqr_evaluation_table(_base(), second, enforce_expected_n=False)
    np.testing.assert_array_equal(left["abs_error_mc_mean"], right["abs_error_mc_mean"])
    assert stage3b.make_provenance()["cqr_q50_error_evaluated"] is False


def test_only_cqr_score_is_final_conformal_width() -> None:
    cqr = _cqr()
    evaluation = stage3b.build_cqr_evaluation_table(
        _base(), cqr, enforce_expected_n=False
    )
    expected = cqr["upper"].to_numpy() - cqr["lower"].to_numpy()
    np.testing.assert_allclose(evaluation[stage3b.CQR_RISK_SCORE], expected)
    assert not np.allclose(expected, cqr["q95"] - cqr["q05"])
    assert stage3b.METHOD_ORDER[-1] == stage3b.CQR_RISK_SCORE


def test_stored_width_mismatch_is_rejected() -> None:
    cqr = _cqr()
    cqr.loc[0, "width"] += 1e-5
    with pytest.raises(ValueError, match="width"):
        stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("lower", np.nan, ValueError),
        ("upper", np.inf, ValueError),
        ("lower", -0.01, ValueError),
        ("upper", 1.01, ValueError),
    ],
)
def test_interval_finite_and_bounds_guards(field: str, value: float, error: type[Exception]) -> None:
    cqr = _cqr()
    cqr.loc[0, field] = value
    if np.isfinite(value):
        cqr.loc[0, "width"] = cqr.loc[0, "upper"] - cqr.loc[0, "lower"]
    with pytest.raises(error):
        stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)


def test_lower_above_upper_is_rejected() -> None:
    cqr = _cqr()
    cqr.loc[0, ["lower", "upper", "width"]] = [0.7, 0.6, -0.1]
    with pytest.raises(ValueError):
        stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)


def test_quantile_crossing_is_rejected() -> None:
    cqr = _cqr()
    cqr.loc[0, "q05"] = cqr.loc[0, "q50"] + 0.01
    cqr.loc[0, "raw_width"] = cqr.loc[0, "q95"] - cqr.loc[0, "q05"]
    with pytest.raises(ValueError):
        stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)


@pytest.mark.parametrize("field", ["q05", "q50", "q95"])
def test_nonfinite_quantiles_are_rejected(field: str) -> None:
    cqr = _cqr()
    cqr.loc[0, field] = np.nan
    with pytest.raises(ValueError):
        stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)


def test_covered_is_inclusive_and_checked() -> None:
    cqr = _cqr()
    cqr.loc[0, "true_L"] = cqr.loc[0, "lower"]
    cqr.loc[0, "covered"] = True
    stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)
    cqr.loc[0, "covered"] = False
    with pytest.raises(ValueError, match="covered"):
        stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)


def test_decision_expected_n_guard() -> None:
    with pytest.raises(ValueError, match="1844"):
        stage3b.build_cqr_evaluation_table(_base(), _cqr())


@pytest.mark.parametrize("role", ["CP_CALIBRATION", "TRAIN", "MODEL_VALIDATION", "RANDOM_TEST"])
def test_non_decision_roles_are_rejected(role: str) -> None:
    cqr = _cqr()
    cqr["role"] = role
    with pytest.raises((PermissionError, ValueError)):
        stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)


def test_duplicate_sample_is_rejected() -> None:
    cqr = _cqr()
    cqr.loc[1, "sample_id"] = cqr.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="sample_id"):
        stage3b.validate_cqr_intervals(cqr, enforce_expected_n=False)


@pytest.mark.parametrize("field", ["image_path", "date", "timestamp", "true_L"])
def test_exact_alignment_fields_are_guarded(field: str) -> None:
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
        stage3b.build_cqr_evaluation_table(_base(), cqr, enforce_expected_n=False)


def test_sample_set_alignment_is_not_row_order_only() -> None:
    cqr = _cqr().iloc[::-1].reset_index(drop=True)
    evaluation = stage3b.build_cqr_evaluation_table(
        _base(), cqr, enforce_expected_n=False
    )
    assert evaluation["sample_id"].tolist() == _base()["sample_id"].tolist()
    cqr.loc[0, "sample_id"] = "synthetic-extra"
    with pytest.raises(ValueError, match="sample_id set mismatch"):
        stage3b.build_cqr_evaluation_table(_base(), cqr, enforce_expected_n=False)


def test_mc_mean_comes_from_frozen_stage2a_base() -> None:
    base = _base()
    altered = base.copy()
    altered["mc_mean"] += 0.005
    first = stage3b.build_cqr_evaluation_table(base, _cqr(), enforce_expected_n=False)
    second = stage3b.build_cqr_evaluation_table(altered, _cqr(), enforce_expected_n=False)
    assert not np.array_equal(first["abs_error_mc_mean"], second["abs_error_mc_mean"])


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.assign(date="2017-06-15"),
        lambda frame: frame.assign(image_path="synthetic/random_test/image.jpg"),
    ],
)
def test_sealed_dates_and_random_test_locators_are_rejected(mutator) -> None:
    with pytest.raises(PermissionError):
        stage3b.validate_cqr_evaluation_table(
            mutator(_evaluation()), enforce_expected_n=False
        )


def test_spearman_is_exact_stage2a_helper_result() -> None:
    evaluation = _evaluation()
    actual = stage3b.build_cqr_spearman(evaluation).iloc[0].to_dict()
    expected = stage3b.stage2a.summarize_risk_score(
        stage3b.CQR_RISK_SCORE,
        evaluation[stage3b.CQR_RISK_SCORE],
        evaluation["abs_error_mc_mean"],
    )
    assert actual == expected


def test_risk_rounding_uses_twelve_decimals() -> None:
    values = np.array([0.1, 0.1 + 4e-13, 0.1 + 2e-12])
    rounded = stage3b.stage2a.risk_rank_values(values)
    assert rounded[0] == rounded[1]
    assert rounded[2] != rounded[0]
    assert stage3b.RISK_TIE_ROUND_DECIMALS == 12


def test_higher_width_means_higher_risk() -> None:
    risk = np.arange(1.0, 21.0)
    rho, constant, _ = stage3b.stage2a.spearman_with_average_ties(risk, risk)
    assert rho == pytest.approx(1.0)
    assert constant is False
    assert "retain lowest risk; reject highest risk" == stage3b.make_config()[
        "risk_retention_direction"
    ]


def test_fixed_method_order_appends_cqr_without_metric_sorting() -> None:
    comparison = stage3b.build_spearman_comparison(
        _baseline_spearman(), stage3b.build_cqr_spearman(_evaluation())
    )
    assert tuple(comparison["risk_score"]) == stage3b.METHOD_ORDER
    assert not stage3b._forbidden_presentation_columns(comparison)
    assert "sort_values" not in inspect.getsource(stage3b.build_spearman_comparison)


def test_stage2b_coverage_grid_is_frozen() -> None:
    assert stage3b.stage2b.COVERAGE_GRID == tuple(np.arange(0.10, 1.001, 0.05).round(2))


@pytest.mark.parametrize(
    ("coverage", "n", "expected"),
    [(0.10, 1844, 184), (0.20, 1844, 368), (0.30, 1844, 553), (1.0, 20, 20)],
)
def test_floor_coverage_count_rule(coverage: float, n: int, expected: int) -> None:
    assert stage3b.stage2b.target_retained_count(coverage, n) == expected


def test_fractional_retention_at_rounded_risk_tie() -> None:
    risk = np.array([0.1, 0.2, 0.2 + 4e-13, 0.9])
    absolute = np.array([1.0, 2.0, 4.0, 8.0])
    metrics = stage3b.stage2b.tie_aware_retention_metrics(
        risk, absolute, absolute**2, 2, use_stage2a_risk_rounding=True
    )
    assert metrics["boundary_tie_group_size"] == 2
    assert metrics["boundary_fraction"] == pytest.approx(0.5)
    assert metrics["MAE_mc_mean"] == pytest.approx((1.0 + 0.5 * 6.0) / 2.0)
    assert metrics["RMSE_mc_mean"] == pytest.approx(math.sqrt((1.0 + 0.5 * 20.0) / 2.0))


def test_cqr_curve_matches_stage2b_reference_at_every_coverage() -> None:
    evaluation = _evaluation()
    actual = stage3b.build_cqr_risk_coverage_curve(evaluation)
    risk = evaluation[stage3b.CQR_RISK_SCORE].to_numpy()
    absolute = evaluation["abs_error_mc_mean"].to_numpy()
    squared = evaluation["sq_error_mc_mean"].to_numpy()
    records = []
    for coverage in stage3b.stage2b.COVERAGE_GRID:
        metrics = stage3b.stage2b.tie_aware_retention_metrics(
            risk,
            absolute,
            squared,
            stage3b.stage2b.target_retained_count(coverage, len(evaluation)),
            use_stage2a_risk_rounding=True,
        )
        records.append(
            {
                "risk_score": stage3b.CQR_RISK_SCORE,
                "evaluation_role": stage3b.EVALUATION_ROLE,
                "coverage_requested": coverage,
                **metrics,
            }
        )
    expected = pd.DataFrame(records).loc[:, stage3b.stage2b.CURVE_COLUMNS]
    pd.testing.assert_frame_equal(actual, expected)


def test_oracle_uses_exact_unrounded_absolute_error() -> None:
    evaluation = _evaluation()
    exact_errors = np.concatenate(
        ([0.01, 0.01 + 4e-13], np.linspace(0.10, 0.27, len(evaluation) - 2))
    )
    evaluation["mc_mean"] = evaluation["true_L"] - exact_errors
    evaluation["abs_error_mc_mean"] = np.abs(
        evaluation["true_L"] - evaluation["mc_mean"]
    )
    evaluation["sq_error_mc_mean"] = evaluation["abs_error_mc_mean"] ** 2
    oracle = stage3b.build_oracle_curve(evaluation)
    expected = stage3b.stage2b.build_curve_for_score(
        stage3b.stage2b.ORACLE,
        evaluation["abs_error_mc_mean"],
        evaluation["abs_error_mc_mean"],
        evaluation["sq_error_mc_mean"],
        oracle=True,
    )
    pd.testing.assert_frame_equal(oracle, expected)
    low = oracle.iloc[0]
    assert low["boundary_tie_group_size"] == 1


def test_coverage_one_converges_to_all_sample_mae_rmse() -> None:
    evaluation = _evaluation()
    full = stage3b.build_cqr_risk_coverage_curve(evaluation).iloc[-1]
    assert full["MAE_mc_mean"] == pytest.approx(evaluation["abs_error_mc_mean"].mean())
    assert full["RMSE_mc_mean"] == pytest.approx(
        np.sqrt(evaluation["sq_error_mc_mean"].mean())
    )


def test_ause_is_stage2b_trapezoid_without_span_normalization() -> None:
    evaluation = _evaluation()
    curve = stage3b.build_cqr_risk_coverage_curve(evaluation)
    oracle = stage3b.build_oracle_curve(evaluation)
    summary = stage3b.build_cqr_ause_summary(curve, oracle, len(evaluation)).iloc[0]
    x = np.asarray(stage3b.stage2b.COVERAGE_GRID)
    expected_mae = stage3b.stage2b.trapezoidal_integral(
        curve["MAE_mc_mean"].to_numpy() - oracle["MAE_mc_mean"].to_numpy(), x
    )
    expected_rmse = stage3b.stage2b.trapezoidal_integral(
        curve["RMSE_mc_mean"].to_numpy() - oracle["RMSE_mc_mean"].to_numpy(), x
    )
    assert summary["AUSE_MAE"] == pytest.approx(expected_mae)
    assert summary["AUSE_RMSE"] == pytest.approx(expected_rmse)
    assert stage3b.make_config()["ause_coverage_span_normalized"] is False


def test_risk_coverage_comparison_only_appends_cqr() -> None:
    evaluation = _evaluation()
    cqr_curve = stage3b.build_cqr_risk_coverage_curve(evaluation)
    comparison = stage3b.build_risk_coverage_comparison(
        _baseline_curves(evaluation),
        cqr_curve,
        evaluation["abs_error_mc_mean"],
        evaluation["sq_error_mc_mean"],
    )
    assert tuple(dict.fromkeys(comparison["risk_score"])) == stage3b.CURVE_COMPARISON_ORDER
    assert len(comparison) == 9 * len(stage3b.stage2b.COVERAGE_GRID)


def test_stage2c_high_error_target_is_frozen() -> None:
    audit = {
        "target_high_error_n": 184,
        "high_error_threshold_abs_error": 0.0853964442488999,
        "n_strict_high_error": 183,
        "target_boundary_tie_group_size": 1,
        "target_boundary_fraction": 1.0,
    }
    stage3b.validate_frozen_high_error_target(audit, 1844)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_high_error_n", 185),
        ("high_error_threshold_abs_error", 0.0854),
        ("n_strict_high_error", 184),
        ("target_boundary_tie_group_size", 2),
        ("target_boundary_fraction", 0.5),
    ],
)
def test_frozen_high_error_target_drift_is_rejected(field: str, value: float) -> None:
    audit = {
        "target_high_error_n": 184,
        "high_error_threshold_abs_error": 0.0853964442488999,
        "n_strict_high_error": 183,
        "target_boundary_tie_group_size": 1,
        "target_boundary_fraction": 1.0,
    }
    audit[field] = value
    with pytest.raises(ValueError):
        stage3b.validate_frozen_high_error_target(audit, 1844)


def test_stage2c_fixed_budgets_and_selected_counts() -> None:
    assert stage3b.stage2c.RISK_BUDGET_FRACTIONS == (0.10, 0.20, 0.30)
    assert tuple(
        stage3b.stage2c.risk_budget_count(fraction, 1844)
        for fraction in stage3b.stage2c.RISK_BUDGET_FRACTIONS
    ) == (184, 368, 553)


def test_high_risk_first_fractional_boundary_selection() -> None:
    risk = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8 + 4e-13, 0.9])
    weights, audit = stage3b.stage2c.build_risk_selection(risk, 0.20)
    assert weights[-1] == 1.0
    assert weights[7] == pytest.approx(0.5)
    assert weights[8] == pytest.approx(0.5)
    assert audit["risk_boundary_tie_group_size"] == 2
    assert audit["risk_boundary_fraction"] == pytest.approx(0.5)


def test_capture_lift_and_random_baseline_match_stage2c() -> None:
    target = np.array([1.0, 1.0, 0.0, 0.0])
    selected = np.array([1.0, 0.0, 1.0, 0.0])
    metrics = stage3b.stage2c.capture_metrics(target, selected, 2, 2)
    assert metrics["expected_captured_high_error"] == 1.0
    assert metrics["capture_rate"] == 0.5
    assert metrics["random_capture_rate"] == 0.5
    assert metrics["random_expected_captured_high_error"] == 1.0
    assert metrics["capture_lift_vs_random"] == 1.0


def test_cqr_capture_matches_stage2c_public_helpers() -> None:
    evaluation = _evaluation()
    actual, audit = stage3b.build_cqr_high_error_capture(
        evaluation, enforce_frozen_target=False
    )
    weights, expected_audit = stage3b.stage2c.build_high_error_target(
        evaluation["abs_error_mc_mean"]
    )
    expected_rows = []
    for budget in stage3b.stage2c.RISK_BUDGET_FRACTIONS:
        selected, selection_audit = stage3b.stage2c.build_risk_selection(
            evaluation[stage3b.CQR_RISK_SCORE], budget
        )
        metrics = stage3b.stage2c.capture_metrics(
            weights,
            selected,
            int(expected_audit["target_high_error_n"]),
            int(selection_audit["risk_budget_n"]),
        )
        expected_rows.append(
            {
                "risk_score": stage3b.CQR_RISK_SCORE,
                "evaluation_role": stage3b.EVALUATION_ROLE,
                "N": len(evaluation),
                "high_error_target_fraction": stage3b.stage2c.HIGH_ERROR_TARGET_FRACTION,
                "target_high_error_n": expected_audit["target_high_error_n"],
                "high_error_threshold_abs_error": expected_audit[
                    "high_error_threshold_abs_error"
                ],
                **selection_audit,
                **metrics,
            }
        )
    expected = pd.DataFrame(expected_rows).loc[:, stage3b.stage2c.SUMMARY_COLUMNS]
    pd.testing.assert_frame_equal(actual, expected)
    assert audit == expected_audit


def test_capture_comparison_only_appends_cqr() -> None:
    cqr_capture, _ = stage3b.build_cqr_high_error_capture(
        _evaluation(), enforce_frozen_target=False
    )
    comparison = stage3b.build_capture_comparison(
        _baseline_capture(cqr_capture), cqr_capture
    )
    assert tuple(dict.fromkeys(comparison["risk_score"])) == stage3b.METHOD_ORDER


@pytest.mark.parametrize("forbidden", ["winner", "best", "rank", "ranking"])
def test_comparisons_reject_winner_and_rank_fields(forbidden: str) -> None:
    baseline = _baseline_spearman()
    baseline[forbidden] = 1
    with pytest.raises(ValueError):
        stage3b.build_spearman_comparison(
            baseline, stage3b.build_cqr_spearman(_evaluation())
        )


@pytest.mark.parametrize("field", list(stage3b.REQUIRED_STAGE3A2_PROVENANCE))
def test_stage3a2_provenance_guards_every_frozen_field(field: str) -> None:
    provenance = _provenance()
    value = provenance[field]
    provenance[field] = (not value) if isinstance(value, bool) else "drifted"
    with pytest.raises(ValueError):
        stage3b.validate_stage3a2_provenance(provenance)


def test_checkpoint_sha_guard_and_shape() -> None:
    sha = stage3b.SOURCE_CQR_CHECKPOINT_SHA256
    assert len(sha) == 64
    assert set(sha) <= set("0123456789abcdef")
    stage3b.validate_stage3a2_provenance(_provenance())
    bad = _provenance()
    bad["source_cqr_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        stage3b.validate_stage3a2_provenance(bad)


@pytest.mark.parametrize(
    "path",
    [Path("synthetic/random_test.csv"), Path("synthetic/cp_calibration.csv")],
)
def test_forbidden_input_paths_are_rejected_before_io(path: Path) -> None:
    with pytest.raises(PermissionError):
        stage3b.validate_authorized_input_path(path, "cqr_predictions")


def test_unauthorized_input_path_is_rejected() -> None:
    with pytest.raises(PermissionError):
        stage3b.validate_authorized_input_path(
            Path("synthetic/decision.csv"), "cqr_predictions"
        )


def test_output_collision_protection(tmp_path: Path) -> None:
    output = tmp_path / "stage3b"
    output.mkdir()
    (output / "existing.txt").write_text("synthetic", encoding="utf-8")
    with pytest.raises(FileExistsError):
        stage3b.ensure_output_available(output)


def test_synthetic_writer_uses_only_expected_output_names(tmp_path: Path) -> None:
    evaluation = _evaluation()
    spearman = stage3b.build_cqr_spearman(evaluation)
    curve = stage3b.build_cqr_risk_coverage_curve(evaluation)
    oracle = stage3b.build_oracle_curve(evaluation)
    ause = stage3b.build_cqr_ause_summary(curve, oracle, len(evaluation))
    capture, _ = stage3b.build_cqr_high_error_capture(
        evaluation, enforce_frozen_target=False
    )
    output = tmp_path / "stage3b"
    stage3b.write_outputs(
        output,
        spearman,
        curve,
        ause,
        capture,
        stage3b.build_spearman_comparison(_baseline_spearman(), spearman),
        stage3b.build_risk_coverage_comparison(
            _baseline_curves(evaluation),
            curve,
            evaluation["abs_error_mc_mean"],
            evaluation["sq_error_mc_mean"],
        ),
        stage3b.build_capture_comparison(_baseline_capture(capture), capture),
        stage3b.make_config(),
        stage3b.make_provenance(),
    )
    assert {path.name for path in output.iterdir()} == {
        "cqr_risk_score_validity.csv",
        "cqr_risk_coverage_curve.csv",
        "cqr_risk_coverage_summary.csv",
        "cqr_high_error_capture.csv",
        "risk_score_comparison_with_stage2a.csv",
        "risk_coverage_comparison_with_stage2b.csv",
        "high_error_capture_comparison_with_stage2c.csv",
        "config.json",
        "provenance.json",
    }


@pytest.mark.parametrize(
    "field",
    [
        "cqr_q50_error_evaluated",
        "raw_cqr_width_evaluated",
        "multiple_cqr_risk_scores_tested",
        "stage2a_definitions_modified",
        "stage2b_definitions_modified",
        "stage2c_definitions_modified",
        "stage3a2_modified",
        "risk_target_modified",
        "new_risk_metrics_added",
        "formal_risk_method_selected",
        "formal_winner_declared",
        "random_test_accessed",
        "sealed_final_dates_accessed",
        "cp_calibration_used_for_risk_evaluation",
        "training_performed",
        "image_inference_performed",
        "mc_dropout_performed",
        "conformal_recalibration_performed",
        "cleaning_decision_performed",
        "economic_decision_performed",
    ],
)
def test_provenance_forbidden_action_flags_are_false(field: str) -> None:
    assert stage3b.make_provenance()[field] is False


def test_provenance_records_one_common_target_and_one_cqr_score() -> None:
    provenance = stage3b.make_provenance()
    assert provenance["common_risk_target"] == "abs(true_L - mc_mean)"
    assert provenance["cqr_risk_score"] == "cqr_conformal_width"
    assert provenance["risk_score_round_decimals"] == 12
    assert provenance["N"] == 1844
    assert provenance["evaluation_role"] == "DECISION_DEVELOPMENT"


def test_stage3b_module_has_no_model_or_recalibration_implementation() -> None:
    source = inspect.getsource(stage3b)
    forbidden_fragments = (
        "torch.",
        "model(",
        "enable_mc_dropout_only",
        "cqr_conformity_scores(",
        "calibrate_cqr(",
        "conformalize_decision_intervals(",
        "abs(true_L - q50)",
    )
    for fragment in forbidden_fragments:
        assert fragment not in source
    assert "sort_values(" not in source


def test_tests_use_no_formal_output_paths() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "outputs/" + "paper1_clean_random_v1" not in source
    assert "pd." + "read_csv(" not in source
    assert "stage3b." + "run(" not in source
