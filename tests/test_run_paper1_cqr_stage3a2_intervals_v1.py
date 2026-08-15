from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.run_paper1_cqr_stage3a2_intervals_v1 as stage3a2


def manifest_frame(
    role: str,
    n: int = 3,
    *,
    prefix: str = "sample",
    date: str = "2017-06-13",
    truths: list[float] | None = None,
    irradiances: list[float] | None = None,
) -> pd.DataFrame:
    truths = truths or [0.2 + index * 0.1 for index in range(n)]
    irradiances = irradiances or [0.3 + index * 0.1 for index in range(n)]
    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}_{index}" for index in range(n)],
            "date": [date] * n,
            "timestamp": [f"{date}T10:{index:02d}:00" for index in range(n)],
            "image_path": [
                "data/raw/PanelImages/"
                f"{prefix}_{index}_L_{truths[index]}_I_{irradiances[index]}.jpg"
                for index in range(n)
            ],
            "role": [role] * n,
        }
    )


def stage3a1_predictions(
    role: str,
    n: int = 3,
    *,
    prefix: str = "sample",
    truths: list[float] | None = None,
    q05: list[float] | None = None,
    q50: list[float] | None = None,
    q95: list[float] | None = None,
    irradiances: list[float] | None = None,
) -> pd.DataFrame:
    q05 = q05 or [0.1 + index * 0.1 for index in range(n)]
    q50 = q50 or [0.2 + index * 0.1 for index in range(n)]
    q95 = q95 or [0.4 + index * 0.1 for index in range(n)]
    irradiances = irradiances or [0.3 + index * 0.1 for index in range(n)]
    metadata = manifest_frame(
        role,
        n,
        prefix=prefix,
        truths=truths,
        irradiances=irradiances,
    )
    result = metadata.copy()
    result["irradiance"] = irradiances
    result["q05"] = q05
    result["q50"] = q50
    result["q95"] = q95
    return result.loc[:, stage3a2.stage3a1.PREDICTION_COLUMNS]


def aligned_frame(
    role: str,
    n: int = 3,
    *,
    prefix: str = "sample",
    truths: list[float] | None = None,
    irradiances: list[float] | None = None,
    q05: list[float] | None = None,
    q50: list[float] | None = None,
    q95: list[float] | None = None,
) -> pd.DataFrame:
    predictions = stage3a1_predictions(
        role,
        n,
        prefix=prefix,
        truths=truths,
        irradiances=irradiances,
        q05=q05,
        q50=q50,
        q95=q95,
    )
    truth = manifest_frame(
        role,
        n,
        prefix=prefix,
        truths=truths,
        irradiances=irradiances,
    )
    return stage3a2.attach_truth_by_sample_id(
        predictions, truth, role, enforce_expected_n=False
    )


def frozen_calibration(qhat: float) -> stage3a2.FrozenCQRCalibration:
    return stage3a2.FrozenCQRCalibration(
        calibration_role=stage3a2.CP_CALIBRATION_ROLE,
        n_calibration_scores=3,
        alpha=stage3a2.ALPHA,
        target_coverage=stage3a2.TARGET_COVERAGE,
        quantile_fraction=1.0,
        order_statistic_rank=3,
        quantile_method="higher",
        score_definition=stage3a2.CQR_SCORE_DEFINITION,
        qhat=qhat,
        qhat_is_negative=qhat < 0.0,
        qhat_is_positive=qhat > 0.0,
    )


def interval_predictions(
    *,
    truths: list[float] | None = None,
    q05: list[float] | None = None,
    q50: list[float] | None = None,
    q95: list[float] | None = None,
    irradiances: list[float] | None = None,
    qhat: float = 0.0,
) -> pd.DataFrame:
    lengths = [
        len(value)
        for value in (truths, q05, q50, q95, irradiances)
        if value is not None
    ]
    n = lengths[0] if lengths else 3
    decision = aligned_frame(
        stage3a2.DECISION_DEVELOPMENT_ROLE,
        n,
        prefix="decision",
        truths=truths,
        irradiances=irradiances,
        q05=q05,
        q50=q50,
        q95=q95,
    )
    return stage3a2.conformalize_decision_intervals(
        decision, frozen_calibration(qhat), enforce_expected_n=False
    )


def baseline_metrics_frame() -> pd.DataFrame:
    rows = []
    for index, method in enumerate(stage3a2.BASELINE_METHOD_ORDER[:-1]):
        rows.append(
            {
                "method": method,
                "evaluation_role": stage3a2.DECISION_DEVELOPMENT_ROLE,
                "N": 1844,
                "alpha": 0.10,
                "target_coverage": 0.90,
                "PICP": 0.80 + index * 0.01,
                "MPIW": 0.20 + index * 0.01,
                "median_width": 0.19 + index * 0.01,
                "coverage_error": 0.10 - index * 0.01,
                "mean_interval_score_alpha_0p10": 0.30 + index * 0.01,
                "fallback_count": 0,
                "fallback_rate": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_frozen_protocol_constants_match_stage1b_and_stage3a1() -> None:
    stage3a2.validate_protocol_constants()
    assert stage3a2.ALPHA == stage3a2.stage1b.ALPHA == 0.10
    assert stage3a2.TARGET_COVERAGE == stage3a2.stage1b.TARGET_COVERAGE == 0.90
    assert stage3a2.QUANTILE_METHOD == "higher"
    assert stage3a2.QUANTILE_LEVELS == (0.05, 0.50, 0.95)


def test_standard_cqr_score_formula_and_negative_scores() -> None:
    scores = stage3a2.cqr_conformity_scores(
        truth=[0.5, 0.95, 0.02],
        q05=[0.1, 0.2, 0.1],
        q95=[0.9, 0.8, 0.7],
    )
    assert scores == pytest.approx([-0.4, 0.15, 0.08])
    assert scores[0] < 0.0


def test_cqr_score_does_not_clamp_at_zero() -> None:
    source = inspect.getsource(stage3a2.cqr_conformity_scores)
    assert "np.zeros" not in source
    assert "maximum.reduce" not in source
    scores = stage3a2.cqr_conformity_scores([0.5], [0.1], [0.9])
    assert scores.tolist() == pytest.approx([-0.4])


@pytest.mark.parametrize(
    "values", [[np.nan], [np.inf], [-np.inf]]
)
def test_cqr_score_finite_guard(values: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        stage3a2.cqr_conformity_scores(values, [0.1], [0.9])


def test_cqr_score_shape_n_and_crossing_guards() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        stage3a2.cqr_conformity_scores([[0.2]], [0.1], [0.3])
    with pytest.raises(ValueError, match="N guard"):
        stage3a2.cqr_conformity_scores([0.2], [0.1], [0.3], expected_n=2)
    with pytest.raises(ValueError, match="crossing"):
        stage3a2.cqr_conformity_scores([0.2], [0.4], [0.3])


def test_finite_sample_rank_fraction_and_known_positive_qhat() -> None:
    calibration = aligned_frame(
        stage3a2.CP_CALIBRATION_ROLE,
        truths=[0.5, 0.95, 0.02],
        q05=[0.1, 0.2, 0.1],
        q50=[0.5, 0.5, 0.4],
        q95=[0.9, 0.8, 0.7],
    )
    result = stage3a2.calibrate_cqr(calibration, enforce_expected_n=False)
    assert result.n_calibration_scores == 3
    assert result.order_statistic_rank == 3
    assert result.quantile_fraction == 1.0
    assert result.quantile_method == "higher"
    assert result.qhat == pytest.approx(0.15)
    assert result.qhat_is_positive is True
    assert result.qhat_is_negative is False


def test_finite_sample_nonclipped_rank_fraction_is_18_over_19(monkeypatch) -> None:
    n = 19
    calibration = aligned_frame(
        stage3a2.CP_CALIBRATION_ROLE,
        n,
        truths=[0.5] * n,
        irradiances=[0.3] * n,
        q05=[0.1] * n,
        q50=[0.5] * n,
        q95=[0.9] * n,
    )
    monkeypatch.setattr(stage3a2.stage1b, "conformal_quantile", lambda scores: -0.4)
    result = stage3a2.calibrate_cqr(calibration, enforce_expected_n=False)
    assert result.order_statistic_rank == 18
    assert result.quantile_fraction == pytest.approx(18 / 19)


def test_known_negative_qhat_is_preserved_and_calibration_is_frozen() -> None:
    calibration = aligned_frame(
        stage3a2.CP_CALIBRATION_ROLE,
        truths=[0.5, 0.5, 0.5],
        q05=[0.1, 0.1, 0.1],
        q50=[0.5, 0.5, 0.5],
        q95=[0.9, 0.9, 0.9],
    )
    result = stage3a2.calibrate_cqr(calibration, enforce_expected_n=False)
    assert result.qhat == pytest.approx(-0.4)
    assert result.qhat_is_negative is True
    with pytest.raises(FrozenInstanceError):
        result.qhat = 0.0


def test_decision_truth_cannot_enter_calibration() -> None:
    decision = aligned_frame(stage3a2.DECISION_DEVELOPMENT_ROLE)
    with pytest.raises(PermissionError, match="Role guard"):
        stage3a2.calibrate_cqr(decision, enforce_expected_n=False)


def test_quantile_implementation_is_exact_stage1b_helper(monkeypatch) -> None:
    observed: dict[str, np.ndarray] = {}

    def fake_quantile(scores):
        observed["scores"] = np.asarray(scores)
        return -0.125

    monkeypatch.setattr(stage3a2.stage1b, "conformal_quantile", fake_quantile)
    calibration = aligned_frame(stage3a2.CP_CALIBRATION_ROLE)
    result = stage3a2.calibrate_cqr(calibration, enforce_expected_n=False)
    assert result.qhat == -0.125
    assert len(observed["scores"]) == 3


def test_stage1b_conformal_quantile_uses_method_higher(monkeypatch) -> None:
    observed: dict[str, object] = {}
    original = np.quantile

    def capture(values, q, *, method):
        observed["method"] = method
        return original(values, q, method=method)

    monkeypatch.setattr(stage3a2.stage1b.np, "quantile", capture)
    stage3a2.stage1b.conformal_quantile(np.array([-0.2, 0.1, 0.3]))
    assert observed["method"] == "higher"


def test_conformal_correction_occurs_before_clipping() -> None:
    predictions = interval_predictions(
        truths=[0.2],
        q05=[0.05],
        q50=[0.5],
        q95=[0.9],
        irradiances=[0.4],
        qhat=0.1,
    )
    assert predictions.loc[0, "lower"] == 0.0
    assert predictions.loc[0, "upper"] == 1.0
    assert bool(predictions.loc[0, "lower_clipped"]) is True
    assert bool(predictions.loc[0, "upper_clipped"]) is False
    assert predictions.loc[0, "raw_width"] == pytest.approx(0.85)


def test_negative_qhat_contracts_without_clamping() -> None:
    predictions = interval_predictions(
        truths=[0.5],
        q05=[0.1],
        q50=[0.5],
        q95=[0.9],
        irradiances=[0.4],
        qhat=-0.05,
    )
    assert predictions.loc[0, "lower"] == pytest.approx(0.15)
    assert predictions.loc[0, "upper"] == pytest.approx(0.85)
    diagnostics = stage3a2.interval_diagnostics(
        predictions, frozen_calibration(-0.05)
    )
    assert diagnostics["interval_correction_direction"] == "contraction"


def test_positive_qhat_expands_interval() -> None:
    predictions = interval_predictions(
        truths=[0.5],
        q05=[0.2],
        q50=[0.5],
        q95=[0.8],
        irradiances=[0.4],
        qhat=0.05,
    )
    assert predictions.loc[0, "lower"] == pytest.approx(0.15)
    assert predictions.loc[0, "upper"] == pytest.approx(0.85)
    diagnostics = stage3a2.interval_diagnostics(
        predictions, frozen_calibration(0.05)
    )
    assert diagnostics["interval_correction_direction"] == "expansion"


def test_lower_greater_than_upper_is_rejected_without_sorting() -> None:
    decision = aligned_frame(
        stage3a2.DECISION_DEVELOPMENT_ROLE,
        1,
        prefix="decision",
        truths=[0.5],
        irradiances=[0.4],
        q05=[0.4],
        q50=[0.5],
        q95=[0.6],
    )
    with pytest.raises(ValueError, match="lower exceeds upper"):
        stage3a2.conformalize_decision_intervals(
            decision, frozen_calibration(-0.2), enforce_expected_n=False
        )
    source = inspect.getsource(stage3a2.conformalize_decision_intervals)
    assert ".sort" not in source
    assert "minimum" not in source


def test_global_picp_is_inclusive_and_width_metrics_match() -> None:
    predictions = interval_predictions(
        truths=[0.1, 0.8],
        q05=[0.1, 0.2],
        q50=[0.2, 0.5],
        q95=[0.3, 0.8],
        irradiances=[0.2, 0.8],
    )
    metrics = stage3a2.compute_global_metrics(
        predictions, enforce_expected_n=False
    )
    assert metrics["PICP"] == 1.0
    assert metrics["MPIW"] == pytest.approx(0.4)
    assert metrics["median_width"] == pytest.approx(0.4)
    assert metrics["coverage_error"] == pytest.approx(0.1)


def test_interval_score_lower_upper_miss_and_boundary_no_penalty() -> None:
    score = stage3a2.stage1b.standard_interval_score(
        truth=[0.0, 1.0, 0.1, 0.9],
        lower=[0.1, 0.1, 0.1, 0.1],
        upper=[0.9, 0.9, 0.9, 0.9],
    )
    assert score == pytest.approx([2.8, 2.8, 0.8, 0.8])


def test_global_metrics_are_exact_stage1b_helper_equivalent() -> None:
    predictions = interval_predictions(qhat=0.03)
    expected = stage3a2.stage1b.compute_interval_metrics(predictions)
    observed = stage3a2.compute_global_metrics(
        predictions, enforce_expected_n=False
    )
    assert observed == expected
    assert tuple(observed) == stage3a2.GLOBAL_METRIC_COLUMNS


def test_raw_q05_q95_diagnostics_are_only_three_allowed_fields() -> None:
    decision = aligned_frame(stage3a2.DECISION_DEVELOPMENT_ROLE)
    diagnostics = stage3a2.raw_q05_q95_diagnostics(decision)
    assert set(diagnostics) == {
        "raw_q05_q95_PICP",
        "raw_q05_q95_MPIW",
        "raw_q05_q95_median_width",
    }


def test_conditional_axes_fixed_bins_order_and_q50_label() -> None:
    predictions = interval_predictions(
        truths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        q05=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        q50=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        q95=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        irradiances=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    diagnostics = stage3a2.conditional_coverage_diagnostics(
        predictions, enforce_expected_n=False
    )
    assert tuple(diagnostics["binning_variable"].drop_duplicates()) == (
        "pred_L=q50",
        "irradiance",
    )
    assert "pred_L=mc_mean" not in set(diagnostics["binning_variable"])
    for axis in ("pred_L=q50", "irradiance"):
        subset = diagnostics[diagnostics["binning_variable"] == axis]
        assert tuple(subset["bin_label"]) == stage3a2.stage1b.FIXED_BIN_LABELS
        assert int(subset["N"].sum()) == 6


def test_conditional_include_lowest_and_right_semantics() -> None:
    labels = stage3a2.stage1b.assign_fixed_bins(
        [0.0, 0.2, 0.2000001, 1.0], stage3a2.stage1b.PRED_L_BINS
    )
    assert labels.astype(str).tolist() == [
        "[0.0,0.2]",
        "[0.0,0.2]",
        "(0.2,0.4]",
        "(0.8,1.0]",
    ]


def test_conditional_empty_and_small_bins_have_no_fallback() -> None:
    predictions = interval_predictions(
        truths=[0.1],
        q05=[0.0],
        q50=[0.1],
        q95=[0.2],
        irradiances=[0.1],
    )
    diagnostics = stage3a2.conditional_coverage_diagnostics(
        predictions, enforce_expected_n=False
    )
    q50_axis = diagnostics[diagnostics["binning_variable"] == "pred_L=q50"]
    first = q50_axis.iloc[0]
    second = q50_axis.iloc[1]
    assert first["N"] == 1
    assert first["PICP"] == 1.0
    assert second["N"] == 0
    assert pd.isna(second["PICP"])
    assert pd.isna(second["MPIW"])
    assert not any("fallback" in column.lower() for column in diagnostics.columns)


def test_cp_and_decision_n_guards() -> None:
    cp = aligned_frame(stage3a2.CP_CALIBRATION_ROLE)
    with pytest.raises(ValueError, match="N guard failed"):
        stage3a2.calibrate_cqr(cp)
    decision = interval_predictions()
    with pytest.raises(ValueError, match="N guard failed"):
        stage3a2.compute_global_metrics(decision)
    assert stage3a2.EXPECTED_N == {
        "CP_CALIBRATION": 2951,
        "DECISION_DEVELOPMENT": 1844,
    }


@pytest.mark.parametrize("field", ["image_path", "role", "date", "timestamp"])
def test_truth_alignment_metadata_mismatch_is_rejected(field: str) -> None:
    predictions = stage3a1_predictions(stage3a2.CP_CALIBRATION_ROLE)
    truth = manifest_frame(stage3a2.CP_CALIBRATION_ROLE)
    if field == "role":
        truth.loc[0, field] = stage3a2.DECISION_DEVELOPMENT_ROLE
        message = "Role guard"
    elif field == "date":
        truth.loc[0, field] = "2017-06-14"
        message = field
    elif field == "timestamp":
        truth.loc[0, field] = "2017-06-13T11:00:00"
        message = field
    elif field == "image_path":
        truth.loc[0, field] = (
            "data/raw/PanelImages/different_0_L_0.2_I_0.3.jpg"
        )
        message = field
    else:
        truth.loc[0, field] = f"different_{field}"
        message = field
    with pytest.raises((ValueError, PermissionError), match=message):
        stage3a2.attach_truth_by_sample_id(
            predictions,
            truth,
            stage3a2.CP_CALIBRATION_ROLE,
            enforce_expected_n=False,
        )


def test_truth_alignment_is_by_sample_id_not_row_order() -> None:
    predictions = stage3a1_predictions(
        stage3a2.CP_CALIBRATION_ROLE, truths=[0.2, 0.3, 0.4]
    )
    truth = manifest_frame(
        stage3a2.CP_CALIBRATION_ROLE, truths=[0.2, 0.3, 0.4]
    ).iloc[::-1]
    aligned = stage3a2.attach_truth_by_sample_id(
        predictions,
        truth,
        stage3a2.CP_CALIBRATION_ROLE,
        enforce_expected_n=False,
    )
    assert aligned["sample_id"].tolist() == predictions["sample_id"].tolist()
    assert aligned["true_L"].tolist() == pytest.approx([0.2, 0.3, 0.4])


def test_sample_id_and_irradiance_alignment_mismatches_are_rejected() -> None:
    predictions = stage3a1_predictions(stage3a2.CP_CALIBRATION_ROLE)
    truth = manifest_frame(stage3a2.CP_CALIBRATION_ROLE)
    truth.loc[0, "sample_id"] = "truth_only"
    with pytest.raises(ValueError, match="sample_id sets differ"):
        stage3a2.attach_truth_by_sample_id(
            predictions,
            truth,
            stage3a2.CP_CALIBRATION_ROLE,
            enforce_expected_n=False,
        )

    truth = manifest_frame(stage3a2.CP_CALIBRATION_ROLE)
    predictions.loc[0, "irradiance"] += 0.01
    with pytest.raises(ValueError, match="irradiance alignment"):
        stage3a2.attach_truth_by_sample_id(
            predictions,
            truth,
            stage3a2.CP_CALIBRATION_ROLE,
            enforce_expected_n=False,
        )


def test_prediction_input_must_be_truth_free_and_truth_is_attached_separately() -> None:
    predictions = stage3a1_predictions(stage3a2.CP_CALIBRATION_ROLE)
    assert "true_L" not in predictions
    invalid = predictions.copy()
    invalid["true_L"] = 0.2
    with pytest.raises(ValueError, match="schema mismatch"):
        stage3a2.attach_truth_by_sample_id(
            invalid,
            manifest_frame(stage3a2.CP_CALIBRATION_ROLE),
            stage3a2.CP_CALIBRATION_ROLE,
            enforce_expected_n=False,
        )
    aligned = stage3a2.attach_truth_by_sample_id(
        predictions,
        manifest_frame(stage3a2.CP_CALIBRATION_ROLE),
        stage3a2.CP_CALIBRATION_ROLE,
        enforce_expected_n=False,
    )
    assert "true_L" in aligned


@pytest.mark.parametrize("overlap_field", ["sample_id", "image_path"])
def test_cp_decision_isolation(overlap_field: str) -> None:
    cp = aligned_frame(stage3a2.CP_CALIBRATION_ROLE, prefix="cp")
    decision = aligned_frame(
        stage3a2.DECISION_DEVELOPMENT_ROLE, prefix="decision"
    )
    decision.loc[0, overlap_field] = cp.loc[0, overlap_field]
    with pytest.raises(ValueError, match="overlap"):
        stage3a2.validate_cp_decision_isolation(cp, decision)


@pytest.mark.parametrize("role", ["TRAIN", "MODEL_VALIDATION", "RANDOM_TEST"])
def test_unauthorized_truth_and_prediction_roles(role: str, tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="Forbidden"):
        stage3a2.validate_prediction_input_path(tmp_path / "x.csv", role)
    with pytest.raises(PermissionError, match="Forbidden"):
        stage3a2.validate_truth_manifest_path(tmp_path / "x.csv", role)


def test_exact_input_path_guards_and_random_test_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    role = stage3a2.CP_CALIBRATION_ROLE
    prediction = tmp_path / "cp_predictions.csv"
    truth = tmp_path / "cp_manifest.csv"
    monkeypatch.setitem(stage3a2.AUTHORIZED_PREDICTION_INPUTS, role, prediction)
    monkeypatch.setitem(stage3a2.AUTHORIZED_TRUTH_MANIFESTS, role, truth)
    assert stage3a2.validate_prediction_input_path(prediction, role) == prediction.resolve()
    assert stage3a2.validate_truth_manifest_path(truth, role) == truth.resolve()
    with pytest.raises(PermissionError, match="not authorized"):
        stage3a2.validate_prediction_input_path(tmp_path / "other.csv", role)
    with pytest.raises(PermissionError, match="RANDOM_TEST"):
        stage3a2.validate_truth_manifest_path(tmp_path / "random_test.csv", role)


@pytest.mark.parametrize("sealed", sorted(stage3a2.stage3a1.SEALED_FINAL_DATES))
def test_sealed_dates_are_rejected(sealed: str) -> None:
    predictions = stage3a1_predictions(stage3a2.CP_CALIBRATION_ROLE, 1)
    truth = manifest_frame(stage3a2.CP_CALIBRATION_ROLE, 1, date=sealed)
    with pytest.raises(PermissionError, match="Sealed final date"):
        stage3a2.attach_truth_by_sample_id(
            predictions,
            truth,
            stage3a2.CP_CALIBRATION_ROLE,
            enforce_expected_n=False,
        )


def test_random_test_locator_is_rejected() -> None:
    predictions = stage3a1_predictions(stage3a2.CP_CALIBRATION_ROLE, 1)
    truth = manifest_frame(stage3a2.CP_CALIBRATION_ROLE, 1)
    truth.loc[0, "image_path"] = "data/random_test/x_L_0.2_I_0.3.jpg"
    with pytest.raises(PermissionError, match="RANDOM_TEST locator"):
        stage3a2.attach_truth_by_sample_id(
            predictions,
            truth,
            stage3a2.CP_CALIBRATION_ROLE,
            enforce_expected_n=False,
        )


def test_q_order_and_finite_guards_flow_from_stage3a1() -> None:
    decision = aligned_frame(stage3a2.DECISION_DEVELOPMENT_ROLE)
    decision.loc[0, "q05"] = 0.9
    with pytest.raises(ValueError, match="crossing"):
        stage3a2.conformalize_decision_intervals(
            decision, frozen_calibration(0.0), enforce_expected_n=False
        )
    decision = aligned_frame(stage3a2.DECISION_DEVELOPMENT_ROLE)
    decision.loc[0, "q50"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        stage3a2.conformalize_decision_intervals(
            decision, frozen_calibration(0.0), enforce_expected_n=False
        )


@pytest.mark.parametrize("field", ["width", "covered"])
def test_width_and_covered_consistency_guards(field: str) -> None:
    predictions = interval_predictions()
    if field == "width":
        predictions.loc[0, field] += 0.1
        message = "width is inconsistent"
    else:
        predictions.loc[0, field] = not bool(predictions.loc[0, field])
        message = "covered field is inconsistent"
    with pytest.raises(ValueError, match=message):
        stage3a2.validate_cqr_predictions(
            predictions, enforce_expected_n=False
        )


def test_stage1b_comparison_fixed_order_schema_and_no_ranking() -> None:
    cqr_metrics = stage3a2.compute_global_metrics(
        interval_predictions(), enforce_expected_n=False
    )
    comparison = stage3a2.build_interval_comparison(
        baseline_metrics_frame().sample(frac=1.0, random_state=7), cqr_metrics
    )
    assert tuple(comparison["method"]) == stage3a2.BASELINE_METHOD_ORDER
    assert tuple(comparison.columns) == stage3a2.COMPARISON_COLUMNS
    assert not any(
        token in column.lower()
        for column in comparison.columns
        for token in ("winner", "best", "rank")
    )


def test_stage1b_comparison_rejects_schema_method_and_role_mismatch() -> None:
    frame = baseline_metrics_frame().drop(columns="PICP")
    with pytest.raises(ValueError, match="missing fields"):
        stage3a2.validate_stage1b_global_metrics(frame)
    frame = baseline_metrics_frame()
    frame.loc[0, "method"] = frame.loc[1, "method"]
    with pytest.raises(ValueError, match="method set"):
        stage3a2.validate_stage1b_global_metrics(frame)
    frame = baseline_metrics_frame()
    frame.loc[0, "evaluation_role"] = "CP_CALIBRATION"
    with pytest.raises(ValueError, match="evaluation role"):
        stage3a2.validate_stage1b_global_metrics(frame)


def test_stage1b_metrics_exact_path_guard(tmp_path: Path, monkeypatch) -> None:
    authorized = tmp_path / "all_interval_metrics.csv"
    monkeypatch.setattr(stage3a2, "STAGE1B_GLOBAL_METRICS_INPUT", authorized)
    assert stage3a2.validate_stage1b_metrics_path(authorized) == authorized.resolve()
    with pytest.raises(PermissionError, match="Unauthorized Stage 1B"):
        stage3a2.validate_stage1b_metrics_path(tmp_path / "other.csv")


def test_output_collision_and_tmp_output_schema(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage3a2.ensure_output_available(occupied)

    predictions = interval_predictions()
    metrics = pd.DataFrame(
        [stage3a2.compute_global_metrics(predictions, enforce_expected_n=False)],
        columns=stage3a2.GLOBAL_METRIC_COLUMNS,
    )
    conditional = stage3a2.conditional_coverage_diagnostics(
        predictions, enforce_expected_n=False
    )
    comparison = stage3a2.build_interval_comparison(
        baseline_metrics_frame(), metrics.iloc[0].to_dict()
    )
    output = tmp_path / "stage3a2"
    stage3a2.write_stage3a2_outputs(
        output, predictions, metrics, conditional, {}, comparison, {}, {}
    )
    assert {path.name for path in output.iterdir()} == {
        "cqr_predictions.csv",
        "cqr_global_metrics.csv",
        "cqr_conditional_coverage.csv",
        "cqr_conformal_calibration.json",
        "interval_comparison_with_stage1b.csv",
        "config.json",
        "provenance.json",
    }


def test_formal_output_path_is_fixed(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="Unauthorized Stage 3A2 output"):
        stage3a2.validate_formal_output_path(tmp_path)


def test_calibration_payload_records_required_qhat_diagnostics() -> None:
    calibration = frozen_calibration(-0.05)
    payload = stage3a2.calibration_payload(
        calibration,
        {"raw_q05_q95_PICP": 0.9},
        {"lower_clipped_count": 1, "upper_clipped_count": 2},
    )
    assert payload["qhat"] == -0.05
    assert payload["qhat_is_negative"] is True
    assert payload["score_definition"] == stage3a2.CQR_SCORE_DEFINITION
    assert payload["finite_sample_quantile_rule"] == stage3a2.FINITE_SAMPLE_RULE


def test_config_and_provenance_freeze_protocol_and_non_action_flags() -> None:
    config = stage3a2.make_config()
    provenance = stage3a2.make_provenance(
        frozen_calibration(-0.05), {"interval_correction_direction": "contraction"}
    )
    assert config["alpha"] == 0.10
    assert config["conditional_axes"] == ["pred_L=q50", "irradiance"]
    assert config["conditional_bin_semantics"] == (
        "pd.cut(include_lowest=True,right=True)"
    )
    assert provenance["source_cqr_checkpoint_sha256"] == (
        "fd5deea62c867fcffe3791f768752da9dc3a39a1c146244b1e225d6b40b0da80"
    )
    assert provenance["qhat_selected_using_decision_truth"] is False
    assert provenance["decision_truth_used_only_for_evaluation"] is True
    for key in (
        "stage1b_metric_definitions_modified",
        "stage1b_bins_modified",
        "random_test_accessed",
        "sealed_final_dates_accessed",
        "training_performed",
        "image_inference_performed",
        "mc_dropout_performed",
        "risk_evaluation_performed",
        "cleaning_decision_performed",
        "economic_decision_performed",
        "formal_cqr_superiority_declared",
        "formal_uq_method_selected",
    ):
        assert provenance[key] is False


def test_formal_run_source_freezes_qhat_before_decision_truth_load() -> None:
    source = inspect.getsource(stage3a2.run)
    freeze_position = source.index("frozen_calibration = calibrate_cqr")
    decision_truth_position = source.index("decision_truth_manifest = load_truth_manifest")
    assert freeze_position < decision_truth_position


def test_source_has_no_training_inference_mc_risk_or_decision_actions() -> None:
    source = Path(stage3a2.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "DataLoader" not in source
    assert "enable_mc_dropout_only" not in source
    assert "model.train(" not in source
    assert "optimizer" not in source.lower()
    assert "risk_stage2" not in source
    assert "cleaning_decision(" not in source
    assert "economic_decision(" not in source
    assert "stage3a1.run(" not in source
    executable_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "run()" not in executable_lines


def test_import_has_no_formal_side_effects() -> None:
    source = inspect.getsource(stage3a2)
    assert "Importing this module reads no formal artifacts" in source
    assert "if __name__ == \"__main__\":" in source
