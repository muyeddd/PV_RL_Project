from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.run_paper1_risk_stage2a_score_validity_v1 as stage2a


def synthetic_base(
    n: int = 4,
    *,
    role: str = stage2a.PRIMARY_ROLE,
    true_l=None,
    point_pred=None,
    mc_mean=None,
    mc_std=None,
    date: str = "2017-06-13",
) -> pd.DataFrame:
    true_l = np.linspace(0.2, 0.8, n) if true_l is None else np.asarray(true_l)
    mc_mean = np.repeat(0.5, n) if mc_mean is None else np.asarray(mc_mean)
    point_pred = np.repeat(0.9, n) if point_pred is None else np.asarray(point_pred)
    mc_std = np.linspace(0.01, 0.04, n) if mc_std is None else np.asarray(mc_std)

    def expand(values):
        return np.repeat(values, n) if np.asarray(values).ndim == 0 else values

    return pd.DataFrame(
        {
            "sample_id": [f"dd-{index}" for index in range(n)],
            "date": [date] * n,
            "timestamp": [f"{date}T10:{index:02d}:00" for index in range(n)],
            "image_path": [f"data/mock/dd-{index}.jpg" for index in range(n)],
            "role": [role] * n,
            "true_L": expand(true_l),
            "irradiance": np.linspace(0.1, 0.9, n),
            "point_pred": expand(point_pred),
            "mc_mean": expand(mc_mean),
            "mc_std": expand(mc_std),
        }
    )


def synthetic_interval(
    base: pd.DataFrame,
    risk_score: str,
    widths,
) -> pd.DataFrame:
    expected_method = stage2a.STAGE1B_INPUT_SPECS[risk_score][1]
    values = np.asarray(widths, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, len(base))
    center = np.repeat(0.5, len(base))
    lower = center - values / 2.0
    upper = center + values / 2.0
    result = base.loc[:, stage2a.STAGE1A_REQUIRED_COLUMNS].copy()
    result["method"] = expected_method
    result["lower"] = lower
    result["upper"] = upper
    result["width"] = values
    truth = result["true_L"].to_numpy(dtype=float)
    result["covered"] = (truth >= lower) & (truth <= upper)
    return result


def all_interval_tables(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        risk_score: synthetic_interval(
            base, risk_score, np.linspace(0.1, 0.2, len(base))
        )
        for risk_score in stage2a.RISK_SCORE_ORDER[1:]
    }


def test_only_decision_development_is_allowed() -> None:
    frame = synthetic_base(role="CP_CALIBRATION")
    with pytest.raises(PermissionError, match="Only DECISION_DEVELOPMENT"):
        stage2a.validate_stage1a_base_frame(frame, enforce_expected_n=False)


def test_expected_n_guard_logic() -> None:
    stage2a.validate_expected_n(pd.DataFrame(index=range(1844)))
    with pytest.raises(ValueError, match="N guard failed"):
        stage2a.validate_expected_n(pd.DataFrame(index=range(1843)))


def test_random_test_input_is_rejected_before_read(tmp_path: Path) -> None:
    path = tmp_path / "random_test_predictions.csv"
    with pytest.raises(PermissionError, match="RANDOM_TEST"):
        stage2a.validate_authorized_input_path(path, "stage1a_decision")
    assert not path.exists()


@pytest.mark.parametrize("sealed_date", sorted(stage2a.SEALED_FINAL_DATES))
def test_sealed_dates_are_rejected(sealed_date: str) -> None:
    frame = synthetic_base(date=sealed_date)
    with pytest.raises(PermissionError, match="Sealed final date"):
        stage2a.validate_stage1a_base_frame(frame, enforce_expected_n=False)


def test_unauthorized_and_cp_calibration_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="Unauthorized Stage 2A input"):
        stage2a.validate_authorized_input_path(
            tmp_path / "legacy_mc.csv", "stage1a_decision"
        )
    with pytest.raises(PermissionError, match="CP_CALIBRATION"):
        stage2a.validate_authorized_input_path(
            tmp_path / "cp_calibration_predictions.csv", "stage1a_decision"
        )


def test_stable_sample_id_alignment() -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2a.RAW_MC_WIDTH, [0.1, 0.2, 0.3, 0.4])
    aligned = stage2a.align_interval_to_base(
        base, interval.sample(frac=1.0, random_state=42), "raw_mc"
    )
    assert aligned["sample_id"].tolist() == base["sample_id"].tolist()


def test_risk_table_is_independent_of_csv_row_order() -> None:
    base = synthetic_base()
    intervals = all_interval_tables(base)
    expected = stage2a.build_aligned_risk_table(base, intervals)
    shuffled = {
        key: value.sample(frac=1.0, random_state=index).reset_index(drop=True)
        for index, (key, value) in enumerate(intervals.items())
    }
    observed = stage2a.build_aligned_risk_table(base, shuffled)
    pd.testing.assert_frame_equal(observed, expected)


def test_mismatched_sample_id_is_rejected() -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2a.RAW_MC_WIDTH, 0.2)
    interval.loc[0, "sample_id"] = "different-id"
    with pytest.raises(ValueError, match="sample_id set mismatch"):
        stage2a.align_interval_to_base(base, interval, "raw_mc")


def test_mismatched_true_l_is_rejected() -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2a.RAW_MC_WIDTH, 0.2)
    interval.loc[0, "true_L"] += 0.01
    with pytest.raises(ValueError, match="true_L mismatch"):
        stage2a.align_interval_to_base(base, interval, "raw_mc")


def test_mismatched_mc_mean_is_rejected() -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2a.RAW_MC_WIDTH, 0.2)
    interval.loc[0, "mc_mean"] += 0.01
    with pytest.raises(ValueError, match="mc_mean mismatch"):
        stage2a.align_interval_to_base(base, interval, "raw_mc")


def test_mismatched_date_is_rejected() -> None:
    base = synthetic_base()
    interval = synthetic_interval(base, stage2a.RAW_MC_WIDTH, 0.2)
    interval.loc[0, "date"] = "2017-06-14"
    with pytest.raises(ValueError, match="date mismatch"):
        stage2a.align_interval_to_base(base, interval, "raw_mc")


def test_risk_target_is_abs_error_of_mc_mean_not_point_pred() -> None:
    base = synthetic_base(
        n=2,
        true_l=[0.2, 0.8],
        mc_mean=[0.3, 0.6],
        point_pred=[0.9, 0.9],
        mc_std=[0.01, 0.02],
    )
    table = stage2a.build_aligned_risk_table(base, all_interval_tables(base))
    assert table["abs_error_mc_mean"].tolist() == pytest.approx([0.1, 0.2])
    assert stage2a.RISK_TARGET_ERROR == "abs(true_L - mc_mean)"


def test_mc_std_risk_is_copied_from_stage1a() -> None:
    base = synthetic_base(mc_std=[0.04, 0.03, 0.02, 0.01])
    table = stage2a.build_aligned_risk_table(base, all_interval_tables(base))
    assert table[stage2a.MC_STD].tolist() == pytest.approx([0.04, 0.03, 0.02, 0.01])


@pytest.mark.parametrize(
    "risk_score",
    [
        stage2a.RAW_MC_WIDTH,
        stage2a.SPLIT_CP_WIDTH,
        stage2a.IRRADIANCE_MONDRIAN_WIDTH,
        stage2a.PRED_L_MONDRIAN_WIDTH,
        stage2a.PRED_L_MC_INTERVAL_WIDTH,
        stage2a.PRED_L_STD_MC_WIDTH,
    ],
)
def test_interval_risks_are_reconstructed_as_upper_minus_lower(risk_score: str) -> None:
    base = synthetic_base()
    intervals = all_interval_tables(base)
    intervals[risk_score] = synthetic_interval(
        base, risk_score, [0.11, 0.12, 0.13, 0.14]
    )
    table = stage2a.build_aligned_risk_table(base, intervals)
    assert table[risk_score].tolist() == pytest.approx([0.11, 0.12, 0.13, 0.14])


def test_spearman_perfect_positive_is_one() -> None:
    rho, constant, _ = stage2a.spearman_with_average_ties(
        [1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]
    )
    assert rho == pytest.approx(1.0)
    assert not constant


def test_spearman_perfect_negative_is_minus_one() -> None:
    rho, constant, _ = stage2a.spearman_with_average_ties(
        [1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1]
    )
    assert rho == pytest.approx(-1.0)
    assert not constant


def test_spearman_ties_use_average_ranks() -> None:
    risk = np.array([1, 1, 2, 3], dtype=float)
    error = np.array([1, 2, 3, 4], dtype=float)
    expected = np.corrcoef([1.5, 1.5, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])[0, 1]
    rho, _, _ = stage2a.spearman_with_average_ties(risk, error)
    assert stage2a.average_ranks(risk).tolist() == [1.5, 1.5, 3.0, 4.0]
    assert rho == pytest.approx(expected)


def test_numerical_near_ties_round_to_one_risk_rank_value() -> None:
    values = np.array(
        [
            0.1771004725719999,
            0.177100472572,
            0.1771004725720001,
        ]
    )
    ranked = stage2a.risk_rank_values(values)
    assert stage2a.RISK_TIE_ROUND_DECIMALS == 12
    assert np.unique(ranked).size == 1
    assert ranked.tolist() == pytest.approx([0.177100472572] * 3)


def test_n_unique_and_constant_flag_use_rounded_risk_values() -> None:
    values = [
        0.1771004725719999,
        0.177100472572,
        0.1771004725720001,
    ]
    summary = stage2a.summarize_risk_score(
        stage2a.SPLIT_CP_WIDTH, values, [0.1, 0.2, 0.3]
    )
    assert summary["n_unique_risk"] == 1
    assert summary["constant_risk_score"] is True
    assert np.isnan(summary["rho_spearman"])
    assert summary["risk_min"] != summary["risk_max"]


def test_qcut_receives_rounded_near_ties(monkeypatch) -> None:
    values = np.array(
        [
            0.1771004725719999,
            0.177100472572,
            0.1771004725720001,
        ]
    )
    original = pd.qcut
    observed: dict[str, np.ndarray] = {}

    def capture(values_arg, *args, **kwargs):
        observed["values"] = np.asarray(values_arg, dtype=float)
        return original(values_arg, *args, **kwargs)

    monkeypatch.setattr(stage2a.pd, "qcut", capture)
    records, summary = stage2a.risk_quantile_diagnostics_for_score(
        stage2a.SPLIT_CP_WIDTH, values, [0.1, 0.2, 0.3]
    )
    assert np.unique(observed["values"]).size == 1
    assert summary["actual_bins"] < 2
    assert len(records) == summary["actual_bins"]


def test_spearman_cannot_exploit_floating_noise_order() -> None:
    risk = [
        0.1771004725719999,
        0.177100472572,
        0.1771004725720001,
    ]
    rho, constant, n_unique = stage2a.spearman_with_average_ties(
        risk, [0.1, 0.2, 0.3]
    )
    assert np.isnan(rho)
    assert constant is True
    assert n_unique == 1


def test_meaningful_risk_difference_survives_12_decimal_rounding() -> None:
    risk = [0.177100472572, 0.177100472600]
    ranked = stage2a.risk_rank_values(risk)
    assert np.unique(ranked).size == 2
    rho, constant, n_unique = stage2a.spearman_with_average_ties(risk, [0.1, 0.2])
    assert rho == pytest.approx(1.0)
    assert constant is False
    assert n_unique == 2


def test_continuous_risk_ranking_remains_unchanged() -> None:
    risk = np.linspace(0.001, 0.100, 100)
    error = np.square(risk)
    rho, constant, n_unique = stage2a.spearman_with_average_ties(risk, error)
    assert rho == pytest.approx(1.0)
    assert constant is False
    assert n_unique == 100


def test_constant_risk_returns_nan_and_flag() -> None:
    summary = stage2a.summarize_risk_score(
        stage2a.SPLIT_CP_WIDTH, [0.2, 0.2, 0.2], [0.1, 0.2, 0.3]
    )
    assert np.isnan(summary["rho_spearman"])
    assert summary["constant_risk_score"] is True
    assert summary["n_unique_risk"] == 1


def test_n_unique_risk_is_correct() -> None:
    summary = stage2a.summarize_risk_score(
        stage2a.MC_STD, [0.1, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]
    )
    assert summary["n_unique_risk"] == 3


def test_qcut_uses_at_most_ten_bins() -> None:
    records, summary = stage2a.risk_quantile_diagnostics_for_score(
        stage2a.MC_STD, np.arange(100), np.linspace(0.0, 1.0, 100)
    )
    assert summary["requested_bins"] == 10
    assert summary["actual_bins"] == 10
    assert len(records) == 10


def test_duplicate_risks_are_not_split_into_fake_deciles() -> None:
    risk = np.array([0.0] * 50 + [1.0] * 50)
    records, summary = stage2a.risk_quantile_diagnostics_for_score(
        stage2a.SPLIT_CP_WIDTH, risk, np.linspace(0.0, 1.0, 100)
    )
    assert summary["actual_bins"] < 10
    assert sum(record["N"] for record in records) == 100


def test_quantile_diagnostic_error_statistics_and_rmse() -> None:
    risk = np.arange(10, dtype=float)
    errors = np.arange(1, 11, dtype=float)
    records, _ = stage2a.risk_quantile_diagnostics_for_score(
        stage2a.MC_STD, risk, errors
    )
    first = records[0]
    assert first["N"] == 1
    assert first["abs_error_mean"] == pytest.approx(1.0)
    assert first["abs_error_median"] == pytest.approx(1.0)
    assert first["abs_error_p90"] == pytest.approx(1.0)
    assert first["RMSE_mc_mean"] == pytest.approx(1.0)


def test_extreme_quantile_error_ratio_is_descriptive() -> None:
    _, summary = stage2a.risk_quantile_diagnostics_for_score(
        stage2a.MC_STD, np.arange(10), np.arange(1, 11)
    )
    assert summary["descriptive_extreme_quantile_error_ratio"] == pytest.approx(10.0)
    assert summary["is_formal_operating_point"] is False
    assert summary["is_formal_reject_threshold"] is False
    assert summary["is_decision_rule"] is False


def test_extreme_ratio_is_nan_when_fewer_than_two_bins() -> None:
    records, summary = stage2a.risk_quantile_diagnostics_for_score(
        stage2a.SPLIT_CP_WIDTH, np.ones(20), np.linspace(0.1, 0.2, 20)
    )
    assert summary["actual_bins"] < 2
    assert np.isnan(summary["descriptive_extreme_quantile_error_ratio"])
    assert len(records) == summary["actual_bins"]


def test_output_collision_protection(tmp_path: Path) -> None:
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage2a.ensure_output_available(output_dir)


def test_formal_output_path_is_fixed(tmp_path: Path) -> None:
    assert stage2a.validate_formal_output_path(stage2a.OUTPUT_DIR) == stage2a.OUTPUT_DIR.resolve()
    with pytest.raises(PermissionError, match="Unauthorized Stage 2A output"):
        stage2a.validate_formal_output_path(tmp_path / "other")


def test_fixed_risk_score_order_is_preserved_without_result_sorting() -> None:
    n = 4
    table = pd.DataFrame(
        {
            "abs_error_mc_mean": [0.1, 0.2, 0.3, 0.4],
            **{
                risk_score: np.arange(n, dtype=float) + index
                for index, risk_score in enumerate(stage2a.RISK_SCORE_ORDER)
            },
        }
    )
    spearman = stage2a.build_spearman_table(table)
    assert tuple(spearman["risk_score"]) == stage2a.RISK_SCORE_ORDER


def test_writer_creates_only_required_stage2a_artifacts(tmp_path: Path) -> None:
    table = pd.DataFrame({"synthetic": [1]})
    output_dir = tmp_path / "stage2a"
    stage2a.write_outputs(output_dir, table, table, table, {}, {})
    assert {path.name for path in output_dir.iterdir()} == {
        "risk_score_spearman.csv",
        "risk_quantile_diagnostics.csv",
        "risk_score_descriptive_summary.csv",
        "config.json",
        "provenance.json",
    }


def test_no_threshold_selection_coverage_ause_or_capture_functions() -> None:
    function_names = {
        node.name.lower()
        for node in ast.walk(ast.parse(inspect.getsource(stage2a)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_terms = (
        "threshold",
        "select",
        "coverage",
        "oracle",
        "ause",
        "capture",
        "accept",
        "reject",
    )
    assert not any(
        term in function_name
        for function_name in function_names
        for term in forbidden_terms
    )


def test_no_image_mc_model_inference_training_or_optimizer_path() -> None:
    tree = ast.parse(inspect.getsource(stage2a))
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


def test_no_cqr_or_cleaning_functions() -> None:
    function_names = {
        node.name.lower()
        for node in ast.walk(ast.parse(inspect.getsource(stage2a)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(
        term in function_name
        for function_name in function_names
        for term in ("cqr", "cleaning")
    )


def test_provenance_records_scope_and_all_non_actions() -> None:
    provenance = stage2a.make_provenance()
    assert provenance["primary_evaluation_role"] == "DECISION_DEVELOPMENT"
    assert provenance["decision_development_n"] == 1844
    assert provenance["risk_target_error"] == "abs(true_L - mc_mean)"
    assert provenance["risk_scores"] == list(stage2a.RISK_SCORE_ORDER)
    assert provenance["risk_tie_round_decimals"] == 12
    assert provenance["risk_tie_policy"] == (
        "round risk scores to 12 decimal places for tie-sensitive ranking and "
        "quantile operations only"
    )
    assert provenance["raw_risk_values_modified"] is False
    false_fields = [
        "formal_risk_score_selected",
        "formal_risk_threshold_frozen",
        "risk_coverage_performed",
        "oracle_risk_coverage_performed",
        "ause_performed",
        "high_error_capture_performed",
        "legacy_20pct_reject_performed",
        "cp_calibration_truth_used_for_risk_evaluation",
        "random_test_accessed",
        "random_test_truth_accessed",
        "random_test_predictions_generated",
        "sealed_final_dates_accessed",
        "image_inference_performed",
        "mc_dropout_performed",
        "training_performed",
        "optimizer_created",
        "model_parameters_updated",
        "cqr_performed",
        "cleaning_decision_performed",
    ]
    assert all(provenance[field] is False for field in false_fields)
