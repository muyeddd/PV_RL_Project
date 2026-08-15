"""Synthetic-only tests for Paper1 Stage6 final evaluator."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import run_paper1_final_stage6_v1 as stage6


def _manifest(target: str = stage6.TARGET_RANDOM_TEST, n: int = 12) -> pd.DataFrame:
    if target == stage6.TARGET_SEALED_DATES:
        dates = np.resize(np.array(stage6.SEALED_DATES), n)
        role = stage6.SEALED_ROLE
    else:
        dates = np.array(["2017-06-13"] * n)
        role = stage6.RANDOM_ROLE
    return pd.DataFrame({
        "sample_id": [f"synthetic-{target}-{i}" for i in range(n)],
        "date": dates,
        "timestamp": [f"2017-01-01T00:{i:02d}:00" for i in range(n)],
        "image_path": [f"synthetic/{target}/image_{i}.jpg" for i in range(n)],
        "role": [role] * n,
    }).loc[:, stage6.MANIFEST_COLUMNS]


def _records(target: str = stage6.TARGET_RANDOM_TEST, n: int = 12) -> pd.DataFrame:
    frame = _manifest(target, n)
    frame["true_L"] = np.linspace(0.02, 0.32, n)
    frame["irradiance"] = np.linspace(0.0, 1.0, n)
    frame["irradiance_normalized"] = frame["irradiance"]
    return frame


def _predictions(target: str = stage6.TARGET_RANDOM_TEST, n: int = 12) -> pd.DataFrame:
    records = _records(target, n)
    q50 = np.linspace(0.03, 0.31, n)
    quantiles = np.column_stack((np.clip(q50 - 0.04, 0, 1), q50, np.clip(q50 + 0.04, 0, 1)))
    point = np.clip(records["true_L"].to_numpy() + np.resize(np.array([-.02, .01, .04]), n), 0, 1)
    return stage6.build_predictions(records, point, quantiles, target, enforce_expected_n=False)


def test_exact_two_target_names_and_order() -> None:
    assert stage6.TARGET_ORDER == ("random_test", "sealed_dates")


@pytest.mark.parametrize("target", ["custom", "RANDOM_TEST", "sealed", "", None])
def test_arbitrary_target_rejected(target) -> None:
    with pytest.raises(PermissionError):
        stage6.validate_target(target)


def test_random_n_guard_2582() -> None:
    assert stage6.EXPECTED_N[stage6.TARGET_RANDOM_TEST] == 2582
    with pytest.raises(ValueError, match="2582"):
        stage6.validate_manifest(_manifest(), stage6.TARGET_RANDOM_TEST)


def test_sealed_pooled_n_guard_8855() -> None:
    assert stage6.EXPECTED_N[stage6.TARGET_SEALED_DATES] == 8855
    with pytest.raises(ValueError, match="8855"):
        stage6.validate_manifest(_manifest(stage6.TARGET_SEALED_DATES), stage6.TARGET_SEALED_DATES)


def test_exact_sealed_date_set() -> None:
    frame = _manifest(stage6.TARGET_SEALED_DATES)
    stage6.validate_manifest(frame, stage6.TARGET_SEALED_DATES, enforce_expected_n=False)
    frame["date"] = "2017-06-15"
    with pytest.raises(PermissionError, match="Exact"):
        stage6.validate_manifest(frame, stage6.TARGET_SEALED_DATES, enforce_expected_n=False)


@pytest.mark.parametrize("date", stage6.SEALED_DATES)
def test_random_mode_rejects_each_sealed_date(date: str) -> None:
    frame = _manifest(); frame.loc[0, "date"] = date
    with pytest.raises(PermissionError):
        stage6.validate_manifest(frame, stage6.TARGET_RANDOM_TEST, enforce_expected_n=False)


def test_sealed_mode_rejects_random_role() -> None:
    frame = _manifest(stage6.TARGET_SEALED_DATES); frame["role"] = stage6.RANDOM_ROLE
    with pytest.raises(PermissionError):
        stage6.validate_manifest(frame, stage6.TARGET_SEALED_DATES, enforce_expected_n=False)


def test_sealed_basename_uses_frozen_image_root_without_changing_row_identity(monkeypatch) -> None:
    source = pd.DataFrame({
        "filename": [
            "solar_Thu_Jun_15_10__0__0_2017_L_0.1_I_0.2.jpg",
            "solar_Sat_Jun_24_10__0__1_2017_L_0.2_I_0.3.jpg",
            "solar_Fri_Jun_30_10__0__2_2017_L_0.3_I_0.4.jpg",
        ],
        "timestamp": ["2017-06-15T10:00:00", "2017-06-24T10:00:01", "2017-06-30T10:00:02"],
        "date": list(stage6.SEALED_DATES),
    })
    monkeypatch.setattr(stage6, "EXPECTED_N", {**stage6.EXPECTED_N, stage6.TARGET_SEALED_DATES: len(source)})
    monkeypatch.setattr(stage6.pd, "read_csv", lambda path, usecols: source.loc[:, usecols].copy())

    result = stage6.load_manifest(stage6.TARGET_SEALED_DATES)

    expected_paths = [
        (stage6.SEALED_SOURCE_IMAGE_DIR.relative_to(stage6.PROJECT_ROOT) / name).as_posix()
        for name in source["filename"]
    ]
    assert result["image_path"].tolist() == expected_paths
    assert result["image_path"].tolist() != source["filename"].tolist()
    assert result["sample_id"].tolist() == [
        stage6.split_builder.sample_id_from_timestamp(value) for value in source["timestamp"]
    ]
    assert result["date"].tolist() == source["date"].tolist()
    assert result["timestamp"].tolist() == source["timestamp"].tolist()
    assert result["role"].tolist() == [stage6.SEALED_ROLE] * len(source)
    assert len(result) == len(source)


def test_random_test_manifest_image_paths_are_loaded_unchanged(monkeypatch) -> None:
    expected = _manifest(stage6.TARGET_RANDOM_TEST, n=4)
    monkeypatch.setattr(stage6, "EXPECTED_N", {**stage6.EXPECTED_N, stage6.TARGET_RANDOM_TEST: len(expected)})

    def fake_read_csv(path, usecols):
        assert path == stage6.RANDOM_MANIFEST
        assert usecols == list(stage6.MANIFEST_COLUMNS)
        return expected.loc[:, usecols].copy()

    monkeypatch.setattr(stage6.pd, "read_csv", fake_read_csv)
    result = stage6.load_manifest(stage6.TARGET_RANDOM_TEST)
    pd.testing.assert_frame_equal(result, expected)


def test_sealed_image_existence_guard_accepts_synthetic_files(tmp_path: Path) -> None:
    records = _manifest(stage6.TARGET_SEALED_DATES, n=3)
    image_dir = tmp_path / "data" / "raw" / "PanelImages"
    image_dir.mkdir(parents=True)
    paths = []
    for index in range(len(records)):
        path = image_dir / f"synthetic_{index}.jpg"
        path.write_bytes(b"synthetic-not-an-opened-image")
        paths.append(path.relative_to(tmp_path).as_posix())
    records["image_path"] = paths

    stage6.guard_sealed_image_paths(records, project_root=tmp_path, enforce_expected_n=False)


def test_sealed_image_existence_guard_hard_fails_with_count_and_first_path(tmp_path: Path) -> None:
    records = _manifest(stage6.TARGET_SEALED_DATES, n=3)
    existing = tmp_path / "existing.jpg"
    existing.write_bytes(b"synthetic-not-an-opened-image")
    records["image_path"] = ["existing.jpg", "missing_first.jpg", "missing_second.jpg"]

    with pytest.raises(FileNotFoundError) as exc_info:
        stage6.guard_sealed_image_paths(records, project_root=tmp_path, enforce_expected_n=False)
    message = str(exc_info.value)
    assert "missing count=2" in message
    assert str((tmp_path / "missing_first.jpg").resolve()) in message


def test_sealed_path_guard_is_before_dataset_dataloader_and_inference() -> None:
    source = inspect.getsource(getattr(stage6, "run"))
    guard = source.index("guard_sealed_image_paths(records)")
    dataset = source.index("Stage3A1InferenceDataset")
    loader = source.index("DataLoader(")
    inference = source.index("predict_deterministic")
    assert guard < dataset < loader < inference


def _random_provenance() -> dict[str, object]:
    return {"protocol": stage6.PROTOCOL, "stage": stage6.STAGE, "target": stage6.TARGET_RANDOM_TEST, "formal_final_evaluation": True, "random_test_accessed": True, "sealed_final_dates_accessed": False, "point_checkpoint_sha256_verified": True, "cqr_checkpoint_sha256_verified": True}


def test_sealed_requires_valid_random_completion_provenance() -> None:
    stage6.validate_random_completion_provenance(_random_provenance())
    bad = _random_provenance(); bad["random_test_accessed"] = False
    with pytest.raises(PermissionError): stage6.validate_random_completion_provenance(bad)


def test_sealed_requires_random_marker_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stage6, "OUTPUT_ROOT", tmp_path / "final_stage6_v1")
    with pytest.raises(FileNotFoundError): stage6.require_random_completion()
    marker = stage6.OUTPUT_ROOT / stage6.TARGET_RANDOM_TEST / "provenance.json"; marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(_random_provenance()), encoding="utf-8")
    stage6.require_random_completion()


def test_checkpoint_paths_and_hash_constants_exact() -> None:
    assert stage6.POINT_CHECKPOINT == stage6.stage1a.CLEAN_CHECKPOINT
    assert stage6.POINT_CHECKPOINT_SHA256 == "97f3ec016cf99f83a78e28e2b4aca24787203f105243447d908da739c295de23"
    assert stage6.CQR_CHECKPOINT == stage6.stage3a1.SOURCE_CQR_CHECKPOINT
    assert stage6.CQR_CHECKPOINT_SHA256 == "fd5deea62c867fcffe3791f768752da9dc3a39a1c146244b1e225d6b40b0da80"


def test_checkpoint_sha_mismatch_rejected_with_synthetic_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"; checkpoint.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        stage6.verify_checkpoint_file(checkpoint, "0" * 64, checkpoint)


def test_checkpoint_exact_path_guard(tmp_path: Path) -> None:
    left = tmp_path / "left"; right = tmp_path / "right"; left.write_bytes(b"x"); right.write_bytes(b"x")
    with pytest.raises(PermissionError): stage6.verify_checkpoint_file(left, stage6.sha256_file(left), right)


@pytest.mark.parametrize("failure_at", ["point", "cqr"])
def test_checkpoint_preflight_failure_precedes_final_manifest_access(monkeypatch, tmp_path: Path, failure_at: str) -> None:
    manifest_accessed = False
    calls = 0

    monkeypatch.setattr(stage6, "OUTPUT_ROOT", tmp_path / "final_stage6_v1")
    monkeypatch.setattr(stage6, "load_frozen_qhat", lambda: stage6.QHAT)

    def fail_verification(path, expected_sha256, authorized_path):
        nonlocal calls
        calls += 1
        if failure_at == "point" or calls == 2:
            raise ValueError(f"synthetic {failure_at} checkpoint failure")
        return expected_sha256

    def final_manifest_sentinel(target):
        nonlocal manifest_accessed
        manifest_accessed = True
        raise AssertionError("final manifest must not be accessed")

    monkeypatch.setattr(stage6, "verify_checkpoint_file", fail_verification)
    monkeypatch.setattr(stage6, "load_manifest", final_manifest_sentinel)
    with pytest.raises(ValueError, match="checkpoint failure"):
        stage6.prepare_frozen_runtime(stage6.TARGET_RANDOM_TEST)
    assert manifest_accessed is False
    assert calls == (1 if failure_at == "point" else 2)


def test_alpha_and_qhat_exact() -> None:
    assert stage6.ALPHA == 0.10
    assert stage6.QHAT == 0.004862844288256299
    assert stage6.validate_qhat_artifact({"alpha": .10, "qhat": stage6.QHAT}) == stage6.QHAT


@pytest.mark.parametrize("field,value", [("alpha", .11), ("qhat", 0.0)])
def test_qhat_artifact_drift_rejected(field: str, value: float) -> None:
    payload = {"alpha": .10, "qhat": stage6.QHAT}; payload[field] = value
    with pytest.raises(ValueError): stage6.validate_qhat_artifact(payload)


def test_qhat_is_not_recomputed_from_final() -> None:
    source = inspect.getsource(stage6)
    assert "conformal_quantile(" not in source
    assert "calibrate_cqr(" not in source
    assert stage6.make_provenance(stage6.TARGET_RANDOM_TEST, point_verified=True, cqr_verified=True, random_completed=False)["qhat_recomputed_from_final"] is False


def test_cqr_interval_formula_and_clipping() -> None:
    q05 = np.array([0.0, .2]); q95 = np.array([.8, 1.0])
    lower, upper, lc, uc = stage6.conformalize(q05, q95)
    np.testing.assert_allclose(lower, [0.0, .2-stage6.QHAT])
    np.testing.assert_allclose(upper, [.8+stage6.QHAT, 1.0])
    assert lc.tolist() == [True, False]; assert uc.tolist() == [False, True]


def test_nonfrozen_qhat_rejected() -> None:
    with pytest.raises(ValueError): stage6.conformalize([.1], [.2], qhat=0.0)


def test_picp_is_inclusive() -> None:
    frame = _predictions(); frame.loc[0, "true_L"] = frame.loc[0, "lower"]; frame.loc[1, "true_L"] = frame.loc[1, "upper"]
    metrics = stage6.interval_metrics(frame)
    expected = ((frame["true_L"] >= frame["lower"]) & (frame["true_L"] <= frame["upper"])).mean()
    assert metrics["PICP"] == expected


def test_interval_score_formula() -> None:
    y = np.array([.1, .5, .9]); lo = np.array([.2, .4, .6]); hi = np.array([.4, .6, .8])
    expected = (hi-lo) + 20*(lo-y)*(y<lo) + 20*(y-hi)*(y>hi)
    np.testing.assert_allclose(stage6.stage1b.standard_interval_score(y, lo, hi), expected)


def test_fixed_conditional_bin_edges_and_labels() -> None:
    assert stage6.BIN_EDGES == (0., .2, .4, .6, .8, 1.)
    assert stage6.BIN_LABELS == stage6.stage1b.FIXED_BIN_LABELS


def test_conditional_cut_boundary_semantics() -> None:
    frame = _predictions(n=5); frame["q50"] = [0., .2, .2001, .8, 1.]
    table = stage6.conditional_coverage(frame, stage6.TARGET_RANDOM_TEST, "q50")
    assert table["N"].tolist() == [2, 1, 0, 1, 1]


def test_point_q50_metric_formulas() -> None:
    y = np.array([0., 1., 2.]); p = np.array([.1, .8, 2.2]); result = stage6.prediction_metrics(y, p); residual = p-y
    assert result["R2"] == pytest.approx(1 - np.square(residual).sum()/2)
    assert result["RMSE"] == pytest.approx(np.sqrt(np.square(residual).mean()))
    assert result["MAE"] == pytest.approx(np.abs(residual).mean()); assert result["bias"] == pytest.approx(residual.mean())


def test_tau_and_method_protocol_exact() -> None:
    assert stage6.TAU_GRID == (.05, .10, .15, .20); assert stage6.REFERENCE_TAU == .15
    assert stage6.METHOD_ORDER == ("point_threshold", "cqr_q50_threshold", "cqr_interval_tristate")


@pytest.mark.parametrize("value,expected", [(.149, stage6.stage4a.WAIT), (.15, stage6.stage4a.WAIT), (.151, stage6.stage4a.CLEAN)])
def test_oracle_point_q50_equality_rules(value: float, expected: str) -> None:
    assert stage6.stage4a.oracle_actions([value], .15)[0] == expected
    assert stage6.stage4a.point_threshold_actions([value], .15)[0] == expected
    assert stage6.stage4a.cqr_q50_threshold_actions([value], .15)[0] == expected


@pytest.mark.parametrize("lower,upper,expected", [(.151,.2,"CLEAN"),(.1,.149,"WAIT"),(.1,.2,"REVIEW"),(.15,.2,"REVIEW"),(.1,.15,"REVIEW"),(.15,.15,"REVIEW")])
def test_tristate_review_boundaries(lower: float, upper: float, expected: str) -> None:
    assert stage6.stage4a.cqr_interval_tristate_actions([lower], [upper], .15)[0] == expected


def test_stage4_metrics_and_adc_identity() -> None:
    predictions = _predictions(); actions = stage6.build_decision_actions(predictions); metrics = stage6.build_decision_metrics(actions, stage6.TARGET_RANDOM_TEST)
    assert np.allclose(metrics["auto_decision_coverage"], 1-metrics["review_rate"])
    binary = metrics["method"].isin((stage6.stage4a.POINT_THRESHOLD, stage6.stage4a.CQR_Q50_THRESHOLD)); assert (metrics.loc[binary,"review_n"] == 0).all()


def test_stage5_normalized_economic_formulas() -> None:
    false = stage6.stage5a.sample_regret_components(.08,.15,"CLEAN"); missed = stage6.stage5a.sample_regret_components(.23,.15,"WAIT"); review = stage6.stage5a.sample_regret_components(.1,.15,"REVIEW")
    assert false["regret_r0"] == pytest.approx(.07); assert missed["regret_r0"] == pytest.approx(.08); assert review["regret_r0"] == 0
    for value in (false, missed, review): assert value["base_action_cost_r0"] == pytest.approx(value["oracle_cost"] + value["regret_r0"])


def test_final_economic_metrics_cost_identity() -> None:
    actions = stage6.build_decision_actions(_predictions()); sample = stage6.build_economic_sample(actions); metrics = stage6.build_economic_metrics(sample, stage6.TARGET_RANDOM_TEST)
    assert np.allclose(metrics["mean_total_cost_r0"], metrics["oracle_mean_cost"] + metrics["mean_regret_r0"])


def test_analytic_break_even_negative_zero_and_substitution() -> None:
    ratio = stage6.stage5a.analytic_break_even_ratio(.01,.02,.25,.1); assert ratio == pytest.approx(-.4)
    assert stage6.stage5a.cqr_mean_regret_at_review_ratio(.02,.25,ratio,.1) == pytest.approx(.01)
    assert math.isnan(stage6.stage5a.analytic_break_even_ratio(.1,.02,0,.1))


def test_sealed_tables_include_pooled_and_each_date() -> None:
    predictions = _predictions(stage6.TARGET_SEALED_DATES, n=12)
    pm = stage6.build_prediction_metric_table(predictions, stage6.TARGET_SEALED_DATES); im = stage6.build_interval_metric_table(predictions, stage6.TARGET_SEALED_DATES)
    assert set(pm["scope"]) == {"pooled","per_date"}; assert set(pm.loc[pm["scope"]=="per_date","date"]) == set(stage6.SEALED_DATES)
    assert set(im.loc[im["scope"]=="per_date","date"]) == set(stage6.SEALED_DATES)


def test_duplicate_sample_and_image_rejected() -> None:
    for field in ("sample_id","image_path"):
        frame = _manifest(); frame.loc[1,field] = frame.loc[0,field]
        with pytest.raises(ValueError): stage6.validate_manifest(frame, stage6.TARGET_RANDOM_TEST, enforce_expected_n=False)


@pytest.mark.parametrize("field", ["true_L","irradiance"])
def test_nonfinite_final_values_rejected(field: str) -> None:
    records = _records(); records.loc[0,field] = np.nan; q = np.column_stack(([.1]*12,[.2]*12,[.3]*12))
    with pytest.raises(ValueError): stage6.build_predictions(records, np.full(12,.2), q, stage6.TARGET_RANDOM_TEST, enforce_expected_n=False)


def test_output_collision_rejected(tmp_path: Path) -> None:
    output = tmp_path/"target"; output.mkdir(); (output/"x").write_text("x")
    with pytest.raises(FileExistsError): stage6.ensure_output_available(output)


def test_forbidden_winner_fields_rejected() -> None:
    frame = pd.DataFrame({"winner":[True]})
    with pytest.raises(ValueError): stage6.ensure_no_forbidden_fields(frame)


def test_provenance_final_flags() -> None:
    random = stage6.make_provenance(stage6.TARGET_RANDOM_TEST, point_verified=True, cqr_verified=True, random_completed=False)
    assert random["formal_final_evaluation"] is True and random["random_test_accessed"] is True and random["sealed_final_dates_accessed"] is False
    assert random["training_performed"] is False and random["mc_dropout_performed"] is False and random["qhat_recomputed_from_final"] is False
    assert "sealed_first_attempt_failed" not in random

    sealed = stage6.make_provenance(stage6.TARGET_SEALED_DATES, point_verified=True, cqr_verified=True, random_completed=True)
    assert sealed["sealed_first_attempt_failed"] is True
    assert sealed["sealed_first_attempt_failure_stage"] == "image_path_resolution_before_successful_image_load"
    assert sealed["sealed_first_attempt_performance_results_generated"] is False
    assert sealed["sealed_retry_reason"] == "fix_relative_image_path_resolution"
    assert sealed["sealed_retry_changes_scientific_method"] is False


def test_tests_do_not_read_final_formal_data_or_call_run() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "pd."+"read_csv(" not in source
    assert "stage6."+"run" not in source
    assert "= "+"stage6."+"run" not in source
    assert "random_"+"test.csv" not in source
    assert "split_"+"manifest.csv" not in source
