"""Paper1 CQR Stage 3B: frozen Stage 2 risk-utility evaluation.

This table-only stage appends one score, ``cqr_conformal_width``, to the
already-frozen Stage 2A/2B/2C risk task.  The common target remains
``abs(true_L - mc_mean)``.  Existing Stage 2 baseline results are read and
appended to, never recomputed.  Importing this module reads no formal data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments import run_paper1_cqr_stage3a2_intervals_v1 as stage3a2
from experiments import run_paper1_risk_stage2a_score_validity_v1 as stage2a
from experiments import run_paper1_risk_stage2b_risk_coverage_v1 as stage2b
from experiments import run_paper1_risk_stage2c_high_error_capture_v1 as stage2c


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"
STAGE = "cqr_stage3b_risk_v1"
EVALUATION_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_N = 1844
COMMON_RISK_TARGET = "abs(true_L - mc_mean)"
CQR_RISK_SCORE = "cqr_conformal_width"
CQR_RISK_SCORE_DEFINITION = (
    "final conformalized clipped interval width (upper-lower)"
)
RISK_TIE_ROUND_DECIMALS = 12
SOURCE_CQR_CHECKPOINT_SHA256 = (
    "fd5deea62c867fcffe3791f768752da9dc3a39a1c146244b1e225d6b40b0da80"
)
WIDTH_ABS_TOLERANCE = 1e-12
ALIGNMENT_ABS_TOLERANCE = 1e-15

STAGE3A2_DIR = stage3a2.OUTPUT_DIR
CQR_PREDICTIONS_INPUT = STAGE3A2_DIR / "cqr_predictions.csv"
STAGE3A2_PROVENANCE_INPUT = STAGE3A2_DIR / "provenance.json"
STAGE2A_SPEARMAN_INPUT = stage2a.OUTPUT_DIR / "risk_score_spearman.csv"
STAGE2B_CURVES_INPUT = stage2b.OUTPUT_DIR / "risk_coverage_curves.csv"
STAGE2B_AUSE_INPUT = stage2b.OUTPUT_DIR / "ause_summary.csv"
STAGE2C_CAPTURE_INPUT = stage2c.OUTPUT_DIR / "high_error_capture_summary.csv"
STAGE2C_TARGET_AUDIT_INPUT = stage2c.OUTPUT_DIR / "high_error_target_audit.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE

AUTHORIZED_INPUTS = {
    "cqr_predictions": CQR_PREDICTIONS_INPUT,
    "stage3a2_provenance": STAGE3A2_PROVENANCE_INPUT,
    "stage2a_spearman": STAGE2A_SPEARMAN_INPUT,
    "stage2b_curves": STAGE2B_CURVES_INPUT,
    "stage2b_ause": STAGE2B_AUSE_INPUT,
    "stage2c_capture": STAGE2C_CAPTURE_INPUT,
    "stage2c_target_audit": STAGE2C_TARGET_AUDIT_INPUT,
}

METHOD_ORDER = stage2a.RISK_SCORE_ORDER + (CQR_RISK_SCORE,)
CURVE_COMPARISON_ORDER = METHOD_ORDER + (stage2b.ORACLE,)
SPEARMAN_COLUMNS = (
    "risk_score",
    "evaluation_role",
    "N",
    "rho_spearman",
    "constant_risk_score",
    "n_unique_risk",
    "risk_min",
    "risk_mean",
    "risk_median",
    "risk_p95",
    "risk_max",
    "abs_error_mean",
    "abs_error_median",
    "abs_error_p95",
)
EVALUATION_COLUMNS = (
    "sample_id",
    "date",
    "timestamp",
    "image_path",
    "role",
    "true_L",
    "mc_mean",
    "q50",
    "abs_error_mc_mean",
    "sq_error_mc_mean",
    CQR_RISK_SCORE,
)
FORBIDDEN_PRESENTATION_FIELDS = frozenset(
    {"winner", "best", "rank", "ranking", "recommended", "selected_method"}
)

FROZEN_HIGH_ERROR_TARGET_N = 184
FROZEN_HIGH_ERROR_THRESHOLD = 0.0853964442488999
FROZEN_HIGH_ERROR_N_STRICT = 183
FROZEN_HIGH_ERROR_BOUNDARY_GROUP_SIZE = 1
FROZEN_HIGH_ERROR_BOUNDARY_FRACTION = 1.0
FROZEN_RISK_BUDGET_COUNTS = (184, 368, 553)

REQUIRED_STAGE3A2_PROVENANCE = {
    "protocol": PROTOCOL,
    "stage": stage3a2.STAGE,
    "alpha": 0.10,
    "calibration_role": "CP_CALIBRATION",
    "evaluation_role": EVALUATION_ROLE,
    "source_cqr_checkpoint_sha256": SOURCE_CQR_CHECKPOINT_SHA256,
    "qhat_selected_using_decision_truth": False,
    "decision_truth_used_only_for_evaluation": True,
    "random_test_accessed": False,
    "sealed_final_dates_accessed": False,
    "training_performed": False,
    "image_inference_performed": False,
    "risk_evaluation_performed": False,
    "cleaning_decision_performed": False,
    "economic_decision_performed": False,
}


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_protocol_constants() -> None:
    if not (PROTOCOL == stage2a.PROTOCOL == stage2b.PROTOCOL == stage2c.PROTOCOL):
        raise ValueError("Stage 2/3 protocol constants disagree")
    if not (
        EVALUATION_ROLE
        == stage2a.PRIMARY_ROLE
        == stage2b.PRIMARY_ROLE
        == stage2c.PRIMARY_ROLE
    ):
        raise ValueError("Stage 2/3 evaluation role constants disagree")
    if not (EXPECTED_N == stage2a.EXPECTED_N == stage2b.EXPECTED_N == stage2c.EXPECTED_N):
        raise ValueError("Stage 2/3 N constants disagree")
    if COMMON_RISK_TARGET != stage2a.RISK_TARGET_ERROR:
        raise ValueError("Stage 3B common risk target must remain Stage 2A's target")
    if RISK_TIE_ROUND_DECIMALS != stage2a.RISK_TIE_ROUND_DECIMALS:
        raise ValueError("Stage 3B risk tie precision drifted")
    if tuple(stage2b.COVERAGE_GRID) != (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ):
        raise ValueError("Stage 2B coverage grid drifted")
    if tuple(stage2c.RISK_BUDGET_FRACTIONS) != (0.10, 0.20, 0.30):
        raise ValueError("Stage 2C risk budgets drifted")


def validate_protocol(protocol: str) -> None:
    validate_protocol_constants()
    if protocol != PROTOCOL:
        raise PermissionError(f"Unauthorized protocol: {protocol!r}")


def validate_authorized_input_path(path: Path, source_key: str) -> Path:
    candidate = _resolved(path)
    lowered = str(candidate).lower()
    if "random_test" in lowered:
        raise PermissionError("RANDOM_TEST input is forbidden")
    if "cp_calibration" in lowered:
        raise PermissionError("CP_CALIBRATION input is forbidden for Stage 3B")
    if source_key not in AUTHORIZED_INPUTS:
        raise PermissionError(f"Unauthorized Stage 3B source key: {source_key}")
    authorized = _resolved(AUTHORIZED_INPUTS[source_key])
    if candidate != authorized:
        raise PermissionError(f"Unauthorized Stage 3B input path: {candidate}")
    return candidate


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return value


def validate_stage3a2_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(REQUIRED_STAGE3A2_PROVENANCE) - set(provenance)
    if missing:
        raise ValueError(f"Stage 3A2 provenance missing fields: {sorted(missing)}")
    mismatches = {
        key: (provenance[key], expected)
        for key, expected in REQUIRED_STAGE3A2_PROVENANCE.items()
        if provenance[key] != expected
    }
    if mismatches:
        raise ValueError(f"Stage 3A2 provenance guard failed: {mismatches}")
    sha = str(provenance["source_cqr_checkpoint_sha256"])
    if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError("Stage 3A2 checkpoint SHA256 must be 64 lowercase hex characters")
    return dict(provenance)


def load_stage3a2_provenance(
    path: Path = STAGE3A2_PROVENANCE_INPUT,
) -> dict[str, Any]:
    authorized = validate_authorized_input_path(path, "stage3a2_provenance")
    return validate_stage3a2_provenance(_load_json_mapping(authorized))


def validate_cqr_intervals(
    frame: pd.DataFrame, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    validated = stage3a2.validate_cqr_predictions(
        frame, enforce_expected_n=enforce_expected_n
    )
    if set(validated["method"].astype(str)) != {stage3a2.METHOD}:
        raise ValueError("Stage 3A2 method must be cqr_v1")
    if set(validated["role"].astype(str)) != {EVALUATION_ROLE}:
        raise PermissionError("Only DECISION_DEVELOPMENT is authorized")
    if validated["sample_id"].isna().any() or validated["sample_id"].duplicated().any():
        raise ValueError("CQR sample_id must be non-null and unique")
    if validated["image_path"].isna().any() or validated["image_path"].duplicated().any():
        raise ValueError("CQR image_path must be non-null and unique")
    lower = validated["lower"].to_numpy(dtype=np.float64)
    upper = validated["upper"].to_numpy(dtype=np.float64)
    stored_width = validated["width"].to_numpy(dtype=np.float64)
    final_width = upper - lower
    if not np.allclose(
        stored_width, final_width, rtol=0.0, atol=WIDTH_ABS_TOLERANCE
    ):
        raise ValueError("Stored CQR width is inconsistent with final upper-lower")
    return validated


def load_cqr_intervals(path: Path = CQR_PREDICTIONS_INPUT) -> pd.DataFrame:
    authorized = validate_authorized_input_path(path, "cqr_predictions")
    return validate_cqr_intervals(pd.read_csv(authorized))


def build_cqr_evaluation_table(
    stage1a_base: pd.DataFrame,
    cqr_intervals: pd.DataFrame,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    base = stage2a.validate_stage1a_base_frame(
        stage1a_base, enforce_expected_n=enforce_expected_n
    )
    cqr = validate_cqr_intervals(
        cqr_intervals, enforce_expected_n=enforce_expected_n
    )
    if len(base) != len(cqr):
        raise ValueError(f"Stage 2A/CQR N mismatch: {len(base)} != {len(cqr)}")
    if base["sample_id"].duplicated().any() or cqr["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique before Stage 3B alignment")
    base_ids = base["sample_id"].astype(str).tolist()
    if set(base_ids) != set(cqr["sample_id"].astype(str)):
        missing = set(base_ids) - set(cqr["sample_id"].astype(str))
        extra = set(cqr["sample_id"].astype(str)) - set(base_ids)
        raise ValueError(
            f"Stage 2A/CQR sample_id set mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    aligned = cqr.set_index(cqr["sample_id"].astype(str), drop=False).loc[base_ids]
    aligned = aligned.reset_index(drop=True)
    base = base.reset_index(drop=True)
    for field in ("image_path", "date", "timestamp", "role"):
        if not np.array_equal(
            base[field].astype(str).to_numpy(), aligned[field].astype(str).to_numpy()
        ):
            raise ValueError(f"Stage 2A/CQR {field} alignment mismatch")
    if not np.allclose(
        base["true_L"].to_numpy(dtype=np.float64),
        aligned["true_L"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=ALIGNMENT_ABS_TOLERANCE,
    ):
        raise ValueError("Stage 2A/CQR true_L alignment mismatch")

    true_l = base["true_L"].to_numpy(dtype=np.float64)
    mc_mean = base["mc_mean"].to_numpy(dtype=np.float64)
    signed_error = true_l - mc_mean
    abs_error = np.abs(signed_error)
    sq_error = np.square(signed_error)
    final_width = (
        aligned["upper"].to_numpy(dtype=np.float64)
        - aligned["lower"].to_numpy(dtype=np.float64)
    )
    if not np.allclose(
        aligned["width"].to_numpy(dtype=np.float64),
        final_width,
        rtol=0.0,
        atol=WIDTH_ABS_TOLERANCE,
    ):
        raise ValueError("CQR risk score must equal stored final interval width")
    result = pd.DataFrame(
        {
            "sample_id": base["sample_id"],
            "date": base["date"],
            "timestamp": base["timestamp"],
            "image_path": base["image_path"],
            "role": base["role"],
            "true_L": true_l,
            "mc_mean": mc_mean,
            "q50": aligned["q50"].to_numpy(dtype=np.float64),
            "abs_error_mc_mean": abs_error,
            "sq_error_mc_mean": sq_error,
            CQR_RISK_SCORE: final_width,
        }
    )
    validate_cqr_evaluation_table(result, enforce_expected_n=enforce_expected_n)
    return result.loc[:, EVALUATION_COLUMNS]


def validate_cqr_evaluation_table(
    frame: pd.DataFrame, *, enforce_expected_n: bool = True
) -> None:
    missing = set(EVALUATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"CQR risk evaluation table missing fields: {sorted(missing)}")
    if frame.empty:
        raise ValueError("CQR risk evaluation table is empty")
    if set(frame["role"].astype(str)) != {EVALUATION_ROLE}:
        raise PermissionError("Only DECISION_DEVELOPMENT is authorized")
    if enforce_expected_n and len(frame) != EXPECTED_N:
        raise ValueError(
            f"DECISION_DEVELOPMENT N guard failed: expected {EXPECTED_N}, got {len(frame)}"
        )
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-null and unique")
    normalized_dates = pd.to_datetime(frame["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    sealed = set(normalized_dates) & stage2a.SEALED_FINAL_DATES
    if sealed:
        raise PermissionError(f"Sealed final date rejected: {sorted(sealed)}")
    locators = frame["image_path"].astype(str)
    if locators.str.lower().str.contains("random_test", regex=False).any():
        raise PermissionError("RANDOM_TEST locator rejected")
    numeric_columns = (
        "true_L",
        "mc_mean",
        "q50",
        "abs_error_mc_mean",
        "sq_error_mc_mean",
        CQR_RISK_SCORE,
    )
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("CQR risk evaluation values must be finite")
    if (numeric[CQR_RISK_SCORE] < 0.0).any():
        raise ValueError("CQR conformal width must be non-negative")
    signed = numeric["true_L"].to_numpy() - numeric["mc_mean"].to_numpy()
    if not np.array_equal(np.abs(signed), numeric["abs_error_mc_mean"].to_numpy()):
        raise ValueError("Common risk target must equal abs(true_L-mc_mean)")
    if not np.array_equal(np.square(signed), numeric["sq_error_mc_mean"].to_numpy()):
        raise ValueError("Squared common error definition mismatch")


def build_cqr_spearman(evaluation: pd.DataFrame) -> pd.DataFrame:
    validate_cqr_evaluation_table(evaluation, enforce_expected_n=False)
    record = stage2a.summarize_risk_score(
        CQR_RISK_SCORE,
        evaluation[CQR_RISK_SCORE].to_numpy(dtype=np.float64),
        evaluation["abs_error_mc_mean"].to_numpy(dtype=np.float64),
    )
    result = pd.DataFrame([record]).loc[:, SPEARMAN_COLUMNS]
    if result.loc[0, "risk_score"] != CQR_RISK_SCORE:
        raise RuntimeError("CQR Spearman risk-score name drifted")
    return result


def build_cqr_risk_coverage_curve(evaluation: pd.DataFrame) -> pd.DataFrame:
    validate_cqr_evaluation_table(evaluation, enforce_expected_n=False)
    stage2b.validate_coverage_grid(stage2b.COVERAGE_GRID)
    risk = evaluation[CQR_RISK_SCORE].to_numpy(dtype=np.float64)
    absolute = evaluation["abs_error_mc_mean"].to_numpy(dtype=np.float64)
    squared = evaluation["sq_error_mc_mean"].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    for coverage in stage2b.COVERAGE_GRID:
        target_n = stage2b.target_retained_count(coverage, len(evaluation))
        metrics = stage2b.tie_aware_retention_metrics(
            risk,
            absolute,
            squared,
            target_n,
            use_stage2a_risk_rounding=True,
        )
        records.append(
            {
                "risk_score": CQR_RISK_SCORE,
                "evaluation_role": EVALUATION_ROLE,
                "coverage_requested": coverage,
                **metrics,
            }
        )
    result = pd.DataFrame.from_records(records).loc[:, stage2b.CURVE_COLUMNS]
    validate_cqr_curve(result, absolute, squared)
    return result


def build_oracle_curve(evaluation: pd.DataFrame) -> pd.DataFrame:
    validate_cqr_evaluation_table(evaluation, enforce_expected_n=False)
    absolute = evaluation["abs_error_mc_mean"].to_numpy(dtype=np.float64)
    squared = evaluation["sq_error_mc_mean"].to_numpy(dtype=np.float64)
    return stage2b.build_curve_for_score(
        stage2b.ORACLE,
        absolute,
        absolute,
        squared,
        oracle=True,
    )


def validate_cqr_curve(
    curve: pd.DataFrame,
    abs_error: Sequence[float] | np.ndarray,
    sq_error: Sequence[float] | np.ndarray,
) -> None:
    if tuple(curve.columns) != stage2b.CURVE_COLUMNS:
        raise ValueError("CQR risk-coverage schema mismatch")
    if tuple(curve["risk_score"].drop_duplicates()) != (CQR_RISK_SCORE,):
        raise ValueError("CQR curve must contain exactly one risk score")
    if tuple(curve["coverage_requested"].astype(float)) != stage2b.COVERAGE_GRID:
        raise ValueError("CQR curve coverage grid mismatch")
    absolute = np.asarray(abs_error, dtype=np.float64)
    squared = np.asarray(sq_error, dtype=np.float64)
    expected_counts = [
        stage2b.target_retained_count(coverage, len(absolute))
        for coverage in stage2b.COVERAGE_GRID
    ]
    if curve["target_retained_n"].astype(int).tolist() != expected_counts:
        raise ValueError("CQR retained-count rule mismatch")
    full = curve.iloc[-1]
    if not math.isclose(
        float(full["MAE_mc_mean"]),
        float(absolute.mean()),
        rel_tol=0.0,
        abs_tol=stage2b.METRIC_QC_TOLERANCE,
    ):
        raise ValueError("CQR coverage=1 MAE mismatch")
    if not math.isclose(
        float(full["RMSE_mc_mean"]),
        float(np.sqrt(squared.mean())),
        rel_tol=0.0,
        abs_tol=stage2b.METRIC_QC_TOLERANCE,
    ):
        raise ValueError("CQR coverage=1 RMSE mismatch")


def build_cqr_ause_summary(
    cqr_curve: pd.DataFrame,
    oracle_curve: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    cqr_rows = cqr_curve.set_index("coverage_requested").loc[
        list(stage2b.COVERAGE_GRID)
    ]
    oracle_rows = oracle_curve.set_index("coverage_requested").loc[
        list(stage2b.COVERAGE_GRID)
    ]
    coverage = np.asarray(stage2b.COVERAGE_GRID, dtype=np.float64)
    ause_mae = stage2b.trapezoidal_integral(
        cqr_rows["MAE_mc_mean"].to_numpy(dtype=np.float64)
        - oracle_rows["MAE_mc_mean"].to_numpy(dtype=np.float64),
        coverage,
    )
    ause_rmse = stage2b.trapezoidal_integral(
        cqr_rows["RMSE_mc_mean"].to_numpy(dtype=np.float64)
        - oracle_rows["RMSE_mc_mean"].to_numpy(dtype=np.float64),
        coverage,
    )
    if ause_mae < -stage2b.AUSE_QC_TOLERANCE or ause_rmse < -stage2b.AUSE_QC_TOLERANCE:
        raise ValueError("CQR AUSE is significantly negative")
    record = {
        "risk_score": CQR_RISK_SCORE,
        "evaluation_role": EVALUATION_ROLE,
        "N": int(n),
        "coverage_min": stage2b.COVERAGE_GRID[0],
        "coverage_max": stage2b.COVERAGE_GRID[-1],
        "n_coverage_points": len(stage2b.COVERAGE_GRID),
        "AUSE_MAE": ause_mae,
        "AUSE_RMSE": ause_rmse,
    }
    return pd.DataFrame([record]).loc[:, stage2b.AUSE_COLUMNS]


def build_cqr_high_error_capture(
    evaluation: pd.DataFrame,
    *,
    enforce_frozen_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_cqr_evaluation_table(
        evaluation, enforce_expected_n=enforce_frozen_target
    )
    stage2c.validate_risk_budget_fractions()
    errors = evaluation["abs_error_mc_mean"].to_numpy(dtype=np.float64)
    target_weights, target_audit = stage2c.build_high_error_target(errors)
    if enforce_frozen_target:
        validate_frozen_high_error_target(target_audit, len(evaluation))
    target_n = int(target_audit["target_high_error_n"])
    risk = evaluation[CQR_RISK_SCORE].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    for budget_fraction in stage2c.RISK_BUDGET_FRACTIONS:
        selection_weights, selection_audit = stage2c.build_risk_selection(
            risk, budget_fraction
        )
        metrics = stage2c.capture_metrics(
            target_weights,
            selection_weights,
            target_n,
            int(selection_audit["risk_budget_n"]),
        )
        records.append(
            {
                "risk_score": CQR_RISK_SCORE,
                "evaluation_role": EVALUATION_ROLE,
                "N": len(evaluation),
                "high_error_target_fraction": stage2c.HIGH_ERROR_TARGET_FRACTION,
                "target_high_error_n": target_n,
                "high_error_threshold_abs_error": target_audit[
                    "high_error_threshold_abs_error"
                ],
                **selection_audit,
                **metrics,
            }
        )
    result = pd.DataFrame.from_records(records).loc[:, stage2c.SUMMARY_COLUMNS]
    validate_cqr_capture_summary(result, len(evaluation), target_n)
    return result, target_audit


def validate_frozen_high_error_target(audit: Mapping[str, Any], n: int) -> None:
    expected = {
        "target_high_error_n": FROZEN_HIGH_ERROR_TARGET_N,
        "n_strict_high_error": FROZEN_HIGH_ERROR_N_STRICT,
        "target_boundary_tie_group_size": FROZEN_HIGH_ERROR_BOUNDARY_GROUP_SIZE,
        "target_boundary_fraction": FROZEN_HIGH_ERROR_BOUNDARY_FRACTION,
    }
    if n != EXPECTED_N:
        raise ValueError(f"Frozen high-error target requires N={EXPECTED_N}")
    mismatches = {
        key: (audit.get(key), value)
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen Stage 2C high-error target mismatch: {mismatches}")
    if not math.isclose(
        float(audit["high_error_threshold_abs_error"]),
        FROZEN_HIGH_ERROR_THRESHOLD,
        rel_tol=0.0,
        abs_tol=ALIGNMENT_ABS_TOLERANCE,
    ):
        raise ValueError("Frozen Stage 2C high-error threshold mismatch")


def validate_cqr_capture_summary(summary: pd.DataFrame, n: int, target_n: int) -> None:
    if tuple(summary.columns) != stage2c.SUMMARY_COLUMNS:
        raise ValueError("CQR high-error capture schema mismatch")
    if tuple(summary["risk_score"].drop_duplicates()) != (CQR_RISK_SCORE,):
        raise ValueError("CQR capture must contain exactly one risk score")
    if tuple(summary["risk_budget_fraction"].astype(float)) != stage2c.RISK_BUDGET_FRACTIONS:
        raise ValueError("CQR risk-budget order mismatch")
    expected_counts = [
        stage2c.risk_budget_count(fraction, n)
        for fraction in stage2c.RISK_BUDGET_FRACTIONS
    ]
    if summary["risk_budget_n"].astype(int).tolist() != expected_counts:
        raise ValueError("CQR risk-budget count rule mismatch")
    captured = summary["expected_captured_high_error"].to_numpy(dtype=np.float64)
    if not np.allclose(
        summary["capture_rate"],
        captured / target_n,
        rtol=0.0,
        atol=stage2c.METRIC_QC_TOLERANCE,
    ):
        raise ValueError("CQR capture-rate definition mismatch")
    expected_random = summary["risk_budget_n"].to_numpy(dtype=np.float64) / n
    if not np.allclose(
        summary["random_capture_rate"],
        expected_random,
        rtol=0.0,
        atol=stage2c.METRIC_QC_TOLERANCE,
    ):
        raise ValueError("CQR random capture-rate mismatch")
    if not np.allclose(
        summary["capture_lift_vs_random"],
        summary["capture_rate"].to_numpy(dtype=np.float64) / expected_random,
        rtol=0.0,
        atol=stage2c.METRIC_QC_TOLERANCE,
    ):
        raise ValueError("CQR capture-lift definition mismatch")


def validate_frozen_target_audit_consistency(
    formal_audit: Mapping[str, Any], recomputed_audit: Mapping[str, Any]
) -> None:
    validate_frozen_high_error_target(formal_audit, EXPECTED_N)
    validate_frozen_high_error_target(recomputed_audit, EXPECTED_N)
    fields = (
        "target_high_error_n",
        "high_error_threshold_abs_error",
        "n_strict_high_error",
        "target_boundary_tie_group_size",
        "target_boundary_fraction",
    )
    for field in fields:
        if not math.isclose(
            float(formal_audit[field]),
            float(recomputed_audit[field]),
            rel_tol=0.0,
            abs_tol=ALIGNMENT_ABS_TOLERANCE,
        ):
            raise ValueError(f"Formal/recomputed Stage 2C target mismatch for {field}")


def _forbidden_presentation_columns(frame: pd.DataFrame) -> set[str]:
    return {
        str(column)
        for column in frame.columns
        if str(column).lower() in FORBIDDEN_PRESENTATION_FIELDS
    }


def validate_stage2a_spearman(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != SPEARMAN_COLUMNS:
        raise ValueError("Frozen Stage 2A Spearman schema mismatch")
    if frame["risk_score"].duplicated().any() or set(frame["risk_score"]) != set(
        stage2a.RISK_SCORE_ORDER
    ):
        raise ValueError("Frozen Stage 2A method set mismatch")
    ordered = frame.set_index("risk_score", drop=False).loc[
        list(stage2a.RISK_SCORE_ORDER)
    ].reset_index(drop=True)
    if _forbidden_presentation_columns(ordered):
        raise ValueError("Winner/rank fields are forbidden")
    return ordered.loc[:, SPEARMAN_COLUMNS]


def build_spearman_comparison(
    stage2a_spearman: pd.DataFrame, cqr_spearman: pd.DataFrame
) -> pd.DataFrame:
    baseline = validate_stage2a_spearman(stage2a_spearman)
    if tuple(cqr_spearman.columns) != SPEARMAN_COLUMNS or len(cqr_spearman) != 1:
        raise ValueError("CQR Spearman schema mismatch")
    result = pd.concat([baseline, cqr_spearman], ignore_index=True)
    if tuple(result["risk_score"]) != METHOD_ORDER:
        raise ValueError("Spearman comparison fixed method order changed")
    return result


def validate_stage2b_curves(
    frame: pd.DataFrame,
    abs_error: Sequence[float] | np.ndarray,
    sq_error: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    stage2b.validate_risk_coverage_curves(frame, abs_error, sq_error)
    if _forbidden_presentation_columns(frame):
        raise ValueError("Winner/rank fields are forbidden")
    ordered_parts = [
        frame.loc[frame["risk_score"] == score]
        .set_index("coverage_requested")
        .loc[list(stage2b.COVERAGE_GRID)]
        .reset_index()
        for score in stage2b.CURVE_SCORE_ORDER
    ]
    return pd.concat(ordered_parts, ignore_index=True).loc[:, stage2b.CURVE_COLUMNS]


def build_risk_coverage_comparison(
    stage2b_curves: pd.DataFrame,
    cqr_curve: pd.DataFrame,
    abs_error: Sequence[float] | np.ndarray,
    sq_error: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    baseline = validate_stage2b_curves(stage2b_curves, abs_error, sq_error)
    baseline_methods = baseline.loc[baseline["risk_score"] != stage2b.ORACLE]
    oracle = baseline.loc[baseline["risk_score"] == stage2b.ORACLE]
    result = pd.concat([baseline_methods, cqr_curve, oracle], ignore_index=True)
    if tuple(dict.fromkeys(result["risk_score"])) != CURVE_COMPARISON_ORDER:
        raise ValueError("Risk-coverage comparison fixed method order changed")
    if _forbidden_presentation_columns(result):
        raise ValueError("Winner/rank fields are forbidden")
    return result.loc[:, stage2b.CURVE_COLUMNS]


def validate_stage2b_ause(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != stage2b.AUSE_COLUMNS:
        raise ValueError("Frozen Stage 2B AUSE schema mismatch")
    if set(frame["risk_score"]) != set(stage2a.RISK_SCORE_ORDER):
        raise ValueError("Frozen Stage 2B AUSE method set mismatch")
    return frame.set_index("risk_score", drop=False).loc[
        list(stage2a.RISK_SCORE_ORDER)
    ].reset_index(drop=True)


def validate_stage2c_capture(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(stage2c.SUMMARY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Frozen Stage 2C capture missing fields: {sorted(missing)}")
    if set(frame["risk_score"]) != set(stage2a.RISK_SCORE_ORDER):
        raise ValueError("Frozen Stage 2C method set mismatch")
    parts = []
    for score in stage2a.RISK_SCORE_ORDER:
        rows = frame.loc[frame["risk_score"] == score].set_index(
            "risk_budget_fraction"
        ).loc[list(stage2c.RISK_BUDGET_FRACTIONS)].reset_index()
        parts.append(rows)
    ordered = pd.concat(parts, ignore_index=True).loc[:, stage2c.SUMMARY_COLUMNS]
    if _forbidden_presentation_columns(ordered):
        raise ValueError("Winner/rank fields are forbidden")
    return ordered


def build_capture_comparison(
    stage2c_capture: pd.DataFrame, cqr_capture: pd.DataFrame
) -> pd.DataFrame:
    baseline = validate_stage2c_capture(stage2c_capture)
    if tuple(cqr_capture.columns) != stage2c.SUMMARY_COLUMNS:
        raise ValueError("CQR capture schema mismatch")
    result = pd.concat([baseline, cqr_capture], ignore_index=True)
    if tuple(dict.fromkeys(result["risk_score"])) != METHOD_ORDER:
        raise ValueError("Capture comparison fixed method order changed")
    return result.loc[:, stage2c.SUMMARY_COLUMNS]


def load_stage2a_spearman() -> pd.DataFrame:
    path = validate_authorized_input_path(STAGE2A_SPEARMAN_INPUT, "stage2a_spearman")
    return validate_stage2a_spearman(pd.read_csv(path))


def load_stage2b_curves(
    abs_error: Sequence[float] | np.ndarray,
    sq_error: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    path = validate_authorized_input_path(STAGE2B_CURVES_INPUT, "stage2b_curves")
    return validate_stage2b_curves(pd.read_csv(path), abs_error, sq_error)


def load_stage2b_ause() -> pd.DataFrame:
    path = validate_authorized_input_path(STAGE2B_AUSE_INPUT, "stage2b_ause")
    return validate_stage2b_ause(pd.read_csv(path))


def load_stage2c_capture() -> pd.DataFrame:
    path = validate_authorized_input_path(STAGE2C_CAPTURE_INPUT, "stage2c_capture")
    return validate_stage2c_capture(pd.read_csv(path))


def load_stage2c_target_audit() -> Mapping[str, Any]:
    path = validate_authorized_input_path(
        STAGE2C_TARGET_AUDIT_INPUT, "stage2c_target_audit"
    )
    return _load_json_mapping(path)


def make_config() -> dict[str, Any]:
    validate_protocol_constants()
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "evaluation_role": EVALUATION_ROLE,
        "N": EXPECTED_N,
        "common_risk_target": COMMON_RISK_TARGET,
        "cqr_risk_score": CQR_RISK_SCORE,
        "cqr_risk_score_definition": CQR_RISK_SCORE_DEFINITION,
        "cqr_q50_error_evaluated": False,
        "raw_cqr_width_evaluated": False,
        "multiple_cqr_risk_scores_tested": False,
        "risk_score_round_decimals": RISK_TIE_ROUND_DECIMALS,
        "method_order": list(METHOD_ORDER),
        "curve_comparison_order": list(CURVE_COMPARISON_ORDER),
        "coverage_grid": list(stage2b.COVERAGE_GRID),
        "coverage_count_rule": "max(1, floor(coverage * N))",
        "risk_retention_direction": "retain lowest risk; reject highest risk",
        "risk_boundary_tie_policy": stage2b.BOUNDARY_TIE_POLICY,
        "oracle_score": COMMON_RISK_TARGET,
        "oracle_risk_rounding": False,
        "ause_integration": "trapezoidal integral on fixed coverage grid",
        "ause_coverage_span_normalized": False,
        "high_error_target_fraction": stage2c.HIGH_ERROR_TARGET_FRACTION,
        "risk_budget_fractions": list(stage2c.RISK_BUDGET_FRACTIONS),
        "risk_budget_counts": list(FROZEN_RISK_BUDGET_COUNTS),
        "risk_selection_direction": "highest risk first",
        "source_stage3a2_predictions": project_relative(CQR_PREDICTIONS_INPUT),
        "source_stage3a2_provenance": project_relative(STAGE3A2_PROVENANCE_INPUT),
        "source_stage2a_spearman": project_relative(STAGE2A_SPEARMAN_INPUT),
        "source_stage2b_curves": project_relative(STAGE2B_CURVES_INPUT),
        "source_stage2b_ause": project_relative(STAGE2B_AUSE_INPUT),
        "source_stage2c_capture": project_relative(STAGE2C_CAPTURE_INPUT),
        "source_stage2c_target_audit": project_relative(
            STAGE2C_TARGET_AUDIT_INPUT
        ),
    }


def make_provenance() -> dict[str, Any]:
    return {
        **make_config(),
        "stage2a_definitions_modified": False,
        "stage2b_definitions_modified": False,
        "stage2c_definitions_modified": False,
        "stage3a2_modified": False,
        "risk_target_modified": False,
        "new_risk_metrics_added": False,
        "formal_risk_method_selected": False,
        "formal_winner_declared": False,
        "random_test_accessed": False,
        "sealed_final_dates_accessed": False,
        "cp_calibration_used_for_risk_evaluation": False,
        "training_performed": False,
        "image_inference_performed": False,
        "mc_dropout_performed": False,
        "conformal_recalibration_performed": False,
        "cleaning_decision_performed": False,
        "economic_decision_performed": False,
    }


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def validate_formal_output_path(output_dir: Path) -> Path:
    candidate = _resolved(output_dir)
    if candidate != _resolved(OUTPUT_DIR):
        raise PermissionError(f"Unauthorized Stage 3B output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_outputs(
    output_dir: Path,
    cqr_spearman: pd.DataFrame,
    cqr_curve: pd.DataFrame,
    cqr_ause: pd.DataFrame,
    cqr_capture: pd.DataFrame,
    spearman_comparison: pd.DataFrame,
    coverage_comparison: pd.DataFrame,
    capture_comparison: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cqr_spearman.to_csv(
        output_dir / "cqr_risk_score_validity.csv", index=False, mode="x"
    )
    cqr_curve.to_csv(
        output_dir / "cqr_risk_coverage_curve.csv", index=False, mode="x"
    )
    cqr_ause.to_csv(
        output_dir / "cqr_risk_coverage_summary.csv", index=False, mode="x"
    )
    cqr_capture.to_csv(
        output_dir / "cqr_high_error_capture.csv", index=False, mode="x"
    )
    spearman_comparison.to_csv(
        output_dir / "risk_score_comparison_with_stage2a.csv",
        index=False,
        mode="x",
    )
    coverage_comparison.to_csv(
        output_dir / "risk_coverage_comparison_with_stage2b.csv",
        index=False,
        mode="x",
    )
    capture_comparison.to_csv(
        output_dir / "high_error_capture_comparison_with_stage2c.csv",
        index=False,
        mode="x",
    )
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "provenance.json", provenance)


def run(
    protocol: str = PROTOCOL,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the formal Stage 3B protocol only when explicitly invoked."""
    validate_protocol(protocol)
    output_dir = validate_formal_output_path(output_dir)
    ensure_output_available(output_dir)
    load_stage3a2_provenance()
    cqr_intervals = load_cqr_intervals()
    stage1a_base = stage2a.load_stage1a_base()
    evaluation = build_cqr_evaluation_table(stage1a_base, cqr_intervals)

    cqr_spearman = build_cqr_spearman(evaluation)
    cqr_curve = build_cqr_risk_coverage_curve(evaluation)
    oracle_curve = build_oracle_curve(evaluation)
    cqr_ause = build_cqr_ause_summary(cqr_curve, oracle_curve, len(evaluation))
    cqr_capture, recomputed_target_audit = build_cqr_high_error_capture(evaluation)

    frozen_spearman = load_stage2a_spearman()
    frozen_curves = load_stage2b_curves(
        evaluation["abs_error_mc_mean"], evaluation["sq_error_mc_mean"]
    )
    load_stage2b_ause()  # Schema/method guard; baseline AUSE is never recomputed.
    frozen_capture = load_stage2c_capture()
    frozen_target_audit = load_stage2c_target_audit()
    validate_frozen_target_audit_consistency(
        frozen_target_audit, recomputed_target_audit
    )

    spearman_comparison = build_spearman_comparison(
        frozen_spearman, cqr_spearman
    )
    coverage_comparison = build_risk_coverage_comparison(
        frozen_curves,
        cqr_curve,
        evaluation["abs_error_mc_mean"],
        evaluation["sq_error_mc_mean"],
    )
    capture_comparison = build_capture_comparison(
        frozen_capture, cqr_capture
    )
    config = make_config()
    provenance = make_provenance()
    write_outputs(
        output_dir,
        cqr_spearman,
        cqr_curve,
        cqr_ause,
        cqr_capture,
        spearman_comparison,
        coverage_comparison,
        capture_comparison,
        config,
        provenance,
    )
    return {"config": config, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(protocol=args.protocol, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
