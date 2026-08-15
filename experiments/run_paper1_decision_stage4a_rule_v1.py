"""Paper1 Stage 4A: predeclared rule-based cleaning decisions.

This table-only stage compares two binary point-threshold rules with a CQR
interval-relative CLEAN/WAIT/REVIEW rule on DECISION_DEVELOPMENT.  Thresholds
are fixed engineering scenarios, not values selected from evaluation results.
Importing this module reads no formal artifact and performs no model work.
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
from experiments import run_paper1_uq_stage1a_inference_v1 as stage1a


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"
STAGE = "decision_stage4a_rule_v1"
EVALUATION_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_N = 1844
TAU_GRID = (0.05, 0.10, 0.15, 0.20)
REFERENCE_TAU = 0.15

POINT_THRESHOLD = "point_threshold"
CQR_Q50_THRESHOLD = "cqr_q50_threshold"
CQR_INTERVAL_TRISTATE = "cqr_interval_tristate"
METHOD_ORDER = (POINT_THRESHOLD, CQR_Q50_THRESHOLD, CQR_INTERVAL_TRISTATE)

CLEAN = "CLEAN"
WAIT = "WAIT"
REVIEW = "REVIEW"
BINARY_ACTIONS = frozenset({CLEAN, WAIT})
TRISTATE_ACTIONS = frozenset({CLEAN, WAIT, REVIEW})

POINT_PREDICTIONS_INPUT = (
    stage1a.OUTPUT_DIR / "decision_development_predictions.csv"
)
CQR_PREDICTIONS_INPUT = stage3a2.OUTPUT_DIR / "cqr_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE
AUTHORIZED_INPUTS = {
    "stage1a_point_predictions": POINT_PREDICTIONS_INPUT,
    "stage3a2_cqr_predictions": CQR_PREDICTIONS_INPUT,
}

ALIGNMENT_ABS_TOLERANCE = 1e-15
IRRADIANCE_ALIGNMENT_ABS_TOLERANCE = 1e-12

ALIGNED_COLUMNS = (
    "sample_id",
    "date",
    "timestamp",
    "image_path",
    "role",
    "true_L",
    "irradiance",
    "point_pred",
    "q50",
    "lower",
    "upper",
)
ACTION_COLUMNS = (
    "sample_id",
    "date",
    "timestamp",
    "image_path",
    "role",
    "true_L",
    "tau",
    "method",
    "oracle_action",
    "predicted_action",
    "point_pred",
    "q50",
    "lower",
    "upper",
)
METRIC_COLUMNS = (
    "method",
    "tau",
    "evaluation_role",
    "N",
    "oracle_clean_n",
    "oracle_wait_n",
    "pred_clean_n",
    "pred_wait_n",
    "pred_review_n",
    "false_clean_n",
    "false_clean_rate_oracle_wait",
    "missed_clean_n",
    "missed_clean_rate_oracle_clean",
    "review_n",
    "review_rate",
    "auto_decision_n",
    "auto_decision_coverage",
    "auto_decision_error_n",
    "auto_decision_error_rate",
    "review_oracle_clean_n",
    "review_oracle_wait_n",
    "auto_correct_n",
)
COUNT_COLUMNS = (
    "method",
    "tau",
    "evaluation_role",
    "N",
    "oracle_clean_n",
    "oracle_wait_n",
    "pred_clean_n",
    "pred_wait_n",
    "pred_review_n",
    "false_clean_n",
    "missed_clean_n",
    "review_n",
    "auto_decision_n",
    "auto_decision_error_n",
    "review_oracle_clean_n",
    "review_oracle_wait_n",
    "auto_correct_n",
)
FORBIDDEN_PRESENTATION_FIELDS = frozenset(
    {"winner", "best", "rank", "ranking", "recommended", "selected_method"}
)


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_protocol_constants() -> None:
    if not (PROTOCOL == stage1a.PROTOCOL == stage3a2.PROTOCOL):
        raise ValueError("Stage 1A/3A2/4A protocol constants disagree")
    if EVALUATION_ROLE != stage3a2.DECISION_DEVELOPMENT_ROLE:
        raise ValueError("Stage 4A evaluation role drifted")
    if EXPECTED_N != stage3a2.EXPECTED_N[EVALUATION_ROLE]:
        raise ValueError("Stage 4A expected N drifted")
    if TAU_GRID != (0.05, 0.10, 0.15, 0.20):
        raise ValueError("Stage 4A tau grid drifted")
    if REFERENCE_TAU != 0.15 or REFERENCE_TAU not in TAU_GRID:
        raise ValueError("Stage 4A reference tau drifted")
    if METHOD_ORDER != (
        "point_threshold",
        "cqr_q50_threshold",
        "cqr_interval_tristate",
    ):
        raise ValueError("Stage 4A method order drifted")


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
        raise PermissionError("CP_CALIBRATION input is forbidden for Stage 4A")
    if source_key not in AUTHORIZED_INPUTS:
        raise PermissionError(f"Unauthorized Stage 4A source key: {source_key}")
    authorized = _resolved(AUTHORIZED_INPUTS[source_key])
    if candidate != authorized:
        raise PermissionError(f"Unauthorized Stage 4A input path: {candidate}")
    return candidate


def validate_point_predictions(
    frame: pd.DataFrame, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    if tuple(frame.columns) != stage1a.PREDICTION_COLUMNS:
        raise ValueError(
            "Stage 1A prediction schema mismatch: "
            f"expected {stage1a.PREDICTION_COLUMNS}, got {tuple(frame.columns)}"
        )
    stage1a.validate_manifest_frame(
        frame.loc[:, ("sample_id", "image_path", "date", "timestamp", "role")],
        EVALUATION_ROLE,
    )
    if enforce_expected_n and len(frame) != EXPECTED_N:
        raise ValueError(
            f"DECISION_DEVELOPMENT N guard failed: expected {EXPECTED_N}, got {len(frame)}"
        )
    numeric_columns = (
        "true_L",
        "irradiance",
        "point_pred",
        "mc_mean",
        "mc_std",
        "abs_error_point",
        "abs_error_mc_mean",
    )
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Stage 1A point artifact values must be finite")
    expected_point_error = np.abs(
        numeric["true_L"].to_numpy(dtype=np.float64)
        - numeric["point_pred"].to_numpy(dtype=np.float64)
    )
    if not np.allclose(
        numeric["abs_error_point"].to_numpy(dtype=np.float64),
        expected_point_error,
        rtol=0.0,
        atol=ALIGNMENT_ABS_TOLERANCE,
    ):
        raise ValueError("Stage 1A abs_error_point consistency guard failed")
    return frame.loc[:, stage1a.PREDICTION_COLUMNS].copy()


def load_point_predictions(path: Path = POINT_PREDICTIONS_INPUT) -> pd.DataFrame:
    authorized = validate_authorized_input_path(path, "stage1a_point_predictions")
    return validate_point_predictions(pd.read_csv(authorized))


def validate_cqr_predictions(
    frame: pd.DataFrame, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    return stage3a2.validate_cqr_predictions(
        frame, enforce_expected_n=enforce_expected_n
    )


def load_cqr_predictions(path: Path = CQR_PREDICTIONS_INPUT) -> pd.DataFrame:
    authorized = validate_authorized_input_path(path, "stage3a2_cqr_predictions")
    return validate_cqr_predictions(pd.read_csv(authorized))


def align_decision_inputs(
    point_predictions: pd.DataFrame,
    cqr_predictions: pd.DataFrame,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    point = validate_point_predictions(
        point_predictions, enforce_expected_n=enforce_expected_n
    )
    cqr = validate_cqr_predictions(
        cqr_predictions, enforce_expected_n=enforce_expected_n
    )
    if len(point) != len(cqr):
        raise ValueError(f"Point/CQR N mismatch: {len(point)} != {len(cqr)}")
    if point["sample_id"].duplicated().any() or cqr["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique before alignment")
    point_ids = point["sample_id"].astype(str).tolist()
    cqr_ids = cqr["sample_id"].astype(str)
    if set(point_ids) != set(cqr_ids):
        missing = set(point_ids) - set(cqr_ids)
        extra = set(cqr_ids) - set(point_ids)
        raise ValueError(
            "Point/CQR sample_id set mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    aligned_cqr = (
        cqr.set_index(cqr_ids, drop=False).loc[point_ids].reset_index(drop=True)
    )
    point = point.reset_index(drop=True)
    for field in ("image_path", "date", "timestamp", "role"):
        if not np.array_equal(
            point[field].astype(str).to_numpy(),
            aligned_cqr[field].astype(str).to_numpy(),
        ):
            raise ValueError(f"Point/CQR {field} alignment mismatch")
    for field, tolerance in (
        ("true_L", ALIGNMENT_ABS_TOLERANCE),
        ("irradiance", IRRADIANCE_ALIGNMENT_ABS_TOLERANCE),
    ):
        if not np.allclose(
            point[field].to_numpy(dtype=np.float64),
            aligned_cqr[field].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(f"Point/CQR {field} alignment mismatch")
    result = pd.DataFrame(
        {
            "sample_id": point["sample_id"],
            "date": point["date"],
            "timestamp": point["timestamp"],
            "image_path": point["image_path"],
            "role": point["role"],
            "true_L": point["true_L"].to_numpy(dtype=np.float64),
            "irradiance": point["irradiance"].to_numpy(dtype=np.float64),
            "point_pred": point["point_pred"].to_numpy(dtype=np.float64),
            "q50": aligned_cqr["q50"].to_numpy(dtype=np.float64),
            "lower": aligned_cqr["lower"].to_numpy(dtype=np.float64),
            "upper": aligned_cqr["upper"].to_numpy(dtype=np.float64),
        }
    )
    validate_aligned_inputs(result, enforce_expected_n=enforce_expected_n)
    return result.loc[:, ALIGNED_COLUMNS]


def validate_aligned_inputs(
    frame: pd.DataFrame, *, enforce_expected_n: bool = True
) -> None:
    missing = set(ALIGNED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Aligned decision input missing fields: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Aligned decision input is empty")
    if enforce_expected_n and len(frame) != EXPECTED_N:
        raise ValueError(
            f"DECISION_DEVELOPMENT N guard failed: expected {EXPECTED_N}, got {len(frame)}"
        )
    if set(frame["role"].astype(str)) != {EVALUATION_ROLE}:
        raise PermissionError("Only DECISION_DEVELOPMENT is authorized")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-null and unique")
    if frame["image_path"].isna().any() or frame["image_path"].duplicated().any():
        raise ValueError("image_path must be non-null and unique")
    normalized_dates = pd.to_datetime(frame["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    sealed = set(normalized_dates) & stage1a.SEALED_FINAL_DATES
    if sealed:
        raise PermissionError(f"Sealed final date rejected: {sorted(sealed)}")
    locators = frame["image_path"].astype(str)
    if locators.str.lower().str.contains("random_test", regex=False).any():
        raise PermissionError("RANDOM_TEST locator rejected")
    for sealed_date in stage1a.SEALED_FINAL_DATES:
        if locators.str.contains(sealed_date, regex=False).any():
            raise PermissionError(f"Sealed final date locator rejected: {sealed_date}")
    numeric_columns = ("true_L", "irradiance", "point_pred", "q50", "lower", "upper")
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Aligned decision values must be finite")
    lower = numeric["lower"].to_numpy(dtype=np.float64)
    upper = numeric["upper"].to_numpy(dtype=np.float64)
    if np.any(lower < 0.0) or np.any(upper > 1.0):
        raise ValueError("CQR interval bounds must lie in [0,1]")
    if np.any(lower > upper):
        raise ValueError("CQR lower exceeds upper")


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def validate_tau(tau: float) -> float:
    value = float(tau)
    if value not in TAU_GRID:
        raise ValueError(f"Unauthorized Stage 4A tau: {tau}")
    return value


def oracle_actions(true_l: Sequence[float] | np.ndarray, tau: float) -> np.ndarray:
    values = _finite_vector(true_l, "true_L")
    threshold = validate_tau(tau)
    return np.where(values > threshold, CLEAN, WAIT)


def point_threshold_actions(
    point_pred: Sequence[float] | np.ndarray, tau: float
) -> np.ndarray:
    values = _finite_vector(point_pred, "point_pred")
    threshold = validate_tau(tau)
    return np.where(values > threshold, CLEAN, WAIT)


def cqr_q50_threshold_actions(
    q50: Sequence[float] | np.ndarray, tau: float
) -> np.ndarray:
    values = _finite_vector(q50, "q50")
    threshold = validate_tau(tau)
    return np.where(values > threshold, CLEAN, WAIT)


def cqr_interval_tristate_actions(
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    tau: float,
) -> np.ndarray:
    lower_values = _finite_vector(lower, "lower")
    upper_values = _finite_vector(upper, "upper")
    if len(lower_values) != len(upper_values):
        raise ValueError("CQR lower/upper lengths differ")
    if np.any(lower_values > upper_values):
        raise ValueError("CQR lower exceeds upper")
    threshold = validate_tau(tau)
    actions = np.full(len(lower_values), REVIEW, dtype=object)
    actions[lower_values > threshold] = CLEAN
    actions[upper_values < threshold] = WAIT
    return actions


def _actions_for_method(frame: pd.DataFrame, method: str, tau: float) -> np.ndarray:
    if method == POINT_THRESHOLD:
        return point_threshold_actions(frame["point_pred"], tau)
    if method == CQR_Q50_THRESHOLD:
        return cqr_q50_threshold_actions(frame["q50"], tau)
    if method == CQR_INTERVAL_TRISTATE:
        return cqr_interval_tristate_actions(frame["lower"], frame["upper"], tau)
    raise ValueError(f"Unauthorized Stage 4A method: {method}")


def build_decision_actions(
    aligned: pd.DataFrame, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    validate_protocol_constants()
    validate_aligned_inputs(aligned, enforce_expected_n=enforce_expected_n)
    records: list[pd.DataFrame] = []
    for method in METHOD_ORDER:
        for tau in TAU_GRID:
            part = aligned.loc[
                :,
                (
                    "sample_id",
                    "date",
                    "timestamp",
                    "image_path",
                    "role",
                    "true_L",
                    "point_pred",
                    "q50",
                    "lower",
                    "upper",
                ),
            ].copy()
            part["tau"] = tau
            part["method"] = method
            part["oracle_action"] = oracle_actions(part["true_L"], tau)
            part["predicted_action"] = _actions_for_method(part, method, tau)
            records.append(part.loc[:, ACTION_COLUMNS])
    result = pd.concat(records, ignore_index=True)
    validate_decision_actions(
        result, n_samples=len(aligned), enforce_expected_n=enforce_expected_n
    )
    return result.loc[:, ACTION_COLUMNS]


def validate_decision_actions(
    actions: pd.DataFrame,
    *,
    n_samples: int,
    enforce_expected_n: bool = True,
) -> None:
    if tuple(actions.columns) != ACTION_COLUMNS:
        raise ValueError("Stage 4A decision-action schema mismatch")
    expected_samples = EXPECTED_N if enforce_expected_n else int(n_samples)
    expected_rows = expected_samples * len(TAU_GRID) * len(METHOD_ORDER)
    if len(actions) != expected_rows:
        raise ValueError(
            f"Stage 4A action row-count guard failed: expected {expected_rows}, "
            f"got {len(actions)}"
        )
    if set(actions["role"].astype(str)) != {EVALUATION_ROLE}:
        raise PermissionError("Only DECISION_DEVELOPMENT actions are authorized")
    if tuple(dict.fromkeys(actions["method"].astype(str))) != METHOD_ORDER:
        raise ValueError("Stage 4A action method order mismatch")
    for method in METHOD_ORDER:
        method_rows = actions.loc[actions["method"] == method]
        if tuple(dict.fromkeys(method_rows["tau"].astype(float))) != TAU_GRID:
            raise ValueError("Stage 4A action tau order mismatch")
    key_columns = ["sample_id", "method", "tau"]
    if actions.loc[:, key_columns].isna().any().any() or actions.duplicated(
        key_columns
    ).any():
        raise ValueError("Each sample/method/tau action must be unique")
    if actions["sample_id"].nunique() != expected_samples:
        raise ValueError("Stage 4A action sample count mismatch")
    if not set(actions["oracle_action"].astype(str)) <= BINARY_ACTIONS:
        raise ValueError("Oracle action must be CLEAN or WAIT")
    for method in METHOD_ORDER:
        allowed = TRISTATE_ACTIONS if method == CQR_INTERVAL_TRISTATE else BINARY_ACTIONS
        observed = set(
            actions.loc[actions["method"] == method, "predicted_action"].astype(str)
        )
        if not observed <= allowed:
            raise ValueError(f"Invalid predicted action for {method}")
    for tau in TAU_GRID:
        for method in METHOD_ORDER:
            rows = actions.loc[
                (actions["tau"] == tau) & (actions["method"] == method)
            ]
            if not np.array_equal(
                rows["oracle_action"].to_numpy(), oracle_actions(rows["true_L"], tau)
            ):
                raise ValueError("Stored oracle action is inconsistent")
            expected_predicted = _actions_for_method(rows, method, tau)
            if not np.array_equal(
                rows["predicted_action"].to_numpy(), expected_predicted
            ):
                raise ValueError("Stored predicted action is inconsistent")


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize_action_group(
    oracle_action: Sequence[str], predicted_action: Sequence[str]
) -> dict[str, int | float]:
    oracle = np.asarray(oracle_action, dtype=object)
    predicted = np.asarray(predicted_action, dtype=object)
    if oracle.ndim != 1 or predicted.ndim != 1 or len(oracle) != len(predicted):
        raise ValueError("Oracle/predicted actions must be equal-length vectors")
    if len(oracle) == 0:
        raise ValueError("Action group is empty")
    if not set(oracle.astype(str)) <= BINARY_ACTIONS:
        raise ValueError("Oracle action must be CLEAN or WAIT")
    if not set(predicted.astype(str)) <= TRISTATE_ACTIONS:
        raise ValueError("Predicted action must be CLEAN, WAIT, or REVIEW")

    oracle_clean = oracle == CLEAN
    oracle_wait = oracle == WAIT
    pred_clean = predicted == CLEAN
    pred_wait = predicted == WAIT
    pred_review = predicted == REVIEW
    false_clean = pred_clean & oracle_wait
    missed_clean = pred_wait & oracle_clean
    auto = pred_clean | pred_wait
    auto_error_n = int(false_clean.sum() + missed_clean.sum())
    review_n = int(pred_review.sum())
    auto_n = int(auto.sum())
    n = len(oracle)
    result: dict[str, int | float] = {
        "N": n,
        "oracle_clean_n": int(oracle_clean.sum()),
        "oracle_wait_n": int(oracle_wait.sum()),
        "pred_clean_n": int(pred_clean.sum()),
        "pred_wait_n": int(pred_wait.sum()),
        "pred_review_n": review_n,
        "false_clean_n": int(false_clean.sum()),
        "false_clean_rate_oracle_wait": _rate(
            int(false_clean.sum()), int(oracle_wait.sum())
        ),
        "missed_clean_n": int(missed_clean.sum()),
        "missed_clean_rate_oracle_clean": _rate(
            int(missed_clean.sum()), int(oracle_clean.sum())
        ),
        "review_n": review_n,
        "review_rate": review_n / n,
        "auto_decision_n": auto_n,
        "auto_decision_coverage": auto_n / n,
        "auto_decision_error_n": auto_error_n,
        "auto_decision_error_rate": _rate(auto_error_n, auto_n),
        "review_oracle_clean_n": int((pred_review & oracle_clean).sum()),
        "review_oracle_wait_n": int((pred_review & oracle_wait).sum()),
        "auto_correct_n": int((auto & (predicted == oracle)).sum()),
    }
    if not math.isclose(
        float(result["auto_decision_coverage"]),
        1.0 - float(result["review_rate"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Auto-decision coverage must equal 1-review rate")
    return result


def build_decision_metrics(
    actions: pd.DataFrame,
    *,
    n_samples: int,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    validate_decision_actions(
        actions, n_samples=n_samples, enforce_expected_n=enforce_expected_n
    )
    records: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for tau in TAU_GRID:
            rows = actions.loc[
                (actions["method"] == method) & (actions["tau"] == tau)
            ]
            records.append(
                {
                    "method": method,
                    "tau": tau,
                    "evaluation_role": EVALUATION_ROLE,
                    **summarize_action_group(
                        rows["oracle_action"], rows["predicted_action"]
                    ),
                }
            )
    result = pd.DataFrame.from_records(records).loc[:, METRIC_COLUMNS]
    validate_decision_metrics(
        result, n_samples=n_samples, enforce_expected_n=enforce_expected_n
    )
    return result


def validate_decision_metrics(
    metrics: pd.DataFrame,
    *,
    n_samples: int,
    enforce_expected_n: bool = True,
) -> None:
    if tuple(metrics.columns) != METRIC_COLUMNS:
        raise ValueError("Stage 4A decision metric schema mismatch")
    if len(metrics) != len(METHOD_ORDER) * len(TAU_GRID):
        raise ValueError("Stage 4A decision metric row count mismatch")
    if tuple(dict.fromkeys(metrics["method"].astype(str))) != METHOD_ORDER:
        raise ValueError("Stage 4A decision metric method order mismatch")
    for method in METHOD_ORDER:
        rows = metrics.loc[metrics["method"] == method]
        if tuple(rows["tau"].astype(float)) != TAU_GRID:
            raise ValueError("Stage 4A decision metric tau order mismatch")
    expected_n = EXPECTED_N if enforce_expected_n else n_samples
    if set(metrics["N"].astype(int)) != {expected_n}:
        raise ValueError("Stage 4A decision metric N mismatch")
    if set(metrics["evaluation_role"].astype(str)) != {EVALUATION_ROLE}:
        raise PermissionError("Stage 4A decision metric role mismatch")
    if set(metrics.columns) & FORBIDDEN_PRESENTATION_FIELDS:
        raise ValueError("Winner/rank fields are forbidden")
    if not np.allclose(
        metrics["auto_decision_coverage"].to_numpy(dtype=np.float64),
        1.0 - metrics["review_rate"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Auto-decision coverage/review-rate identity failed")
    binary = metrics["method"].isin((POINT_THRESHOLD, CQR_Q50_THRESHOLD))
    if not (metrics.loc[binary, "review_n"] == 0).all():
        raise ValueError("Binary point baselines cannot REVIEW")
    if not (metrics.loc[binary, "auto_decision_coverage"] == 1.0).all():
        raise ValueError("Binary point baselines must have full auto coverage")


def build_decision_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    missing = set(COUNT_COLUMNS) - set(metrics.columns)
    if missing:
        raise ValueError(f"Decision counts source missing fields: {sorted(missing)}")
    return metrics.loc[:, COUNT_COLUMNS].copy()


def make_config() -> dict[str, Any]:
    validate_protocol_constants()
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "evaluation_role": EVALUATION_ROLE,
        "N": EXPECTED_N,
        "tau_grid": list(TAU_GRID),
        "reference_tau": REFERENCE_TAU,
        "thresholds_are_scenarios": True,
        "universal_optimal_threshold_claimed": False,
        "thresholds_selected_using_decision_results": False,
        "threshold_variable": "DeepSolarEye relative power loss L",
        "literature_reference_variable": "temperature-corrected performance ratio",
        "threshold_variables_are_physically_equivalent": False,
        "all_tau_reported_at_equal_status": True,
        "reference_tau_used_for_tuning": False,
        "method_order": list(METHOD_ORDER),
        "oracle_rule": "true_L > tau => CLEAN else WAIT",
        "point_rule": "point_pred > tau => CLEAN else WAIT",
        "cqr_q50_rule": "q50 > tau => CLEAN else WAIT",
        "cqr_interval_rule": (
            "lower > tau => CLEAN; upper < tau => WAIT; otherwise REVIEW"
        ),
        "source_stage1a_point_predictions": project_relative(
            POINT_PREDICTIONS_INPUT
        ),
        "stage1a_deterministic_point_column": "point_pred",
        "stage1a_mc_mean_used_as_point_prediction": False,
        "source_stage3a2_cqr_predictions": project_relative(
            CQR_PREDICTIONS_INPUT
        ),
        "stage3a2_interval_recomputed": False,
        "action_columns": list(ACTION_COLUMNS),
        "metric_columns": list(METRIC_COLUMNS),
        "count_columns": list(COUNT_COLUMNS),
        "expected_action_rows": EXPECTED_N * len(TAU_GRID) * len(METHOD_ORDER),
    }


def make_provenance() -> dict[str, Any]:
    return {
        **make_config(),
        "review_rate_preselected": False,
        "stage3b_risk_budget_used_as_review_rule": False,
        "decision_safety_must_be_interpreted_with_auto_coverage": True,
        "nominal_interval_coverage_implies_decision_accuracy": False,
        "training_performed": False,
        "image_inference_performed": False,
        "mc_dropout_performed": False,
        "conformal_recalibration_performed": False,
        "risk_score_development_performed": False,
        "economic_analysis_performed": False,
        "random_test_accessed": False,
        "sealed_final_dates_accessed": False,
        "formal_decision_method_selected": False,
        "formal_winner_declared": False,
    }


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def validate_formal_output_path(output_dir: Path) -> Path:
    candidate = _resolved(output_dir)
    if candidate != _resolved(OUTPUT_DIR):
        raise PermissionError(f"Unauthorized Stage 4A output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_outputs(
    output_dir: Path,
    actions: pd.DataFrame,
    metrics: pd.DataFrame,
    counts: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions.to_csv(output_dir / "decision_actions.csv", index=False, mode="x")
    metrics.to_csv(output_dir / "decision_metrics.csv", index=False, mode="x")
    counts.to_csv(output_dir / "decision_counts.csv", index=False, mode="x")
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "provenance.json", provenance)


def run(
    protocol: str = PROTOCOL,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the formal Stage 4A protocol only when explicitly invoked."""
    validate_protocol(protocol)
    output_dir = validate_formal_output_path(output_dir)
    ensure_output_available(output_dir)
    point = load_point_predictions()
    cqr = load_cqr_predictions()
    aligned = align_decision_inputs(point, cqr)
    actions = build_decision_actions(aligned)
    metrics = build_decision_metrics(actions, n_samples=len(aligned))
    counts = build_decision_counts(metrics)
    config = make_config()
    provenance = make_provenance()
    write_outputs(output_dir, actions, metrics, counts, config, provenance)
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
