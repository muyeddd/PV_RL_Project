"""Paper1 CQR Stage 3A2: conformal calibration and interval evaluation.

This table-only stage consumes frozen Stage 3A1 quantile predictions.  It
calibrates exactly once on CP_CALIBRATION, freezes the resulting CQR qhat, and
only then attaches DECISION_DEVELOPMENT truth for interval evaluation.  It
performs no training, image inference, MC Dropout, risk analysis, cleaning, or
economic decision.  Importing this module reads no formal artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments import run_paper1_cqr_stage3a1_inference_v1 as stage3a1
from experiments import run_paper1_uq_stage1b_intervals_v1 as stage1b


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"
STAGE = "cqr_stage3a2_intervals_v1"
METHOD = "cqr_v1"
ALPHA = 0.10
TARGET_COVERAGE = 0.90
QUANTILE_LEVELS = (0.05, 0.50, 0.95)
QUANTILE_METHOD = "higher"
CQR_SCORE_DEFINITION = "max(q05-y, y-q95), negative values allowed"
FINITE_SAMPLE_RULE = "min(ceil((n+1)*(1-alpha))/n,1.0)"

CP_CALIBRATION_ROLE = stage3a1.CP_CALIBRATION_ROLE
DECISION_DEVELOPMENT_ROLE = stage3a1.DECISION_DEVELOPMENT_ROLE
EXPECTED_N = dict(stage3a1.EXPECTED_N)
SOURCE_CQR_CHECKPOINT_SHA256 = stage3a1.SOURCE_CQR_CHECKPOINT_SHA256

STAGE3A1_INPUT_DIR = stage3a1.OUTPUT_DIR
CP_PREDICTIONS_INPUT = STAGE3A1_INPUT_DIR / "cp_calibration_predictions.csv"
DECISION_PREDICTIONS_INPUT = (
    STAGE3A1_INPUT_DIR / "decision_development_predictions.csv"
)
AUTHORIZED_PREDICTION_INPUTS = {
    CP_CALIBRATION_ROLE: CP_PREDICTIONS_INPUT,
    DECISION_DEVELOPMENT_ROLE: DECISION_PREDICTIONS_INPUT,
}
AUTHORIZED_TRUTH_MANIFESTS = {
    CP_CALIBRATION_ROLE: stage3a1.CP_CALIBRATION_MANIFEST,
    DECISION_DEVELOPMENT_ROLE: stage3a1.DECISION_DEVELOPMENT_MANIFEST,
}

STAGE1B_GLOBAL_METRICS_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / PROTOCOL
    / "uq_stage1b_intervals_v1"
    / "all_interval_metrics.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE

BASELINE_METHOD_ORDER = (
    "raw_mc",
    "split_cp",
    "irradiance_mondrian_cp",
    "pred_l_mondrian_cp",
    "pred_l_mondrian_mc_interval_cp",
    "pred_l_mondrian_std_mc_cp",
    METHOD,
)
GLOBAL_METRIC_COLUMNS = (
    "method",
    "evaluation_role",
    "N",
    "alpha",
    "target_coverage",
    "PICP",
    "MPIW",
    "median_width",
    "coverage_error",
    "mean_interval_score_alpha_0p10",
)
COMPARISON_COLUMNS = (
    "method",
    "PICP",
    "MPIW",
    "median_width",
    "coverage_error",
    "mean_interval_score_alpha_0p10",
)
CONDITIONAL_COLUMNS = (
    "method",
    "evaluation_role",
    "binning_variable",
    "bin_label",
    "N",
    "PICP",
    "MPIW",
)
ALIGNED_COLUMNS = (
    "sample_id",
    "date",
    "timestamp",
    "image_path",
    "role",
    "true_L",
    "irradiance",
    "q05",
    "q50",
    "q95",
)
PREDICTION_OUTPUT_COLUMNS = ALIGNED_COLUMNS + (
    "method",
    "lower",
    "upper",
    "width",
    "covered",
    "raw_width",
    "lower_clipped",
    "upper_clipped",
)
FORBIDDEN_COMPARISON_COLUMNS = frozenset({"winner", "best", "rank", "ranking"})
IRRADIANCE_ALIGNMENT_ABS_TOL = 1e-12
INTERVAL_CONSISTENCY_ABS_TOL = 1e-15


@dataclass(frozen=True)
class FrozenCQRCalibration:
    calibration_role: str
    n_calibration_scores: int
    alpha: float
    target_coverage: float
    quantile_fraction: float
    order_statistic_rank: int
    quantile_method: str
    score_definition: str
    qhat: float
    qhat_is_negative: bool
    qhat_is_positive: bool


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_protocol_constants() -> None:
    if PROTOCOL != stage1b.PROTOCOL or PROTOCOL != stage3a1.PROTOCOL:
        raise ValueError("Paper1 protocol constants disagree")
    if ALPHA != stage1b.ALPHA or TARGET_COVERAGE != stage1b.TARGET_COVERAGE:
        raise ValueError("Stage 3A2 must use the frozen Stage 1B alpha and coverage")
    if QUANTILE_METHOD != "higher":
        raise ValueError("Stage 3A2 conformal quantile method must be higher")
    if QUANTILE_LEVELS != stage3a1.QUANTILE_LEVELS:
        raise ValueError("Stage 3A2 quantile levels disagree with Stage 3A1")


def validate_protocol(protocol: str) -> None:
    validate_protocol_constants()
    if protocol != PROTOCOL:
        raise PermissionError(f"Unauthorized protocol: {protocol!r}")


def validate_prediction_input_path(path: Path, expected_role: str) -> Path:
    if expected_role not in AUTHORIZED_PREDICTION_INPUTS:
        raise PermissionError(f"Forbidden Stage 3A2 prediction role: {expected_role}")
    candidate = _resolved(path)
    lowered = str(candidate).lower()
    if "random_test" in lowered:
        raise PermissionError("RANDOM_TEST prediction input is forbidden")
    authorized = _resolved(AUTHORIZED_PREDICTION_INPUTS[expected_role])
    if candidate != authorized:
        raise PermissionError(
            f"Prediction input is not authorized for {expected_role}: {candidate}"
        )
    return candidate


def load_stage3a1_predictions(path: Path, expected_role: str) -> pd.DataFrame:
    authorized = validate_prediction_input_path(path, expected_role)
    frame = pd.read_csv(authorized)
    return stage3a1.validate_prediction_frame(frame, expected_role)


def validate_truth_manifest_path(path: Path, expected_role: str) -> Path:
    if expected_role not in AUTHORIZED_TRUTH_MANIFESTS:
        raise PermissionError(f"Forbidden Stage 3A2 truth role: {expected_role}")
    candidate = _resolved(path)
    lowered = str(candidate).lower()
    if "random_test" in lowered:
        raise PermissionError("RANDOM_TEST truth manifest is forbidden")
    authorized = _resolved(AUTHORIZED_TRUTH_MANIFESTS[expected_role])
    if candidate != authorized:
        raise PermissionError(
            f"Truth manifest is not authorized for {expected_role}: {candidate}"
        )
    return candidate


def load_truth_manifest(path: Path, expected_role: str) -> pd.DataFrame:
    authorized = validate_truth_manifest_path(path, expected_role)
    # Stage 3A1's loader applies exact path/role/N/locator/date guards and reads
    # metadata only.  Truth is attached separately below by the frozen parser.
    return stage3a1.load_role_manifest(authorized, expected_role)


def _validate_expected_role_n(
    frame: pd.DataFrame,
    expected_role: str,
    *,
    enforce_expected_n: bool,
) -> None:
    if expected_role not in EXPECTED_N:
        raise PermissionError(f"Forbidden Stage 3A2 role: {expected_role}")
    if frame.empty:
        raise ValueError(f"{expected_role} table is empty")
    if set(frame["role"].astype(str)) != {expected_role}:
        raise PermissionError(f"Role guard failed for {expected_role}")
    if enforce_expected_n and len(frame) != EXPECTED_N[expected_role]:
        raise ValueError(
            f"{expected_role} N guard failed: expected {EXPECTED_N[expected_role]}, "
            f"got {len(frame)}"
        )


def attach_truth_by_sample_id(
    predictions: pd.DataFrame,
    truth_manifest: pd.DataFrame,
    expected_role: str,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    prediction_frame = stage3a1.validate_prediction_frame(
        predictions, expected_role, enforce_expected_n=enforce_expected_n
    )
    manifest_frame = stage3a1.validate_manifest_frame(
        truth_manifest, expected_role, enforce_expected_n=enforce_expected_n
    )
    _validate_expected_role_n(
        prediction_frame, expected_role, enforce_expected_n=enforce_expected_n
    )
    _validate_expected_role_n(
        manifest_frame, expected_role, enforce_expected_n=enforce_expected_n
    )

    prediction_ids = set(prediction_frame["sample_id"])
    truth_ids = set(manifest_frame["sample_id"])
    if prediction_ids != truth_ids:
        raise ValueError(
            "Prediction/truth sample_id sets differ; "
            f"prediction_only={sorted(prediction_ids - truth_ids)[:5]}, "
            f"truth_only={sorted(truth_ids - prediction_ids)[:5]}"
        )

    truth_values = stage3a1.cqr_train.point_train.attach_development_values(
        manifest_frame
    )
    truth_values = truth_values.loc[
        :,
        (
            "sample_id",
            "date",
            "timestamp",
            "image_path",
            "role",
            "true_L",
            "irradiance_raw",
        ),
    ].copy()
    prediction_frame = prediction_frame.copy()
    prediction_frame["_prediction_order"] = np.arange(len(prediction_frame))
    merged = prediction_frame.merge(
        truth_values,
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_prediction", "_truth"),
        indicator=True,
        sort=False,
    )
    if len(merged) != len(prediction_frame) or set(merged["_merge"]) != {"both"}:
        raise ValueError("Prediction/truth one-to-one merge failed")
    for field in ("image_path", "role", "date", "timestamp"):
        prediction_values = merged[f"{field}_prediction"].astype(str).to_numpy()
        truth_field_values = merged[f"{field}_truth"].astype(str).to_numpy()
        if not np.array_equal(prediction_values, truth_field_values):
            raise ValueError(f"Prediction/truth {field} alignment mismatch")
    if not np.allclose(
        merged["irradiance"].to_numpy(dtype=np.float64),
        merged["irradiance_raw"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=IRRADIANCE_ALIGNMENT_ABS_TOL,
    ):
        raise ValueError("Prediction/truth irradiance alignment mismatch")

    merged = merged.sort_values("_prediction_order", kind="stable")
    result = pd.DataFrame(
        {
            "sample_id": merged["sample_id"],
            "date": merged["date_prediction"],
            "timestamp": merged["timestamp_prediction"],
            "image_path": merged["image_path_prediction"],
            "role": merged["role_prediction"],
            "true_L": pd.to_numeric(merged["true_L"], errors="raise"),
            "irradiance": pd.to_numeric(merged["irradiance"], errors="raise"),
            "q05": pd.to_numeric(merged["q05"], errors="raise"),
            "q50": pd.to_numeric(merged["q50"], errors="raise"),
            "q95": pd.to_numeric(merged["q95"], errors="raise"),
        }
    ).reset_index(drop=True)
    numeric = result.loc[:, ["true_L", "irradiance", "q05", "q50", "q95"]]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Aligned prediction/truth values must be finite")
    if np.any(result["true_L"].to_numpy(dtype=np.float64) < 0.0) or np.any(
        result["true_L"].to_numpy(dtype=np.float64) > 1.0
    ):
        raise ValueError("Aligned truth falls outside [0, 1]")
    return result.loc[:, ALIGNED_COLUMNS]


def validate_cp_decision_isolation(
    calibration: pd.DataFrame, decision: pd.DataFrame
) -> None:
    if set(calibration["role"].astype(str)) != {CP_CALIBRATION_ROLE}:
        raise PermissionError("Calibration table must contain CP_CALIBRATION only")
    if set(decision["role"].astype(str)) != {DECISION_DEVELOPMENT_ROLE}:
        raise PermissionError("Evaluation table must contain DECISION_DEVELOPMENT only")
    if set(calibration["sample_id"]) & set(decision["sample_id"]):
        raise ValueError("CP_CALIBRATION/DECISION_DEVELOPMENT sample_id overlap")
    if set(calibration["image_path"]) & set(decision["image_path"]):
        raise ValueError("CP_CALIBRATION/DECISION_DEVELOPMENT image_path overlap")


def _finite_vector(
    values: Sequence[float] | np.ndarray, name: str, *, expected_n: int | None = None
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if expected_n is not None and len(vector) != expected_n:
        raise ValueError(f"{name} N guard failed: expected {expected_n}, got {len(vector)}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def cqr_conformity_scores(
    truth: Sequence[float] | np.ndarray,
    q05: Sequence[float] | np.ndarray,
    q95: Sequence[float] | np.ndarray,
    *,
    expected_n: int | None = None,
) -> np.ndarray:
    y = _finite_vector(truth, "CQR calibration truth", expected_n=expected_n)
    lower = _finite_vector(q05, "CQR q05", expected_n=expected_n)
    upper = _finite_vector(q95, "CQR q95", expected_n=expected_n)
    if not (len(y) == len(lower) == len(upper)):
        raise ValueError("CQR score inputs must have equal length")
    if np.any(lower > upper):
        raise ValueError("CQR score inputs contain quantile crossing")
    # Standard CQR deliberately permits negative scores.  Do not add zero.
    return np.maximum(lower - y, y - upper)


def calibrate_cqr(
    calibration: pd.DataFrame,
    *,
    enforce_expected_n: bool = True,
) -> FrozenCQRCalibration:
    validate_protocol_constants()
    _validate_expected_role_n(
        calibration,
        CP_CALIBRATION_ROLE,
        enforce_expected_n=enforce_expected_n,
    )
    n = EXPECTED_N[CP_CALIBRATION_ROLE] if enforce_expected_n else len(calibration)
    stage3a1.validate_quantile_array(
        calibration.loc[:, ["q05", "q50", "q95"]].to_numpy(), n
    )
    scores = cqr_conformity_scores(
        calibration["true_L"],
        calibration["q05"],
        calibration["q95"],
        expected_n=n,
    )
    raw_rank = math.ceil((n + 1) * (1.0 - ALPHA))
    rank = min(raw_rank, n)
    fraction = min(raw_rank / n, 1.0)
    qhat = float(stage1b.conformal_quantile(scores))
    if not math.isfinite(qhat):
        raise ValueError("CQR qhat must be finite")
    return FrozenCQRCalibration(
        calibration_role=CP_CALIBRATION_ROLE,
        n_calibration_scores=n,
        alpha=ALPHA,
        target_coverage=TARGET_COVERAGE,
        quantile_fraction=float(fraction),
        order_statistic_rank=int(rank),
        quantile_method=QUANTILE_METHOD,
        score_definition=CQR_SCORE_DEFINITION,
        qhat=qhat,
        qhat_is_negative=qhat < 0.0,
        qhat_is_positive=qhat > 0.0,
    )


def validate_cqr_predictions(
    predictions: pd.DataFrame,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    if tuple(predictions.columns) != PREDICTION_OUTPUT_COLUMNS:
        raise ValueError(
            "CQR interval prediction schema mismatch: "
            f"expected {PREDICTION_OUTPUT_COLUMNS}, got {tuple(predictions.columns)}"
        )
    _validate_expected_role_n(
        predictions,
        DECISION_DEVELOPMENT_ROLE,
        enforce_expected_n=enforce_expected_n,
    )
    metadata = predictions.loc[:, stage3a1.MANIFEST_COLUMNS]
    stage3a1.validate_manifest_frame(
        metadata,
        DECISION_DEVELOPMENT_ROLE,
        enforce_expected_n=enforce_expected_n,
    )
    expected_n = (
        EXPECTED_N[DECISION_DEVELOPMENT_ROLE]
        if enforce_expected_n
        else len(predictions)
    )
    stage3a1.validate_quantile_array(
        predictions.loc[:, ["q05", "q50", "q95"]].to_numpy(), expected_n
    )
    numeric_columns = (
        "true_L",
        "irradiance",
        "lower",
        "upper",
        "width",
        "raw_width",
    )
    numeric = predictions.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("CQR interval predictions must be finite")
    lower = numeric["lower"].to_numpy(dtype=np.float64)
    upper = numeric["upper"].to_numpy(dtype=np.float64)
    truth = numeric["true_L"].to_numpy(dtype=np.float64)
    width = numeric["width"].to_numpy(dtype=np.float64)
    if np.any(lower < 0.0) or np.any(upper > 1.0):
        raise ValueError("CQR interval bounds fall outside [0, 1]")
    if np.any(lower > upper):
        raise ValueError("CQR interval lower bound exceeds upper bound")
    if np.any(width < 0.0):
        raise ValueError("CQR interval width must be non-negative")
    expected_width = upper - lower
    if not np.allclose(
        width,
        expected_width,
        rtol=0.0,
        atol=INTERVAL_CONSISTENCY_ABS_TOL,
    ):
        raise ValueError("CQR interval width is inconsistent with upper-lower")
    expected_covered = (truth >= lower) & (truth <= upper)
    if not np.array_equal(
        predictions["covered"].astype(bool).to_numpy(), expected_covered
    ):
        raise ValueError("CQR covered field is inconsistent with inclusive bounds")
    expected_raw_width = (
        predictions["q95"].to_numpy(dtype=np.float64)
        - predictions["q05"].to_numpy(dtype=np.float64)
    )
    if not np.allclose(
        numeric["raw_width"].to_numpy(dtype=np.float64),
        expected_raw_width,
        rtol=0.0,
        atol=INTERVAL_CONSISTENCY_ABS_TOL,
    ):
        raise ValueError("raw_width is inconsistent with q95-q05")
    if set(predictions["method"].astype(str)) != {METHOD}:
        raise ValueError(f"CQR interval method must be {METHOD}")
    return predictions.loc[:, PREDICTION_OUTPUT_COLUMNS].copy()


def conformalize_decision_intervals(
    decision: pd.DataFrame,
    calibration: FrozenCQRCalibration,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    if not isinstance(calibration, FrozenCQRCalibration):
        raise TypeError("A frozen CP calibration object is required")
    if calibration.calibration_role != CP_CALIBRATION_ROLE:
        raise PermissionError("qhat must be calibrated on CP_CALIBRATION")
    if not math.isfinite(calibration.qhat):
        raise ValueError("CQR qhat must be finite")
    _validate_expected_role_n(
        decision,
        DECISION_DEVELOPMENT_ROLE,
        enforce_expected_n=enforce_expected_n,
    )
    expected_n = (
        EXPECTED_N[DECISION_DEVELOPMENT_ROLE]
        if enforce_expected_n
        else len(decision)
    )
    stage3a1.validate_quantile_array(
        decision.loc[:, ["q05", "q50", "q95"]].to_numpy(), expected_n
    )
    raw_lower = decision["q05"].to_numpy(dtype=np.float64)
    raw_upper = decision["q95"].to_numpy(dtype=np.float64)
    lower_unclipped = raw_lower - calibration.qhat
    upper_unclipped = raw_upper + calibration.qhat
    lower = np.clip(lower_unclipped, 0.0, 1.0)
    upper = np.clip(upper_unclipped, 0.0, 1.0)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Conformalized interval bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("Conformalized lower exceeds upper after clipping")
    truth = decision["true_L"].to_numpy(dtype=np.float64)
    result = decision.loc[:, ALIGNED_COLUMNS].copy()
    result["method"] = METHOD
    result["lower"] = lower
    result["upper"] = upper
    result["width"] = upper - lower
    result["covered"] = (truth >= lower) & (truth <= upper)
    result["raw_width"] = raw_upper - raw_lower
    result["lower_clipped"] = lower != lower_unclipped
    result["upper_clipped"] = upper != upper_unclipped
    return validate_cqr_predictions(
        result.loc[:, PREDICTION_OUTPUT_COLUMNS],
        enforce_expected_n=enforce_expected_n,
    )


def compute_global_metrics(
    predictions: pd.DataFrame,
    *,
    enforce_expected_n: bool = True,
) -> dict[str, Any]:
    validated = validate_cqr_predictions(
        predictions, enforce_expected_n=enforce_expected_n
    )
    metrics = stage1b.compute_interval_metrics(validated)
    if tuple(metrics) != GLOBAL_METRIC_COLUMNS:
        raise ValueError("Stage 1B global metric schema changed unexpectedly")
    expected_n = (
        EXPECTED_N[DECISION_DEVELOPMENT_ROLE]
        if enforce_expected_n
        else len(validated)
    )
    if int(metrics["N"]) != expected_n:
        raise ValueError("CQR global metric N guard failed")
    return metrics


def raw_q05_q95_diagnostics(decision: pd.DataFrame) -> dict[str, float]:
    truth = decision["true_L"].to_numpy(dtype=np.float64)
    lower = decision["q05"].to_numpy(dtype=np.float64)
    upper = decision["q95"].to_numpy(dtype=np.float64)
    width = upper - lower
    covered = (truth >= lower) & (truth <= upper)
    return {
        "raw_q05_q95_PICP": float(covered.mean()),
        "raw_q05_q95_MPIW": float(width.mean()),
        "raw_q05_q95_median_width": float(np.median(width)),
    }


def interval_diagnostics(
    predictions: pd.DataFrame, calibration: FrozenCQRCalibration
) -> dict[str, Any]:
    if calibration.qhat < 0.0:
        direction = "contraction"
    elif calibration.qhat > 0.0:
        direction = "expansion"
    else:
        direction = "unchanged"
    return {
        "qhat": calibration.qhat,
        "qhat_is_negative": calibration.qhat_is_negative,
        "qhat_is_positive": calibration.qhat_is_positive,
        "interval_correction_direction": direction,
        "lower_clipped_count": int(predictions["lower_clipped"].astype(bool).sum()),
        "upper_clipped_count": int(predictions["upper_clipped"].astype(bool).sum()),
    }


def _conditional_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"N": 0, "PICP": None, "MPIW": None}
    return {
        "N": int(len(frame)),
        "PICP": float(frame["covered"].astype(bool).mean()),
        "MPIW": float(frame["width"].mean()),
    }


def conditional_coverage_diagnostics(
    predictions: pd.DataFrame,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    validated = validate_cqr_predictions(
        predictions, enforce_expected_n=enforce_expected_n
    )
    records: list[dict[str, Any]] = []
    variable_specs = (
        ("pred_L=q50", "q50", stage1b.PRED_L_BINS),
        ("irradiance", "irradiance", stage1b.IRRADIANCE_BINS),
    )
    for variable_name, column, bins in variable_specs:
        labels = stage1b.assign_fixed_bins(validated[column], bins)
        label_values = labels.astype(object)
        for label in stage1b.FIXED_BIN_LABELS:
            subset = validated.loc[np.asarray(label_values == label, dtype=bool)]
            records.append(
                {
                    "method": METHOD,
                    "evaluation_role": DECISION_DEVELOPMENT_ROLE,
                    "binning_variable": variable_name,
                    "bin_label": label,
                    **_conditional_metrics(subset),
                }
            )
        invalid = validated.loc[np.asarray(labels.isna(), dtype=bool)]
        if len(invalid):
            records.append(
                {
                    "method": METHOD,
                    "evaluation_role": DECISION_DEVELOPMENT_ROLE,
                    "binning_variable": variable_name,
                    "bin_label": "OUT_OF_RANGE_OR_NAN",
                    **_conditional_metrics(invalid),
                }
            )
    result = pd.DataFrame.from_records(records, columns=CONDITIONAL_COLUMNS)
    for variable_name, _, _ in variable_specs:
        axis = result.loc[result["binning_variable"] == variable_name]
        if int(axis["N"].sum()) != len(validated):
            raise ValueError(f"Conditional bin counts do not sum to N for {variable_name}")
        observed_fixed = tuple(
            axis.loc[axis["bin_label"] != "OUT_OF_RANGE_OR_NAN", "bin_label"]
        )
        if observed_fixed != stage1b.FIXED_BIN_LABELS:
            raise ValueError("Conditional fixed bin order changed")
    return result


def validate_stage1b_metrics_path(path: Path) -> Path:
    candidate = _resolved(path)
    if candidate != _resolved(STAGE1B_GLOBAL_METRICS_INPUT):
        raise PermissionError(f"Unauthorized Stage 1B metrics input: {candidate}")
    return candidate


def validate_stage1b_global_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(GLOBAL_METRIC_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Stage 1B global metrics missing fields: {sorted(missing)}")
    baseline_order = BASELINE_METHOD_ORDER[:-1]
    methods = frame["method"].astype(str)
    if methods.duplicated().any() or set(methods) != set(baseline_order):
        raise ValueError("Stage 1B baseline method set/order source is invalid")
    indexed = frame.set_index("method", drop=False)
    ordered = indexed.loc[list(baseline_order), list(GLOBAL_METRIC_COLUMNS)].reset_index(
        drop=True
    )
    numeric_columns = tuple(
        column for column in GLOBAL_METRIC_COLUMNS if column not in {"method", "evaluation_role"}
    )
    numeric = ordered.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Stage 1B global metrics must be finite")
    if set(ordered["evaluation_role"].astype(str)) != {DECISION_DEVELOPMENT_ROLE}:
        raise ValueError("Stage 1B global metrics evaluation role mismatch")
    if set(ordered["N"].astype(int)) != {EXPECTED_N[DECISION_DEVELOPMENT_ROLE]}:
        raise ValueError("Stage 1B global metrics N mismatch")
    if not np.allclose(ordered["alpha"].astype(float), ALPHA, rtol=0.0, atol=0.0):
        raise ValueError("Stage 1B alpha mismatch")
    if not np.allclose(
        ordered["target_coverage"].astype(float),
        TARGET_COVERAGE,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Stage 1B target coverage mismatch")
    return ordered


def load_stage1b_global_metrics(
    path: Path = STAGE1B_GLOBAL_METRICS_INPUT,
) -> pd.DataFrame:
    authorized = validate_stage1b_metrics_path(path)
    return validate_stage1b_global_metrics(pd.read_csv(authorized))


def build_interval_comparison(
    stage1b_metrics: pd.DataFrame, cqr_global_metrics: Mapping[str, Any]
) -> pd.DataFrame:
    baselines = validate_stage1b_global_metrics(stage1b_metrics)
    missing_cqr = set(GLOBAL_METRIC_COLUMNS) - set(cqr_global_metrics)
    if missing_cqr:
        raise ValueError(f"CQR global metrics missing fields: {sorted(missing_cqr)}")
    if cqr_global_metrics["method"] != METHOD:
        raise ValueError("CQR comparison method must be cqr_v1")
    baseline_comparison = baselines.loc[:, COMPARISON_COLUMNS]
    cqr_row = pd.DataFrame(
        [{column: cqr_global_metrics[column] for column in COMPARISON_COLUMNS}]
    )
    result = pd.concat([baseline_comparison, cqr_row], ignore_index=True)
    if tuple(result.columns) != COMPARISON_COLUMNS:
        raise ValueError("Interval comparison schema mismatch")
    if tuple(result["method"].astype(str)) != BASELINE_METHOD_ORDER:
        raise ValueError("Interval comparison fixed method order changed")
    if FORBIDDEN_COMPARISON_COLUMNS & {str(column).lower() for column in result.columns}:
        raise ValueError("Ranking/winner fields are forbidden in interval comparison")
    return result


def calibration_payload(
    calibration: FrozenCQRCalibration,
    raw_diagnostics: Mapping[str, Any],
    conformal_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **asdict(calibration),
        "finite_sample_quantile_rule": FINITE_SAMPLE_RULE,
        "raw_q05_q95_diagnostics": dict(raw_diagnostics),
        "conformal_interval_diagnostics": dict(conformal_diagnostics),
    }


def make_config() -> dict[str, Any]:
    validate_protocol_constants()
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "source_stage3a1_dir": project_relative(STAGE3A1_INPUT_DIR),
        "cp_predictions_input": project_relative(CP_PREDICTIONS_INPUT),
        "decision_predictions_input": project_relative(DECISION_PREDICTIONS_INPUT),
        "stage1b_global_metrics_input": project_relative(
            STAGE1B_GLOBAL_METRICS_INPUT
        ),
        "alpha": ALPHA,
        "target_coverage": TARGET_COVERAGE,
        "quantile_levels": list(QUANTILE_LEVELS),
        "cqr_score_definition": CQR_SCORE_DEFINITION,
        "finite_sample_quantile_rule": FINITE_SAMPLE_RULE,
        "quantile_method": QUANTILE_METHOD,
        "calibration_role": CP_CALIBRATION_ROLE,
        "evaluation_role": DECISION_DEVELOPMENT_ROLE,
        "expected_N": dict(EXPECTED_N),
        "prediction_output_columns": list(PREDICTION_OUTPUT_COLUMNS),
        "global_metric_columns": list(GLOBAL_METRIC_COLUMNS),
        "conditional_columns": list(CONDITIONAL_COLUMNS),
        "conditional_axes": ["pred_L=q50", "irradiance"],
        "conditional_bins": list(stage1b.PRED_L_BINS),
        "conditional_bin_labels": list(stage1b.FIXED_BIN_LABELS),
        "conditional_bin_semantics": "pd.cut(include_lowest=True,right=True)",
        "comparison_method_order": list(BASELINE_METHOD_ORDER),
        "comparison_columns": list(COMPARISON_COLUMNS),
    }


def make_provenance(
    calibration: FrozenCQRCalibration,
    conformal_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "source_stage3a1_dir": project_relative(STAGE3A1_INPUT_DIR),
        "source_cqr_checkpoint_sha256": SOURCE_CQR_CHECKPOINT_SHA256,
        "alpha": ALPHA,
        "quantile_levels": list(QUANTILE_LEVELS),
        "cqr_score_definition": CQR_SCORE_DEFINITION,
        "finite_sample_quantile_rule": FINITE_SAMPLE_RULE,
        "quantile_method": QUANTILE_METHOD,
        "calibration_role": CP_CALIBRATION_ROLE,
        "evaluation_role": DECISION_DEVELOPMENT_ROLE,
        "qhat": calibration.qhat,
        "qhat_is_negative": calibration.qhat_is_negative,
        "qhat_is_positive": calibration.qhat_is_positive,
        "conformal_interval_diagnostics": dict(conformal_diagnostics),
        "qhat_selected_using_decision_truth": False,
        "decision_truth_used_only_for_evaluation": True,
        "stage1b_metric_definitions_modified": False,
        "stage1b_bins_modified": False,
        "random_test_accessed": False,
        "sealed_final_dates_accessed": False,
        "training_performed": False,
        "image_inference_performed": False,
        "mc_dropout_performed": False,
        "risk_evaluation_performed": False,
        "cleaning_decision_performed": False,
        "economic_decision_performed": False,
        "formal_cqr_superiority_declared": False,
        "formal_uq_method_selected": False,
    }


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def validate_formal_output_path(output_dir: Path) -> Path:
    candidate = _resolved(output_dir)
    if candidate != _resolved(OUTPUT_DIR):
        raise PermissionError(f"Unauthorized Stage 3A2 output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_stage3a2_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    global_metrics: pd.DataFrame,
    conditional_coverage: pd.DataFrame,
    calibration: Mapping[str, Any],
    comparison: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "cqr_predictions.csv", index=False, mode="x")
    global_metrics.to_csv(
        output_dir / "cqr_global_metrics.csv", index=False, mode="x"
    )
    conditional_coverage.to_csv(
        output_dir / "cqr_conditional_coverage.csv", index=False, mode="x"
    )
    _write_json_exclusive(
        output_dir / "cqr_conformal_calibration.json", calibration
    )
    comparison.to_csv(
        output_dir / "interval_comparison_with_stage1b.csv", index=False, mode="x"
    )
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "provenance.json", provenance)


def run(
    protocol: str = PROTOCOL,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the formal Stage 3A2 protocol only when explicitly invoked."""
    validate_protocol(protocol)
    output_dir = validate_formal_output_path(output_dir)
    ensure_output_available(output_dir)

    # Calibration phase: CP truth is attached solely to select and freeze qhat.
    cp_predictions = load_stage3a1_predictions(
        CP_PREDICTIONS_INPUT, CP_CALIBRATION_ROLE
    )
    cp_truth_manifest = load_truth_manifest(
        AUTHORIZED_TRUTH_MANIFESTS[CP_CALIBRATION_ROLE], CP_CALIBRATION_ROLE
    )
    cp_aligned = attach_truth_by_sample_id(
        cp_predictions, cp_truth_manifest, CP_CALIBRATION_ROLE
    )
    frozen_calibration = calibrate_cqr(cp_aligned)

    # Evaluation phase starts only after FrozenCQRCalibration exists.  Decision
    # truth cannot flow into calibrate_cqr because that function accepts CP only.
    decision_predictions = load_stage3a1_predictions(
        DECISION_PREDICTIONS_INPUT, DECISION_DEVELOPMENT_ROLE
    )
    decision_truth_manifest = load_truth_manifest(
        AUTHORIZED_TRUTH_MANIFESTS[DECISION_DEVELOPMENT_ROLE],
        DECISION_DEVELOPMENT_ROLE,
    )
    stage3a1.validate_role_isolation(cp_predictions, decision_predictions)
    stage3a1.validate_role_isolation(cp_truth_manifest, decision_truth_manifest)
    decision_aligned = attach_truth_by_sample_id(
        decision_predictions,
        decision_truth_manifest,
        DECISION_DEVELOPMENT_ROLE,
    )
    validate_cp_decision_isolation(cp_aligned, decision_aligned)

    predictions = conformalize_decision_intervals(
        decision_aligned, frozen_calibration
    )
    global_record = compute_global_metrics(predictions)
    global_metrics = pd.DataFrame([global_record], columns=GLOBAL_METRIC_COLUMNS)
    conditional = conditional_coverage_diagnostics(predictions)
    raw_diagnostics = raw_q05_q95_diagnostics(decision_aligned)
    conformal_diagnostics = interval_diagnostics(predictions, frozen_calibration)
    calibration_json = calibration_payload(
        frozen_calibration, raw_diagnostics, conformal_diagnostics
    )
    stage1b_metrics = load_stage1b_global_metrics()
    comparison = build_interval_comparison(stage1b_metrics, global_record)
    config = make_config()
    provenance = make_provenance(frozen_calibration, conformal_diagnostics)
    write_stage3a2_outputs(
        output_dir,
        predictions,
        global_metrics,
        conditional,
        calibration_json,
        comparison,
        config,
        provenance,
    )
    return {
        "calibration": calibration_json,
        "global_metrics": global_record,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(protocol=args.protocol, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
