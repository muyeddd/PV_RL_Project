"""Paper1 Clean UQ Stage 1B: six interval baselines and interval metrics.

The formal entry point reads only the two sealed Stage 1A prediction tables.
It performs table-only interval construction: no image/model inference, MC
Dropout, training, optimization, risk ranking, CQR, or cleaning decisions.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"

ALPHA = 0.10
TARGET_COVERAGE = 0.90
MC_Z_90 = 1.645
PRED_L_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
IRRADIANCE_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
FIXED_BIN_LABELS = (
    "[0.0,0.2]",
    "(0.2,0.4]",
    "(0.4,0.6]",
    "(0.6,0.8]",
    "(0.8,1.0]",
)
MIN_CALIB_PER_BIN = 30
STD_MC_EPSILON = 1e-8

CP_CALIBRATION_ROLE = "CP_CALIBRATION"
DECISION_DEVELOPMENT_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_N = {
    CP_CALIBRATION_ROLE: 2951,
    DECISION_DEVELOPMENT_ROLE: 1844,
}
SEALED_FINAL_DATES = frozenset({"2017-06-15", "2017-06-24", "2017-06-30"})

STAGE1A_INPUT_DIR = (
    PROJECT_ROOT / "outputs" / PROTOCOL / "uq_stage1a_inference_v1"
)
CP_CALIBRATION_INPUT = STAGE1A_INPUT_DIR / "cp_calibration_predictions.csv"
DECISION_DEVELOPMENT_INPUT = (
    STAGE1A_INPUT_DIR / "decision_development_predictions.csv"
)
AUTHORIZED_INPUTS = {
    CP_CALIBRATION_ROLE: CP_CALIBRATION_INPUT,
    DECISION_DEVELOPMENT_ROLE: DECISION_DEVELOPMENT_INPUT,
}
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / "uq_stage1b_intervals_v1"

STABLE_STAGE1A_COLUMNS = (
    "sample_id",
    "date",
    "timestamp",
    "image_path",
    "role",
    "true_L",
    "irradiance",
    "point_pred",
    "mc_mean",
    "mc_std",
)
NUMERIC_STAGE1A_COLUMNS = (
    "true_L",
    "irradiance",
    "point_pred",
    "mc_mean",
    "mc_std",
)
COMMON_INTERVAL_COLUMNS = STABLE_STAGE1A_COLUMNS + (
    "method",
    "lower",
    "upper",
    "width",
    "covered",
)
MONDRIAN_INTERVAL_COLUMNS = COMMON_INTERVAL_COLUMNS + (
    "bin_label",
    "q_used",
    "used_global_fallback",
)

RAW_MC = "raw_mc"
SPLIT_CP = "split_cp"
IRRADIANCE_MONDRIAN = "irradiance_mondrian_cp"
PRED_L_MONDRIAN = "pred_l_mondrian_cp"
PRED_L_MC_INTERVAL = "pred_l_mondrian_mc_interval_cp"
PRED_L_STD_MC = "pred_l_mondrian_std_mc_cp"
METHOD_OUTPUTS = {
    RAW_MC: "raw_mc_predictions.csv",
    SPLIT_CP: "split_cp_predictions.csv",
    IRRADIANCE_MONDRIAN: "irradiance_mondrian_predictions.csv",
    PRED_L_MONDRIAN: "pred_l_mondrian_predictions.csv",
    PRED_L_MC_INTERVAL: "pred_l_mc_interval_predictions.csv",
    PRED_L_STD_MC: "pred_l_std_mc_predictions.csv",
}


def validate_protocol(protocol: str) -> None:
    if protocol != PROTOCOL:
        raise PermissionError(f"Unauthorized protocol: {protocol!r}")


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_stage1a_input_path(path: Path, expected_role: str) -> Path:
    candidate = _resolved(path)
    lowered = str(candidate).lower()
    if "random_test" in lowered:
        raise PermissionError("RANDOM_TEST input is forbidden")
    if expected_role not in AUTHORIZED_INPUTS:
        raise PermissionError(f"Forbidden Stage 1A role: {expected_role}")
    authorized = _resolved(AUTHORIZED_INPUTS[expected_role])
    if candidate != authorized:
        raise PermissionError(
            f"Only the sealed Stage 1A {expected_role} table is authorized: {candidate}"
        )
    return candidate


def validate_expected_n(frame: pd.DataFrame, expected_role: str) -> None:
    expected = EXPECTED_N.get(expected_role)
    if expected is None:
        raise PermissionError(f"Forbidden role for N guard: {expected_role}")
    if len(frame) != expected:
        raise ValueError(
            f"{expected_role} N guard failed: expected {expected}, got {len(frame)}"
        )


def validate_stage1a_frame(
    frame: pd.DataFrame, expected_role: str, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    missing = set(STABLE_STAGE1A_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Stage 1A schema missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{expected_role} Stage 1A table is empty")
    if set(frame["role"].astype(str)) != {expected_role}:
        raise PermissionError(f"Role guard failed for {expected_role}")
    if expected_role not in AUTHORIZED_INPUTS:
        raise PermissionError(f"Forbidden Stage 1A role: {expected_role}")
    if enforce_expected_n:
        validate_expected_n(frame, expected_role)

    normalized_dates = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    sealed = set(normalized_dates) & SEALED_FINAL_DATES
    if sealed:
        raise PermissionError(f"Sealed final date rejected: {sorted(sealed)}")

    locators = frame["image_path"].astype(str)
    if locators.str.lower().str.contains("random_test", regex=False).any():
        raise PermissionError("RANDOM_TEST locator rejected")
    for sealed_date in SEALED_FINAL_DATES:
        if locators.str.contains(sealed_date, regex=False).any():
            raise PermissionError(f"Sealed final date locator rejected: {sealed_date}")

    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-null and unique")
    if frame["image_path"].isna().any() or frame["image_path"].duplicated().any():
        raise ValueError("image_path must be non-null and unique")
    numeric = frame.loc[:, NUMERIC_STAGE1A_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Stage 1A numeric columns must all be finite")
    if (numeric["mc_std"] < 0).any():
        raise ValueError("mc_std must be non-negative")
    return frame.loc[:, STABLE_STAGE1A_COLUMNS].copy()


def load_stage1a_frame(path: Path, expected_role: str) -> pd.DataFrame:
    authorized = validate_stage1a_input_path(path, expected_role)
    return validate_stage1a_frame(pd.read_csv(authorized), expected_role)


def validate_role_isolation(cp: pd.DataFrame, decision: pd.DataFrame) -> None:
    if set(cp["role"]) != {CP_CALIBRATION_ROLE}:
        raise PermissionError("Calibration table must contain CP_CALIBRATION only")
    if set(decision["role"]) != {DECISION_DEVELOPMENT_ROLE}:
        raise PermissionError(
            "Primary evaluation table must contain DECISION_DEVELOPMENT only"
        )
    if set(cp["sample_id"]) & set(decision["sample_id"]):
        raise ValueError("CP_CALIBRATION/DECISION_DEVELOPMENT sample_id overlap")
    if set(cp["image_path"]) & set(decision["image_path"]):
        raise ValueError("CP_CALIBRATION/DECISION_DEVELOPMENT image_path overlap")


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def conformal_quantile(scores: Sequence[float] | np.ndarray) -> float:
    """Finite-sample split-conformal quantile with the fixed Stage 1B alpha."""
    values = _finite_vector(scores, "calibration scores")
    n = values.size
    rank_level = min(math.ceil((n + 1) * (1.0 - ALPHA)) / n, 1.0)
    return float(np.quantile(values, rank_level, method="higher"))


def absolute_residual_scores(frame: pd.DataFrame) -> np.ndarray:
    return np.abs(
        frame["true_L"].to_numpy(dtype=np.float64)
        - frame["mc_mean"].to_numpy(dtype=np.float64)
    )


def raw_mc_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    center = frame["mc_mean"].to_numpy(dtype=np.float64)
    spread = frame["mc_std"].to_numpy(dtype=np.float64)
    return (
        np.clip(center - MC_Z_90 * spread, 0.0, 1.0),
        np.clip(center + MC_Z_90 * spread, 0.0, 1.0),
    )


def mc_interval_nonconformity_scores(frame: pd.DataFrame) -> np.ndarray:
    lower, upper = raw_mc_bounds(frame)
    truth = frame["true_L"].to_numpy(dtype=np.float64)
    return np.maximum.reduce([lower - truth, truth - upper, np.zeros(len(frame))])


def std_mc_nonconformity_scores(frame: pd.DataFrame) -> np.ndarray:
    return absolute_residual_scores(frame) / (
        frame["mc_std"].to_numpy(dtype=np.float64) + STD_MC_EPSILON
    )


def assign_fixed_bins(
    values: Sequence[float] | pd.Series | np.ndarray,
    bins: Sequence[float],
) -> pd.Series:
    fixed = tuple(float(value) for value in bins)
    if fixed not in {PRED_L_BINS, IRRADIANCE_BINS}:
        raise ValueError("Only the fixed Stage 1B bins are authorized")
    numeric = pd.to_numeric(pd.Series(values, copy=False), errors="coerce")
    return pd.cut(
        numeric,
        bins=list(fixed),
        labels=list(FIXED_BIN_LABELS),
        include_lowest=True,
        right=True,
    )


def calibrate_mondrian_quantiles(
    scores: Sequence[float] | np.ndarray,
    calibration_values: Sequence[float] | pd.Series | np.ndarray,
    *,
    binning_variable: str,
    bins: Sequence[float],
) -> dict[str, Any]:
    score_values = _finite_vector(scores, "calibration scores")
    if len(calibration_values) != len(score_values):
        raise ValueError("Calibration values and scores must have equal length")
    labels = assign_fixed_bins(calibration_values, bins)
    global_q = conformal_quantile(score_values)
    counts: dict[str, int] = {}
    q_by_bin: dict[str, float | None] = {}
    for label in FIXED_BIN_LABELS:
        mask = np.asarray(labels == label, dtype=bool)
        count = int(mask.sum())
        counts[label] = count
        q_by_bin[label] = (
            conformal_quantile(score_values[mask])
            if count >= MIN_CALIB_PER_BIN
            else None
        )
    return {
        "conformal_calibration": True,
        "alpha": ALPHA,
        "finite_sample_correction": "min(ceil((n+1)*(1-alpha))/n,1.0)",
        "quantile_method": "higher",
        "score_n": int(len(score_values)),
        "global_q": global_q,
        "binning_variable": binning_variable,
        "bins": list(float(value) for value in bins),
        "bin_semantics": "pd.cut(include_lowest=True,right=True)",
        "bin_labels": list(FIXED_BIN_LABELS),
        "min_calib_per_bin": MIN_CALIB_PER_BIN,
        "calibration_bin_counts": counts,
        "q_by_bin": q_by_bin,
        "calibration_out_of_range_or_nan_count": int(labels.isna().sum()),
    }


def apply_mondrian_quantiles(
    values: Sequence[float] | pd.Series | np.ndarray,
    calibration: Mapping[str, Any],
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    labels = assign_fixed_bins(values, calibration["bins"])
    global_q = float(calibration["global_q"])
    q_by_bin = calibration["q_by_bin"]
    q_used = np.empty(len(labels), dtype=np.float64)
    fallback = np.empty(len(labels), dtype=bool)
    for index, label in enumerate(labels.astype(object).tolist()):
        bin_q = q_by_bin.get(label) if isinstance(label, str) else None
        if bin_q is None:
            q_used[index] = global_q
            fallback[index] = True
        else:
            q_used[index] = float(bin_q)
            fallback[index] = False
    return labels, q_used, fallback


def _finalize_predictions(
    frame: pd.DataFrame,
    method: str,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    *,
    bin_label: pd.Series | None = None,
    q_used: np.ndarray | None = None,
    used_global_fallback: np.ndarray | None = None,
) -> pd.DataFrame:
    lower_values = np.clip(_finite_vector(lower, "lower bounds"), 0.0, 1.0)
    upper_values = np.clip(_finite_vector(upper, "upper bounds"), 0.0, 1.0)
    if len(frame) != len(lower_values) or len(frame) != len(upper_values):
        raise ValueError("Interval bounds do not match prediction table length")
    if np.any(lower_values > upper_values):
        raise ValueError("Interval lower bound exceeds upper bound")
    result = frame.loc[:, STABLE_STAGE1A_COLUMNS].copy()
    truth = result["true_L"].to_numpy(dtype=np.float64)
    result["method"] = method
    result["lower"] = lower_values
    result["upper"] = upper_values
    result["width"] = upper_values - lower_values
    result["covered"] = (truth >= lower_values) & (truth <= upper_values)
    mondrian_fields = (bin_label, q_used, used_global_fallback)
    if any(value is not None for value in mondrian_fields):
        if any(value is None for value in mondrian_fields):
            raise ValueError("Mondrian audit fields must be supplied together")
        result["bin_label"] = pd.Series(bin_label).astype(object).where(
            pd.Series(bin_label).notna(), "OUT_OF_RANGE_OR_NAN"
        ).to_numpy()
        result["q_used"] = np.asarray(q_used, dtype=np.float64)
        result["used_global_fallback"] = np.asarray(
            used_global_fallback, dtype=bool
        )
        return result.loc[:, MONDRIAN_INTERVAL_COLUMNS]
    return result.loc[:, COMMON_INTERVAL_COLUMNS]


def _with_fallback_summary(
    calibration: Mapping[str, Any], predictions: pd.DataFrame
) -> dict[str, Any]:
    result = dict(calibration)
    fallback_count = int(predictions["used_global_fallback"].sum())
    result["decision_fallback_count"] = fallback_count
    result["decision_fallback_rate"] = fallback_count / len(predictions)
    return result


def raw_mc_intervals(decision: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    lower, upper = raw_mc_bounds(decision)
    predictions = _finalize_predictions(decision, RAW_MC, lower, upper)
    return predictions, {
        "conformal_calibration": False,
        "mc_z_90": MC_Z_90,
        "decision_fallback_count": 0,
        "decision_fallback_rate": 0.0,
    }


def split_cp_intervals(
    calibration: pd.DataFrame, decision: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = absolute_residual_scores(calibration)
    q_global = conformal_quantile(scores)
    center = decision["mc_mean"].to_numpy(dtype=np.float64)
    predictions = _finalize_predictions(
        decision, SPLIT_CP, center - q_global, center + q_global
    )
    return predictions, {
        "conformal_calibration": True,
        "alpha": ALPHA,
        "finite_sample_correction": "min(ceil((n+1)*(1-alpha))/n,1.0)",
        "quantile_method": "higher",
        "score": "abs(true_L-mc_mean)",
        "base_predictor": "mc_mean",
        "score_n": int(len(scores)),
        "global_q": q_global,
        "decision_fallback_count": 0,
        "decision_fallback_rate": 0.0,
    }


def _mondrian_symmetric_intervals(
    calibration: pd.DataFrame,
    decision: pd.DataFrame,
    *,
    method: str,
    calibration_values: pd.Series,
    decision_values: pd.Series,
    binning_variable: str,
    bins: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quantiles = calibrate_mondrian_quantiles(
        absolute_residual_scores(calibration),
        calibration_values,
        binning_variable=binning_variable,
        bins=bins,
    )
    labels, q_used, fallback = apply_mondrian_quantiles(decision_values, quantiles)
    center = decision["mc_mean"].to_numpy(dtype=np.float64)
    predictions = _finalize_predictions(
        decision,
        method,
        center - q_used,
        center + q_used,
        bin_label=labels,
        q_used=q_used,
        used_global_fallback=fallback,
    )
    return predictions, _with_fallback_summary(quantiles, predictions)


def irradiance_mondrian_intervals(
    calibration: pd.DataFrame, decision: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _mondrian_symmetric_intervals(
        calibration,
        decision,
        method=IRRADIANCE_MONDRIAN,
        calibration_values=calibration["irradiance"],
        decision_values=decision["irradiance"],
        binning_variable="irradiance",
        bins=IRRADIANCE_BINS,
    )


def pred_l_mondrian_intervals(
    calibration: pd.DataFrame, decision: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _mondrian_symmetric_intervals(
        calibration,
        decision,
        method=PRED_L_MONDRIAN,
        calibration_values=calibration["mc_mean"],
        decision_values=decision["mc_mean"],
        binning_variable="pred_L=mc_mean",
        bins=PRED_L_BINS,
    )


def pred_l_mc_interval_intervals(
    calibration: pd.DataFrame, decision: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quantiles = calibrate_mondrian_quantiles(
        mc_interval_nonconformity_scores(calibration),
        calibration["mc_mean"],
        binning_variable="pred_L=mc_mean",
        bins=PRED_L_BINS,
    )
    labels, q_used, fallback = apply_mondrian_quantiles(
        decision["mc_mean"], quantiles
    )
    mc_lower, mc_upper = raw_mc_bounds(decision)
    predictions = _finalize_predictions(
        decision,
        PRED_L_MC_INTERVAL,
        mc_lower - q_used,
        mc_upper + q_used,
        bin_label=labels,
        q_used=q_used,
        used_global_fallback=fallback,
    )
    quantiles = dict(quantiles)
    quantiles["score"] = "max(mc_lower-true_L,true_L-mc_upper,0)"
    return predictions, _with_fallback_summary(quantiles, predictions)


def pred_l_std_mc_intervals(
    calibration: pd.DataFrame, decision: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quantiles = calibrate_mondrian_quantiles(
        std_mc_nonconformity_scores(calibration),
        calibration["mc_mean"],
        binning_variable="pred_L=mc_mean",
        bins=PRED_L_BINS,
    )
    labels, q_used, fallback = apply_mondrian_quantiles(
        decision["mc_mean"], quantiles
    )
    center = decision["mc_mean"].to_numpy(dtype=np.float64)
    half_width = q_used * (
        decision["mc_std"].to_numpy(dtype=np.float64) + STD_MC_EPSILON
    )
    predictions = _finalize_predictions(
        decision,
        PRED_L_STD_MC,
        center - half_width,
        center + half_width,
        bin_label=labels,
        q_used=q_used,
        used_global_fallback=fallback,
    )
    quantiles = dict(quantiles)
    quantiles["score"] = "abs(true_L-mc_mean)/(mc_std+epsilon)"
    quantiles["epsilon"] = STD_MC_EPSILON
    return predictions, _with_fallback_summary(quantiles, predictions)


def standard_interval_score(
    truth: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> np.ndarray:
    y = _finite_vector(truth, "truth")
    lo = _finite_vector(lower, "lower")
    hi = _finite_vector(upper, "upper")
    if not (len(y) == len(lo) == len(hi)):
        raise ValueError("Interval score inputs must have equal length")
    if np.any(lo > hi):
        raise ValueError("Interval score lower bound exceeds upper bound")
    width = hi - lo
    return (
        width
        + (2.0 / ALPHA) * (lo - y) * (y < lo)
        + (2.0 / ALPHA) * (y - hi) * (y > hi)
    )


def compute_interval_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    truth = predictions["true_L"].to_numpy(dtype=np.float64)
    lower = predictions["lower"].to_numpy(dtype=np.float64)
    upper = predictions["upper"].to_numpy(dtype=np.float64)
    covered = (truth >= lower) & (truth <= upper)
    width = upper - lower
    picp = float(covered.mean())
    return {
        "method": str(predictions["method"].iloc[0]),
        "evaluation_role": DECISION_DEVELOPMENT_ROLE,
        "N": int(len(predictions)),
        "alpha": ALPHA,
        "target_coverage": TARGET_COVERAGE,
        "PICP": picp,
        "MPIW": float(width.mean()),
        "median_width": float(np.median(width)),
        "coverage_error": abs(picp - TARGET_COVERAGE),
        "mean_interval_score_alpha_0p10": float(
            standard_interval_score(truth, lower, upper).mean()
        ),
    }


def conditional_coverage_diagnostics(
    predictions_by_method: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    variable_specs = (
        ("pred_L=mc_mean", "mc_mean", PRED_L_BINS),
        ("irradiance", "irradiance", IRRADIANCE_BINS),
    )
    for method, predictions in predictions_by_method.items():
        for variable_name, column, bins in variable_specs:
            labels = assign_fixed_bins(predictions[column], bins)
            label_values = labels.astype(object)
            for label in FIXED_BIN_LABELS:
                subset = predictions.loc[np.asarray(label_values == label, dtype=bool)]
                metrics = _conditional_metrics(subset)
                records.append(
                    {
                        "method": method,
                        "evaluation_role": DECISION_DEVELOPMENT_ROLE,
                        "binning_variable": variable_name,
                        "bin_label": label,
                        **metrics,
                    }
                )
            invalid = predictions.loc[np.asarray(labels.isna(), dtype=bool)]
            if len(invalid):
                records.append(
                    {
                        "method": method,
                        "evaluation_role": DECISION_DEVELOPMENT_ROLE,
                        "binning_variable": variable_name,
                        "bin_label": "OUT_OF_RANGE_OR_NAN",
                        **_conditional_metrics(invalid),
                    }
                )
    return pd.DataFrame.from_records(records)


def _conditional_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"N": 0, "PICP": None, "MPIW": None}
    return {
        "N": int(len(frame)),
        "PICP": float(frame["covered"].astype(bool).mean()),
        "MPIW": float(frame["width"].mean()),
    }


def build_bin_diagnostics(
    quantiles: Mapping[str, Mapping[str, Any]],
    predictions_by_method: Mapping[str, pd.DataFrame],
    decision: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, info in quantiles.items():
        if not info.get("conformal_calibration"):
            continue
        if "q_by_bin" not in info:
            records.append(
                {
                    "diagnostic_type": "calibration_quantile",
                    "method": method,
                    "role": CP_CALIBRATION_ROLE,
                    "binning_variable": "GLOBAL",
                    "bin_label": "GLOBAL",
                    "N": int(info["score_n"]),
                    "q_global": float(info["global_q"]),
                    "q_bin": float(info["global_q"]),
                    "calibration_eligible": True,
                    "used_global_fallback": False,
                }
            )
            continue
        for label in FIXED_BIN_LABELS:
            q_bin = info["q_by_bin"][label]
            count = int(info["calibration_bin_counts"][label])
            records.append(
                {
                    "diagnostic_type": "calibration_quantile",
                    "method": method,
                    "role": CP_CALIBRATION_ROLE,
                    "binning_variable": info["binning_variable"],
                    "bin_label": label,
                    "N": count,
                    "q_global": float(info["global_q"]),
                    "q_bin": q_bin,
                    "calibration_eligible": count >= MIN_CALIB_PER_BIN,
                    "used_global_fallback": q_bin is None,
                }
            )

    for variable_name, column, bins in (
        ("pred_L=mc_mean", "mc_mean", PRED_L_BINS),
        ("irradiance", "irradiance", IRRADIANCE_BINS),
    ):
        labels = assign_fixed_bins(decision[column], bins)
        for label in FIXED_BIN_LABELS:
            records.append(
                {
                    "diagnostic_type": "decision_bin_count",
                    "method": "ALL_METHODS",
                    "role": DECISION_DEVELOPMENT_ROLE,
                    "binning_variable": variable_name,
                    "bin_label": label,
                    "N": int((labels == label).sum()),
                }
            )
        if labels.isna().any():
            records.append(
                {
                    "diagnostic_type": "decision_bin_count",
                    "method": "ALL_METHODS",
                    "role": DECISION_DEVELOPMENT_ROLE,
                    "binning_variable": variable_name,
                    "bin_label": "OUT_OF_RANGE_OR_NAN",
                    "N": int(labels.isna().sum()),
                }
            )

    for method, predictions in predictions_by_method.items():
        fallback_count = (
            int(predictions["used_global_fallback"].sum())
            if "used_global_fallback" in predictions
            else 0
        )
        records.append(
            {
                "diagnostic_type": "decision_fallback_summary",
                "method": method,
                "role": DECISION_DEVELOPMENT_ROLE,
                "N": int(len(predictions)),
                "fallback_count": fallback_count,
                "fallback_rate": fallback_count / len(predictions),
            }
        )
    return pd.DataFrame.from_records(records)


def run_all_methods(
    calibration: pd.DataFrame, decision: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    results = (
        (RAW_MC, raw_mc_intervals(decision)),
        (SPLIT_CP, split_cp_intervals(calibration, decision)),
        (
            IRRADIANCE_MONDRIAN,
            irradiance_mondrian_intervals(calibration, decision),
        ),
        (PRED_L_MONDRIAN, pred_l_mondrian_intervals(calibration, decision)),
        (
            PRED_L_MC_INTERVAL,
            pred_l_mc_interval_intervals(calibration, decision),
        ),
        (PRED_L_STD_MC, pred_l_std_mc_intervals(calibration, decision)),
    )
    predictions = {method: result[0] for method, result in results}
    quantiles = {method: result[1] for method, result in results}
    return predictions, quantiles


def make_config() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "stage1a_input_dir": project_relative(STAGE1A_INPUT_DIR),
        "alpha": ALPHA,
        "target_coverage": TARGET_COVERAGE,
        "mc_z_90": MC_Z_90,
        "pred_l_bins": list(PRED_L_BINS),
        "irradiance_bins": list(IRRADIANCE_BINS),
        "bin_semantics": "pd.cut(include_lowest=True,right=True)",
        "min_calib_per_bin": MIN_CALIB_PER_BIN,
        "std_mc_epsilon": STD_MC_EPSILON,
        "split_cp_base_predictor": "mc_mean",
        "pred_l_definition": "mc_mean",
        "conformal_quantile_finite_sample_correction": (
            "min(ceil((n+1)*(1-alpha))/n,1.0)"
        ),
        "conformal_quantile_method": "higher",
        "cp_calibration_role": CP_CALIBRATION_ROLE,
        "primary_evaluation_role": DECISION_DEVELOPMENT_ROLE,
        "methods": list(METHOD_OUTPUTS),
        "prediction_outputs": METHOD_OUTPUTS,
        "interval_metrics": [
            "PICP",
            "MPIW",
            "median_width",
            "coverage_error",
            "mean_interval_score_alpha_0p10",
        ],
    }


def make_provenance() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "stage1a_input_dir": project_relative(STAGE1A_INPUT_DIR),
        "alpha": ALPHA,
        "target_coverage": TARGET_COVERAGE,
        "mc_z_90": MC_Z_90,
        "pred_l_bins": list(PRED_L_BINS),
        "irradiance_bins": list(IRRADIANCE_BINS),
        "min_calib_per_bin": MIN_CALIB_PER_BIN,
        "std_mc_epsilon": STD_MC_EPSILON,
        "split_cp_base_predictor": "mc_mean",
        "pred_l_definition": "mc_mean",
        "cp_calibration_role": CP_CALIBRATION_ROLE,
        "primary_evaluation_role": DECISION_DEVELOPMENT_ROLE,
        "cp_calibration_n": EXPECTED_N[CP_CALIBRATION_ROLE],
        "decision_development_n": EXPECTED_N[DECISION_DEVELOPMENT_ROLE],
        "random_test_accessed": False,
        "random_test_truth_accessed": False,
        "random_test_predictions_generated": False,
        "sealed_final_dates_accessed": False,
        "legacy_outputs_used": False,
        "legacy_checkpoint_loaded": False,
        "image_inference_performed": False,
        "mc_dropout_performed": False,
        "training_performed": False,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "risk_screening_performed": False,
        "cqr_performed": False,
        "cleaning_decision_performed": False,
        "decision_truth_used_for_cp_quantile": False,
        "decision_truth_used_to_optimize_bins": False,
        "decision_truth_used_to_optimize_alpha": False,
        "decision_truth_used_to_optimize_epsilon": False,
    }


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def validate_formal_output_path(output_dir: Path) -> Path:
    candidate = _resolved(output_dir)
    if candidate != _resolved(OUTPUT_DIR):
        raise PermissionError(f"Unauthorized Stage 1B output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_stage1b_outputs(
    output_dir: Path,
    predictions_by_method: Mapping[str, pd.DataFrame],
    metrics: pd.DataFrame,
    quantiles: Mapping[str, Mapping[str, Any]],
    bin_diagnostics: pd.DataFrame,
    conditional_diagnostics: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    if set(predictions_by_method) != set(METHOD_OUTPUTS):
        raise ValueError("Exactly the six authorized Stage 1B methods are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "all_interval_metrics.csv", index=False, mode="x")
    for method, filename in METHOD_OUTPUTS.items():
        predictions_by_method[method].to_csv(
            output_dir / filename, index=False, mode="x"
        )
    _write_json_exclusive(output_dir / "conformal_quantiles.json", quantiles)
    bin_diagnostics.to_csv(output_dir / "bin_diagnostics.csv", index=False, mode="x")
    conditional_diagnostics.to_csv(
        output_dir / "conditional_coverage_diagnostics.csv", index=False, mode="x"
    )
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "provenance.json", provenance)


def run(
    protocol: str = PROTOCOL,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    validate_protocol(protocol)
    output_dir = validate_formal_output_path(output_dir)
    ensure_output_available(output_dir)
    calibration = load_stage1a_frame(CP_CALIBRATION_INPUT, CP_CALIBRATION_ROLE)
    decision = load_stage1a_frame(
        DECISION_DEVELOPMENT_INPUT, DECISION_DEVELOPMENT_ROLE
    )
    validate_role_isolation(calibration, decision)
    predictions, quantiles = run_all_methods(calibration, decision)
    metric_records = []
    for method in METHOD_OUTPUTS:
        record = compute_interval_metrics(predictions[method])
        record["fallback_count"] = int(
            quantiles[method]["decision_fallback_count"]
        )
        record["fallback_rate"] = float(
            quantiles[method]["decision_fallback_rate"]
        )
        metric_records.append(record)
    metrics = pd.DataFrame(metric_records)
    bin_diagnostics = build_bin_diagnostics(quantiles, predictions, decision)
    conditional_diagnostics = conditional_coverage_diagnostics(predictions)
    config = make_config()
    provenance = make_provenance()
    write_stage1b_outputs(
        output_dir,
        predictions,
        metrics,
        quantiles,
        bin_diagnostics,
        conditional_diagnostics,
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
