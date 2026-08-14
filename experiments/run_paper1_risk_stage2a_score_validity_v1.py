"""Paper1 Risk Stage 2A: descriptive risk-score validity diagnostics.

This table-only stage measures whether larger fixed uncertainty scores tend to
coincide with larger ``abs(true_L - mc_mean)`` on DECISION_DEVELOPMENT.  It
does not select a score, set a threshold, reject samples, run coverage curves,
or perform image/model inference, MC Dropout, training, CQR, or decisions.
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
PRIMARY_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_N = 1844
REQUESTED_QUANTILE_BINS = 10
RISK_TIE_ROUND_DECIMALS = 12
RISK_TIE_POLICY = (
    "round risk scores to 12 decimal places for tie-sensitive ranking and "
    "quantile operations only"
)
SEALED_FINAL_DATES = frozenset({"2017-06-15", "2017-06-24", "2017-06-30"})
RISK_TARGET_ERROR = "abs(true_L - mc_mean)"

STAGE1A_SOURCE = (
    PROJECT_ROOT
    / "outputs"
    / PROTOCOL
    / "uq_stage1a_inference_v1"
    / "decision_development_predictions.csv"
)
STAGE1B_SOURCE_DIR = (
    PROJECT_ROOT / "outputs" / PROTOCOL / "uq_stage1b_intervals_v1"
)

MC_STD = "mc_std"
RAW_MC_WIDTH = "raw_mc_width"
SPLIT_CP_WIDTH = "split_cp_width"
IRRADIANCE_MONDRIAN_WIDTH = "irradiance_mondrian_width"
PRED_L_MONDRIAN_WIDTH = "pred_l_mondrian_width"
PRED_L_MC_INTERVAL_WIDTH = "pred_l_mc_interval_width"
PRED_L_STD_MC_WIDTH = "pred_l_std_mc_width"
RISK_SCORE_ORDER = (
    MC_STD,
    RAW_MC_WIDTH,
    SPLIT_CP_WIDTH,
    IRRADIANCE_MONDRIAN_WIDTH,
    PRED_L_MONDRIAN_WIDTH,
    PRED_L_MC_INTERVAL_WIDTH,
    PRED_L_STD_MC_WIDTH,
)

STAGE1B_INPUT_SPECS = {
    RAW_MC_WIDTH: (
        STAGE1B_SOURCE_DIR / "raw_mc_predictions.csv",
        "raw_mc",
    ),
    SPLIT_CP_WIDTH: (
        STAGE1B_SOURCE_DIR / "split_cp_predictions.csv",
        "split_cp",
    ),
    IRRADIANCE_MONDRIAN_WIDTH: (
        STAGE1B_SOURCE_DIR / "irradiance_mondrian_predictions.csv",
        "irradiance_mondrian_cp",
    ),
    PRED_L_MONDRIAN_WIDTH: (
        STAGE1B_SOURCE_DIR / "pred_l_mondrian_predictions.csv",
        "pred_l_mondrian_cp",
    ),
    PRED_L_MC_INTERVAL_WIDTH: (
        STAGE1B_SOURCE_DIR / "pred_l_mc_interval_predictions.csv",
        "pred_l_mondrian_mc_interval_cp",
    ),
    PRED_L_STD_MC_WIDTH: (
        STAGE1B_SOURCE_DIR / "pred_l_std_mc_predictions.csv",
        "pred_l_mondrian_std_mc_cp",
    ),
}
AUTHORIZED_INPUTS = {
    "stage1a_decision": STAGE1A_SOURCE,
    **{risk_score: spec[0] for risk_score, spec in STAGE1B_INPUT_SPECS.items()},
}
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / PROTOCOL / "risk_stage2a_score_validity_v1"
)

STAGE1A_REQUIRED_COLUMNS = (
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
INTERVAL_REQUIRED_COLUMNS = STAGE1A_REQUIRED_COLUMNS + (
    "method",
    "lower",
    "upper",
    "width",
    "covered",
)
ALIGNMENT_COLUMNS = ("date", "true_L", "mc_mean")


def validate_protocol(protocol: str) -> None:
    if protocol != PROTOCOL:
        raise PermissionError(f"Unauthorized protocol: {protocol!r}")


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_authorized_input_path(path: Path, source_key: str) -> Path:
    candidate = _resolved(path)
    lowered = str(candidate).lower()
    if "random_test" in lowered:
        raise PermissionError("RANDOM_TEST input is forbidden")
    if "cp_calibration" in lowered:
        raise PermissionError("CP_CALIBRATION input is forbidden for Stage 2A")
    if source_key not in AUTHORIZED_INPUTS:
        raise PermissionError(f"Unauthorized Stage 2A source key: {source_key}")
    authorized = _resolved(AUTHORIZED_INPUTS[source_key])
    if candidate != authorized:
        raise PermissionError(f"Unauthorized Stage 2A input path: {candidate}")
    return candidate


def validate_expected_n(frame: pd.DataFrame) -> None:
    if len(frame) != EXPECTED_N:
        raise ValueError(
            f"{PRIMARY_ROLE} N guard failed: expected {EXPECTED_N}, got {len(frame)}"
        )


def _validate_common_decision_fields(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    enforce_expected_n: bool,
) -> None:
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Input schema missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Stage 2A input table is empty")
    if set(frame["role"].astype(str)) != {PRIMARY_ROLE}:
        raise PermissionError(f"Only {PRIMARY_ROLE} is authorized")
    if enforce_expected_n:
        validate_expected_n(frame)

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
    numeric_columns = ["true_L", "mc_mean", "mc_std"]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Core Stage 2A numeric values must be finite")
    if (numeric["mc_std"] < 0).any():
        raise ValueError("mc_std must be non-negative")


def validate_stage1a_base_frame(
    frame: pd.DataFrame, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    _validate_common_decision_fields(
        frame, STAGE1A_REQUIRED_COLUMNS, enforce_expected_n=enforce_expected_n
    )
    return frame.loc[:, STAGE1A_REQUIRED_COLUMNS].copy()


def validate_stage1b_interval_frame(
    frame: pd.DataFrame,
    expected_method: str,
    *,
    enforce_expected_n: bool = True,
) -> pd.DataFrame:
    _validate_common_decision_fields(
        frame, INTERVAL_REQUIRED_COLUMNS, enforce_expected_n=enforce_expected_n
    )
    if set(frame["method"].astype(str)) != {expected_method}:
        raise PermissionError(f"Stage 1B method guard failed for {expected_method}")
    bounds = frame.loc[:, ["lower", "upper", "width"]].apply(
        pd.to_numeric, errors="raise"
    )
    values = bounds.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Stage 1B interval bounds/width must be finite")
    if (bounds["lower"] < 0).any() or (bounds["upper"] > 1).any():
        raise ValueError("Stage 1B interval bounds must lie in [0,1]")
    if (bounds["lower"] > bounds["upper"]).any():
        raise ValueError("Stage 1B lower bound exceeds upper bound")
    reconstructed_width = bounds["upper"] - bounds["lower"]
    if not np.allclose(
        bounds["width"], reconstructed_width, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Stage 1B width is inconsistent with upper-lower")
    return frame.loc[:, INTERVAL_REQUIRED_COLUMNS].copy()


def load_stage1a_base() -> pd.DataFrame:
    path = validate_authorized_input_path(STAGE1A_SOURCE, "stage1a_decision")
    return validate_stage1a_base_frame(pd.read_csv(path))


def load_stage1b_intervals(risk_score: str) -> pd.DataFrame:
    if risk_score not in STAGE1B_INPUT_SPECS:
        raise PermissionError(f"Unauthorized interval risk score: {risk_score}")
    path, expected_method = STAGE1B_INPUT_SPECS[risk_score]
    authorized = validate_authorized_input_path(path, risk_score)
    return validate_stage1b_interval_frame(
        pd.read_csv(authorized), expected_method
    )


def align_interval_to_base(
    base: pd.DataFrame,
    interval: pd.DataFrame,
    expected_method: str,
) -> pd.DataFrame:
    if len(interval) != len(base):
        raise ValueError(
            f"Aligned N mismatch: base={len(base)}, interval={len(interval)}"
        )
    if base["sample_id"].duplicated().any() or interval["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique before alignment")
    base_ids = base["sample_id"].tolist()
    if set(base_ids) != set(interval["sample_id"]):
        missing = set(base_ids) - set(interval["sample_id"])
        extra = set(interval["sample_id"]) - set(base_ids)
        raise ValueError(
            "sample_id set mismatch: "
            f"missing={sorted(missing, key=str)}, extra={sorted(extra, key=str)}"
        )
    if set(interval["method"].astype(str)) != {expected_method}:
        raise PermissionError(f"Stage 1B method mismatch for {expected_method}")

    aligned = interval.set_index("sample_id", drop=False).loc[base_ids].reset_index(drop=True)
    base_ordered = base.reset_index(drop=True)
    if not np.array_equal(
        base_ordered["date"].astype(str).to_numpy(),
        aligned["date"].astype(str).to_numpy(),
    ):
        raise ValueError("Aligned date mismatch")
    for column in ("true_L", "mc_mean"):
        if not np.array_equal(
            base_ordered[column].to_numpy(dtype=np.float64),
            aligned[column].to_numpy(dtype=np.float64),
        ):
            raise ValueError(f"Aligned {column} mismatch")
    return aligned


def build_aligned_risk_table(
    base: pd.DataFrame,
    interval_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    expected_interval_scores = set(RISK_SCORE_ORDER) - {MC_STD}
    if set(interval_tables) != expected_interval_scores:
        raise ValueError("Exactly the six fixed Stage 1B interval tables are required")
    result = base.loc[:, ["sample_id", "date", "true_L", "mc_mean"]].copy()
    result["abs_error_mc_mean"] = np.abs(
        base["true_L"].to_numpy(dtype=np.float64)
        - base["mc_mean"].to_numpy(dtype=np.float64)
    )
    result[MC_STD] = base["mc_std"].to_numpy(dtype=np.float64)
    for risk_score in RISK_SCORE_ORDER[1:]:
        expected_method = STAGE1B_INPUT_SPECS[risk_score][1]
        aligned = align_interval_to_base(
            base, interval_tables[risk_score], expected_method
        )
        result[risk_score] = (
            aligned["upper"].to_numpy(dtype=np.float64)
            - aligned["lower"].to_numpy(dtype=np.float64)
        )
    return result


def average_ranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = _finite_vector(values, "rank values")
    return pd.Series(vector).rank(method="average").to_numpy(dtype=np.float64)


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def risk_rank_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the rounded view used only by tie-sensitive risk operations."""
    raw_values = _finite_vector(values, "risk score")
    return np.round(raw_values, decimals=RISK_TIE_ROUND_DECIMALS)


def spearman_with_average_ties(
    risk: Sequence[float] | np.ndarray,
    abs_error: Sequence[float] | np.ndarray,
) -> tuple[float, bool, int]:
    risk_values = _finite_vector(risk, "risk score")
    error_values = _finite_vector(abs_error, "absolute error")
    if len(risk_values) != len(error_values):
        raise ValueError("Risk and absolute error lengths differ")
    ranked_risk_values = risk_rank_values(risk_values)
    n_unique_risk = int(np.unique(ranked_risk_values).size)
    constant_risk = n_unique_risk == 1
    if constant_risk:
        return float("nan"), True, n_unique_risk
    risk_ranks = average_ranks(ranked_risk_values)
    error_ranks = average_ranks(error_values)
    risk_centered = risk_ranks - risk_ranks.mean()
    error_centered = error_ranks - error_ranks.mean()
    denominator = math.sqrt(
        float(np.dot(risk_centered, risk_centered))
        * float(np.dot(error_centered, error_centered))
    )
    rho = (
        float(np.dot(risk_centered, error_centered) / denominator)
        if denominator > 0
        else float("nan")
    )
    return rho, False, n_unique_risk


def summarize_risk_score(
    risk_score: str,
    risk: Sequence[float] | np.ndarray,
    abs_error: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    risk_values = _finite_vector(risk, "risk score")
    errors = _finite_vector(abs_error, "absolute error")
    rho, constant_risk, n_unique_risk = spearman_with_average_ties(
        risk_values, errors
    )
    return {
        "risk_score": risk_score,
        "evaluation_role": PRIMARY_ROLE,
        "N": int(len(risk_values)),
        "rho_spearman": rho,
        "constant_risk_score": constant_risk,
        "n_unique_risk": n_unique_risk,
        "risk_min": float(risk_values.min()),
        "risk_mean": float(risk_values.mean()),
        "risk_median": float(np.median(risk_values)),
        "risk_p95": float(np.quantile(risk_values, 0.95)),
        "risk_max": float(risk_values.max()),
        "abs_error_mean": float(errors.mean()),
        "abs_error_median": float(np.median(errors)),
        "abs_error_p95": float(np.quantile(errors, 0.95)),
    }


def build_spearman_table(risk_table: pd.DataFrame) -> pd.DataFrame:
    errors = risk_table["abs_error_mc_mean"].to_numpy(dtype=np.float64)
    records = [
        summarize_risk_score(
            risk_score,
            risk_table[risk_score].to_numpy(dtype=np.float64),
            errors,
        )
        for risk_score in RISK_SCORE_ORDER
    ]
    return pd.DataFrame(records)


def risk_quantile_diagnostics_for_score(
    risk_score: str,
    risk: Sequence[float] | np.ndarray,
    abs_error: Sequence[float] | np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    risk_values = _finite_vector(risk, "risk score")
    errors = _finite_vector(abs_error, "absolute error")
    if len(risk_values) != len(errors):
        raise ValueError("Risk and absolute error lengths differ")
    ranked_risk_values = risk_rank_values(risk_values)
    labels = pd.qcut(
        pd.Series(ranked_risk_values),
        q=REQUESTED_QUANTILE_BINS,
        duplicates="drop",
    )
    categories = list(labels.cat.categories)
    actual_bins = len(categories)
    records: list[dict[str, Any]] = []
    for bin_index, category in enumerate(categories, start=1):
        mask = np.asarray(labels == category, dtype=bool)
        bin_risk = risk_values[mask]
        bin_error = errors[mask]
        records.append(
            {
                "risk_score": risk_score,
                "evaluation_role": PRIMARY_ROLE,
                "requested_bins": REQUESTED_QUANTILE_BINS,
                "actual_bins": actual_bins,
                "quantile_bin_index": bin_index,
                "quantile_bin_label": str(category),
                "N": int(mask.sum()),
                "risk_min": float(bin_risk.min()),
                "risk_median": float(np.median(bin_risk)),
                "risk_max": float(bin_risk.max()),
                "abs_error_mean": float(bin_error.mean()),
                "abs_error_median": float(np.median(bin_error)),
                "abs_error_p90": float(np.quantile(bin_error, 0.90)),
                "RMSE_mc_mean": float(np.sqrt(np.mean(np.square(bin_error)))),
            }
        )
    ratio = float("nan")
    if actual_bins >= 2:
        low_mean = float(records[0]["abs_error_mean"])
        high_mean = float(records[-1]["abs_error_mean"])
        if low_mean > 0:
            ratio = high_mean / low_mean
    descriptive_summary = {
        "risk_score": risk_score,
        "evaluation_role": PRIMARY_ROLE,
        "N": int(len(risk_values)),
        "requested_bins": REQUESTED_QUANTILE_BINS,
        "actual_bins": actual_bins,
        "descriptive_extreme_quantile_error_ratio": ratio,
        "is_formal_operating_point": False,
        "is_formal_reject_threshold": False,
        "is_decision_rule": False,
    }
    return records, descriptive_summary


def build_quantile_outputs(
    risk_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = risk_table["abs_error_mc_mean"].to_numpy(dtype=np.float64)
    diagnostic_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for risk_score in RISK_SCORE_ORDER:
        records, summary = risk_quantile_diagnostics_for_score(
            risk_score,
            risk_table[risk_score].to_numpy(dtype=np.float64),
            errors,
        )
        diagnostic_records.extend(records)
        summary_records.append(summary)
    return (
        pd.DataFrame.from_records(diagnostic_records),
        pd.DataFrame.from_records(summary_records),
    )


def make_config() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "primary_evaluation_role": PRIMARY_ROLE,
        "decision_development_n": EXPECTED_N,
        "risk_target_error": RISK_TARGET_ERROR,
        "stage1a_source": project_relative(STAGE1A_SOURCE),
        "stage1b_source": project_relative(STAGE1B_SOURCE_DIR),
        "risk_scores_fixed_order": list(RISK_SCORE_ORDER),
        "risk_quantile_requested_bins": REQUESTED_QUANTILE_BINS,
        "risk_quantile_tie_policy": "qcut_duplicates_drop",
        "risk_tie_round_decimals": RISK_TIE_ROUND_DECIMALS,
        "risk_tie_policy": RISK_TIE_POLICY,
        "raw_risk_values_modified": False,
        "risk_tie_handling_purpose": (
            "numerical floating-point tie handling only; not risk threshold tuning, "
            "method selection, or performance tuning"
        ),
        "spearman_tie_policy": "average_rank_then_Pearson_correlation",
        "constant_risk_rho": "NaN",
        "split_cp_width_interpretation": (
            "negative/control baseline; without [0,1] clipping width is approximately "
            "constant 2q, so observed sample variation is mainly boundary clipping"
        ),
        "descriptive_extreme_quantile_error_ratio_interpretation": (
            "descriptive only; not an operating point, reject threshold, or decision rule"
        ),
    }


def make_provenance() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "primary_evaluation_role": PRIMARY_ROLE,
        "decision_development_n": EXPECTED_N,
        "risk_target_error": RISK_TARGET_ERROR,
        "stage1a_source": project_relative(STAGE1A_SOURCE),
        "stage1b_source": project_relative(STAGE1B_SOURCE_DIR),
        "risk_scores": list(RISK_SCORE_ORDER),
        "risk_quantile_requested_bins": REQUESTED_QUANTILE_BINS,
        "risk_quantile_tie_policy": "qcut_duplicates_drop",
        "risk_tie_round_decimals": RISK_TIE_ROUND_DECIMALS,
        "risk_tie_policy": RISK_TIE_POLICY,
        "raw_risk_values_modified": False,
        "risk_tie_handling_purpose": (
            "numerical floating-point tie handling only; not risk threshold tuning, "
            "method selection, or performance tuning"
        ),
        "formal_risk_score_selected": False,
        "formal_risk_threshold_frozen": False,
        "risk_coverage_performed": False,
        "oracle_risk_coverage_performed": False,
        "ause_performed": False,
        "high_error_capture_performed": False,
        "legacy_20pct_reject_performed": False,
        "cp_calibration_truth_used_for_risk_evaluation": False,
        "random_test_accessed": False,
        "random_test_truth_accessed": False,
        "random_test_predictions_generated": False,
        "sealed_final_dates_accessed": False,
        "image_inference_performed": False,
        "mc_dropout_performed": False,
        "training_performed": False,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "cqr_performed": False,
        "cleaning_decision_performed": False,
    }


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def validate_formal_output_path(output_dir: Path) -> Path:
    candidate = _resolved(output_dir)
    if candidate != _resolved(OUTPUT_DIR):
        raise PermissionError(f"Unauthorized Stage 2A output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_outputs(
    output_dir: Path,
    spearman: pd.DataFrame,
    quantile_diagnostics: pd.DataFrame,
    descriptive_summary: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spearman.to_csv(output_dir / "risk_score_spearman.csv", index=False, mode="x")
    quantile_diagnostics.to_csv(
        output_dir / "risk_quantile_diagnostics.csv", index=False, mode="x"
    )
    descriptive_summary.to_csv(
        output_dir / "risk_score_descriptive_summary.csv", index=False, mode="x"
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
    base = load_stage1a_base()
    intervals = {
        risk_score: load_stage1b_intervals(risk_score)
        for risk_score in RISK_SCORE_ORDER[1:]
    }
    risk_table = build_aligned_risk_table(base, intervals)
    spearman = build_spearman_table(risk_table)
    quantile_diagnostics, descriptive_summary = build_quantile_outputs(risk_table)
    config = make_config()
    provenance = make_provenance()
    write_outputs(
        output_dir,
        spearman,
        quantile_diagnostics,
        descriptive_summary,
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
