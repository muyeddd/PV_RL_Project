"""Paper1 Risk Stage 2B: tie-aware risk-coverage, Oracle, and AUSE.

The formal entry point reuses the sealed Stage 2A input guards, alignment,
risk-score definitions, and 12-decimal numerical tie handling.  It does not
select a risk score, freeze a threshold/reject rate, run model inference or
training, or perform high-error capture, CQR, or cleaning decisions.
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

import experiments.run_paper1_risk_stage2a_score_validity_v1 as stage2a


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = stage2a.PROTOCOL
STAGE = "risk_stage2b_risk_coverage_v1"
PRIMARY_ROLE = stage2a.PRIMARY_ROLE
EXPECTED_N = stage2a.EXPECTED_N
RISK_TARGET_ERROR = stage2a.RISK_TARGET_ERROR
RISK_SCORE_ORDER = stage2a.RISK_SCORE_ORDER
MC_STD = stage2a.MC_STD
RAW_MC_WIDTH = stage2a.RAW_MC_WIDTH
SPLIT_CP_WIDTH = stage2a.SPLIT_CP_WIDTH
IRRADIANCE_MONDRIAN_WIDTH = stage2a.IRRADIANCE_MONDRIAN_WIDTH
PRED_L_MONDRIAN_WIDTH = stage2a.PRED_L_MONDRIAN_WIDTH
PRED_L_MC_INTERVAL_WIDTH = stage2a.PRED_L_MC_INTERVAL_WIDTH
PRED_L_STD_MC_WIDTH = stage2a.PRED_L_STD_MC_WIDTH
RISK_TIE_ROUND_DECIMALS = stage2a.RISK_TIE_ROUND_DECIMALS
RISK_TIE_POLICY = (
    "round risk scores to 12 decimal places for tie-sensitive risk grouping only"
)
BOUNDARY_TIE_POLICY = "fractional expected retention within boundary tie group"
ORACLE = "oracle"
CURVE_SCORE_ORDER = RISK_SCORE_ORDER + (ORACLE,)
COVERAGE_GRID = (
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
)
EXPECTED_RETAINED_TOLERANCE = 1e-9
METRIC_QC_TOLERANCE = 1e-12
AUSE_QC_TOLERANCE = 1e-12
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE

CURVE_COLUMNS = (
    "risk_score",
    "evaluation_role",
    "coverage_requested",
    "target_retained_n",
    "expected_retained_n",
    "n_fully_retained_lower",
    "boundary_tie_group_size",
    "boundary_fraction",
    "fractional_boundary_used",
    "MAE_mc_mean",
    "RMSE_mc_mean",
)
AUSE_COLUMNS = (
    "risk_score",
    "evaluation_role",
    "N",
    "coverage_min",
    "coverage_max",
    "n_coverage_points",
    "AUSE_MAE",
    "AUSE_RMSE",
)


def validate_protocol(protocol: str) -> None:
    stage2a.validate_protocol(protocol)


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_coverage_grid(coverage_grid: Sequence[float]) -> None:
    observed = tuple(float(value) for value in coverage_grid)
    if observed != COVERAGE_GRID:
        raise ValueError("Coverage grid must equal the fixed 19-point Stage 2B grid")


def target_retained_count(coverage: float, n: int) -> int:
    if not math.isfinite(coverage) or coverage <= 0.0 or coverage > 1.0:
        raise ValueError("Coverage must lie in (0,1]")
    if n <= 0:
        raise ValueError("N must be positive")
    return max(1, math.floor(coverage * n))


def build_evaluation_table(
    base: pd.DataFrame,
    interval_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    risk_table = stage2a.build_aligned_risk_table(base, interval_tables)
    result = risk_table.copy()
    result.insert(2, "role", base.reset_index(drop=True)["role"].to_numpy())
    signed_error = (
        result["true_L"].to_numpy(dtype=np.float64)
        - result["mc_mean"].to_numpy(dtype=np.float64)
    )
    result["signed_error_mc_mean"] = signed_error
    result["abs_error_mc_mean"] = np.abs(signed_error)
    result["sq_error_mc_mean"] = np.square(signed_error)
    return result


def validate_evaluation_table(
    frame: pd.DataFrame, *, enforce_expected_n: bool = True
) -> None:
    required = {
        "sample_id",
        "date",
        "role",
        "true_L",
        "mc_mean",
        "signed_error_mc_mean",
        "abs_error_mc_mean",
        "sq_error_mc_mean",
        *RISK_SCORE_ORDER,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Stage 2B evaluation table missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Stage 2B evaluation table is empty")
    if set(frame["role"].astype(str)) != {PRIMARY_ROLE}:
        raise PermissionError(f"Only {PRIMARY_ROLE} is authorized")
    if enforce_expected_n and len(frame) != EXPECTED_N:
        raise ValueError(
            f"{PRIMARY_ROLE} N guard failed: expected {EXPECTED_N}, got {len(frame)}"
        )
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-null and unique")

    normalized_dates = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    sealed = set(normalized_dates) & stage2a.SEALED_FINAL_DATES
    if sealed:
        raise PermissionError(f"Sealed final date rejected: {sorted(sealed)}")
    numeric_columns = [
        "true_L",
        "mc_mean",
        "signed_error_mc_mean",
        "abs_error_mc_mean",
        "sq_error_mc_mean",
        *RISK_SCORE_ORDER,
    ]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Stage 2B evaluation values must all be finite")
    if (numeric.loc[:, RISK_SCORE_ORDER] < 0).any().any():
        raise ValueError("All Stage 2B risk scores must be non-negative")

    signed = numeric["true_L"] - numeric["mc_mean"]
    if not np.array_equal(signed.to_numpy(), numeric["signed_error_mc_mean"].to_numpy()):
        raise ValueError("signed_error_mc_mean definition mismatch")
    if not np.array_equal(np.abs(signed.to_numpy()), numeric["abs_error_mc_mean"].to_numpy()):
        raise ValueError("abs_error_mc_mean definition mismatch")
    if not np.array_equal(np.square(signed.to_numpy()), numeric["sq_error_mc_mean"].to_numpy()):
        raise ValueError("sq_error_mc_mean definition mismatch")

    for risk_score in RISK_SCORE_ORDER:
        raw = numeric[risk_score].to_numpy(dtype=np.float64)
        rounded = stage2a.risk_rank_values(raw)
        if not np.array_equal(
            rounded, np.round(raw, decimals=RISK_TIE_ROUND_DECIMALS)
        ):
            raise ValueError("Stage 2A 12-decimal risk tie handling drift detected")


def tie_aware_retention_metrics(
    raw_score: Sequence[float] | np.ndarray,
    abs_error: Sequence[float] | np.ndarray,
    sq_error: Sequence[float] | np.ndarray,
    target_n: int,
    *,
    use_stage2a_risk_rounding: bool,
) -> dict[str, Any]:
    score = _finite_vector(raw_score, "score")
    absolute = _finite_vector(abs_error, "absolute error")
    squared = _finite_vector(sq_error, "squared error")
    if not (len(score) == len(absolute) == len(squared)):
        raise ValueError("Score/error lengths differ")
    if target_n < 1 or target_n > len(score):
        raise ValueError("target_n must lie in [1,N]")
    if (absolute < 0).any() or (squared < 0).any():
        raise ValueError("Absolute and squared errors must be non-negative")
    tie_score = (
        stage2a.risk_rank_values(score)
        if use_stage2a_risk_rounding
        else score.copy()
    )
    unique_scores, inverse, group_counts = np.unique(
        tie_score, return_inverse=True, return_counts=True
    )
    cumulative_counts = np.cumsum(group_counts)
    boundary_group_index = int(np.searchsorted(cumulative_counts, target_n, side="left"))
    lower_mask = inverse < boundary_group_index
    boundary_mask = inverse == boundary_group_index
    n_lower = int(lower_mask.sum())
    n_boundary = int(boundary_mask.sum())
    needed = target_n - n_lower
    boundary_fraction = needed / n_boundary
    if not (0.0 < boundary_fraction <= 1.0):
        raise RuntimeError("Boundary fraction must lie in (0,1]")
    expected_retained_n = n_lower + boundary_fraction * n_boundary
    weighted_abs_sum = float(absolute[lower_mask].sum()) + boundary_fraction * float(
        absolute[boundary_mask].sum()
    )
    weighted_sq_sum = float(squared[lower_mask].sum()) + boundary_fraction * float(
        squared[boundary_mask].sum()
    )
    if not math.isclose(
        expected_retained_n,
        target_n,
        rel_tol=0.0,
        abs_tol=EXPECTED_RETAINED_TOLERANCE,
    ):
        raise RuntimeError("Expected retained count does not match target")
    return {
        "target_retained_n": int(target_n),
        "expected_retained_n": float(expected_retained_n),
        "n_fully_retained_lower": n_lower,
        "boundary_tie_group_size": n_boundary,
        "boundary_fraction": float(boundary_fraction),
        "fractional_boundary_used": bool(boundary_fraction < 1.0),
        "MAE_mc_mean": weighted_abs_sum / target_n,
        "RMSE_mc_mean": math.sqrt(weighted_sq_sum / target_n),
        "boundary_group_score": float(unique_scores[boundary_group_index]),
    }


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def build_curve_for_score(
    risk_score: str,
    raw_score: Sequence[float] | np.ndarray,
    abs_error: Sequence[float] | np.ndarray,
    sq_error: Sequence[float] | np.ndarray,
    *,
    oracle: bool = False,
) -> pd.DataFrame:
    if oracle:
        if risk_score != ORACLE:
            raise ValueError("Oracle curve must use risk_score='oracle'")
    elif risk_score not in RISK_SCORE_ORDER:
        raise ValueError(f"Unauthorized Stage 2B risk score: {risk_score}")
    validate_coverage_grid(COVERAGE_GRID)
    score = _finite_vector(raw_score, "score")
    records: list[dict[str, Any]] = []
    for coverage in COVERAGE_GRID:
        target_n = target_retained_count(coverage, len(score))
        metrics = tie_aware_retention_metrics(
            score,
            abs_error,
            sq_error,
            target_n,
            use_stage2a_risk_rounding=not oracle,
        )
        records.append(
            {
                "risk_score": risk_score,
                "evaluation_role": PRIMARY_ROLE,
                "coverage_requested": coverage,
                **metrics,
            }
        )
    return pd.DataFrame.from_records(records).loc[:, CURVE_COLUMNS]


def build_all_curves(evaluation: pd.DataFrame) -> pd.DataFrame:
    validate_evaluation_table(evaluation)
    absolute = evaluation["abs_error_mc_mean"].to_numpy(dtype=np.float64)
    squared = evaluation["sq_error_mc_mean"].to_numpy(dtype=np.float64)
    curves = [
        build_curve_for_score(
            risk_score,
            evaluation[risk_score].to_numpy(dtype=np.float64),
            absolute,
            squared,
        )
        for risk_score in RISK_SCORE_ORDER
    ]
    curves.append(
        build_curve_for_score(
            ORACLE,
            absolute,
            absolute,
            squared,
            oracle=True,
        )
    )
    result = pd.concat(curves, ignore_index=True)
    validate_risk_coverage_curves(result, absolute, squared)
    return result


def validate_risk_coverage_curves(
    curves: pd.DataFrame,
    abs_error: Sequence[float] | np.ndarray,
    sq_error: Sequence[float] | np.ndarray,
) -> None:
    missing = set(CURVE_COLUMNS) - set(curves.columns)
    if missing:
        raise ValueError(f"Risk-coverage curves missing columns: {sorted(missing)}")
    if not np.isfinite(
        curves[
            [
                "coverage_requested",
                "target_retained_n",
                "expected_retained_n",
                "boundary_fraction",
                "MAE_mc_mean",
                "RMSE_mc_mean",
            ]
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Risk-coverage curves contain NaN/inf")
    if tuple(dict.fromkeys(curves["risk_score"])) != CURVE_SCORE_ORDER:
        raise ValueError("Risk-coverage score order must remain fixed")
    absolute = _finite_vector(abs_error, "absolute error")
    squared = _finite_vector(sq_error, "squared error")
    overall_mae = float(absolute.mean())
    overall_rmse = float(np.sqrt(squared.mean()))

    oracle_rows = _ordered_curve_rows(curves, ORACLE)
    if np.any(np.diff(oracle_rows["MAE_mc_mean"].to_numpy()) < -METRIC_QC_TOLERANCE):
        raise ValueError("Oracle MAE must be nondecreasing with coverage")
    if np.any(np.diff(oracle_rows["RMSE_mc_mean"].to_numpy()) < -METRIC_QC_TOLERANCE):
        raise ValueError("Oracle RMSE must be nondecreasing with coverage")

    for risk_score in CURVE_SCORE_ORDER:
        rows = _ordered_curve_rows(curves, risk_score)
        if len(rows) != len(COVERAGE_GRID):
            raise ValueError(f"{risk_score} must have exactly 19 coverage points")
        if tuple(rows["coverage_requested"].astype(float)) != COVERAGE_GRID:
            raise ValueError(f"{risk_score} coverage grid mismatch")
        expected_targets = [
            target_retained_count(coverage, len(absolute))
            for coverage in COVERAGE_GRID
        ]
        if rows["target_retained_n"].astype(int).tolist() != expected_targets:
            raise ValueError(f"{risk_score} retained-count rule mismatch")
        if not np.allclose(
            rows["expected_retained_n"],
            rows["target_retained_n"],
            rtol=0.0,
            atol=EXPECTED_RETAINED_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} expected retained count mismatch")
        fractions = rows["boundary_fraction"].to_numpy(dtype=np.float64)
        if np.any(fractions <= 0.0) or np.any(fractions > 1.0):
            raise ValueError(f"{risk_score} boundary fraction outside (0,1]")
        full = rows.iloc[-1]
        if not math.isclose(
            float(full["MAE_mc_mean"]),
            overall_mae,
            rel_tol=0.0,
            abs_tol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} coverage=1 MAE mismatch")
        if not math.isclose(
            float(full["RMSE_mc_mean"]),
            overall_rmse,
            rel_tol=0.0,
            abs_tol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} coverage=1 RMSE mismatch")
        if risk_score != ORACLE:
            if np.any(
                rows["MAE_mc_mean"].to_numpy()
                < oracle_rows["MAE_mc_mean"].to_numpy() - METRIC_QC_TOLERANCE
            ):
                raise ValueError(f"{risk_score} MAE is unexpectedly better than Oracle")
            if np.any(
                rows["RMSE_mc_mean"].to_numpy()
                < oracle_rows["RMSE_mc_mean"].to_numpy() - METRIC_QC_TOLERANCE
            ):
                raise ValueError(f"{risk_score} RMSE is unexpectedly better than Oracle")


def _ordered_curve_rows(curves: pd.DataFrame, risk_score: str) -> pd.DataFrame:
    rows = curves.loc[curves["risk_score"] == risk_score].copy()
    if set(rows["coverage_requested"].astype(float)) != set(COVERAGE_GRID):
        raise ValueError(f"{risk_score} does not contain the fixed coverage grid")
    return rows.set_index("coverage_requested").loc[list(COVERAGE_GRID)].reset_index()


def trapezoidal_integral(
    values: Sequence[float] | np.ndarray,
    x: Sequence[float] | np.ndarray,
) -> float:
    y_values = _finite_vector(values, "integrand")
    x_values = _finite_vector(x, "integration grid")
    if len(y_values) != len(x_values) or len(y_values) < 2:
        raise ValueError("Trapezoidal integration inputs must have equal length >=2")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("Integration grid must be strictly increasing")
    return float(
        np.sum(np.diff(x_values) * (y_values[:-1] + y_values[1:]) / 2.0)
    )


def build_ause_summary(curves: pd.DataFrame) -> pd.DataFrame:
    oracle_rows = _ordered_curve_rows(curves, ORACLE)
    coverage = np.asarray(COVERAGE_GRID, dtype=np.float64)
    oracle_mae = oracle_rows["MAE_mc_mean"].to_numpy(dtype=np.float64)
    oracle_rmse = oracle_rows["RMSE_mc_mean"].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    for risk_score in RISK_SCORE_ORDER:
        method_rows = _ordered_curve_rows(curves, risk_score)
        ause_mae = trapezoidal_integral(
            method_rows["MAE_mc_mean"].to_numpy(dtype=np.float64) - oracle_mae,
            coverage,
        )
        ause_rmse = trapezoidal_integral(
            method_rows["RMSE_mc_mean"].to_numpy(dtype=np.float64) - oracle_rmse,
            coverage,
        )
        if ause_mae < -AUSE_QC_TOLERANCE or ause_rmse < -AUSE_QC_TOLERANCE:
            raise ValueError(f"{risk_score} AUSE is significantly negative")
        records.append(
            {
                "risk_score": risk_score,
                "evaluation_role": PRIMARY_ROLE,
                "N": EXPECTED_N,
                "coverage_min": COVERAGE_GRID[0],
                "coverage_max": COVERAGE_GRID[-1],
                "n_coverage_points": len(COVERAGE_GRID),
                "AUSE_MAE": ause_mae,
                "AUSE_RMSE": ause_rmse,
            }
        )
    result = pd.DataFrame.from_records(records).loc[:, AUSE_COLUMNS]
    if tuple(result["risk_score"]) != RISK_SCORE_ORDER:
        raise RuntimeError("AUSE output order drifted")
    return result


def make_config() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "primary_evaluation_role": PRIMARY_ROLE,
        "decision_development_n": EXPECTED_N,
        "risk_target_error": RISK_TARGET_ERROR,
        "stage1a_source": project_relative(stage2a.STAGE1A_SOURCE),
        "stage1b_source": project_relative(stage2a.STAGE1B_SOURCE_DIR),
        "risk_scores": list(RISK_SCORE_ORDER),
        "risk_tie_round_decimals": RISK_TIE_ROUND_DECIMALS,
        "risk_tie_policy": RISK_TIE_POLICY,
        "raw_risk_values_modified": False,
        "boundary_tie_policy": BOUNDARY_TIE_POLICY,
        "coverage_grid": list(COVERAGE_GRID),
        "retained_count_rule": "max(1, floor(coverage * N))",
        "oracle_score": "abs(true_L - mc_mean)",
        "oracle_tie_policy": "exact equality only; no 12-decimal risk rounding",
        "ause_integration": (
            "trapezoidal integral of method risk minus oracle risk over fixed "
            "coverage grid"
        ),
        "ause_coverage_span_normalized": False,
        "curve_score_order": list(CURVE_SCORE_ORDER),
        "output_files": [
            "risk_coverage_curves.csv",
            "ause_summary.csv",
            "config.json",
            "provenance.json",
        ],
    }


def make_provenance() -> dict[str, Any]:
    return {
        **make_config(),
        "formal_risk_score_selected": False,
        "formal_risk_threshold_frozen": False,
        "formal_reject_rate_frozen": False,
        "high_error_capture_performed": False,
        "legacy_20pct_reject_performed": False,
        "cleaning_decision_performed": False,
        "cqr_performed": False,
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
    }


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def validate_formal_output_path(output_dir: Path) -> Path:
    candidate = _resolved(output_dir)
    if candidate != _resolved(OUTPUT_DIR):
        raise PermissionError(f"Unauthorized Stage 2B output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_outputs(
    output_dir: Path,
    curves: pd.DataFrame,
    ause_summary: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / "risk_coverage_curves.csv", index=False, mode="x")
    ause_summary.to_csv(output_dir / "ause_summary.csv", index=False, mode="x")
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "provenance.json", provenance)


def run(
    protocol: str = PROTOCOL,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    validate_protocol(protocol)
    output_dir = validate_formal_output_path(output_dir)
    ensure_output_available(output_dir)
    base = stage2a.load_stage1a_base()
    intervals = {
        risk_score: stage2a.load_stage1b_intervals(risk_score)
        for risk_score in RISK_SCORE_ORDER[1:]
    }
    evaluation = build_evaluation_table(base, intervals)
    validate_evaluation_table(evaluation)
    curves = build_all_curves(evaluation)
    ause_summary = build_ause_summary(curves)
    config = make_config()
    provenance = make_provenance()
    write_outputs(output_dir, curves, ause_summary, config, provenance)
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
