"""Paper1 Risk Stage 2C: tie-aware high-error capture diagnostics.

This descriptive stage measures how much of the true top-absolute-error 10%
is expected to be captured by fixed top-risk budgets of 10%, 20%, and 30%.
It does not select a score, threshold, reject rate, or budget and performs no
model inference, training, CQR, cleaning, or economic decision.
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
import experiments.run_paper1_risk_stage2b_risk_coverage_v1 as stage2b


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = stage2a.PROTOCOL
STAGE = "risk_stage2c_high_error_capture_v1"
PRIMARY_ROLE = stage2a.PRIMARY_ROLE
EXPECTED_N = stage2a.EXPECTED_N
RISK_TARGET_ERROR = stage2a.RISK_TARGET_ERROR
RISK_SCORE_ORDER = stage2a.RISK_SCORE_ORDER
RISK_TIE_ROUND_DECIMALS = stage2a.RISK_TIE_ROUND_DECIMALS
RISK_TIE_POLICY = (
    "round risk scores to 12 decimal places for tie-sensitive risk grouping only"
)
HIGH_ERROR_TARGET_FRACTION = 0.10
RISK_BUDGET_FRACTIONS = (0.10, 0.20, 0.30)
TARGET_TIE_POLICY = (
    "fractional target membership for exact abs-error boundary ties"
)
RISK_BOUNDARY_TIE_POLICY = (
    "fractional expected selection within boundary risk tie group"
)
WEIGHT_SUM_TOLERANCE = 1e-9
METRIC_QC_TOLERANCE = 1e-12
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE

SUMMARY_COLUMNS = (
    "risk_score",
    "evaluation_role",
    "N",
    "high_error_target_fraction",
    "target_high_error_n",
    "high_error_threshold_abs_error",
    "risk_budget_fraction",
    "risk_budget_n",
    "expected_captured_high_error",
    "capture_rate",
    "precision_high_error",
    "random_capture_rate",
    "random_expected_captured_high_error",
    "random_precision_high_error",
    "capture_lift_vs_random",
    "oracle_capture_rate_ceiling",
    "n_fully_selected_higher_risk",
    "risk_boundary_tie_group_size",
    "risk_boundary_fraction",
    "risk_fractional_boundary_used",
    "expected_selected_n",
)


def validate_protocol(protocol: str) -> None:
    stage2a.validate_protocol(protocol)


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_risk_budget_fractions(
    fractions: Sequence[float] = RISK_BUDGET_FRACTIONS,
) -> None:
    observed = tuple(float(value) for value in fractions)
    if observed != RISK_BUDGET_FRACTIONS:
        raise ValueError("Risk budgets must remain fixed at 0.10, 0.20, and 0.30")


def fractional_count(fraction: float, n: int) -> int:
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("Fraction must lie in (0,1]")
    if n <= 0:
        raise ValueError("N must be positive")
    return max(1, math.floor(fraction * n))


def target_high_error_count(n: int) -> int:
    return fractional_count(HIGH_ERROR_TARGET_FRACTION, n)


def risk_budget_count(fraction: float, n: int) -> int:
    validate_risk_budget_fractions()
    if float(fraction) not in RISK_BUDGET_FRACTIONS:
        raise ValueError(f"Unauthorized risk budget fraction: {fraction}")
    return fractional_count(float(fraction), n)


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def fractional_top_group_weights(
    raw_score: Sequence[float] | np.ndarray,
    selected_n: int,
    *,
    use_stage2a_risk_rounding: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select highest score groups with fractional expected boundary weight."""
    raw = _finite_vector(raw_score, "selection score")
    if selected_n < 1 or selected_n > len(raw):
        raise ValueError("selected_n must lie in [1,N]")
    tie_score = (
        stage2a.risk_rank_values(raw)
        if use_stage2a_risk_rounding
        else raw.copy()
    )
    unique_scores, inverse, group_counts = np.unique(
        tie_score, return_inverse=True, return_counts=True
    )
    descending_counts = group_counts[::-1]
    descending_cumulative = np.cumsum(descending_counts)
    boundary_from_high = int(
        np.searchsorted(descending_cumulative, selected_n, side="left")
    )
    boundary_group_index = len(unique_scores) - 1 - boundary_from_high
    higher_mask = inverse > boundary_group_index
    boundary_mask = inverse == boundary_group_index
    n_higher = int(higher_mask.sum())
    n_boundary = int(boundary_mask.sum())
    needed = selected_n - n_higher
    boundary_fraction = needed / n_boundary
    if not (0.0 < boundary_fraction <= 1.0):
        raise RuntimeError("Boundary fraction must lie in (0,1]")
    weights = np.zeros(len(raw), dtype=np.float64)
    weights[higher_mask] = 1.0
    weights[boundary_mask] = boundary_fraction
    weight_sum = float(weights.sum())
    if not math.isclose(
        weight_sum,
        selected_n,
        rel_tol=0.0,
        abs_tol=WEIGHT_SUM_TOLERANCE,
    ):
        raise RuntimeError("Fractional selection weight sum does not match target")
    return weights, {
        "selected_n": int(selected_n),
        "boundary_score": float(unique_scores[boundary_group_index]),
        "n_strictly_higher": n_higher,
        "boundary_tie_group_size": n_boundary,
        "boundary_fraction": float(boundary_fraction),
        "fractional_boundary_used": bool(boundary_fraction < 1.0),
        "weight_sum": weight_sum,
    }


def build_high_error_target(
    abs_error: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    errors = _finite_vector(abs_error, "absolute error")
    if (errors < 0).any():
        raise ValueError("Absolute error must be non-negative")
    target_n = target_high_error_count(len(errors))
    weights, generic_audit = fractional_top_group_weights(
        errors,
        target_n,
        use_stage2a_risk_rounding=False,
    )
    audit = {
        "high_error_target_fraction": HIGH_ERROR_TARGET_FRACTION,
        "target_high_error_n": target_n,
        "high_error_threshold_abs_error": generic_audit["boundary_score"],
        "target_tie_policy": (
            "exact absolute-error equality only; no 12-decimal risk rounding"
        ),
        "n_strict_high_error": generic_audit["n_strictly_higher"],
        "target_boundary_tie_group_size": generic_audit[
            "boundary_tie_group_size"
        ],
        "target_boundary_fraction": generic_audit["boundary_fraction"],
        "target_fractional_boundary_used": generic_audit[
            "fractional_boundary_used"
        ],
        "target_weight_sum": generic_audit["weight_sum"],
        "target_definition": "top absolute errors of abs(true_L - mc_mean)",
        "target_used_for_risk_ranking": False,
    }
    validate_high_error_target(weights, audit, len(errors))
    return weights, audit


def validate_high_error_target(
    weights: Sequence[float] | np.ndarray,
    audit: Mapping[str, Any],
    n: int,
) -> None:
    weight_values = _finite_vector(weights, "high-error target weights")
    expected_n = target_high_error_count(n)
    if int(audit["target_high_error_n"]) != expected_n:
        raise ValueError("High-error target count rule mismatch")
    if len(weight_values) != n or np.any(weight_values < 0) or np.any(weight_values > 1):
        raise ValueError("High-error target weights must lie in [0,1] with length N")
    if not math.isclose(
        float(weight_values.sum()),
        expected_n,
        rel_tol=0.0,
        abs_tol=WEIGHT_SUM_TOLERANCE,
    ):
        raise ValueError("High-error target weight sum mismatch")
    fraction = float(audit["target_boundary_fraction"])
    if not (0.0 < fraction <= 1.0):
        raise ValueError("Target boundary fraction must lie in (0,1]")


def build_risk_selection(
    raw_risk: Sequence[float] | np.ndarray,
    budget_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    risk = _finite_vector(raw_risk, "risk score")
    if (risk < 0).any():
        raise ValueError("Risk score must be non-negative")
    budget_n = risk_budget_count(budget_fraction, len(risk))
    weights, generic_audit = fractional_top_group_weights(
        risk,
        budget_n,
        use_stage2a_risk_rounding=True,
    )
    audit = {
        "risk_budget_fraction": float(budget_fraction),
        "risk_budget_n": budget_n,
        "n_fully_selected_higher_risk": generic_audit["n_strictly_higher"],
        "risk_boundary_tie_group_size": generic_audit[
            "boundary_tie_group_size"
        ],
        "risk_boundary_fraction": generic_audit["boundary_fraction"],
        "risk_fractional_boundary_used": generic_audit[
            "fractional_boundary_used"
        ],
        "expected_selected_n": generic_audit["weight_sum"],
    }
    return weights, audit


def capture_metrics(
    target_weights: Sequence[float] | np.ndarray,
    selection_weights: Sequence[float] | np.ndarray,
    target_n: int,
    budget_n: int,
) -> dict[str, float]:
    target = _finite_vector(target_weights, "target weights")
    selected = _finite_vector(selection_weights, "selection weights")
    if len(target) != len(selected):
        raise ValueError("Target and selection weight lengths differ")
    n = len(target)
    expected_capture = float(np.dot(target, selected))
    capture_rate = expected_capture / target_n
    precision = expected_capture / budget_n
    random_capture_rate = budget_n / n
    random_expected_capture = target_n * random_capture_rate
    random_precision = target_n / n
    lift = capture_rate / random_capture_rate
    precision_lift = precision / random_precision
    if not math.isclose(
        lift,
        precision_lift,
        rel_tol=0.0,
        abs_tol=METRIC_QC_TOLERANCE,
    ):
        raise RuntimeError("Capture-lift equivalent definitions disagree")
    return {
        "expected_captured_high_error": expected_capture,
        "capture_rate": capture_rate,
        "precision_high_error": precision,
        "random_capture_rate": random_capture_rate,
        "random_expected_captured_high_error": random_expected_capture,
        "random_precision_high_error": random_precision,
        "capture_lift_vs_random": lift,
        "oracle_capture_rate_ceiling": min(1.0, budget_n / target_n),
    }


def build_capture_summary(
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage2b.validate_evaluation_table(evaluation)
    validate_risk_budget_fractions()
    errors = evaluation["abs_error_mc_mean"].to_numpy(dtype=np.float64)
    target_weights, target_audit = build_high_error_target(errors)
    target_n = int(target_audit["target_high_error_n"])
    records: list[dict[str, Any]] = []
    for risk_score in RISK_SCORE_ORDER:
        raw_risk = evaluation[risk_score].to_numpy(dtype=np.float64)
        for budget_fraction in RISK_BUDGET_FRACTIONS:
            selection_weights, selection_audit = build_risk_selection(
                raw_risk, budget_fraction
            )
            metrics = capture_metrics(
                target_weights,
                selection_weights,
                target_n,
                int(selection_audit["risk_budget_n"]),
            )
            records.append(
                {
                    "risk_score": risk_score,
                    "evaluation_role": PRIMARY_ROLE,
                    "N": len(evaluation),
                    "high_error_target_fraction": HIGH_ERROR_TARGET_FRACTION,
                    "target_high_error_n": target_n,
                    "high_error_threshold_abs_error": target_audit[
                        "high_error_threshold_abs_error"
                    ],
                    **selection_audit,
                    **metrics,
                }
            )
    summary = pd.DataFrame.from_records(records).loc[:, SUMMARY_COLUMNS]
    validate_capture_summary(summary, len(evaluation), target_n)
    return summary, target_audit


def validate_capture_summary(
    summary: pd.DataFrame,
    n: int,
    target_n: int,
) -> None:
    missing = set(SUMMARY_COLUMNS) - set(summary.columns)
    if missing:
        raise ValueError(f"Capture summary missing columns: {sorted(missing)}")
    numeric_columns = [
        column
        for column in SUMMARY_COLUMNS
        if column not in {"risk_score", "evaluation_role", "risk_fractional_boundary_used"}
    ]
    if not np.isfinite(summary[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Capture summary contains NaN/inf")
    if tuple(dict.fromkeys(summary["risk_score"])) != RISK_SCORE_ORDER:
        raise ValueError("Risk-score order must remain fixed")
    if set(summary["evaluation_role"]) != {PRIMARY_ROLE}:
        raise PermissionError("Capture summary role mismatch")
    if set(summary["N"].astype(int)) != {n}:
        raise ValueError("Capture summary N mismatch")
    if set(summary["target_high_error_n"].astype(int)) != {target_n}:
        raise ValueError("Capture summary target count mismatch")
    if not np.allclose(
        summary["high_error_target_fraction"],
        HIGH_ERROR_TARGET_FRACTION,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Capture summary target fraction mismatch")

    for risk_score in RISK_SCORE_ORDER:
        rows = summary.loc[summary["risk_score"] == risk_score].reset_index(drop=True)
        if tuple(rows["risk_budget_fraction"].astype(float)) != RISK_BUDGET_FRACTIONS:
            raise ValueError(f"{risk_score} risk-budget order mismatch")
        expected_budget_counts = [
            risk_budget_count(fraction, n) for fraction in RISK_BUDGET_FRACTIONS
        ]
        if rows["risk_budget_n"].astype(int).tolist() != expected_budget_counts:
            raise ValueError(f"{risk_score} budget-count rule mismatch")
        if not np.allclose(
            rows["expected_selected_n"],
            rows["risk_budget_n"],
            rtol=0.0,
            atol=WEIGHT_SUM_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} expected selected count mismatch")
        fractions = rows["risk_boundary_fraction"].to_numpy(dtype=np.float64)
        if np.any(fractions <= 0.0) or np.any(fractions > 1.0):
            raise ValueError(f"{risk_score} boundary fraction outside (0,1]")
        captured = rows["expected_captured_high_error"].to_numpy(dtype=np.float64)
        budgets = rows["risk_budget_n"].to_numpy(dtype=np.float64)
        if np.any(captured < -METRIC_QC_TOLERANCE):
            raise ValueError(f"{risk_score} expected capture is negative")
        if np.any(captured > target_n + METRIC_QC_TOLERANCE):
            raise ValueError(f"{risk_score} expected capture exceeds target")
        if np.any(captured > budgets + METRIC_QC_TOLERANCE):
            raise ValueError(f"{risk_score} expected capture exceeds budget")
        if not np.allclose(
            rows["capture_rate"],
            captured / target_n,
            rtol=0.0,
            atol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} capture-rate definition mismatch")
        if not np.allclose(
            rows["precision_high_error"],
            captured / budgets,
            rtol=0.0,
            atol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} precision definition mismatch")
        for metric in ("capture_rate", "precision_high_error"):
            values = rows[metric].to_numpy(dtype=np.float64)
            if np.any(values < -METRIC_QC_TOLERANCE) or np.any(
                values > 1.0 + METRIC_QC_TOLERANCE
            ):
                raise ValueError(f"{risk_score} {metric} outside [0,1]")
        expected_random_rate = budgets / n
        if not np.allclose(
            rows["random_capture_rate"],
            expected_random_rate,
            rtol=0.0,
            atol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} random capture-rate mismatch")
        if not np.allclose(
            rows["random_expected_captured_high_error"],
            target_n * expected_random_rate,
            rtol=0.0,
            atol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} random expected-capture mismatch")
        if not np.allclose(
            rows["random_precision_high_error"],
            target_n / n,
            rtol=0.0,
            atol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} random precision mismatch")
        lift_from_capture = (
            rows["capture_rate"].to_numpy(dtype=np.float64)
            / rows["random_capture_rate"].to_numpy(dtype=np.float64)
        )
        lift_from_precision = (
            rows["precision_high_error"].to_numpy(dtype=np.float64)
            / rows["random_precision_high_error"].to_numpy(dtype=np.float64)
        )
        if not np.allclose(
            rows["capture_lift_vs_random"],
            lift_from_capture,
            rtol=0.0,
            atol=METRIC_QC_TOLERANCE,
        ) or not np.allclose(
            rows["capture_lift_vs_random"],
            lift_from_precision,
            rtol=0.0,
            atol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} capture-lift equivalence mismatch")
        expected_ceiling = np.minimum(1.0, budgets / target_n)
        if not np.allclose(
            rows["oracle_capture_rate_ceiling"],
            expected_ceiling,
            rtol=0.0,
            atol=METRIC_QC_TOLERANCE,
        ):
            raise ValueError(f"{risk_score} Oracle ceiling mismatch")
        if not np.allclose(expected_ceiling, 1.0):
            raise ValueError("Fixed Stage 2C budgets must all have Oracle ceiling 1")


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
        "high_error_target_fraction": HIGH_ERROR_TARGET_FRACTION,
        "target_count_rule": "max(1, floor(high_error_target_fraction * N))",
        "high_error_target_tie_policy": TARGET_TIE_POLICY,
        "risk_budget_fractions": list(RISK_BUDGET_FRACTIONS),
        "risk_budget_count_rule": "max(1, floor(risk_budget_fraction * N))",
        "risk_selection_direction": "highest risk first",
        "risk_tie_round_decimals": RISK_TIE_ROUND_DECIMALS,
        "risk_tie_policy": RISK_TIE_POLICY,
        "raw_risk_values_modified": False,
        "risk_boundary_tie_policy": RISK_BOUNDARY_TIE_POLICY,
        "capture_definition": (
            "expected overlap between high-error target weights and "
            "risk-selection weights"
        ),
        "random_baseline": "uniform random sample of equal budget size",
        "output_files": [
            "high_error_capture_summary.csv",
            "high_error_target_audit.json",
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
        "formal_risk_budget_selected": False,
        "stage2b_definition_modified": False,
        "cqr_performed": False,
        "cleaning_decision_performed": False,
        "economic_decision_performed": False,
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
        raise PermissionError(f"Unauthorized Stage 2C output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_outputs(
    output_dir: Path,
    summary: pd.DataFrame,
    target_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output_dir / "high_error_capture_summary.csv", index=False, mode="x"
    )
    _write_json_exclusive(
        output_dir / "high_error_target_audit.json", target_audit
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
    base = stage2a.load_stage1a_base()
    intervals = {
        risk_score: stage2a.load_stage1b_intervals(risk_score)
        for risk_score in RISK_SCORE_ORDER[1:]
    }
    evaluation = stage2b.build_evaluation_table(base, intervals)
    stage2b.validate_evaluation_table(evaluation)
    summary, target_audit = build_capture_summary(evaluation)
    config = make_config()
    provenance = make_provenance()
    write_outputs(output_dir, summary, target_audit, config, provenance)
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
