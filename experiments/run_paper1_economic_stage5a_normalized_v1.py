"""Paper1 Stage 5-A1: normalized decision economics.

This table-only stage consumes the frozen Stage 4A decision actions and uses
normalized recoverable value V=1.  It introduces no currency, site scale,
training, inference, recalibration, threshold search, or method selection.
Importing this module reads no formal artifact.
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

from experiments import run_paper1_decision_stage4a_rule_v1 as stage4a


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"
STAGE = "economic_stage5a_normalized_v1"
EVALUATION_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_N = 1844
TAU_GRID = (0.05, 0.10, 0.15, 0.20)
REFERENCE_TAU = 0.15
METHOD_ORDER = (
    "point_threshold",
    "cqr_q50_threshold",
    "cqr_interval_tristate",
)
NORMALIZED_RECOVERABLE_VALUE = 1.0

POINT_THRESHOLD = stage4a.POINT_THRESHOLD
CQR_Q50_THRESHOLD = stage4a.CQR_Q50_THRESHOLD
CQR_INTERVAL_TRISTATE = stage4a.CQR_INTERVAL_TRISTATE
CLEAN = stage4a.CLEAN
WAIT = stage4a.WAIT
REVIEW = stage4a.REVIEW

STAGE4A_ACTIONS_INPUT = stage4a.OUTPUT_DIR / "decision_actions.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE
AUTHORIZED_INPUTS = {"stage4a_decision_actions": STAGE4A_ACTIONS_INPUT}

EXPECTED_FORMAL_ROWS = EXPECTED_N * len(TAU_GRID) * len(METHOD_ORDER)
NUMERIC_ABS_TOLERANCE = 1e-12
BREAK_EVEN_ABS_TOLERANCE = 1e-12
FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "winner",
        "best",
        "rank",
        "ranking",
        "recommended_method",
        "selected_method",
        "optimal_tau",
    }
)

SAMPLE_REGRET_COLUMNS = stage4a.ACTION_COLUMNS + (
    "oracle_cost",
    "base_action_cost_r0",
    "regret_r0",
    "false_clean_regret",
    "missed_clean_regret",
    "automatic_error_regret",
    "review_indicator",
)
METRIC_COLUMNS = (
    "method",
    "tau",
    "evaluation_role",
    "N",
    "oracle_mean_cost",
    "mean_regret_r0",
    "false_clean_regret_sum",
    "false_clean_regret_mean_per_N",
    "missed_clean_regret_sum",
    "missed_clean_regret_mean_per_N",
    "automatic_error_regret_sum",
    "review_n",
    "review_rate",
    "mean_total_cost_r0",
)
BREAK_EVEN_COLUMNS = (
    "tau",
    "evaluation_role",
    "N",
    "cqr_method",
    "review_n",
    "review_rate",
    "cqr_mean_regret_r0",
    "point_mean_regret_r0",
    "q50_mean_regret_r0",
    "break_even_vs_point",
    "break_even_vs_q50",
)


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_protocol_constants() -> None:
    if PROTOCOL != stage4a.PROTOCOL:
        raise ValueError("Stage 4A/5-A1 protocol constants disagree")
    if EVALUATION_ROLE != stage4a.EVALUATION_ROLE:
        raise ValueError("Stage 5-A1 evaluation role drifted")
    if EXPECTED_N != stage4a.EXPECTED_N:
        raise ValueError("Stage 5-A1 expected N drifted")
    if TAU_GRID != stage4a.TAU_GRID or TAU_GRID != (0.05, 0.10, 0.15, 0.20):
        raise ValueError("Stage 5-A1 tau grid drifted")
    if METHOD_ORDER != stage4a.METHOD_ORDER or METHOD_ORDER != (
        "point_threshold",
        "cqr_q50_threshold",
        "cqr_interval_tristate",
    ):
        raise ValueError("Stage 5-A1 method order drifted")
    if REFERENCE_TAU != stage4a.REFERENCE_TAU or REFERENCE_TAU != 0.15:
        raise ValueError("Stage 5-A1 reference tau drifted")
    if NORMALIZED_RECOVERABLE_VALUE != 1.0:
        raise ValueError("Normalized recoverable value must equal one")


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
        raise PermissionError("CP_CALIBRATION input is forbidden")
    if source_key not in AUTHORIZED_INPUTS:
        raise PermissionError(f"Unauthorized Stage 5-A1 source key: {source_key}")
    authorized = _resolved(AUTHORIZED_INPUTS[source_key])
    if candidate != authorized:
        raise PermissionError(f"Unauthorized Stage 5-A1 input path: {candidate}")
    return candidate


def _forbidden_columns(frame: pd.DataFrame) -> set[str]:
    return {
        str(column).lower()
        for column in frame.columns
        if str(column).lower() in FORBIDDEN_OUTPUT_FIELDS
    }


def validate_stage4_actions(
    actions: pd.DataFrame, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    if tuple(actions.columns) != stage4a.ACTION_COLUMNS:
        raise ValueError(
            "Frozen Stage 4A action schema mismatch: "
            f"expected {stage4a.ACTION_COLUMNS}, got {tuple(actions.columns)}"
        )
    n_samples = EXPECTED_N if enforce_expected_n else int(actions["sample_id"].nunique())
    stage4a.validate_decision_actions(
        actions,
        n_samples=n_samples,
        enforce_expected_n=enforce_expected_n,
    )
    if enforce_expected_n and len(actions) != EXPECTED_FORMAL_ROWS:
        raise ValueError(
            f"Formal Stage 4A row-count guard failed: expected {EXPECTED_FORMAL_ROWS}, "
            f"got {len(actions)}"
        )
    if set(actions["role"].astype(str)) != {EVALUATION_ROLE}:
        raise PermissionError("Only DECISION_DEVELOPMENT is authorized")
    if actions.duplicated(["sample_id", "method", "tau"]).any():
        raise ValueError("Duplicate sample/method/tau action rejected")
    normalized_dates = pd.to_datetime(actions["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    sealed = set(normalized_dates) & stage4a.stage1a.SEALED_FINAL_DATES
    if sealed:
        raise PermissionError(f"Sealed final date rejected: {sorted(sealed)}")
    locators = actions["image_path"].astype(str)
    if locators.str.lower().str.contains("random_test", regex=False).any():
        raise PermissionError("RANDOM_TEST locator rejected")
    for sealed_date in stage4a.stage1a.SEALED_FINAL_DATES:
        if locators.str.contains(sealed_date, regex=False).any():
            raise PermissionError(f"Sealed final date locator rejected: {sealed_date}")
    true_l = pd.to_numeric(actions["true_L"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(true_l).all():
        raise ValueError("true_L must be finite")
    grouped = actions.groupby("sample_id", sort=False, dropna=False)
    if not (grouped.size() == len(TAU_GRID) * len(METHOD_ORDER)).all():
        raise ValueError("Each sample must have every fixed method/tau action")
    for field in ("date", "timestamp", "image_path", "role", "true_L"):
        if not (grouped[field].nunique(dropna=False) == 1).all():
            raise ValueError(f"Per-sample {field} must be consistent")
    if _forbidden_columns(actions):
        raise ValueError("Winner/ranking/optimal-tau fields are forbidden")
    return actions.loc[:, stage4a.ACTION_COLUMNS].copy()


def load_stage4_actions(path: Path = STAGE4A_ACTIONS_INPUT) -> pd.DataFrame:
    authorized = validate_authorized_input_path(path, "stage4a_decision_actions")
    return validate_stage4_actions(pd.read_csv(authorized))


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_tau(tau: float) -> float:
    value = _finite_scalar(tau, "tau")
    if value not in TAU_GRID:
        raise ValueError(f"Unauthorized Stage 5-A1 tau: {tau}")
    return value


def oracle_cost(true_l: float, tau: float) -> float:
    loss = _finite_scalar(true_l, "true_L")
    threshold = validate_tau(tau)
    return min(threshold, loss)


def review_regret(review_cost_ratio: float, tau: float) -> float:
    ratio = _finite_scalar(review_cost_ratio, "review cost ratio")
    if ratio < 0.0:
        raise ValueError("Physical review cost ratio must be non-negative")
    threshold = validate_tau(tau)
    return ratio * threshold


def review_total_cost(true_l: float, tau: float, review_cost_ratio: float) -> float:
    return oracle_cost(true_l, tau) + review_regret(review_cost_ratio, tau)


def action_cost_r0(true_l: float, tau: float, predicted_action: str) -> float:
    loss = _finite_scalar(true_l, "true_L")
    threshold = validate_tau(tau)
    if predicted_action == CLEAN:
        return threshold
    if predicted_action == WAIT:
        return loss
    if predicted_action == REVIEW:
        return min(threshold, loss)
    raise ValueError(f"Illegal predicted action: {predicted_action!r}")


def sample_regret_components(
    true_l: float, tau: float, predicted_action: str
) -> dict[str, float | bool]:
    loss = _finite_scalar(true_l, "true_L")
    threshold = validate_tau(tau)
    oracle = min(threshold, loss)
    action = action_cost_r0(loss, threshold, predicted_action)
    regret = action - oracle
    if regret < -NUMERIC_ABS_TOLERANCE:
        raise ValueError("Normalized regret must be non-negative")
    false_clean = predicted_action == CLEAN and loss <= threshold
    missed_clean = predicted_action == WAIT and loss > threshold
    false_clean_regret = threshold - loss if false_clean else 0.0
    missed_clean_regret = loss - threshold if missed_clean else 0.0
    automatic_error_regret = false_clean_regret + missed_clean_regret
    if not math.isclose(
        regret,
        automatic_error_regret,
        rel_tol=0.0,
        abs_tol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Sample regret decomposition failed")
    if (false_clean or missed_clean) and not math.isclose(
        regret,
        abs(loss - threshold),
        rel_tol=0.0,
        abs_tol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Automatic-error regret must equal abs(true_L-tau)")
    return {
        "oracle_cost": oracle,
        "base_action_cost_r0": action,
        "regret_r0": regret,
        "false_clean_regret": false_clean_regret,
        "missed_clean_regret": missed_clean_regret,
        "automatic_error_regret": automatic_error_regret,
        "review_indicator": predicted_action == REVIEW,
    }


def build_sample_regrets(
    actions: pd.DataFrame, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    validated = validate_stage4_actions(
        actions, enforce_expected_n=enforce_expected_n
    )
    components = [
        sample_regret_components(row.true_L, row.tau, row.predicted_action)
        for row in validated.itertuples(index=False)
    ]
    result = pd.concat(
        [validated.reset_index(drop=True), pd.DataFrame.from_records(components)],
        axis=1,
    ).loc[:, SAMPLE_REGRET_COLUMNS]
    validate_sample_regrets(result, enforce_expected_n=enforce_expected_n)
    return result


def validate_sample_regrets(
    frame: pd.DataFrame, *, enforce_expected_n: bool = True
) -> None:
    if tuple(frame.columns) != SAMPLE_REGRET_COLUMNS:
        raise ValueError("Stage 5-A1 sample-regret schema mismatch")
    validate_stage4_actions(
        frame.loc[:, stage4a.ACTION_COLUMNS],
        enforce_expected_n=enforce_expected_n,
    )
    numeric_columns = (
        "oracle_cost",
        "base_action_cost_r0",
        "regret_r0",
        "false_clean_regret",
        "missed_clean_regret",
        "automatic_error_regret",
    )
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Stage 5-A1 sample economics must be finite")
    if np.any(values < -NUMERIC_ABS_TOLERANCE):
        raise ValueError("Stage 5-A1 costs/regrets must be non-negative")
    if not np.allclose(
        numeric["base_action_cost_r0"],
        numeric["oracle_cost"] + numeric["regret_r0"],
        rtol=0.0,
        atol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Sample total cost must equal oracle cost plus regret")
    if not np.allclose(
        numeric["automatic_error_regret"],
        numeric["false_clean_regret"] + numeric["missed_clean_regret"],
        rtol=0.0,
        atol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Automatic-error regret decomposition failed")
    expected_review = frame["predicted_action"].astype(str) == REVIEW
    if not np.array_equal(frame["review_indicator"].astype(bool), expected_review):
        raise ValueError("Review indicator is inconsistent")
    recomputed = pd.DataFrame.from_records(
        [
            sample_regret_components(row.true_L, row.tau, row.predicted_action)
            for row in frame.itertuples(index=False)
        ]
    )
    for field in numeric_columns:
        if not np.allclose(
            numeric[field].to_numpy(dtype=np.float64),
            recomputed[field].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=NUMERIC_ABS_TOLERANCE,
        ):
            raise ValueError(f"Stored sample economics mismatch for {field}")


def summarize_economic_group(group: pd.DataFrame) -> dict[str, int | float]:
    if group.empty:
        raise ValueError("Economic metric group is empty")
    n = len(group)
    oracle_costs = group["oracle_cost"].to_numpy(dtype=np.float64)
    regrets = group["regret_r0"].to_numpy(dtype=np.float64)
    false_regrets = group["false_clean_regret"].to_numpy(dtype=np.float64)
    missed_regrets = group["missed_clean_regret"].to_numpy(dtype=np.float64)
    action_costs = group["base_action_cost_r0"].to_numpy(dtype=np.float64)
    review_n = int(group["review_indicator"].astype(bool).sum())
    result: dict[str, int | float] = {
        "N": n,
        "oracle_mean_cost": float(oracle_costs.mean()),
        "mean_regret_r0": float(regrets.mean()),
        "false_clean_regret_sum": float(false_regrets.sum()),
        "false_clean_regret_mean_per_N": float(false_regrets.sum() / n),
        "missed_clean_regret_sum": float(missed_regrets.sum()),
        "missed_clean_regret_mean_per_N": float(missed_regrets.sum() / n),
        "automatic_error_regret_sum": float(
            group["automatic_error_regret"].sum()
        ),
        "review_n": review_n,
        "review_rate": review_n / n,
        "mean_total_cost_r0": float(action_costs.mean()),
    }
    if not math.isclose(
        float(result["mean_total_cost_r0"]),
        float(result["oracle_mean_cost"]) + float(result["mean_regret_r0"]),
        rel_tol=0.0,
        abs_tol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Mean total cost identity failed")
    if not math.isclose(
        float(result["automatic_error_regret_sum"]),
        float(regrets.sum()),
        rel_tol=0.0,
        abs_tol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Mean regret must contain only automatic-error regret at r=0")
    return result


def build_economic_metrics(
    sample_regrets: pd.DataFrame, *, enforce_expected_n: bool = True
) -> pd.DataFrame:
    validate_sample_regrets(
        sample_regrets, enforce_expected_n=enforce_expected_n
    )
    records: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for tau in TAU_GRID:
            group = sample_regrets.loc[
                (sample_regrets["method"] == method)
                & (sample_regrets["tau"] == tau)
            ]
            records.append(
                {
                    "method": method,
                    "tau": tau,
                    "evaluation_role": EVALUATION_ROLE,
                    **summarize_economic_group(group),
                }
            )
    result = pd.DataFrame.from_records(records).loc[:, METRIC_COLUMNS]
    n_samples = EXPECTED_N if enforce_expected_n else int(
        sample_regrets["sample_id"].nunique()
    )
    validate_economic_metrics(result, n_samples=n_samples)
    return result


def validate_economic_metrics(metrics: pd.DataFrame, *, n_samples: int) -> None:
    if tuple(metrics.columns) != METRIC_COLUMNS:
        raise ValueError("Stage 5-A1 metric schema mismatch")
    if len(metrics) != len(METHOD_ORDER) * len(TAU_GRID):
        raise ValueError("Stage 5-A1 metric row-count mismatch")
    if tuple(dict.fromkeys(metrics["method"].astype(str))) != METHOD_ORDER:
        raise ValueError("Stage 5-A1 metric method order mismatch")
    for method in METHOD_ORDER:
        rows = metrics.loc[metrics["method"] == method]
        if tuple(rows["tau"].astype(float)) != TAU_GRID:
            raise ValueError("Stage 5-A1 metric tau order mismatch")
    if set(metrics["evaluation_role"].astype(str)) != {EVALUATION_ROLE}:
        raise PermissionError("Stage 5-A1 metric role mismatch")
    if set(metrics["N"].astype(int)) != {n_samples}:
        raise ValueError("Stage 5-A1 metric N mismatch")
    numeric_columns = tuple(
        column
        for column in METRIC_COLUMNS
        if column not in {"method", "tau", "evaluation_role", "N"}
    )
    numeric = metrics.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Stage 5-A1 metrics must be finite")
    nonnegative_columns = tuple(column for column in numeric_columns)
    if np.any(numeric.loc[:, nonnegative_columns].to_numpy() < -NUMERIC_ABS_TOLERANCE):
        raise ValueError("Stage 5-A1 metrics must be non-negative")
    if not np.allclose(
        metrics["mean_total_cost_r0"],
        metrics["oracle_mean_cost"] + metrics["mean_regret_r0"],
        rtol=0.0,
        atol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Metric total-cost identity failed")
    if not np.allclose(
        metrics["automatic_error_regret_sum"],
        metrics["false_clean_regret_sum"] + metrics["missed_clean_regret_sum"],
        rtol=0.0,
        atol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Metric automatic-error regret decomposition failed")
    if not np.allclose(
        metrics["mean_regret_r0"] * metrics["N"],
        metrics["automatic_error_regret_sum"],
        rtol=0.0,
        atol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Metric mean-regret aggregation failed")
    if not np.allclose(
        metrics["false_clean_regret_mean_per_N"] * metrics["N"],
        metrics["false_clean_regret_sum"],
        rtol=0.0,
        atol=NUMERIC_ABS_TOLERANCE,
    ) or not np.allclose(
        metrics["missed_clean_regret_mean_per_N"] * metrics["N"],
        metrics["missed_clean_regret_sum"],
        rtol=0.0,
        atol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Per-N regret aggregation failed")
    if not np.allclose(
        metrics["review_rate"],
        metrics["review_n"] / metrics["N"],
        rtol=0.0,
        atol=NUMERIC_ABS_TOLERANCE,
    ):
        raise ValueError("Review-rate aggregation failed")
    for tau in TAU_GRID:
        oracle_costs = metrics.loc[metrics["tau"] == tau, "oracle_mean_cost"]
        if not np.allclose(
            oracle_costs,
            float(oracle_costs.iloc[0]),
            rtol=0.0,
            atol=NUMERIC_ABS_TOLERANCE,
        ):
            raise ValueError("Oracle mean cost must be method-independent")
    binary = metrics["method"].isin((POINT_THRESHOLD, CQR_Q50_THRESHOLD))
    if not (metrics.loc[binary, "review_n"] == 0).all():
        raise ValueError("Binary baselines cannot contain REVIEW")
    if not (metrics.loc[binary, "review_rate"] == 0.0).all():
        raise ValueError("Binary baseline review rate must be zero")
    if _forbidden_columns(metrics):
        raise ValueError("Winner/ranking/optimal-tau fields are forbidden")


def analytic_break_even_ratio(
    baseline_mean_regret_r0: float,
    cqr_mean_regret_r0: float,
    review_rate: float,
    tau: float,
) -> float:
    baseline = _finite_scalar(baseline_mean_regret_r0, "baseline mean regret")
    cqr = _finite_scalar(cqr_mean_regret_r0, "CQR mean regret")
    rate = _finite_scalar(review_rate, "review rate")
    threshold = validate_tau(tau)
    if rate < 0.0 or rate > 1.0:
        raise ValueError("Review rate must lie in [0,1]")
    if rate == 0.0:
        return float("nan")
    return (baseline - cqr) / (rate * threshold)


def cqr_mean_regret_at_review_ratio(
    cqr_mean_regret_r0: float,
    review_rate: float,
    review_cost_ratio: float,
    tau: float,
) -> float:
    cqr = _finite_scalar(cqr_mean_regret_r0, "CQR mean regret")
    rate = _finite_scalar(review_rate, "review rate")
    ratio = _finite_scalar(review_cost_ratio, "review cost ratio")
    threshold = validate_tau(tau)
    return cqr + rate * ratio * threshold


def build_break_even_review_cost(metrics: pd.DataFrame) -> pd.DataFrame:
    n_samples = int(metrics["N"].iloc[0])
    validate_economic_metrics(metrics, n_samples=n_samples)
    indexed = metrics.set_index(["method", "tau"], drop=False)
    records: list[dict[str, Any]] = []
    for tau in TAU_GRID:
        point = indexed.loc[(POINT_THRESHOLD, tau)]
        q50 = indexed.loc[(CQR_Q50_THRESHOLD, tau)]
        cqr = indexed.loc[(CQR_INTERVAL_TRISTATE, tau)]
        review_rate = float(cqr["review_rate"])
        cqr_regret = float(cqr["mean_regret_r0"])
        point_regret = float(point["mean_regret_r0"])
        q50_regret = float(q50["mean_regret_r0"])
        point_ratio = analytic_break_even_ratio(
            point_regret, cqr_regret, review_rate, tau
        )
        q50_ratio = analytic_break_even_ratio(
            q50_regret, cqr_regret, review_rate, tau
        )
        for baseline_regret, ratio, name in (
            (point_regret, point_ratio, "point"),
            (q50_regret, q50_ratio, "q50"),
        ):
            if review_rate > 0.0 and not math.isclose(
                cqr_mean_regret_at_review_ratio(
                    cqr_regret, review_rate, ratio, tau
                ),
                baseline_regret,
                rel_tol=0.0,
                abs_tol=BREAK_EVEN_ABS_TOLERANCE,
            ):
                raise ValueError(f"Break-even substitution failed versus {name}")
        records.append(
            {
                "tau": tau,
                "evaluation_role": EVALUATION_ROLE,
                "N": n_samples,
                "cqr_method": CQR_INTERVAL_TRISTATE,
                "review_n": int(cqr["review_n"]),
                "review_rate": review_rate,
                "cqr_mean_regret_r0": cqr_regret,
                "point_mean_regret_r0": point_regret,
                "q50_mean_regret_r0": q50_regret,
                "break_even_vs_point": point_ratio,
                "break_even_vs_q50": q50_ratio,
            }
        )
    result = pd.DataFrame.from_records(records).loc[:, BREAK_EVEN_COLUMNS]
    validate_break_even_table(result, n_samples=n_samples)
    return result


def validate_break_even_table(frame: pd.DataFrame, *, n_samples: int) -> None:
    if tuple(frame.columns) != BREAK_EVEN_COLUMNS:
        raise ValueError("Stage 5-A1 break-even schema mismatch")
    if tuple(frame["tau"].astype(float)) != TAU_GRID:
        raise ValueError("Stage 5-A1 break-even tau order mismatch")
    if set(frame["evaluation_role"].astype(str)) != {EVALUATION_ROLE}:
        raise PermissionError("Stage 5-A1 break-even role mismatch")
    if set(frame["N"].astype(int)) != {n_samples}:
        raise ValueError("Stage 5-A1 break-even N mismatch")
    if set(frame["cqr_method"].astype(str)) != {CQR_INTERVAL_TRISTATE}:
        raise ValueError("Only CQR tri-state may have break-even review cost")
    for row in frame.itertuples(index=False):
        for field in ("break_even_vs_point", "break_even_vs_q50"):
            ratio = float(getattr(row, field))
            if row.review_rate == 0.0 and not math.isnan(ratio):
                raise ValueError("Zero review rate requires undefined break-even ratio")
            if row.review_rate > 0.0 and not math.isfinite(ratio):
                raise ValueError("Positive review rate requires finite break-even ratio")
    if _forbidden_columns(frame):
        raise ValueError("Winner/ranking/optimal-tau fields are forbidden")


def make_config() -> dict[str, Any]:
    validate_protocol_constants()
    return {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "evaluation_role": EVALUATION_ROLE,
        "N": EXPECTED_N,
        "tau_grid": list(TAU_GRID),
        "reference_tau": REFERENCE_TAU,
        "method_order": list(METHOD_ORDER),
        "normalized_recoverable_value": NORMALIZED_RECOVERABLE_VALUE,
        "clean_cost": "tau",
        "wait_cost": "true_L",
        "oracle_cost": "min(tau,true_L)",
        "review_cost": "min(tau,true_L) + review_cost_ratio*tau",
        "review_regret": "review_cost_ratio*tau",
        "review_benchmark_interpretation": (
            "perfect-review is an optimistic lower bound on cost and upper bound on benefit"
        ),
        "real_human_review_claimed_perfect": False,
        "break_even_formula": "(baseline_mean_regret_r0-cqr_mean_regret_r0)/(review_rate*tau)",
        "break_even_negative_values_truncated": False,
        "review_cost_ratio_scanned": False,
        "source_stage4a_actions": project_relative(STAGE4A_ACTIONS_INPUT),
        "source_stage4a_action_columns": list(stage4a.ACTION_COLUMNS),
        "sample_regret_columns": list(SAMPLE_REGRET_COLUMNS),
        "metric_columns": list(METRIC_COLUMNS),
        "break_even_columns": list(BREAK_EVEN_COLUMNS),
        "expected_formal_rows": EXPECTED_FORMAL_ROWS,
    }


def make_provenance() -> dict[str, Any]:
    return {
        **make_config(),
        "normalized_economics": True,
        "currency_used": False,
        "gansu_price_used_in_core_evaluation": False,
        "actual_station_scale_claimed": False,
        "review_is_perfect_resolution_benchmark": True,
        "review_cost_ratio_selected_from_results": False,
        "tau_selected_from_economic_results": False,
        "random_test_accessed": False,
        "sealed_final_dates_accessed": False,
        "training_performed": False,
        "inference_performed": False,
        "conformal_recalibration_performed": False,
        "risk_score_development_performed": False,
        "threshold_optimization_performed": False,
        "method_selection_performed": False,
        "economic_winner_declared": False,
    }


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def validate_formal_output_path(output_dir: Path) -> Path:
    candidate = _resolved(output_dir)
    if candidate != _resolved(OUTPUT_DIR):
        raise PermissionError(f"Unauthorized Stage 5-A1 output directory: {candidate}")
    return candidate


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_outputs(
    output_dir: Path,
    sample_regrets: pd.DataFrame,
    metrics: pd.DataFrame,
    break_even: pd.DataFrame,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    ensure_output_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_regrets.to_csv(
        output_dir / "economic_sample_regrets.csv", index=False, mode="x"
    )
    metrics.to_csv(output_dir / "economic_metrics.csv", index=False, mode="x")
    break_even.to_csv(
        output_dir / "break_even_review_cost.csv", index=False, mode="x"
    )
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "provenance.json", provenance)


def run(
    protocol: str = PROTOCOL,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the formal Stage 5-A1 protocol only when explicitly invoked."""
    validate_protocol(protocol)
    output_dir = validate_formal_output_path(output_dir)
    ensure_output_available(output_dir)
    actions = load_stage4_actions()
    sample_regrets = build_sample_regrets(actions)
    metrics = build_economic_metrics(sample_regrets)
    break_even = build_break_even_review_cost(metrics)
    config = make_config()
    provenance = make_provenance()
    write_outputs(output_dir, sample_regrets, metrics, break_even, config, provenance)
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
