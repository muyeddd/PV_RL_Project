"""Deterministic, label-free rules for PV cleaning recommendations.

The rule engine deliberately reads only the approved prediction columns. Ground
truth, realized errors, and interval-coverage columns are never consulted.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

import numpy as np
import pandas as pd


class Decision(str, Enum):
    """Available cleaning decisions."""

    CLEAN = "CLEAN"
    WAIT = "WAIT"
    MONITOR = "MONITOR"
    REVIEW = "REVIEW"


class ReasonCode(str, Enum):
    """Reason codes emitted by the rule engine."""

    REVIEW_INVALID_INPUT = "REVIEW_INVALID_INPUT"
    MONITOR_LOW_IRRADIANCE = "MONITOR_LOW_IRRADIANCE"
    REVIEW_HIGH_UNCERTAINTY = "REVIEW_HIGH_UNCERTAINTY"
    CLEAN_CONFIDENT_LOSS = "CLEAN_CONFIDENT_LOSS"
    WAIT_CONFIDENT_LOW_LOSS = "WAIT_CONFIDENT_LOW_LOSS"
    MONITOR_THRESHOLD_CROSSING = "MONITOR_THRESHOLD_CROSSING"


REQUIRED_INPUT_COLUMNS = (
    "filename",
    "pred_L",
    "pred_std",
    "irradiance",
    "pred_l_mondrian_std_mc_lower",
    "pred_l_mondrian_std_mc_upper",
    "pred_l_mondrian_std_mc_width",
)

REQUIRED_CONFIG_FIELDS = (
    "schema_version",
    "config_version",
    "interval_method",
    "confidence",
    "loss_threshold",
    "min_irradiance",
    "max_interval_width",
)

THRESHOLD_FIELDS = (
    "loss_threshold",
    "min_irradiance",
    "max_interval_width",
)

SUPPORTED_INTERVAL_METHOD = "pred_l_mondrian_std_mc"


def _validated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping containing explicit rule settings")

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    null_thresholds = [
        field
        for field in THRESHOLD_FIELDS
        if config[field] is None or pd.isna(config[field])
    ]
    if null_thresholds:
        raise ValueError(
            "Rule thresholds must be explicitly configured; null values are not "
            f"allowed for: {', '.join(null_thresholds)}. No threshold defaults are assumed."
        )

    validated = dict(config)
    for field in (*THRESHOLD_FIELDS, "confidence"):
        value = validated[field]
        if isinstance(value, bool):
            raise ValueError(f"Config field '{field}' must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{field}' must be a finite number") from exc
        if not np.isfinite(value):
            raise ValueError(f"Config field '{field}' must be a finite number")
        validated[field] = value

    if not 0.0 <= validated["loss_threshold"] <= 1.0:
        raise ValueError("loss_threshold must be within [0, 1]")
    if validated["min_irradiance"] < 0.0:
        raise ValueError("min_irradiance must be non-negative")
    if not 0.0 <= validated["max_interval_width"] <= 1.0:
        raise ValueError("max_interval_width must be within [0, 1]")
    if not 0.0 < validated["confidence"] <= 1.0:
        raise ValueError("confidence must be within (0, 1]")
    if validated["interval_method"] != SUPPORTED_INTERVAL_METHOD:
        raise ValueError(
            "interval_method must be "
            f"'{SUPPORTED_INTERVAL_METHOD}' for the approved input columns"
        )
    if validated["config_version"] is None or not str(validated["config_version"]).strip():
        raise ValueError("config_version must be a non-empty value")

    return validated


def _numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric view without mutating the caller's DataFrame."""

    return pd.to_numeric(df[name], errors="coerce").astype(float)


def apply_cleaning_rules(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply rule-based cleaning decisions in a fixed priority order.

    Parameters
    ----------
    df:
        Per-observation prediction data. Only ``REQUIRED_INPUT_COLUMNS`` are
        read; label and evaluation columns are ignored even if present.
    config:
        Rule configuration with explicit, non-null thresholds.

    Returns
    -------
    pandas.DataFrame
        One decision row per input row, preserving input order. ``clean_flag``
        is 1 for CLEAN, 0 for WAIT, and missing for MONITOR/REVIEW.

    Notes
    -----
    ``margin_to_threshold`` is point prediction minus the loss threshold.
    ``priority_score`` is the non-economic lower-bound margin for CLEAN rows;
    it is zero for other valid rows and missing for invalid rows.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    settings = _validated_config(config)
    missing_columns = [name for name in REQUIRED_INPUT_COLUMNS if name not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required input columns: {', '.join(missing_columns)}")

    filename = df["filename"].astype("string")
    pred_loss = _numeric_column(df, "pred_L")
    pred_std = _numeric_column(df, "pred_std")
    irradiance = _numeric_column(df, "irradiance")
    interval_lower = _numeric_column(df, "pred_l_mondrian_std_mc_lower")
    interval_upper = _numeric_column(df, "pred_l_mondrian_std_mc_upper")
    interval_width = _numeric_column(df, "pred_l_mondrian_std_mc_width")

    numeric = np.column_stack(
        [
            pred_loss.to_numpy(),
            pred_std.to_numpy(),
            irradiance.to_numpy(),
            interval_lower.to_numpy(),
            interval_upper.to_numpy(),
            interval_width.to_numpy(),
        ]
    )
    finite = np.isfinite(numeric).all(axis=1)
    filename_valid = filename.notna().to_numpy() & filename.str.strip().ne("").fillna(False).to_numpy()
    width_consistent = np.isclose(
        interval_width.to_numpy(),
        interval_upper.to_numpy() - interval_lower.to_numpy(),
        rtol=1e-7,
        atol=1e-9,
        equal_nan=False,
    )

    valid_domain = (
        pred_loss.between(0.0, 1.0).to_numpy()
        & pred_std.ge(0.0).to_numpy()
        & irradiance.ge(0.0).to_numpy()
        & interval_lower.between(0.0, 1.0).to_numpy()
        & interval_upper.between(0.0, 1.0).to_numpy()
        & interval_width.between(0.0, 1.0).to_numpy()
        & interval_lower.le(interval_upper).to_numpy()
        & width_consistent
    )
    invalid = ~(finite & filename_valid & valid_domain)

    row_count = len(df)
    decisions = np.full(row_count, Decision.REVIEW.value, dtype=object)
    reasons = np.full(row_count, ReasonCode.REVIEW_INVALID_INPUT.value, dtype=object)
    reliable = np.zeros(row_count, dtype=bool)
    clean_flag = pd.array([pd.NA] * row_count, dtype="Int64")

    remaining = ~invalid

    low_irradiance = remaining & irradiance.lt(settings["min_irradiance"]).to_numpy()
    decisions[low_irradiance] = Decision.MONITOR.value
    reasons[low_irradiance] = ReasonCode.MONITOR_LOW_IRRADIANCE.value
    remaining &= ~low_irradiance

    high_uncertainty = remaining & interval_width.gt(settings["max_interval_width"]).to_numpy()
    decisions[high_uncertainty] = Decision.REVIEW.value
    reasons[high_uncertainty] = ReasonCode.REVIEW_HIGH_UNCERTAINTY.value
    remaining &= ~high_uncertainty

    clean = remaining & interval_lower.ge(settings["loss_threshold"]).to_numpy()
    decisions[clean] = Decision.CLEAN.value
    reasons[clean] = ReasonCode.CLEAN_CONFIDENT_LOSS.value
    reliable[clean] = True
    clean_flag[clean] = 1
    remaining &= ~clean

    wait = remaining & interval_upper.lt(settings["loss_threshold"]).to_numpy()
    decisions[wait] = Decision.WAIT.value
    reasons[wait] = ReasonCode.WAIT_CONFIDENT_LOW_LOSS.value
    reliable[wait] = True
    clean_flag[wait] = 0
    remaining &= ~wait

    decisions[remaining] = Decision.MONITOR.value
    reasons[remaining] = ReasonCode.MONITOR_THRESHOLD_CROSSING.value
    reliable[remaining] = True

    margin = pred_loss.to_numpy() - settings["loss_threshold"]
    margin[invalid] = np.nan
    priority = np.zeros(row_count, dtype=float)
    priority[clean] = np.maximum(
        interval_lower.to_numpy()[clean] - settings["loss_threshold"],
        0.0,
    )
    priority[invalid] = np.nan

    return pd.DataFrame(
        {
            "filename": filename.reset_index(drop=True),
            "pred_loss": pred_loss.reset_index(drop=True),
            "pred_std": pred_std.reset_index(drop=True),
            "irradiance": irradiance.reset_index(drop=True),
            "interval_lower": interval_lower.reset_index(drop=True),
            "interval_upper": interval_upper.reset_index(drop=True),
            "interval_width": interval_width.reset_index(drop=True),
            "decision": decisions,
            "clean_flag": clean_flag,
            "reason_code": reasons,
            "is_reliable": reliable,
            "margin_to_threshold": margin,
            "priority_score": priority,
            "config_version": str(settings["config_version"]),
        }
    )
