#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0C-3B
Leave-One-Date-Out validation of the Paper1 conditional perception emulator.

Scientific goal
---------------
Validate, before any WAPP perception trajectory is generated, whether a
date-balanced conditional resampling emulator can reproduce Paper1 perception
statistics on an entirely held-out Paper1 date.

Authorized data
---------------
DECISION_DEVELOPMENT only.
RANDOM_TEST and SEALED_DATES are forbidden.

Emulator v1 validated here
--------------------------
For each held-out target row with true loss L*:

1) Source pool = all OTHER Paper1 dates.
2) Candidate source rows satisfy |true_L_source - L*| <= 0.01.
3) Support gate per target:
       >= 20 candidate source rows
       AND >= 3 distinct source dates.
   No fallback / extrapolation is allowed in this validation.
4) Date-balanced source selection:
       choose one eligible source date uniformly;
       within that date, sample one candidate row with a Gaussian
       distance weight exp(-0.5*(d/h)^2), h=0.005.
5) Transport the JOINT source quantile residual tuple:
       delta = L* - true_L_source
       q05* = clip(q05_source + delta, 0, 1)
       q50* = clip(q50_source + delta, 0, 1)
       q95* = clip(q95_source + delta, 0, 1)
6) Rebuild the frozen conformal interval using the Paper1 qhat:
       lower* = clip(q05* - qhat, 0, 1)
       upper* = clip(q95* + qhat, 0, 1)
       width* = upper* - lower*

This preserves q05/q50/q95 joint structure and explicitly preserves
lower-bound clipping semantics.

Validation strategy
-------------------
- 12 leave-one-date-out folds.
- 50 Monte Carlo replicates per fold.
- Compare actual held-out-date perception metrics with generated metrics.
- Main gates are applied to DATE-MACRO averages so dense source/target dates
  do not dominate the validation conclusion.
- A WAPP-relevant low-loss subset is evaluated separately using the observed
  P2-0C-1B maximum L + 0.01 margin.

This stage still does NOT generate a WAPP q50/width trajectory.

Outputs
-------
lodo_date_metrics.csv
lodo_rep_metrics.csv
lodo_target_support.csv
audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 1844
EXPECTED_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_DATES = 12
EXPECTED_WAPP_VALID = 729

QHAT = 0.004862844288256299

LOCAL_RADIUS = 0.01
LOCAL_MIN_SAMPLES = 20
LOCAL_MIN_DATES = 3
KERNEL_BANDWIDTH = 0.005

MC_REPS = 50
BASE_SEED = 20260905
WAPP_MARGIN = 0.01

# Predeclared gates for date-macro metrics.
GATE_SUPPORT_FRACTION = 0.99

GLOBAL_GATES = {
    "bias_abs_diff": 0.010,
    "mae_abs_diff": 0.010,
    "width_median_abs_diff": 0.020,
    "coverage_abs_diff": 0.030,
    "lower_clipped_abs_diff": 0.050,
    "rho_width_abs_error_abs_diff": 0.100,
}

WAPP_RELEVANT_GATES = {
    "bias_abs_diff": 0.010,
    "mae_abs_diff": 0.010,
    "width_median_abs_diff": 0.020,
    "coverage_abs_diff": 0.040,
    "lower_clipped_abs_diff": 0.070,
    "rho_width_abs_error_abs_diff": 0.120,
}


def qstats(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "max": float(x.max()),
    }


def spearman(a, b) -> float:
    aa = pd.to_numeric(pd.Series(a), errors="coerce")
    bb = pd.to_numeric(pd.Series(b), errors="coerce")
    ok = np.isfinite(aa) & np.isfinite(bb)
    if int(ok.sum()) < 3:
        return float("nan")
    if aa[ok].nunique() < 2 or bb[ok].nunique() < 2:
        return float("nan")
    return float(aa[ok].corr(bb[ok], method="spearman"))


def load_paper1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "sample_id", "date", "role", "true_L",
        "q05", "q50", "q95", "lower", "upper", "width",
        "covered", "lower_clipped", "upper_clipped",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing Paper1 columns: {sorted(missing)}")

    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} rows, found {len(df)}")

    roles = set(df["role"].astype(str))
    if roles != {EXPECTED_ROLE}:
        raise PermissionError(
            f"Only {EXPECTED_ROLE} is authorized; found {sorted(roles)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if df["date"].nunique() != EXPECTED_DATES:
        raise RuntimeError(
            f"Expected {EXPECTED_DATES} dates, found {df['date'].nunique()}"
        )

    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise RuntimeError("sample_id must be unique and non-null.")

    num_cols = ["true_L", "q05", "q50", "q95", "lower", "upper", "width"]
    num = df[num_cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(num.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite Paper1 numeric values.")

    if (num["true_L"] < 0).any() or (num["true_L"] > 1).any():
        raise RuntimeError("true_L outside [0,1].")

    if not (
        (num["q05"] <= num["q50"]) & (num["q50"] <= num["q95"])
    ).all():
        raise RuntimeError("q05/q50/q95 ordering violated.")

    # Validate the frozen CQR reconstruction.
    expected_lower = np.clip(
        num["q05"].to_numpy(dtype=float) - QHAT, 0.0, 1.0
    )
    expected_upper = np.clip(
        num["q95"].to_numpy(dtype=float) + QHAT, 0.0, 1.0
    )
    if not np.allclose(
        expected_lower,
        num["lower"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("Frozen qhat does not reproduce Paper1 lower bound.")
    if not np.allclose(
        expected_upper,
        num["upper"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("Frozen qhat does not reproduce Paper1 upper bound.")
    if not np.allclose(
        expected_upper - expected_lower,
        num["width"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("Frozen qhat does not reproduce Paper1 width.")

    covered_expected = (
        (num["true_L"] >= num["lower"])
        & (num["true_L"] <= num["upper"])
    )
    if not np.array_equal(
        covered_expected.to_numpy(dtype=bool),
        df["covered"].astype(bool).to_numpy(),
    ):
        raise RuntimeError("covered flag inconsistent with interval.")

    return df.reset_index(drop=True)


def load_wapp(path: Path) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"L_power_proxy", "bridge_valid", "power_bridge_model"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing WAPP bridge columns: {sorted(missing)}")

    valid = df["bridge_valid"].fillna(False).astype(bool)
    if int(valid.sum()) != EXPECTED_WAPP_VALID:
        raise RuntimeError(
            f"Expected {EXPECTED_WAPP_VALID} valid WAPP days, "
            f"found {int(valid.sum())}"
        )

    models = set(df.loc[valid, "power_bridge_model"].astype(str))
    if models != {"COMMON_TEMPERATURE_PVWATTS_RATIO"}:
        raise RuntimeError(f"Unexpected WAPP bridge model(s): {sorted(models)}")

    l = pd.to_numeric(
        df.loc[valid, "L_power_proxy"], errors="raise"
    )
    if not np.isfinite(l).all():
        raise RuntimeError("Non-finite WAPP L_power_proxy.")

    return df.loc[valid].copy(), float(l.max())


def metric_dict(true_l, q50, lower, upper) -> dict:
    true_l = np.asarray(true_l, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    width = upper - lower
    err = q50 - true_l
    abs_err = np.abs(err)
    covered = (true_l >= lower) & (true_l <= upper)
    lower_clip = np.isclose(lower, 0.0, rtol=0.0, atol=1e-12)

    return {
        "N": int(len(true_l)),
        "bias": float(np.mean(err)),
        "mae": float(np.mean(abs_err)),
        "width_mean": float(np.mean(width)),
        "width_median": float(np.median(width)),
        "coverage": float(np.mean(covered)),
        "lower_clipped_fraction": float(np.mean(lower_clip)),
        "rho_width_abs_error": spearman(width, abs_err),
    }


def weighted_pick(
    rng: np.random.Generator,
    candidate_indices: np.ndarray,
    distances: np.ndarray,
) -> int:
    weights = np.exp(
        -0.5 * (distances / KERNEL_BANDWIDTH) ** 2
    )
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Invalid kernel weights.")
    weights = weights / total
    return int(rng.choice(candidate_indices, p=weights))


def build_target_candidate_map(
    source: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    src_l = pd.to_numeric(
        source["true_L"], errors="raise"
    ).to_numpy(dtype=float)

    source_dates = source["date"].to_numpy(dtype=str)
    target_rows = []
    candidate_maps = []

    for target_pos, (_, row) in enumerate(targets.iterrows()):
        lstar = float(row["true_L"])
        dist = np.abs(src_l - lstar)
        m = dist <= LOCAL_RADIUS

        idx = np.flatnonzero(m)
        dates = np.unique(source_dates[idx])

        support_ok = (
            len(idx) >= LOCAL_MIN_SAMPLES
            and len(dates) >= LOCAL_MIN_DATES
        )

        by_date = {}
        if support_ok:
            for d in dates:
                date_idx = idx[source_dates[idx] == d]
                by_date[str(d)] = {
                    "indices": date_idx,
                    "distances": dist[date_idx],
                }

        candidate_maps.append(
            {
                "support_ok": support_ok,
                "by_date": by_date,
            }
        )
        target_rows.append(
            {
                "target_position": target_pos,
                "target_sample_id": row["sample_id"],
                "target_date": row["date"],
                "target_true_L": lstar,
                "candidate_samples": int(len(idx)),
                "candidate_dates": int(len(dates)),
                "support_ok": bool(support_ok),
            }
        )

    return candidate_maps, pd.DataFrame(target_rows)


def generate_one_rep(
    rng: np.random.Generator,
    source: pd.DataFrame,
    targets: pd.DataFrame,
    candidate_maps: list[dict],
) -> dict[str, np.ndarray]:
    n = len(targets)

    q05_gen = np.empty(n, dtype=float)
    q50_gen = np.empty(n, dtype=float)
    q95_gen = np.empty(n, dtype=float)
    lower_gen = np.empty(n, dtype=float)
    upper_gen = np.empty(n, dtype=float)
    source_date_chosen = np.empty(n, dtype=object)

    source_q05 = pd.to_numeric(source["q05"], errors="raise").to_numpy(dtype=float)
    source_q50 = pd.to_numeric(source["q50"], errors="raise").to_numpy(dtype=float)
    source_q95 = pd.to_numeric(source["q95"], errors="raise").to_numpy(dtype=float)
    source_l = pd.to_numeric(source["true_L"], errors="raise").to_numpy(dtype=float)

    target_l = pd.to_numeric(
        targets["true_L"], errors="raise"
    ).to_numpy(dtype=float)

    for i, cmap in enumerate(candidate_maps):
        if not cmap["support_ok"]:
            raise RuntimeError(
                "LODO target lacks predeclared local support; no fallback allowed."
            )

        eligible_dates = sorted(cmap["by_date"].keys())
        chosen_date = str(rng.choice(eligible_dates))
        block = cmap["by_date"][chosen_date]

        chosen_src = weighted_pick(
            rng,
            np.asarray(block["indices"], dtype=int),
            np.asarray(block["distances"], dtype=float),
        )

        delta = target_l[i] - source_l[chosen_src]

        q05 = np.clip(source_q05[chosen_src] + delta, 0.0, 1.0)
        q50 = np.clip(source_q50[chosen_src] + delta, 0.0, 1.0)
        q95 = np.clip(source_q95[chosen_src] + delta, 0.0, 1.0)

        # Adding the same delta preserves ordering before clipping; monotone
        # clipping preserves it afterward.
        if not (q05 <= q50 <= q95):
            raise RuntimeError("Generated quantile ordering violated.")

        lower = np.clip(q05 - QHAT, 0.0, 1.0)
        upper = np.clip(q95 + QHAT, 0.0, 1.0)

        q05_gen[i] = q05
        q50_gen[i] = q50
        q95_gen[i] = q95
        lower_gen[i] = lower
        upper_gen[i] = upper
        source_date_chosen[i] = chosen_date

    return {
        "q05": q05_gen,
        "q50": q50_gen,
        "q95": q95_gen,
        "lower": lower_gen,
        "upper": upper_gen,
        "source_date": source_date_chosen,
    }


def macro_metric(date_metrics: pd.DataFrame, prefix: str, metric: str) -> float:
    col = f"{prefix}_{metric}"
    x = pd.to_numeric(date_metrics[col], errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(x.mean())


def gate_comparison(
    actual: dict,
    generated: dict,
    gates: dict,
) -> dict:
    out = {}
    all_pass = True
    for metric, tol in gates.items():
        if metric == "bias_abs_diff":
            name = "bias"
        elif metric == "mae_abs_diff":
            name = "mae"
        elif metric == "width_median_abs_diff":
            name = "width_median"
        elif metric == "coverage_abs_diff":
            name = "coverage"
        elif metric == "lower_clipped_abs_diff":
            name = "lower_clipped_fraction"
        elif metric == "rho_width_abs_error_abs_diff":
            name = "rho_width_abs_error"
        else:
            raise KeyError(metric)

        a = float(actual[name])
        g = float(generated[name])
        diff = abs(g - a)
        passed = bool(np.isfinite(diff) and diff <= tol)
        out[name] = {
            "actual_macro": a,
            "generated_macro": g,
            "absolute_difference": diff,
            "tolerance": tol,
            "pass": passed,
        }
        all_pass &= passed
    out["all_pass"] = bool(all_pass)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-0C-3B leave-one-date-out perception emulator validation."
    )
    p.add_argument("--paper1-dev-cqr", required=True, type=Path)
    p.add_argument("--wapp-power-bridge", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0c_3b_lodo_emulator_validation_v1"
        ),
    )
    args = p.parse_args()

    paper1_path = args.paper1_dev_cqr.expanduser().resolve()
    wapp_path = args.wapp_power_bridge.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for path in (paper1_path, wapp_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/8] Load authorized Paper1 DECISION_DEVELOPMENT CQR asset")
    p1 = load_paper1(paper1_path)

    print("[2/8] Load WAPP query range for relevance audit")
    _, wapp_max = load_wapp(wapp_path)
    relevance_hi = min(1.0, wapp_max + WAPP_MARGIN)

    print("[3/8] Build 12 leave-one-date-out support maps")
    support_tables = []
    fold_data = {}

    for heldout_date in sorted(p1["date"].unique()):
        targets = p1[p1["date"].eq(heldout_date)].copy().reset_index(drop=True)
        source = p1[~p1["date"].eq(heldout_date)].copy().reset_index(drop=True)

        cmap, support = build_target_candidate_map(source, targets)
        support["wapp_relevant"] = support["target_true_L"] <= relevance_hi
        support_tables.append(support)

        fold_data[heldout_date] = {
            "targets": targets,
            "source": source,
            "candidate_maps": cmap,
        }

    support_all = pd.concat(support_tables, ignore_index=True)
    support_fraction = float(support_all["support_ok"].mean())

    print("[4/8] Run Monte Carlo LODO emulator validation")
    rep_rows = []
    date_rows = []

    for fold_idx, heldout_date in enumerate(sorted(fold_data.keys())):
        targets_full = fold_data[heldout_date]["targets"]
        source = fold_data[heldout_date]["source"]
        cmaps_full = fold_data[heldout_date]["candidate_maps"]

        # IMPORTANT: the protocol gate is support_fraction >= 0.99, not 1.00.
        # Unsupported targets receive no fallback and are excluded from the
        # generated-vs-actual distribution comparison. They remain explicit
        # failures in the support audit below.
        supported_mask = np.array(
            [c["support_ok"] for c in cmaps_full], dtype=bool
        )

        if int(supported_mask.sum()) < 3:
            raise RuntimeError(
                f"Held-out date {heldout_date} has fewer than 3 supported "
                "targets; fold-level distribution validation is impossible."
            )

        targets = (
            targets_full.loc[supported_mask]
            .copy()
            .reset_index(drop=True)
        )
        cmaps = [
            c for c, ok in zip(cmaps_full, supported_mask) if ok
        ]

        true_l = pd.to_numeric(
            targets["true_L"], errors="raise"
        ).to_numpy(dtype=float)

        actual_all = metric_dict(
            true_l,
            pd.to_numeric(targets["q50"], errors="raise"),
            pd.to_numeric(targets["lower"], errors="raise"),
            pd.to_numeric(targets["upper"], errors="raise"),
        )

        relevant_mask = true_l <= relevance_hi
        actual_rel = metric_dict(
            true_l[relevant_mask],
            pd.to_numeric(targets.loc[relevant_mask, "q50"], errors="raise"),
            pd.to_numeric(targets.loc[relevant_mask, "lower"], errors="raise"),
            pd.to_numeric(targets.loc[relevant_mask, "upper"], errors="raise"),
        ) if relevant_mask.sum() >= 3 else None

        generated_metrics_all = []
        generated_metrics_rel = []

        for rep in range(MC_REPS):
            rng = np.random.default_rng(
                BASE_SEED + fold_idx * 10000 + rep
            )
            gen = generate_one_rep(rng, source, targets, cmaps)

            gm_all = metric_dict(
                true_l,
                gen["q50"],
                gen["lower"],
                gen["upper"],
            )
            generated_metrics_all.append(gm_all)

            gm_rel = None
            if relevant_mask.sum() >= 3:
                gm_rel = metric_dict(
                    true_l[relevant_mask],
                    gen["q50"][relevant_mask],
                    gen["lower"][relevant_mask],
                    gen["upper"][relevant_mask],
                )
                generated_metrics_rel.append(gm_rel)

            row = {
                "heldout_date": heldout_date,
                "rep": rep,
                "scope": "ALL_SUPPORTED",
                **gm_all,
            }
            rep_rows.append(row)

            if gm_rel is not None:
                rep_rows.append(
                    {
                        "heldout_date": heldout_date,
                        "rep": rep,
                        "scope": "WAPP_RELEVANT_SUPPORTED",
                        **gm_rel,
                    }
                )

        def mean_generated(metric: str, rows: list[dict]) -> float:
            vals = [
                r[metric] for r in rows
                if np.isfinite(r[metric])
            ]
            return float(np.mean(vals)) if vals else np.nan

        full_true_l = pd.to_numeric(
            targets_full["true_L"], errors="raise"
        ).to_numpy(dtype=float)
        full_rel_mask = full_true_l <= relevance_hi
        full_rel_supported = supported_mask & full_rel_mask

        date_row = {
            "heldout_date": heldout_date,
            "N_full": int(len(targets_full)),
            "N_supported": int(supported_mask.sum()),
            "N_unsupported": int((~supported_mask).sum()),
            "support_fraction": float(supported_mask.mean()),
            "N_wapp_relevant_full": int(full_rel_mask.sum()),
            "N_wapp_relevant_supported": int(full_rel_supported.sum()),
            "wapp_relevant_support_fraction": (
                float(full_rel_supported.sum() / full_rel_mask.sum())
                if int(full_rel_mask.sum()) > 0 else np.nan
            ),
            "unsupported_true_L_min": (
                float(full_true_l[~supported_mask].min())
                if (~supported_mask).any() else np.nan
            ),
            "unsupported_true_L_max": (
                float(full_true_l[~supported_mask].max())
                if (~supported_mask).any() else np.nan
            ),
        }
        for metric in [
            "bias", "mae", "width_median", "coverage",
            "lower_clipped_fraction", "rho_width_abs_error",
        ]:
            date_row[f"actual_{metric}"] = actual_all[metric]
            date_row[f"generated_{metric}"] = mean_generated(
                metric, generated_metrics_all
            )
            date_row[f"absdiff_{metric}"] = abs(
                date_row[f"generated_{metric}"]
                - date_row[f"actual_{metric}"]
            )

            if actual_rel is not None:
                date_row[f"actual_rel_{metric}"] = actual_rel[metric]
                date_row[f"generated_rel_{metric}"] = mean_generated(
                    metric, generated_metrics_rel
                )
                date_row[f"absdiff_rel_{metric}"] = abs(
                    date_row[f"generated_rel_{metric}"]
                    - date_row[f"actual_rel_{metric}"]
                )
            else:
                date_row[f"actual_rel_{metric}"] = np.nan
                date_row[f"generated_rel_{metric}"] = np.nan
                date_row[f"absdiff_rel_{metric}"] = np.nan

        date_rows.append(date_row)

    date_metrics = pd.DataFrame(date_rows)
    rep_metrics = pd.DataFrame(rep_rows)

    print("[5/8] Build date-macro validation metrics")
    metric_names = [
        "bias", "mae", "width_median", "coverage",
        "lower_clipped_fraction", "rho_width_abs_error",
    ]

    actual_macro = {}
    generated_macro = {}
    actual_macro_rel = {}
    generated_macro_rel = {}

    for metric in metric_names:
        actual_macro[metric] = float(
            pd.to_numeric(
                date_metrics[f"actual_{metric}"], errors="coerce"
            ).dropna().mean()
        )
        generated_macro[metric] = float(
            pd.to_numeric(
                date_metrics[f"generated_{metric}"], errors="coerce"
            ).dropna().mean()
        )

        rel_actual = pd.to_numeric(
            date_metrics[f"actual_rel_{metric}"], errors="coerce"
        ).dropna()
        rel_gen = pd.to_numeric(
            date_metrics[f"generated_rel_{metric}"], errors="coerce"
        ).dropna()

        actual_macro_rel[metric] = (
            float(rel_actual.mean()) if len(rel_actual) else np.nan
        )
        generated_macro_rel[metric] = (
            float(rel_gen.mean()) if len(rel_gen) else np.nan
        )

    global_gate_result = gate_comparison(
        actual_macro, generated_macro, GLOBAL_GATES
    )
    relevant_gate_result = gate_comparison(
        actual_macro_rel,
        generated_macro_rel,
        WAPP_RELEVANT_GATES,
    )

    print("[6/8] Audit support without fallback")
    support_ok = support_all["support_ok"].astype(bool)
    support_fraction = float(support_ok.mean())
    support_gate_pass = bool(
        support_fraction >= GATE_SUPPORT_FRACTION
    )

    relevant_support = support_all[
        support_all["wapp_relevant"].astype(bool)
    ].copy()
    wapp_relevant_support_fraction = float(
        relevant_support["support_ok"].astype(bool).mean()
    )
    wapp_support_gate_pass = bool(
        wapp_relevant_support_fraction >= GATE_SUPPORT_FRACTION
    )
    unsupported = support_all[~support_ok].copy()

    print("[7/8] Evaluate predeclared LODO scientific gates")
    all_primary = bool(
        support_gate_pass
        and wapp_support_gate_pass
        and global_gate_result["all_pass"]
        and relevant_gate_result["all_pass"]
    )

    summary = {
        "stage": "P2-0C-3B",
        "validation_only": True,
        "revision": (
            "unsupported targets are recorded and excluded from distribution "
            "comparison; no fallback or threshold relaxation"
        ),
        "emulator_design_validated": False,
        "wapp_perception_trajectory_generated": False,
        "paper1_asset": {
            "role": EXPECTED_ROLE,
            "rows": int(len(p1)),
            "dates": int(p1["date"].nunique()),
            "random_test_used": False,
            "sealed_dates_used": False,
            "qhat": QHAT,
        },
        "emulator_v1": {
            "conditioning_variables": ["true_L"],
            "local_radius_abs_L": LOCAL_RADIUS,
            "local_min_samples": LOCAL_MIN_SAMPLES,
            "local_min_dates": LOCAL_MIN_DATES,
            "fallback_for_unsupported_targets": False,
            "source_date_selection": "UNIFORM_OVER_ELIGIBLE_SOURCE_DATES",
            "within_date_selection": (
                "GAUSSIAN_DISTANCE_WEIGHTED_WITH_BANDWIDTH_0.005"
            ),
            "joint_transport": [
                "q05-true_L",
                "q50-true_L",
                "q95-true_L",
            ],
            "quantile_prediction_projection": "[0,1]",
            "conformal_interval_rebuilt_with_frozen_qhat": True,
            "source_frame_tuple_sampled_jointly": True,
            "wapp_irradiance_used": False,
        },
        "lodo_protocol": {
            "folds": int(p1["date"].nunique()),
            "mc_reps_per_fold": MC_REPS,
            "base_seed": BASE_SEED,
            "wapp_relevance_upper_true_L": relevance_hi,
            "metrics_compared_on_same_supported_subset": True,
        },
        "support": {
            "target_rows": int(len(support_all)),
            "supported_target_rows": int(support_ok.sum()),
            "unsupported_target_rows": int((~support_ok).sum()),
            "supported_fraction": support_fraction,
            "gate": GATE_SUPPORT_FRACTION,
            "global_support_pass": support_gate_pass,
            "wapp_relevant_target_rows": int(len(relevant_support)),
            "wapp_relevant_supported_rows": int(
                relevant_support["support_ok"].astype(bool).sum()
            ),
            "wapp_relevant_supported_fraction": wapp_relevant_support_fraction,
            "wapp_relevant_support_pass": wapp_support_gate_pass,
            "candidate_sample_count_distribution": qstats(
                support_all["candidate_samples"]
            ),
            "candidate_date_count_distribution": qstats(
                support_all["candidate_dates"]
            ),
            "unsupported_true_L_distribution": qstats(
                unsupported["target_true_L"]
            ),
            "per_date_support_fraction_distribution": qstats(
                date_metrics["support_fraction"]
            ),
            "per_date_wapp_relevant_support_fraction_distribution": qstats(
                date_metrics["wapp_relevant_support_fraction"]
            ),
        },
        "date_macro_global": {
            "actual": actual_macro,
            "generated": generated_macro,
            "gate_comparison": global_gate_result,
        },
        "date_macro_wapp_relevant": {
            "actual": actual_macro_rel,
            "generated": generated_macro_rel,
            "gate_comparison": relevant_gate_result,
        },
        "date_level_discrepancy_diagnostics": {
            "mae_absdiff_distribution": qstats(
                date_metrics["absdiff_mae"]
            ),
            "width_median_absdiff_distribution": qstats(
                date_metrics["absdiff_width_median"]
            ),
            "coverage_absdiff_distribution": qstats(
                date_metrics["absdiff_coverage"]
            ),
            "rho_absdiff_distribution": qstats(
                date_metrics["absdiff_rho_width_abs_error"]
            ),
            "wapp_relevant_mae_absdiff_distribution": qstats(
                date_metrics["absdiff_rel_mae"]
            ),
            "wapp_relevant_width_absdiff_distribution": qstats(
                date_metrics["absdiff_rel_width_median"]
            ),
        },
        "gates": {
            "global_support_pass": support_gate_pass,
            "wapp_relevant_support_pass": wapp_support_gate_pass,
            "global_distributional_metrics_pass": bool(
                global_gate_result["all_pass"]
            ),
            "wapp_relevant_distributional_metrics_pass": bool(
                relevant_gate_result["all_pass"]
            ),
            "all_primary_gates_pass": all_primary,
        },
        "next_step_if_pass": (
            "Freeze emulator-v1 design, then generate WAPP perception "
            "trajectories with independent seeds and audit marginal/joint "
            "reproduction before any PPO training."
        ),
        "next_step_if_fail": (
            "Do not widen radius or add fallback automatically. Diagnose "
            "unsupported L regimes, date regime shift, boundary transport, "
            "or source-date balancing."
        ),
        "limitations": [
            "LODO validates conditional distribution transport across Paper1 dates, not cross-domain image generalization.",
            "The emulator remains a stochastic surrogate for Paper1 perception errors, not a new image model.",
            "Minute-scale Paper1 temporal dependence is intentionally not transferred to daily WAPP trajectories.",
            "A separate daily-persistence sensitivity and within-q50 width-shuffle RL ablation remain mandatory.",
        ],
    }

    print("[8/8] Write validation outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    date_metrics.to_csv(
        out_dir / "lodo_date_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rep_metrics.to_csv(
        out_dir / "lodo_rep_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    support_all.to_csv(
        out_dir / "lodo_target_support.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "lodo_date_metrics.csv")
    print(out_dir / "lodo_rep_metrics.csv")
    print(out_dir / "lodo_target_support.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: LODO validation only. Do NOT generate the formal WAPP "
        "perception trajectory unless all primary gates are reviewed and pass."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
