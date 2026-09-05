#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0C-3B2
Boundary-state-preserving LODO validation of perception emulator v2.

WHY V2 EXISTS
-------------
P2-0C-3B emulator-v1 was rejected. Failure diagnosis showed:
- actual WAPP deployment queries remain very well supported under removal of
  any one Paper1 date;
- the main deployment-relevant mismatch is lower-bound clipping / coverage;
- v1 transported q05/q50/q95 and then rebuilt the conformal interval, which
  can turn a source row that was already lower-clipped into an unclipped
  target row after a positive L shift.

V2 PRINCIPLE
------------
Emulate the FINAL Paper1 perception output geometry rather than re-running the
latent pre-conformal quantile construction.

For one source row:
    e50 = q50_source - true_L_source
    elo = lower_source - true_L_source
    eup = upper_source - true_L_source

For target loss L*:
    q50* = clip(L* + e50, 0, 1)

    if source lower_clipped:
        lower* = 0
    else:
        lower* = clip(L* + elo, 0, 1)

    upper* = clip(L* + eup, 0, 1)
    width* = upper* - lower*

Thus the sampled source row's final lower-bound state is explicitly preserved.
The whole final tuple is sampled jointly from one real Paper1 row.

This is NOT a change to Paper1 qhat. The source lower/upper already contain the
frozen Paper1 conformal correction (qhat = 0.004862844288256299). V2 emulates
those final outputs directly.

VALIDATION SCOPE
----------------
The emulator is deployment-specific: its intended WAPP query domain is the
actual P2-0C-1B L_power_proxy range [0, WAPP max].

Primary support gate:
- Recompute the 729 ACTUAL WAPP queries after removing each Paper1 date.
- Minimum supported fraction over the 12 removals must be >= 0.99.
- Support definition is unchanged:
      |true_L_source - L_query| <= 0.01
      >=20 candidate rows
      >=3 source dates
- no fallback / no radius widening.

LODO distribution validation:
- 12 held-out Paper1 dates;
- only held-out rows inside the actual WAPP deployment L range are used;
- unsupported validation rows are explicitly excluded, with no fallback;
- actual and generated metrics are always compared on the same supported rows;
- 50 Monte Carlo replicates per fold;
- date-balanced source selection;
- SAME WAPP-relevant metric tolerances as rejected v1:
      |bias diff| <= 0.010
      |MAE diff| <= 0.010
      |width-median diff| <= 0.020
      |coverage diff| <= 0.040
      |lower-clip diff| <= 0.070
      |rho(width,abs-error) diff| <= 0.120

This stage still does NOT generate a formal WAPP perception trajectory.

Outputs
-------
v2_lodo_date_metrics.csv
v2_lodo_rep_metrics.csv
v2_lodo_target_support.csv
v2_wapp_support_by_removed_date.csv
audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 1844
EXPECTED_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_DATES = 12
EXPECTED_WAPP_VALID = 729

FROZEN_QHAT = 0.004862844288256299

LOCAL_RADIUS = 0.01
LOCAL_MIN_SAMPLES = 20
LOCAL_MIN_DATES = 3
KERNEL_BANDWIDTH = 0.005

MC_REPS = 50
BASE_SEED = 20260905

DEPLOYMENT_SUPPORT_GATE = 0.99

DEPLOYMENT_METRIC_GATES = {
    "bias": 0.010,
    "mae": 0.010,
    "width_median": 0.020,
    "coverage": 0.040,
    "lower_clipped_fraction": 0.070,
    "rho_width_abs_error": 0.120,
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
        "q50", "lower", "upper", "width",
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

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    if df["date"].nunique() != EXPECTED_DATES:
        raise RuntimeError(
            f"Expected {EXPECTED_DATES} dates, found {df['date'].nunique()}"
        )

    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise RuntimeError("sample_id must be unique and non-null.")

    for col in ["true_L", "q50", "lower", "upper", "width"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    arr = df[["true_L", "q50", "lower", "upper", "width"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite Paper1 numeric values.")

    if not ((df["true_L"] >= 0) & (df["true_L"] <= 1)).all():
        raise RuntimeError("Paper1 true_L outside [0,1].")
    if not ((df["lower"] >= 0) & (df["upper"] <= 1)).all():
        raise RuntimeError("Paper1 interval outside [0,1].")
    if not (df["lower"] <= df["q50"]).all():
        raise RuntimeError("Paper1 lower > q50.")
    if not (df["q50"] <= df["upper"]).all():
        raise RuntimeError("Paper1 q50 > upper.")
    if not np.allclose(
        df["upper"].to_numpy(dtype=float)
        - df["lower"].to_numpy(dtype=float),
        df["width"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("Paper1 width != upper-lower.")

    covered_expected = (
        (df["true_L"] >= df["lower"])
        & (df["true_L"] <= df["upper"])
    )
    if not np.array_equal(
        covered_expected.to_numpy(dtype=bool),
        df["covered"].astype(bool).to_numpy(),
    ):
        raise RuntimeError("Paper1 covered flag inconsistent.")

    lower_clip_expected = np.isclose(
        df["lower"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    if not np.array_equal(
        lower_clip_expected,
        df["lower_clipped"].astype(bool).to_numpy(),
    ):
        raise RuntimeError("Paper1 lower_clipped flag inconsistent.")

    if df["upper_clipped"].astype(bool).any():
        raise RuntimeError(
            "Unexpected Paper1 upper clipping; v2 assumes frozen asset has none."
        )

    return df.reset_index(drop=True)


def load_wapp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "L_power_proxy", "bridge_valid", "power_bridge_model"
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing WAPP columns: {sorted(missing)}")

    valid = df["bridge_valid"].fillna(False).astype(bool)
    out = df.loc[valid].copy().reset_index(drop=True)

    if len(out) != EXPECTED_WAPP_VALID:
        raise RuntimeError(
            f"Expected {EXPECTED_WAPP_VALID} valid WAPP rows, found {len(out)}"
        )

    models = set(out["power_bridge_model"].astype(str))
    if models != {"COMMON_TEMPERATURE_PVWATTS_RATIO"}:
        raise RuntimeError(f"Unexpected bridge models: {sorted(models)}")

    out["L_power_proxy"] = pd.to_numeric(
        out["L_power_proxy"], errors="raise"
    )
    if not np.isfinite(out["L_power_proxy"]).all():
        raise RuntimeError("Non-finite WAPP L_power_proxy.")

    return out


def metric_dict(true_l, q50, lower, upper) -> dict:
    true_l = np.asarray(true_l, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if len(true_l) == 0:
        return {
            "N": 0,
            "bias": np.nan,
            "mae": np.nan,
            "width_mean": np.nan,
            "width_median": np.nan,
            "coverage": np.nan,
            "lower_clipped_fraction": np.nan,
            "rho_width_abs_error": np.nan,
        }

    width = upper - lower
    error = q50 - true_l
    abs_error = np.abs(error)
    covered = (true_l >= lower) & (true_l <= upper)
    lower_clip = np.isclose(lower, 0.0, rtol=0.0, atol=1e-12)

    return {
        "N": int(len(true_l)),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(abs_error)),
        "width_mean": float(np.mean(width)),
        "width_median": float(np.median(width)),
        "coverage": float(np.mean(covered)),
        "lower_clipped_fraction": float(np.mean(lower_clip)),
        "rho_width_abs_error": spearman(width, abs_error),
    }


def support_counts(
    source_l: np.ndarray,
    source_dates: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(len(queries), dtype=int)
    date_counts = np.zeros(len(queries), dtype=int)

    for i, q in enumerate(queries):
        m = np.abs(source_l - q) <= LOCAL_RADIUS
        counts[i] = int(m.sum())
        date_counts[i] = int(np.unique(source_dates[m]).size)

    ok = (
        (counts >= LOCAL_MIN_SAMPLES)
        & (date_counts >= LOCAL_MIN_DATES)
    )
    return counts, date_counts, ok


def wapp_support_after_date_removal(
    p1: pd.DataFrame,
    wapp: pd.DataFrame,
) -> pd.DataFrame:
    q = wapp["L_power_proxy"].to_numpy(dtype=float)
    rows = []

    for removed in sorted(p1["date"].unique()):
        source = p1[~p1["date"].eq(removed)]
        src_l = source["true_L"].to_numpy(dtype=float)
        src_dates = source["date"].to_numpy(dtype=str)

        counts, date_counts, ok = support_counts(
            src_l, src_dates, q
        )

        rows.append({
            "removed_paper1_date": removed,
            "wapp_queries": int(len(q)),
            "supported_queries": int(ok.sum()),
            "unsupported_queries": int((~ok).sum()),
            "supported_fraction": float(ok.mean()),
            "candidate_samples_min": int(counts.min()),
            "candidate_samples_q05": float(np.quantile(counts, 0.05)),
            "candidate_samples_median": float(np.median(counts)),
            "candidate_dates_min": int(date_counts.min()),
            "candidate_dates_q05": float(np.quantile(date_counts, 0.05)),
            "candidate_dates_median": float(np.median(date_counts)),
        })

    return pd.DataFrame(rows)


def build_candidate_map(
    source: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    src_l = source["true_L"].to_numpy(dtype=float)
    src_dates = source["date"].to_numpy(dtype=str)

    maps = []
    rows = []

    for pos, (_, row) in enumerate(targets.iterrows()):
        lstar = float(row["true_L"])
        dist = np.abs(src_l - lstar)
        m = dist <= LOCAL_RADIUS

        idx = np.flatnonzero(m)
        dates = np.unique(src_dates[idx])
        ok = (
            len(idx) >= LOCAL_MIN_SAMPLES
            and len(dates) >= LOCAL_MIN_DATES
        )

        by_date = {}
        if ok:
            for d in dates:
                didx = idx[src_dates[idx] == d]
                by_date[str(d)] = {
                    "indices": didx,
                    "distances": dist[didx],
                }

        maps.append({
            "support_ok": bool(ok),
            "by_date": by_date,
        })

        rows.append({
            "target_position": pos,
            "target_sample_id": row["sample_id"],
            "target_date": row["date"],
            "target_true_L": lstar,
            "candidate_samples": int(len(idx)),
            "candidate_dates": int(len(dates)),
            "support_ok": bool(ok),
        })

    return maps, pd.DataFrame(rows)


def weighted_pick(
    rng: np.random.Generator,
    indices: np.ndarray,
    distances: np.ndarray,
) -> int:
    w = np.exp(-0.5 * (distances / KERNEL_BANDWIDTH) ** 2)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Invalid Gaussian weights.")
    w = w / total
    return int(rng.choice(indices, p=w))


def generate_v2_rep(
    rng: np.random.Generator,
    source: pd.DataFrame,
    targets: pd.DataFrame,
    candidate_maps: list[dict],
) -> dict[str, np.ndarray]:
    n = len(targets)

    src_l = source["true_L"].to_numpy(dtype=float)
    src_q50 = source["q50"].to_numpy(dtype=float)
    src_lower = source["lower"].to_numpy(dtype=float)
    src_upper = source["upper"].to_numpy(dtype=float)
    src_lower_clipped = source["lower_clipped"].astype(bool).to_numpy()

    target_l = targets["true_L"].to_numpy(dtype=float)

    q50_gen = np.empty(n, dtype=float)
    lower_gen = np.empty(n, dtype=float)
    upper_gen = np.empty(n, dtype=float)
    source_date_gen = np.empty(n, dtype=object)
    source_lower_clip_gen = np.empty(n, dtype=bool)

    for i, cmap in enumerate(candidate_maps):
        if not cmap["support_ok"]:
            raise RuntimeError(
                "Unsupported target entered v2 generation; implementation error."
            )

        eligible_dates = sorted(cmap["by_date"].keys())
        chosen_date = str(rng.choice(eligible_dates))
        block = cmap["by_date"][chosen_date]

        src_idx = weighted_pick(
            rng,
            np.asarray(block["indices"], dtype=int),
            np.asarray(block["distances"], dtype=float),
        )

        e50 = src_q50[src_idx] - src_l[src_idx]
        eup = src_upper[src_idx] - src_l[src_idx]

        q50 = np.clip(target_l[i] + e50, 0.0, 1.0)
        upper = np.clip(target_l[i] + eup, 0.0, 1.0)

        if src_lower_clipped[src_idx]:
            lower = 0.0
        else:
            elo = src_lower[src_idx] - src_l[src_idx]
            lower = np.clip(target_l[i] + elo, 0.0, 1.0)

        if lower > q50 + 1e-12 or q50 > upper + 1e-12:
            raise RuntimeError(
                "Boundary-preserving transport violated interval ordering."
            )

        q50_gen[i] = q50
        lower_gen[i] = lower
        upper_gen[i] = upper
        source_date_gen[i] = chosen_date
        source_lower_clip_gen[i] = src_lower_clipped[src_idx]

    return {
        "q50": q50_gen,
        "lower": lower_gen,
        "upper": upper_gen,
        "source_date": source_date_gen,
        "source_lower_clipped": source_lower_clip_gen,
    }


def macro_mean(df: pd.DataFrame, col: str) -> float:
    x = pd.to_numeric(df[col], errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def compare_gates(actual: dict, generated: dict) -> dict:
    out = {}
    all_pass = True

    for metric, tol in DEPLOYMENT_METRIC_GATES.items():
        a = float(actual[metric])
        g = float(generated[metric])
        diff = abs(g - a)
        passed = bool(np.isfinite(diff) and diff <= tol)
        out[metric] = {
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
        description=(
            "P2-0C-3B2 boundary-state-preserving LODO validation "
            "for perception emulator v2."
        )
    )
    p.add_argument("--paper1-dev-cqr", required=True, type=Path)
    p.add_argument("--wapp-power-bridge", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0c_3b2_boundary_preserving_lodo_v1"
        ),
    )
    args = p.parse_args()

    paper1_path = args.paper1_dev_cqr.expanduser().resolve()
    wapp_path = args.wapp_power_bridge.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for path in [paper1_path, wapp_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/8] Load Paper1 DECISION_DEVELOPMENT and WAPP deployment domain")
    p1 = load_paper1(paper1_path)
    wapp = load_wapp(wapp_path)
    wapp_max = float(wapp["L_power_proxy"].max())

    print("[2/8] Re-audit actual WAPP support after each Paper1 date removal")
    wapp_support = wapp_support_after_date_removal(p1, wapp)
    deployment_support_min = float(
        wapp_support["supported_fraction"].min()
    )
    deployment_support_pass = bool(
        deployment_support_min >= DEPLOYMENT_SUPPORT_GATE
    )

    print("[3/8] Build deployment-domain LODO target support maps")
    fold_data = {}
    support_tables = []

    for heldout_date in sorted(p1["date"].unique()):
        full_target = (
            p1[p1["date"].eq(heldout_date)]
            .copy()
            .reset_index(drop=True)
        )
        target = (
            full_target[full_target["true_L"] <= wapp_max]
            .copy()
            .reset_index(drop=True)
        )
        source = (
            p1[~p1["date"].eq(heldout_date)]
            .copy()
            .reset_index(drop=True)
        )

        maps, support = build_candidate_map(source, target)
        support["heldout_date"] = heldout_date
        support_tables.append(support)

        fold_data[heldout_date] = {
            "target": target,
            "source": source,
            "maps": maps,
        }

    support_all = pd.concat(support_tables, ignore_index=True)

    print("[4/8] Run boundary-state-preserving Monte Carlo LODO")
    date_rows = []
    rep_rows = []

    for fold_idx, heldout_date in enumerate(sorted(fold_data.keys())):
        target_full = fold_data[heldout_date]["target"]
        source = fold_data[heldout_date]["source"]
        maps_full = fold_data[heldout_date]["maps"]

        support_mask = np.array(
            [m["support_ok"] for m in maps_full],
            dtype=bool,
        )

        if int(support_mask.sum()) < 3:
            # Some Paper1 dates have very few rows in the WAPP deployment
            # range. They remain visible in support output but cannot provide
            # a stable date-level correlation metric.
            continue

        target = (
            target_full.loc[support_mask]
            .copy()
            .reset_index(drop=True)
        )
        maps = [
            m for m, ok in zip(maps_full, support_mask) if ok
        ]

        true_l = target["true_L"].to_numpy(dtype=float)

        actual = metric_dict(
            true_l,
            target["q50"].to_numpy(dtype=float),
            target["lower"].to_numpy(dtype=float),
            target["upper"].to_numpy(dtype=float),
        )

        generated_rep_metrics = []

        for rep in range(MC_REPS):
            rng = np.random.default_rng(
                BASE_SEED + fold_idx * 10000 + rep
            )
            gen = generate_v2_rep(
                rng,
                source,
                target,
                maps,
            )

            gm = metric_dict(
                true_l,
                gen["q50"],
                gen["lower"],
                gen["upper"],
            )
            generated_rep_metrics.append(gm)

            rep_rows.append({
                "heldout_date": heldout_date,
                "rep": rep,
                **gm,
            })

        def rep_mean(metric: str) -> float:
            vals = [
                x[metric]
                for x in generated_rep_metrics
                if np.isfinite(x[metric])
            ]
            return float(np.mean(vals)) if vals else np.nan

        row = {
            "heldout_date": heldout_date,
            "N_deployment_domain": int(len(target_full)),
            "N_supported": int(support_mask.sum()),
            "N_unsupported": int((~support_mask).sum()),
            "support_fraction": float(support_mask.mean())
            if len(support_mask) else np.nan,
        }

        for metric in DEPLOYMENT_METRIC_GATES:
            row[f"actual_{metric}"] = actual[metric]
            row[f"generated_{metric}"] = rep_mean(metric)
            row[f"absdiff_{metric}"] = abs(
                row[f"generated_{metric}"]
                - row[f"actual_{metric}"]
            )

        date_rows.append(row)

    date_metrics = pd.DataFrame(date_rows)
    rep_metrics = pd.DataFrame(rep_rows)

    print("[5/8] Build date-macro deployment-domain metrics")
    actual_macro = {
        metric: macro_mean(date_metrics, f"actual_{metric}")
        for metric in DEPLOYMENT_METRIC_GATES
    }
    generated_macro = {
        metric: macro_mean(date_metrics, f"generated_{metric}")
        for metric in DEPLOYMENT_METRIC_GATES
    }

    gate_comparison = compare_gates(
        actual_macro,
        generated_macro,
    )

    print("[6/8] Audit boundary-state reproduction")
    validation_support_fraction = float(
        support_all["support_ok"].astype(bool).mean()
    )
    per_date_support = (
        support_all.groupby("heldout_date")["support_ok"]
        .mean()
        .reset_index(name="support_fraction")
    )

    print("[7/8] Evaluate predeclared v2 gates")
    all_primary_pass = bool(
        deployment_support_pass
        and gate_comparison["all_pass"]
    )

    summary = {
        "stage": "P2-0C-3B2",
        "validation_only": True,
        "emulator_v2_frozen": False,
        "wapp_perception_trajectory_generated": False,
        "rejected_v1_not_overwritten": True,
        "emulator_v2": {
            "name": "BOUNDARY_STATE_PRESERVING_FINAL_CQR_TRANSPORT",
            "conditioning_variables": ["true_L"],
            "local_radius_abs_L": LOCAL_RADIUS,
            "local_min_samples": LOCAL_MIN_SAMPLES,
            "local_min_dates": LOCAL_MIN_DATES,
            "source_date_selection": "UNIFORM_OVER_ELIGIBLE_SOURCE_DATES",
            "within_date_selection": (
                "GAUSSIAN_DISTANCE_WEIGHTED_BANDWIDTH_0.005"
            ),
            "transported_joint_outputs": [
                "q50-true_L",
                "final lower-true_L",
                "final upper-true_L",
                "source lower-clipped boundary state",
            ],
            "source_lower_clipped_state_preserved": True,
            "frozen_qhat_provenance": FROZEN_QHAT,
            "qhat_recalibrated": False,
            "fallback_for_unsupported_queries": False,
            "wapp_irradiance_used": False,
        },
        "deployment_scope": {
            "wapp_valid_days": int(len(wapp)),
            "wapp_L_max": wapp_max,
            "primary_validation_domain": (
                "Paper1 held-out rows with true_L <= actual WAPP maximum"
            ),
            "not_claimed": (
                "emulator validity over the full Paper1 high-loss domain"
            ),
        },
        "actual_wapp_support_after_date_removal": {
            "minimum_supported_fraction": deployment_support_min,
            "mean_supported_fraction": float(
                wapp_support["supported_fraction"].mean()
            ),
            "gate": DEPLOYMENT_SUPPORT_GATE,
            "pass": deployment_support_pass,
            "worst_removed_date": str(
                wapp_support.loc[
                    wapp_support["supported_fraction"].idxmin(),
                    "removed_paper1_date",
                ]
            ),
        },
        "lodo_validation_support_diagnostic": {
            "deployment_domain_target_rows": int(len(support_all)),
            "supported_rows": int(
                support_all["support_ok"].astype(bool).sum()
            ),
            "supported_fraction": validation_support_fraction,
            "per_date_support_fraction_distribution": qstats(
                per_date_support["support_fraction"]
            ),
            "note": (
                "This is a validation-data coverage diagnostic, not the "
                "primary deployment support gate; actual WAPP queries are "
                "audited separately above."
            ),
        },
        "date_macro_deployment_domain": {
            "actual": actual_macro,
            "generated": generated_macro,
            "gate_comparison": gate_comparison,
        },
        "date_level_discrepancy_diagnostics": {
            "bias_absdiff_distribution": qstats(
                date_metrics["absdiff_bias"]
            ),
            "mae_absdiff_distribution": qstats(
                date_metrics["absdiff_mae"]
            ),
            "width_median_absdiff_distribution": qstats(
                date_metrics["absdiff_width_median"]
            ),
            "coverage_absdiff_distribution": qstats(
                date_metrics["absdiff_coverage"]
            ),
            "lower_clip_absdiff_distribution": qstats(
                date_metrics["absdiff_lower_clipped_fraction"]
            ),
            "rho_absdiff_distribution": qstats(
                date_metrics["absdiff_rho_width_abs_error"]
            ),
        },
        "gates": {
            "actual_wapp_deployment_support_pass": deployment_support_pass,
            "deployment_distributional_metrics_pass": bool(
                gate_comparison["all_pass"]
            ),
            "all_primary_gates_pass": all_primary_pass,
        },
        "next_step_if_pass": (
            "Freeze emulator-v2 design, then generate multiple independent "
            "729-day WAPP imperfect-perception trajectories and audit their "
            "joint/marginal properties before RL."
        ),
        "next_step_if_fail": (
            "Do not widen L radius or tune Paper1 qhat. Diagnose whether "
            "remaining mismatch requires an explicit latent date/regime "
            "mixture or another final-output transport rule."
        ),
        "limitations": [
            "V2 is intentionally deployment-specific to the observed WAPP loss domain.",
            "It does not claim validity in Paper1 high-loss regimes that the WAPP deployment never queries.",
            "It emulates final Paper1 perception outputs; it does not recreate images or rerun the vision model.",
            "Daily regime persistence is not imposed here and remains a later sensitivity analysis.",
            "Within-q50 width-shuffle remains mandatory in the later RL ablation.",
        ],
    }

    print("[8/8] Write v2 validation outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    date_metrics.to_csv(
        out_dir / "v2_lodo_date_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rep_metrics.to_csv(
        out_dir / "v2_lodo_rep_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    support_all.to_csv(
        out_dir / "v2_lodo_target_support.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wapp_support.to_csv(
        out_dir / "v2_wapp_support_by_removed_date.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "v2_lodo_date_metrics.csv")
    print(out_dir / "v2_lodo_rep_metrics.csv")
    print(out_dir / "v2_lodo_target_support.csv")
    print(out_dir / "v2_wapp_support_by_removed_date.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: v2 LODO validation only. Do NOT generate the formal WAPP "
        "perception trajectory unless all primary gates are reviewed and pass."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
