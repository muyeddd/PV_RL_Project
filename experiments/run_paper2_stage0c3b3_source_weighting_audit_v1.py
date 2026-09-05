#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0C-3B3
Source-weighting mechanism audit for boundary-preserving perception emulator.

AUDIT ONLY. No WAPP perception trajectory and no RL state are generated.

Background
----------
P2-0C-3B2 fixed lower-bound transport. In the actual WAPP deployment domain,
bias, MAE, width median, lower clipping, and width-error correlation passed,
but coverage still failed. Final-output transport approximately preserves a
sampled source row's coverage state, so the remaining mismatch is likely a
source-selection / weighting issue.

Fixed settings
--------------
- condition on true_L only
- |L_source - L_target| <= 0.01
- >=20 candidate rows and >=3 source dates
- Gaussian distance bandwidth = 0.005
- preserve final lower-clipped state
- no qhat recalibration
- no fallback
- no radius widening

Selectors compared
------------------
UNIFORM_DATE:
    Current 3B2 control. Choose eligible source date uniformly, then choose
    a local source row in that date using Gaussian distance weights.

POOLED_ROW:
    Choose directly from all local source rows using Gaussian distance weights.

BLOCK10:
    Cluster source rows into fixed 10-minute date-time blocks; choose an
    eligible block uniformly, then choose a local row in that block using
    Gaussian distance weights. This reduces repeated-frame dominance without
    imposing equal weight on entire dates.

Validation domain
-----------------
Paper1 held-out rows with true_L <= actual WAPP max.

Unchanged gates
---------------
|bias diff| <= 0.010
|MAE diff| <= 0.010
|width median diff| <= 0.020
|coverage diff| <= 0.040
|lower clipping diff| <= 0.070
|rho(width,|error|) diff| <= 0.120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = 1844
EXPECTED_DATES = 12
EXPECTED_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_WAPP_VALID = 729

LOCAL_RADIUS = 0.01
LOCAL_MIN_SAMPLES = 20
LOCAL_MIN_DATES = 3
KERNEL_BANDWIDTH = 0.005
BLOCK_MINUTES = 10

MC_REPS = 50
BASE_SEED = 20260905
DEPLOYMENT_SUPPORT_GATE = 0.99

GATES = {
    "bias": 0.010,
    "mae": 0.010,
    "width_median": 0.020,
    "coverage": 0.040,
    "lower_clipped_fraction": 0.070,
    "rho_width_abs_error": 0.120,
}

SELECTORS = ["UNIFORM_DATE", "POOLED_ROW", "BLOCK10"]


def qstats(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.5]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "max": float(x.max()),
    }


def spearman(a, b):
    aa = pd.to_numeric(pd.Series(a), errors="coerce")
    bb = pd.to_numeric(pd.Series(b), errors="coerce")
    ok = np.isfinite(aa) & np.isfinite(bb)
    if int(ok.sum()) < 3:
        return float("nan")
    if aa[ok].nunique() < 2 or bb[ok].nunique() < 2:
        return float("nan")
    return float(aa[ok].corr(bb[ok], method="spearman"))


def load_paper1(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "sample_id", "date", "timestamp", "role", "true_L",
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
        raise PermissionError(f"Unexpected role(s): {sorted(roles)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    if df["date"].nunique() != EXPECTED_DATES:
        raise RuntimeError("Unexpected number of Paper1 dates.")
    if not np.array_equal(
        df["timestamp"].dt.strftime("%Y-%m-%d").to_numpy(),
        df["date"].to_numpy(),
    ):
        raise RuntimeError("timestamp/date mismatch.")

    for col in ["true_L", "q50", "lower", "upper", "width"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    arr = df[["true_L", "q50", "lower", "upper", "width"]].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite Paper1 values.")
    if not ((df["lower"] <= df["q50"]) & (df["q50"] <= df["upper"])).all():
        raise RuntimeError("Interval ordering violated.")
    if not np.allclose(
        df["upper"].to_numpy(dtype=float) - df["lower"].to_numpy(dtype=float),
        df["width"].to_numpy(dtype=float),
        atol=1e-10, rtol=0.0,
    ):
        raise RuntimeError("width != upper-lower.")

    lower_clip_expected = np.isclose(
        df["lower"].to_numpy(dtype=float), 0.0, atol=1e-12, rtol=0.0
    )
    if not np.array_equal(
        lower_clip_expected,
        df["lower_clipped"].astype(bool).to_numpy(),
    ):
        raise RuntimeError("lower_clipped inconsistent.")
    if df["upper_clipped"].astype(bool).any():
        raise RuntimeError("Unexpected upper clipping.")

    df["block10_id"] = (
        df["date"]
        + "|"
        + df["timestamp"].dt.floor(f"{BLOCK_MINUTES}min").astype(str)
    )
    return df.reset_index(drop=True)


def load_wapp(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "L_power_proxy", "bridge_valid", "power_bridge_model"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing WAPP columns: {sorted(missing)}")

    valid = df["bridge_valid"].fillna(False).astype(bool)
    out = df.loc[valid].copy().reset_index(drop=True)
    if len(out) != EXPECTED_WAPP_VALID:
        raise RuntimeError(f"Expected {EXPECTED_WAPP_VALID} valid WAPP rows.")

    models = set(out["power_bridge_model"].astype(str))
    if models != {"COMMON_TEMPERATURE_PVWATTS_RATIO"}:
        raise RuntimeError(f"Unexpected bridge model(s): {sorted(models)}")

    out["L_power_proxy"] = pd.to_numeric(out["L_power_proxy"], errors="raise")
    return out


def metric_dict(true_l, q50, lower, upper):
    true_l = np.asarray(true_l, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    width = upper - lower
    err = q50 - true_l
    abs_err = np.abs(err)

    return {
        "bias": float(np.mean(err)),
        "mae": float(np.mean(abs_err)),
        "width_median": float(np.median(width)),
        "coverage": float(np.mean((true_l >= lower) & (true_l <= upper))),
        "lower_clipped_fraction": float(
            np.mean(np.isclose(lower, 0.0, atol=1e-12, rtol=0.0))
        ),
        "rho_width_abs_error": spearman(width, abs_err),
    }


def gaussian_weights(distances):
    w = np.exp(-0.5 * (distances / KERNEL_BANDWIDTH) ** 2)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Invalid Gaussian weights.")
    return w / total


def candidate_info(source, lstar):
    src_l = source["true_L"].to_numpy(dtype=float)
    dist_all = np.abs(src_l - lstar)
    idx = np.flatnonzero(dist_all <= LOCAL_RADIUS)
    dates = np.unique(source.loc[idx, "date"].to_numpy(dtype=str))
    ok = len(idx) >= LOCAL_MIN_SAMPLES and len(dates) >= LOCAL_MIN_DATES
    return {
        "ok": bool(ok),
        "indices": idx,
        "distances": dist_all[idx],
        "dates": dates,
    }


def choose_source_index(rng, source, info, selector):
    idx = np.asarray(info["indices"], dtype=int)
    dist = np.asarray(info["distances"], dtype=float)

    if selector == "POOLED_ROW":
        return int(rng.choice(idx, p=gaussian_weights(dist)))

    if selector == "UNIFORM_DATE":
        candidate_dates = source.loc[idx, "date"].to_numpy(dtype=str)
        chosen_date = str(rng.choice(sorted(np.unique(candidate_dates))))
        m = candidate_dates == chosen_date
        return int(rng.choice(idx[m], p=gaussian_weights(dist[m])))

    if selector == "BLOCK10":
        candidate_blocks = source.loc[idx, "block10_id"].to_numpy(dtype=str)
        chosen_block = str(rng.choice(sorted(np.unique(candidate_blocks))))
        m = candidate_blocks == chosen_block
        return int(rng.choice(idx[m], p=gaussian_weights(dist[m])))

    raise KeyError(selector)


def transport_final_output(source_row, target_l):
    src_l = float(source_row["true_L"])
    q50 = float(np.clip(target_l + (float(source_row["q50"]) - src_l), 0.0, 1.0))
    upper = float(np.clip(target_l + (float(source_row["upper"]) - src_l), 0.0, 1.0))

    if bool(source_row["lower_clipped"]):
        lower = 0.0
    else:
        lower = float(
            np.clip(target_l + (float(source_row["lower"]) - src_l), 0.0, 1.0)
        )

    if lower > q50 + 1e-12 or q50 > upper + 1e-12:
        raise RuntimeError("Transport violated interval ordering.")
    return q50, lower, upper


def audit_wapp_support_after_date_removal(p1, wapp):
    queries = wapp["L_power_proxy"].to_numpy(dtype=float)
    rows = []
    for removed in sorted(p1["date"].unique()):
        source = p1[~p1["date"].eq(removed)].reset_index(drop=True)
        oks = np.array([candidate_info(source, float(q))["ok"] for q in queries])
        rows.append({
            "removed_paper1_date": removed,
            "supported_fraction": float(oks.mean()),
            "unsupported_queries": int((~oks).sum()),
        })
    return pd.DataFrame(rows)


def compare_gates(actual, generated):
    result = {}
    all_pass = True
    for metric, tol in GATES.items():
        diff = abs(float(generated[metric]) - float(actual[metric]))
        passed = bool(np.isfinite(diff) and diff <= tol)
        result[metric] = {
            "actual_macro": float(actual[metric]),
            "generated_macro": float(generated[metric]),
            "absolute_difference": diff,
            "tolerance": tol,
            "pass": passed,
        }
        all_pass &= passed
    result["all_pass"] = bool(all_pass)
    return result


def main():
    p = argparse.ArgumentParser(
        description="P2-0C-3B3 source-weighting mechanism audit."
    )
    p.add_argument("--paper1-dev-cqr", required=True, type=Path)
    p.add_argument("--wapp-power-bridge", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0c_3b3_source_weighting_audit_v1"
        ),
    )
    args = p.parse_args()

    p1_path = args.paper1_dev_cqr.expanduser().resolve()
    wapp_path = args.wapp_power_bridge.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for path in [p1_path, wapp_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/8] Load frozen Paper1 DEV and WAPP deployment domain")
    p1 = load_paper1(p1_path)
    wapp = load_wapp(wapp_path)
    wapp_max = float(wapp["L_power_proxy"].max())

    print("[2/8] Re-audit actual WAPP support after each date removal")
    wapp_support = audit_wapp_support_after_date_removal(p1, wapp)
    min_support = float(wapp_support["supported_fraction"].min())
    deployment_support_pass = bool(min_support >= DEPLOYMENT_SUPPORT_GATE)

    print("[3/8] Build deployment-domain held-out folds")
    fold_data = {}
    support_rows = []

    for heldout in sorted(p1["date"].unique()):
        target_full = (
            p1[p1["date"].eq(heldout) & (p1["true_L"] <= wapp_max)]
            .copy()
            .reset_index(drop=True)
        )
        source = (
            p1[~p1["date"].eq(heldout)]
            .copy()
            .reset_index(drop=True)
        )

        infos = []
        ok_mask = []
        for _, row in target_full.iterrows():
            info = candidate_info(source, float(row["true_L"]))
            infos.append(info)
            ok_mask.append(info["ok"])
            support_rows.append({
                "heldout_date": heldout,
                "target_sample_id": row["sample_id"],
                "target_true_L": float(row["true_L"]),
                "support_ok": bool(info["ok"]),
                "candidate_samples": int(len(info["indices"])),
                "candidate_dates": int(len(info["dates"])),
            })

        fold_data[heldout] = {
            "target_full": target_full,
            "source": source,
            "infos": infos,
            "ok_mask": np.asarray(ok_mask, dtype=bool),
        }

    support_df = pd.DataFrame(support_rows)

    print("[4/8] Run three source-selection mechanisms")
    date_metric_rows = []
    usage_rows = []

    for selector_idx, selector in enumerate(SELECTORS):
        selector_chosen_dates = []

        for fold_idx, heldout in enumerate(sorted(fold_data.keys())):
            block = fold_data[heldout]
            target_full = block["target_full"]
            source = block["source"]
            ok_mask = block["ok_mask"]

            if int(ok_mask.sum()) < 3:
                continue

            target = target_full.loc[ok_mask].copy().reset_index(drop=True)
            infos = [x for x, ok in zip(block["infos"], ok_mask) if ok]
            true_l = target["true_L"].to_numpy(dtype=float)

            actual = metric_dict(
                true_l,
                target["q50"].to_numpy(dtype=float),
                target["lower"].to_numpy(dtype=float),
                target["upper"].to_numpy(dtype=float),
            )

            rep_metrics = []

            for rep in range(MC_REPS):
                rng = np.random.default_rng(
                    BASE_SEED + selector_idx * 1_000_000 + fold_idx * 10_000 + rep
                )

                q50_gen = np.empty(len(target))
                lower_gen = np.empty(len(target))
                upper_gen = np.empty(len(target))

                for i, ((_, trow), info) in enumerate(zip(target.iterrows(), infos)):
                    src_idx = choose_source_index(rng, source, info, selector)
                    src_row = source.iloc[src_idx]
                    q50, lower, upper = transport_final_output(
                        src_row, float(trow["true_L"])
                    )
                    q50_gen[i] = q50
                    lower_gen[i] = lower
                    upper_gen[i] = upper
                    selector_chosen_dates.append(str(src_row["date"]))

                rep_metrics.append(
                    metric_dict(true_l, q50_gen, lower_gen, upper_gen)
                )

            generated = {}
            for metric in GATES:
                vals = [m[metric] for m in rep_metrics if np.isfinite(m[metric])]
                generated[metric] = float(np.mean(vals)) if vals else np.nan

            row = {
                "selector": selector,
                "heldout_date": heldout,
                "N_deployment_domain": int(len(target_full)),
                "N_supported": int(ok_mask.sum()),
                "support_fraction": float(ok_mask.mean()),
            }
            for metric in GATES:
                row[f"actual_{metric}"] = actual[metric]
                row[f"generated_{metric}"] = generated[metric]
                row[f"absdiff_{metric}"] = abs(
                    generated[metric] - actual[metric]
                )
            date_metric_rows.append(row)

        if selector_chosen_dates:
            usage = pd.Series(selector_chosen_dates).value_counts(normalize=True)
            for date, share in usage.items():
                usage_rows.append({
                    "selector": selector,
                    "source_date": str(date),
                    "selection_share": float(share),
                })

    date_metrics = pd.DataFrame(date_metric_rows)
    usage_df = pd.DataFrame(usage_rows)

    print("[5/8] Build selector-level date-macro metrics")
    macro_rows = []
    gate_rows = []

    for selector in SELECTORS:
        g = date_metrics[date_metrics["selector"].eq(selector)]
        actual_macro = {}
        generated_macro = {}

        for metric in GATES:
            actual_macro[metric] = float(
                pd.to_numeric(g[f"actual_{metric}"], errors="coerce").dropna().mean()
            )
            generated_macro[metric] = float(
                pd.to_numeric(g[f"generated_{metric}"], errors="coerce").dropna().mean()
            )

        comp = compare_gates(actual_macro, generated_macro)

        macro_row = {"selector": selector}
        for metric in GATES:
            macro_row[f"actual_{metric}"] = actual_macro[metric]
            macro_row[f"generated_{metric}"] = generated_macro[metric]
            macro_row[f"absdiff_{metric}"] = comp[metric]["absolute_difference"]
            gate_rows.append({
                "selector": selector,
                "metric": metric,
                "actual_macro": comp[metric]["actual_macro"],
                "generated_macro": comp[metric]["generated_macro"],
                "absolute_difference": comp[metric]["absolute_difference"],
                "tolerance": comp[metric]["tolerance"],
                "pass": comp[metric]["pass"],
            })

        macro_row["all_distribution_gates_pass"] = bool(comp["all_pass"])
        macro_rows.append(macro_row)

    macro_df = pd.DataFrame(macro_rows)
    gate_df = pd.DataFrame(gate_rows)

    print("[6/8] Audit source-date concentration")
    concentration = {}
    if len(usage_df):
        for selector, g in usage_df.groupby("selector"):
            concentration[selector] = {
                "max_source_date_share": float(g["selection_share"].max()),
                "source_dates_used": int(g["source_date"].nunique()),
                "selection_share_distribution": qstats(g["selection_share"]),
            }

    print("[7/8] Evaluate mechanism candidates without auto-selecting")
    passing_selectors = macro_df.loc[
        macro_df["all_distribution_gates_pass"].astype(bool),
        "selector",
    ].astype(str).tolist()

    summary = {
        "stage": "P2-0C-3B3",
        "audit_only": True,
        "wapp_perception_trajectory_generated": False,
        "no_qhat_recalibration": True,
        "no_radius_change": True,
        "no_support_threshold_change": True,
        "no_fallback": True,
        "deployment_domain": {
            "wapp_valid_days": int(len(wapp)),
            "wapp_L_max": wapp_max,
        },
        "actual_wapp_support_after_date_removal": {
            "minimum_supported_fraction": min_support,
            "gate": DEPLOYMENT_SUPPORT_GATE,
            "pass": deployment_support_pass,
        },
        "fixed_transport": {
            "name": "BOUNDARY_STATE_PRESERVING_FINAL_CQR_TRANSPORT",
            "local_radius_abs_L": LOCAL_RADIUS,
            "local_min_samples": LOCAL_MIN_SAMPLES,
            "local_min_dates": LOCAL_MIN_DATES,
            "kernel_bandwidth": KERNEL_BANDWIDTH,
            "source_lower_clipped_state_preserved": True,
        },
        "selectors_tested": {
            "UNIFORM_DATE": "eligible source date uniform, then local row kernel",
            "POOLED_ROW": "all local source rows pooled with Gaussian L-distance kernel",
            "BLOCK10": "fixed 10-minute source block uniform, then local row kernel",
        },
        "selector_macro_results": macro_df.to_dict(orient="records"),
        "passing_selectors": passing_selectors,
        "source_date_concentration": concentration,
        "interpretation_rule": (
            "Do not auto-select solely on passing. If BLOCK10 passes with "
            "reasonable date concentration, prefer it as the cluster-aware "
            "candidate. If only POOLED_ROW passes, retain it as a candidate "
            "but require a clustered-source sensitivity before freezing."
        ),
    }

    print("[8/8] Write mechanism-audit outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_df.to_csv(
        out_dir / "selector_date_macro_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_df.to_csv(
        out_dir / "selector_gate_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    date_metrics.to_csv(
        out_dir / "selector_date_level_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    usage_df.to_csv(
        out_dir / "selector_source_date_usage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    support_df.to_csv(
        out_dir / "selector_target_support.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wapp_support.to_csv(
        out_dir / "wapp_support_by_removed_date.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "selector_date_macro_metrics.csv")
    print(out_dir / "selector_gate_comparison.csv")
    print(out_dir / "selector_date_level_metrics.csv")
    print(out_dir / "selector_source_date_usage.csv")
    print(out_dir / "selector_target_support.csv")
    print(out_dir / "wapp_support_by_removed_date.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: mechanism audit only. Do NOT generate WAPP perception "
        "trajectories until one source selector is reviewed and frozen."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
