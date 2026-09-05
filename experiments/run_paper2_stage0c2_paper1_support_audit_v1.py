#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P2-0C-2 Paper1 support-domain / extrapolation-risk audit.

Audit only: no emulator fitting/calling, no RANDOM_TEST/SEALED_DATES, no WAPP
irradiance conditioning. The only conditioning variable is Paper1 true_L.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_P1_N = 1844
EXPECTED_ROLE = "DECISION_DEVELOPMENT"
EXPECTED_P1_DATES = 12
EXPECTED_WAPP_DAYS = 730
EXPECTED_WAPP_VALID = 729

ZERO_BOUNDARY_TOL = 0.001
LOCAL_RADIUS = 0.01
LOCAL_MIN_SAMPLES = 30
LOCAL_MIN_DATES = 3
LOCAL_SUPPORT_FRACTION_GATE = 0.95
RANGE_SUPPORT_FRACTION_GATE = 0.99
NN_UNIQUE_P95_GATE = 0.005
NN_UNIQUE_P99_GATE = 0.010
REPORT_BIN_EDGES = [0.0,0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.40,0.60,0.80,1.00]


def qstats(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q = x.quantile([0.01,0.05,0.25,0.50,0.75,0.95,0.99])
    return {
        "n": int(len(x)), "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x)>1 else 0.0,
        "min": float(x.min()), "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]), "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]), "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]), "q99": float(q.loc[0.99]),
        "max": float(x.max()),
    }


def load_bridge(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    req = {"date","L_power_proxy","S_soil_observed","S_soil_physical","bridge_valid","power_bridge_model"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing P2-0C-1B columns: {sorted(miss)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) != EXPECTED_WAPP_DAYS or df["date"].duplicated().any():
        raise RuntimeError("Unexpected WAPP daily asset shape/date uniqueness.")
    models = set(df["power_bridge_model"].dropna().astype(str))
    if models != {"COMMON_TEMPERATURE_PVWATTS_RATIO"}:
        raise RuntimeError(f"Unexpected bridge model(s): {sorted(models)}")
    valid = df["bridge_valid"].fillna(False).astype(bool)
    if int(valid.sum()) != EXPECTED_WAPP_VALID:
        raise RuntimeError(f"Expected {EXPECTED_WAPP_VALID} valid WAPP days, found {int(valid.sum())}")
    l = pd.to_numeric(df.loc[valid,"L_power_proxy"], errors="raise")
    if not np.isfinite(l).all() or (l<0).any() or (l>1).any():
        raise RuntimeError("Valid L_power_proxy must be finite in [0,1].")
    return df


def load_p1(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    req = {"sample_id","date","role","true_L","q50","lower","upper","width"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing Paper1 DEV columns: {sorted(miss)}")
    if len(df) != EXPECTED_P1_N:
        raise RuntimeError(f"Paper1 DEV N guard failed: {len(df)}")
    roles = set(df["role"].astype(str))
    if roles != {EXPECTED_ROLE}:
        raise PermissionError(f"Only {EXPECTED_ROLE} authorized, found {sorted(roles)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if df["date"].nunique() != EXPECTED_P1_DATES:
        raise RuntimeError("Paper1 DEV date-count guard failed.")
    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise RuntimeError("Paper1 sample_id must be unique/non-null.")
    num = df[["true_L","q50","lower","upper","width"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(num.to_numpy(float)).all():
        raise RuntimeError("Paper1 DEV numeric fields non-finite.")
    if (num["true_L"]<0).any() or (num["true_L"]>1).any():
        raise RuntimeError("Paper1 true_L outside [0,1].")
    if (num["lower"]<0).any() or (num["upper"]>1).any() or (num["lower"]>num["upper"]).any():
        raise RuntimeError("Paper1 CQR interval invalid.")
    if not np.allclose((num["upper"]-num["lower"]).to_numpy(float), num["width"].to_numpy(float), rtol=0, atol=1e-10):
        raise RuntimeError("Paper1 width != upper-lower.")
    return df


def reference_table(p1):
    return (p1.groupby("true_L",as_index=False)
            .agg(n_samples=("sample_id","size"), n_dates=("date","nunique"),
                 q50_median=("q50","median"), width_median=("width","median"),
                 width_q25=("width",lambda x: float(pd.Series(x).quantile(.25))),
                 width_q75=("width",lambda x: float(pd.Series(x).quantile(.75))))
            .sort_values("true_L").reset_index(drop=True))


def nearest_unique_distance(query, unique_l):
    pos = np.searchsorted(unique_l, query, side="left")
    li = np.clip(pos-1,0,len(unique_l)-1)
    ri = np.clip(pos,0,len(unique_l)-1)
    return np.minimum(np.abs(query-unique_l[li]), np.abs(query-unique_l[ri]))


def bin_table(p1, wapp):
    edges = np.asarray(REPORT_BIN_EDGES,float)
    labels = [f"[{edges[i]:.3f},{edges[i+1]:.3f})" for i in range(len(edges)-1)]
    pb = pd.cut(pd.to_numeric(p1["true_L"]), bins=edges, labels=labels, include_lowest=True, right=False)
    wb = pd.cut(pd.to_numeric(wapp["L_power_proxy"]), bins=edges, labels=labels, include_lowest=True, right=False)
    rows=[]
    for label in labels:
        pm = pb.astype(str).eq(label); wm = wb.astype(str).eq(label)
        rows.append({"bin":label,"paper1_samples":int(pm.sum()),"paper1_dates":int(p1.loc[pm,"date"].nunique()),
                     "paper1_unique_true_L":int(p1.loc[pm,"true_L"].nunique()),"wapp_days":int(wm.sum()),
                     "wapp_fraction_valid":float(wm.mean())})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(description="P2-0C-2 Paper1 support-domain / extrapolation-risk audit")
    ap.add_argument("--power-bridge",required=True,type=Path)
    ap.add_argument("--paper1-dev-cqr",required=True,type=Path)
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/paper2_uncertainty_rl_v1/p2_0c_2_paper1_support_audit_v1"))
    a=ap.parse_args()
    bp=a.power_bridge.expanduser().resolve(); pp=a.paper1_dev_cqr.expanduser().resolve(); out=a.output_dir.expanduser().resolve()
    for p in (bp,pp):
        if not p.exists(): raise FileNotFoundError(p)

    print("[1/7] Load + validate P2-0C-1B power-loss proxy")
    bridge=load_bridge(bp)
    print("[2/7] Load authorized Paper1 DECISION_DEVELOPMENT CQR asset")
    p1=load_p1(pp)
    print("[3/7] Build Paper1 unique-target support reference")
    ref=reference_table(p1)
    p1_l=pd.to_numeric(p1["true_L"],errors="raise").to_numpy(float)
    p1_sorted=np.sort(p1_l); unique_l=np.sort(pd.unique(p1_l))
    p1_min=float(p1_sorted[0]); p1_max=float(p1_sorted[-1]); zero_eligible=bool(p1_min<=ZERO_BOUNDARY_TOL)

    print("[4/7] Audit each valid WAPP query against Paper1 support")
    vm=bridge["bridge_valid"].fillna(False).astype(bool); valid=bridge.loc[vm].copy()
    q=pd.to_numeric(valid["L_power_proxy"],errors="raise").to_numpy(float)
    strict=(q>=p1_min)&(q<=p1_max); zero=np.isclose(q,0.0,rtol=0,atol=1e-15); zero_sup=zero&zero_eligible; range_zero=strict|zero_sup
    nn=nearest_unique_distance(q,unique_l)
    percentile=np.searchsorted(p1_sorted,q,side="right")/float(len(p1_sorted))

    local_n=np.zeros(len(q),int); local_dates=np.zeros(len(q),int); local_unique=np.zeros(len(q),int)
    local_q50=np.full(len(q),np.nan); local_wmed=np.full(len(q),np.nan); local_wiqr=np.full(len(q),np.nan)
    for i,v in enumerate(q):
        m=np.abs(p1_l-v)<=LOCAL_RADIUS
        local_n[i]=int(m.sum()); local_dates[i]=int(p1.loc[m,"date"].nunique()); local_unique[i]=int(p1.loc[m,"true_L"].nunique())
        if m.any():
            local_q50[i]=float(pd.to_numeric(p1.loc[m,"q50"]).median())
            w=pd.to_numeric(p1.loc[m,"width"]); local_wmed[i]=float(w.median()); local_wiqr[i]=float(w.quantile(.75)-w.quantile(.25))
    local_pass=(local_n>=LOCAL_MIN_SAMPLES)&(local_dates>=LOCAL_MIN_DATES)

    valid["paper1_strict_range_supported"]=strict; valid["paper1_zero_boundary_supported"]=zero_sup
    valid["paper1_range_or_zero_supported"]=range_zero; valid["paper1_empirical_percentile"]=percentile
    valid["nearest_unique_true_L_distance"]=nn; valid["local_radius"]=LOCAL_RADIUS
    valid["local_sample_count"]=local_n; valid["local_date_count"]=local_dates; valid["local_unique_true_L_count"]=local_unique
    valid["local_q50_median_diagnostic"]=local_q50; valid["local_width_median_diagnostic"]=local_wmed
    valid["local_width_iqr_diagnostic"]=local_wiqr; valid["local_support_pass"]=local_pass

    print("[5/7] Build marginal/bin and local-support diagnostics")
    bins=bin_table(p1,valid); occupied=bins[bins["wapp_days"]>0]
    range_fraction=float(range_zero.mean()); local_fraction=float(local_pass.mean()); nn95=float(np.quantile(nn,.95)); nn99=float(np.quantile(nn,.99))

    print("[6/7] Evaluate predeclared support gates")
    g_range=range_fraction>=RANGE_SUPPORT_FRACTION_GATE; g_local=local_fraction>=LOCAL_SUPPORT_FRACTION_GATE
    g95=nn95<=NN_UNIQUE_P95_GATE; g99=nn99<=NN_UNIQUE_P99_GATE; gzero=(not zero.any()) or zero_eligible
    all_g=bool(g_range and g_local and g95 and g99 and gzero)

    summary={
      "stage":"P2-0C-2","audit_only":True,"paper1_emulator_fitted":False,"paper1_emulator_called":False,"rl_state_generated":False,
      "authorized_paper1_asset":{"role":EXPECTED_ROLE,"rows":int(len(p1)),"dates":int(p1['date'].nunique()),"uses_random_test":False,"uses_sealed_dates":False,
          "conditioning_variables":["true_L"],"wapp_irradiance_used":False,"reason_no_irradiance":"Paper1 irradiance units/semantics have not been proven commensurate with WAPP physical irradiance."},
      "paper1_true_L_reference":{"distribution":qstats(p1_l),"unique_true_L_levels":int(len(unique_l)),"empirical_min":p1_min,"empirical_max":p1_max,
          "zero_boundary_tolerance":ZERO_BOUNDARY_TOL,"zero_boundary_eligible":zero_eligible},
      "wapp_queries":{"valid_days":int(len(valid)),"L_power_proxy_distribution":qstats(q),"zero_loss_days":int(zero.sum()),"strict_empirical_range_days":int(strict.sum()),
          "range_or_zero_supported_days":int(range_zero.sum()),"range_or_zero_supported_fraction":range_fraction,"paper1_empirical_percentile_distribution":qstats(percentile)},
      "local_support":{"radius_abs_L":LOCAL_RADIUS,"min_samples":LOCAL_MIN_SAMPLES,"min_dates":LOCAL_MIN_DATES,"supported_days":int(local_pass.sum()),"supported_fraction":local_fraction,
          "sample_count_distribution":qstats(local_n),"date_count_distribution":qstats(local_dates),"unique_true_L_count_distribution":qstats(local_unique),
          "local_width_median_distribution_diagnostic":qstats(local_wmed),"local_width_iqr_distribution_diagnostic":qstats(local_wiqr)},
      "nearest_unique_target":{"distance_distribution":qstats(nn),"p95":nn95,"p99":nn99,"p95_gate":NN_UNIQUE_P95_GATE,"p99_gate":NN_UNIQUE_P99_GATE},
      "occupied_reporting_bins":{"bins":int(len(occupied)),"minimum_paper1_samples_in_occupied_bin":int(occupied['paper1_samples'].min()),"minimum_paper1_dates_in_occupied_bin":int(occupied['paper1_dates'].min())},
      "gates":{"range_support_fraction_gate":RANGE_SUPPORT_FRACTION_GATE,"range_support_fraction_pass":bool(g_range),"local_support_fraction_gate":LOCAL_SUPPORT_FRACTION_GATE,
          "local_support_fraction_pass":bool(g_local),"nearest_unique_p95_pass":bool(g95),"nearest_unique_p99_pass":bool(g99),"physical_zero_boundary_pass":bool(gzero),"all_primary_support_gates_pass":all_g},
      "interpretation_limits":[
          "Passing scalar/local support does not prove cross-domain image generalization.",
          "This audit only establishes support for conditional resampling of Paper1 perception errors given L.",
          "The WAPP marginal L distribution may differ strongly from the Paper1 marginal distribution; that is covariate shift, not automatically conditional out-of-support.",
          "Temporal dependence of Paper1 residuals is not modeled here and must be audited before generating longitudinal perception trajectories.",
          "Width-vs-q50 confounding / within-q50 width-shuffle value must be tested later before attributing all UA-PPO gains uniquely to uncertainty."
      ]}

    print("[7/7] Write audit outputs")
    out.mkdir(parents=True,exist_ok=True)
    valid.to_csv(out/"daily_support_audit.csv",index=False,encoding="utf-8-sig")
    ref.to_csv(out/"paper1_support_reference.csv",index=False,encoding="utf-8-sig")
    bins.to_csv(out/"local_support_bins.csv",index=False,encoding="utf-8-sig")
    with (out/"audit_summary.json").open("w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(out/"daily_support_audit.csv"); print(out/"paper1_support_reference.csv"); print(out/"local_support_bins.csv"); print(out/"audit_summary.json")
    print("IMPORTANT: support audit only. Do NOT fit/call the Paper1 emulator unless support gates are reviewed and passed.")

if __name__ == "__main__":
    main()
