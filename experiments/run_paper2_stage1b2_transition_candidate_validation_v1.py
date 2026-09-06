#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_DRY = 494
EXPECTED_RAIN_DAILY = 201
MIN_DRY_POOL = 20
MIN_RAIN_POOL = 10
MIN_DRY_PER_YEAR = 150
MIN_RAIN_PER_YEAR = 40
SIMPLICITY_TOL = 0.05

STATE_BINS = [-np.inf, 0.01, 0.03, 0.06, np.inf]
STATE_LABELS = ["<0.01", "0.01-0.03", "0.03-0.06", ">=0.06"]

DRY_CANDIDATES = ["D0_GLOBAL", "D1_SEASON2", "D2_STATE4"]
RAIN_CANDIDATES = ["R0_GLOBAL", "R1_PATTERN3", "R2_STATE4"]
COMPLEXITY = {
    "D0_GLOBAL": 0, "D1_SEASON2": 1, "D2_STATE4": 1,
    "R0_GLOBAL": 0, "R1_PATTERN3": 1, "R2_STATE4": 1,
}

def parse_bool(s, name):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.isna().any() or (~x.isin([0,1])).any():
            raise RuntimeError(f"{name}: bad boolean column")
        return x.astype(int).astype(bool)
    mp = {"true":True,"false":False,"1":True,"0":False,"yes":True,"no":False,"y":True,"n":False}
    x = s.astype(str).str.strip().str.lower().map(mp)
    if x.isna().any():
        raise RuntimeError(f"{name}: bad boolean values")
    return x.astype(bool)

def load_ledger(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"date","audit_period","state_valid","L_power_proxy","rain_day","rain_mm_day"}
    miss = need.difference(df.columns)
    if miss: raise RuntimeError(f"ledger missing {sorted(miss)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["state_valid"] = parse_bool(df["state_valid"],"state_valid")
    df["rain_day"] = parse_bool(df["rain_day"],"rain_day")
    df["L_power_proxy"] = pd.to_numeric(df["L_power_proxy"], errors="coerce")
    df["rain_mm_day"] = pd.to_numeric(df["rain_mm_day"], errors="coerce")
    return df

def load_trans(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"transition_index","source_date","dest_date","source_rain_day","dest_rain_day",
            "transition_class","delta_L_power_proxy"}
    miss = need.difference(df.columns)
    if miss: raise RuntimeError(f"transition missing {sorted(miss)}")
    df["source_date"] = pd.to_datetime(df["source_date"], errors="raise").dt.normalize()
    df["dest_date"] = pd.to_datetime(df["dest_date"], errors="raise").dt.normalize()
    df["source_rain_day"] = parse_bool(df["source_rain_day"],"source_rain_day")
    df["dest_rain_day"] = parse_bool(df["dest_rain_day"],"dest_rain_day")
    df["delta_L_power_proxy"] = pd.to_numeric(df["delta_L_power_proxy"], errors="coerce")
    return df

def load_rain_events(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"rain_event_id","start_date","end_date","rain_mm_total","S_pre","S_post","resolved","cleaning_confounded"}
    miss = need.difference(df.columns)
    if miss: raise RuntimeError(f"rain event missing {sorted(miss)}")
    for c in ["start_date","end_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
    df["resolved"] = parse_bool(df["resolved"],"resolved")
    df["cleaning_confounded"] = parse_bool(df["cleaning_confounded"],"cleaning_confounded")
    for c in ["rain_mm_total","S_pre","S_post"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_context(t, ledger):
    src = ledger[["date","audit_period","L_power_proxy"]].rename(
        columns={"date":"source_date","audit_period":"source_period","L_power_proxy":"source_L"})
    dst = ledger[["date","L_power_proxy"]].rename(
        columns={"date":"dest_date","L_power_proxy":"dest_L"})
    x = t.merge(src,on="source_date",how="left",validate="many_to_one").merge(
        dst,on="dest_date",how="left",validate="many_to_one")
    x["dest_month"] = x["dest_date"].dt.month.astype(int)
    x["season2"] = np.where(
        x["dest_month"].isin([11,12,1,2,3,4]), "NOV_APR", "MAY_OCT")
    x["state4"] = pd.cut(
        x["source_L"], bins=STATE_BINS, labels=STATE_LABELS, right=False).astype(str)
    sr = x["source_rain_day"].to_numpy(bool)
    dr = x["dest_rain_day"].to_numpy(bool)
    patt = np.full(len(x),"NOT_RAIN_AFFECTED",dtype=object)
    patt[(~sr)&dr] = "ONSET"
    patt[sr&dr] = "CONTINUATION"
    patt[sr&(~dr)] = "POST_RAIN"
    x["rain_pattern3"] = patt
    return x

def crps_emp(pool, y):
    p = np.asarray(pool,float)
    return float(np.mean(np.abs(p-y)) - 0.5*np.mean(np.abs(p[:,None]-p[None,:])))

def pred_metrics(pool, y):
    q05,q50,q95 = np.quantile(pool,[0.05,0.5,0.95])
    return {
        "crps": crps_emp(pool,y),
        "median_abs_error": float(abs(q50-y)),
        "covered90": bool(q05 <= y <= q95),
        "width90": float(q95-q05),
    }

def key_for(row, cand):
    if cand.endswith("GLOBAL"): return "GLOBAL"
    if cand=="D1_SEASON2": return str(row["season2"])
    if cand in ["D2_STATE4","R2_STATE4"]: return str(row["state4"])
    if cand=="R1_PATTERN3": return str(row["rain_pattern3"])
    raise KeyError(cand)

def min_pool(cand):
    return MIN_DRY_POOL if cand.startswith("D") else MIN_RAIN_POOL

def one_fold(train,test,cand,train_period,test_period):
    groups={}
    if cand.endswith("GLOBAL"):
        groups["GLOBAL"] = train["delta_L_power_proxy"].to_numpy(float)
    else:
        col={"D1_SEASON2":"season2","D2_STATE4":"state4","R1_PATTERN3":"rain_pattern3","R2_STATE4":"state4"}[cand]
        for k,g in train.groupby(col,dropna=False):
            groups[str(k)] = g["delta_L_power_proxy"].to_numpy(float)
    rows=[]
    for _,r in test.iterrows():
        k=key_for(r,cand)
        pool=groups.get(k)
        ok = pool is not None and len(pool)>=min_pool(cand) and np.isfinite(pool).all()
        rec={"candidate":cand,"train_period":train_period,"test_period":test_period,
             "transition_index":int(r["transition_index"]),"group_key":k,
             "train_pool_n":0 if pool is None else int(len(pool)),
             "supported":bool(ok),"observed_delta_L":float(r["delta_L_power_proxy"])}
        if ok: rec.update(pred_metrics(pool,float(r["delta_L_power_proxy"])))
        else: rec.update({"crps":np.nan,"median_abs_error":np.nan,"covered90":False,"width90":np.nan})
        rows.append(rec)
    pred=pd.DataFrame(rows)
    sup=pred[pred["supported"]]
    fold={
        "candidate":cand,"train_period":train_period,"test_period":test_period,
        "train_n":int(len(train)),"test_n":int(len(test)),
        "supported_n":int(len(sup)),
        "support_fraction":float(len(sup)/len(test)) if len(test) else np.nan,
        "crps_mean":float(sup["crps"].mean()) if len(sup) else np.nan,
        "median_abs_error_mean":float(sup["median_abs_error"].mean()) if len(sup) else np.nan,
        "coverage90":float(sup["covered90"].mean()) if len(sup) else np.nan,
        "width90_mean":float(sup["width90"].mean()) if len(sup) else np.nan,
    }
    return fold,pred

def run_cv(data,cands):
    folds=[]; preds=[]
    for tr,te in [("YEAR1","YEAR2"),("YEAR2","YEAR1")]:
        train=data[data["source_period"].eq(tr)].copy()
        test=data[data["source_period"].eq(te)].copy()
        for cand in cands:
            f,p=one_fold(train,test,cand,tr,te)
            folds.append(f); preds.append(p)
    return pd.DataFrame(folds),pd.concat(preds,ignore_index=True)

def summarize(folds,cands):
    rows=[]
    for cand in cands:
        g=folds[folds["candidate"].eq(cand)]
        rows.append({
            "candidate":cand,
            "full_support_both_folds":bool(len(g)==2 and np.isclose(g["support_fraction"],1.0).all()),
            "support_fraction_macro":float(g["support_fraction"].mean()),
            "crps_macro":float(g["crps_mean"].mean()),
            "median_abs_error_macro":float(g["median_abs_error_mean"].mean()),
            "coverage90_macro":float(g["coverage90"].mean()),
            "coverage90_abs_error":float(abs(g["coverage90"].mean()-0.90)),
            "width90_macro":float(g["width90_mean"].mean()),
            "complexity_rank":int(COMPLEXITY[cand]),
        })
    return pd.DataFrame(rows)

def recommend(summary):
    v=summary[summary["full_support_both_folds"] & np.isfinite(summary["crps_macro"])].copy()
    if len(v)==0: return {"recommended":None,"reason":"no full-support candidate"}
    best=float(v["crps_macro"].min())
    near=v[v["crps_macro"] <= best*(1+SIMPLICITY_TOL)].copy()
    near=near.sort_values(["complexity_rank","crps_macro","coverage90_abs_error"])
    r=near.iloc[0]
    return {
        "recommended":str(r["candidate"]),
        "best_crps":best,
        "simplicity_threshold_crps":float(best*(1+SIMPLICITY_TOL)),
        "chosen_crps":float(r["crps_macro"]),
        "chosen_complexity_rank":int(r["complexity_rank"]),
        "rule":"100% bidirectional support; simplest candidate within 5% of best CRPS",
    }

def group_support(data,kind):
    defs=[("SEASON2","season2"),("STATE4","state4")] if kind=="DRY" else [("PATTERN3","rain_pattern3"),("STATE4","state4")]
    rows=[]
    for fam,col in defs:
        for per,pg in data.groupby("source_period"):
            for key,g in pg.groupby(col,dropna=False):
                rows.append({"kind":kind,"family":fam,"source_period":str(per),
                             "group_key":str(key),"n":int(len(g)),
                             "delta_mean":float(g["delta_L_power_proxy"].mean()),
                             "delta_median":float(g["delta_L_power_proxy"].median()),
                             "fraction_negative":float((g["delta_L_power_proxy"]<0).mean())})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(description="P2-1B-2 natural-dynamics candidate validation.")
    ap.add_argument("--master-ledger",required=True,type=Path)
    ap.add_argument("--transition-audit",required=True,type=Path)
    ap.add_argument("--rain-event-audit",required=True,type=Path)
    ap.add_argument("--output-dir",type=Path,default=Path(
        "outputs/paper2_uncertainty_rl_v1/p2_1b2_transition_candidate_validation_v1"))
    args=ap.parse_args()

    lp=args.master_ledger.expanduser().resolve()
    tp=args.transition_audit.expanduser().resolve()
    rp=args.rain_event_audit.expanduser().resolve()
    for p in [lp,tp,rp]:
        if not p.exists(): raise FileNotFoundError(p)

    print("[1/8] Load frozen P2-1A assets")
    ledger=load_ledger(lp); trans=load_trans(tp); rain_events=load_rain_events(rp)
    x=add_context(trans,ledger)

    print("[2/8] Build dry and daily rain samples")
    dry=x[x["transition_class"].eq("DRY_NATURAL")].copy()
    rain=x[x["transition_class"].eq("RAIN_AFFECTED")].copy()
    dry_gate=bool(len(dry)==EXPECTED_DRY and np.isfinite(dry["delta_L_power_proxy"]).all())
    rain_gate=bool(len(rain)==EXPECTED_RAIN_DAILY and np.isfinite(rain["delta_L_power_proxy"]).all())
    dry_counts=dry["source_period"].value_counts().to_dict()
    rain_counts=rain["source_period"].value_counts().to_dict()
    dry_year_gate=all(dry_counts.get(y,0)>=MIN_DRY_PER_YEAR for y in ["YEAR1","YEAR2"])
    rain_year_gate=all(rain_counts.get(y,0)>=MIN_RAIN_PER_YEAR for y in ["YEAR1","YEAR2"])

    print("[3/8] Audit candidate group support")
    gsup=pd.concat([group_support(dry,"DRY"),group_support(rain,"RAIN")],ignore_index=True)

    print("[4/8] Bidirectional OOT dry validation")
    dry_folds,dry_preds=run_cv(dry,DRY_CANDIDATES)
    dry_summary=summarize(dry_folds,DRY_CANDIDATES)
    dry_rec=recommend(dry_summary)

    print("[5/8] Bidirectional OOT rain validation")
    rain_folds,rain_preds=run_cv(rain,RAIN_CANDIDATES)
    rain_summary=summarize(rain_folds,RAIN_CANDIDATES)
    rain_rec=recommend(rain_summary)

    print("[6/8] Build event-level rain sanity reference")
    rr=rain_events[rain_events["resolved"] & (~rain_events["cleaning_confounded"])].copy()
    rr=rr[np.isfinite(rr[["rain_mm_total","S_pre","S_post"]].to_numpy(float)).all(axis=1)].copy()
    rr["delta_S"]=rr["S_post"]-rr["S_pre"]
    rain_ref={
        "rows_total":int(len(rain_events)),
        "resolved_unconfounded_finite":int(len(rr)),
        "fraction_soiling_decreased":float((rr["delta_S"]<0).mean()) if len(rr) else np.nan,
        "delta_S_median":float(rr["delta_S"].median()) if len(rr) else np.nan,
        "rho_S_pre_vs_removal":float(rr["S_pre"].corr(-rr["delta_S"],method="spearman")) if len(rr)>=3 else np.nan,
        "rho_rain_mm_vs_removal":float(rr["rain_mm_total"].corr(-rr["delta_S"],method="spearman")) if len(rr)>=3 else np.nan,
    }

    print("[7/8] Evaluate readiness")
    dry_valid=dry_summary[dry_summary["full_support_both_folds"]]["candidate"].tolist()
    rain_valid=rain_summary[rain_summary["full_support_both_folds"]]["candidate"].tolist()
    all_pass=bool(dry_gate and rain_gate and dry_year_gate and rain_year_gate and dry_valid and rain_valid
                  and dry_rec["recommended"] is not None and rain_rec["recommended"] is not None)

    summary={
        "stage":"P2-1B-2",
        "audit_and_model_selection_only":True,
        "counterfactual_environment_built":False,
        "rl_started":False,
        "predeclared_candidates":{"dry":DRY_CANDIDATES,"rain_daily":RAIN_CANDIDATES},
        "validation_protocol":{
            "folds":["TRAIN_YEAR1_TEST_YEAR2","TRAIN_YEAR2_TEST_YEAR1"],
            "proper_score":"EMPIRICAL_CRPS",
            "no_fallback":True,
            "dry_min_group_pool":MIN_DRY_POOL,
            "rain_min_group_pool":MIN_RAIN_POOL,
            "simplicity_tolerance_relative_to_best_crps":SIMPLICITY_TOL,
        },
        "sample_support":{
            "dry_total":int(len(dry)),"dry_by_year":{str(k):int(v) for k,v in dry_counts.items()},
            "rain_daily_total":int(len(rain)),"rain_daily_by_year":{str(k):int(v) for k,v in rain_counts.items()},
        },
        "dry_candidate_summary":dry_summary.to_dict(orient="records"),
        "rain_candidate_summary":rain_summary.to_dict(orient="records"),
        "dry_recommendation":dry_rec,
        "rain_recommendation":rain_rec,
        "rain_event_physical_reference":rain_ref,
        "primary_gates":{
            "dry_count_pass":dry_gate,
            "rain_daily_count_pass":rain_gate,
            "dry_year_support_pass":bool(dry_year_gate),
            "rain_year_support_pass":bool(rain_year_gate),
            "at_least_one_full_support_dry_candidate":bool(len(dry_valid)>=1),
            "at_least_one_full_support_rain_candidate":bool(len(rain_valid)>=1),
            "all_primary_gates_pass":all_pass,
        },
        "next_step_if_pass":"Review recommendations, then freeze stochastic dry/rain kernels before CLEAN-action mechanics. Do not train PPO yet.",
        "next_step_if_fail":"Do not build simulator. Diagnose candidate support or cross-year shift."
    }

    print("[8/8] Write outputs")
    out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    dry_folds.to_csv(out/"dry_cv_fold_metrics.csv",index=False,encoding="utf-8-sig")
    dry_summary.to_csv(out/"dry_candidate_summary.csv",index=False,encoding="utf-8-sig")
    dry_preds.to_csv(out/"dry_cv_predictions.csv",index=False,encoding="utf-8-sig")
    rain_folds.to_csv(out/"rain_cv_fold_metrics.csv",index=False,encoding="utf-8-sig")
    rain_summary.to_csv(out/"rain_candidate_summary.csv",index=False,encoding="utf-8-sig")
    rain_preds.to_csv(out/"rain_cv_predictions.csv",index=False,encoding="utf-8-sig")
    gsup.to_csv(out/"candidate_group_support.csv",index=False,encoding="utf-8-sig")
    with (out/"audit_summary.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print(out/"audit_summary.json")
    return 0

if __name__=="__main__":
    sys.exit(main())
