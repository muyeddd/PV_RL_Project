#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P2-0C-3C: generate a fixed multi-seed 729-day WAPP perception bank.
Frozen emulator: boundary-preserving final-output transport + BLOCK10.
No RL training, no WAPP irradiance conditioning, no fallback, no qhat tuning.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_P1_N=1844; EXPECTED_P1_DATES=12; EXPECTED_ROLE='DECISION_DEVELOPMENT'; EXPECTED_WAPP_VALID=729
RADIUS=0.01; MIN_SAMPLES=20; MIN_DATES=3; BW=0.005; BLOCK_MIN=10
DEV_SEED=20260906; FORMAL_SEEDS=[20260907,20260908,20260909,20260910,20260911]; ALL_SEEDS=[DEV_SEED]+FORMAL_SEEDS

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def spearman(a,b):
    a=pd.to_numeric(pd.Series(a),errors='coerce'); b=pd.to_numeric(pd.Series(b),errors='coerce')
    ok=np.isfinite(a)&np.isfinite(b)
    if int(ok.sum())<3 or a[ok].nunique()<2 or b[ok].nunique()<2: return float('nan')
    return float(a[ok].corr(b[ok],method='spearman'))

def qstats(v):
    x=pd.to_numeric(pd.Series(v),errors='coerce'); x=x[np.isfinite(x)]
    if len(x)==0:return {}
    q=x.quantile([.01,.05,.25,.5,.75,.95,.99])
    return {'n':int(len(x)),'mean':float(x.mean()),'std':float(x.std(ddof=1)) if len(x)>1 else 0.0,'min':float(x.min()),'q01':float(q.loc[.01]),'q05':float(q.loc[.05]),'q25':float(q.loc[.25]),'q50':float(q.loc[.5]),'q75':float(q.loc[.75]),'q95':float(q.loc[.95]),'q99':float(q.loc[.99]),'max':float(x.max())}

def load_p1(path):
    d=pd.read_csv(path,encoding='utf-8-sig')
    req={'sample_id','date','timestamp','role','true_L','q50','lower','upper','width','covered','lower_clipped','upper_clipped'}
    miss=req-set(d.columns)
    if miss: raise RuntimeError(f'Missing Paper1 columns: {sorted(miss)}')
    if len(d)!=EXPECTED_P1_N: raise RuntimeError(f'Expected {EXPECTED_P1_N} Paper1 rows, found {len(d)}')
    if set(d['role'].astype(str))!={EXPECTED_ROLE}: raise PermissionError('Only DECISION_DEVELOPMENT is authorized.')
    d['date']=pd.to_datetime(d['date'],errors='raise').dt.strftime('%Y-%m-%d'); d['timestamp']=pd.to_datetime(d['timestamp'],errors='raise')
    if d['date'].nunique()!=EXPECTED_P1_DATES: raise RuntimeError('Unexpected Paper1 date count.')
    if not np.array_equal(d['timestamp'].dt.strftime('%Y-%m-%d').to_numpy(),d['date'].to_numpy()): raise RuntimeError('Paper1 timestamp/date mismatch.')
    if d['sample_id'].isna().any() or d['sample_id'].duplicated().any(): raise RuntimeError('sample_id must be unique/non-null.')
    for c in ['true_L','q50','lower','upper','width']: d[c]=pd.to_numeric(d[c],errors='raise')
    if not np.isfinite(d[['true_L','q50','lower','upper','width']].to_numpy(float)).all(): raise RuntimeError('Non-finite Paper1 values.')
    if not ((d['lower']<=d['q50'])&(d['q50']<=d['upper'])).all(): raise RuntimeError('Paper1 interval ordering violated.')
    if not np.allclose(d['upper']-d['lower'],d['width'],atol=1e-10,rtol=0): raise RuntimeError('Paper1 width mismatch.')
    lc=np.isclose(d['lower'].to_numpy(float),0.0,atol=1e-12,rtol=0)
    if not np.array_equal(lc,d['lower_clipped'].astype(bool).to_numpy()): raise RuntimeError('lower_clipped inconsistent.')
    if d['upper_clipped'].astype(bool).any(): raise RuntimeError('Unexpected Paper1 upper clipping.')
    d['block10_id']=d['date']+'|'+d['timestamp'].dt.floor(f'{BLOCK_MIN}min').astype(str)
    return d.reset_index(drop=True)

def load_wapp(path):
    d=pd.read_csv(path,encoding='utf-8-sig')
    req={'date','L_power_proxy','bridge_valid','power_bridge_model'}; miss=req-set(d.columns)
    if miss: raise RuntimeError(f'Missing WAPP columns: {sorted(miss)}')
    d=d.loc[d['bridge_valid'].fillna(False).astype(bool)].copy().reset_index(drop=True)
    if len(d)!=EXPECTED_WAPP_VALID: raise RuntimeError(f'Expected {EXPECTED_WAPP_VALID} valid WAPP days, found {len(d)}')
    if set(d['power_bridge_model'].astype(str))!={'COMMON_TEMPERATURE_PVWATTS_RATIO'}: raise RuntimeError('Unexpected WAPP bridge model.')
    d['date']=pd.to_datetime(d['date'],errors='raise').dt.strftime('%Y-%m-%d'); d['L_power_proxy']=pd.to_numeric(d['L_power_proxy'],errors='raise')
    if d['date'].duplicated().any() or not np.isfinite(d['L_power_proxy']).all(): raise RuntimeError('Invalid WAPP date/L values.')
    return d

def weights(dist):
    w=np.exp(-.5*(dist/BW)**2); s=float(w.sum())
    if not np.isfinite(s) or s<=0: raise RuntimeError('Invalid Gaussian weights.')
    return w/s

def build_support(p1,wapp):
    sl=p1['true_L'].to_numpy(float); sd=p1['date'].to_numpy(str); sb=p1['block10_id'].to_numpy(str)
    rows=[]; maps=[]
    for i,r in wapp.iterrows():
        L=float(r['L_power_proxy']); da=np.abs(sl-L); idx=np.flatnonzero(da<=RADIUS); dates=np.unique(sd[idx]); blocks=np.unique(sb[idx])
        ok=len(idx)>=MIN_SAMPLES and len(dates)>=MIN_DATES and len(blocks)>=1
        bm={}
        if ok:
            cand_blocks=sb[idx]
            for b in blocks:
                m=cand_blocks==b; bidx=idx[m]; bm[str(b)]={'indices':bidx,'distances':da[bidx]}
        rows.append({'wapp_row':int(i),'date':r['date'],'L_true':L,'candidate_samples':int(len(idx)),'candidate_dates':int(len(dates)),'candidate_blocks':int(len(blocks)),'support_ok':bool(ok)})
        maps.append({'support_ok':bool(ok),'blocks':bm})
    return pd.DataFrame(rows),maps

def choose_source(rng,qmap):
    if not qmap['support_ok']: raise RuntimeError('Unsupported query entered generation.')
    bid=str(rng.choice(sorted(qmap['blocks'].keys()))); b=qmap['blocks'][bid]
    idx=np.asarray(b['indices'],int); dist=np.asarray(b['distances'],float)
    return int(rng.choice(idx,p=weights(dist)))

def transport(s,L):
    sl=float(s['true_L']); q50=float(np.clip(L+(float(s['q50'])-sl),0,1)); upper=float(np.clip(L+(float(s['upper'])-sl),0,1))
    lower=0.0 if bool(s['lower_clipped']) else float(np.clip(L+(float(s['lower'])-sl),0,1))
    if lower>q50+1e-12 or q50>upper+1e-12: raise RuntimeError('Generated interval ordering violated.')
    return q50,lower,upper

def generate(p1,wapp,support,maps,seed,role):
    rng=np.random.default_rng(seed); out=[]
    for i,r in wapp.iterrows():
        if not bool(support.loc[i,'support_ok']): raise RuntimeError(f'No fallback allowed at WAPP row {i}.')
        si=choose_source(rng,maps[i]); s=p1.iloc[si]; L=float(r['L_power_proxy']); q50,lo,up=transport(s,L); width=up-lo; err=q50-L
        out.append({'date':r['date'],'L_true':L,'q50':q50,'lower':lo,'upper':up,'width':width,'error':err,'abs_error':abs(err),'covered':bool(lo<=L<=up),'lower_clipped':bool(np.isclose(lo,0,atol=1e-12,rtol=0)),'perception_seed':int(seed),'trajectory_role':role,'source_sample_id':s['sample_id'],'source_date':s['date'],'source_block10_id':s['block10_id'],'source_true_L':float(s['true_L']),'source_lower_clipped':bool(s['lower_clipped']),'source_covered':bool(s['covered']),'candidate_samples':int(support.loc[i,'candidate_samples']),'candidate_dates':int(support.loc[i,'candidate_dates']),'candidate_blocks':int(support.loc[i,'candidate_blocks'])})
    d=pd.DataFrame(out)
    if len(d)!=EXPECTED_WAPP_VALID: raise RuntimeError('Trajectory row count mismatch.')
    return d

def audit(d):
    num=d[['L_true','q50','lower','upper','width','error','abs_error']].to_numpy(float)
    finite=bool(np.isfinite(num).all()); order=bool(((d['lower']>=0)&(d['lower']<=d['q50'])&(d['q50']<=d['upper'])&(d['upper']<=1)).all()); wid=bool(np.allclose(d['upper']-d['lower'],d['width'],atol=1e-12,rtol=0))
    src_clip=d['source_lower_clipped'].astype(bool).to_numpy(); lo=d['lower'].to_numpy(float); preserve=bool(np.isclose(lo[src_clip],0,atol=1e-12,rtol=0).all())
    return {'rows':int(len(d)),'finite_pass':finite,'ordering_pass':order,'width_identity_pass':wid,'source_lower_clipped_preserved_pass':preserve,'bias':float(d['error'].mean()),'mae':float(d['abs_error'].mean()),'width_median':float(d['width'].median()),'coverage':float(d['covered'].astype(bool).mean()),'lower_clipped_fraction':float(d['lower_clipped'].astype(bool).mean()),'rho_width_abs_error':spearman(d['width'],d['abs_error']),'unique_source_dates':int(d['source_date'].nunique()),'unique_source_blocks':int(d['source_block10_id'].nunique()),'max_source_date_share':float(d['source_date'].value_counts(normalize=True).max()),'max_source_block_share':float(d['source_block10_id'].value_counts(normalize=True).max())}

def main():
    ap=argparse.ArgumentParser(description='P2-0C-3C fixed WAPP perception bank generation.')
    ap.add_argument('--paper1-dev-cqr',required=True,type=Path); ap.add_argument('--wapp-power-bridge',required=True,type=Path); ap.add_argument('--output-dir',type=Path,default=Path('outputs/paper2_uncertainty_rl_v1/p2_0c_3c_perception_trajectory_bank_v1'))
    a=ap.parse_args(); p1p=a.paper1_dev_cqr.expanduser().resolve(); wp=a.wapp_power_bridge.expanduser().resolve(); out=a.output_dir.expanduser().resolve()
    for p in [p1p,wp]:
        if not p.exists(): raise FileNotFoundError(p)
    print('[1/8] Load frozen Paper1 DEV and WAPP bridge assets'); p1=load_p1(p1p); w=load_wapp(wp)
    print('[2/8] Audit full-pool support for all 729 actual WAPP queries'); support,maps=build_support(p1,w); sf=float(support['support_ok'].astype(bool).mean())
    if sf!=1.0: raise RuntimeError(f'3C requires 100% support; found {int((~support.support_ok.astype(bool)).sum())} unsupported queries. No fallback allowed.')
    print('[3/8] Freeze perception-seed roles'); roles={DEV_SEED:'DEV',**{s:'FORMAL_EVAL' for s in FORMAL_SEEDS}}
    print('[4/8] Generate 6 fixed 729-day BLOCK10 perception trajectories'); out.mkdir(parents=True,exist_ok=True); parts=[]; metrics=[]
    for seed in ALL_SEEDS:
        d=generate(p1,w,support,maps,seed,roles[seed]); au=audit(d); au.update({'perception_seed':seed,'trajectory_role':roles[seed]}); metrics.append(au); d.to_csv(out/f'trajectory_seed_{seed}.csv',index=False,encoding='utf-8-sig'); parts.append(d)
    bank=pd.concat(parts,ignore_index=True); sm=pd.DataFrame(metrics)
    print('[5/8] Audit structural invariants and multi-seed completeness'); expected=EXPECTED_WAPP_VALID*len(ALL_SEEDS); complete=bool(len(bank)==expected and (sm['rows']==EXPECTED_WAPP_VALID).all()); structural=bool(sm[['finite_pass','ordering_pass','width_identity_pass','source_lower_clipped_preserved_pass']].astype(bool).all().all()); primary=bool(sf==1.0 and complete and structural)
    print('[6/8] Build source-diversity and seed diagnostics'); du=bank.groupby(['perception_seed','trajectory_role','source_date']).size().reset_index(name='n'); du['share']=du['n']/du.groupby('perception_seed')['n'].transform('sum'); bu=bank.groupby(['perception_seed','trajectory_role','source_date','source_block10_id']).size().reset_index(name='n'); bu['share']=bu['n']/bu.groupby('perception_seed')['n'].transform('sum')
    manifest={'stage':'P2-0C-3C','bank_version':'v1','emulator_status':'FROZEN_FROM_P2-0C-3B3','emulator_name':'BOUNDARY_PRESERVING_FINAL_CQR_TRANSPORT_PLUS_BLOCK10','paper1_asset':{'path':str(p1p),'sha256':sha256_file(p1p),'role':EXPECTED_ROLE,'rows':len(p1),'dates':p1.date.nunique()},'wapp_bridge_asset':{'path':str(wp),'sha256':sha256_file(wp),'valid_days':len(w),'L_min':float(w.L_power_proxy.min()),'L_median':float(w.L_power_proxy.median()),'L_max':float(w.L_power_proxy.max())},'frozen_protocol':{'conditioning_variables':['true_L'],'local_radius_abs_L':RADIUS,'local_min_samples':MIN_SAMPLES,'local_min_dates':MIN_DATES,'selector':'BLOCK10','block_minutes':BLOCK_MIN,'within_block_weighting':'GAUSSIAN_L_DISTANCE','kernel_bandwidth':BW,'final_output_transport':True,'preserve_source_lower_clipped_state':True,'fallback':False,'wapp_irradiance_used':False,'daily_perception_persistence_imposed':False},'perception_seeds':{'DEV':[DEV_SEED],'FORMAL_EVAL':FORMAL_SEEDS},'note':'Perception seeds are not RL training seeds; all competing policies must use identical formal perception trajectories.'}
    summary={'stage':'P2-0C-3C','trajectory_bank_generated':True,'trajectory_bank_frozen':False,'emulator_frozen':True,'support':{'wapp_queries':len(support),'supported_queries':int(support.support_ok.astype(bool).sum()),'supported_fraction':sf,'candidate_samples_distribution':qstats(support.candidate_samples),'candidate_dates_distribution':qstats(support.candidate_dates),'candidate_blocks_distribution':qstats(support.candidate_blocks)},'bank':{'requested_seeds':ALL_SEEDS,'dev_seed':DEV_SEED,'formal_eval_seeds':FORMAL_SEEDS,'rows_per_seed':EXPECTED_WAPP_VALID,'expected_total_rows':expected,'actual_total_rows':len(bank)},'primary_gates':{'full_wapp_support_100pct':bool(sf==1.0),'all_seed_rows_complete':complete,'all_structural_invariants_pass':structural,'all_primary_gates_pass':primary},'seed_metrics':sm.to_dict(orient='records'),'limitations':['Paper1-derived stochastic perception surrogate, not WAPP field-image validation.','No daily perception-regime persistence in the primary bank.','BLOCK10 is a source-resampling cluster, not a WAPP daily timescale.','No RL reward/action/transition logic is included.'],'next_step_if_pass':'Review and freeze the perception bank, then build the counterfactual WAPP cleaning environment before PPO.','next_step_if_fail':'Do not train RL; diagnose support or implementation failures without relaxing the frozen emulator.'}
    print('[7/8] Write bank, manifest, and audit summary'); bank.to_csv(out/'trajectory_bank.csv',index=False,encoding='utf-8-sig'); sm.to_csv(out/'seed_metrics.csv',index=False,encoding='utf-8-sig'); du.to_csv(out/'source_date_usage.csv',index=False,encoding='utf-8-sig'); bu.to_csv(out/'source_block_usage.csv',index=False,encoding='utf-8-sig'); support.to_csv(out/'support_audit.csv',index=False,encoding='utf-8-sig'); (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[8/8] Completed P2-0C-3C generation'); print(out/'trajectory_bank.csv'); print(out/'seed_metrics.csv'); print(out/'support_audit.csv'); print(out/'manifest.json'); print(out/'audit_summary.json'); print('IMPORTANT: review audit_summary.json before freezing this bank or starting the RL environment.')
    return 0

if __name__=='__main__': raise SystemExit(main())
