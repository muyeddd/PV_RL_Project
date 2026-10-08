"""P2-2F-0A: integrate accepted DEVELOPMENT opportunity and PPO evidence.

Read-only inputs: 4AR/4B fixed-continuation contrasts and 5B/6A/2E-1R
policy episodes. No model/env runtime. Selected pre-first-force anchors are
NOT representative of all free-choice states. Never claim optimal Q or
infer population-level opportunity prevalence from 63/163.
Import inert; only --mode formal writes NEW immutable output.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/paper2_uncertainty_rl_v1'
SELF = 'experiments/run_paper2_stage2f0a_existing_opportunity_evidence_audit_v1.py'
D4B = BASE / 'p2_2d_4b_safe_action_distinction_decomposition_audit_v1/formal'
P5B = BASE / 'p2_2d_5b_masked_point_ppo_development_v1/formal'
U6A = BASE / 'p2_2d_6a_masked_ua_ppo_development_v1/formal'
H1R = BASE / 'p2_2e_1r_history_point_ppo_development_recovery_v1/formal'
NEW = BASE / 'p2_2f_0a_existing_opportunity_evidence_audit_v1'
OUT = NEW / 'formal'
SEEDS = (510001, 510002)
ATOL = 1e-12


def require(ok, msg):
    if not bool(ok): raise RuntimeError(msg)


def git(*args):
    return subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],
                          cwd=ROOT,check=True,capture_output=True,text=True,timeout=30).stdout.strip()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()


def load_json(path): return json.loads(Path(path).read_bytes())


def load_csv(path):
    with Path(path).open('r',encoding='utf-8',newline='') as f: return list(csv.DictReader(f))


def encoded(x): return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def csv_encoded(rows):
    require(bool(rows),'CSV nonempty')
    fields=list(rows[0]); require(all(set(r)==set(fields) for r in rows),'CSV schema')
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
    w.writeheader(); w.writerows(rows); return f.getvalue().encode()


def write(path,content):
    path=Path(path)
    require(path.parent.resolve()==OUT.resolve(),'Output parent restricted')
    with path.open('xb') as f:f.write(content)
    require(path.read_bytes()==content,'Write readback')


def audit_authority(path,stage,required_csv):
    a=load_json(path/'audit_summary.json')
    require(a['stage']==stage and a['stage_pass'] is True and not a['failed_gates']
            and all(a['gates'].values()),stage+' PASS')
    for key in ('formal_RL_access_count','RANDOM_TEST_access_count','SEALED_DATES_access_count'):
        require(a[key]==0,stage+' protected access '+key)
    require(required_csv in a['output_sha256'] and sha(path/required_csv)==a['output_sha256'][required_csv],stage+' CSV hash')
    return sha(path/'audit_summary.json')


def episodes(path):
    rows=load_csv(path); require(len(rows)==1200,'Frozen 1200 episodes')
    index={}
    for r in rows:
        key=(int(r['parent_rl_seed']),r['year'],int(r['trajectory_id']))
        require(key not in index,'Unique episode key')
        index[key]=(float(r['J_total']),int(r['N_voluntary_clean']),int(r['N_shield_interventions']))
    require(len(index)==1200 and set(index)=={
        (seed,year,tid) for seed in SEEDS for year in ('YEAR1','YEAR2') for tid in range(300)
    },'Exact two-seed DEVELOPMENT key set')
    return index


def static_contract():
    tree=ast.parse((ROOT/SELF).read_text(encoding='utf-8'))
    attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    require(not ({'step','reset','learn','predict','save','load'} & attrs),'No PPO/env runtime')
    return {'no_training':True,'no_environment_runtime':True,'existing_4AR_4B_only':True,
            'three_frozen_PPO_episode_tables':True,'fixed_continuation_not_optimal_Q':True,
            'no_protected_roles':True,'performance_not_PASS_gate':True}


def run_formal():
    require(not NEW.exists(),'Exclusive stage output; no overwrite/resume')
    require(git('status','--porcelain','--untracked-files=all')=='','Committed clean worktree')
    head=git('rev-parse','HEAD')
    require(git('rev-parse',f'{head}:{SELF}')==git('hash-object',f'--path={SELF}',SELF)
            ==git('rev-parse',f':{SELF}'),'Exact committed script')
    stat=static_contract()
    import run_paper2_stage2d4b_safe_action_distinction_decomposition_audit_v1 as prior
    require(all(prior.static_contract().values()),'Prior 4B static contract')
    old_auth=prior.verify_predecessor(head)  # SHA pins ALL frozen 4AR outputs
    anchors=prior.audit_rows()
    require(len(anchors)==163,'Exactly 163 frozen anchors')
    prior4bsha=audit_authority(D4B,'P2-2D-4B-v1','distinction_row_audit.csv')
    pointsha=audit_authority(P5B,'P2-2D-5B-v1','masked_point_development_episode_metrics.csv')
    uasha=audit_authority(U6A,'P2-2D-6A-v1','masked_ua_development_episode_metrics.csv')
    hpsha=audit_authority(H1R,'P2-2E-1R-v1','history_point_development_episode_metrics.csv')
    point=episodes(P5B/'masked_point_development_episode_metrics.csv')
    ua=episodes(U6A/'masked_ua_development_episode_metrics.csv')
    hp=episodes(H1R/'history_point_development_episode_metrics.csv')
    pairs=[]
    for a in anchors:
        year,tid=a['year'],int(a['trajectory_id'])
        for seed in SEEDS:
            key=(seed,year,tid)
            p,u,h=point[key],ua[key],hp[key]
            pairs.append({
                'seed':seed,'year':year,'trajectory_id':tid,
                'anchor_label':a['anchor_label'],'anchor_day':a['anchor_day'],
                'fixed_continuation_CLEANminusWAIT':a['remaining_horizon_gap_CLEANminusWAIT'],
                'fixed_continuation_classification':a['continuation_classification'],
                'Static_Point_J':p[0],'Static_UA_J':u[0],'History_Point_J':h[0],
                'Static_Point_voluntary_CLEAN':p[1],
                'Static_UA_voluntary_CLEAN':u[1],
                'History_Point_voluntary_CLEAN':h[1],
                'Static_Point_shield_interventions':p[2],
                'Static_UA_shield_interventions':u[2],
                'History_Point_shield_interventions':h[2],
            })
    require(len(pairs)==326,'Paired two-seed anchor overlay')
    benefit=[a for a in anchors if a['continuation_classification']=='CLEAN_BENEFICIAL']
    episode_keys={(a['year'],a['trajectory_id']) for a in anchors}
    positive_episode_keys={(a['year'],a['trajectory_id']) for a in benefit}
    by_label={}
    for label in ('MID_PRE_FORCE','PRE_FORCE'):
        sub=[r for r in anchors if r['anchor_label']==label]
        by_label[label]={
            'n_selected_anchors':len(sub),
            'CLEAN_beneficial':sum(r['continuation_classification']=='CLEAN_BENEFICIAL' for r in sub),
            'WAIT_beneficial':sum(r['continuation_classification']=='WAIT_BENEFICIAL' for r in sub),
            'tie':sum(r['continuation_classification']=='COST_TIE_WITHIN_ATOL' for r in sub),
        }
    diag={
        'frozen_selected_anchors':len(anchors),
        'anchor_eligible_distinct_episodes':len(episode_keys),
        'episodes_without_selected_pre_first_force_anchor':600-len(episode_keys),
        'CLEAN_beneficial_selected_anchors':len(benefit),
        'CLEAN_beneficial_distinct_episodes':len(positive_episode_keys),
        'WAIT_beneficial_selected_anchors':sum(r['continuation_classification']=='WAIT_BENEFICIAL' for r in anchors),
        'tie_selected_anchors':sum(r['continuation_classification']=='COST_TIE_WITHIN_ATOL' for r in anchors),
        'selected_anchor_CLEAN_beneficial_fraction_not_population_rate':len(benefit)/len(anchors),
        'by_anchor':by_label,
        'policies_all_voluntary_CLEAN_zero_in_1200_development_episodes':{
            name:all(v[1]==0 for v in dataset.values())
            for name,dataset in [('Static_Point',point),('Static_UA',ua),('History_Point',hp)]},
        'policy_episode_costs_tie_to_Static_Point_for_all_1200':{
            'Static_UA':all(abs(point[k][0]-ua[k][0])<=ATOL for k in point),
            'History_Point':all(abs(point[k][0]-hp[k][0])<=ATOL for k in point)},
        'aligned_beneficial_anchors_all_three_policies_have_zero_voluntary_CLEAN':all(
            p['Static_Point_voluntary_CLEAN']==p['Static_UA_voluntary_CLEAN']==p['History_Point_voluntary_CLEAN']==0
            for p in pairs if p['fixed_continuation_classification']=='CLEAN_BENEFICIAL'),
        'performance_diagnostics_not_pass_gates':True,
    }
    interpretation={
        'accepted_claim':'Selected DEVELOPMENT free-choice sites contain CLEAN-beneficial choices under the same frozen WAIT-proposal-plus-shield continuation.',
        'not_population_prevalence':'163 anchors are conditional on first forced CLEAN; multiple anchors may come from one episode. 63/163 is NOT population opportunity rate.',
        'no_optimal_Q_claim':True,
        'paired_seeds_do_not_double_counterfactual_sample_size':True,
        'cannot_separate_information_and_PPO_from_observed_WAIT_only':True,
        'follow_up':'Design fresh representative post-force and pre-force free-choice sample, with original reward component decomposition and identical exogenous trajectories. Only after that test observation identifiability. No PPO tuning or protected roles.'
    }
    provenance={'HEAD':head,'frozen_4AR_authority':old_auth,'audit_sha256':{
        '4B':prior4bsha,'5B':pointsha,'6A':uasha,'2E1R':hpsha},
        'new_source_sha256':sha(ROOT/SELF),'protected_roles_used':[],
        'environment_steps':0,'PPO_training_runs':0}
    NEW.mkdir(parents=True,exist_ok=False); OUT.mkdir(exist_ok=False)
    try:
        content={'diagnostic_summary.json':encoded(diag),
                 'interpretation_limits.json':encoded(interpretation),
                 'source_artifact_provenance.json':encoded(provenance),
                 'aligned_anchor_policy_comparison.csv':csv_encoded(pairs)}
        for name,payload in content.items(): write(OUT/name,payload)
        hashes={name:hashlib.sha256(payload).hexdigest() for name,payload in content.items()}
        extra=encoded({'hashes':hashes,'audit_summary':'published last'})
        write(OUT/'output_hashes.json',extra)
        require(all(sha(OUT/name)==digest for name,digest in hashes.items()),'Output hashes match')
        require(git('rev-parse','HEAD')==head and git('status','--porcelain','--untracked-files=all')=='','Source worktree unchanged')
        prior.verify_predecessor(head)
        for directory,stage,filename in [
            (D4B,'P2-2D-4B-v1','distinction_row_audit.csv'),
            (P5B,'P2-2D-5B-v1','masked_point_development_episode_metrics.csv'),
            (U6A,'P2-2D-6A-v1','masked_ua_development_episode_metrics.csv'),
            (H1R,'P2-2E-1R-v1','history_point_development_episode_metrics.csv')]:
            audit_authority(directory,stage,filename)
        gates={'frozen_4AR_all_outputs_verified':True,'4B_5B_6A_2E1R_hash_pinned':True,
               'selected_anchors_join_to_three_policies':True,
               'all_1200_development_episodes_per_policy_present':True,
               'no_model_or_env_runtime':True,'no_protected_role_access':True,
               'scientific_results_not_PASS_gates':True,'output_hashes_verified':True}
        require(all(gates.values()) and all(stat.values()),'Structural pass')
        audit={'stage':'P2-2F-0A-v1','mode':'formal','stage_pass':True,
               'scientific_status':'EXISTING_OPPORTUNITY_EVIDENCE_INTEGRATION_COMPLETE',
               'HEAD':head,'gates':gates,'failed_gates':[],
               'PPO_training_runs':0,'PPO_model_loads':0,'environment_steps':0,
               'formal_RL_access_count':0,'RANDOM_TEST_access_count':0,'SEALED_DATES_access_count':0,
               'performance_diagnostics_not_pass_gates':diag,
               'output_sha256':{**hashes,'output_hashes.json':hashlib.sha256(extra).hexdigest()},
               'audit_summary_published_last':True}
        write(OUT/'audit_summary.json',encoded(audit))
        return audit
    except Exception as exc:
        if not (OUT/'execution_failure.json').exists():
            write(OUT/'execution_failure.json',encoded({
                'stage':'P2-2F-0A-v1','stage_pass':False,'HEAD':head,
                'error_type':type(exc).__name__,'message':str(exc),
                'action':'STOP; PRESERVE EVIDENCE; NO RETRY/RESEED/TUNING'}))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('formal',))
    p.parse_args()
    print(json.dumps(run_formal(),indent=2,allow_nan=False))

if __name__=='__main__': main()
