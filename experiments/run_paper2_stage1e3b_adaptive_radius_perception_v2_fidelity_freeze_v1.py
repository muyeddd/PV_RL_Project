"""P2-1E-3B-v1 two-population adaptive perception fidelity validation.
Population Amendment v1 fixed before v2 results. Inert import; explicit formal
mode only. No PPO, environment execution, protected roles or formal RL assets.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage1e3b_adaptive_radius_perception_v2_fidelity_freeze_v1.py"
MODULE = "experiments/paper2_adaptive_radius_perception_v2.py"
LEGACY_SOURCE = "experiments/run_paper2_stage0c3b3_source_weighting_audit_v1.py"
FROZEN_SOURCE = "experiments/run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1.py"
GEOMETRY_SOURCE = "experiments/run_paper2_stage1e3a_adaptive_radius_support_feasibility_audit_v1.py"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
LEGACY = BASE / "p2_0c_3b3_source_weighting_audit_v1"
PREVIOUS = BASE / "p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal"
INTERVALS = BASE / "p2_1e_2_rl_perception_support_closure_audit_v1/formal/perception_support_intervals.csv"
STAGE = BASE / "p2_1e_3b_adaptive_radius_perception_v2_fidelity_freeze_v1"
OUTPUT = STAGE / "formal"
CHECKPOINT = 'fe042d5eea9f6b823bf027994a6e8a77d07a76c1'
PINS = {'experiments/run_paper2_stage0c3b3_source_weighting_audit_v1.py': 'd951ee5b4e6c60b4996521497556619a2e83a0af9c602fef825cd3c64cf3aab0',
 'experiments/run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1.py': '379808b83ac0d47e7f8da593b8455a8de4914c7a33b3b1df78e1409be6d0c6e1',
 'experiments/run_paper2_stage1e3a_adaptive_radius_support_feasibility_audit_v1.py': '38e0e718e31cd0870f5e4e642184555ed4e06ea7df2b4ea684ca59e15f2ad2ee',
 'outputs/paper2_uncertainty_rl_v1/p2_0c_3b3_source_weighting_audit_v1/audit_summary.json': '12d60c31e0688947075f8ed857920680518b262ceecec628159ff79aac8b6e3d',
 'outputs/paper2_uncertainty_rl_v1/p2_0c_3b3_source_weighting_audit_v1/selector_date_level_metrics.csv': '1f6f2fff4967f8efc495b0a9d2340c22e0481b118164989ceb56562c1daa2102',
 'outputs/paper2_uncertainty_rl_v1/p2_0c_3b3_source_weighting_audit_v1/selector_date_macro_metrics.csv': '0348c64ba65859125cca423e10dc9eb2d0aed0651a2e8d7994709c5ac72071fb',
 'outputs/paper2_uncertainty_rl_v1/p2_0c_3b3_source_weighting_audit_v1/selector_gate_comparison.csv': '6e7a5244f28fe850b399b0313150408e7739333e1a3a74d4d85b1977da036bef',
 'outputs/paper2_uncertainty_rl_v1/p2_0c_3b3_source_weighting_audit_v1/selector_source_date_usage.csv': '0967c1453670efdd7d8d65b14cf2ebf032cf63924496eefd56f85f94c8378897',
 'outputs/paper2_uncertainty_rl_v1/p2_0c_3b3_source_weighting_audit_v1/selector_target_support.csv': 'acd31de8d98793f58c16d6119134897a8ab4ab3ef4ff68f5bdb554aa1a114fb1',
 'outputs/paper2_uncertainty_rl_v1/p2_0c_3b3_source_weighting_audit_v1/wapp_support_by_removed_date.csv': 'a2f907995f50d0268e26ea626b07c8f2d1fbc54200889fa756d93fe2bb81e610',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_2_rl_perception_support_closure_audit_v1/formal/perception_support_intervals.csv': 'cf97c31a4645dc7ce5fd0b36c49bc4968a2a7c9ee294f514aa5908bbcdfd3f6f',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/adaptive_radius_summary.json': '6081f47eb9685e9b92028a5f374f3dbe4c51207e32136af3fb42931f6c704fc0',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/audit_summary.json': '11bdd845d421dacef040ac6c394ee34fbe4decb704a98836261a275f6affee67',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/full_domain_radius_geometry.csv': '967b23462bb191b1b584a5b4fc3ef7c54abd6ca7a4535d392701601fdd42f4c7',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/output_hashes.json': 'a7875f2419fec8ab9eb22da4f81791e42c369013ea512e57bfbe5b9f969de32d',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/protocol_manifest.json': 'c5ae356678a9ee78328c414e9a6370ff2cbf90be6fd8c8727f2f22451081f320',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/reachable_radius_burden.csv': '7bd2f796bb4facd72b3bb331611de70a5ac07ce1fd6c83767b0cb4368e7a7a44',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/source_artifact_hashes.json': 'd1c2e6421ee65b1abfe6959951a7a3f7a19031c70ff87e56effb54c5a94fce75',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/unsupported_interval_radius_requirements.csv': '31bcb3eacecd0b3ea391e49ac4d480f4aa8664ee7ea7e5f649e0c22bc2c7a254',
 'outputs/paper2_uncertainty_rl_v1/p2_1e_3a_adaptive_radius_support_feasibility_audit_v1/formal/witness_radius_requirement.json': '27ba30678e8545cc6ba21f49f07381756f7f3dc535e0716a24b859b4b607da09'}
BLOBS = {'experiments/run_paper2_stage0c3b3_source_weighting_audit_v1.py': '2a094decbfbc4883d22abcb3e41e4501ab09e956',
 'experiments/run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1.py': '6f940272ce1a262f325af310c9e72ed0adc401eb',
 'experiments/run_paper2_stage1e3a_adaptive_radius_support_feasibility_audit_v1.py': 'b4aa9bc2a661cda358b04e4d4ba49ff629e336d1'}

DECLARATION = ('MINIMUM-ADAPTIVE-RADIUS PERCEPTION V2 FROZEN AFTER INDEPENDENT SOURCE-DATE FIDELITY VALIDATION; '
               'V1-SUPPORTED QUERIES ARE EXACTLY BACKWARD-COMPATIBLE; 20-SAMPLE/3-DATE QUALITY THRESHOLDS, '
               'BLOCK10 SELECTOR, BANDWIDTH AND TRANSPORT ARE UNCHANGED; NO PPO OR FORMAL RL PERFORMANCE WAS ACCESSED')
SCOPE = ('Paper1 DECISION_DEVELOPMENT internal leave-one-source-date-out reconstruction only; '
         'no external generalization, WAPP image perception, protected Paper1 role, or formal RL validation claim')


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT,
                          capture_output=True, text=True, check=True, timeout=30).stdout.strip()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def clean(value):
    # Undefined descriptive correlations are null, never silently made zero.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_json(name, value):
    data = (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    write_bytes(name, data)


def write_bytes(name, data):
    with (OUTPUT/name).open('xb') as f:
        f.write(data)
    require((OUTPUT/name).read_bytes() == data, 'Exact readback: '+name)


def write_csv(name, rows):
    require(bool(rows), 'Nonempty output '+name)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(clean(rows))
    write_bytes(name, stream.getvalue().encode())


def exact_number(a, b):
    # Historical 3B3 defines no aggregate-reconstruction epsilon: use exact
    # binary64 identity after round-trip decimal decoding, including signed zero.
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or a.hex() == b.hex()


def verify_static_contract(v2):
    old = ast.parse((ROOT/GEOMETRY_SOURCE).read_text(encoding='utf-8'))
    new = ast.parse((ROOT/MODULE).read_text(encoding='utf-8'))
    def methods(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RadiusGeometry')
        return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    a, b = methods(old), methods(new)
    require(all(a[n] == b[n] for n in ('__init__', 'counts', 'admissible', 'required')), '3A helper AST exact')
    for name in ('bits', 'number'):
        x = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
        y = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
        require(ast.dump(x) == ast.dump(y), 'Binary64 helper AST exact')
    cls = v2.AdaptiveRadiusOnlineBlock10EmulatorV2
    require(cls.sample_one is v2.frozen.OnlineBlock10Emulator.sample_one
            and 'sample_one' not in cls.__dict__, 'Inherited frozen sample_one only')
    sampler = ast.parse((ROOT/FROZEN_SOURCE).read_text(encoding='utf-8'))
    sample = next(n for n in ast.walk(sampler) if isinstance(n, ast.FunctionDef) and n.name == 'sample_one')
    require(not any(isinstance(n, ast.Name) and n.id == 'LOCAL_RADIUS' for n in ast.walk(sample)),
            'No hidden fixed radius in inherited sample_one')


def provenance(head):
    for name, expected in PINS.items():
        require(sha(ROOT/name) == expected, 'Frozen input SHA256 '+name)
    git('merge-base', '--is-ancestor', CHECKPOINT, head)
    for name, blob in BLOBS.items():
        require(git('rev-parse', f'{CHECKPOINT}:{name}') == git('rev-parse', f'{head}:{name}')
                == git('rev-parse', ':'+name) == git('hash-object', '--path='+name, name) == blob,
                'Frozen source blob '+name)
    previous = read_json(PREVIOUS/'audit_summary.json')
    require(previous['HEAD'] == CHECKPOINT and previous['stage_pass'] is True
            and previous['failed_gates'] == []
            and previous['scientific_status'] == 'ADAPTIVE_RADIUS_SUPPORT_FEASIBILITY_DIAGNOSTIC_COMPLETE'
            and previous['perception_repair_applied'] is False
            and previous['perception_samples_generated'] == previous['PPO_training_runs']
            == previous['formal_trajectory_access_count'] == previous['formal_perception_seed_access_count'] == 0,
            'Accepted 3A diagnostic only')
    for name, expected in previous['output_sha256'].items():
        require((PREVIOUS/name).resolve().is_relative_to(PREVIOUS)
                and sha(PREVIOUS/name) == expected, '3A complete output provenance')
    require(read_json(PREVIOUS/'output_hashes.json')['hashes'] ==
            {n:h for n,h in previous['output_sha256'].items() if n != 'output_hashes.json'}, '3A hash index')
    return read_json(LEGACY/'audit_summary.json')


def seed_formula(authority):
    """Extract the original expression; do not redefine seed constants."""
    tree = ast.parse((ROOT/LEGACY_SOURCE).read_text(encoding='utf-8'))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'default_rng']
    require(len(nodes) == 1, 'One frozen reconstruction RNG initialization')
    expression = nodes[0].args[0]
    require({n.id for n in ast.walk(expression) if isinstance(n, ast.Name)} ==
            {'BASE_SEED', 'selector_idx', 'fold_idx', 'rep'}, 'Original RNG namespace')
    program = compile(ast.Expression(expression), '<frozen 3B3 seed expression>', 'eval')
    index = authority.SELECTORS.index('BLOCK10')
    def value(fold, rep):
        return eval(program, {'__builtins__': {}}, {'BASE_SEED':authority.BASE_SEED,
                    'selector_idx':index, 'fold_idx':fold, 'rep':rep})
    return value, ast.unparse(expression)


def finite_sample(sample):
    require(all(math.isfinite(float(sample[k])) for k in ('q50', 'lower', 'upper', 'width'))
            and sample['lower'] <= sample['q50'] <= sample['upper']
            and sample['width'] == sample['upper']-sample['lower']
            and sample['fallback_used'] is False, 'Finite inherited transport, no fallback')


def metrics(authority, records):
    return authority.metric_dict([r['query_L'] for r in records], [r['q50'] for r in records],
                                 [r['lower'] for r in records], [r['upper'] for r in records])


def reference_metrics(authority, frame):
    return authority.metric_dict(frame['true_L'], frame['q50'], frame['lower'], frame['upper'])


def mean_finite(values):
    import numpy as np
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(values)) if values else float('nan')


def date_row(authority, actual, rep_metrics, **metadata):
    generated = {k:mean_finite([r[k] for r in rep_metrics]) for k in authority.GATES}
    return {**metadata, **{'actual_'+k:float(actual[k]) for k in authority.GATES},
            **{'generated_'+k:generated[k] for k in authority.GATES},
            **{'absdiff_'+k:abs(generated[k]-actual[k]) for k in authority.GATES}}


def macro_gates(authority, rows):
    # Same date-macro weighting and finite-replicate handling as frozen 3B3.
    import pandas as pd
    def macro(prefix):
        return {k:float(pd.to_numeric(pd.Series([r[prefix+k] for r in rows], dtype=float),
                     errors='coerce').dropna().mean()) for k in authority.GATES}
    return authority.compare_gates(macro('actual_'), macro('generated_'))


def legacy_population(authority, v2, source, legacy, seed):
    """A: historical mask/order and independent continuous RNG streams unchanged."""
    import numpy as np
    dates, cases, date_rows = sorted(source['date'].unique()), [], []
    cap = legacy['deployment_domain']['wapp_L_max']
    original_support = read_csv(LEGACY/'selector_target_support.csv')
    recovered_support = []
    for fold, heldout in enumerate(dates):
        pool = source[~source['date'].eq(heldout)].copy().reset_index(drop=True)
        target_full = source[source['date'].eq(heldout) & (source['true_L'] <= cap)].copy().reset_index(drop=True)
        infos = [authority.candidate_info(pool, float(t)) for t in target_full['true_L']]
        mask = np.asarray([i['ok'] for i in infos], dtype=bool)
        for (_, row), info in zip(target_full.iterrows(), infos):
            recovered_support.append({'heldout_date':str(heldout), 'target_sample_id':str(row['sample_id']),
                'target_true_L':float(row['true_L']), 'support_ok':info['ok'],
                'candidate_samples':len(info['indices']), 'candidate_dates':len(info['dates'])})
        if int(mask.sum()) < 3:  # EXACT original predeclared population rule, not bad-date removal.
            continue
        target = target_full.loc[mask].copy().reset_index(drop=True)
        selected_infos = [i for i, ok in zip(infos, mask) if ok]
        parent, model = v2.frozen.OnlineBlock10Emulator(pool), v2.AdaptiveRadiusOnlineBlock10EmulatorV2(pool)
        for q, info in zip(target['true_L'], selected_infos):
            require(np.array_equal(parent._candidate_indices(q), info['indices'])
                    and np.array_equal(model._candidate_indices(q), info['indices'])
                    and model.radius_requirement(q)['required_radius'] == v2.frozen.LOCAL_RADIUS,
                    'A exact historical candidate order and base radius')
        repetitions = []
        for rep in range(authority.MC_REPS):
            # A historical, v1, v2 streams each start from the SAME original seed.
            historical_rng = np.random.default_rng(seed(fold, rep))
            parent_rng = np.random.default_rng(seed(fold, rep))
            new_rng = np.random.default_rng(seed(fold, rep))
            generated = []
            for (_, row), info in zip(target.iterrows(), selected_infos):
                q = float(row['true_L'])
                index = authority.choose_source_index(historical_rng, pool, info, 'BLOCK10')
                historical = authority.transport_final_output(pool.iloc[index], q)
                old, new = parent.sample_one(q, parent_rng), model.sample_one(q, new_rng)
                finite_sample(new)
                require(old == new and new['source_index'] == index
                        and all(exact_number(new[k], x) for k,x in zip(('q50','lower','upper'), historical))
                        and historical_rng.bit_generator.state == parent_rng.bit_generator.state == new_rng.bit_generator.state,
                        'A exact output/source/RNG identity')
                generated.append(new)
                cases.append({'population':'A', 'heldout_date':str(heldout), 'rep':rep,
                    'target_sample_id':str(row['sample_id']), 'query_L':q, 'source_sample_id':str(new['source_sample_id']),
                    'source_date':new['source_date'], 'source_block':new['source_block_id'],
                    'candidate_exact':True, 'outputs_exact':True, 'rng_state_exact':True})
            repetitions.append(metrics(authority, generated))
        date_rows.append(date_row(authority, reference_metrics(authority,target), repetitions,
                         heldout_date=str(heldout), N_deployment_domain=len(target_full), N_supported=int(mask.sum()),
                         support_fraction=float(mask.mean())))
    require(len(recovered_support) == len(original_support), 'A target inventory exact')
    for current, frozen in zip(recovered_support, original_support):
        require(all(str(current[k]) == frozen[k] if k != 'target_true_L'
                    else exact_number(current[k], frozen[k]) for k in current), 'A historical targets/order/support mask exact')
    expected_dates = [r for r in read_csv(LEGACY/'selector_date_level_metrics.csv') if r['selector'] == 'BLOCK10']
    require(len(expected_dates) == len(date_rows), 'A historical fold count exact')
    for row, expected in zip(date_rows, expected_dates):
        require(row['heldout_date'] == expected['heldout_date'] and
                all(exact_number(v, expected[k]) for k,v in row.items() if k != 'heldout_date'), 'A date metrics exact')
    macro = macro_gates(authority, date_rows)
    historical_macro = next(r for r in legacy['selector_macro_results'] if r['selector'] == 'BLOCK10')
    for k in authority.GATES:
        require(exact_number(macro[k]['actual_macro'], historical_macro['actual_'+k])
                and exact_number(macro[k]['generated_macro'], historical_macro['generated_'+k])
                and exact_number(macro[k]['absolute_difference'], historical_macro['absdiff_'+k]), 'A aggregate exact')
    require(macro['all_pass'] is True, 'Original A distribution gates')
    return cases, {'pass':True, 'case_count':len(cases), 'date_count':len(date_rows),
        'exact_identity_fraction':1.0, 'date_metrics':date_rows, 'historical_macro':macro,
        'role':'Historical regression only; NOT adaptive fidelity evidence'}


def band(radius, base):
    if radius == base:
        return 'r == base'
    for endpoint in (.015, .020, .030, .040):
        if radius <= endpoint:
            return f'r <= {endpoint:.3f}'
    return 'r > 0.040'


def adaptive_population(authority, v2, source, seed):
    """B: all rows, every date; an independent pass with original seed formula.

    FULL and ADAPTIVE_ONLY use original replicate-mean then date-macro metrics.
    A date with no adaptive targets is absent only from that stratum, explicitly
    reported as empty. Undefined correlation follows original finite handling.
    No target is excluded from FULL, no cap or v1-support filter is applied.
    """
    import numpy as np
    dates, targets, rows, generated_all = sorted(source['date'].unique()), [], [], []
    stratum_names = ('FULL', 'LEGACY_SUPPORTED', 'ADAPTIVE_ONLY')
    for fold, heldout in enumerate(dates):
        pool = source[~source['date'].eq(heldout)].copy().reset_index(drop=True)
        target = source[source['date'].eq(heldout)].copy().reset_index(drop=True)
        require(len(target) > 0 and not pool['date'].eq(heldout).any(), 'LODO disjoint source and target')
        model = v2.AdaptiveRadiusOnlineBlock10EmulatorV2(pool)
        metadata = []
        for _, row in target.iterrows():
            radius = model.radius_requirement(float(row['true_L']))
            legacy = radius['required_radius'] == v2.frozen.LOCAL_RADIUS
            item = {'heldout_date':str(heldout), 'target_sample_id':str(row['sample_id']),
                    'stratum':'LEGACY_SUPPORTED' if legacy else 'ADAPTIVE_ONLY',
                    'radius_band':band(radius['required_radius'],v2.frozen.LOCAL_RADIUS), **radius}
            targets.append(item)
            metadata.append(item)
        masks = {'FULL':np.ones(len(target),dtype=bool),
                 'LEGACY_SUPPORTED':np.asarray([r['stratum']=='LEGACY_SUPPORTED' for r in metadata]),
                 'ADAPTIVE_ONLY':np.asarray([r['stratum']=='ADAPTIVE_ONLY' for r in metadata])}
        rep_metrics = {s:[] for s in stratum_names}
        for rep in range(authority.MC_REPS):
            rng = np.random.default_rng(seed(fold, rep))  # independent of A, never reused state
            samples = []
            for (_, row), info in zip(target.iterrows(), metadata):
                result = model.sample_one(float(row['true_L']), rng)
                finite_sample(result)
                record = {**result, 'heldout_date':str(heldout), 'rep':rep,
                          'target_sample_id':info['target_sample_id'], 'stratum':info['stratum'],
                          'required_radius':info['required_radius'], 'radius_band':info['radius_band']}
                samples.append(record)
                generated_all.append(record)
            for name, mask in masks.items():
                if mask.any():
                    rep_metrics[name].append(metrics(authority,[r for r,ok in zip(samples,mask) if ok]))
        for name, mask in masks.items():
            subset = target.loc[mask]
            if len(subset):
                rows.append(date_row(authority, reference_metrics(authority,subset), rep_metrics[name],
                            population='B', stratum=name, heldout_date=str(heldout), N_targets=len(subset),
                            empty_stratum=False))
            else:
                rows.append({'population':'B','stratum':name,'heldout_date':str(heldout),
                             'N_targets':0,'empty_stratum':True})
    require(len(targets) == authority.EXPECTED_N == 1844 and len(dates) == authority.EXPECTED_DATES == 12
            and len({t['target_sample_id'] for t in targets}) == len(targets)
            and len(generated_all) == len(targets)*authority.MC_REPS, 'Complete B targets and repeats, no deletion')
    gates = {s:macro_gates(authority,[r for r in rows if r['stratum']==s and not r['empty_stratum']])
             for s in ('FULL','ADAPTIVE_ONLY')}
    counts = {s:sum(t['stratum']==s for t in targets) for s in ('LEGACY_SUPPORTED','ADAPTIVE_ONLY')}
    require(counts['ADAPTIVE_ONLY'] > 0, 'Adaptive stratum must be tested')
    usage = {}
    for s in ('LEGACY_SUPPORTED','ADAPTIVE_ONLY'):
        group = [t for t in targets if t['stratum']==s]
        usage[s] = {'N_targets':len(group), 'heldout_dates':sorted({r['heldout_date'] for r in group}),
                    'radius_distribution':authority.qstats([r['required_radius'] for r in group])}
    by_band = []
    for label in ('r == base','r <= 0.015','r <= 0.020','r <= 0.030','r <= 0.040','r > 0.040'):
        ts = [r for r in targets if r['radius_band']==label]
        samples = [r for r in generated_all if r['radius_band']==label]
        base_row = {'radius_band':label, 'N_queries':len(ts), 'N_draws':len(samples),
                    'target_dates_represented':len({r['heldout_date'] for r in ts}),
                    'source_dates_represented':len({r['source_date'] for r in samples}),
                    'descriptive_only':True}
        if samples:
            radii = [r['required_radius'] for r in ts]
            base_row.update(mean_required_radius=float(np.mean(radii)), median_required_radius=float(np.median(radii)))
            base_row.update(metrics(authority,samples))
        by_band.append(base_row)
    return rows, by_band, targets, {'gates':gates, 'full_and_adaptive_only_pass':all(g['all_pass'] for g in gates.values()),
        'legacy_supported_n':counts['LEGACY_SUPPORTED'], 'adaptive_only_n':counts['ADAPTIVE_ONLY'],
        'N_targets':len(targets), 'N_source_dates':len(dates), 'per_date_hard_gates':False,
        'hard_gate_weighting':'Original 3B3: mean finite replicate metrics within date, then date macro; not pooled episodes',
        'stratum_radius_usage':usage, 'scope':SCOPE}


def supported_domain_regression(v2, source):
    """Structural all-query identity plus frozen support-map boundary regressions."""
    import numpy as np
    parent, model = v2.frozen.OnlineBlock10Emulator(source), v2.AdaptiveRadiusOnlineBlock10EmulatorV2(source)
    cases = []
    for interval in read_csv(INTERVALS):
        if interval['supported'] != 'True':
            continue
        left, right = float(interval['interval_left']), float(interval['interval_right'])
        points = {left if interval['left_closed']=='True' else math.nextafter(left,math.inf),
                  float(interval['representative_L']),
                  right if interval['right_closed']=='True' else math.nextafter(right,-math.inf)}
        for q in sorted(points):
            require(np.array_equal(parent._candidate_indices(q), model._candidate_indices(q))
                    and model.radius_requirement(q)['required_radius'] == v2.frozen.LOCAL_RADIUS, 'Supported-domain candidate identity')
            a = np.random.default_rng(v2.frozen.DEV_PERCEPTION_SEED)
            b = np.random.default_rng(v2.frozen.DEV_PERCEPTION_SEED)
            old, new = parent.sample_one(q,a), model.sample_one(q,b)
            require(old == new and a.bit_generator.state == b.bit_generator.state, 'Supported-domain output/RNG identity')
            cases.append({'population':'SUPPORTED_DOMAIN_BOUNDARY', 'query_L':q,
                          'candidate_exact':True, 'outputs_exact':True, 'rng_state_exact':True})
    require(bool(cases), 'Frozen supported-domain regression population')
    return cases


def witness_regression(v2, source):
    import numpy as np
    old = read_json(PREVIOUS/'witness_radius_requirement.json')
    model = v2.AdaptiveRadiusOnlineBlock10EmulatorV2(source)
    result = model.radius_requirement(old['representative_L'])
    require(result == {k:old[k] for k in result}, 'Witness radius exact vs accepted 3A')
    require(result['base_local_rows'] == 19 and result['base_local_dates'] == 6
            and result['required_radius'] == 0.010029416603574828
            and result['rows_at_required_radius'] == 20 and result['dates_at_required_radius'] == 6
            and result['previous_binary64_radius_fails'] is True, 'Witness expected regression')
    sample = model.sample_one(old['representative_L'], np.random.default_rng(v2.frozen.DEV_PERCEPTION_SEED))
    finite_sample(sample)
    # NumPy source IDs are normalized for JSON only, not scientific comparison.
    sample['source_sample_id'] = str(sample['source_sample_id'])
    return {'pass':True, 'radius':result, 'sample':sample, 'diagnostic_seed':v2.frozen.DEV_PERCEPTION_SEED,
            'role':'Integration proof only; no performance claim'}


def final_integrity(head, sources, blobs):
    require(git('rev-parse','HEAD') == head and git('status','--porcelain','--untracked-files=all') == '', 'HEAD/tree unchanged')
    require(all(sha(ROOT/n) == h for n,h in sources.items()), 'Inputs and sources unchanged')
    require(all(git('rev-parse', f'{head}:{n}') == git('rev-parse', ':'+n)
                == git('hash-object','--path='+n,n) == b for n,b in blobs.items()), 'Published source unchanged')


def publish_audit(passed, head, failed_gates, expected=None):
    existing = {p.name for p in OUTPUT.iterdir()}
    require('audit_summary.json' not in existing and 'output_hashes.json' not in existing, 'No repeat publication')
    if expected is not None:
        require(existing == expected, 'Complete output inventory')
    hashes = {name:sha(OUTPUT/name) for name in sorted(existing)}
    write_json('output_hashes.json', {'hashes':hashes, 'exclusions':{
        'output_hashes.json':'Pinned by final audit', 'audit_summary.json':'Last publication; no self-hash'}})
    require(all(sha(OUTPUT/n) == h for n,h in hashes.items()), 'Output hashes readback exact')
    hashes['output_hashes.json'] = sha(OUTPUT/'output_hashes.json')
    write_json('audit_summary.json', {'stage':'P2-1E-3B-v1', 'stage_pass':passed, 'HEAD':head,
        'scientific_status':'ADAPTIVE_RADIUS_PERCEPTION_V2_FIDELITY_PASS_FROZEN' if passed else
                            'ADAPTIVE_RADIUS_PERCEPTION_V2_FIDELITY_FAIL',
        'declaration':DECLARATION if passed else 'POINT/UA PPO REMAINS BLOCKED; NO RETRY OR PARAMETER CHANGE',
        'failed_gates':failed_gates, 'scope':SCOPE, 'PPO_training_runs':0, 'formal_RL_access_count':0,
        'protected_Paper1_roles_access_count':0, 'formal_perception_seed_access_count':0,
        'output_sha256':hashes, 'audit_summary_published_last':True})


def run_formal():
    require(not STAGE.exists(), 'Immutable stage: no overwrite, resume or retry')
    require(git('status','--porcelain','--untracked-files=all') == '', 'Both new sources must be committed unchanged')
    head = git('rev-parse','HEAD')
    legacy = provenance(head)
    sources, blobs = dict(PINS), dict(BLOBS)
    for name in (SELF,MODULE):
        sources[name] = sha(ROOT/name)
        blobs[name] = git('rev-parse', f'{head}:{name}')
    final_integrity(head,sources,blobs)
    import paper2_adaptive_radius_perception_v2 as v2
    import run_paper2_stage0c3b3_source_weighting_audit_v1 as authority
    import run_paper2_stage1e3a_adaptive_radius_support_feasibility_audit_v1 as geometry_reference
    verify_static_contract(v2)
    f = v2.frozen
    require(f.LOCAL_RADIUS == authority.LOCAL_RADIUS == legacy['fixed_transport']['local_radius_abs_L']
            and f.MIN_LOCAL_SAMPLES == authority.LOCAL_MIN_SAMPLES == 20
            and f.MIN_LOCAL_DATES == authority.LOCAL_MIN_DATES == 3
            and f.KERNEL_BANDWIDTH == authority.KERNEL_BANDWIDTH == legacy['fixed_transport']['kernel_bandwidth']
            and f.BLOCK_MINUTES == authority.BLOCK_MINUTES == 10, 'Frozen constants exact')
    gates = [r for r in read_csv(LEGACY/'selector_gate_comparison.csv') if r['selector']=='BLOCK10']
    require({r['metric']:float(r['tolerance']) for r in gates} == authority.GATES
            and all(r['pass']=='True' for r in gates) and 'BLOCK10' in legacy['passing_selectors'], 'Accepted tolerance authority')
    provenance_3a = read_json(PREVIOUS/'source_artifact_hashes.json')
    paths = [Path(n) for n,h in provenance_3a['sha256'].items() if h == f.EXPECTED_HASHES['paper1_cqr']]
    require(len(paths)==1, 'Unique pinned Paper1 DECISION_DEVELOPMENT source')
    path = paths[0] if paths[0].is_absolute() else ROOT/paths[0]
    require(sha(path)==f.EXPECTED_HASHES['paper1_cqr'], 'Paper1 exact hash before reading')
    source = f.load_paper1_cqr(path)
    require(len(source)==authority.EXPECTED_N==1844 and source['date'].nunique()==authority.EXPECTED_DATES==12
            and set(source['role'])=={authority.EXPECTED_ROLE}=={f.EXPECTED_ROLE}, 'Source role/N/dates exact')
    source['block10_id'] = source['source_block_id']  # original 3B3 column alias only
    sources[str(path)] = f.EXPECTED_HASHES['paper1_cqr']
    seed, seed_expression = seed_formula(authority)
    # Date/rep namespace stays the exact original BLOCK10 formula in both passes.
    require(all(seed(i,j) not in f.FORMAL_PERCEPTION_SEEDS for i in range(authority.EXPECTED_DATES)
                for j in range(authority.MC_REPS)), 'No formal perception seed initialization')
    STAGE.mkdir(parents=True,exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    try:
        write_json('v2_source_provenance.json', {'HEAD':head,'predecessor_HEAD':CHECKPOINT,'sha256':sources,'git_blobs':blobs})
        write_json('v2_protocol_manifest.json', {
            'stage':'P2-1E-3B-v1','amendment':'Population Amendment v1; fixed before v2 fidelity results',
            'population_A':'Exact historical WAPP deployment-domain population, v1 support mask, original row order/RNG; regression only',
            'population_A_cap':legacy['deployment_domain']['wapp_L_max'],
            'population_B':'All 1844 DECISION_DEVELOPMENT rows / 12 dates LODO; no deployment cap or v1-supported filtering',
            'population_B_strata':['LEGACY_SUPPORTED','ADAPTIVE_ONLY'], 'separate_validation_passes':True,
            'hard_gate_populations':['B FULL','B ADAPTIVE_ONLY'], 'per_date_hard_gates':False,
            'tolerance_authority':LEGACY_SOURCE, 'tolerances':authority.GATES,
            'MC_REPS':authority.MC_REPS,'BASE_SEED':authority.BASE_SEED,'seed_expression':seed_expression,
            'BLOCK10_selector_index':authority.SELECTORS.index('BLOCK10'),'rng':'np.random.default_rng; original fold/rep initialization',
            'target_order':'Original Paper1 CSV row order within each sorted heldout-date fold',
            'aggregation':'Original metric_dict; finite replicate means then equally weighted finite date means',
            'radius_bands':['r == base','base < r <= 0.015','0.015 < r <= 0.020','0.020 < r <= 0.030','0.030 < r <= 0.040','r > 0.040'],
            'radius_bands_descriptive_only':True,'support_constants':{'base':f.LOCAL_RADIUS,'min_rows':f.MIN_LOCAL_SAMPLES,
                'min_dates':f.MIN_LOCAL_DATES,'bandwidth':f.KERNEL_BANDWIDTH,'block_minutes':f.BLOCK_MINUTES},
            'selector':'BLOCK10','transport':'BOUNDARY-PRESERVING FINAL-CQR TRANSPORT','fallback':'NONE',
            'sample_one_inherited':True,'historical_conclusions_unchanged':True,'scope':SCOPE,
            'environment_integration':'Deferred; no old environment or runner modified'} )
        numerical_cases = read_csv(PREVIOUS/'unsupported_interval_radius_requirements.csv')
        a = v2.RadiusGeometry(source.true_L.to_numpy(),source.date.to_numpy(),f.LOCAL_RADIUS,f.MIN_LOCAL_SAMPLES,f.MIN_LOCAL_DATES)
        b = geometry_reference.RadiusGeometry(source.true_L.to_numpy(),source.date.to_numpy(),f.LOCAL_RADIUS,f.MIN_LOCAL_SAMPLES,f.MIN_LOCAL_DATES)
        for row in numerical_cases:
            q = float(row['representative_L'])
            require(a.required(q)==b.required(q) and exact_number(a.required(q)['required_radius'],row['required_radius']), '3A numerical radius regression')
        identity_cases, identity = legacy_population(authority,v2,source,legacy,seed)
        identity_cases += supported_domain_regression(v2,source)
        identity['universal_identity_basis'] = 'Exact parent candidates on all base-supported queries + inherited sample_one + no RNG in candidate hook'
        identity['supported_domain_boundary_cases'] = len(identity_cases)-identity['case_count']
        write_json('v1_backward_identity_summary.json',identity)
        write_csv('v1_backward_identity_cases.csv',identity_cases)
        write_json('witness_v2_regression.json',witness_regression(v2,source))
        by_date, by_band, targets, result = adaptive_population(authority,v2,source,seed)
        write_json('adaptive_v2_fidelity_metrics.json',result)
        write_csv('adaptive_v2_fidelity_by_date.csv',by_date)
        write_csv('adaptive_v2_fidelity_by_radius_band.csv',by_band)
        write_csv('adaptive_v2_target_inventory.csv',targets)
        write_json('radius_usage_summary.json',{'strata':result['stratum_radius_usage'],
                   'legacy_supported_n':result['legacy_supported_n'],'adaptive_only_n':result['adaptive_only_n'],
                   'radius_bands_descriptive_only':True})
        final_integrity(head,sources,blobs)
        expected = {'v2_protocol_manifest.json','v2_source_provenance.json','v1_backward_identity_summary.json',
                    'v1_backward_identity_cases.csv','adaptive_v2_fidelity_metrics.json','adaptive_v2_fidelity_by_date.csv',
                    'adaptive_v2_fidelity_by_radius_band.csv','adaptive_v2_target_inventory.csv',
                    'witness_v2_regression.json','radius_usage_summary.json'}
        failed = [f'B {s} {k}' for s,g in result['gates'].items() for k in authority.GATES if not g[k]['pass']]
        publish_audit(not failed,head,failed,expected)
    except Exception as error:
        # Preserve every completed diagnostic; never tune, retry, replace, or overwrite.
        if not (OUTPUT/'audit_summary.json').exists() and not (OUTPUT/'output_hashes.json').exists():
            write_json('execution_failure.json',{'error_type':type(error).__name__,'message':str(error),
                       'action':'STOP; POINT/UA PPO REMAINS BLOCKED; NO RETRY OR PARAMETER CHANGE'})
            publish_audit(False,head,[str(error)])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('formal',))
    parser.parse_args()
    run_formal()


if __name__ == '__main__':
    main()
