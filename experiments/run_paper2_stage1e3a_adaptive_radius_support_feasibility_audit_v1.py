"""P2-1E-3A-v1 Minimal Adaptive-Radius Perception Support Feasibility Audit.

Inert import; explicit --mode formal is a geometry diagnostic only. No sampler,
environment, PPO, prediction, repair or parameter mutation is executed here.
SUPPORT-DEFINITION FEASIBILITY is not PERCEPTION FIDELITY / VALIDITY.
"""
from __future__ import annotations

import argparse
import ast
import bisect
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import io
from itertools import zip_longest
import json
import math
from pathlib import Path
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage1e3a_adaptive_radius_support_feasibility_audit_v1.py"
PREDECESSOR = "experiments/run_paper2_stage1e2_rl_perception_support_closure_audit_v1.py"
PREDECESSOR_BLOB = "8838f844e69d7420cca8257d4feaf5ac69547d59"
CHECKPOINT = "8c730ab65847401d99ed7fc94f1a70a4a59b6841"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
FROZEN = BASE / "p2_1e_2_rl_perception_support_closure_audit_v1/formal"
STAGE_DIRECTORY = BASE / "p2_1e_3a_adaptive_radius_support_feasibility_audit_v1"
OUTPUT = STAGE_DIRECTORY / "formal"
PINS = {
    "audit_summary.json": "f0dfdeec474c43491de97880a2265502f3628b62256e31b65a3053ac37025749",
    "support_closure_summary.json": "9fca7595c858b307ae09f82def1ae28c275ae6aa6f60617158335480b0003ab7",
    "perception_support_intervals.csv": "cf97c31a4645dc7ce5fd0b36c49bc4968a2a7c9ee294f514aa5908bbcdfd3f6f",
    "reachable_support_overlap.csv": "6c968bf46c7f7dfca251a2ebe6748f1a825e83ca734fb89b2e1b5b4e19410689",
    "failure_witness_reproduction.json": "694872056e1f08cb7a040299741483092744811a07d0c971c7b4b434eb27108d",
    "reachable_state_envelope.csv": "4799f4c55a4b32339fdf5ac2987476cfe6f8b2f984bee434a6cc75f70d62a71a",
    "protocol_manifest.json": "7a931c5be207e0692c89b1daef4dc4a3100f91730bfc49dc0b94315d5d5087de",
    "source_artifact_hashes.json": "1be8ba2a8742ab5917c518adba19eb24bc0f3da2ea2508bc80a1598127822f90",
    "output_hashes.json": "bb2c422c110e7851e55206716ffaf7d1f9a206074dbfd45322af83e05462c6e7",
}
DECLARATION = ("SUPPORT-DEFINITION FEASIBILITY ONLY; NOT PERCEPTION FIDELITY / VALIDITY; "
               "NO PERCEPTION REPAIR APPLIED; POINT/UA PPO REMAINS BLOCKED PENDING INDEPENDENT VALIDATION")


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
                          check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def csv_bytes(rows):
    require(rows and all(set(r) == set(rows[0]) for r in rows), "Consistent CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def number(value):
    return struct.unpack(">d", struct.pack(">Q", value))[0]


def ceil_float(value):
    result = float(value)
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    require(Fraction(result) >= value, "Outward rounded geometric upper bound")
    return result


class RadiusGeometry:
    """Immutable source geometry; no OnlineBlock10Emulator or perception draws.

    Point queries use exactly the pinned retrieval inequalities fl(L-r) <= s
    <= fl(L+r), retaining the frozen row/date thresholds. Geometry diagnostics
    may vary a LOCAL variable r; no frozen module attribute is ever assigned.
    """
    def __init__(self, source_values, source_dates, base, min_rows, min_dates):
        import numpy as np
        values = np.asarray(source_values, dtype=float)
        dates = np.asarray(source_dates, dtype=str)
        require(values.ndim == dates.ndim == 1 and len(values) == len(dates)
                and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Finite source geometry in [0,1]")
        order = np.argsort(values, kind="mergesort")
        self.values, self.dates = values[order], dates[order]
        self.base, self.min_rows, self.min_dates = base, min_rows, min_dates
        require(0 < base <= 1 and len(values) >= min_rows and len(set(dates)) >= min_dates, "Global source can meet unchanged thresholds")

    def counts(self, query, radius):
        import numpy as np
        lo = int(np.searchsorted(self.values, query - radius, side="left"))
        hi = int(np.searchsorted(self.values, query + radius, side="right"))
        return hi - lo, len(np.unique(self.dates[lo:hi]))

    def admissible(self, query, radius):
        rows, dates = self.counts(query, radius)
        return rows >= self.min_rows and dates >= self.min_dates

    def required(self, query):
        """Distance event bracket then exact binary64 lower_bound, never a grid.

        Rounded source distances are event approximations, not blindly accepted
        radii. The first successful distance brackets the answer with the prior
        failed event; binary64 refinement handles addition/subtraction rounding.
        Immediate predecessor failure certifies global minimality by monotonicity.
        """
        import numpy as np
        query = float(query)
        require(math.isfinite(query) and 0 <= query <= 1, "Query domain; no clipping")
        base_rows, base_dates = self.counts(query, self.base)
        radius = self.base
        predecessor_fails = None
        if not self.admissible(query, radius):
            changes = sorted(set(float(d) for d in np.abs(self.values - query) if d > self.base) | {1.0})
            left, right = 0, len(changes) - 1
            require(self.admissible(query, changes[right]), "Source-distance upper bracket")
            while left < right:
                mid = (left + right) // 2
                if self.admissible(query, changes[mid]):
                    right = mid
                else:
                    left = mid + 1
            failing = self.base if left == 0 else changes[left - 1]
            require(not self.admissible(query, failing), "Failed lower distance event")
            low, high = bits(failing) + 1, bits(changes[left])
            while low < high:
                mid = (low + high) // 2
                if self.admissible(query, number(mid)):
                    high = mid
                else:
                    low = mid + 1
            radius = number(low)
            predecessor_fails = not self.admissible(query, math.nextafter(radius, -math.inf))
            require(predecessor_fails, "Minimal binary64 radius certificate")
        require(radius >= self.base and self.admissible(query, radius), "Radius passes frozen thresholds")
        rows, dates = self.counts(query, radius)
        return {"representative_L": query, "base_local_rows": base_rows, "base_local_dates": base_dates,
                "required_radius": radius, "radius_expansion": radius - self.base,
                "radius_multiplier": radius / self.base, "rows_at_required_radius": rows,
                "dates_at_required_radius": dates, "previous_binary64_radius_fails": predecessor_fails}

    def build_continuous_geometry(self):
        """Exact rational lower envelope of all minimal admissible source windows.

        Any centered candidate interval contains a contiguous sorted-source
        window. For each left row, a two-pointer scan finds its minimal right
        row meeting BOTH frozen thresholds. Dominated windows are removed.
        For a window [a,b], required radius is max(base,L-a,b-L). Adjacent
        undominated windows switch at (a_previous+b_next)/2. These switches,
        window centers and base-radius intersections are ALL geometry events.
        Fractions represent the input binary64 source values exactly.
        """
        right, dates, by_left = 0, Counter(), {}
        for left in range(len(self.values)):
            while right < len(self.values) and (right - left < self.min_rows or len(dates) < self.min_dates):
                dates[self.dates[right]] += 1
                right += 1
            if right - left >= self.min_rows and len(dates) >= self.min_dates:
                a, b = Fraction(float(self.values[left])), Fraction(float(self.values[right - 1]))
                by_left[a] = min(by_left.get(a, b), b)
            dates[self.dates[left]] -= 1
            if dates[self.dates[left]] == 0:
                del dates[self.dates[left]]
        windows = []
        for a, b in sorted(by_left.items()):
            while windows and b <= windows[-1][1]:
                windows.pop()
            windows.append((a, b))
        require(windows and all(windows[i][0] < windows[i+1][0] and windows[i][1] < windows[i+1][1]
                                for i in range(len(windows)-1)), "Undominated ordered windows")
        base = Fraction(self.base)
        switches = [(windows[i][0] + windows[i+1][1]) / 2 for i in range(len(windows)-1)]
        self.segments = []
        for i, (a, b) in enumerate(windows):
            left = Fraction(0) if i == 0 else switches[i-1]
            right = Fraction(1) if i == len(windows)-1 else switches[i]
            points = sorted({left, right} | {x for x in ((a+b)/2, b-base, a+base) if left < x < right})
            for x, y in zip(points, points[1:]):
                if x == y:
                    continue
                midpoint = (x+y)/2
                if base >= max(midpoint-a, b-midpoint):
                    slope, intercept = 0, base
                elif b-midpoint >= midpoint-a:
                    slope, intercept = -1, b
                else:
                    slope, intercept = 1, -a
                self.segments.append((x, y, slope, intercept))
        require(self.segments[0][0] == 0 and self.segments[-1][1] == 1, "Entire physical domain geometry")
        for first, second in zip(self.segments, self.segments[1:]):
            require(first[1] == second[0] and first[2]*first[1]+first[3] == second[2]*second[0]+second[3], "Exact continuous geometry partition")
        self.starts = [s[0] for s in self.segments]
        self.knots = self.starts + [Fraction(1)]
        values = [self.value(x) for x in self.knots]
        size = 1
        while size < len(values):
            size *= 2
        self.tree_size = size
        self.tree = [Fraction(0)] * (2*size)
        self.tree[size:size+len(values)] = values
        for i in range(size-1, 0, -1):
            self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
        self.window_count = len(windows)

    def value(self, latent):
        index = min(bisect.bisect_right(self.starts, latent)-1, len(self.segments)-1)
        require(index >= 0, "Geometry query domain")
        _, _, slope, intercept = self.segments[index]
        return slope*latent+intercept

    def envelope_bound(self, lower, upper):
        """Exact real-geometric maximum, rounded outward: certified float upper bound.

        If r >= |L-source_L| in real arithmetic, correctly rounded L +/- r
        still includes the representable source endpoint. Thus outward rounding
        the continuous maximum covers every binary64 L in this envelope. It
        need not equal the minimum radius for every floating-point query.
        """
        low, high = Fraction(lower), Fraction(upper)
        require(0 <= low <= high <= 1, "Valid recorded envelope")
        result = max(self.value(low), self.value(high))
        left = bisect.bisect_left(self.knots, low) + self.tree_size
        right = bisect.bisect_right(self.knots, high) + self.tree_size
        while left < right:
            if left % 2:
                result = max(result, self.tree[left]); left += 1
            if right % 2:
                right -= 1; result = max(result, self.tree[right])
            left //= 2; right //= 2
        return ceil_float(result)

    def full_domain_summary(self):
        """Uniform Lebesgue-L quantiles, exact event sweep; never a sampled grid."""
        events, atoms = defaultdict(int), defaultdict(Fraction)
        maximum, worst = Fraction(-1), None
        for left, right, slope, intercept in self.segments:
            a, b = slope*left+intercept, slope*right+intercept
            for value, location in ((a,left),(b,right)):
                if value > maximum:
                    maximum, worst = value, location
            if slope == 0:
                atoms[a] += right-left
            else:
                events[min(a,b)] += 1
                events[max(a,b)] -= 1
        knots = sorted(set(events) | set(atoms))
        density, mass, previous, quantiles = 0, Fraction(0), knots[0], {}
        targets = {q: Fraction(q,100) for q in (50,90,95,99)}
        for radius in knots:
            before_atom = mass + density*(radius-previous)
            for q, target in targets.items():
                if q not in quantiles and target <= before_atom:
                    require(density > 0, "Positive continuous quantile density")
                    quantiles[q] = previous+(target-mass)/density
            mass = before_atom + atoms[radius]
            for q, target in targets.items():
                if q not in quantiles and target <= mass:
                    quantiles[q] = radius
            density += events[radius]
            previous = radius
        require(mass == 1 and len(quantiles) == 4, "Exact full-domain probability mass")
        return {"max": float(maximum), **{f"P{q}": float(v) for q,v in quantiles.items()},
                "worst_L_location": float(worst), "exact_max_fraction": str(maximum),
                "exact_quantile_fractions": {str(q): str(v) for q,v in quantiles.items()},
                "exact_worst_L_fraction": str(worst), "certified_binary64_max_upper_bound": ceil_float(maximum),
                "measure": "Uniform Lebesgue measure in L over [0,1]; exact real source-distance geometry",
                "floating_point_scope": "These are continuous geometry statistics, not a uniform distribution over binary64 numbers"}


def read_evidence(head):
    sources = {}
    for name, expected in PINS.items():
        require(digest(FROZEN/name) == expected, "Frozen P2-1E-2 artifact: "+name)
        sources[(FROZEN/name).relative_to(ROOT).as_posix()] = expected
    audit = json.loads((FROZEN/"audit_summary.json").read_bytes())
    require(audit["HEAD"] == CHECKPOINT and audit["stage_pass"] is True and audit["support_closure_pass"] is False
            and audit["scientific_status"] == "RL_PERCEPTION_SUPPORT_CLOSURE_FAIL_CONFIRMED"
            and audit["witness_reproduction_pass"] is True and audit["failed_gates"] == []
            and audit["formal_trajectory_access_count"] == audit["formal_perception_seed_access_count"] == audit["stochastic_perception_draws"] == 0,
            "Accepted support closure failure")
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, expected in audit["output_sha256"].items():
        path = (FROZEN/name).resolve()
        require(path.is_relative_to(FROZEN) and digest(path) == expected, "Complete predecessor output provenance")
        sources[path.relative_to(ROOT).as_posix()] = expected
    require(json.loads((FROZEN/"output_hashes.json").read_bytes())["hashes"] ==
            {n:h for n,h in audit["output_sha256"].items() if n != "output_hashes.json"}, "Predecessor hash index identity")
    provenance = json.loads((FROZEN/"source_artifact_hashes.json").read_bytes())
    require(provenance["HEAD"] == CHECKPOINT and provenance["git_blobs"][PREDECESSOR] == PREDECESSOR_BLOB, "Predecessor source identity")
    for name, expected in provenance["sha256"].items():
        require(digest(ROOT/name) == expected, "Inherited frozen input: "+name)
        sources[name] = expected
    blobs = provenance["git_blobs"]
    for name, blob in blobs.items():
        require(git("rev-parse", f"{CHECKPOINT}:{name}") == git("rev-parse", f"{head}:{name}")
                == git("rev-parse", f":{name}") == git("hash-object", f"--path={name}", name) == blob, "Frozen source: "+name)
    return sources, blobs


def burden_statistics(values, base):
    import numpy as np
    a = np.asarray(values, dtype=float)
    require(len(a) > 0 and np.isfinite(a).all(), "Finite nonempty envelope burden")
    return {"envelope_day_count": len(a), "max_required_radius_upper_bound": float(a.max()),
            **{f"P{q}_required_radius_upper_bound": float(np.percentile(a,q,method="linear")) for q in (50,90,95,99)},
            **{f"fraction_upper_bound_above_{b:g}": float(np.mean(a>b)) for b in (base,.015,.02,.03,.05)}}


def read_reachable_burden(geometry, protocol):
    """Consume paired frozen CSVs; never construct or rerun a trajectory."""
    groups, role_values = [], {"TRAINING": [], "DEVELOPMENT": []}
    expected = ((role, year, root, tid, day) for role in role_values
                for part in protocol.partitions if part.role == role
                for year,root in zip(part.years,part.roots) for tid in part.trajectory_ids
                for day in range(protocol.decision_reward_steps))
    current, values = None, []
    with (FROZEN/"reachable_state_envelope.csv").open(newline="",encoding="utf-8") as ef, \
         (FROZEN/"reachable_support_overlap.csv").open(newline="",encoding="utf-8") as of:
        for row, overlap, key in zip_longest(csv.DictReader(ef), csv.DictReader(of), expected):
            require(row is not None and overlap is not None and key is not None, "Exact frozen envelope population")
            actual = (row["role"],row["year"],int(row["environment_root"]),int(row["trajectory_id"]),int(row["day_index"]))
            require(actual == key and all(overlap[k] == v for k,v in row.items()), "TRAINING/DEVELOPMENT only; exact paired artifact keys/values")
            group = key[:4]
            if current is not None and group != current:
                groups.append({"role":current[0],"year":current[1],"environment_root":current[2],"trajectory_id":current[3],
                               **burden_statistics(values,geometry.base)})
                values = []
            current = group
            bound = geometry.envelope_bound(float(row["L_lower"]),float(row["L_upper"]))
            values.append(bound)
            role_values[key[0]].append(bound)
    require(current is not None, "Nonempty reachable population")
    groups.append({"role":current[0],"year":current[1],"environment_root":current[2],"trajectory_id":current[3],
                   **burden_statistics(values,geometry.base)})
    require(all(g["envelope_day_count"] == protocol.decision_reward_steps for g in groups), "Complete year/trajectory aggregates")
    return groups, {role:burden_statistics(v,geometry.base) for role,v in role_values.items()}


def run_formal():
    require(not STAGE_DIRECTORY.exists(), "Exclusive output; no overwrite/resume")
    require(git("status","--porcelain","--untracked-files=all") == "", "Committed unchanged clean tree required")
    head = git("rev-parse","HEAD")
    self_blob = git("rev-parse",f"{head}:{SELF}")
    require(self_blob == git("rev-parse",f":{SELF}") == git("hash-object",f"--path={SELF}",SELF), "Published source required")
    sources, blobs = read_evidence(head)
    sources[SELF], blobs[SELF] = digest(ROOT/SELF), self_blob
    import paper2_gym_pomdp_env_v1 as core
    import paper2_ppo_protocol_v1 as frozen
    import run_paper2_stage1e2_rl_perception_support_closure_audit_v1 as predecessor
    # Metadata/source reads only: no Paper2EnvAssets or OnlineBlock10Emulator.
    failure, online, authority, inherited_sources, inherited_blobs = predecessor.read_evidence(head)
    sources.update(inherited_sources)
    blobs.update(inherited_blobs)
    sampler = core.load_frozen_module("adaptive_radius_source_reader",core.HELPERS["perception"][0])
    constants = {n:getattr(sampler,n) for n in ("LOCAL_RADIUS","MIN_LOCAL_SAMPLES","MIN_LOCAL_DATES","KERNEL_BANDWIDTH","BLOCK_MINUTES")}
    authority_fields = {"LOCAL_RADIUS":"local_radius_abs_L", "MIN_LOCAL_SAMPLES":"local_min_samples",
                        "MIN_LOCAL_DATES":"local_min_dates", "KERNEL_BANDWIDTH":"kernel_bandwidth",
                        "BLOCK_MINUTES":"block_minutes"}
    require(all(constants[n] == online["frozen_perception_design"][field] for n,field in authority_fields.items()),
            "All frozen perception constants exactly inherited")
    require(constants["MIN_LOCAL_SAMPLES"] == online["frozen_perception_design"]["local_min_samples"] == 20
            and constants["MIN_LOCAL_DATES"] == online["frozen_perception_design"]["local_min_dates"] == 3
            and constants["LOCAL_RADIUS"] == online["frozen_perception_design"]["local_radius_abs_L"], "Unchanged frozen support thresholds")
    # The source loader preserves accepted row/date geometry; no prediction fields are used.
    source_path = core.DEFAULT_PAPER1_CQR
    require(digest(source_path) == sampler.EXPECTED_HASHES["paper1_cqr"], "Same frozen Paper1 source")
    source = sampler.load_paper1_cqr(source_path)
    require(digest(source_path) == sampler.EXPECTED_HASHES["paper1_cqr"], "Source unchanged across loading")
    sources[str(source_path)] = sampler.EXPECTED_HASHES["paper1_cqr"]
    # Guard the exact retrieval algebra used by counts(), including inclusivity.
    tree = ast.parse((ROOT/core.HELPERS["perception"][0]).read_text(encoding="utf-8"))
    retrieval = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == "_candidate_indices")
    require(hashlib.sha256(ast.dump(retrieval).encode()).hexdigest() == predecessor.AST_PINS["_candidate_indices"], "Pinned inclusive candidate geometry")
    geometry = RadiusGeometry(source.true_L.to_numpy(),source.date.astype(str).to_numpy(),
                              constants["LOCAL_RADIUS"],constants["MIN_LOCAL_SAMPLES"],constants["MIN_LOCAL_DATES"])
    witness_old = json.loads((FROZEN/"failure_witness_reproduction.json").read_bytes())
    context = witness_old["original_context"]
    require(context == failure["context"], "Exact original Point-PPO failure context")
    witness = geometry.required(context["L_true"])
    require(witness_old["reproduction_pass"] is True and witness["base_local_rows"] == context["local_rows"] == 19
            and witness["base_local_dates"] == context["local_dates"] == 6
            and witness["required_radius"] > geometry.base, "Frozen constructive witness consistency")
    witness.update({"original_context":context,"diagnostic_only":True,"perception_sample_generated":False})
    interval_rows = []
    with (FROZEN/"perception_support_intervals.csv").open(newline="",encoding="utf-8") as stream:
        for interval in csv.DictReader(stream):
            if interval["supported"] == "True":
                continue
            require(interval["supported"] == "False", "Frozen support flag")
            left,right = float(interval["interval_left"]),float(interval["interval_right"])
            right_query = right if interval["right_closed"] == "True" else math.nextafter(right,-math.inf)
            left_query = left if interval["left_closed"] == "True" else math.nextafter(left,math.inf)
            for label,query in (("left_endpoint_or_limit",left_query),("representative",float(interval["representative_L"])),("right_endpoint_or_limit",right_query)):
                require(left <= query <= right, "Interval query membership")
                result = geometry.required(query)
                require(result["required_radius"] > geometry.base, "Unsupported interval base-radius regression")
                interval_rows.append({"interval_left":left,"interval_right":right,"probe_role":label,**result})
    closure = json.loads((FROZEN/"support_closure_summary.json").read_bytes())
    require(len(interval_rows) == 3*closure["number_of_unsupported_intervals"], "Every unsupported interval tested")
    geometry.build_continuous_geometry()
    full = geometry.full_domain_summary()
    protocol = frozen.Protocol()
    frozen.validate_protocol(protocol)
    burden, role_stats = read_reachable_burden(geometry,protocol)
    # Exact binary64 point radii must be bounded by the continuous certificate.
    require(all(geometry.envelope_bound(r["representative_L"],r["representative_L"]) >= r["required_radius"]
                for r in interval_rows+[witness]), "Real-geometry certificates dominate exact binary64 queries")
    summary = {"stage":"P2-1E-3A-v1","base_radius":geometry.base,"quality_thresholds_unchanged":True,
        "perception_repair_applied":False,"witness_required_radius":witness["required_radius"],
        "reachable_max_required_radius":max(r["max_required_radius_upper_bound"] for r in role_stats.values()),
        "reachable_max_scope":"CERTIFIED UPPER BOUND over conservative recorded envelopes, not observed policy occupancy",
        "reachable_by_role":role_stats,"reachable_statistical_unit":"One conservative trajectory/day envelope, NOT independent RL seed",
        "reachable_percentile_method":"linear percentiles of per-envelope certified worst-case upper bounds",
        "fraction_semantics":"Fractions of upper bounds exceeding descriptive bins; conservative, not exact fractions of visited states",
        "full_domain_max_required_radius":full["max"],"full_domain":full,
        "source_window_count":geometry.window_count,"geometry_segment_count":len(geometry.segments),
        "declaration":DECLARATION,"independent_fidelity_validation_still_required":True}
    manifest = {"stage":"P2-1E-3A-v1","mode":"formal","role":"SUPPORT-DEFINITION FEASIBILITY ONLY",
        "frozen_constants":constants,"development_perception_seed_authority":authority["perception_seed"],
        "point_query_algorithm":"Source-distance event search then inclusive frozen-retrieval binary64 refinement; previous radius must fail",
        "continuous_algorithm":"Exact rational lower envelope of minimal row/date-admissible source windows; event-based CDF and RMQ; no grid",
        "rounding_certificate":"Outward rounding of real-geometric envelope maximum includes representable source endpoints under monotone IEEE rounding",
        "full_domain_measure":"Uniform Lebesgue L on [0,1], not uniform binary64 or source-row distribution",
        "descriptive_bins":[geometry.base,.015,.02,.03,.05],"bins_are_selection_rules":False,
        "no_fidelity_claim":True,"no_repair_selection":True,"no_perception_sampling":True,"no_environment_execution":True,
        "declaration":DECLARATION}
    documents = {"adaptive_radius_summary.json":summary,"witness_radius_requirement.json":witness,
        "protocol_manifest.json":manifest,"source_artifact_hashes.json":{"HEAD":head,"predecessor_HEAD":CHECKPOINT,
            "sha256":sources,"git_blobs":blobs}}
    content = {n:encode(v) for n,v in documents.items()}
    content["unsupported_interval_radius_requirements.csv"] = csv_bytes(interval_rows)
    content["reachable_radius_burden.csv"] = csv_bytes(burden)
    content["full_domain_radius_geometry.csv"] = csv_bytes([{"left_exact":str(a),"right_exact":str(b),
        "slope":s,"intercept_exact":str(c)} for a,b,s,c in geometry.segments])
    # Claim the directory once; no old output is ever written or removed.
    STAGE_DIRECTORY.mkdir(parents=True,exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    for name,data in content.items():
        with (OUTPUT/name).open("xb") as stream:
            stream.write(data)
        require((OUTPUT/name).read_bytes() == data,"Exact output readback: "+name)
    hashes = {n:hashlib.sha256(data).hexdigest() for n,data in content.items()}
    index = encode({"hashes":hashes,"exclusions":{"output_hashes.json":"SHA256 in final audit; no self-reference","audit_summary.json":"Last completion marker; geometry only"}})
    with (OUTPUT/"output_hashes.json").open("xb") as stream:
        stream.write(index)
    require((OUTPUT/"output_hashes.json").read_bytes() == index and {p.name for p in OUTPUT.iterdir()} == set(content)|{"output_hashes.json"}
            and all(digest(OUTPUT/n) == h for n,h in hashes.items()),"Complete exact output inventory and hashes")
    require(git("rev-parse","HEAD") == head and git("status","--porcelain","--untracked-files=all") == ""
            and all(digest(ROOT/n) == h for n,h in sources.items()),"Source/HEAD/evidence unchanged")
    require(all(git("rev-parse",f"{head}:{n}") == git("rev-parse",f":{n}") == git("hash-object",f"--path={n}",n) == b
                for n,b in blobs.items()),"Publication source integrity")
    audit = {"stage":"P2-1E-3A-v1","stage_pass":True,"scientific_status":"ADAPTIVE_RADIUS_SUPPORT_FEASIBILITY_DIAGNOSTIC_COMPLETE",
        "stage_pass_meaning":"GEOMETRY DIAGNOSTIC EXECUTED; NOT PERCEPTION VALIDATION OR REPAIR APPROVAL",
        "declaration":DECLARATION,"HEAD":head,"failed_gates":[],"quality_thresholds_unchanged":True,
        "perception_repair_applied":False,"perception_samples_generated":0,"PPO_training_runs":0,
        "formal_trajectory_access_count":0,"formal_perception_seed_access_count":0,
        "output_sha256":{**hashes,"output_hashes.json":hashlib.sha256(index).hexdigest()}}
    data = encode(audit)
    with (OUTPUT/"audit_summary.json").open("xb") as stream:
        stream.write(data)
    require((OUTPUT/"audit_summary.json").read_bytes() == data,"Final audit readback")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(),indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
