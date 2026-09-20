"""P2-1E-2-v1: RL-Reachability / Frozen-Perception Support Closure Audit.

Inert import; --mode formal explicitly executes a diagnostic, never PPO or
perception repair. Support means the frozen sampler reached its first random
draw, not that a random perception sample was generated. Existing evidence is
read-only. Any inconsistency stops without a successful audit marker.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage1e2_rl_perception_support_closure_audit_v1.py"
SMOKE_SOURCE = "experiments/run_paper2_stage2d1_point_ppo_smoke_audit_v1.py"
SMOKE_BLOB = "435800547721559f45d54f0cd5576dfa642b14c8"
CHECKPOINT = "61d4a746c72d2a86e323b8fe7936420bd1639a04"
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
FAILED = BASE / "p2_2d_1_point_ppo_smoke_audit_v1/smoke"
FAILURE_SHA = "edfd260e5377ac2a5c0e1778054b8e38417c0b770404c12842e0077b91b9339b"
MODEL_SHA = "34596f06eb0754b9f86d5602d11564590b1141c70294262d68d03d5cc72d9381"
ONLINE_AUDIT = BASE / "p2_1c2b_online_counterfactual_perception_audit_v1_1/audit_summary.json"
ONLINE_SHA = "959fa7cd77b5b8ba914d63d4e9caa0a63be26c6d8d2de939fae80739ff7575c9"
STAGE_DIRECTORY = BASE / "p2_1e_2_rl_perception_support_closure_audit_v1"
OUTPUT = STAGE_DIRECTORY / "formal"
STATUS = "RL_PERCEPTION_SUPPORT_CLOSURE_FAIL_CONFIRMED"
INCONSISTENT = "FAIL_CLOSED_INCONSISTENT_SUPPORT_EVIDENCE"
DECLARATION = ("FROZEN PERCEPTION SUPPORT DOES NOT COVER THE CURRENT RL-REACHABLE STATE DOMAIN; "
               "P2-2D POINT/UA PPO REMAINS BLOCKED; NO SUPPORT PARAMETER WAS CHANGED")
AST_PINS = {
    "rain_next_samples": "7c00d651eafc9947ca3d4e4c5c9fd3f5bd7c3834349bfcc55cef9620aa7c9c98",
    "dry_next_samples": "31b045f29e732b68b5a4e37194400441089520ba53d679d57622a6a7e53326e7",
    "_candidate_indices": "651fdb4bd4d3b152e0c623859d007c60e101e3abf113249bd4054732773737aa",
    "sample_one": "af4a18f134f86f3ef11491f2b6395e1d647eeadae1db7295b1e7f6106b3ddfbc",
}


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
                          check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def write_json(name, value):
    data = encode(value)
    with (OUTPUT / name).open("xb") as stream:
        stream.write(data)
    require((OUTPUT / name).read_bytes() == data, "JSON readback: " + name)


class VerifiedCSV:
    """Hash intended serialized bytes while streaming; readback verifies them."""
    def __init__(self, stream, fields):
        self.stream, self.fields = stream, list(fields)
        self.hash = hashlib.sha256()
        self.row_count = 0
        self._write(self.fields)

    def _write(self, values):
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerow(values)
        text = buffer.getvalue()
        self.stream.write(text)
        self.hash.update(text.encode("utf-8"))

    def writerow(self, row):
        require(set(row) == set(self.fields), "Exact CSV schema")
        self._write([row[k] for k in self.fields])
        self.row_count += 1

    def evidence(self):
        return {"sha256": self.hash.hexdigest(), "rows": self.row_count}


class SUPPORTED_BEFORE_RANDOM_DRAW(Exception):
    """Frozen admissibility checks passed; stop before any random draw."""


class NoSamplingRNG:
    def choice(self, *args, **kwargs):
        raise SUPPORTED_BEFORE_RANDOM_DRAW()


def support_probe(emulator, latent):
    """Classifier exclusively delegates to frozen sample_one. Counts are diagnostic."""
    import numpy as np
    require(math.isfinite(latent) and 0 <= latent <= 1, "Probe domain")
    try:
        emulator.sample_one(float(latent), NoSamplingRNG())
    except SUPPORTED_BEFORE_RANDOM_DRAW:
        supported = True
    except RuntimeError as exc:
        require(str(exc).startswith("Unsupported query "), "Unexpected sampler failure: " + str(exc))
        supported = False
    else:
        raise RuntimeError("Sampler returned a sample instead of reaching RNG sentinel")
    indices = emulator._candidate_indices(float(latent))
    return {"supported": supported, "representative_L": float(latent), "local_rows": len(indices),
            "local_dates": len(np.unique(emulator.date[indices])),
            "candidate_blocks": len(np.unique(emulator.block[indices])),
            "candidate_indices_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()}


def float_bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def bits_float(value):
    return struct.unpack(">d", struct.pack(">Q", value))[0]


def first_true_float(predicate):
    """Exact binary64 lower_bound on [0,1], not a numeric grid/tolerance search."""
    low, high = float_bits(0.0), float_bits(1.0)
    if not predicate(1.0):
        return None
    while low < high:
        mid = (low + high) // 2
        if predicate(bits_float(mid)):
            high = mid
        else:
            low = mid + 1
    return bits_float(low)


def support_cells(emulator, radius):
    """Exhaustive candidate-change cells, including floating-point boundary effects.

    Pinned _candidate_indices uses source_L >= fl(q-radius) and source_L <=
    fl(q+radius). These monotone comparisons are inverted over ALL nonnegative
    binary64 numbers. No local-row/date support criterion is reproduced here.
    Also retain every clipped source_L +/- radius as requested. Between cuts,
    no candidate can enter/leave; every cell is classified by frozen sample_one.

    Cells use [left,right), with {1} last. This is an exact map for the frozen
    simulator's binary64 states. Its extension between adjacent floating-point
    numbers is a declared interval convention, not a claim about real arithmetic.
    """
    cuts = {0.0, 1.0}
    for value in sorted(set(map(float, emulator.L))):
        require(math.isfinite(value), "Finite frozen source true_L")
        cuts.update((max(0.0, min(1.0, value - radius)), max(0.0, min(1.0, value + radius))))
        # These are retrieval geometry only; classification remains the sentinel.
        for point in (first_true_float(lambda q: q + radius >= value),
                      first_true_float(lambda q: q - radius > value)):
            if point is not None:
                cuts.add(point)
    ordered = sorted(cuts)
    for i, left in enumerate(ordered):
        right = ordered[i + 1] if i + 1 < len(ordered) else left
        representative = left + (right - left) / 2
        if representative >= right:
            representative = left
        probe = support_probe(emulator, representative)
        # Probe every cut and both relevant one-sided limits, without draws.
        points = {left, representative}
        if right > left:
            points.add(math.nextafter(right, -math.inf))
        for point in points:
            other = support_probe(emulator, point)
            require(other["supported"] == probe["supported"]
                    and other["candidate_indices_sha256"] == probe["candidate_indices_sha256"], "Candidate-constant exact cell")
        yield {"interval_left": left, "interval_right": right, "left_closed": True,
               "right_closed": left == right, **probe}


def merge_cells(cells):
    """Maximal equal-support intervals; count diagnostics refer to representative_L."""
    regions = []
    for cell in cells:
        if regions and regions[-1]["supported"] == cell["supported"]:
            require(regions[-1]["interval_right"] == cell["interval_left"], "Exact adjacent partition")
            regions[-1]["interval_right"] = cell["interval_right"]
            regions[-1]["right_closed"] = cell["right_closed"]
        else:
            regions.append(dict(cell))
    require(regions and regions[0]["interval_left"] == 0 and regions[-1]["interval_right"] == 1
            and regions[-1]["right_closed"], "Complete [0,1] support map")
    return regions


def locate(regions, starts, latent):
    index = bisect.bisect_right(starts, latent) - 1
    require(index >= 0, "State inside support map")
    row = regions[index]
    require(row["interval_left"] <= latent <= row["interval_right"]
            and (latent < row["interval_right"] or row["right_closed"]), "Interval endpoint membership")
    return index


def first_overlap(unsupported, ends, lower, upper):
    """Closed envelope vs half-open support regions; exact endpoint handling."""
    index = bisect.bisect_left(ends, lower)
    while index < len(unsupported):
        row = unsupported[index]
        if row["interval_right"] == lower and not row["right_closed"]:
            index += 1
            continue
        if row["interval_left"] > upper:
            return None
        return row
    return None


def monotonicity_audit(assets, source_paths):
    """Verify frozen analytic form and its coefficient conditions, not just prose.

    Pinned dry is clip(x+innovation); rain is clip(x+a+b*x+residual).
    The real-arithmetic slopes are 1 and 1+b. CLEAN multiplier is 1-eta.
    Frozen functions additionally check endpoints for every innovation. Actual
    paired binary64 trajectories must remain ordered at EVERY recorded step.
    This does not claim every real number inside an envelope is reachable.
    """
    import numpy as np
    found = {}
    for path in source_paths:
        for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name in AST_PINS:
                found[node.name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
    require(found == AST_PINS, "Frozen sampler geometry and affine-clipped dynamics AST identity")
    require(np.isfinite([assets.eta, assets.r3["slope"], assets.r3["intercept"]]).all()
            and 0 <= assets.eta <= 1 and 1 + assets.r3["slope"] >= 0, "Monotone transition/CLEAN coefficient conditions")
    rain = {**assets.r3, "candidate": assets.reward_protocol["environment_parameters"]["rain"]}
    for first, last in ((assets.natural.dry_next_samples(0.0, assets.dry), assets.natural.dry_next_samples(1.0, assets.dry)),
                        (assets.natural.rain_next_samples(0.0, rain), assets.natural.rain_next_samples(1.0, rain))):
        require(np.isfinite(first).all() and np.isfinite(last).all() and (first <= last).all(), "Frozen-function endpoint monotonic regression")
    return {"passed": True, "AST_sha256": found, "CLEAN_multiplier": 1 - assets.eta,
            "dry_slope": 1, "rain_slope": 1 + assets.r3["slope"],
            "proof": "Pinned affine maps with nonnegative slopes composed with monotone physical projection and CLEAN multiplier",
            "floating_point_check": "Every actual lower/upper state pair must be ordered; no envelope tolerance or state clipping added",
            "claim": "Conservative enclosing intervals; not every interior real is binary-policy reachable"}


def read_evidence(head):
    sources = {}

    def capture(path, expected):
        path = Path(path)
        require(digest(path) == expected, "Pinned evidence: " + str(path))
        key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        sources[key] = expected

    capture(FAILED / "perception_support_failure.json", FAILURE_SHA)
    failure = json.loads((FAILED / "perception_support_failure.json").read_bytes())
    require(failure.get("HEAD") == CHECKPOINT and failure.get("stage_pass") is False
            and failure.get("scientific_status") == "POINT_PPO_SMOKE_PERCEPTION_SUPPORT_FAIL"
            and failure.get("action") == "STOP; NO RETRY, RESAMPLING, FALLBACK OR PERCEPTION PARAMETER CHANGE", "Accepted failure identity")
    expected_context = {"year": "YEAR1", "trajectory_id": 0, "day_index": 133, "date": "2021-12-20",
                        "L_true": 0.21687332636557485, "perception_seed": 20260906,
                        "local_rows": 19, "local_dates": 6, "candidate_blocks": 17}
    require(failure["context"] == expected_context, "Exact observed witness regression")
    # Opaque archive evidence only; no PPO import/load/predict.
    capture(FAILED / "final_smoke_model.zip", MODEL_SHA)
    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    require(git("rev-parse", f"{CHECKPOINT}:{SMOKE_SOURCE}") == git("rev-parse", f"{head}:{SMOKE_SOURCE}")
            == git("hash-object", f"--path={SMOKE_SOURCE}", SMOKE_SOURCE) == SMOKE_BLOB, "Frozen failed-smoke execution order")
    blobs = {}
    for name, expected in failure["source_sha256"].items():
        capture(ROOT / name, expected)
        if name.startswith("experiments/") and name.endswith(".py"):
            blob = git("rev-parse", f"{CHECKPOINT}:{name}")
            require(blob == git("rev-parse", f"{head}:{name}") == git("rev-parse", f":{name}")
                    == git("hash-object", f"--path={name}", name), "Frozen source unchanged: " + name)
            blobs[name] = blob
    capture(ONLINE_AUDIT, ONLINE_SHA)
    online = json.loads(ONLINE_AUDIT.read_bytes())
    require(online["stage_pass"] is True and online["implementation_version"] == "v1.1_R1_VERIFIED_FROZEN_SAMPLER"
            and online["frozen_perception_design"]["fallback"] is False, "Accepted P2-1C perception authority")
    # Accepted P2-1E-1 pins and DEV-seed resolution are reused without trajectories.
    import paper2_perception_ppo_runner_v1 as bridge
    authority = bridge.read_perception_authority()
    for name, expected in bridge.AUTHORITY_HASHES.items():
        capture(bridge.AUTHORITY / name, expected)
    require(bridge.get_perception_contract().development_seed == failure["context"]["perception_seed"], "DEV seed authority")
    return failure, online, authority, sources, blobs


def audit_envelopes(runner, assets, regions, witness_context):
    """All frozen TRAINING/DEVELOPMENT specs; two True-State core delegates only."""
    import numpy as np
    p = runner.get_protocol()
    unsupported = [r for r in regions if not r["supported"]]
    ends = [r["interval_right"] for r in unsupported]
    ledger = runner.AccessLedger()
    totals = {role: {"rows": 0, "unsupported_rows": 0} for role in ("TRAINING", "DEVELOPMENT")}
    affected_years, affected_trajectories, witness = set(), set(), None
    envelope_fields = ["role", "year", "environment_root", "trajectory_id", "day_index", "date", "L_lower", "L_upper"]
    overlap_fields = envelope_fields + ["unsupported_overlap", "first_unsupported_left", "first_unsupported_right"]
    with (OUTPUT / "reachable_state_envelope.csv").open("x", newline="", encoding="utf-8") as ef, \
         (OUTPUT / "reachable_support_overlap.csv").open("x", newline="", encoding="utf-8") as of:
        ew, ow = VerifiedCSV(ef, envelope_fields), VerifiedCSV(of, overlap_fields)
        for role in ("TRAINING", "DEVELOPMENT"):
            part = runner.partition(role)
            for year, root in zip(part.years, part.roots):
                for tid in part.trajectory_ids:
                    row = {"year": year, "environment_root": root, "trajectory_id": tid,
                           "population_size": len(part.trajectory_ids), "perception_seed": None}
                    lower = runner.construct_core(assets, ledger, role, row)
                    try:
                        upper = runner.construct_core(assets, ledger, role, row)
                        try:
                            lo_obs, _ = lower.reset()
                            hi_obs, _ = upper.reset()
                            runner.audit_observation(lower, lo_obs)
                            runner.audit_observation(upper, hi_obs)
                            require(np.array_equal(lower._indices, upper._indices) and np.array_equal(lower._rain, upper._rain), "Paired exogenous trajectory")
                            actions = {name: action for action, name in p.actions}
                            for day in range(p.decision_reward_steps):
                                low, high = lower._audit_snapshot(), upper._audit_snapshot()
                                require(low["day_index"] == high["day_index"] == day and low["date"] == high["date"]
                                        and low["L_pre"] <= high["L_pre"], "Ordered frozen reachable envelope")
                                result = {"role": role, "year": year, "environment_root": root, "trajectory_id": tid,
                                          "day_index": day, "date": low["date"], "L_lower": low["L_pre"], "L_upper": high["L_pre"]}
                                hit = first_overlap(unsupported, ends, low["L_pre"], high["L_pre"])
                                ew.writerow(result)
                                ow.writerow({**result, "unsupported_overlap": hit is not None,
                                             "first_unsupported_left": "" if hit is None else hit["interval_left"],
                                             "first_unsupported_right": "" if hit is None else hit["interval_right"]})
                                totals[role]["rows"] += 1
                                if hit is not None:
                                    totals[role]["unsupported_rows"] += 1
                                    affected_years.add((role, year))
                                    affected_trajectories.add((role, year, root, tid))
                                if role == "DEVELOPMENT" and (year, tid, day) == (witness_context["year"], witness_context["trajectory_id"], witness_context["day_index"]):
                                    require(witness is None and result["date"] == witness_context["date"]
                                            and low["L_pre"] <= witness_context["L_true"] <= high["L_pre"] and hit is not None,
                                            INCONSISTENT + ": observed witness outside envelope")
                                    witness = result
                                for env, action in ((lower, actions["CLEAN"]), (upper, actions["WAIT"])):
                                    obs, reward, terminated, truncated, _ = env.step(action)
                                    snap = runner.audit_observation(env, obs, terminal=terminated)
                                    require(not truncated and terminated == (day == p.decision_reward_steps - 1)
                                            and np.isfinite(reward) and snap["perception_query_count"] == 0, "Frozen horizon/finite reward/no perception sample")
                        finally:
                            upper.close()
                    finally:
                        lower.close()
    for role in totals:
        require(totals[role]["rows"] == runner.partition(role).population_size * p.decision_reward_steps, "Full envelope population")
    require(witness is not None and ledger.heldout_ppo_trajectory_access_count == ledger.denied_accesses == 0, "Witness present and no formal access")
    require(len(ledger.records) == 2 * sum(runner.partition(role).population_size for role in totals), "Exactly two envelopes per spec")
    require(assets.perception_sample_count == 0 and not assets._perception_cache, "Zero stochastic perception draws")
    allowed_banks = {(year, root, len(runner.partition(role).trajectory_ids)) for role in totals
                     for year, root in zip(runner.partition(role).years, runner.partition(role).roots)}
    require(set(assets._banks) <= allowed_banks, "Actual bank cache contains only permitted roots")
    return {"populations": totals, "years_affected": sorted(affected_years), "trajectories_affected": sorted(affected_trajectories),
            "affected_trajectory_counts": {role: sum(r[0] == role for r in affected_trajectories) for role in totals},
            "witness_envelope": witness, "heldout_ppo_trajectory_access_count": 0, "stochastic_perception_draws": 0,
            "csv_evidence": {"reachable_state_envelope.csv": ew.evidence(), "reachable_support_overlap.csv": ow.evidence()}}


def run_formal():
    require(not STAGE_DIRECTORY.exists(), "Exclusive new stage directory; no overwrite/resume")
    require(git("status", "--porcelain", "--untracked-files=all") == "", "Committed unchanged clean tree required")
    head = git("rev-parse", "HEAD")
    self_blob = git("rev-parse", f"{head}:{SELF}")
    require(self_blob == git("rev-parse", f":{SELF}") == git("hash-object", f"--path={SELF}", SELF), "Published unchanged diagnostic source")
    failure, online, authority, sources, blobs = read_evidence(head)
    sources[SELF], blobs[SELF] = digest(ROOT / SELF), self_blob
    import paper2_true_state_ppo_runner_v1 as runner
    import run_paper2_stage2c1_true_state_ppo_smoke_audit_v1 as smoke
    p = runner.get_protocol()
    require(smoke.dependency_versions() == dict(p.dependencies), "Frozen dependencies")
    assets = runner.load_assets()
    assets.ensure_perception()
    sampler = assets.perception
    constants = {name: getattr(sampler, name) for name in
                 ("LOCAL_RADIUS", "MIN_LOCAL_SAMPLES", "MIN_LOCAL_DATES", "KERNEL_BANDWIDTH", "BLOCK_MINUTES")}
    fields = {"LOCAL_RADIUS": "local_radius_abs_L", "MIN_LOCAL_SAMPLES": "local_min_samples",
              "MIN_LOCAL_DATES": "local_min_dates", "KERNEL_BANDWIDTH": "kernel_bandwidth", "BLOCK_MINUTES": "block_minutes"}
    require(all(constants[n] == online["frozen_perception_design"][key] for n, key in fields.items()), "Frozen support constants mechanically consumed")
    source_paths = [ROOT / runner.core.HELPERS[name][0] for name in ("natural", "perception")]
    monotone = monotonicity_audit(assets, source_paths)
    STAGE_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    try:
        probe = support_probe(assets.emulator, failure["context"]["L_true"])
        require(not probe["supported"] and all(probe[k] == failure["context"][k]
                for k in ("local_rows", "local_dates", "candidate_blocks")), INCONSISTENT + ": witness support reproduction")
        # Full candidate-constant cells are preserved as additional audit evidence.
        with (OUTPUT / "support_breakpoint_cells.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = None

            def recorded_cells():
                nonlocal writer
                for cell in support_cells(assets.emulator, constants["LOCAL_RADIUS"]):
                    if writer is None:
                        writer = VerifiedCSV(stream, list(cell))
                    writer.writerow(cell)
                    yield cell

            regions = merge_cells(recorded_cells())
            csv_evidence = {"support_breakpoint_cells.csv": writer.evidence()}
        with (OUTPUT / "perception_support_intervals.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = VerifiedCSV(stream, list(regions[0]))
            for region in regions:
                writer.writerow(region)
            csv_evidence["perception_support_intervals.csv"] = writer.evidence()
        idx = locate(regions, [r["interval_left"] for r in regions], failure["context"]["L_true"])
        require(not regions[idx]["supported"], INCONSISTENT + ": witness interval classified supported")
        envelope = audit_envelopes(runner, assets, regions, failure["context"])
        csv_evidence.update(envelope["csv_evidence"])
        witness = {"classification": "REACHABLE_UNSUPPORTED_WITNESS", "reproduction_pass": True,
                   "original_context": failure["context"], "support_probe": probe, "support_interval": regions[idx],
                   "reachable_envelope": envelope["witness_envelope"], "failure_artifact_sha256": FAILURE_SHA,
                   "root_resolution": "DEVELOPMENT inferred from pinned smoke source: pairing precedes training; saved model precedes DEVELOPMENT evaluation",
                   "model_evidence": "Archive hashed only; no loading, prediction, performance analysis or rollout replay"}
        unsupported = [r for r in regions if not r["supported"]]
        supported_measure = math.fsum(r["interval_right"] - r["interval_left"] for r in regions if r["supported"])
        unsupported_measure = math.fsum(r["interval_right"] - r["interval_left"] for r in unsupported)
        require(abs(supported_measure + unsupported_measure - 1) <= math.ulp(1.0), "Full support-map measure")
        counts = envelope["populations"]
        require(counts["DEVELOPMENT"]["unsupported_rows"] > 0, INCONSISTENT + ": closure contradicts witness")
        summary = {"support_closure_pass": False, "number_of_supported_intervals": len(regions) - len(unsupported),
            "number_of_unsupported_intervals": len(unsupported), "total_supported_L_measure": supported_measure,
            "total_unsupported_L_measure": unsupported_measure,
            "first_unsupported_interval_above_zero": next((r for r in unsupported if r["interval_right"] > 0), None),
            "highest_fully_supported_L_from_origin": regions[0]["interval_right"] if regions[0]["supported"] else None,
            "origin_supported_component_right_closed": regions[0]["right_closed"] if regions[0]["supported"] else None,
            "origin_extent_semantics": "Supremum of fully supported origin component; see endpoint flag",
            "witness_interval": regions[idx], "witness_reproduction_pass": True,
            "development_unsupported_envelope_count": counts["DEVELOPMENT"]["unsupported_rows"],
            "development_unsupported_envelope_fraction": counts["DEVELOPMENT"]["unsupported_rows"] / counts["DEVELOPMENT"]["rows"],
            **envelope, "monotonicity": monotone,
            "limitations": "Envelope overlap is conservative; only observed witness is constructive, not all interior real states",
            "interval_measure_semantics": "Length of declared half-open binary64-cut interval extension on [0,1]"}
        manifest = {"stage": "P2-1E-2-v1", "role": "DIAGNOSTIC ONLY; NO REPAIR", "support_constants": constants,
            "perception_seed_authority": authority["perception_seed"], "support_admissibility_seed_independent": True,
            "support_definition": "Frozen L/data admissibility before FIRST random choice; no stochastic samples",
            "support_map": "All source_L +/- LOCAL_RADIUS cuts plus exact binary64 candidate-entry/exit cuts; endpoints and one-sided limits probed",
            "diagnostic_counts_scope": "Counts/hash refer to representative_L, not every point in a merged interval",
            "roles": ["TRAINING", "DEVELOPMENT"], "monotonicity": monotone,
            "perception_draws": 0, "PPO_training": False, "model_loading": False, "repair": False,
            "historical_unsupported_context_role": "Corroboration only; never changes support rules"}
        write_json("failure_witness_reproduction.json", witness)
        write_json("support_closure_summary.json", summary)
        write_json("protocol_manifest.json", manifest)
        write_json("source_artifact_hashes.json", {"HEAD": head, "failure_HEAD": CHECKPOINT, "sha256": sources,
                   "git_blobs": blobs, "core_assets_provenance": assets.provenance})
        names = {"perception_support_intervals.csv", "support_breakpoint_cells.csv", "reachable_state_envelope.csv",
                 "reachable_support_overlap.csv", "failure_witness_reproduction.json", "support_closure_summary.json",
                 "protocol_manifest.json", "source_artifact_hashes.json"}
        require({x.name for x in OUTPUT.iterdir()} == names, "Exact output inventory")
        # CSV semantic readback: population, schema, nonmissing values and interval identity.
        for name in sorted(n for n in names if n.endswith(".csv")):
            require(digest(OUTPUT / name) == csv_evidence[name]["sha256"], "Exact intended CSV bytes read back: " + name)
            with (OUTPUT / name).open(newline="", encoding="utf-8") as stream:
                rows = csv.DictReader(stream)
                count = 0
                for row in rows:
                    require(None not in row and None not in row.values(), "CSV schema/readback")
                    if name == "perception_support_intervals.csv":
                        require(row == {k: str(v) for k, v in regions[count].items()}, "Exact interval readback")
                    count += 1
                expected = len(regions) if name == "perception_support_intervals.csv" else sum(c["rows"] for c in counts.values())
                require(count > 0 and (name == "support_breakpoint_cells.csv" or count == expected), "CSV population readback")
                require(count == csv_evidence[name]["rows"], "CSV generated/readback row identity")
        hashes = {name: digest(OUTPUT / name) for name in names}
        write_json("output_hashes.json", {"hashes": hashes, "exclusions": {
            "output_hashes.json": "SHA256 in audit_summary to avoid self-reference", "audit_summary.json": "Last successful audit marker; not closure PASS"}})
        require(all(digest(OUTPUT / n) == h for n, h in hashes.items()), "Output hash completeness")
        require(git("rev-parse", "HEAD") == head and git("status", "--porcelain", "--untracked-files=all") == ""
                and all(digest(ROOT / n) == h for n, h in sources.items()), "Unchanged HEAD/sources/failure evidence")
        require(all(git("rev-parse", f"{head}:{n}") == git("hash-object", f"--path={n}", n)
                    == git("rev-parse", f":{n}") == b for n, b in blobs.items()), "Publication source integrity")
        result = {"stage": "P2-1E-2-v1", "stage_pass": True, "stage_pass_meaning": "AUDIT EXECUTED SUCCESSFULLY",
                  "support_closure_pass": False, "scientific_status": STATUS, "declaration": DECLARATION,
                  "HEAD": head, "failed_gates": [], "witness_reproduction_pass": True,
                  "formal_trajectory_access_count": 0, "formal_perception_seed_access_count": 0,
                  "stochastic_perception_draws": 0, "output_sha256": {**hashes, "output_hashes.json": digest(OUTPUT / "output_hashes.json")}}
        write_json("audit_summary.json", result)
        return result
    except Exception as exc:
        # Diagnostics only; original outputs/models are never touched or rerun.
        if not (OUTPUT / "audit_summary.json").exists():
            write_json("diagnostic_failure.json", {"stage_pass": False, "support_closure_pass": None,
                "scientific_status": INCONSISTENT if INCONSISTENT in str(exc) else "SUPPORT_CLOSURE_AUDIT_EXECUTION_FAIL",
                "error": str(exc), "action": "STOP; NO REPAIR OR RETRY", "HEAD": head})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
