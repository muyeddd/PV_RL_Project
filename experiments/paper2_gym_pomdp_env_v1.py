"""Single scientific episode Gymnasium environment; no import-time experiments.

All downstream baselines and training harnesses must import this implementation.
Episode scheduling belongs to the caller. Gym seeds never choose scientific
episodes. Private audit accessors are outside the policy information contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
REWARD_DIRECTORY = BASE / "p2_1d_3_final_reward_contract_audit_v1/formal"
CHECKPOINT = "25fca93a7187d235564ae76c2a998af7992a68c7"
REWARD_SCRIPT = "experiments/run_paper2_stage1d3_final_reward_contract_audit_v1.py"
REWARD_BLOB = "04c72b41dec121c93db306193cac6af1485f29ca"
REWARD_HASHES = {
    "audit_summary.json": "554ad015113e4a50d213394b4bb050c32e05315f21b88d2e5b9a62726d255753",
    "reward_contract_manifest.json": "6eeb18ada84874b55ef245ba217ef84864d123d97ba77d18d8a72218b3897fe9",
    "protocol_manifest.json": "5085693a023f39f9fc09de580b0c6da38e448e04f24c333694df80c2f2e06e4c",
    "daily_reward_trace.csv": "de0d78496a94c507e78080ac5c5748ed40b467b6e1079da0514814f1cfe81f01",
    "trajectory_reward_regression.csv": "3749203c846e7ce18150a56a86c2e85af58db17989bf49139084bcae339f8fcb",
    "reward_distribution_summary.csv": "fbf369399fca1555f2050fd8a228a1e17bc0764d591e3062b5a3b5d4eafe3847",
}
HELPERS = {
    "mechanics": ("experiments/run_paper2_stage1c2a_clean_mechanics_operational_support_audit_v1.py",
                  "3be8472e8963e0ee97b6173d54916a92dbc05277"),
    "natural": ("experiments/run_paper2_stage1c1r_b_rain_repair_candidate_validation_v1.py",
                "c1430b7b4ccac776433294489984b07af0c82c5f"),
    "perception": ("experiments/run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1.py",
                   "6f940272ce1a262f325af310c9e72ed0adc401eb"),
}
ASSET_HASHES = {
    "ledger": (BASE / "p2_1a_timeline_intervention_audit_v1/environment_master_ledger.csv",
               "7906680af03d94f4989efc16787d9f5431bfb53b87bd2eb0f8586c6a340fde04"),
    "transitions": (BASE / "p2_1a_timeline_intervention_audit_v1/transition_audit.csv",
                    "195a23f1bb893b4b92c8f567e17824007c351886d2cadd1600cd5f3022d7d75d"),
    "energy": (BASE / "p2_1d_1a_daily_ghi_energy_value_audit_v1/daily_energy_value_proxy.csv",
               "443a023c42e9b0db9cf888c120eba53b5ad092779ea0875cd6de32826414c155"),
}
DEFAULT_PAPER1_CQR = Path(r"E:\PV_RL_Project\outputs\paper1_clean_random_v1\cqr_stage3a2_intervals_v1\cqr_predictions.csv")
DEFAULT_WAPP_BRIDGE = BASE / "p2_0c_1b_common_temp_power_bridge_v1/daily_common_temp_power_bridge.csv"
OBSERVATION_MODES = ("True-State", "Point", "UA")
ETA = 0.95
LAMBDA_MAIN = 0.19925428989934618
REWARD_SCALE = 1.0


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    """Read-only provenance; called only on explicit asset construction/audit."""
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def script_record(path, head, expected=None):
    committed = git("rev-parse", "--verify", f"{head}:{path}")
    working = git("hash-object", f"--path={path}", path)
    require(committed == working and (expected is None or committed == expected),
            f"Required committed unchanged script blob: {path}")
    if expected is not None:
        require(git("rev-parse", "--verify", f"{CHECKPOINT}:{path}") == expected,
                f"Frozen checkpoint helper blob: {path}")
    return {"path": path, "git_blob": committed, "sha256": sha((ROOT / path).read_bytes())}


def load_frozen_module(name, path):
    spec = importlib.util.spec_from_file_location(f"_paper2_env_frozen_{name}", ROOT / path)
    require(spec is not None and spec.loader is not None, f"Module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def readonly(values):
    """Immutable bytes-backed arrays also prevent re-enabling write flags."""
    array = np.ascontiguousarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class EpisodeSpec:
    year: str
    environment_root: int
    trajectory_id: int
    population_size: int
    perception_seed: int | None = None

    def __post_init__(self):
        require(self.year in ("YEAR1", "YEAR2"), "Unknown scientific year")
        for name in ("environment_root", "trajectory_id", "population_size"):
            value = getattr(self, name)
            require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                    and value >= 0, f"Explicit nonnegative integer required: {name}")
        require(self.population_size > 0 and self.trajectory_id < self.population_size,
                "trajectory_id must identify a column of the full population")
        if self.perception_seed is not None:
            require(isinstance(self.perception_seed, (int, np.integer))
                    and not isinstance(self.perception_seed, (bool, np.bool_)) and self.perception_seed >= 0,
                    "Explicit nonnegative integer perception seed required")


class PerceptionSupportError(RuntimeError):
    def __init__(self, context, cause):
        self.context = dict(context)
        super().__init__(f"Frozen perception support failure: {json.dumps(context, sort_keys=True)}; {cause}")


class Paper2EnvAssets:
    """Load and verify once; lazily construct perception only for Point/UA.

    The frozen fit_r3 helper reconstructs the pinned parameters from verified
    inputs; it does not select or introduce a dynamics model. Full environment
    banks are cached by year/root/population, independently of information mode.
    """

    def __init__(self, *, paper1_cqr=DEFAULT_PAPER1_CQR, wapp_power_bridge=DEFAULT_WAPP_BRIDGE):
        self.paper1_cqr = Path(paper1_cqr)
        self.wapp_power_bridge_path = Path(wapp_power_bridge)
        head = git("rev-parse", "HEAD")
        require(git("rev-parse", "--verify", f"{CHECKPOINT}^{{commit}}") == CHECKPOINT, "Reward checkpoint exists")
        git("merge-base", "--is-ancestor", CHECKPOINT, head)
        self.provenance = {"HEAD": head, "checkpoint": CHECKPOINT, "checkpoint_ancestor": True,
                           "reward_script": script_record(REWARD_SCRIPT, head, REWARD_BLOB)}
        self.reward_payloads = {}
        for name, digest in REWARD_HASHES.items():
            payload = (REWARD_DIRECTORY / name).read_bytes()
            require(sha(payload) == digest, f"Frozen reward artifact hash: {name}")
            self.reward_payloads[name] = payload
        self.provenance["reward_hashes"] = dict(REWARD_HASHES)
        audit = json.loads(self.reward_payloads["audit_summary.json"])
        self.reward_contract = json.loads(self.reward_payloads["reward_contract_manifest.json"])
        contract = self.reward_contract
        require(audit.get("stage_pass") is True and audit.get("failed_gates") == []
                and audit.get("scientific_status") == "REWARD_CONTRACT_PASS_FROZEN"
                and audit.get("reward_contract_status") == "PASS_FROZEN", "Frozen reward audit PASS")
        require(contract.get("scientific_status") == "REWARD_CONTRACT_PASS_FROZEN"
                and contract.get("reward_contract_status") == "PASS_FROZEN"
                and contract.get("lambda_c_main") == LAMBDA_MAIN and contract.get("eta") == ETA
                and contract.get("reward_scale") == REWARD_SCALE
                and contract.get("declaration") == "NO FURTHER REWARD TUNING AFTER THIS STAGE",
                "Frozen reward contract exact")
        self.lambda_main, self.eta, self.reward_scale = contract["lambda_c_main"], contract["eta"], contract["reward_scale"]
        self.reward_protocol = json.loads(self.reward_payloads["protocol_manifest.json"])
        self.provenance["helpers"] = {name: script_record(path, head, blob)
                                      for name, (path, blob) in HELPERS.items()}
        modules = {name: load_frozen_module(name, path) for name, (path, _) in HELPERS.items()}
        self.mechanics, self.natural, self.perception = (modules[name] for name in ("mechanics", "natural", "perception"))
        self.reward_helper = load_frozen_module("reward", REWARD_SCRIPT)
        snapshots = {}
        for name, (path, digest) in ASSET_HASHES.items():
            snapshots[name] = path.read_bytes()
            require(sha(snapshots[name]) == digest, f"Frozen timeline/energy hash: {name}")
        self.provenance["assets"] = {name: {"path": str(path), "sha256": digest}
                                     for name, (path, digest) in ASSET_HASHES.items()}
        self.ledger = self.mechanics.load_ledger(io.BytesIO(snapshots["ledger"]))
        self.transitions = self.mechanics.load_transitions(io.BytesIO(snapshots["transitions"]))
        self.ledger["bridge_valid"] = self.mechanics.parse_bool_series(self.ledger.bridge_valid, "bridge_valid")
        energy = pd.read_csv(io.BytesIO(snapshots["energy"]), float_precision="round_trip", encoding="utf-8-sig")
        energy["date"] = pd.to_datetime(energy.date, errors="raise")
        for frame in (energy, self.ledger):
            require(len(frame) == 730 and not frame.date.duplicated().any()
                    and pd.DatetimeIndex(frame.date).equals(pd.date_range("2021-08-09", "2023-08-08")),
                    "Exact ordered 730-day calendar")
        self.energy = energy.set_index("date").E_clean_GHI.copy()
        require(np.isfinite(self.energy).all() and self.energy.gt(0).all()
                and np.isclose(self.energy.mean(), 1, atol=1e-12, rtol=0), "Frozen normalized positive energy")
        require(len(self.transitions) == 729
                and np.array_equal(self.transitions.transition_index, np.arange(729))
                and np.array_equal(self.transitions.source_date, self.ledger.date.iloc[:-1])
                and np.array_equal(self.transitions.dest_date, self.ledger.date.iloc[1:]), "Transition alignment")
        context = self.mechanics.attach_context(self.transitions, self.ledger)
        self.dry = readonly(context.loc[context.transition_class.eq("DRY_NATURAL"), "delta_L_power_proxy"].to_numpy(float))
        rain_rows = context.loc[context.transition_class.eq("RAIN_AFFECTED")]
        self.r3 = self.mechanics.fit_r3(rain_rows)
        self.r3["residuals"] = readonly(self.r3["residuals"])
        values = [self.r3["intercept"], self.r3["slope"], self.r3["source_max"],
                  np.std(self.r3["residuals"], ddof=1)]
        expected = [0.00090811689781017, -0.48383380267297343, 0.0490334865821413, 0.004494874318643408]
        require(len(self.dry) == 494 and len(rain_rows) == 201 and np.isfinite(self.dry).all()
                and np.isfinite(self.r3["residuals"]).all()
                and np.allclose(values, expected, atol=1e-15, rtol=0), "Frozen D0/R3 exact regression")
        parameters = self.reward_protocol["environment_parameters"]
        require(parameters["dry"] == "D0_GLOBAL" and parameters["rain"] == "R3_OLS_RESIDUAL"
                and sha(self.dry.tobytes()) == parameters["dry_pool_sha256"]
                and sha(self.r3["residuals"].tobytes()) == parameters["residual_pool_sha256"], "Frozen innovation pools")
        self.provenance["natural_regression"] = {"dry_rows": len(self.dry), "rain_rows": len(rain_rows),
                                                "actual": list(map(float, values)), "atol": 1e-15, "rtol": 0}
        self._banks = {}
        self.bank_generation_count = 0
        self.emulator = None
        self.perception_source = None
        self.wapp_power_bridge = None
        self.perception_construction_count = 0
        self.perception_sample_count = 0
        self._perception_cache = {}

    def ensure_perception(self):
        if self.emulator is not None:
            return
        frozen = self.perception
        records = {}
        for name, path in (("paper1_cqr", self.paper1_cqr), ("wapp_power_bridge", self.wapp_power_bridge_path)):
            digest = frozen.sha256_file(path)
            require(digest == frozen.EXPECTED_HASHES[name], f"Frozen external perception source: {name}")
            records[name] = {"path": str(path), "sha256": digest}
        source = frozen.load_paper1_cqr(self.paper1_cqr)
        bridge = frozen.load_wapp_power_bridge(self.wapp_power_bridge_path)
        # Detect source changes across path-based frozen loaders.
        for name, path in (("paper1_cqr", self.paper1_cqr), ("wapp_power_bridge", self.wapp_power_bridge_path)):
            require(frozen.sha256_file(path) == records[name]["sha256"], f"Perception source changed: {name}")
        self.perception_source, self.wapp_power_bridge = source, bridge
        self.emulator = frozen.OnlineBlock10Emulator(source)
        self.perception_construction_count += 1
        self.provenance["perception_sources"] = records

    def episode_bank(self, episode):
        development_roots = self.reward_protocol["environment_roots"]
        is_development = episode.environment_root in development_roots.values()
        if is_development:
            require(episode.environment_root == development_roots[episode.year]
                    and episode.population_size == 300, "Frozen development year/root and 300-column bank required")
        key = (episode.year, int(episode.environment_root), int(episode.population_size))
        if key not in self._banks:
            calendar = self.mechanics.get_year_calendar(self.ledger, episode.year)
            expected_dates = pd.date_range("2021-08-09" if episode.year == "YEAR1" else "2022-08-09", periods=365)
            require(pd.DatetimeIndex(calendar.date).equals(expected_dates), "Year calendar")
            initial = float(calendar.L_power_proxy.iloc[0])
            require(bool(calendar.state_valid.iloc[0]) and np.isfinite(initial) and 0 <= initial <= 1,
                    "Historical initial latent state")
            bank, rain, report = self.mechanics.build_crn_indices(
                calendar, len(self.dry), len(self.r3["residuals"]), episode.population_size, episode.environment_root)
            full = readonly(np.stack(bank))
            require(full.shape == (364, episode.population_size) and report["same_seed_exact_reproduction"], "Full CRN bank")
            if is_development:
                frozen = self.reward_protocol["CRN"][episode.year]
                require(report == frozen["full_bank"] and sha(full.tobytes()) == frozen["full_bank_sha256"]
                        and initial == frozen["initial_L"], "Frozen development CRN and initialization exact")
            self._banks[key] = (calendar.copy(), initial, full, readonly(rain), dict(report))
            self.bank_generation_count += 1
        return self._banks[key]

    def perception_record(self, episode, day, latent, date):
        require(self.emulator is not None, "Point/UA must construct the verified emulator")
        # Shared immutable content: Point/UA never draw independent records.
        key = (int(episode.perception_seed), episode.year, int(episode.trajectory_id), int(day), float(latent))
        if key not in self._perception_cache:
            seed = self.perception.perception_substream_seed(
                episode.perception_seed, episode.year, episode.trajectory_id, day)
            self.perception_sample_count += 1
            try:
                rec = self.emulator.sample_one(float(latent), np.random.default_rng(seed))
            except RuntimeError as exc:
                if "Unsupported query" not in str(exc):
                    raise
                context = {"year": episode.year, "trajectory_id": int(episode.trajectory_id),
                           "day_index": int(day), "date": date, "L_true": float(latent),
                           "perception_seed": int(episode.perception_seed), "local_rows": None}
                # Read-only diagnostics from the frozen helper; never resample.
                try:
                    cand = self.emulator._candidate_indices(float(latent))
                    context.update(local_rows=int(len(cand)), local_dates=int(len(np.unique(self.emulator.date[cand]))),
                                   candidate_blocks=int(len(np.unique(self.emulator.block[cand]))))
                except Exception as diagnostic_error:
                    context["diagnostic_error"] = str(diagnostic_error)
                raise PerceptionSupportError(context, str(exc)) from exc
            q50, width = float(rec["q50"]), float(rec["width"])
            require(np.isfinite([q50, width]).all() and 0 <= q50 <= 1 and 0 <= width <= 1
                    and rec["fallback_used"] is False, "Frozen perception finite bounded/no fallback")
            self._perception_cache[key] = (q50, width)
        return self._perception_cache[key]


class Paper2CleaningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, assets: Paper2EnvAssets, episode_spec: EpisodeSpec, observation_mode="True-State"):
        super().__init__()
        require(observation_mode in OBSERVATION_MODES, "Unknown observation mode")
        require(isinstance(episode_spec, EpisodeSpec), "Fixed EpisodeSpec required")
        self._assets, self._episode_spec, self._observation_mode = assets, episode_spec, observation_mode
        if observation_mode != "True-State":
            require(episode_spec.perception_seed is not None, "Point/UA require explicit perception_seed")
            assets.ensure_perception()
        calendar, initial, full, rain, report = assets.episode_bank(episode_spec)
        self._calendar = calendar.copy()
        self._initial = initial
        self._indices = readonly(full[:, episode_spec.trajectory_id])
        self._rain = rain
        self._full_bank_hash = sha(full.tobytes())
        self._bank_report = dict(report)
        self._energy = readonly(assets.energy.loc[calendar.date].to_numpy(float))
        self.action_space = gym.spaces.Discrete(2)
        low = [0, 0, -1, -1] if observation_mode == "UA" else [0, -1, -1]
        high = [1, 1, 1, 1] if observation_mode == "UA" else [1, 1, 1]
        self.observation_space = gym.spaces.Box(np.array(low, dtype=np.float32), np.array(high, dtype=np.float32), dtype=np.float32)
        self._needs_reset = True
        self._terminated = False
        self._day = 0
        self._latent = float(initial)
        self._last = None
        self._transition_count = self._clean_count = self._perception_query_count = 0
        self._episode_return = 0.0

    @property
    def episode_spec(self):
        return self._episode_spec

    @property
    def observation_mode(self):
        return self._observation_mode

    def _observation(self, day, latent):
        date = self._calendar.date.iloc[day]
        phase = 2 * np.pi * (date.dayofyear - 1) / 365
        season = [np.sin(phase), np.cos(phase)]
        if self.observation_mode == "True-State":
            content = [float(latent)]
        else:
            self._perception_query_count += 1
            q50, width = self._assets.perception_record(self.episode_spec, day, latent, date.strftime("%Y-%m-%d"))
            content = [q50, width] if self.observation_mode == "UA" else [q50]
        observation = np.asarray(content + season, dtype=np.float32)
        require(np.isfinite(observation).all() and self.observation_space.contains(observation), "Observation contract")
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        require(options is None or options == {}, "EpisodeSpec is fixed; reset options cannot schedule episodes")
        self._needs_reset = True
        self._day, self._latent, self._terminated = 0, float(self._initial), False
        self._last = None
        self._transition_count = self._clean_count = self._perception_query_count = 0
        self._episode_return = 0.0
        obs = self._observation(0, self._latent)
        self._needs_reset = False
        return obs, self._info(0)

    def _info(self, day, action=None, terminal=False):
        info = {"date": self._calendar.date.iloc[day].strftime("%Y-%m-%d"), "day_index": int(day),
                "observation_mode": self.observation_mode, "terminal_observation_sentinel": bool(terminal)}
        if action is not None:
            info["action_name"] = "CLEAN" if action else "WAIT"
        if terminal:
            info.update(terminated_reason="365_DAY_HORIZON", episode_return=float(self._episode_return),
                        clean_count=int(self._clean_count))
        return info

    def step(self, action):
        # Reject invalid actions before touching any scientific state or counter.
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)) or int(action) not in (0, 1):
            raise ValueError("Action must be integer 0 (WAIT) or 1 (CLEAN)")
        if self._needs_reset or self._terminated:
            raise RuntimeError("Episode unavailable or terminated; call reset()")
        action = int(action)
        day, pre = self._day, float(self._latent)
        post = (1 - self._assets.eta) * pre if action else pre
        soiling, cleaning, total, raw, final = self._assets.reward_helper.settlement(
            float(self._energy[day]), post, action, self._assets.lambda_main)
        require(np.isfinite([pre, post, soiling, cleaning, total, final]).all(), "Finite reward algebra")
        terminal = day == 364
        following = None
        # If a genuine perception failure occurs, require explicit reset. There
        # is no fallback observation and no silently skipped settlement/day.
        self._needs_reset = True
        if not terminal:
            idx = int(self._indices[day])
            if self._rain[day]:
                r3 = self._assets.r3
                model = {"candidate": "R3_OLS_RESIDUAL", "intercept": r3["intercept"], "slope": r3["slope"],
                         "residuals": r3["residuals"][idx:idx + 1]}
                following = float(self._assets.natural.rain_next_samples(post, model)[0])
            else:
                following = float(self._assets.natural.dry_next_samples(post, self._assets.dry[idx:idx + 1])[0])
            require(np.isfinite(following) and 0 <= following <= 1, "Frozen physical projection")
            obs = self._observation(day + 1, following)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._last = {"year": self.episode_spec.year, "trajectory_id": int(self.episode_spec.trajectory_id),
                      "day_index": day, "date": self._calendar.date.iloc[day].strftime("%Y-%m-%d"),
                      "L_pre": pre, "action": action, "L_post": float(post), "E_clean_GHI": float(self._energy[day]),
                      "soiling_cost": float(soiling), "cleaning_cost": float(cleaning), "total_cost": float(total),
                      "raw_reward": float(raw), "reward_scale": self._assets.reward_scale, "final_reward": float(final),
                      "transition_executed": not terminal, "rain_affected_transition": bool(self._rain[day]) if not terminal else False,
                      "L_next": following}
        self._episode_return += float(final)
        self._clean_count += action
        if not terminal:
            self._day, self._latent = day + 1, following
            self._transition_count += 1
        self._terminated = terminal
        self._needs_reset = False
        return obs, float(final), bool(terminal), False, self._info(day, action, terminal)

    def _audit_snapshot(self):
        """Private debug channel; NEVER merge this into observation or info."""
        return {"year": self.episode_spec.year, "trajectory_id": int(self.episode_spec.trajectory_id),
                "day_index": int(self._day), "date": self._calendar.date.iloc[self._day].strftime("%Y-%m-%d"),
                "L_pre": float(self._latent), "L_post_last": None if self._last is None else self._last["L_post"],
                "environment_root": int(self.episode_spec.environment_root),
                "perception_seed": None if self.episode_spec.perception_seed is None else int(self.episode_spec.perception_seed),
                "last_reward_components": None if self._last is None else dict(self._last),
                "transition_count": int(self._transition_count), "clean_count": int(self._clean_count),
                "episode_return": float(self._episode_return), "terminated": self._terminated,
                "needs_reset": self._needs_reset, "perception_query_count": self._perception_query_count,
                "indices_sha256": sha(self._indices.tobytes()), "full_bank_sha256": self._full_bank_hash,
                "indices_read_only": not self._indices.flags.writeable}
