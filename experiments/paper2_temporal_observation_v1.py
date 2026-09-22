
"""Compact temporal observation wrapper for Paper2 masked Point/UA environments.

This module introduces NO new dynamics, reward, safety rule, perception model,
training rule, RNG, or data access. It only transforms already-public Point/UA
observations plus executed action history into compact temporal features.

Frozen candidate observations
-----------------------------
History-Point:
    [q50_t, delta_q50_t, tau_t, sin_DOY_t, cos_DOY_t]

History-UA:
    [q50_t, delta_q50_t, width_t, delta_width_t,
     tau_t, sin_DOY_t, cos_DOY_t]

where:
    delta_q50_0 = 0
    delta_width_0 = 0
    tau_0 = 0

and, for nonterminal transition t -> t+1:
    delta_q50_{t+1} = q50_{t+1} - q50_t
    delta_width_{t+1} = width_{t+1} - width_t
    age_days_{t+1} = 1 if executed_action_t == CLEAN
                     age_days_t + 1 otherwise
    tau_{t+1} = age_days_{t+1} / 364

tau is deliberately "since last in-episode CLEAN or episode start". Before the
first in-episode CLEAN it measures age since episode start; unknown pre-episode
cleaning age is not imputed.

Only information available by the current decision time is used. The wrapper
never reads latent L, future weather, future rain, reward components,
transition innovations, or private environment audit state to construct policy
features.

The underlying q50-history action mask and frozen q50 shield remain unchanged.
Width, delta-width and tau are policy inputs only; they never enter safety.

Import is inert.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

import paper2_masked_perception_ppo_runner_v1_1 as masked


require = masked.require
get_protocol = masked.get_protocol

HISTORY_POINT = "History-Point"
HISTORY_UA = "History-UA"
TEMPORAL_MODES = (HISTORY_POINT, HISTORY_UA)

FEATURES = {
    HISTORY_POINT: (
        "q50",
        "delta_q50",
        "tau_since_last_clean_or_episode_start",
        "sin_DOY",
        "cos_DOY",
    ),
    HISTORY_UA: (
        "q50",
        "delta_q50",
        "width",
        "delta_width",
        "tau_since_last_clean_or_episode_start",
        "sin_DOY",
        "cos_DOY",
    ),
}

BASE_MODE = {
    HISTORY_POINT: "Point",
    HISTORY_UA: "UA",
}

OBSERVATION_DIMENSION = {
    HISTORY_POINT: 5,
    HISTORY_UA: 7,
}

TAU_DENOMINATOR = 364.0


def temporal_contract():
    return {
        "modes": {
            HISTORY_POINT: {
                "base_mode": BASE_MODE[HISTORY_POINT],
                "features": list(FEATURES[HISTORY_POINT]),
                "dimension": OBSERVATION_DIMENSION[HISTORY_POINT],
            },
            HISTORY_UA: {
                "base_mode": BASE_MODE[HISTORY_UA],
                "features": list(FEATURES[HISTORY_UA]),
                "dimension": OBSERVATION_DIMENSION[HISTORY_UA],
            },
        },
        "day0": {
            "delta_q50": 0.0,
            "delta_width": 0.0,
            "tau_since_last_clean_or_episode_start": 0.0,
        },
        "delta_semantics":
            "current public observation minus immediately previous public observation",
        "tau_semantics":
            "days since most recent executed in-episode CLEAN; before first CLEAN, days since episode start; divided by 364",
        "post_clean_next_day_tau": 1.0 / TAU_DENOMINATOR,
        "pre_episode_clean_age_imputed": False,
        "history_window_hyperparameter": None,
        "future_information_used": False,
        "latent_state_used": False,
        "safety_rule_changed": False,
    }


class TemporalMaskedObservationWrapper(gym.Wrapper):
    metadata = {"render_modes": []}

    def __init__(self, env, temporal_mode):
        require(
            temporal_mode in TEMPORAL_MODES,
            "Known compact temporal observation mode required",
        )
        require(
            isinstance(env, masked.Q50MaskedShieldedPerceptionEnv),
            "Frozen masked Point/UA environment required",
        )
        base_mode = BASE_MODE[temporal_mode]
        require(
            env.observation_mode == base_mode,
            "Temporal mode must match frozen base perception mode",
        )
        require(
            get_protocol().natural_transitions == int(TAU_DENOMINATOR),
            "Tau denominator must equal frozen 364-transition horizon",
        )
        super().__init__(env)

        self.temporal_mode = temporal_mode
        self.base_observation_mode = base_mode
        self.action_space = env.action_space

        if temporal_mode == HISTORY_POINT:
            low = np.asarray([0.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)
            high = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        else:
            low = np.asarray(
                [0.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0],
                dtype=np.float32,
            )
            high = np.asarray(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

        self._previous_public_observation = None
        self._age_days = None
        self._temporal_decision_index = None
        self._last_temporal_observation = None

    @property
    def observation_mode(self):
        return self.temporal_mode

    @property
    def last_shield_decision(self):
        return self.env.last_shield_decision

    def action_masks(self):
        return self.env.action_masks()

    def _build_temporal_observation(
        self,
        public_observation,
        *,
        delta_q50,
        delta_width,
        age_days,
    ):
        obs = np.asarray(public_observation, dtype=np.float32)
        require(
            obs.shape == self.env.observation_space.shape
            and np.isfinite(obs).all(),
            "Finite frozen public Point/UA observation required",
        )

        q50 = np.float32(obs[0])
        tau = np.float32(age_days / TAU_DENOMINATOR)

        if self.temporal_mode == HISTORY_POINT:
            sin_doy = np.float32(obs[1])
            cos_doy = np.float32(obs[2])
            values = np.asarray(
                [q50, np.float32(delta_q50), tau, sin_doy, cos_doy],
                dtype=np.float32,
            )
        else:
            width = np.float32(obs[1])
            sin_doy = np.float32(obs[2])
            cos_doy = np.float32(obs[3])
            values = np.asarray(
                [
                    q50,
                    np.float32(delta_q50),
                    width,
                    np.float32(delta_width),
                    tau,
                    sin_doy,
                    cos_doy,
                ],
                dtype=np.float32,
            )

        require(
            values.shape == self.observation_space.shape
            and values.dtype == self.observation_space.dtype
            and np.isfinite(values).all()
            and self.observation_space.contains(values),
            "Compact temporal observation contract",
        )
        return values

    def reset(self, *, seed=None, options=None):
        public_obs, info = self.env.reset(seed=seed, options=options)
        require(
            not bool(info.get("terminal_observation_sentinel", False)),
            "Reset must expose a live public observation",
        )
        self._previous_public_observation = np.asarray(
            public_obs,
            dtype=np.float32,
        ).copy()
        self._age_days = 0
        self._temporal_decision_index = 0

        temporal = self._build_temporal_observation(
            public_obs,
            delta_q50=np.float32(0.0),
            delta_width=np.float32(0.0),
            age_days=0,
        )
        self._last_temporal_observation = temporal.copy()
        return temporal, info

    def step(self, action):
        require(
            self._previous_public_observation is not None
            and self._age_days is not None
            and self._temporal_decision_index is not None,
            "Temporal wrapper must be reset before step",
        )

        public_obs, reward, terminated, truncated, info = self.env.step(action)
        decision = self.env.last_shield_decision
        require(
            decision is not None
            and decision["intervention"] is False
            and decision["executed_action"] == int(action),
            "Temporal features require the frozen masked zero-intervention path",
        )

        if terminated:
            temporal = np.zeros(
                self.observation_space.shape,
                dtype=self.observation_space.dtype,
            )
            self._last_temporal_observation = temporal.copy()
            self._temporal_decision_index += 1
            return temporal, reward, terminated, truncated, info

        current = np.asarray(public_obs, dtype=np.float32)
        previous = self._previous_public_observation

        delta_q50 = np.float32(current[0] - previous[0])
        if self.temporal_mode == HISTORY_UA:
            delta_width = np.float32(current[1] - previous[1])
        else:
            delta_width = np.float32(0.0)

        executed = int(decision["executed_action"])
        if executed == 1:
            age_days = 1
        else:
            age_days = int(self._age_days) + 1

        require(
            1 <= age_days <= get_protocol().natural_transitions,
            "Known in-episode action-history age bound",
        )
        temporal = self._build_temporal_observation(
            current,
            delta_q50=delta_q50,
            delta_width=delta_width,
            age_days=age_days,
        )

        self._previous_public_observation = current.copy()
        self._age_days = age_days
        self._temporal_decision_index += 1
        self._last_temporal_observation = temporal.copy()
        return temporal, reward, terminated, truncated, info

    def _temporal_audit_snapshot(self):
        return {
            "temporal_mode": self.temporal_mode,
            "base_observation_mode": self.base_observation_mode,
            "feature_names": list(FEATURES[self.temporal_mode]),
            "dimension": OBSERVATION_DIMENSION[self.temporal_mode],
            "decision_index": self._temporal_decision_index,
            "age_days_since_last_clean_or_episode_start": self._age_days,
            "tau_denominator": TAU_DENOMINATOR,
            "previous_public_observation":
                None
                if self._previous_public_observation is None
                else self._previous_public_observation.tolist(),
            "last_temporal_observation":
                None
                if self._last_temporal_observation is None
                else self._last_temporal_observation.tolist(),
        }

    @property
    def schedule(self):
        return getattr(self.env, "schedule")

    @property
    def ledger(self):
        return getattr(self.env, "ledger")

    @property
    def assignments(self):
        return getattr(self.env, "assignments")

    @property
    def started(self):
        return getattr(self.env, "started")

    @property
    def done(self):
        return getattr(self.env, "done")

    @property
    def failed(self):
        return getattr(self.env, "failed")

    @property
    def total_steps(self):
        return getattr(self.env, "total_steps")

    @property
    def observations_checked(self):
        return getattr(self.env, "observations_checked")

    @property
    def episodes_completed(self):
        return getattr(self.env, "episodes_completed")

    @property
    def perception_query_count(self):
        return getattr(self.env, "perception_query_count")

    def _mask_audit_snapshot(self):
        return self.env._mask_audit_snapshot()


def wrap_temporal_masked_env(env, temporal_mode):
    return TemporalMaskedObservationWrapper(env, temporal_mode)
