
"""Q50-only set-membership predictive shield for Paper2.

This module is a reusable CONTROL-SIDE component. It does not load project
assets, trajectories, outputs or PPO at import.

Scientific design
-----------------
1. The safety domain [0, K] is supplied by the already-audited P2-1F-0 robust
   support-invariant kernel.
2. The current hidden L is NEVER an input.
3. The UA-only interval width is NEVER an input. Point and UA therefore use the
   same shield information.
4. An observed float32 q50 is inverted through the FROZEN empirical perception
   generator. A source row is possible only over the exact binary64 query range
   in which that row belongs to the frozen radius candidate set. The float32
   rounding bin is conservatively inverted, including clipping at 0/1.
5. The observer carries a conservative latent interval hull through time using
   BOTH frozen dry and rain disturbance supports; no future weather/rain flag is
   used.
6. WAIT is passed only if every next state represented by the observer remains
   in [0, K]. Otherwise CLEAN is used if robustly safe. If neither action is
   certified, the shield fails closed.

The interval hull may include latent states that are not jointly reachable. This
can make the shield conservative, never optimistic. Feasibility/conservatism are
audited separately before PPO.

Import is inert.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def float_bits(value):
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def bits_float(value):
    return struct.unpack(">d", struct.pack(">Q", int(value)))[0]


def first_true_float(predicate):
    """Exact binary64 lower_bound over nonnegative [0,1]."""
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


def float32_rounding_bin(observed):
    """Conservative real interval whose values may cast to observed float32.

    Tie points are included on both sides deliberately. This is a safety-side
    over-approximation, not an attempt to reproduce IEEE tie ownership.
    """
    q = np.float32(observed)
    require(np.isfinite(q) and np.float32(0.0) <= q <= np.float32(1.0),
            "Observed q50 must be finite float32 in [0,1]")
    prev_q = np.nextafter(q, np.float32(-np.inf))
    next_q = np.nextafter(q, np.float32(np.inf))
    lower = 0.0 if q == np.float32(0.0) else (float(prev_q) + float(q)) / 2.0
    upper = 1.0 if q == np.float32(1.0) else (float(q) + float(next_q)) / 2.0
    lower = max(0.0, math.nextafter(lower, -math.inf))
    upper = min(1.0, math.nextafter(upper, math.inf))
    require(0.0 <= lower <= float(q) <= upper <= 1.0, "Float32 rounding bin")
    return lower, upper, float(q)


@dataclass(frozen=True)
class ObservationBound:
    lower: float
    upper: float
    compatible_source_rows: int
    q50_float32: float
    rounding_bin_lower: float
    rounding_bin_upper: float


class Q50ObservationInverse:
    """Conservative inverse of the frozen q50 transport on [0, kernel_upper]."""

    def __init__(self, emulator, local_radius, kernel_upper):
        self.kernel_upper = float(kernel_upper)
        self.local_radius = float(local_radius)
        require(
            0.0 < self.local_radius < 1.0
            and 0.0 < self.kernel_upper < 1.0,
            "Valid inverse geometry constants",
        )
        self.source_L = np.asarray(emulator.L, dtype=float)
        self.dq50 = np.asarray(emulator.dq50, dtype=float)
        require(
            self.source_L.ndim == self.dq50.ndim == 1
            and len(self.source_L) == len(self.dq50) > 0
            and np.isfinite(self.source_L).all()
            and np.isfinite(self.dq50).all(),
            "Finite frozen source arrays",
        )

        left = np.empty(len(self.source_L), dtype=float)
        right = np.empty(len(self.source_L), dtype=float)
        active = np.zeros(len(self.source_L), dtype=bool)

        for i, source_value in enumerate(self.source_L):
            v = float(source_value)
            entry = first_true_float(lambda q, v=v: q + self.local_radius >= v)
            require(entry is not None, "Every source row can enter by q=1")
            exit_point = first_true_float(
                lambda q, v=v: q - self.local_radius > v
            )
            candidate_left = max(0.0, float(entry))
            if exit_point is None:
                candidate_right = self.kernel_upper
            else:
                candidate_right = min(
                    self.kernel_upper,
                    math.nextafter(float(exit_point), -math.inf),
                )
            left[i], right[i] = candidate_left, candidate_right
            active[i] = (
                candidate_left <= self.kernel_upper
                and candidate_left <= candidate_right
            )

        require(active.any(), "At least one frozen source row intersects kernel")
        self.candidate_left = left
        self.candidate_right = right
        self.active = active

    def bounds(self, q50_float32):
        bin_lower, bin_upper, q = float32_rounding_bin(q50_float32)

        # y = clip(L + dq50, 0, 1), then policy sees float32(y).
        # If observed q==0, clipping means raw values below zero are possible:
        # there is no finite lower raw bound. Likewise q==1 has no upper raw bound.
        if q == 0.0:
            residual_left = np.zeros(len(self.dq50), dtype=float)
        else:
            residual_left = np.full(len(self.dq50), bin_lower) - self.dq50
            residual_left = np.nextafter(residual_left, -np.inf)

        if q == 1.0:
            residual_right = np.full(len(self.dq50), self.kernel_upper, dtype=float)
        else:
            residual_right = np.full(len(self.dq50), bin_upper) - self.dq50
            residual_right = np.nextafter(residual_right, np.inf)

        lo = np.maximum.reduce(
            [
                self.candidate_left,
                residual_left,
                np.zeros(len(self.dq50), dtype=float),
            ]
        )
        hi = np.minimum.reduce(
            [
                self.candidate_right,
                residual_right,
                np.full(len(self.dq50), self.kernel_upper, dtype=float),
            ]
        )
        valid = self.active & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
        require(valid.any(), "Observed q50 has no model-consistent latent state")
        return ObservationBound(
            lower=float(np.min(lo[valid])),
            upper=float(np.max(hi[valid])),
            compatible_source_rows=int(np.sum(valid)),
            q50_float32=q,
            rounding_bin_lower=float(bin_lower),
            rounding_bin_upper=float(bin_upper),
        )


@dataclass(frozen=True)
class TransitionBound:
    lower: float
    upper: float
    dry_lower: float
    dry_upper: float
    rain_lower: float
    rain_upper: float


class FrozenRobustTransitionEnvelope:
    """Analytic envelope of frozen D0/R3 supports; no future weather input."""

    def __init__(self, dry_pool, r3_intercept, r3_slope, r3_residuals,
                 clean_multiplier, kernel_upper):
        dry = np.asarray(dry_pool, dtype=float)
        residual = np.asarray(r3_residuals, dtype=float)
        require(
            dry.ndim == residual.ndim == 1
            and len(dry) > 0 and len(residual) > 0
            and np.isfinite(dry).all() and np.isfinite(residual).all(),
            "Finite frozen disturbance pools",
        )
        self.dry_min = float(np.min(dry))
        self.dry_max = float(np.max(dry))
        self.rain_residual_min = float(np.min(residual))
        self.rain_residual_max = float(np.max(residual))
        self.rain_intercept = float(r3_intercept)
        self.rain_state_slope = 1.0 + float(r3_slope)
        self.clean_multiplier = float(clean_multiplier)
        self.kernel_upper = float(kernel_upper)
        require(
            0.0 <= self.clean_multiplier <= 1.0
            and self.rain_state_slope >= 0.0
            and 0.0 < self.kernel_upper < 1.0,
            "Monotone frozen transition coefficients",
        )

    @staticmethod
    def _clip(value):
        return float(np.clip(float(value), 0.0, 1.0))

    def next_bounds(self, lower, upper, action):
        lo, hi = float(lower), float(upper)
        require(
            0.0 <= lo <= hi <= self.kernel_upper
            and type(action) is int and action in (0, 1),
            "Belief/action domain",
        )
        multiplier = self.clean_multiplier if action == 1 else 1.0
        post_lo, post_hi = multiplier * lo, multiplier * hi

        dry_lo = self._clip(post_lo + self.dry_min)
        dry_hi = self._clip(post_hi + self.dry_max)
        rain_lo = self._clip(
            self.rain_state_slope * post_lo
            + self.rain_intercept
            + self.rain_residual_min
        )
        rain_hi = self._clip(
            self.rain_state_slope * post_hi
            + self.rain_intercept
            + self.rain_residual_max
        )
        result = TransitionBound(
            lower=min(dry_lo, rain_lo),
            upper=max(dry_hi, rain_hi),
            dry_lower=dry_lo,
            dry_upper=dry_hi,
            rain_lower=rain_lo,
            rain_upper=rain_hi,
        )
        require(0.0 <= result.lower <= result.upper <= 1.0, "Transition envelope")
        return result

    def action_safe(self, lower, upper, action):
        bound = self.next_bounds(lower, upper, action)
        return bool(bound.upper <= self.kernel_upper), bound


@dataclass(frozen=True)
class ShieldDecision:
    proposed_action: int
    executed_action: int
    intervention: bool
    proposed_safe: bool
    clean_safe: bool
    belief_lower: float
    belief_upper: float
    proposed_next_upper: float
    executed_next_lower: float
    executed_next_upper: float


class Q50SetMembershipShield:
    """History-carrying q50-only shield.

    Public control methods intentionally accept only q50 observations and
    proposed/executed actions. Current L, width, rain flags and future weather are
    not arguments.
    """

    def __init__(self, observation_inverse, transition_envelope):
        self.inverse = observation_inverse
        self.transition = transition_envelope
        require(
            self.inverse.kernel_upper == self.transition.kernel_upper,
            "Common frozen kernel",
        )
        self.kernel_upper = self.inverse.kernel_upper
        self.reset()

    def reset(self):
        self.prior_lower = 0.0
        self.prior_upper = self.kernel_upper
        self.belief_lower = None
        self.belief_upper = None
        self.observation_count = 0
        self.intervention_count = 0
        self.executed_clean_count = 0

    def observe(self, q50_float32):
        obs = self.inverse.bounds(q50_float32)
        lo = max(float(self.prior_lower), obs.lower)
        hi = min(float(self.prior_upper), obs.upper)
        require(lo <= hi, "Prediction/observation consistency set is empty")
        self.belief_lower, self.belief_upper = float(lo), float(hi)
        self.observation_count += 1
        return {
            "belief_lower": self.belief_lower,
            "belief_upper": self.belief_upper,
            "belief_width": self.belief_upper - self.belief_lower,
            "prior_lower": float(self.prior_lower),
            "prior_upper": float(self.prior_upper),
            "observation_lower": obs.lower,
            "observation_upper": obs.upper,
            "compatible_source_rows": obs.compatible_source_rows,
            "q50_float32": obs.q50_float32,
            "rounding_bin_lower": obs.rounding_bin_lower,
            "rounding_bin_upper": obs.rounding_bin_upper,
        }

    def filter_action(self, proposed_action):
        require(
            self.belief_lower is not None
            and self.belief_upper is not None
            and type(proposed_action) is int
            and proposed_action in (0, 1),
            "Observed live belief and discrete proposed action required",
        )
        proposed_safe, proposed_bound = self.transition.action_safe(
            self.belief_lower, self.belief_upper, proposed_action
        )
        clean_safe, clean_bound = self.transition.action_safe(
            self.belief_lower, self.belief_upper, 1
        )
        if proposed_safe:
            executed = proposed_action
            executed_bound = proposed_bound
        else:
            require(clean_safe, "Neither proposed action nor CLEAN is robustly safe")
            executed = 1
            executed_bound = clean_bound

        intervention = executed != proposed_action
        self.intervention_count += int(intervention)
        self.executed_clean_count += int(executed == 1)
        return ShieldDecision(
            proposed_action=proposed_action,
            executed_action=executed,
            intervention=intervention,
            proposed_safe=proposed_safe,
            clean_safe=clean_safe,
            belief_lower=float(self.belief_lower),
            belief_upper=float(self.belief_upper),
            proposed_next_upper=float(proposed_bound.upper),
            executed_next_lower=float(executed_bound.lower),
            executed_next_upper=float(executed_bound.upper),
        )

    def predict_after_action(self, executed_action):
        require(
            self.belief_lower is not None
            and self.belief_upper is not None
            and type(executed_action) is int
            and executed_action in (0, 1),
            "Live belief/action required",
        )
        safe, bound = self.transition.action_safe(
            self.belief_lower, self.belief_upper, executed_action
        )
        require(safe, "Executed action must be robustly kernel-safe")
        self.prior_lower, self.prior_upper = float(bound.lower), float(bound.upper)
        self.belief_lower = self.belief_upper = None
        return {
            "prior_lower": self.prior_lower,
            "prior_upper": self.prior_upper,
        }
