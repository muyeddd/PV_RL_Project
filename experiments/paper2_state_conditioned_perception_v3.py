"""State-conditioned residual-transport perception v3.

Scientific contract
-------------------
* Frozen v1-supported queries return the parent v1/v2 sample EXACTLY.
* Only queries that are undefined under the frozen radius=0.01 support rule use
  the already-audited minimum-adaptive-radius candidate geometry from v2.
* Source selection, BLOCK10 block-uniform sampling, Gaussian within-block row
  sampling, RNG consumption, 20-row/3-date thresholds, bandwidth and source
  data remain inherited.
* The only new operation is a sampling-law-weighted local-linear correction of
  dq50/dlower/dupper residual trend with true_L on adaptive-only queries.
* No ridge, slope clipping, radius cap, fallback, model selection or new RNG.

Import is inert: no assets are read, no RNG is initialized and no output is
created.
"""
from __future__ import annotations

import math
import numpy as np

import paper2_adaptive_radius_perception_v2 as v2


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


class StateConditionedResidualTransportV3(v2.AdaptiveRadiusOnlineBlock10EmulatorV2):
    """Minimum-adaptive-radius BLOCK10 with local-linear residual detrending.

    For base-supported queries this class returns ``super().sample_one`` without
    modifying the returned dictionary. Therefore the legacy path is bitwise
    identical and consumes exactly the same RNG state.

    For adaptive-only queries, ``super().sample_one`` is still called first.
    Hence candidate selection, block choice, source-row choice and RNG
    consumption are exactly the v2/frozen sampling law. The sampled row's
    residual vector is then transported to the query state by removing only the
    locally estimated linear residual trend.
    """

    RESIDUAL_NAMES = ("dq50", "dlower", "dupper")

    def _base_supported(self, query_L):
        q = float(query_L)
        original = v2.frozen.OnlineBlock10Emulator._candidate_indices(self, q)
        dates = np.unique(self.date[original])
        return bool(
            len(original) >= v2.frozen.MIN_LOCAL_SAMPLES
            and len(dates) >= v2.frozen.MIN_LOCAL_DATES
        )

    def _sampling_law_weights(self, query_L, cand):
        """Unconditional per-row probabilities implied by frozen BLOCK10 law."""
        q = float(query_L)
        cand = np.asarray(cand, dtype=int)
        require(len(cand) >= v2.frozen.MIN_LOCAL_SAMPLES, "Adaptive candidate rows")
        blocks = sorted(np.unique(self.block[cand]).tolist())
        require(bool(blocks), "At least one adaptive BLOCK10 block")

        p = np.zeros(len(cand), dtype=float)
        candidate_blocks = self.block[cand]
        for block in blocks:
            pos = np.flatnonzero(candidate_blocks == block)
            rows = cand[pos]
            distances = np.abs(self.L[rows] - q)
            kernel = np.exp(
                -0.5 * np.square(distances / v2.frozen.KERNEL_BANDWIDTH)
            )
            total = float(kernel.sum())
            require(
                np.isfinite(total) and total > 0.0,
                "Finite positive frozen within-block kernel mass",
            )
            p[pos] = (kernel / total) / float(len(blocks))

        require(np.isfinite(p).all() and (p > 0.0).all(), "Finite positive sampling probabilities")
        require(abs(float(p.sum()) - 1.0) <= 1e-12, "Sampling-law probabilities sum to one")
        return p

    @staticmethod
    def _weighted_slope(x, y, p):
        """Unregularized WLS slope; no tuning constant or shrinkage."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        require(len(x) == len(y) == len(p) and len(x) > 1, "WLS aligned arrays")
        xbar = float(np.sum(p * x))
        ybar = float(np.sum(p * y))
        xc = x - xbar
        denom = float(np.sum(p * xc * xc))
        numer = float(np.sum(p * xc * (y - ybar)))
        require(np.isfinite(denom) and denom > 0.0, "Positive local WLS state variance")
        slope = numer / denom
        require(math.isfinite(slope), "Finite local WLS slope")
        return float(slope)

    def trend_diagnostics(self, query_L):
        """Deterministic adaptive-only local trend; no RNG or prediction."""
        q = float(query_L)
        require(math.isfinite(q) and 0.0 <= q <= 1.0, "Query domain; no clipping")
        requirement = self.radius_requirement(q)
        if requirement["required_radius"] == v2.frozen.LOCAL_RADIUS:
            return {
                "v3_applied": False,
                "required_radius": v2.frozen.LOCAL_RADIUS,
                "candidate_samples": int(requirement["rows_at_required_radius"]),
                "candidate_dates": int(requirement["dates_at_required_radius"]),
                "candidate_blocks": int(
                    len(
                        np.unique(
                            self.block[
                                v2.frozen.OnlineBlock10Emulator._candidate_indices(self, q)
                            ]
                        )
                    )
                ),
                "sampling_probability_ess": None,
                "slope_dq50": None,
                "slope_dlower": None,
                "slope_dupper": None,
            }

        cand = self._candidate_indices(q)
        dates = np.unique(self.date[cand])
        require(
            len(cand) >= v2.frozen.MIN_LOCAL_SAMPLES
            and len(dates) >= v2.frozen.MIN_LOCAL_DATES,
            "Adaptive candidate set meets frozen 20/3 rule",
        )
        p = self._sampling_law_weights(q, cand)
        x = self.L[cand] - q
        slopes = {
            name: self._weighted_slope(x, getattr(self, name)[cand], p)
            for name in self.RESIDUAL_NAMES
        }
        ess = 1.0 / float(np.sum(np.square(p)))
        require(math.isfinite(ess) and ess > 0.0, "Finite sampling-probability ESS")
        return {
            "v3_applied": True,
            "required_radius": float(requirement["required_radius"]),
            "candidate_samples": int(len(cand)),
            "candidate_dates": int(len(dates)),
            "candidate_blocks": int(len(np.unique(self.block[cand]))),
            "sampling_probability_ess": float(ess),
            "slope_dq50": float(slopes["dq50"]),
            "slope_dlower": float(slopes["dlower"]),
            "slope_dupper": float(slopes["dupper"]),
        }

    def sample_one(self, query_L, rng):
        """Sample with frozen selection/RNG; adjust transport only if adaptive."""
        q = float(query_L)
        require(math.isfinite(q) and 0.0 <= q <= 1.0, "Query domain; no clipping")

        # This is the exact v2/frozen selection+transport call. It consumes all
        # RNG before any v3 arithmetic and returns the unchanged legacy sample.
        raw = super().sample_one(q, rng)

        if self._base_supported(q):
            return raw

        diag = self.trend_diagnostics(q)
        require(diag["v3_applied"] is True, "Adaptive-only v3 branch")
        src = int(raw["source_index"])
        source_delta = float(self.L[src] - q)

        e50 = float(self.dq50[src]) - diag["slope_dq50"] * source_delta
        elower = float(self.dlower[src]) - diag["slope_dlower"] * source_delta
        eupper = float(self.dupper[src]) - diag["slope_dupper"] * source_delta

        q50 = float(np.clip(q + e50, 0.0, 1.0))
        upper = float(np.clip(q + eupper, 0.0, 1.0))
        raw_lower = float(q + elower)

        if bool(raw["source_lower_clipped"]):
            lower = 0.0
        else:
            lower = float(np.clip(raw_lower, 0.0, 1.0))

        if lower > q50 + 1e-12:
            raise RuntimeError(
                f"V3 transport broke lower <= q50 at query {q:.17g}"
            )
        if q50 > upper + 1e-12:
            raise RuntimeError(
                f"V3 transport broke q50 <= upper at query {q:.17g}"
            )

        generated_lower_clipped = bool(
            np.isclose(lower, 0.0, atol=1e-12, rtol=0.0)
        )
        if bool(raw["source_lower_clipped"]) and not generated_lower_clipped:
            raise RuntimeError("V3 violated frozen source-clipping implication")

        out = dict(raw)
        out.update(
            {
                "q50": q50,
                "lower": lower,
                "upper": upper,
                "width": float(upper - lower),
                "covered": bool(lower <= q <= upper),
                "lower_clipped": generated_lower_clipped,
                "raw_transported_lower_before_clip": raw_lower,
                "new_transport_lower_clip": bool(
                    (not bool(raw["source_lower_clipped"]))
                    and generated_lower_clipped
                ),
                "v3_applied": True,
                "v3_required_radius": float(diag["required_radius"]),
                "v3_sampling_probability_ess": float(
                    diag["sampling_probability_ess"]
                ),
                "v3_slope_dq50": float(diag["slope_dq50"]),
                "v3_slope_dlower": float(diag["slope_dlower"]),
                "v3_slope_dupper": float(diag["slope_dupper"]),
            }
        )
        require(
            math.isfinite(out["width"]) and out["width"] >= 0.0,
            "Finite nonnegative v3 interval width",
        )
        return out
