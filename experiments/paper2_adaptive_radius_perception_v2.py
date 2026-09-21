"""Minimum-Adaptive-Radius BLOCK10 v2; frozen v1 remains unmodified.

No assets, sampling, RNG initialization, or output on import. Only candidate
geometry is new. sample_one is inherited verbatim, including transport/RNG.
The helper methods are mechanically extracted from accepted P2-1E-3A and
AST-checked by the separate freeze audit; this module never imports that audit.
"""
from __future__ import annotations
import math
import struct
import numpy as np
import run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1 as frozen

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)

def bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]

def number(value):
    return struct.unpack(">d", struct.pack(">Q", value))[0]


class RadiusGeometry:
    """Exact binary64 geometry helper; no prediction or stochastic operations."""

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


class AdaptiveRadiusOnlineBlock10EmulatorV2(frozen.OnlineBlock10Emulator):
    """Only candidate selection differs; v1-supported path returns super exactly.

    Universal identity argument: for every v1-supported query the exact parent
    candidate array is returned without any RNG access. Both versions then run
    the SAME sample_one function with identical self source arrays and RNG state.
    Thus candidate/block/row/output and post-call RNG state are identical, not
    merely close. Finite-case tests supplement this structural guarantee.
    Inherited support_summary/candidate_block_counts remain BASE-v1 diagnostics;
    use radius_requirement() for v2 support diagnostics.
    """
    def __init__(self, source):
        require(set(source['role'].astype(str)) == {frozen.EXPECTED_ROLE},
                'DECISION_DEVELOPMENT source only, including LODO subsets')
        require(source['sample_id'].is_unique, 'Unique source rows')
        super().__init__(source)
        self._geometry = RadiusGeometry(self.L, self.date, frozen.LOCAL_RADIUS,
                                        frozen.MIN_LOCAL_SAMPLES, frozen.MIN_LOCAL_DATES)

    def radius_requirement(self, query_L):
        return self._geometry.required(query_L)

    def _candidate_indices(self, query_L):
        query_L = float(query_L)
        require(math.isfinite(query_L) and 0 <= query_L <= 1, 'Query domain; no clipping')
        original = super()._candidate_indices(query_L)
        if (len(original) >= frozen.MIN_LOCAL_SAMPLES
                and len(np.unique(self.date[original])) >= frozen.MIN_LOCAL_DATES):
            return original
        radius = self.radius_requirement(query_L)['required_radius']
        lo = np.searchsorted(self.sorted_L, query_L-radius, side='left')
        hi = np.searchsorted(self.sorted_L, query_L+radius, side='right')
        return np.sort(self.order[lo:hi])
