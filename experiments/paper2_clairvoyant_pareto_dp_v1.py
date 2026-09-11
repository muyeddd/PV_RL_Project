"""Exact reachable-label Pareto DP for deterministic, monotone callbacks.

Caller must establish monotonicity of stage cost and next state in current
state for each fixed action. This module knows no environment, files or RNG.
Float64 exact comparisons determine membership; no comparison tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


MAX_FRONTIER_LABELS = 1_000_000


class FrontierExplosion(RuntimeError):
    status = "EXACT_SOLVER_FRONTIER_EXPLOSION"

    def __init__(self, day, size, statistics):
        self.day = day
        self.size = size
        self.statistics = tuple(statistics)
        super().__init__(f"{self.status}: day={day}, frontier_size={size}, cap={MAX_FRONTIER_LABELS}")


@dataclass(frozen=True, slots=True)
class Label:
    state: float
    cumulative_cost: float
    clean_count: int
    parent_index: int
    action: int
    creation_order: int
    post_state: float
    stage_cost: float


@dataclass(frozen=True, slots=True)
class DayStatistics:
    day_index: int
    frontier_in: int
    candidates_generated: int
    dominated_pruned: int
    exact_duplicates_pruned: int
    frontier_out: int
    terminal_unselected: int = 0


@dataclass(frozen=True, slots=True)
class TraceStep:
    day_index: int
    L_pre: float
    action: int
    L_post: float
    stage_cost: float
    L_next: float | None
    cumulative_cost: float


@dataclass(frozen=True, slots=True)
class Solution:
    total_cost: float
    clean_count: int
    actions: tuple[int, ...]
    trace: tuple[TraceStep, ...]
    statistics: tuple[DayStatistics, ...]
    tie_detected: bool
    tie_count: int


def _finite_state(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Callback state must be finite and in [0,1]")
    return value


def _finite_cost(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Callback/cumulative cost must be finite and nonnegative")
    return value


def pareto_prune(candidates, day, statistics):
    """Sorted sweep: strictly decreasing costs along increasing states.

    Exact duplicate pairs retain the earliest creation order. A candidate with
    cost >= the smallest preceding cost is dominated (equal pairs handled
    separately). Sorting ensures the preceding state's <= relation. No epsilon.
    """
    ordered = sorted(candidates, key=lambda label: (label.state, label.cumulative_cost, label.creation_order))
    retained = []
    best_cost = math.inf
    previous_pair = None
    dominated = duplicates = 0
    for candidate in ordered:
        pair = (candidate.state, candidate.cumulative_cost)
        if pair == previous_pair:
            duplicates += 1
        elif candidate.cumulative_cost < best_cost:
            retained.append(candidate)
            best_cost = candidate.cumulative_cost
            if len(retained) > MAX_FRONTIER_LABELS:
                raise FrontierExplosion(day, len(retained), statistics)
        else:
            dominated += 1
        previous_pair = pair
    return retained, dominated, duplicates


def solve_pareto_dp(initial_state: float, horizon: int,
                    settlement_callback: Callable[[int, float, int], tuple[float, float]],
                    transition_callback: Callable[[int, float], float]) -> Solution:
    """Callbacks return (post_state, cost) and next_state respectively.

    Days are zero-based within the supplied horizon. WAIT precedes CLEAN.
    No terminal transition. Parent pointers recover the sequence without a
    second solve. Terminal ties are among surviving terminal candidates, ordered
    by (cost, post_state, clean_count, creation_order).
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Positive integer horizon required")
    initial = _finite_state(initial_state)
    layers = [[Label(initial, 0.0, 0, -1, -1, 0, initial, 0.0)]]
    statistics = []
    creation_order = 0
    tie_count = 0
    for day in range(horizon):
        frontier = layers[-1]
        candidates = []
        terminal = day == horizon - 1
        for parent_index, parent in enumerate(frontier):
            for action in (0, 1):
                post, cost = settlement_callback(day, parent.state, action)
                post, cost = _finite_state(post), _finite_cost(cost)
                state = post if terminal else _finite_state(transition_callback(day, post))
                creation_order += 1
                candidates.append(Label(state, _finite_cost(parent.cumulative_cost + cost),
                                        parent.clean_count + action, parent_index, action,
                                        creation_order, post, cost))
        if terminal:
            winner = min(candidates, key=lambda label: (label.cumulative_cost, label.post_state,
                                                        label.clean_count, label.creation_order))
            tie_count = sum(label.cumulative_cost == winner.cumulative_cost for label in candidates)
            retained, dominated, duplicates = [winner], 0, 0
        else:
            retained, dominated, duplicates = pareto_prune(candidates, day, statistics)
        statistics.append(DayStatistics(day, len(frontier), len(candidates), dominated, duplicates,
                                         len(retained), len(candidates) - 1 if terminal else 0))
        layers.append(retained)
    index = 0
    reverse_trace = []
    for day in range(horizon - 1, -1, -1):
        label = layers[day + 1][index]
        parent = layers[day][label.parent_index]
        reverse_trace.append(TraceStep(day, parent.state, label.action, label.post_state, label.stage_cost,
                                      None if day == horizon - 1 else label.state, label.cumulative_cost))
        index = label.parent_index
    trace = tuple(reversed(reverse_trace))
    winner = layers[-1][0]
    return Solution(winner.cumulative_cost, winner.clean_count, tuple(row.action for row in trace),
                    trace, tuple(statistics), tie_count > 1, tie_count)
