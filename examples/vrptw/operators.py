"""Operators for VRPTW strategies."""

from __future__ import annotations

from vibedescent.core import Operator
from .problem import VRPTWSolution, VRPTWInstance
from .solvers import (
    nearest_time_window_heuristic,
    savings_heuristic,
    randomized_insertion_heuristic,
    two_phase_heuristic,
)


class VRPTWOperators:
    @staticmethod
    def apply(operator: Operator, instance: VRPTWInstance) -> VRPTWSolution:
        params = operator.params or {}
        strategy = operator.type

        if strategy == "nearest_time_window":
            return nearest_time_window_heuristic(instance, params)
        if strategy == "savings":
            return savings_heuristic(instance, params)
        if strategy == "randomized_insertion":
            return randomized_insertion_heuristic(instance, params)
        if strategy == "two_phase":
            return two_phase_heuristic(instance, params)

        # Default fallback
        return nearest_time_window_heuristic(instance, params)

