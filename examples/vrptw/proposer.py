"""Adaptive proposer for VRPTW strategies."""

from __future__ import annotations

from typing import List
import random

from vibedescent.core import Operator, Candidate, Problem
from vibedescent.proposer import AdaptiveProposer
from .problem import VRPTWSolution


class VRPTWProposer(AdaptiveProposer[VRPTWSolution]):
    def __init__(self):
        operator_pool = [
            {"type": "nearest_time_window"},
            {"type": "savings"},
            {"type": "randomized_insertion", "seed": 0},
            {"type": "randomized_insertion", "seed": 1},
            {"type": "two_phase", "seed": 2},
        ]
        super().__init__(operator_pool=operator_pool, use_momentum=True, exploration_rate=0.3)

    def propose_operators(
        self,
        problem: Problem,
        current_solution: VRPTWSolution,
        history: List[Candidate],
        trust_region: float,
        k: int = 4,
    ) -> List[Operator]:
        ops = super().propose_operators(problem, current_solution, history, trust_region, k)

        # Add slight randomness to randomized strategies
        for op in ops:
            if op.type == "randomized_insertion":
                op.params["seed"] = random.randint(0, 10_000)
            if op.type == "two_phase":
                op.params["seed"] = random.randint(0, 10_000)
        return ops

