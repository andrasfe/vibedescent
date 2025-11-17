"""Proposer for Falling Squares operators."""

from __future__ import annotations

from typing import List

from vibedescent.core import Operator, Candidate, Problem
from vibedescent.proposer import AdaptiveProposer
from .problem import FallingSquaresSolution


class FallingSquaresProposer(AdaptiveProposer[FallingSquaresSolution]):
    """Adaptive proposer specialized for Falling Squares strategies."""

    def __init__(self):
        operator_pool = [
            {"type": "set_strategy", "strategy": "naive"},
            {"type": "set_strategy", "strategy": "segment_tree"},
            {"type": "set_strategy", "strategy": "segment_tree_lazy"},
            {"type": "set_strategy", "strategy": "interval_map"},
            {"type": "set_strategy", "strategy": "diff_array"},
            {"type": "set_strategy", "strategy": "bucket_grid"},
            {"type": "set_strategy", "strategy": "bucket_grid_large"},
            {"type": "set_strategy", "strategy": "bitset"},
            {"type": "set_strategy", "strategy": "bitset_small_span"},
            {"type": "set_strategy", "strategy": "bitset_large_span"},
            {"type": "set_strategy", "strategy": "fenwick"},
            {"type": "set_strategy", "strategy": "block_dp"},
            {"type": "set_strategy", "strategy": "block_dp_fine"},
            {"type": "set_strategy", "strategy": "sparse_tree"},
            {"type": "toggle_lazy"},
            {"type": "reset_params"},
            {"type": "adjust_param", "param": "bucket_size", "delta": 32, "min": 16, "max": 512, "strategy": "bucket_grid", "default": 64},
            {"type": "adjust_param", "param": "bucket_size", "delta": -32, "min": 16, "max": 512, "strategy": "bucket_grid", "default": 64},
            {"type": "adjust_param", "param": "block_size", "delta": 16, "min": 16, "max": 256, "strategy": "block_dp", "default": 64},
            {"type": "adjust_param", "param": "block_size", "delta": -16, "min": 16, "max": 256, "strategy": "block_dp", "default": 64},
            {"type": "adjust_param", "param": "block_size", "delta": 8, "min": 16, "max": 128, "strategy": "block_dp_fine", "default": 32},
            {"type": "adjust_param", "param": "block_size", "delta": -8, "min": 16, "max": 128, "strategy": "block_dp_fine", "default": 32},
            {"type": "adjust_param", "param": "span_limit", "delta": 500, "min": 2000, "max": 20000, "strategy": "bitset", "default": 6000},
            {"type": "adjust_param", "param": "span_limit", "delta": -500, "min": 2000, "max": 20000, "strategy": "bitset", "default": 6000},
            {"type": "adjust_param", "param": "span_limit", "delta": 400, "min": 1500, "max": 8000, "strategy": "bitset_small_span", "default": 4000},
            {"type": "adjust_param", "param": "span_limit", "delta": -400, "min": 1500, "max": 8000, "strategy": "bitset_small_span", "default": 4000},
            {"type": "adjust_param", "param": "span_limit", "delta": 800, "min": 6000, "max": 25000, "strategy": "bitset_large_span", "default": 12000},
            {"type": "adjust_param", "param": "span_limit", "delta": -800, "min": 6000, "max": 25000, "strategy": "bitset_large_span", "default": 12000},
        ]
        super().__init__(operator_pool=operator_pool, use_momentum=True, exploration_rate=0.45)

    def propose_operators(
        self,
        problem: Problem,
        current_solution: FallingSquaresSolution,
        history: List[Candidate],
        trust_region: float,
        k: int = 4,
    ) -> List[Operator]:
        context = problem.get_context()
        if context.get("num_cases", 0) > 8 and current_solution.strategy == "naive":
            # Bias toward alternative strategies on harder workloads
            extra = Operator(type="set_strategy", params={"strategy": "diff_array"})
            ops = super().propose_operators(problem, current_solution, history, trust_region, k - 1)
            ops.append(extra)
            return ops
        return super().propose_operators(problem, current_solution, history, trust_region, k)

