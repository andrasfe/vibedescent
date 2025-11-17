"""Problem definition for Falling Squares optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Sequence, Optional
import random
import time

from vibedescent.core import Problem, Solution
from .solvers import (
    run_solver,
    falling_squares_segment_tree,
    falling_squares_naive,
    list_strategies,
)


Positions = Sequence[Sequence[int]]


def _default_cases() -> List[List[List[int]]]:
    """Seed cases derived from the PDF examples."""
    return [
        [[1, 2], [2, 3], [6, 1]],
        [[100, 100], [200, 100]],
        [[9, 7], [1, 9], [3, 1], [2, 2], [7, 2], [1, 9]],
    ]


def _random_case(rng: random.Random, length: int, coord_limit: int, size_limit: int) -> List[List[int]]:
    case: List[List[int]] = []
    left = 0
    for _ in range(length):
        left = rng.randint(0, coord_limit)
        size = rng.randint(1, size_limit)
        case.append([left, size])
    return case


@dataclass
class FallingSquaresSolution(Solution):
    """Represents a solver configuration."""

    strategy: str = "naive"
    params: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "FallingSquaresSolution":
        return FallingSquaresSolution(strategy=self.strategy, params=dict(self.params))

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "params": self.params}


@dataclass
class FallingSquaresProblem(Problem):
    """Problem containing multiple Falling Squares instances."""

    name: str = "falling_squares"
    test_cases: List[List[List[int]]] = field(default_factory=_default_cases)
    num_random_cases: int = 5
    random_case_length: int = 60
    coord_limit: int = 400
    size_limit: int = 50
    seed: int = 42

    reference_outputs: List[List[int]] = field(init=False)
    best_runtime_ms: float = field(init=False)
    worst_runtime_ms: float = field(init=False)
    baseline_memory_mb: float = field(init=False)
    baseline_allocations: float = field(init=False)

    def __post_init__(self):
        rng = random.Random(self.seed)
        for _ in range(self.num_random_cases):
            self.test_cases.append(
                _random_case(rng, self.random_case_length, self.coord_limit, self.size_limit)
            )

        self.reference_outputs = []
        total_best = 0.0
        total_worst = 0.0
        total_memory = 0.0
        total_allocs = 0.0

        for case in self.test_cases:
            start = time.perf_counter()
            ref, stats = falling_squares_segment_tree(case, return_stats=True)
            total_best += (time.perf_counter() - start) * 1000
            self.reference_outputs.append(ref)
            total_memory += stats["memory_bytes"]
            total_allocs += stats["allocations"]

            start = time.perf_counter()
            falling_squares_naive(case)
            total_worst += (time.perf_counter() - start) * 1000

        count = len(self.test_cases)
        self.best_runtime_ms = max(total_best / count, 1e-6)
        self.worst_runtime_ms = max(total_worst / count, self.best_runtime_ms * 1.5)
        self.baseline_memory_mb = max((total_memory / count) / 1_000_000.0, 1e-6)
        self.baseline_allocations = max(total_allocs / count, 1.0)

    def get_context(self) -> Dict[str, Any]:
        return {
            "num_cases": len(self.test_cases),
            "avg_case_length": sum(len(c) for c in self.test_cases) / len(self.test_cases),
            "best_runtime_ms": self.best_runtime_ms,
            "baseline_memory_mb": self.baseline_memory_mb,
            "baseline_allocations": self.baseline_allocations,
            "strategies": list_strategies(),
        }

