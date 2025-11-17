"""Evaluator for Falling Squares solver strategies."""

from __future__ import annotations

from typing import Dict, Any, List
import time

from vibedescent.core import EvalResult, Evaluator as BaseEvaluator
from .problem import FallingSquaresProblem, FallingSquaresSolution
from .solvers import run_solver


class FallingSquaresEvaluator(BaseEvaluator[FallingSquaresSolution]):
    """Evaluates solver configurations on reference test cases."""

    def __init__(self, weights: Dict[str, float] | None = None):
        super().__init__()
        self.weights = weights or {
            "gap": 0.8,
            "runtime": 0.15,
            "violation": 0.05,
        }

    def evaluate(
        self,
        problem: FallingSquaresProblem,
        solution: FallingSquaresSolution,
    ) -> EvalResult:
        total_runtime = 0.0
        total_memory_bytes = 0.0
        total_allocations = 0.0
        feasible = True
        violations = 0

        for case, reference in zip(problem.test_cases, problem.reference_outputs):
            start = time.perf_counter()
            result = run_solver(
                solution.strategy,
                solution.params,
                case,
                return_stats=True,
            )
            output, stats = result  # type: ignore[misc]
            runtime = (time.perf_counter() - start) * 1000
            total_runtime += runtime
            if output != reference:
                feasible = False
                violations += 1
                break
            total_memory_bytes += stats["memory_bytes"]
            total_allocations += stats["allocations"]

        avg_runtime = total_runtime / max(1, len(problem.test_cases))
        gap_pct = (
            max(0.0, (avg_runtime - problem.best_runtime_ms) / problem.best_runtime_ms * 100.0)
            if feasible
            else float("inf")
        )

        avg_memory_mb = (total_memory_bytes / max(1, len(problem.test_cases))) / 1_000_000.0
        avg_allocations = total_allocations / max(1, len(problem.test_cases))
        allocation_norm = (
            avg_allocations / problem.baseline_allocations if feasible else 1.0
        )

        metadata = {
            "strategy": solution.strategy,
            "avg_runtime_ms": avg_runtime,
            "params": solution.params,
            "avg_memory_mb": avg_memory_mb,
            "avg_allocations": avg_allocations,
        }

        return EvalResult(
            feasible=feasible,
            constraints_satisfied={"correctness": feasible},
            objective_value=avg_runtime,
            lower_bound=problem.best_runtime_ms,
            gap_pct=gap_pct,
            runtime_ms=avg_runtime,
            memory_mb=avg_memory_mb,
            quality_scores={
                "violations": violations,
                "allocations": allocation_norm,
            },
            metadata=metadata,
        )

