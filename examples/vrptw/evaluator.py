"""Evaluator for VRPTW solutions."""

from __future__ import annotations

from typing import Dict, Any

from vibedescent.core import EvalResult
from vibedescent.evaluator import Evaluator as BaseEvaluator
from .problem import VRPTWInstance, VRPTWSolution


class VRPTWEvaluator(BaseEvaluator):
    def __init__(self, weights: Dict[str, float] | None = None):
        super().__init__()
        self.weights = weights or {
            "runtime": 0.5,
            "gap": 0.1,
            "memory": 0.1,
            "allocations": 0.0,
            "violation": 0.3,
        }

    def evaluate(self, problem: VRPTWInstance, solution: VRPTWSolution) -> EvalResult:
        feasible = True
        total_distance = 0.0
        total_lateness = 0.0
        capacity = problem.vehicle_capacity

        served = set()
        for route in solution.routes:
            load = 0.0
            time = 0.0
            last = 0
            for cid in route:
                cust = problem.customers[cid - 1]
                served.add(cid)
                load += cust.demand
                if load > capacity:
                    feasible = False
                travel = problem.travel_matrix[last][cust.idx]
                total_distance += travel
                time += travel
                if time < cust.ready_time:
                    time = cust.ready_time
                if time > cust.due_time:
                    total_lateness += time - cust.due_time
                time += cust.service_time
                last = cust.idx
            total_distance += problem.travel_matrix[last][0]

        missing = len(problem.customers) - len(served)
        if missing > 0:
            feasible = False

        objective = total_distance + 5.0 * total_lateness + 50.0 * missing

        return EvalResult(
            feasible=feasible,
            constraints_satisfied={"all_customers": missing == 0, "capacity": feasible},
            objective_value=objective,
            gap_pct=0.0,
            runtime_ms=objective,
            memory_mb=0.0,
            quality_scores={
                "lateness": total_lateness,
                "missing_customers": missing,
            },
            metadata={
                "distance": total_distance,
                "routes": len(solution.routes),
                "lateness": total_lateness,
                "missing": missing,
            },
        )

