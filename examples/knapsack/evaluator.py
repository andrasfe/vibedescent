"""
Evaluator for knapsack problem.
"""

from typing import Optional
import time

from vibedescent.core import Evaluator, EvalResult
from .problem import KnapsackProblem, KnapsackSolution


class KnapsackEvaluator(Evaluator[KnapsackSolution]):
    """
    Evaluator for knapsack solutions.
    
    Provides:
    - Feasibility checking
    - Objective value
    - Upper bound via fractional relaxation
    - Optional exact optimum via DP (for small instances)
    """
    
    def __init__(
        self,
        weights: dict = None,
        compute_exact: bool = True,
        dp_limit: int = 20000
    ):
        """
        Initialize knapsack evaluator.
        
        Args:
            weights: Loss weights
            compute_exact: Whether to compute exact optimum for small instances
            dp_limit: Maximum n*C product for exact DP
        """
        super().__init__()
        self.weights = weights or {
            'gap': 0.70,
            'runtime': 0.20,
            'memory': 0.05,
            'violation': 0.05,
        }
        self.compute_exact = compute_exact
        self.dp_limit = dp_limit
    
    def evaluate(
        self,
        problem: KnapsackProblem,
        solution: KnapsackSolution
    ) -> EvalResult:
        """Evaluate a knapsack solution."""
        t0 = time.time()
        
        # Basic metrics
        total_value = solution.value(problem)
        total_weight = solution.weight(problem)
        feasible = solution.feasible(problem)
        
        # Upper bound via fractional relaxation
        upper_bound = self._fractional_upper_bound(problem)
        
        # Exact optimum if feasible
        exact_opt = None
        if self.compute_exact:
            exact_opt = self._exact_dp(problem)
        
        # Compute gap
        if upper_bound > 0:
            gap_pct = max(0.0, (upper_bound - total_value) / upper_bound * 100.0)
        else:
            gap_pct = float('inf')
        
        runtime_ms = (time.time() - t0) * 1000
        
        return EvalResult(
            feasible=feasible,
            constraints_satisfied={'capacity': feasible},
            objective_value=float(total_value),
            upper_bound=upper_bound,
            gap_pct=gap_pct,
            runtime_ms=runtime_ms,
            metadata={
                'total_weight': total_weight,
                'capacity': problem.capacity,
                'capacity_used_pct': (total_weight / problem.capacity * 100) if problem.capacity > 0 else 0,
                'exact_optimum': exact_opt,
                'num_items': len(solution.picked),
            }
        )
    
    def _fractional_upper_bound(self, problem: KnapsackProblem) -> float:
        """
        Compute upper bound using fractional knapsack relaxation.
        
        Sort items by value/weight ratio and greedily fill, allowing
        fractional items.
        """
        items = problem.items
        
        # Sort by density (value/weight ratio)
        sorted_indices = sorted(
            range(len(items)),
            key=lambda i: items[i].density(),
            reverse=True
        )
        
        remaining_capacity = problem.capacity
        upper_bound = 0.0
        
        for idx in sorted_indices:
            item = items[idx]
            
            if remaining_capacity <= 0:
                break
            
            if item.weight <= remaining_capacity:
                # Take full item
                upper_bound += item.value
                remaining_capacity -= item.weight
            else:
                # Take fractional item
                fraction = remaining_capacity / item.weight
                upper_bound += item.value * fraction
                break
        
        return upper_bound
    
    def _exact_dp(self, problem: KnapsackProblem) -> Optional[int]:
        """
        Compute exact optimum using dynamic programming.
        
        Uses DP by weight: dp[w] = max value with weight exactly w
        Only computed if n * C is below threshold.
        """
        n = len(problem.items)
        C = problem.capacity
        
        if n * C > self.dp_limit:
            return None
        
        # DP table
        NEG_INF = -10**12
        dp = [NEG_INF] * (C + 1)
        dp[0] = 0
        
        for item in problem.items:
            w, v = item.weight, item.value
            
            # Iterate backwards to avoid using same item twice
            for c in range(C, w - 1, -1):
                if dp[c - w] != NEG_INF:
                    dp[c] = max(dp[c], dp[c - w] + v)
        
        return max(dp) if max(dp) > NEG_INF else 0

