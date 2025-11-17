"""
Operators (neighborhoods/heuristics) for knapsack problem.
"""

import random
from typing import List
from bisect import bisect_right

from vibedescent.core import Operator
from .problem import KnapsackProblem, KnapsackSolution


class KnapsackOperators:
    """
    Collection of operators for knapsack problem.
    
    Provides various neighborhoods and construction heuristics:
    - Greedy constructions (by density, by value)
    - Local search (1-out/1-in swaps)
    - Large neighborhood search (ruin-and-recreate)
    - Randomized construction
    - Exact solver for small instances
    """
    
    @staticmethod
    def apply(
        operator: Operator,
        problem: KnapsackProblem,
        solution: KnapsackSolution
    ) -> KnapsackSolution:
        """
        Apply an operator to a solution.
        
        Routes to appropriate method based on operator type.
        """
        op_type = operator.type
        params = operator.params
        
        if op_type == "greedy_density":
            return KnapsackOperators.greedy_by_density(problem)
        
        elif op_type == "greedy_value":
            return KnapsackOperators.greedy_by_value(problem)
        
        elif op_type == "repair_fill":
            return KnapsackOperators.repair_and_fill(problem, solution)
        
        elif op_type == "two_opt_swap":
            tries = params.get('tries', 100)
            return KnapsackOperators.two_opt_swap(problem, solution, tries)
        
        elif op_type == "ruin_recreate":
            destroy_frac = params.get('destroy_frac', 0.2)
            strategy = params.get('strategy', 'density')
            return KnapsackOperators.ruin_and_recreate(
                problem, solution, destroy_frac, strategy
            )
        
        elif op_type == "randomized_greedy":
            alpha = params.get('alpha', 0.2)
            return KnapsackOperators.randomized_greedy(problem, alpha)
        
        elif op_type == "meet_in_the_middle":
            return KnapsackOperators.meet_in_the_middle(problem)
        
        else:
            # Unknown operator, return copy
            return solution.copy()
    
    @staticmethod
    def greedy_by_density(problem: KnapsackProblem) -> KnapsackSolution:
        """
        Greedy construction by value/weight ratio.
        
        Sort items by density and greedily pack.
        """
        solution = KnapsackSolution()
        
        # Sort by density (descending)
        sorted_items = sorted(
            problem.items,
            key=lambda item: item.density(),
            reverse=True
        )
        
        total_weight = 0
        for item in sorted_items:
            if total_weight + item.weight <= problem.capacity:
                solution.add_item(item.index)
                total_weight += item.weight
        
        return solution
    
    @staticmethod
    def greedy_by_value(problem: KnapsackProblem) -> KnapsackSolution:
        """
        Greedy construction by value.
        
        Sort items by value and greedily pack.
        """
        solution = KnapsackSolution()
        
        # Sort by value (descending)
        sorted_items = sorted(
            problem.items,
            key=lambda item: item.value,
            reverse=True
        )
        
        total_weight = 0
        for item in sorted_items:
            if total_weight + item.weight <= problem.capacity:
                solution.add_item(item.index)
                total_weight += item.weight
        
        return solution
    
    @staticmethod
    def repair_to_feasible(
        problem: KnapsackProblem,
        solution: KnapsackSolution
    ) -> KnapsackSolution:
        """
        Repair infeasible solution by removing low-density items.
        """
        if solution.feasible(problem):
            return solution
        
        result = solution.copy()
        
        # Sort picked items by density (ascending = worst first)
        picked_items = sorted(
            [problem.items[i] for i in result.picked],
            key=lambda item: item.density()
        )
        
        # Remove items until feasible
        for item in picked_items:
            if result.feasible(problem):
                break
            result.remove_item(item.index)
        
        return result
    
    @staticmethod
    def add_greedy_fill(
        problem: KnapsackProblem,
        solution: KnapsackSolution
    ) -> KnapsackSolution:
        """
        Greedily add items to solution until full.
        """
        result = solution.copy()
        
        # Get unpicked items sorted by density
        unpicked = [
            item for item in problem.items
            if not result.has_item(item.index)
        ]
        unpicked.sort(key=lambda item: item.density(), reverse=True)
        
        # Add items greedily
        current_weight = result.weight(problem)
        for item in unpicked:
            if current_weight + item.weight <= problem.capacity:
                result.add_item(item.index)
                current_weight += item.weight
        
        return result
    
    @staticmethod
    def repair_and_fill(
        problem: KnapsackProblem,
        solution: KnapsackSolution
    ) -> KnapsackSolution:
        """Repair to feasible then fill greedily."""
        result = KnapsackOperators.repair_to_feasible(problem, solution)
        result = KnapsackOperators.add_greedy_fill(problem, result)
        return result
    
    @staticmethod
    def two_opt_swap(
        problem: KnapsackProblem,
        solution: KnapsackSolution,
        tries: int = 100
    ) -> KnapsackSolution:
        """
        Local search: try 1-out/1-in swaps.
        
        Randomly sample pairs and accept improving moves.
        """
        result = solution.copy()
        best_value = result.value(problem)
        
        picked = list(result.picked)
        unpicked = [i for i in range(len(problem.items)) if i not in result.picked]
        
        for _ in range(int(tries)):
            if not picked or not unpicked:
                break
            
            # Random swap
            out_idx = random.choice(picked)
            in_idx = random.choice(unpicked)
            
            # Try swap
            candidate = result.copy()
            candidate.remove_item(out_idx)
            candidate.add_item(in_idx)
            
            # Repair and fill if needed
            candidate = KnapsackOperators.repair_to_feasible(problem, candidate)
            candidate = KnapsackOperators.add_greedy_fill(problem, candidate)
            
            # Accept if improving
            cand_value = candidate.value(problem)
            if cand_value > best_value:
                result = candidate
                best_value = cand_value
                picked = list(result.picked)
                unpicked = [i for i in range(len(problem.items)) if i not in result.picked]
        
        return result
    
    @staticmethod
    def ruin_and_recreate(
        problem: KnapsackProblem,
        solution: KnapsackSolution,
        destroy_frac: float = 0.2,
        strategy: str = "density"
    ) -> KnapsackSolution:
        """
        Large neighborhood search: remove fraction of items and rebuild.
        """
        result = solution.copy()
        
        if not result.picked:
            return KnapsackOperators.greedy_by_density(problem)
        
        # Determine items to remove
        k = max(1, int(len(result.picked) * destroy_frac))
        to_remove = list(result.picked)
        
        if strategy == "density":
            # Remove worst density items
            to_remove.sort(key=lambda i: problem.items[i].density())
        else:
            # Random removal
            random.shuffle(to_remove)
        
        # Remove k items
        for item_idx in to_remove[:k]:
            result.remove_item(item_idx)
        
        # Rebuild with greedy fill
        result = KnapsackOperators.add_greedy_fill(problem, result)
        
        # Apply local search
        result = KnapsackOperators.two_opt_swap(problem, result, tries=30)
        
        return result
    
    @staticmethod
    def randomized_greedy(
        problem: KnapsackProblem,
        alpha: float = 0.2
    ) -> KnapsackSolution:
        """
        Randomized greedy with restricted candidate list.
        
        At each step, randomly select from top alpha fraction of items.
        """
        solution = KnapsackSolution()
        
        # Sort items by density
        items = sorted(
            problem.items,
            key=lambda item: item.density(),
            reverse=True
        )
        
        available = list(items)
        total_weight = 0
        rcl_size = max(1, int(len(items) * alpha))
        
        while available:
            # Restricted candidate list: top rcl_size items
            rcl = available[:min(rcl_size, len(available))]
            
            # Random selection from RCL
            item = random.choice(rcl)
            available.remove(item)
            
            # Add if fits
            if total_weight + item.weight <= problem.capacity:
                solution.add_item(item.index)
                total_weight += item.weight
        
        # Final greedy fill
        solution = KnapsackOperators.add_greedy_fill(problem, solution)
        
        return solution
    
    @staticmethod
    def meet_in_the_middle(problem: KnapsackProblem) -> KnapsackSolution:
        """
        Exact solution using meet-in-the-middle algorithm.
        
        Only works well for n <= ~46. Falls back to greedy for larger instances.
        """
        n = len(problem.items)
        
        if n > 46:
            return KnapsackOperators.greedy_by_density(problem)
        
        # Split items into two halves
        mid = n // 2
        items_A = problem.items[:mid]
        items_B = problem.items[mid:]
        
        # Enumerate all subsets of A
        subsets_A = []
        for mask in range(1 << len(items_A)):
            weight = sum(items_A[i].weight for i in range(len(items_A)) if mask & (1 << i))
            value = sum(items_A[i].value for i in range(len(items_A)) if mask & (1 << i))
            
            if weight <= problem.capacity:
                subsets_A.append((weight, value, mask))
        
        # Sort and prune dominated solutions
        subsets_A.sort()
        pruned_A = []
        max_value = -1
        
        for weight, value, mask in subsets_A:
            if value > max_value:
                pruned_A.append((weight, value, mask))
                max_value = value
        
        # Extract for binary search
        weights_A = [w for w, v, m in pruned_A]
        values_A = [v for w, v, m in pruned_A]
        masks_A = [m for w, v, m in pruned_A]
        
        # Enumerate all subsets of B and find best combination
        best_value = 0
        best_masks = (0, 0)
        
        for mask_B in range(1 << len(items_B)):
            weight_B = sum(items_B[i].weight for i in range(len(items_B)) if mask_B & (1 << i))
            value_B = sum(items_B[i].value for i in range(len(items_B)) if mask_B & (1 << i))
            
            if weight_B > problem.capacity:
                continue
            
            # Find best compatible subset from A
            remaining = problem.capacity - weight_B
            idx = bisect_right(weights_A, remaining) - 1
            
            if idx >= 0:
                total_value = value_B + values_A[idx]
                if total_value > best_value:
                    best_value = total_value
                    best_masks = (masks_A[idx], mask_B)
        
        # Build solution from best masks
        solution = KnapsackSolution()
        mask_A, mask_B = best_masks
        
        for i in range(len(items_A)):
            if mask_A & (1 << i):
                solution.add_item(items_A[i].index)
        
        for i in range(len(items_B)):
            if mask_B & (1 << i):
                solution.add_item(items_B[i].index)
        
        return solution

