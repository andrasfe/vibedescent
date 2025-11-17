"""
Proposer for knapsack problem.
"""

from typing import List, Dict, Any

from vibedescent.core import Proposer, Operator, Candidate
from vibedescent.proposer import AdaptiveProposer
from .problem import KnapsackProblem, KnapsackSolution


class KnapsackProposer(AdaptiveProposer[KnapsackSolution]):
    """
    Proposer for knapsack problem.
    
    Extends adaptive proposer with knapsack-specific operator pool.
    """
    
    def __init__(
        self,
        use_momentum: bool = True,
        exploration_rate: float = 0.2
    ):
        """Initialize knapsack proposer with default operators."""
        
        # Define operator pool
        operator_pool = [
            # Greedy constructions
            {'type': 'greedy_density'},
            {'type': 'greedy_value'},
            {'type': 'repair_fill'},
            
            # Local search with varying intensities
            {'type': 'two_opt_swap', 'tries': 50},
            {'type': 'two_opt_swap', 'tries': 100},
            {'type': 'two_opt_swap', 'tries': 200},
            
            # Large neighborhood search with varying destroy sizes
            {'type': 'ruin_recreate', 'destroy_frac': 0.1, 'strategy': 'density'},
            {'type': 'ruin_recreate', 'destroy_frac': 0.2, 'strategy': 'density'},
            {'type': 'ruin_recreate', 'destroy_frac': 0.3, 'strategy': 'density'},
            {'type': 'ruin_recreate', 'destroy_frac': 0.2, 'strategy': 'random'},
            
            # Randomized constructions
            {'type': 'randomized_greedy', 'alpha': 0.1},
            {'type': 'randomized_greedy', 'alpha': 0.2},
            {'type': 'randomized_greedy', 'alpha': 0.3},
            
            # Exact solver (will only work for small n)
            {'type': 'meet_in_the_middle'},
        ]
        
        super().__init__(
            operator_pool=operator_pool,
            use_momentum=use_momentum,
            exploration_rate=exploration_rate
        )
    
    def propose_operators(
        self,
        problem: KnapsackProblem,
        current_solution: KnapsackSolution,
        history: List[Candidate],
        trust_region: float,
        k: int = 6
    ) -> List[Operator]:
        """
        Propose operators with problem-specific adjustments.
        
        For small instances, biases toward exact solver.
        For large instances, focuses on heuristics.
        """
        context = problem.get_context()
        n_items = context['n_items']
        
        # If small enough, definitely try exact solver
        if n_items <= 46:
            # Include meet-in-the-middle
            ops = super().propose_operators(
                problem, current_solution, history, trust_region, k - 1
            )
            ops.append(Operator(type='meet_in_the_middle', params={}))
            return ops
        
        # Otherwise use standard adaptive proposal
        return super().propose_operators(
            problem, current_solution, history, trust_region, k
        )


def example_model_proposer(
    history: List[Candidate],
    context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Example reasoning model function for proposing operators.
    
    This is a simple heuristic that mimics what a reasoning model might do.
    Replace with actual LLM call in production.
    """
    n_items = context.get('n_items', 0)
    
    # For small instances, prefer exact solver
    if n_items <= 46:
        return [{'type': 'meet_in_the_middle'}]
    
    # If no history, start with strong baselines
    if not history:
        return [
            {'type': 'greedy_density'},
            {'type': 'greedy_value'},
            {'type': 'two_opt_swap', 'tries': 150},
            {'type': 'ruin_recreate', 'destroy_frac': 0.2, 'strategy': 'density'},
            {'type': 'randomized_greedy', 'alpha': 0.2},
            {'type': 'repair_fill'},
        ]
    
    # Analyze recent performance
    recent = history[-5:]
    avg_gap = sum(c.evaluation.gap_pct for c in recent) / len(recent)
    
    # If gap is high, try more aggressive operators
    if avg_gap > 5.0:
        return [
            {'type': 'ruin_recreate', 'destroy_frac': 0.35, 'strategy': 'density'},
            {'type': 'ruin_recreate', 'destroy_frac': 0.4, 'strategy': 'random'},
            {'type': 'two_opt_swap', 'tries': 200},
            {'type': 'randomized_greedy', 'alpha': 0.3},
            {'type': 'greedy_density'},
        ]
    
    # If gap is low, fine-tune with local search
    else:
        return [
            {'type': 'two_opt_swap', 'tries': 150},
            {'type': 'two_opt_swap', 'tries': 100},
            {'type': 'repair_fill'},
            {'type': 'ruin_recreate', 'destroy_frac': 0.15, 'strategy': 'density'},
            {'type': 'randomized_greedy', 'alpha': 0.15},
        ]

