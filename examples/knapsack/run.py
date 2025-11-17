"""
Run Vibe Descent optimization on knapsack problem.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vibedescent import VibeDescentTrainer, ObjectiveConfig
from vibedescent.critic import SimpleCritic
from examples.knapsack import (
    KnapsackProblem,
    KnapsackSolution,
    KnapsackEvaluator,
    KnapsackProposer,
    KnapsackOperators,
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Vibe Descent on 0/1 Knapsack problem"
    )
    
    # Problem parameters
    parser.add_argument(
        '--n', type=int, default=100,
        help='Number of items'
    )
    parser.add_argument(
        '--capacity-ratio', type=float, default=0.4,
        help='Capacity as ratio of total weight'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--iters', type=int, default=30,
        help='Maximum iterations'
    )
    parser.add_argument(
        '--k', type=int, default=6,
        help='Candidates per iteration'
    )
    parser.add_argument(
        '--target-gap', type=float, default=0.5,
        help='Target optimality gap (%)'
    )
    parser.add_argument(
        '--patience', type=int, default=3,
        help='Early stopping patience'
    )
    
    # Other options
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to objective config YAML file'
    )
    
    args = parser.parse_args()
    
    # Create problem
    print(f"Generating random knapsack instance (n={args.n}, seed={args.seed})...")
    problem = KnapsackProblem.random(
        name=f"random_n{args.n}",
        n=args.n,
        capacity_ratio=args.capacity_ratio,
        seed=args.seed
    )
    
    print(f"Problem: {problem.name}")
    print(f"  Items: {len(problem.items)}")
    print(f"  Capacity: {problem.capacity}")
    print(f"  Total weight: {sum(item.weight for item in problem.items)}")
    print(f"  Total value: {sum(item.value for item in problem.items)}")
    
    # Create configuration
    if args.config:
        config = ObjectiveConfig.from_yaml(args.config)
    else:
        config = ObjectiveConfig.default_knapsack()
        config.stop['max_iters'] = args.iters
        config.stop['target_gap_pct'] = args.target_gap
        config.stop['patience'] = args.patience
    
    # Create components
    evaluator = KnapsackEvaluator(
        weights=config.weights,
        compute_exact=True
    )
    
    proposer = KnapsackProposer(
        use_momentum=True,
        exploration_rate=0.2
    )
    
    critic = SimpleCritic()
    
    # Create initial solution (greedy)
    print("\nGenerating initial solution...")
    initial_solution = KnapsackOperators.greedy_by_density(problem)
    
    # Create trainer with operator application
    trainer = VibeDescentTrainer(
        problem=problem,
        evaluator=evaluator,
        proposer=proposer,
        critic=critic,
        config=config,
        initial_solution=initial_solution,
        verbose=not args.quiet
    )
    
    # Hook up operator application
    original_apply = trainer._apply_operator
    
    def apply_with_operators(operator, solution):
        return KnapsackOperators.apply(operator, problem, solution)
    
    trainer._apply_operator = apply_with_operators
    
    # Run optimization
    best_solution, best_eval, stats = trainer.train(
        candidates_per_iteration=args.k
    )
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Objective value: {best_eval.objective_value:.0f}")
    print(f"Optimality gap: {best_eval.gap_pct:.2f}%")
    
    if best_eval.metadata.get('exact_optimum'):
        opt = best_eval.metadata['exact_optimum']
        print(f"Exact optimum: {opt}")
        print(f"Gap to optimum: {(opt - best_eval.objective_value) / opt * 100:.2f}%")
    
    print(f"Weight used: {best_eval.metadata['total_weight']}/{problem.capacity} "
          f"({best_eval.metadata['capacity_used_pct']:.1f}%)")
    print(f"Items selected: {best_eval.metadata['num_items']}/{len(problem.items)}")
    print(f"\nIterations: {stats.iterations}")
    print(f"Improvements: {stats.improvements}")
    print(f"Total evaluations: {stats.total_evaluations}")
    print(f"Total time: {stats.total_time_ms/1000:.2f}s")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

