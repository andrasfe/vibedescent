"""Run Vibe Descent optimization for the Falling Squares problem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vibedescent import ObjectiveConfig, VibeDescentTrainer
from vibedescent.critic import SimpleCritic
from vibedescent.utils import Logger
from examples.falling_squares import (
    FallingSquaresProblem,
    FallingSquaresSolution,
    FallingSquaresEvaluator,
    FallingSquaresOperators,
    FallingSquaresProposer,
)


def build_config(target_gap: float, max_iters: int, patience: int) -> ObjectiveConfig:
    return ObjectiveConfig(
        hard_constraints=["correctness"],
        weights={
            "gap": 0.15,
            "runtime": 0.45,
            "memory": 0.25,
            "allocations": 0.10,
            "violation": 0.05,
        },
        trust_region={
            "min": 0.5,
            "max": 2.0,
            "init": 1.0,
            "expand": 1.2,
            "shrink": 0.8,
        },
        stop={
            "target_gap_pct": target_gap,
            "patience": patience,
            "max_iters": max_iters,
            "min_iters": 4,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize Falling Squares solver using Vibe Descent")
    parser.add_argument("--random-cases", type=int, default=5, help="Number of random cases to add")
    parser.add_argument("--case-length", type=int, default=60, help="Length of each random case")
    parser.add_argument("--iters", type=int, default=15, help="Maximum iterations")
    parser.add_argument("--k", type=int, default=4, help="Candidates per iteration")
    parser.add_argument("--target-gap", type=float, default=0.5, help="Target runtime gap (%%)")
    parser.add_argument("--patience", type=int, default=4, help="Patience before stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="logs/falling_squares", help="Directory to store logs")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    problem = FallingSquaresProblem(
        num_random_cases=args.random_cases,
        random_case_length=args.case_length,
        seed=args.seed,
    )

    config = build_config(args.target_gap, args.iters, args.patience)

    evaluator = FallingSquaresEvaluator(weights=config.weights)
    proposer = FallingSquaresProposer()
    critic = SimpleCritic()
    initial_solution = FallingSquaresSolution(strategy="segment_tree_lazy")

    logger = Logger(log_dir=args.log_dir, verbose=not args.quiet)

    def log_callback(iteration, candidate, state):
        logger.log_iteration(iteration, candidate, state.get_status())

    trainer = VibeDescentTrainer(
        problem=problem,
        evaluator=evaluator,
        proposer=proposer,
        critic=critic,
        config=config,
        initial_solution=initial_solution,
        verbose=not args.quiet,
        log_callback=log_callback,
    )

    def apply_operator(op, sol):
        return FallingSquaresOperators.apply(op, sol)

    trainer._apply_operator = apply_operator

    best_solution, best_eval, stats = trainer.train(candidates_per_iteration=args.k)

    if not args.quiet:
        print("\nBest configuration:")
        print(f"  Strategy: {best_solution.strategy}")
        print(f"  Params: {best_solution.params}")
        print(f"  Avg runtime: {best_eval.runtime_ms:.3f} ms")
        print(f"  Gap vs reference: {best_eval.gap_pct:.3f}%")
        print(f"  Logs saved to: {logger.log_dir}")
    logger.save_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

