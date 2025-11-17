"""Run Vibe Descent on VRPTW."""

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
from examples.vrptw import (
    generate_random_instance,
    VRPTWSolution,
    VRPTWEvaluator,
    VRPTWOperators,
    VRPTWProposer,
)


def build_config(target: float, max_iters: int, patience: int) -> ObjectiveConfig:
    return ObjectiveConfig(
        hard_constraints=["all_customers", "capacity"],
        weights={
            "runtime": 0.5,
            "memory": 0.0,
            "gap": 0.05,
            "allocations": 0.0,
            "violation": 0.45,
        },
        trust_region={
            "min": 0.5,
            "max": 2.0,
            "init": 1.0,
            "expand": 1.2,
            "shrink": 0.8,
        },
        stop={
            "target_gap_pct": target,
            "patience": patience,
            "max_iters": max_iters,
            "min_iters": 4,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="VRPTW optimization via Vibe Descent")
    parser.add_argument("--customers", type=int, default=25)
    parser.add_argument("--capacity", type=float, default=40.0)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--target-gap", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="logs/vrptw")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    instance = generate_random_instance(
        num_customers=args.customers,
        capacity=args.capacity,
        seed=args.seed,
    )
    config = build_config(args.target_gap, args.iters, args.patience)

    evaluator = VRPTWEvaluator(weights=config.weights)
    proposer = VRPTWProposer()
    critic = SimpleCritic()
    initial_solution = VRPTWSolution(routes=[[cust.idx] for cust in instance.customers])

    logger = Logger(log_dir=args.log_dir, verbose=not args.quiet)

    def log_callback(iteration, candidate, state):
        logger.log_iteration(iteration, candidate, state.get_status())

    trainer = VibeDescentTrainer(
        problem=instance,
        evaluator=evaluator,
        proposer=proposer,
        critic=critic,
        config=config,
        initial_solution=initial_solution,
        verbose=not args.quiet,
        log_callback=log_callback,
    )

    def apply_operator(op, sol):
        return VRPTWOperators.apply(op, instance)

    trainer._apply_operator = apply_operator

    best_solution, best_eval, stats = trainer.train(candidates_per_iteration=args.k)

    if not args.quiet:
        print("\nBest configuration:")
        print(f"  Objective: {best_eval.objective_value:.2f}")
        print(f"  Routes: {best_solution.routes}")
        print(f"  Metadata: {best_eval.metadata}")
        print(f"  Logs saved to: {logger.log_dir}")
    logger.save_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

