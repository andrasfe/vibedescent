"""
Vibe Descent: A human-in-the-loop optimization framework for agentic systems.

Inspired by gradient descent in machine learning, Vibe Descent treats:
- The problem state as the "model"
- Patches/changes as "parameter updates"
- Feedback (human or automated) as the "loss signal"
- Operators as the "optimizer"

The framework iteratively proposes changes, evaluates them against an objective,
and selects improvements until convergence or budget exhaustion.
"""

from .core import (
    Solution,
    Problem,
    EvalResult,
    Operator,
    Candidate,
    Evaluator as BaseEvaluator,
    Proposer as BaseProposer,
    Critic as BaseCritic,
)

from .evaluator import Evaluator
from .proposer import AdaptiveProposer, ModelProposer
from .critic import SimpleCritic, DiversityCritic
from .optimizer import OptimizerState, TrustRegion
from .trainer import VibeDescentTrainer
from .config import ObjectiveConfig
from .strategy import Strategy, StrategyRegistry

__version__ = "0.1.0"

__all__ = [
    "Solution",
    "Problem",
    "EvalResult",
    "Operator",
    "Candidate",
    "BaseEvaluator",
    "BaseProposer",
    "BaseCritic",
    "Evaluator",
    "AdaptiveProposer",
    "ModelProposer",
    "SimpleCritic",
    "DiversityCritic",
    "OptimizerState",
    "TrustRegion",
    "VibeDescentTrainer",
    "ObjectiveConfig",
    "Strategy",
    "StrategyRegistry",
]

