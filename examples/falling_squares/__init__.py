"""Falling Squares optimization example using Vibe Descent."""

from .problem import FallingSquaresProblem, FallingSquaresSolution
from .evaluator import FallingSquaresEvaluator
from .operators import FallingSquaresOperators
from .proposer import FallingSquaresProposer

__all__ = [
    "FallingSquaresProblem",
    "FallingSquaresSolution",
    "FallingSquaresEvaluator",
    "FallingSquaresOperators",
    "FallingSquaresProposer",
]

