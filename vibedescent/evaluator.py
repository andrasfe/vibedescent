"""
Concrete evaluator implementations.
"""

from typing import Dict
from .core import Evaluator as BaseEvaluator, EvalResult, Problem, Solution


class Evaluator(BaseEvaluator[Solution]):
    """
    Standard evaluator implementation.
    
    Delegates to problem-specific evaluation logic but provides
    standard loss computation.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize evaluator with loss weights.
        
        Args:
            weights: Dictionary of metric weights for loss computation
        """
        self.weights = weights or {
            'gap': 0.70,
            'runtime': 0.20,
            'memory': 0.05,
            'violation': 0.05,
        }
    
    def evaluate(self, problem: Problem, solution: Solution) -> EvalResult:
        """
        Evaluate solution - must be implemented by problem-specific evaluator.
        
        Override this in subclasses or use composition with problem-specific
        evaluation functions.
        """
        raise NotImplementedError(
            "Evaluator.evaluate must be overridden or use a problem-specific evaluator"
        )
    
    def set_weights(self, weights: Dict[str, float]):
        """Update loss weights."""
        self.weights.update(weights)

