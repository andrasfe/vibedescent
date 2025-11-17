"""
Core abstractions for the Vibe Descent framework.

This module defines the base classes and interfaces that all problem-specific
implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generic, TypeVar
import time


# Type variable for problem-specific solution types
SolutionType = TypeVar('SolutionType', bound='Solution')


@dataclass
class Solution(ABC):
    """
    Base class for problem solutions.
    
    A solution represents a specific state or configuration for the problem
    being optimized. Subclasses must implement problem-specific logic.
    """
    
    @abstractmethod
    def copy(self) -> 'Solution':
        """Create a deep copy of this solution."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize solution to dictionary for logging/storage."""
        pass


@dataclass
class Problem(ABC):
    """
    Base class for problem instances.
    
    A problem encapsulates the problem definition, constraints, and instance data.
    """
    
    name: str
    
    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
        """
        Return problem context for proposers/critics.
        
        This should include problem size, structure, and any metadata
        that operators might need to make informed decisions.
        """
        pass


@dataclass
class EvalResult:
    """
    Evaluation result for a candidate solution.
    
    Contains all metrics needed to compute loss and make decisions.
    """
    
    # Hard constraints
    feasible: bool
    constraints_satisfied: Dict[str, bool] = field(default_factory=dict)
    
    # Objective metrics
    objective_value: float = 0.0
    
    # Bounds and gaps
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    gap_pct: float = float('inf')
    
    # Performance metrics
    runtime_ms: float = 0.0
    memory_mb: float = 0.0
    
    # Quality metrics
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_gap(self) -> float:
        """Compute optimality gap as percentage."""
        if self.upper_bound is not None and self.upper_bound > 0:
            return max(0.0, (self.upper_bound - self.objective_value) / self.upper_bound * 100.0)
        if self.lower_bound is not None and self.lower_bound > 0:
            return max(0.0, (self.objective_value - self.lower_bound) / abs(self.lower_bound) * 100.0)
        return float('inf')


@dataclass
class Operator:
    """
    Specification for an operator/move to apply to a solution.
    
    Operators represent transformations (neighborhoods, heuristics, etc.)
    that can be applied to modify solutions.
    """
    
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    magnitude: str = "medium"  # tiny, small, medium, large
    
    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        if param_str:
            return f"{self.type}({param_str})"
        return self.type


@dataclass
class Candidate(Generic[SolutionType]):
    """
    A candidate solution with its operator and evaluation.
    
    Represents one step in the optimization, linking the operator applied,
    the resulting solution, and its evaluation.
    """
    
    operator: Operator
    solution: SolutionType
    evaluation: EvalResult
    loss: float = float('inf')
    
    def __lt__(self, other: 'Candidate') -> bool:
        """Enable sorting by loss."""
        return self.loss < other.loss


class Evaluator(ABC, Generic[SolutionType]):
    """
    Base class for solution evaluators.
    
    The evaluator is the source of truth - it performs all checks and measurements
    without relying on model opinions.
    """
    
    @abstractmethod
    def evaluate(self, problem: Problem, solution: SolutionType) -> EvalResult:
        """
        Evaluate a solution against the problem.
        
        Must check:
        - Feasibility (hard constraints)
        - Objective value
        - Bounds (if available)
        - Performance metrics
        - Quality scores
        """
        pass
    
    def compute_loss(self, eval_result: EvalResult, weights: Dict[str, float]) -> float:
        """
        Compute scalar loss from evaluation result.
        
        Default implementation:
        - Infeasible solutions get huge penalty
        - Otherwise weighted sum of normalized metrics
        """
        if not eval_result.feasible:
            return 1e9
        
        loss = 0.0
        
        # Gap contribution
        if 'gap' in weights and eval_result.gap_pct < float('inf'):
            loss += weights['gap'] * (eval_result.gap_pct / 100.0)
        
        # Runtime contribution (normalized)
        if 'runtime' in weights:
            loss += weights['runtime'] * (eval_result.runtime_ms / 1000.0)
        
        # Memory contribution (normalized)
        if 'memory' in weights:
            loss += weights['memory'] * (eval_result.memory_mb / 1024.0)
        
        # Violation contribution
        if 'violation' in weights:
            violations = sum(1 for v in eval_result.constraints_satisfied.values() if not v)
            loss += weights['violation'] * violations
        
        # Quality score contributions
        for key, weight in weights.items():
            if key in eval_result.quality_scores:
                # Assume quality scores are in [0, 1] where 0 is best
                loss += weight * eval_result.quality_scores[key]
        
        return loss


class Proposer(ABC, Generic[SolutionType]):
    """
    Base class for operator proposers.
    
    The proposer generates candidate operators to apply, potentially using
    a reasoning model or adaptive heuristics.
    """
    
    @abstractmethod
    def propose_operators(
        self,
        problem: Problem,
        current_solution: SolutionType,
        history: List[Candidate],
        trust_region: float,
        k: int = 6
    ) -> List[Operator]:
        """
        Propose k operators to try next.
        
        Args:
            problem: The problem instance
            current_solution: Current best solution
            history: Previous candidates tried
            trust_region: Current trust region size
            k: Number of operators to propose
            
        Returns:
            List of operators to evaluate
        """
        pass


class Critic(ABC):
    """
    Base class for critics that judge and guide the search.
    
    The critic reads evaluation results and provides guidance for the next
    steps, but its opinions are advisory - only measured improvements count.
    """
    
    @abstractmethod
    def select_best(
        self,
        candidates: List[Candidate],
        context: Dict[str, Any]
    ) -> Candidate:
        """
        Select the best candidate from a batch.
        
        Default behavior is to select by lowest loss, but critics can
        use additional logic (diversity, exploration, etc.)
        """
        pass
    
    @abstractmethod
    def analyze_progress(
        self,
        history: List[Candidate],
        current_best: Candidate
    ) -> Dict[str, Any]:
        """
        Analyze optimization progress and provide insights.
        
        Returns a dictionary with analysis results that can guide
        trust region updates and stopping decisions.
        """
        pass

