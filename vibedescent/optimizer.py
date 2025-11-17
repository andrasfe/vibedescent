"""
Optimizer state and trust region management.
"""

from dataclasses import dataclass, field
from typing import Dict
from collections import defaultdict


@dataclass
class TrustRegion:
    """
    Trust region configuration and state.
    
    Controls the step size / magnitude of changes per iteration,
    adapting based on progress.
    """
    
    min_size: float = 0.5
    max_size: float = 2.0
    initial_size: float = 1.0
    expand_factor: float = 1.2
    shrink_factor: float = 0.8
    
    current_size: float = field(init=False)
    
    def __post_init__(self):
        self.current_size = self.initial_size
    
    def expand(self) -> float:
        """Expand trust region after successful step."""
        self.current_size = min(self.max_size, self.current_size * self.expand_factor)
        return self.current_size
    
    def shrink(self) -> float:
        """Shrink trust region after failed step."""
        self.current_size = max(self.min_size, self.current_size * self.shrink_factor)
        return self.current_size
    
    def reset(self):
        """Reset to initial size."""
        self.current_size = self.initial_size


@dataclass
class OptimizerState:
    """
    Maintains optimizer state across iterations.
    
    Tracks best solutions, momentum, patience, and trust region.
    """
    
    # Trust region
    trust_region: TrustRegion = field(default_factory=TrustRegion)
    
    # Best tracking
    best_loss: float = float('inf')
    best_objective: float = 0.0
    best_gap: float = float('inf')
    
    # Momentum: track successful operator types
    operator_momentum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    # Patience for early stopping
    patience_left: int = 3
    patience_max: int = 3
    
    # Iteration tracking
    iterations: int = 0
    total_evaluations: int = 0
    
    # Improvement tracking
    improvements: int = 0
    stagnation_count: int = 0
    
    def update_on_improvement(self, loss: float, objective: float, gap: float):
        """Update state after an improving step."""
        self.best_loss = loss
        self.best_objective = objective
        self.best_gap = gap
        self.improvements += 1
        self.stagnation_count = 0
        self.patience_left = self.patience_max
        self.trust_region.expand()
    
    def update_on_stagnation(self):
        """Update state after a non-improving step."""
        self.stagnation_count += 1
        self.patience_left -= 1
        self.trust_region.shrink()
    
    def should_stop(
        self,
        target_gap: float | None = None,
        max_iterations: int | None = None,
        min_iterations: int = 1,
    ) -> bool:
        """
        Check if optimization should stop.
        
        Args:
            target_gap: Target optimality gap (stop if reached)
            max_iterations: Maximum iterations (stop if reached)
            min_iterations: Minimum iterations to run before stopping
        
        Returns:
            True if should stop
        """
        if self.iterations < min_iterations:
            return False
        
        # Patience exhausted
        if self.patience_left <= 0:
            return True
        
        # Target gap reached
        if target_gap is not None and self.best_gap <= target_gap:
            return True
        
        # Max iterations reached
        if max_iterations is not None and self.iterations >= max_iterations:
            return True
        
        return False
    
    def get_status(self) -> Dict[str, any]:
        """Get current status as dictionary."""
        return {
            'iteration': self.iterations,
            'evaluations': self.total_evaluations,
            'best_loss': self.best_loss,
            'best_objective': self.best_objective,
            'best_gap': self.best_gap,
            'trust_region': self.trust_region.current_size,
            'patience_left': self.patience_left,
            'improvements': self.improvements,
            'stagnation_count': self.stagnation_count,
        }

