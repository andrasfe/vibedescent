"""
Configuration system for Vibe Descent objectives.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path


@dataclass
class ObjectiveConfig:
    """
    Configuration for a Vibe Descent objective.
    
    Defines hard constraints, loss weights, trust region parameters,
    and stopping conditions.
    """
    
    # Hard constraints (must be satisfied)
    hard_constraints: List[str] = field(default_factory=list)
    
    # Loss weights for different metrics
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Trust region configuration
    trust_region: Dict[str, float] = field(default_factory=dict)
    
    # Stopping conditions
    stop: Dict[str, Any] = field(default_factory=dict)
    
    # Targets for metrics
    targets: Dict[str, float] = field(default_factory=dict)
    
    # Review/feedback configuration
    review: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ObjectiveConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            hard_constraints=data.get('hard_constraints', []),
            weights=data.get('weights', {}),
            trust_region=data.get('trust_region', {}),
            stop=data.get('stop', {}),
            targets=data.get('targets', {}),
            review=data.get('review', {}),
            metadata=data.get('metadata', {})
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            'hard_constraints': self.hard_constraints,
            'weights': self.weights,
            'trust_region': self.trust_region,
            'stop': self.stop,
            'targets': self.targets,
            'review': self.review,
            'metadata': self.metadata,
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hard_constraints': self.hard_constraints,
            'weights': self.weights,
            'trust_region': self.trust_region,
            'stop': self.stop,
            'targets': self.targets,
            'review': self.review,
            'metadata': self.metadata,
        }
    
    @classmethod
    def default_knapsack(cls) -> 'ObjectiveConfig':
        """Create default configuration for knapsack problem."""
        return cls(
            hard_constraints=['feasible', 'no_violations'],
            weights={
                'gap': 0.70,
                'runtime': 0.20,
                'memory': 0.05,
                'violation': 0.05,
            },
            trust_region={
                'min': 0.5,
                'max': 2.0,
                'init': 1.0,
                'expand': 1.2,
                'shrink': 0.8,
            },
            stop={
                'target_gap_pct': 0.5,
                'patience': 3,
                'max_iters': 30,
            },
            targets={
                'gap_pct': 1.0,
                'runtime_ms': 1000.0,
            }
        )
    
    @classmethod
    def default_tsp(cls) -> 'ObjectiveConfig':
        """Create default configuration for TSP problem."""
        return cls(
            hard_constraints=['valid_tour', 'all_cities_visited'],
            weights={
                'gap': 0.75,
                'runtime': 0.15,
                'memory': 0.05,
                'violation': 0.05,
            },
            trust_region={
                'min': 0.5,
                'max': 2.0,
                'init': 1.0,
                'expand': 1.15,
                'shrink': 0.85,
            },
            stop={
                'target_gap_pct': 1.0,
                'patience': 5,
                'max_iters': 50,
            },
            targets={
                'gap_pct': 2.0,
                'runtime_ms': 5000.0,
            }
        )
    
    def get_weight(self, metric: str, default: float = 0.0) -> float:
        """Get weight for a metric with default."""
        return self.weights.get(metric, default)
    
    def get_target(self, metric: str, default: float = float('inf')) -> float:
        """Get target for a metric with default."""
        return self.targets.get(metric, default)

