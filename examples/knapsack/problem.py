"""
Knapsack problem definition and solution representation.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional
import random

from vibedescent.core import Problem, Solution


@dataclass
class Item:
    """An item with value and weight."""
    
    index: int
    value: int
    weight: int
    
    def density(self) -> float:
        """Value per unit weight."""
        return self.value / self.weight if self.weight > 0 else float('inf')


@dataclass
class KnapsackSolution(Solution):
    """
    Solution to knapsack problem.
    
    Represented as a set of item indices that are included.
    """
    
    picked: Set[int] = field(default_factory=set)
    
    def copy(self) -> 'KnapsackSolution':
        """Create a deep copy."""
        return KnapsackSolution(picked=set(self.picked))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'picked': sorted(list(self.picked)),
            'size': len(self.picked)
        }
    
    def weight(self, problem: 'KnapsackProblem') -> int:
        """Total weight of picked items."""
        return sum(problem.items[i].weight for i in self.picked)
    
    def value(self, problem: 'KnapsackProblem') -> int:
        """Total value of picked items."""
        return sum(problem.items[i].value for i in self.picked)
    
    def feasible(self, problem: 'KnapsackProblem') -> bool:
        """Check if solution is feasible (within capacity)."""
        return self.weight(problem) <= problem.capacity
    
    def add_item(self, item_idx: int):
        """Add an item to the solution."""
        self.picked.add(item_idx)
    
    def remove_item(self, item_idx: int):
        """Remove an item from the solution."""
        self.picked.discard(item_idx)
    
    def has_item(self, item_idx: int) -> bool:
        """Check if item is in solution."""
        return item_idx in self.picked


@dataclass
class KnapsackProblem(Problem):
    """
    0/1 Knapsack problem instance.
    """
    
    capacity: int
    items: List[Item]
    
    def __post_init__(self):
        """Validate problem instance."""
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not self.items:
            raise ValueError("Must have at least one item")
    
    def get_context(self) -> Dict[str, Any]:
        """Get problem context for operators."""
        return {
            'n_items': len(self.items),
            'capacity': self.capacity,
            'total_weight': sum(item.weight for item in self.items),
            'total_value': sum(item.value for item in self.items),
            'avg_weight': sum(item.weight for item in self.items) / len(self.items),
            'avg_value': sum(item.value for item in self.items) / len(self.items),
        }
    
    @classmethod
    def random(
        cls,
        name: str,
        n: int,
        capacity_ratio: float = 0.4,
        value_range: tuple = (1, 99),
        weight_range: tuple = (1, 99),
        seed: Optional[int] = None
    ) -> 'KnapsackProblem':
        """
        Generate a random knapsack instance.
        
        Args:
            name: Problem name
            n: Number of items
            capacity_ratio: Capacity as fraction of total weight
            value_range: Min and max value for items
            weight_range: Min and max weight for items
            seed: Random seed
        
        Returns:
            Random knapsack problem
        """
        rng = random.Random(seed)
        
        items = []
        total_weight = 0
        
        for i in range(n):
            weight = rng.randint(*weight_range)
            value = rng.randint(*value_range)
            items.append(Item(index=i, value=value, weight=weight))
            total_weight += weight
        
        capacity = max(1, int(total_weight * capacity_ratio))
        
        return cls(name=name, capacity=capacity, items=items)
    
    @classmethod
    def from_file(cls, name: str, filepath: str) -> 'KnapsackProblem':
        """
        Load knapsack instance from file.
        
        Expected format:
        Line 1: n capacity
        Lines 2..n+1: value weight
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        n, capacity = map(int, lines[0].split())
        items = []
        
        for i, line in enumerate(lines[1:n+1]):
            value, weight = map(int, line.split())
            items.append(Item(index=i, value=value, weight=weight))
        
        return cls(name=name, capacity=capacity, items=items)

