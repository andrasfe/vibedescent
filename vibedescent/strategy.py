"""
Generic strategy registration utilities for Vibe Descent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


StrategyRunner = Callable[[Any, Optional[Dict[str, Any]], bool], Any]


@dataclass
class Strategy:
    """Metadata describing a runnable strategy."""

    name: str
    runner: StrategyRunner
    description: str = ""
    supports_params: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyRegistry:
    """Registry that keeps track of available strategies."""

    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}

    def register(self, strategy: Strategy) -> None:
        if strategy.name in self._strategies:
            raise ValueError(f"Strategy '{strategy.name}' already registered")
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Strategy:
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' is not registered")
        return self._strategies[name]

    def list_names(self) -> List[str]:
        return list(self._strategies.keys())

    def run(self, name: str, positions: Any, params: Optional[Dict[str, Any]] = None, return_stats: bool = False):
        strategy = self.get(name)
        return strategy.runner(positions, params, return_stats)


