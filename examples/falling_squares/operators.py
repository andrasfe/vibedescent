"""Operators for modifying Falling Squares solver configurations."""

from __future__ import annotations

from typing import Any

from vibedescent.core import Operator
from .problem import FallingSquaresSolution


class FallingSquaresOperators:
    """Applies operator transformations to solutions."""

    @staticmethod
    def apply(operator: Operator, solution: FallingSquaresSolution) -> FallingSquaresSolution:
        new_solution = solution.copy()

        if operator.type == "set_strategy":
            strategy = operator.params.get("strategy")
            if strategy:
                new_solution.strategy = strategy
                if strategy == "bucket_grid" and "bucket_size" not in new_solution.params:
                    new_solution.params["bucket_size"] = 64
                if strategy == "block_dp" and "block_size" not in new_solution.params:
                    new_solution.params["block_size"] = 64
        elif operator.type == "toggle_lazy":
            current = new_solution.params.get("lazy", False)
            new_solution.params["lazy"] = not current
            if not new_solution.strategy.startswith("segment_tree"):
                new_solution.strategy = "segment_tree_lazy"
        elif operator.type == "reset_params":
            new_solution.params = {}
        elif operator.type == "adjust_param":
            param = operator.params.get("param")
            delta = operator.params.get("delta", 0)
            minimum = operator.params.get("min", 1)
            maximum = operator.params.get("max", 4096)
            target_strategy = operator.params.get("strategy")
            if param and (not target_strategy or target_strategy == new_solution.strategy):
                current = new_solution.params.get(param, operator.params.get("default", minimum))
                new_value = int(max(minimum, min(maximum, current + delta)))
                new_solution.params[param] = new_value

        return new_solution

