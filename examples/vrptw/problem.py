"""VRPTW problem definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import math
import random

from vibedescent.core import Problem, Solution


@dataclass
class VRPTWCustomer:
    idx: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float


@dataclass
class VRPTWInstance(Problem):
    name: str
    vehicle_capacity: float
    depot: VRPTWCustomer
    customers: List[VRPTWCustomer]
    travel_matrix: List[List[float]]

    def get_context(self) -> Dict[str, Any]:
        return {
            "num_customers": len(self.customers),
            "vehicle_capacity": self.vehicle_capacity,
            "max_due_time": max(c.due_time for c in self.customers),
        }


@dataclass
class VRPTWSolution(Solution):
    routes: List[List[int]] = field(default_factory=list)

    def copy(self) -> VRPTWSolution:
        return VRPTWSolution(routes=[route[:] for route in self.routes])

    def to_dict(self) -> Dict[str, Any]:
        return {"routes": self.routes}


def generate_random_instance(
    name: str = "vrptw-rand",
    num_customers: int = 25,
    capacity: float = 40.0,
    coord_range: Tuple[int, int] = (0, 100),
    demand_range: Tuple[int, int] = (1, 10),
    time_window: Tuple[int, int] = (0, 200),
    service_time: Tuple[int, int] = (5, 15),
    seed: Optional[int] = None,
) -> VRPTWInstance:
    rng = random.Random(seed)
    customers: List[VRPTWCustomer] = []

    depot = VRPTWCustomer(
        idx=0,
        x=(coord_range[0] + coord_range[1]) / 2,
        y=(coord_range[0] + coord_range[1]) / 2,
        demand=0,
        ready_time=time_window[0],
        due_time=time_window[1],
        service_time=0,
    )

    for i in range(1, num_customers + 1):
        x = rng.uniform(*coord_range)
        y = rng.uniform(*coord_range)
        demand = rng.randint(*demand_range)
        ready = rng.uniform(time_window[0], time_window[1] * 0.7)
        due = ready + rng.uniform(20, 60)
        due = min(due, time_window[1])
        service = rng.uniform(*service_time)
        customers.append(
            VRPTWCustomer(
                idx=i,
                x=x,
                y=y,
                demand=demand,
                ready_time=ready,
                due_time=due,
                service_time=service,
            )
        )

    all_nodes = [depot] + customers
    travel_matrix = []
    for i in range(len(all_nodes)):
        row = []
        for j in range(len(all_nodes)):
            dx = all_nodes[i].x - all_nodes[j].x
            dy = all_nodes[i].y - all_nodes[j].y
            row.append(math.hypot(dx, dy))
        travel_matrix.append(row)

    return VRPTWInstance(
        name=name,
        vehicle_capacity=capacity,
        depot=depot,
        customers=customers,
        travel_matrix=travel_matrix,
    )

