"""Heuristic solvers for VRPTW."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import random

from .problem import VRPTWInstance, VRPTWSolution, VRPTWCustomer


def _is_feasible_route(
    instance: VRPTWInstance,
    route: List[int],
    capacity: float,
) -> bool:
    total_demand = 0.0
    for cid in route:
        total_demand += instance.customers[cid - 1].demand
        if total_demand > capacity:
            return False
    return True


def _evaluate_route_schedule(instance: VRPTWInstance, route: List[int]) -> float:
    time = 0.0
    last = 0
    lateness = 0.0
    for cid in route:
        cust = instance.customers[cid - 1]
        travel = instance.travel_matrix[last][cust.idx]
        time += travel
        if time < cust.ready_time:
            time = cust.ready_time
        if time > cust.due_time:
            lateness += time - cust.due_time
        time += cust.service_time
        last = cust.idx
    time += instance.travel_matrix[last][0]
    return lateness


def nearest_time_window_heuristic(
    instance: VRPTWInstance,
    params: Optional[Dict[str, Any]] = None,
) -> VRPTWSolution:
    remaining = set(c.idx for c in instance.customers)
    routes: List[List[int]] = []
    capacity = instance.vehicle_capacity

    while remaining:
        route = []
        load = 0.0
        current = 0
        time = 0.0
        while True:
            feasible = []
            for cid in remaining:
                cust = instance.customers[cid - 1]
                demand = cust.demand
                if load + demand > capacity:
                    continue
                travel = instance.travel_matrix[current][cid]
                arrive = time + travel
                if arrive > cust.due_time:
                    continue
                wait = max(0.0, cust.ready_time - arrive)
                feasible.append((arrive + wait, travel, cid))
            if not feasible:
                break
            feasible.sort()
            _, travel, chosen = feasible[0]
            cust = instance.customers[chosen - 1]
            time += travel
            if time < cust.ready_time:
                time = cust.ready_time
            time += cust.service_time
            load += cust.demand
            route.append(chosen)
            remaining.remove(chosen)
            current = chosen
        if route:
            routes.append(route)
        else:
            # fallback if no feasible
            cid = remaining.pop()
            routes.append([cid])
    return VRPTWSolution(routes=routes)


def savings_heuristic(
    instance: VRPTWInstance,
    params: Optional[Dict[str, Any]] = None,
) -> VRPTWSolution:
    capacity = instance.vehicle_capacity
    routes = [[cust.idx] for cust in instance.customers]

    savings_list = []
    for i in range(len(instance.customers)):
        for j in range(i + 1, len(instance.customers)):
            ci = instance.customers[i]
            cj = instance.customers[j]
            saving = (
                instance.travel_matrix[0][ci.idx]
                + instance.travel_matrix[0][cj.idx]
                - instance.travel_matrix[ci.idx][cj.idx]
            )
            savings_list.append((saving, ci.idx, cj.idx))
    savings_list.sort(reverse=True)

    for _, i, j in savings_list:
        route_i = None
        route_j = None
        for route in routes:
            if route[0] == i:
                route_i = route
            if route[-1] == j:
                route_j = route
        if route_i is None or route_j is None or route_i == route_j:
            continue
        merged = route_j + route_i
        if _is_feasible_route(instance, merged, capacity):
            routes.remove(route_i)
            routes.remove(route_j)
            routes.append(merged)

    return VRPTWSolution(routes=routes)


def randomized_insertion_heuristic(
    instance: VRPTWInstance,
    params: Optional[Dict[str, Any]] = None,
) -> VRPTWSolution:
    rng = random.Random((params or {}).get("seed", 0))
    remaining = list(c.idx for c in instance.customers)
    rng.shuffle(remaining)
    routes: List[List[int]] = []
    capacity = instance.vehicle_capacity

    for cid in remaining:
        placed = False
        for route in routes:
            for pos in range(len(route) + 1):
                trial = route[:pos] + [cid] + route[pos:]
                if _is_feasible_route(instance, trial, capacity):
                    route[:] = trial
                    placed = True
                    break
            if placed:
                break
        if not placed:
            routes.append([cid])
    return VRPTWSolution(routes=routes)


def two_phase_heuristic(
    instance: VRPTWInstance,
    params: Optional[Dict[str, Any]] = None,
) -> VRPTWSolution:
    initial = nearest_time_window_heuristic(instance, params)
    rng = random.Random((params or {}).get("seed", 1))

    for _ in range(10):
        if len(initial.routes) < 2:
            break
        r1, r2 = rng.sample(range(len(initial.routes)), 2)
        if not initial.routes[r1] or not initial.routes[r2]:
            continue
        cid = initial.routes[r1].pop()
        initial.routes[r2].append(cid)
        if not _is_feasible_route(instance, initial.routes[r2], instance.vehicle_capacity):
            initial.routes[r2].pop()
            initial.routes[r1].append(cid)
    return initial

