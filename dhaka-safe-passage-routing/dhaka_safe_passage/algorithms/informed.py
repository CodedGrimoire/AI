"""Informed search algorithms using exponential heuristic.

Algorithms:
- Greedy best-first search        f(n) = h(n)
- A* search                       f(n) = g(n) + h(n)
- Weighted A*                     f(n) = g(n) + w*h(n)
"""

from __future__ import annotations

import heapq
from time import perf_counter
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import networkx as nx

from .common import SearchResult, edge_step_cost, compute_path_cost, reconstruct_path


HeuristicFn = Callable[[Hashable], float]


def _run_best_first(
    G: nx.MultiDiGraph,
    start: Hashable,
    goal: Hashable,
    eval_fn: Callable[[float, float], float],
    heuristic: HeuristicFn,
    *,
    algorithm_name: str,
    weight: Optional[float] = None,
) -> SearchResult:
    t0 = perf_counter()
    frontier: List[Tuple[float, Hashable, float]] = []
    start_h = heuristic(start)
    heapq.heappush(frontier, (eval_fn(0.0, start_h), start, 0.0))

    parents: Dict[Hashable, Optional[Hashable]] = {start: None}
    g_scores: Dict[Hashable, float] = {start: 0.0}
    expanded = 0
    max_frontier = 1
    visited_count = 0
    expanded_nodes: List[Hashable] = []

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        f_u, u, g_u = heapq.heappop(frontier)
        visited_count += 1
        expanded_nodes.append(u)

        if u == goal:
            path = reconstruct_path(parents, goal)
            cost = compute_path_cost(G, path)
            return SearchResult(
                algorithm_name=algorithm_name,
                path=path,
                path_found=True,
                total_path_cost=cost,
                nodes_expanded=expanded,
                execution_time=perf_counter() - t0,
                max_frontier_size=max_frontier,
                path_length=len(path),
                visited_count=visited_count,
                start_node=start,
                goal_node=goal,
                weight=weight,
                expanded_nodes=expanded_nodes,
            )

        expanded += 1

        for _, v, _, data in G.out_edges(u, keys=True, data=True):
            step = edge_step_cost(data)
            g_v = g_u + step
            if v not in g_scores or g_v < g_scores[v]:
                g_scores[v] = g_v
                parents[v] = u
                h_v = heuristic(v)
                heapq.heappush(frontier, (eval_fn(g_v, h_v), v, g_v))

    return SearchResult(
        algorithm_name=algorithm_name,
        path=[],
        path_found=False,
        total_path_cost=float("inf"),
        nodes_expanded=expanded,
        execution_time=perf_counter() - t0,
        max_frontier_size=max_frontier,
        path_length=0,
        visited_count=visited_count,
        start_node=start,
        goal_node=goal,
        weight=weight,
        expanded_nodes=expanded_nodes,
    )


def greedy_best_first_search(G: nx.MultiDiGraph, start: Hashable, goal: Hashable, heuristic: HeuristicFn) -> SearchResult:
    return _run_best_first(
        G,
        start,
        goal,
        eval_fn=lambda g, h: h,
        heuristic=heuristic,
        algorithm_name="Greedy best-first search",
    )


def a_star_search(G: nx.MultiDiGraph, start: Hashable, goal: Hashable, heuristic: HeuristicFn) -> SearchResult:
    return _run_best_first(
        G,
        start,
        goal,
        eval_fn=lambda g, h: g + h,
        heuristic=heuristic,
        algorithm_name="A* search",
    )


def weighted_a_star_search(
    G: nx.MultiDiGraph, start: Hashable, goal: Hashable, heuristic: HeuristicFn, w: float = 1.0
) -> SearchResult:
    return _run_best_first(
        G,
        start,
        goal,
        eval_fn=lambda g, h: g + w * h,
        heuristic=heuristic,
        algorithm_name="Weighted A*",
        weight=w,
    )
