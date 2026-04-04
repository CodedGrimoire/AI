"""
Classic search algorithms on an OSMnx road graph with a custom edge cost.

Prereqs (already in your setup):
- G is a NetworkX/OSMnx MultiDiGraph
- Each edge has data['custom_cost'] (float)
- A Euclidean heuristic function is available (heuristic implemented below)

This module implements from-scratch versions of:
- Dijkstra / Uniform Cost Search
- Greedy Best-First Search
- A* Search
- Weighted A* Search

All algorithms use heapq, avoid NetworkX shortest_path helpers, and return:
- path (list of nodes)
- total path cost (g-cost)
- nodes expanded
- execution time (seconds)

You can import these functions or run this file directly for a small demo.
"""

from __future__ import annotations


import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Data structures & low-level helpers
# ---------------------------------------------------------------------------
@dataclass(order=True)
class PQItem:
    priority: float
    node: Hashable = field(compare=False)
    g_cost: float = field(compare=False)
    parent: Optional[Hashable] = field(compare=False)


def reconstruct_path(parents: Dict[Hashable, Hashable], goal: Hashable) -> List[Hashable]:
    path: List[Hashable] = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents.get(cur)
    return list(reversed(path))


def edge_step_cost(data: Dict[str, Any]) -> float:
    """Read the custom edge cost with a safe default."""
    return float(data.get("custom_cost", 1.0))


def compute_path_cost(G: nx.MultiDiGraph, path: List[Hashable]) -> float:
    """Compute total cost along a path using the minimum custom_cost over parallel edges."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_datas = G.get_edge_data(u, v)
        if not edge_datas:
            return float("inf")
        step = min(edge_step_cost(d) for d in edge_datas.values())
        total += step
    return total


def euclidean_heuristic(G: nx.MultiDiGraph, n1: Hashable, n2: Hashable, scale: float = 1.0) -> float:
    """Euclidean distance between two nodes (using x, y), optionally scaled."""
    x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
    x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]
    return scale * ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Generic search engine
# ---------------------------------------------------------------------------
def _run_search(
    G: nx.MultiDiGraph,
    start: Hashable,
    goal: Hashable,
    priority_fn,
    heuristic_fn,
) -> Tuple[List[Hashable], float, int, float]:
    t0 = time.time()
    frontier: List[PQItem] = []
    start_h = heuristic_fn(start)
    heapq.heappush(frontier, PQItem(priority_fn(0.0, start_h), start, 0.0, None))

    parents = {start: None}
    g_scores = {start: 0.0}
    expanded = 0

    while frontier:
        current = heapq.heappop(frontier)
        u = current.node
        g_u = current.g_cost

        if u == goal:
            elapsed = time.time() - t0
            path = reconstruct_path(parents, goal)
            return path, g_u, expanded, elapsed

        expanded += 1

        for _, v, _, data in G.out_edges(u, keys=True, data=True):
            step = edge_step_cost(data)
            g_v = g_u + step
            if v not in g_scores or g_v < g_scores[v]:
                g_scores[v] = g_v
                parents[v] = u
                h_v = heuristic_fn(v)
                heapq.heappush(frontier, PQItem(priority_fn(g_v, h_v), v, g_v, u))

    return [], float("inf"), expanded, time.time() - t0  # no path found


# ---------------------------------------------------------------------------
# Algorithm wrappers
# ---------------------------------------------------------------------------
def dijkstra_search(G, start, goal, heuristic=lambda n: 0.0):
    return _run_search(G, start, goal, priority_fn=lambda g, h: g, heuristic_fn=heuristic)


def greedy_best_first_search(G, start, goal, heuristic):
    return _run_search(G, start, goal, priority_fn=lambda g, h: h, heuristic_fn=heuristic)


def a_star_search(G, start, goal, heuristic):
    return _run_search(G, start, goal, priority_fn=lambda g, h: g + h, heuristic_fn=heuristic)


def weighted_a_star_search(G, start, goal, heuristic, w: float = 1.0):
    return _run_search(G, start, goal, priority_fn=lambda g, h: g + w * h, heuristic_fn=heuristic)


# ---------------------------------------------------------------------------
# Demo / comparison runner (optional)
# ---------------------------------------------------------------------------
def run_demo(G: nx.MultiDiGraph, start: Hashable, goal: Hashable, heuristic_scale: float = 1.0):
    h_fn = lambda n: euclidean_heuristic(G, n, goal, scale=heuristic_scale)

    runs = [
        ("Dijkstra", lambda: dijkstra_search(G, start, goal, heuristic=lambda n: 0.0)),
        ("Greedy", lambda: greedy_best_first_search(G, start, goal, h_fn)),
        ("A*", lambda: a_star_search(G, start, goal, h_fn)),
        ("WA* w=1.5", lambda: weighted_a_star_search(G, start, goal, h_fn, w=1.5)),
    ]

    results = []
    for name, fn in runs:
        path, g_cost, expanded, elapsed = fn()
        total_cost = compute_path_cost(G, path)
        results.append((name, total_cost, expanded, elapsed, len(path)))

    print("\nAlgorithm Comparison")
    print("{:<12} {:>12} {:>12} {:>10} {:>10}".format("Algorithm", "Path Cost", "Expanded", "Time(s)", "Path len"))
    for name, cost, expd, t, plen in results:
        print(f"{name:<12} {cost:>12.2f} {expd:>12d} {t:>10.4f} {plen:>10d}")


if __name__ == "__main__":
    import osmnx as ox

    print("[info] Demo: building a small graph (replace with your own G/start/goal)")
    G_demo = ox.graph_from_point((23.746, 90.376), dist=1200, network_type="drive")
    G_demo = ox.project_graph(G_demo)

    # If custom_cost is missing, fall back to length for the demo
    for _, _, _, d in G_demo.edges(keys=True, data=True):
        d.setdefault("custom_cost", float(d.get("length", 1.0)))

    nodes = list(G_demo.nodes)
    start_node, goal_node = nodes[0], nodes[-1]
    run_demo(G_demo, start_node, goal_node, heuristic_scale=1.0)
