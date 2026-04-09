"""Uninformed search algorithms using distance-only edge costs.

Algorithms:
- Breadth-first search (BFS)              frontier: FIFO queue, f = depth
- Uniform cost search                     frontier: min-heap on g(n)
- Depth-first search (DFS)                frontier: LIFO stack
- Depth Limited Search                    DFS with depth limit
- Iterative Deepening Search              repeated DLS
- Bidirectional Search                    simultaneous BFS from start/goal

All return a unified SearchResult for easy comparison.
"""

from __future__ import annotations

from collections import deque
import heapq
from time import perf_counter
from typing import Dict, Hashable, List, Optional, Set, Tuple

import networkx as nx

from .common import SearchResult, edge_step_cost, compute_path_cost, reconstruct_path


def breadth_first_search(G: nx.MultiDiGraph, start: Hashable, goal: Hashable) -> SearchResult:
    t0 = perf_counter()
    frontier = deque([start])
    parents: Dict[Hashable, Optional[Hashable]] = {start: None}
    visited: Set[Hashable] = {start}
    expanded = 0
    max_frontier = 1
    expanded_nodes: List[Hashable] = []

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        u = frontier.popleft()
        expanded += 1
        expanded_nodes.append(u)
        if u == goal:
            path = reconstruct_path(parents, goal)
            cost = compute_path_cost(G, path)
            return SearchResult(
                algorithm_name="Breadth-first search (BFS)",
                path=path,
                path_found=True,
                total_path_cost=cost,
                nodes_expanded=expanded,
                execution_time=perf_counter() - t0,
                max_frontier_size=max_frontier,
                path_length=len(path),
                visited_count=len(visited),
                start_node=start,
                goal_node=goal,
                expanded_nodes=expanded_nodes,
            )
        for _, v, _, data in G.out_edges(u, keys=True, data=True):
            if v not in visited:
                visited.add(v)
                parents[v] = u
                frontier.append(v)

    return SearchResult(
        "Breadth-first search (BFS)",
        [],
        False,
        float("inf"),
        expanded,
        perf_counter() - t0,
        max_frontier,
        0,
        len(visited),
        start_node=start,
        goal_node=goal,
        expanded_nodes=expanded_nodes,
    )


def uniform_cost_search(G: nx.MultiDiGraph, start: Hashable, goal: Hashable) -> SearchResult:
    t0 = perf_counter()
    frontier: List[Tuple[float, Hashable]] = [(0.0, start)]
    parents: Dict[Hashable, Optional[Hashable]] = {start: None}
    g_scores: Dict[Hashable, float] = {start: 0.0}
    expanded = 0
    max_frontier = 1
    visited = 0
    expanded_nodes: List[Hashable] = []

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        g_u, u = heapq.heappop(frontier)
        visited += 1
        expanded_nodes.append(u)
        if u == goal:
            path = reconstruct_path(parents, goal)
            cost = compute_path_cost(G, path)
            return SearchResult(
                algorithm_name="Uniform cost search",
                path=path,
                path_found=True,
                total_path_cost=cost,
                nodes_expanded=expanded,
                execution_time=perf_counter() - t0,
                max_frontier_size=max_frontier,
                path_length=len(path),
                visited_count=visited,
                start_node=start,
                goal_node=goal,
                expanded_nodes=expanded_nodes,
            )

        expanded += 1

        for _, v, _, data in G.out_edges(u, keys=True, data=True):
            step = edge_step_cost(data)
            g_v = g_u + step
            if v not in g_scores or g_v < g_scores[v]:
                g_scores[v] = g_v
                parents[v] = u
                heapq.heappush(frontier, (g_v, v))

    return SearchResult(
        "Uniform cost search",
        [],
        False,
        float("inf"),
        expanded,
        perf_counter() - t0,
        max_frontier,
        0,
        visited,
        start_node=start,
        goal_node=goal,
        expanded_nodes=expanded_nodes,
    )


def depth_first_search(G: nx.MultiDiGraph, start: Hashable, goal: Hashable) -> SearchResult:
    t0 = perf_counter()
    stack = [start]
    parents: Dict[Hashable, Optional[Hashable]] = {start: None}
    visited: Set[Hashable] = {start}
    expanded = 0
    max_frontier = 1
    expanded_nodes: List[Hashable] = []

    while stack:
        max_frontier = max(max_frontier, len(stack))
        u = stack.pop()
        expanded += 1
        expanded_nodes.append(u)
        if u == goal:
            path = reconstruct_path(parents, goal)
            cost = compute_path_cost(G, path)
            return SearchResult(
                algorithm_name="Depth-first search (DFS)",
                path=path,
                path_found=True,
                total_path_cost=cost,
                nodes_expanded=expanded,
                execution_time=perf_counter() - t0,
                max_frontier_size=max_frontier,
                path_length=len(path),
                visited_count=len(visited),
                start_node=start,
                goal_node=goal,
                expanded_nodes=expanded_nodes,
            )

        for _, v, _, data in reversed(list(G.out_edges(u, keys=True, data=True))):
            if v not in visited:
                visited.add(v)
                parents[v] = u
                stack.append(v)

    return SearchResult(
        "Depth-first search (DFS)",
        [],
        False,
        float("inf"),
        expanded,
        perf_counter() - t0,
        max_frontier,
        0,
        len(visited),
        start_node=start,
        goal_node=goal,
        expanded_nodes=expanded_nodes,
    )


def depth_limited_search(G: nx.MultiDiGraph, start: Hashable, goal: Hashable, limit: int) -> SearchResult:
    t0 = perf_counter()
    expanded = 0
    max_frontier = 0
    cutoff_occurred = False
    visited: Set[Hashable] = set()
    expanded_nodes: List[Hashable] = []

    def recursive_dls(node: Hashable, depth: int, parents: Dict[Hashable, Optional[Hashable]]) -> Optional[List[Hashable]]:
        nonlocal expanded, max_frontier, cutoff_occurred
        visited.add(node)
        expanded_nodes.append(node)
        max_frontier = max(max_frontier, depth + 1)
        if node == goal:
            return reconstruct_path(parents, node)
        if depth == limit:
            cutoff_occurred = True
            return None
        expanded += 1
        for _, v, _, _ in G.out_edges(node, keys=True, data=True):
            if v not in parents:  # avoids cycles on current path
                parents[v] = node
                res = recursive_dls(v, depth + 1, parents)
                if res is not None:
                    return res
        return None

    parents: Dict[Hashable, Optional[Hashable]] = {start: None}
    path = recursive_dls(start, 0, parents)
    cost = compute_path_cost(G, path) if path else float("inf")
    return SearchResult(
        algorithm_name="Depth Limited Search",
        path=path or [],
        path_found=bool(path),
        total_path_cost=cost,
        nodes_expanded=expanded,
        execution_time=perf_counter() - t0,
        max_frontier_size=max_frontier,
        path_length=len(path or []),
        visited_count=len(visited),
        depth_reached=limit,
        cutoff_occurred=cutoff_occurred,
        start_node=start,
        goal_node=goal,
        expanded_nodes=expanded_nodes,
    )


def iterative_deepening_search(G: nx.MultiDiGraph, start: Hashable, goal: Hashable, max_depth: int = 50) -> SearchResult:
    t0 = perf_counter()
    total_expanded = 0
    max_frontier = 0
    visited_total = 0
    expanded_nodes: List[Hashable] = []

    for limit in range(max_depth + 1):
        res = depth_limited_search(G, start, goal, limit)
        total_expanded += res.nodes_expanded
        max_frontier = max(max_frontier, res.max_frontier_size)
        visited_total += res.visited_count
        if res.expanded_nodes:
            expanded_nodes.extend(res.expanded_nodes)
        if res.path_found:
            res.algorithm_name = "Iterative Deepening Search"
            res.nodes_expanded = total_expanded
            res.execution_time = perf_counter() - t0
            res.max_frontier_size = max_frontier
            res.visited_count = visited_total
            res.depth_reached = limit
            res.start_node = start
            res.goal_node = goal
            res.expanded_nodes = expanded_nodes
            return res
        if not res.cutoff_occurred:
            break

    return SearchResult(
        algorithm_name="Iterative Deepening Search",
        path=[],
        path_found=False,
        total_path_cost=float("inf"),
        nodes_expanded=total_expanded,
        execution_time=perf_counter() - t0,
        max_frontier_size=max_frontier,
        path_length=0,
        visited_count=visited_total,
        depth_reached=max_depth,
        cutoff_occurred=True,
        start_node=start,
        goal_node=goal,
        expanded_nodes=expanded_nodes,
    )


def bidirectional_search(G: nx.MultiDiGraph, start: Hashable, goal: Hashable) -> SearchResult:
    t0 = perf_counter()
    if start == goal:
        return SearchResult(
            algorithm_name="Bidirectional Search",
            path=[start],
            path_found=True,
            total_path_cost=0.0,
            nodes_expanded=0,
            execution_time=0.0,
            max_frontier_size=1,
            path_length=1,
            visited_count=1,
            start_node=start,
            goal_node=goal,
            expanded_nodes=[start],
        )

    frontier_f = deque([start])
    frontier_b = deque([goal])
    parents_f: Dict[Hashable, Optional[Hashable]] = {start: None}
    parents_b: Dict[Hashable, Optional[Hashable]] = {goal: None}
    visited_f: Set[Hashable] = {start}
    visited_b: Set[Hashable] = {goal}
    expanded = 0
    max_frontier = 2
    expanded_nodes: List[Hashable] = []

    def _merge(meet: Hashable) -> List[Hashable]:
        path_f = reconstruct_path(parents_f, meet)
        path_b = reconstruct_path(parents_b, meet)
        path_b = path_b[::-1]
        return path_f[:-1] + path_b

    while frontier_f and frontier_b:
        max_frontier = max(max_frontier, len(frontier_f) + len(frontier_b))

        # Expand one step forward
        u = frontier_f.popleft()
        expanded += 1
        expanded_nodes.append(u)
        for _, v, _, _ in G.out_edges(u, keys=True, data=True):
            if v not in visited_f:
                visited_f.add(v)
                parents_f[v] = u
                if v in visited_b:
                    path = _merge(v)
                    cost = compute_path_cost(G, path)
                    return SearchResult(
                        algorithm_name="Bidirectional Search",
                        path=path,
                        path_found=True,
                        total_path_cost=cost,
                        nodes_expanded=expanded,
                        execution_time=perf_counter() - t0,
                        max_frontier_size=max_frontier,
                        path_length=len(path),
                        visited_count=len(visited_f) + len(visited_b),
                        meeting_node=v,
                        start_node=start,
                        goal_node=goal,
                        expanded_nodes=expanded_nodes,
                    )
                frontier_f.append(v)

        # Expand one step backward (reverse direction; for directed graphs we consider in_edges)
        u = frontier_b.popleft()
        expanded += 1
        expanded_nodes.append(u)
        for v, _, _ in G.in_edges(u, keys=True):
            if v not in visited_b:
                visited_b.add(v)
                parents_b[v] = u
                if v in visited_f:
                    path = _merge(v)
                    cost = compute_path_cost(G, path)
                    return SearchResult(
                        algorithm_name="Bidirectional Search",
                        path=path,
                        path_found=True,
                        total_path_cost=cost,
                        nodes_expanded=expanded,
                        execution_time=perf_counter() - t0,
                        max_frontier_size=max_frontier,
                        path_length=len(path),
                        visited_count=len(visited_f) + len(visited_b),
                        meeting_node=v,
                        start_node=start,
                        goal_node=goal,
                        expanded_nodes=expanded_nodes,
                    )
                frontier_b.append(v)

    return SearchResult(
        "Bidirectional Search",
        [],
        False,
        float("inf"),
        expanded,
        perf_counter() - t0,
        max_frontier,
        0,
        len(visited_f) + len(visited_b),
        start_node=start,
        goal_node=goal,
        expanded_nodes=expanded_nodes,
    )
