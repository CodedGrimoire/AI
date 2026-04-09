"""Shared utilities and result structures for search algorithms."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple

import networkx as nx


@dataclass
class SearchResult:
    algorithm_name: str
    path: List[Hashable]
    path_found: bool
    total_path_cost: float
    nodes_expanded: int
    execution_time: float
    max_frontier_size: int
    path_length: int
    visited_count: int
    depth_reached: Optional[int] = None
    cutoff_occurred: Optional[bool] = None
    meeting_node: Optional[Hashable] = None
    weight: Optional[float] = None


def edge_step_cost(data: Dict[str, Any]) -> float:
    """Distance-only edge cost (custom_cost set by data pipeline)."""
    return float(data.get("custom_cost", 1.0))


def compute_path_cost(G: nx.MultiDiGraph, path: List[Hashable]) -> float:
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


def reconstruct_path(parents: Dict[Hashable, Hashable], meet: Hashable) -> List[Hashable]:
    path: List[Hashable] = []
    cur: Optional[Hashable] = meet
    while cur is not None:
        path.append(cur)
        cur = parents.get(cur)
    path.reverse()
    return path


def timed(fn):
    """Decorator to measure execution time (seconds)."""

    def wrapper(*args, **kwargs):
        start = time.time()
        out = fn(*args, **kwargs)
        return out, time.time() - start

    return wrapper

