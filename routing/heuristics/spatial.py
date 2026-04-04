"""Heuristic functions for routing."""

from __future__ import annotations

from typing import Hashable
import networkx as nx
import math


def euclidean_heuristic(G: nx.MultiDiGraph, n1: Hashable, n2: Hashable, scale: float = 1.0) -> float:
    """Straight-line distance between two nodes using (x, y) coordinates."""
    x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
    x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]
    return scale * math.hypot(x1 - x2, y1 - y2)

