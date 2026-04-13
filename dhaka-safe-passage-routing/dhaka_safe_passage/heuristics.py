"""Heuristics for informed routing under safe-passage objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable
import math

import networkx as nx


@dataclass(frozen=True)
class HeuristicWeights:
    alpha_traffic: float = 0.7
    beta_accident: float = 0.9
    gamma_bumpiness: float = 0.5
    lambda_safety: float = 0.8
    min_factor: float = 0.05


def euclidean_distance(G: nx.MultiDiGraph, n1: Hashable, n2: Hashable) -> float:
    x1, y1 = float(G.nodes[n1]["x"]), float(G.nodes[n1]["y"])
    x2, y2 = float(G.nodes[n2]["x"]), float(G.nodes[n2]["y"])
    return math.hypot(x1 - x2, y1 - y2)


def contextual_heuristic(
    G: nx.MultiDiGraph,
    goal: Hashable,
    weights: HeuristicWeights = HeuristicWeights(),
) -> Callable[[Hashable], float]:
    """Return h(n) = d(n,g) * max(min_factor, 1 + aT + bA + cB - lS)."""

    def h(n: Hashable) -> float:
        d = euclidean_distance(G, n, goal)
        t = float(G.nodes[n].get("traffic_level", 0.0))
        a = float(G.nodes[n].get("accident_risk", 0.0))
        b = float(G.nodes[n].get("bumpiness", 0.0))
        s = float(G.nodes[n].get("safety_score", 0.0))
        factor = 1.0 + weights.alpha_traffic * t + weights.beta_accident * a + weights.gamma_bumpiness * b - weights.lambda_safety * s
        return d * max(weights.min_factor, factor)

    return h
