"""Heuristic functions for routing."""

from __future__ import annotations

from typing import Hashable, Callable
import networkx as nx
import math


def euclidean_heuristic(G: nx.MultiDiGraph, n1: Hashable, n2: Hashable, scale: float = 1.0) -> float:
    """Straight-line distance between two nodes using (x, y) coordinates."""
    x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
    x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]
    return scale * math.hypot(x1 - x2, y1 - y2)


# Benefit-based heuristic parameters (single source of truth)
HEURISTIC_ALPHA = 0.5
HEURISTIC_BETA = 0.4
HEURISTIC_GAMMA = 0.3
HEURISTIC_LAMBDA = 0.6


def exponential_feature_heuristic(
    G: nx.MultiDiGraph,
    goal: Hashable,
    *,
    alpha: float = HEURISTIC_ALPHA,
    beta: float = HEURISTIC_BETA,
    gamma: float = HEURISTIC_GAMMA,
    lambd: float = HEURISTIC_LAMBDA,
) -> Callable[[Hashable], float]:
    """Return benefit-based heuristic:

    L_T(n) = T_max - T(n)
    L_A(n) = A_max - A(n)
    L_B(n) = B_max - B(n)
    z(n)   = alpha*L_T + beta*L_A + gamma*L_B + lambda*S(n)
    h(n)   = max(0, d(n,g) - z(n))

    The function name is preserved for backward compatibility with existing
    experiment/search pipelines.
    """

    def _safe_feature(n: Hashable, key: str) -> float:
        val = G.nodes[n].get(key, 0.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _feature_max(key: str) -> float:
        vals = []
        for n in G.nodes:
            try:
                vals.append(float(G.nodes[n].get(key, 0.0)))
            except (TypeError, ValueError):
                continue
        if not vals:
            return 1.0
        m = max(vals)
        return m if m > 0.0 else 1.0

    # Features in this project are clamped to [0,1], so these are usually 1.0.
    # We still infer maxima from graph data for robustness.
    t_max = _feature_max("traffic_level")
    a_max = _feature_max("accident_risk")
    b_max = _feature_max("bumpiness")

    def heuristic(n: Hashable) -> float:
        d = euclidean_heuristic(G, n, goal, scale=1.0)
        t = _safe_feature(n, "traffic_level")
        a = _safe_feature(n, "accident_risk")
        b = _safe_feature(n, "bumpiness")
        s = _safe_feature(n, "safety_score")

        low_traffic_benefit = max(0.0, t_max - t)
        low_accident_benefit = max(0.0, a_max - a)
        low_bumpiness_benefit = max(0.0, b_max - b)
        z = (
            alpha * low_traffic_benefit
            + beta * low_accident_benefit
            + gamma * low_bumpiness_benefit
            + lambd * s
        )
        return max(0.0, d - z)

    return heuristic
