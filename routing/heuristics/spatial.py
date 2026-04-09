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


# Exponential heuristic parameters (defaults here for single source of truth)
HEURISTIC_ALPHA = 0.5
HEURISTIC_BETA = 0.4
HEURISTIC_GAMMA = 0.3
HEURISTIC_LAMBDA = 0.6
HEURISTIC_EXP_CLAMP = 4.0  # prevent overflow/instability in exp


def exponential_feature_heuristic(
    G: nx.MultiDiGraph,
    goal: Hashable,
    *,
    alpha: float = HEURISTIC_ALPHA,
    beta: float = HEURISTIC_BETA,
    gamma: float = HEURISTIC_GAMMA,
    lambd: float = HEURISTIC_LAMBDA,
    exp_clamp: float = HEURISTIC_EXP_CLAMP,
) -> Callable[[Hashable], float]:
    """Return h(n) = d(n, goal) * exp(alpha*T + beta*A + gamma*B - lambda*S).

    Assumes node attributes traffic_level, accident_risk, bumpiness, safety_score
    are already populated and clamped to [0, 1].
    """

    def _safe_feature(n: Hashable, key: str) -> float:
        val = G.nodes[n].get(key, 0.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def heuristic(n: Hashable) -> float:
        d = euclidean_heuristic(G, n, goal, scale=1.0)
        t = _safe_feature(n, "traffic_level")
        a = _safe_feature(n, "accident_risk")
        b = _safe_feature(n, "bumpiness")
        s = _safe_feature(n, "safety_score")

        exponent = alpha * t + beta * a + gamma * b - lambd * s
        # Clamp exponent to avoid numerical explosion
        exponent = max(-exp_clamp, min(exp_clamp, exponent))
        return d * math.exp(exponent)

    return heuristic
