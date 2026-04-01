"""Synthetic feature assignment and custom edge cost computation."""

from __future__ import annotations

import random
from typing import Any, Dict
import networkx as nx


RANDOM_SEED = 12345
random.seed(RANDOM_SEED)


def assign_synthetic_features(G: nx.MultiDiGraph) -> None:
    """Add structured synthetic attributes to each edge.

    Traffic: grows with length.
    Accident risk: clustered high-risk zones (~20% edges).
    Bumpiness: slightly correlated with length + noise.
    Safety: inverse of accident with small noise.
    """

    # Choose high-risk edges (clustered subset)
    edges = list(G.edges(keys=True))
    high_risk_count = max(1, int(0.2 * len(edges)))
    high_risk_edges = set(random.sample(edges, high_risk_count))

    for edge in edges:
        u, v, k = edge
        data = G[u][v][k]
        length = float(data.get("length", 30.0))

        # Traffic increases with length (scale ~ 800m)
        traffic = min(1.0, length / 800.0)

        # Accident risk clustered
        if edge in high_risk_edges:
            accident = random.uniform(0.8, 1.0)
        else:
            accident = random.uniform(0.05, 0.3)

        # Bumpiness slightly correlated with length + noise
        bump = max(0.0, min(1.0, 0.3 + length / 1500.0 + random.uniform(-0.15, 0.15)))

        # Safety inverse to accident with small noise
        safety = max(0.0, min(1.0, 1.0 - accident + random.uniform(-0.05, 0.05)))

        data["traffic_level"] = traffic
        data["accident_risk"] = accident
        data["bumpiness"] = bump
        data["safety_score"] = safety


def edge_cost(data: Dict[str, Any], weights: Dict[str, float]) -> float:
    return (
        weights["w_distance"] * float(data.get("length", 1.0))
        + weights["w_accident"] * float(data.get("accident_risk", 0.0))
        + weights["w_traffic"] * float(data.get("traffic_level", 0.0))
        + weights["w_bump"] * float(data.get("bumpiness", 0.0))
        - weights["w_safety"] * float(data.get("safety_score", 0.0))
    )


def apply_cost(G: nx.MultiDiGraph, weights: Dict[str, float]) -> None:
    for _, _, _, data in G.edges(keys=True, data=True):
        data["custom_cost"] = edge_cost(data, weights)
