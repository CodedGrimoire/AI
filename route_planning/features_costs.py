"""Synthetic feature assignment and custom edge cost computation."""

from __future__ import annotations

import random
from typing import Any, Dict
import networkx as nx


RANDOM_SEED = 12345
random.seed(RANDOM_SEED)


def assign_synthetic_features(G: nx.MultiDiGraph) -> None:
    """Add accident_risk, traffic_level, bumpiness, safety_score to each edge."""
    for _, _, _, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 30.0))
        base = min(1.0, length / 1200.0)

        def clip(x: float) -> float:
            return max(0.0, min(1.0, x))

        def noisy(base_val: float, scale: float = 0.15) -> float:
            return clip(base_val + random.uniform(-scale, scale))

        data["accident_risk"] = noisy(base, 0.12)
        data["traffic_level"] = noisy(base, 0.12)
        data["bumpiness"] = noisy(base, 0.18)
        data["safety_score"] = clip(1.0 - base + random.uniform(-0.10, 0.10))


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

