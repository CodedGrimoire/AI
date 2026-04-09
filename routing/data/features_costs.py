"""Synthetic feature assignment and distance-only edge cost computation.

This module now separates *features* from the actual traversal cost:
- Synthetic features (traffic_level, accident_risk, bumpiness, safety_score)
  are still generated and stored on edges for analysis/debugging.
- The traversal cost used by all search algorithms is *only* the physical
  distance between adjacent nodes (edge length or Euclidean fallback).

It also computes per-node feature averages so heuristic functions can read
T(n), A(n), B(n), S(n) directly from node attributes.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable

import networkx as nx


RANDOM_SEED = 12345
random.seed(RANDOM_SEED)

SMALL_COST_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Synthetic feature generation (unchanged semantics)
# ---------------------------------------------------------------------------
def assign_synthetic_features(G: nx.MultiDiGraph) -> None:
    """Add structured synthetic attributes to each edge.

    Traffic: grows with length.
    Accident risk: clustered high-risk zones (~20% edges).
    Bumpiness: slightly correlated with length + noise.
    Safety: inverse of accident with small noise.
    """

    edges = list(G.edges(keys=True))
    if not edges:
        return

    high_risk_count = max(1, int(0.2 * len(edges)))
    high_risk_edges = set(random.sample(edges, high_risk_count))

    for u, v, k in edges:
        data = G[u][v][k]
        length = float(data.get("length", 30.0))

        traffic = min(1.0, length / 800.0)
        accident = random.uniform(0.8, 1.0) if (u, v, k) in high_risk_edges else random.uniform(0.05, 0.3)
        bump = max(0.0, min(1.0, 0.3 + length / 1500.0 + random.uniform(-0.15, 0.15)))
        safety = max(0.0, min(1.0, 1.0 - accident + random.uniform(-0.05, 0.05)))

        data["traffic_level"] = traffic
        data["accident_risk"] = accident
        data["bumpiness"] = bump
        data["safety_score"] = safety


# ---------------------------------------------------------------------------
# Distance-only cost model
# ---------------------------------------------------------------------------
def _euclidean_length(G: nx.MultiDiGraph, u, v) -> float:
    """Fallback Euclidean distance using node coordinates."""
    ux, uy = G.nodes[u].get("x"), G.nodes[u].get("y")
    vx, vy = G.nodes[v].get("x"), G.nodes[v].get("y")
    if ux is None or uy is None or vx is None or vy is None:
        return 1.0  # best-effort default
    return math.hypot(ux - vx, uy - vy)


def compute_edge_distance(G: nx.MultiDiGraph, u, v, data: Dict[str, Any]) -> float:
    """Return the physical distance for an edge, with safe defaults."""
    length = data.get("length")
    if length is None:
        return _euclidean_length(G, u, v)
    try:
        return float(length)
    except (TypeError, ValueError):
        return _euclidean_length(G, u, v)


def apply_cost(G: nx.MultiDiGraph) -> None:
    """Attach `custom_cost` = physical distance to every edge.

    Previous composite models (traffic/accident/bumpiness/safety) are no longer
    part of the traversal cost. Those features remain available on edges and
    are also aggregated to nodes for heuristic use.
    """

    for u, v, k, data in G.edges(keys=True, data=True):
        dist = compute_edge_distance(G, u, v, data)
        data["custom_cost"] = max(dist, SMALL_COST_FLOOR)

    aggregate_edge_features_to_nodes(G)


# ---------------------------------------------------------------------------
# Node-level feature aggregation (for heuristics)
# ---------------------------------------------------------------------------
FEATURE_KEYS = ("traffic_level", "accident_risk", "bumpiness", "safety_score")


def _iter_edge_features(datas: Iterable[Dict[str, Any]], key: str) -> Iterable[float]:
    for d in datas:
        val = d.get(key)
        if val is not None:
            try:
                yield float(val)
            except (TypeError, ValueError):
                continue


def aggregate_edge_features_to_nodes(G: nx.MultiDiGraph) -> None:
    """Store mean feature values on each node for heuristic consumption."""

    for n in G.nodes:
        incident_edges = []
        incident_edges.extend(data for _, _, data in G.in_edges(n, data=True))
        incident_edges.extend(data for _, _, data in G.out_edges(n, data=True))

        if not incident_edges:
            # No incident edges: default to 0.0 for all features
            for key in FEATURE_KEYS:
                G.nodes[n][key] = 0.0
            continue

        for key in FEATURE_KEYS:
            vals = list(_iter_edge_features(incident_edges, key))
            mean_val = sum(vals) / len(vals) if vals else 0.0
            # Clamp to [0,1] to keep heuristic stable
            G.nodes[n][key] = max(0.0, min(1.0, mean_val))


__all__ = [
    "assign_synthetic_features",
    "apply_cost",
    "compute_edge_distance",
    "aggregate_edge_features_to_nodes",
    "FEATURE_KEYS",
]

