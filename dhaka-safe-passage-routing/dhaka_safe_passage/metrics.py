"""Path-quality metrics and safe-passage scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import networkx as nx


@dataclass(frozen=True)
class SafeScoreWeights:
    w_safety: float = 1.0
    w_traffic: float = 0.9
    w_accident: float = 1.1
    w_bumpiness: float = 0.6


def _edge_for_step(G: nx.MultiDiGraph, u: Hashable, v: Hashable) -> dict:
    edge_data = G.get_edge_data(u, v)
    if not edge_data:
        return {}
    # Select edge with minimum active search cost.
    return min(edge_data.values(), key=lambda d: float(d.get("custom_cost", d.get("length", 1.0))))


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def compute_route_quality(
    G: nx.MultiDiGraph,
    path: list[Hashable],
    score_weights: SafeScoreWeights = SafeScoreWeights(),
) -> dict:
    if len(path) < 2:
        return {
            "total_path_length": 0.0,
            "average_traffic": 0.0,
            "average_accident_risk": 0.0,
            "average_bumpiness": 0.0,
            "average_safety": 0.0,
            "cumulative_safety": 0.0,
            "cumulative_contextual_penalty": 0.0,
            "safe_passage_score": 0.0,
        }

    lengths = []
    traffic = []
    accident = []
    bumpiness = []
    safety = []

    for u, v in zip(path[:-1], path[1:]):
        d = _edge_for_step(G, u, v)
        lengths.append(float(d.get("length", 0.0)))
        traffic.append(float(d.get("traffic_level", 0.0)))
        accident.append(float(d.get("accident_risk", 0.0)))
        bumpiness.append(float(d.get("bumpiness", 0.0)))
        safety.append(float(d.get("safety_score", 0.0)))

    avg_traffic = _mean(traffic)
    avg_accident = _mean(accident)
    avg_bumpiness = _mean(bumpiness)
    avg_safety = _mean(safety)

    cumulative_penalty = sum(
        score_weights.w_traffic * t + score_weights.w_accident * a + score_weights.w_bumpiness * b - score_weights.w_safety * s
        for t, a, b, s in zip(traffic, accident, bumpiness, safety)
    )

    # Explicit report-friendly score:
    # safe_passage_score = w1*avg_safety - w2*avg_traffic - w3*avg_accident - w4*avg_bumpiness
    safe_passage_score = (
        score_weights.w_safety * avg_safety
        - score_weights.w_traffic * avg_traffic
        - score_weights.w_accident * avg_accident
        - score_weights.w_bumpiness * avg_bumpiness
    )

    return {
        "total_path_length": sum(lengths),
        "average_traffic": avg_traffic,
        "average_accident_risk": avg_accident,
        "average_bumpiness": avg_bumpiness,
        "average_safety": avg_safety,
        "cumulative_safety": sum(safety),
        "cumulative_contextual_penalty": cumulative_penalty,
        "safe_passage_score": safe_passage_score,
    }
