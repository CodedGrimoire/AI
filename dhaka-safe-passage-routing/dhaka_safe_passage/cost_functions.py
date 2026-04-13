"""Edge cost models.

Uninformed algorithms can keep using distance-only `custom_cost`.
Informed experiments switch to contextual costs while preserving positivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class CostWeights:
    alpha_traffic: float = 0.9
    beta_accident: float = 1.1
    gamma_bumpiness: float = 0.6
    lambda_safety: float = 0.8
    min_multiplier: float = 0.05


def apply_distance_cost(G: nx.MultiDiGraph) -> None:
    for _, _, _, data in G.edges(keys=True, data=True):
        data["custom_cost"] = max(float(data.get("length", 1.0)), 1e-6)


def apply_contextual_cost(G: nx.MultiDiGraph, weights: CostWeights = CostWeights()) -> None:
    """Set `custom_cost` for informed routing.

    Formula:
    cost(u,v) = length(u,v) * max(min_multiplier,
                   1 + alpha*T + beta*A + gamma*B - lambda*S)
    """

    for _, _, _, data in G.edges(keys=True, data=True):
        length = max(float(data.get("length", 1.0)), 1e-6)
        t = float(data.get("traffic_level", 0.0))
        a = float(data.get("accident_risk", 0.0))
        b = float(data.get("bumpiness", 0.0))
        s = float(data.get("safety_score", 0.0))

        mult = 1.0 + weights.alpha_traffic * t + weights.beta_accident * a + weights.gamma_bumpiness * b - weights.lambda_safety * s
        mult = max(weights.min_multiplier, mult)
        data["custom_cost"] = length * mult
