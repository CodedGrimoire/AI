"""Synthetic feature assignment and custom edge cost computation."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional
import networkx as nx


RANDOM_SEED = 12345
random.seed(RANDOM_SEED)

# Config flags (default to legacy behaviour)
USE_NONLINEAR = False
USE_THRESHOLD = False
USE_TIME = False
VERY_LARGE_COST = 1e9
SMALL_COST_FLOOR = 1e-6

# Predefined user profiles (weights mirror legacy defaults unless overridden)
USER_PROFILES: Dict[str, Dict[str, float]] = {
    # Balanced roughly matches DEFAULT_WEIGHTS from existing experiments
    "balanced": {
        "w_distance": 1.0,
        "w_accident": 35.0,
        "w_traffic": 20.0,
        "w_bump": 12.0,
        "w_safety": 15.0,
    },
    # Prioritise speed: higher distance weight, softer accident/traffic penalties
    "fastest": {
        "w_distance": 1.5,
        "w_accident": 15.0,
        "w_traffic": 10.0,
        "w_bump": 10.0,
        "w_safety": 8.0,
    },
    # Prioritise safety: strong accident penalty and safety reward
    "safest": {
        "w_distance": 0.8,
        "w_accident": 55.0,
        "w_traffic": 22.0,
        "w_bump": 12.0,
        "w_safety": 25.0,
    },
    # Eco/comfort: heavier bumpiness penalty to avoid rough roads
    "eco": {
        "w_distance": 1.0,
        "w_accident": 28.0,
        "w_traffic": 18.0,
        "w_bump": 22.0,
        "w_safety": 15.0,
    },
}


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


def _resolve_weights(weights: Optional[Dict[str, float]], user_profile: Optional[str]) -> Dict[str, float]:
    """Choose weights; prefer explicit weights to keep backward compatibility."""

    if weights is not None:
        return weights

    if user_profile and user_profile in USER_PROFILES:
        return USER_PROFILES[user_profile]

    # Fallback to balanced if nothing else is provided
    return USER_PROFILES["balanced"]


def _adjust_traffic_level(base_level: float, time_of_day: Optional[str], use_time: bool) -> float:
    """Simple time-of-day scaling for traffic."""

    if not use_time:
        return base_level

    if time_of_day == "morning":
        return base_level * 1.5
    if time_of_day == "night":
        return base_level * 0.7
    # noon / None → unchanged
    return base_level


def edge_cost(
    data: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    *,
    user_profile: Optional[str] = None,
    use_nonlinear: bool = USE_NONLINEAR,
    use_threshold: bool = USE_THRESHOLD,
    time_of_day: Optional[str] = None,
    use_time: bool = USE_TIME,
    very_large_cost: float = VERY_LARGE_COST,
) -> float:
    """Compute edge cost with optional advanced models.

    Backward compatible: if called with only (data, weights), behaviour matches the
    original linear model. Extra features are opt-in via keyword arguments.
    """

    w = _resolve_weights(weights, user_profile)

    length = float(data.get("length", 1.0))
    accident = float(data.get("accident_risk", 0.0))
    traffic = _adjust_traffic_level(float(data.get("traffic_level", 0.0)), time_of_day, use_time)
    bumpiness = float(data.get("bumpiness", 0.0))
    safety = float(data.get("safety_score", 0.0))

    if use_threshold and accident > 0.9:
        return very_large_cost

    if use_nonlinear:
        cost = (
            w["w_distance"] * length
            + w["w_accident"] * (accident ** 2)
            + w["w_traffic"] * (traffic ** 2)
            + w["w_bump"] * bumpiness
            - w["w_safety"] * safety
        )
    else:
        cost = (
            w["w_distance"] * length
            + w["w_accident"] * accident
            + w["w_traffic"] * traffic
            + w["w_bump"] * bumpiness
            - w["w_safety"] * safety
        )

    # Safety: ensure non-negative, avoid zero
    return max(cost, SMALL_COST_FLOOR)


def apply_cost(
    G: nx.MultiDiGraph,
    weights: Optional[Dict[str, float]] = None,
    *,
    user_profile: str = "balanced",
    use_nonlinear: bool = USE_NONLINEAR,
    use_threshold: bool = USE_THRESHOLD,
    use_time: bool = USE_TIME,
    time_of_day: Optional[str] = None,
    print_config: bool = False,
) -> None:
    """Attach `custom_cost` to edges.

    Defaults preserve legacy linear behaviour when only `weights` is supplied.
    """

    if print_config:
        print("[Cost Model]")
        print(f"Nonlinear: {use_nonlinear}")
        print(f"Threshold: {use_threshold}")
        print(f"Profile: {user_profile}")
        print(f"Time: {time_of_day if use_time else 'disabled'}")

    resolved_weights = _resolve_weights(weights, user_profile)

    for _, _, _, data in G.edges(keys=True, data=True):
        data["custom_cost"] = edge_cost(
            data,
            resolved_weights,
            user_profile=user_profile,
            use_nonlinear=use_nonlinear,
            use_threshold=use_threshold,
            time_of_day=time_of_day,
            use_time=use_time,
        )
