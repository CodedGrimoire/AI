"""Deterministic, spatially-meaningful contextual feature generation.

Attributes are generated in [0, 1] and stored on edges:
- traffic_level
- accident_risk
- bumpiness
- safety_score

Node-level averages are also stored for heuristic use.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import networkx as nx


@dataclass(frozen=True)
class FeatureConfig:
    seed: int = 2026


FEATURE_KEYS = ("traffic_level", "accident_risk", "bumpiness", "safety_score")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _deterministic_noise(seed: int, u, v, k, channel: str) -> float:
    blob = f"{seed}:{u}:{v}:{k}:{channel}".encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    n = int(digest[:12], 16)
    return (n % 1_000_000) / 1_000_000.0


def _road_priority(highway) -> float:
    if isinstance(highway, list):
        highway = highway[0] if highway else "unclassified"
    h = str(highway or "unclassified")
    if "motorway" in h or "trunk" in h:
        return 1.0
    if "primary" in h:
        return 0.8
    if "secondary" in h:
        return 0.6
    if "tertiary" in h:
        return 0.45
    return 0.3


def assign_contextual_features(G: nx.MultiDiGraph, cfg: FeatureConfig = FeatureConfig()) -> None:
    xs = [float(d["x"]) for _, d in G.nodes(data=True) if "x" in d]
    ys = [float(d["y"]) for _, d in G.nodes(data=True) if "y" in d]
    if not xs or not ys:
        raise RuntimeError("Graph nodes missing projected x/y coordinates.")

    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    max_radius = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) or 1.0

    for u, v, k, data in G.edges(keys=True, data=True):
        ux, uy = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
        vx, vy = float(G.nodes[v]["x"]), float(G.nodes[v]["y"])

        mx, my = 0.5 * (ux + vx), 0.5 * (uy + vy)
        center_proximity = 1.0 - _clamp01(math.hypot(mx - cx, my - cy) / max_radius)

        length = float(data.get("length", math.hypot(ux - vx, uy - vy)))
        length_norm = _clamp01(length / 300.0)
        road_priority = _road_priority(data.get("highway"))

        noise_t = _deterministic_noise(cfg.seed, u, v, k, "traffic") - 0.5
        noise_a = _deterministic_noise(cfg.seed, u, v, k, "accident") - 0.5
        noise_b = _deterministic_noise(cfg.seed, u, v, k, "bump") - 0.5

        # Dhaka-inspired synthetic patterns:
        # - central corridors and major roads carry heavier traffic.
        traffic = _clamp01(0.45 * center_proximity + 0.30 * road_priority + 0.20 * length_norm + 0.10 * noise_t + 0.05)

        # - accident risk is elevated on major/fast roads and congested zones.
        accident = _clamp01(0.40 * road_priority + 0.35 * traffic + 0.10 * center_proximity + 0.15 * noise_a)

        # - bumpiness tends to be worse toward peripheral/local roads.
        peripheral_factor = 1.0 - center_proximity
        local_road_factor = 1.0 - road_priority
        bumpiness = _clamp01(0.35 * peripheral_factor + 0.35 * local_road_factor + 0.15 * length_norm + 0.15 * noise_b)

        # Safety is modeled inversely from harmful attributes.
        safety = _clamp01(1.0 - (0.50 * accident + 0.30 * traffic + 0.20 * bumpiness))

        data["traffic_level"] = traffic
        data["accident_risk"] = accident
        data["bumpiness"] = bumpiness
        data["safety_score"] = safety

    aggregate_edge_features_to_nodes(G)


def _iter_vals(datas: Iterable[dict], key: str) -> Iterable[float]:
    for d in datas:
        try:
            yield float(d.get(key, 0.0))
        except (TypeError, ValueError):
            yield 0.0


def aggregate_edge_features_to_nodes(G: nx.MultiDiGraph) -> None:
    for n in G.nodes:
        edges = [d for _, _, d in G.in_edges(n, data=True)] + [d for _, _, d in G.out_edges(n, data=True)]
        if not edges:
            for key in FEATURE_KEYS:
                G.nodes[n][key] = 0.0
            continue

        for key in FEATURE_KEYS:
            vals = list(_iter_vals(edges, key))
            G.nodes[n][key] = _clamp01(sum(vals) / max(1, len(vals)))
