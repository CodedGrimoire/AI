"""Graph loading utilities for whole-Dhaka experiments.

The graph is cached under this repo's dedicated `graph_cache/` directory so this
project does not interfere with caches in your original repository.
"""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("PROJ_NETWORK", "OFF")

import networkx as nx
import osmnx as ox


def _cache_paths(graph_dir: Path) -> tuple[Path, Path]:
    graph_dir.mkdir(parents=True, exist_ok=True)
    return graph_dir / "dhaka_drive.graphml", graph_dir / "dhaka_drive_projected.graphml"


def load_or_build_dhaka_graph(graph_dir: Path) -> nx.MultiDiGraph:
    """Load Dhaka drive graph from cache, otherwise download and cache it.

    Keeps the largest strongly connected component to make directed routing
    queries mutually reachable.
    """

    raw_path, projected_path = _cache_paths(graph_dir)

    if projected_path.exists():
        G = ox.load_graphml(projected_path)
        G.graph["graph_label"] = "Dhaka road network (projected cache)"
        return G

    if raw_path.exists():
        G = ox.load_graphml(raw_path)
    else:
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.timeout = 120
        G = ox.graph_from_place("Dhaka, Bangladesh", network_type="drive", simplify=True)
        ox.save_graphml(G, raw_path)

    G = ox.project_graph(G)

    if len(G) == 0:
        raise RuntimeError("Loaded empty Dhaka graph.")

    try:
        largest = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        G.graph["component_type"] = "strongly_connected"
    except Exception:
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        G.graph["component_type"] = "weakly_connected"

    G.graph["graph_label"] = "Dhaka road network (OSM)"
    ox.save_graphml(G, projected_path)
    return G
