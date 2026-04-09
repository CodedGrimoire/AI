"""Graph generation utilities using OSMnx.

Default behaviour now builds the full Dhaka, Bangladesh road network
(`network_type="drive"`) and keeps the largest strongly connected component
to guarantee reachability for directed searches. A synthetic grid fallback is
kept for offline scenarios.
"""

from __future__ import annotations

import os

# Disable network downloads for PROJ data *before* importing osmnx/pyproj
# to avoid long hangs in restricted environments.
os.environ.setdefault("PROJ_NETWORK", "OFF")

import osmnx as ox
import networkx as nx
import time
import math


def _synthetic_grid(target_nodes: int = 400, spacing: float = 30.0) -> nx.MultiDiGraph:
    """Offline fallback: build a grid graph near the requested size.

    Tries to hit `target_nodes` exactly by constructing a nearly square grid and
    trimming the last row if necessary.
    """

    # Choose near-square dimensions
    rows = max(2, math.isqrt(target_nodes))
    cols = max(2, math.ceil(target_nodes / rows))

    # Build base grid
    base = nx.grid_2d_graph(rows, cols, create_using=nx.Graph)

    # Trim excess nodes from the last row to match target_nodes (keeps grid shape)
    extra = len(base) - target_nodes
    if extra > 0:
        # nodes are (r, c); trim from the highest row, right to left
        max_r = rows - 1
        candidates = sorted([(r, c) for r, c in base.nodes if r == max_r], reverse=True)
        for node in candidates:
            if extra <= 0:
                break
            base.remove_node(node)
            extra -= 1

    G = nx.MultiDiGraph()
    G.graph["crs"] = "epsg:3857"
    G.graph["graph_label"] = "Synthetic grid"
    for (r, c) in base.nodes:
        G.add_node((r, c), x=float(c * spacing), y=float(r * spacing))
    for u, v in base.edges:
        G.add_edge(u, v, length=spacing)
        G.add_edge(v, u, length=spacing)
    return G


def generate_graph(use_osm: bool = True) -> nx.MultiDiGraph:
    """Build the full Dhaka city drive network or a synthetic fallback.

    Connectivity handling:
    - For directed graphs, keep the largest strongly connected component so
      start/goal pairs remain mutually reachable.
    - If that fails, fall back to the largest weakly connected component.
    """

    if not use_osm:
        G = _synthetic_grid(target_nodes=400)
        G.graph["graph_label"] = "Synthetic grid"
        print(f"[info] Synthetic grid | Nodes: {len(G)} | Edges: {len(G.edges())}")
        return G

    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.timeout = 60  # allow larger download

    place_query = "Dhaka, Bangladesh"
    try:
        print(f"[info] Downloading OSM graph for: {place_query} (drive)")
        G = ox.graph_from_place(place_query, network_type="drive", simplify=True)
        G = ox.project_graph(G)

        # Keep largest strongly connected component for directed routing.
        try:
            largest_strong = max(nx.strongly_connected_components(G), key=len)
            G = G.subgraph(largest_strong).copy()
            G.graph["component_type"] = "strongly_connected"
        except Exception:
            largest_weak = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest_weak).copy()
            G.graph["component_type"] = "weakly_connected"

        G.graph["graph_label"] = "Dhaka road network (OSM)"
        print(f"[info] Dhaka graph ready | Nodes: {len(G)} | Edges: {len(G.edges())} | component: {G.graph.get('component_type')}")
        return G
    except Exception as e:
        print(f"[warn] OSM Dhaka load failed ({e}); using offline synthetic grid")
        G = _synthetic_grid(target_nodes=400)
        G.graph["graph_label"] = "Synthetic grid"
        print(f"[info] Synthetic grid | Nodes: {len(G)} | Edges: {len(G.edges())}")
        return G
