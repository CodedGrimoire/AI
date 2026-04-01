"""Graph generation utilities using OSMnx.

Sets PROJ_NETWORK=OFF up front so pyproj/osmnx avoid network fetches for
grid files, which can hang in restricted environments.
"""

from __future__ import annotations

import os

# Disable network downloads for PROJ data *before* importing osmnx/pyproj
# to avoid long hangs in restricted environments.
os.environ.setdefault("PROJ_NETWORK", "OFF")

import osmnx as ox
import networkx as nx
import time


def _synthetic_grid(target_nodes: int = 400, spacing: float = 30.0) -> nx.MultiDiGraph:
    """Offline fallback: build a simple grid graph with x/y coords and lengths."""
    side = max(2, int(target_nodes ** 0.5))
    rows = cols = side
    base = nx.grid_2d_graph(rows, cols, create_using=nx.Graph)
    G = nx.MultiDiGraph()
    G.graph["crs"] = "epsg:3857"
    for (r, c) in base.nodes:
        G.add_node((r, c), x=float(c * spacing), y=float(r * spacing))
    for u, v in base.edges:
        G.add_edge(u, v, length=spacing)
        G.add_edge(v, u, length=spacing)
    return G


def generate_graph(center: tuple[float, float], min_nodes: int = 50, max_nodes: int = 200) -> nx.MultiDiGraph:
    use_osm = os.environ.get("USE_OSM", "0") == "1"
    if not use_osm:
        G = _synthetic_grid(target_nodes=max_nodes)
        print(f"[info] Synthetic grid | Nodes: {len(G)} | Edges: {len(G.edges())}")
        return G

    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.timeout = 20  # seconds per request

    radius = 1000  # meters
    G: nx.MultiDiGraph | None = None
    start_time = time.time()

    try:
        while True:
            G = ox.graph_from_point(center, dist=radius, network_type="drive")
            if len(G) >= min_nodes:
                break
            radius = int(radius * 1.5)
            if time.time() - start_time > 60:
                raise TimeoutError("OSM download taking too long; falling back to synthetic grid")

        if len(G) > max_nodes:
            nodes = list(G.nodes)[:max_nodes]
            G = G.subgraph(nodes).copy()

        G = ox.project_graph(G)
        print(f"[info] OSM graph loaded | Nodes: {len(G)} | Edges: {len(G.edges())}")
        return G
    except Exception as e:
        print(f"[warn] OSM graph load failed ({e}); using offline synthetic grid")
        G = _synthetic_grid(target_nodes=max_nodes)
        print(f"[info] Synthetic grid | Nodes: {len(G)} | Edges: {len(G.edges())}")
        return G
