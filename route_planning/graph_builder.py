"""Graph generation utilities using OSMnx."""

from __future__ import annotations

import osmnx as ox
import networkx as nx


def generate_graph(center: tuple[float, float], min_nodes: int = 50, max_nodes: int = 200) -> nx.MultiDiGraph:
    ox.settings.log_console = False
    ox.settings.use_cache = True

    radius = 1000  # meters
    G = None

    while True:
        G = ox.graph_from_point(center, dist=radius, network_type="drive")
        if len(G) >= min_nodes:
            break
        radius = int(radius * 1.5)

    if len(G) > max_nodes:
        nodes = list(G.nodes)[:max_nodes]
        G = G.subgraph(nodes).copy()

    print(f"Nodes: {len(G)} | Edges: {len(G.edges())}")
    G = ox.project_graph(G)

    # Non-blocking plot saved to file (prevents GUI hang)
    # Skip plotting to keep runs non-blocking and fast.
    return G
