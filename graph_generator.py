"""Generate and plot a manageable OSMnx road network graph.

Function:
    generate_graph(center, min_nodes=50, max_nodes=200)

Behavior:
    - Grows search radius until graph has at least min_nodes.
    - If graph exceeds 300 nodes, it trims to ~max_nodes using a node-induced subgraph.
    - Projects the graph for metric distance calculations.
    - Prints node/edge counts and plots the graph.

Usage:
    python graph_generator.py
"""

from __future__ import annotations

import osmnx as ox
import networkx as nx


def generate_graph(center: tuple[float, float], min_nodes: int = 50, max_nodes: int = 200) -> nx.MultiDiGraph:
    ox.settings.log_console = False
    ox.settings.use_cache = True

    radius = 1000  # meters
    G = None

    # Grow radius until we have enough nodes
    while True:
        G = ox.graph_from_point(center, dist=radius, network_type="drive")
        if len(G) >= min_nodes:
            break
        radius = int(radius * 1.5)

    # Trim if too large
    if len(G) > 300:
        nodes = list(G.nodes)[:max_nodes]
        G = G.subgraph(nodes).copy()

    print(f"Nodes: {len(G)} | Edges: {len(G.edges())}")

    # Project for metric calculations
    G = ox.project_graph(G)

    # Plot for visual confirmation
    ox.plot_graph(G, node_size=5, edge_color="#444", bgcolor="white")
    return G


if __name__ == "__main__":
    # Example: Dhaka center point
    center_point = (23.746, 90.376)
    generate_graph(center_point)
