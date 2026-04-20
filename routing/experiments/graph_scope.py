"""Utilities for selecting full-map vs node-limited experiment scopes."""

from __future__ import annotations

import random
from typing import Optional

import networkx as nx

from routing.data.graph_builder import generate_graph
from routing.data.features_costs import apply_cost, assign_synthetic_features


def _largest_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the largest connected component for robust routing experiments."""
    if G.is_directed():
        comp = max(nx.strongly_connected_components(G), key=len)
    else:
        comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def restrict_graph_nodes(
    G: nx.MultiDiGraph,
    max_nodes: Optional[int] = None,
    *,
    seed: int = 42,
) -> nx.MultiDiGraph:
    """Restrict graph size by BFS sampling while keeping a connected component."""
    G = _largest_component(G)
    if max_nodes is None or max_nodes <= 0 or len(G) <= max_nodes:
        return G

    rng = random.Random(seed)
    UG = G.to_undirected(as_view=True)
    start = rng.choice(list(UG.nodes))
    bfs_nodes = list(nx.bfs_tree(UG, start).nodes())[:max_nodes]
    H = G.subgraph(bfs_nodes).copy()
    return _largest_component(H)


def load_graph_with_costs(
    *,
    use_osm: bool = True,
    max_nodes: Optional[int] = None,
    seed: int = 42,
) -> nx.MultiDiGraph:
    """Build graph, assign features/costs, and apply optional node budget."""
    G = generate_graph(use_osm=use_osm)
    assign_synthetic_features(G)
    apply_cost(G)
    return restrict_graph_nodes(G, max_nodes=max_nodes, seed=seed)
