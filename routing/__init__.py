"""Routing toolkit: algorithms, data generation, heuristics, and visualization."""

from routing.algorithms.search import (
    dijkstra_search,
    greedy_best_first_search,
    a_star_search,
    weighted_a_star_search,
    compute_path_cost,
)
from routing.data.graph_builder import generate_graph
from routing.data.features_costs import assign_synthetic_features, apply_cost
from routing.heuristics.spatial import euclidean_heuristic
from routing.viz.plotting import plot_all_routes, plot_single_route

__all__ = [
    "dijkstra_search",
    "greedy_best_first_search",
    "a_star_search",
    "weighted_a_star_search",
    "compute_path_cost",
    "generate_graph",
    "assign_synthetic_features",
    "apply_cost",
    "euclidean_heuristic",
    "plot_all_routes",
    "plot_single_route",
]
