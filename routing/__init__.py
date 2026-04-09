"""Routing toolkit public API."""

from routing.algorithms import (
    SearchResult,
    compute_path_cost,
    breadth_first_search,
    uniform_cost_search,
    depth_first_search,
    depth_limited_search,
    iterative_deepening_search,
    bidirectional_search,
    greedy_best_first_search,
    a_star_search,
    weighted_a_star_search,
)
from routing.data.graph_builder import generate_graph
from routing.data.features_costs import assign_synthetic_features, apply_cost
from routing.heuristics.spatial import euclidean_heuristic, exponential_feature_heuristic
from routing.viz.plotting import plot_all_routes, plot_single_route

__all__ = [
    "SearchResult",
    "compute_path_cost",
    "breadth_first_search",
    "uniform_cost_search",
    "depth_first_search",
    "depth_limited_search",
    "iterative_deepening_search",
    "bidirectional_search",
    "greedy_best_first_search",
    "a_star_search",
    "weighted_a_star_search",
    "generate_graph",
    "assign_synthetic_features",
    "apply_cost",
    "euclidean_heuristic",
    "exponential_feature_heuristic",
    "plot_all_routes",
    "plot_single_route",
]

