"""Search algorithms package.

Uninformed implementations are intentionally copied from the base project
without behavioral changes.
"""

from .common import SearchResult, compute_path_cost
from .uninformed import (
    breadth_first_search,
    uniform_cost_search,
    depth_first_search,
    depth_limited_search,
    iterative_deepening_search,
    bidirectional_search,
)
from .informed import greedy_best_first_search, a_star_search, weighted_a_star_search

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
]
