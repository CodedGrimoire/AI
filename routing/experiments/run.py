"""Unified experiment runner comparing uninformed and informed search algorithms.

Naming strictly follows lecture notes:
- Breadth-first search (BFS)
- Uniform cost search
- Depth-first search (DFS)
- Depth Limited Search
- Iterative Deepening Search
- Bidirectional Search
- Greedy best-first search
- A* search
- Weighted A*

Edge cost: distance-only (custom_cost set by data pipeline).
Heuristic: exponential feature heuristic defined in routing.heuristics.spatial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from routing.data.graph_builder import generate_graph
from routing.data.features_costs import assign_synthetic_features, apply_cost
from routing.heuristics.spatial import exponential_feature_heuristic, euclidean_heuristic
from routing.viz.plotting import plot_all_routes, plot_single_route
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


OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)


def choose_start_goal(G: nx.MultiDiGraph):
    """Pick a well-separated start/goal pair (farthest pair heuristic)."""
    if not nx.is_weakly_connected(G):
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    nodes = list(G.nodes)
    start = nodes[0]
    goal = max(nodes, key=lambda n: euclidean_heuristic(G, start, n, 1.0))
    start = max(nodes, key=lambda n: euclidean_heuristic(G, n, goal, 1.0))
    return start, goal, G


def run_all_algorithms(G: nx.MultiDiGraph, start, goal) -> List[SearchResult]:
    h_fn = exponential_feature_heuristic(G, goal)

    algorithms: List[Callable[[], SearchResult]] = [
        lambda: breadth_first_search(G, start, goal),
        lambda: uniform_cost_search(G, start, goal),
        lambda: depth_first_search(G, start, goal),
        lambda: depth_limited_search(G, start, goal, limit=20),
        lambda: iterative_deepening_search(G, start, goal, max_depth=40),
        lambda: bidirectional_search(G, start, goal),
        lambda: greedy_best_first_search(G, start, goal, h_fn),
        lambda: a_star_search(G, start, goal, h_fn),
    ]

    # Weighted A* sweep (required weights)
    for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
        algorithms.append(lambda w=w: weighted_a_star_search(G, start, goal, h_fn, w=w))

    results: List[SearchResult] = []
    for fn in algorithms:
        res = fn()
        results.append(res)
    return results


def results_to_df(results: List[SearchResult]) -> pd.DataFrame:
    records = []
    for r in results:
        records.append(
            {
                "algorithm_name": r.algorithm_name,
                "path_found": r.path_found,
                "total_path_cost": r.total_path_cost,
                "nodes_expanded": r.nodes_expanded,
                "execution_time": r.execution_time,
                "max_frontier_size": r.max_frontier_size,
                "path_length": r.path_length,
                "visited_count": r.visited_count,
                "depth_reached": r.depth_reached,
                "cutoff_occurred": r.cutoff_occurred,
                "meeting_node": r.meeting_node,
                "weight": r.weight,
            }
        )
    return pd.DataFrame(records)


def save_csv(df: pd.DataFrame, filename: Path):
    df.to_csv(filename, index=False)


def plot_bars(df: pd.DataFrame):
    metrics = [
        ("total_path_cost", "Total Path Cost"),
        ("nodes_expanded", "Nodes Expanded"),
        ("execution_time", "Execution Time (s)"),
        ("path_length", "Path Length (nodes)"),
        ("max_frontier_size", "Max Frontier Size"),
    ]
    for key, title in metrics:
        plt.figure(figsize=(8, 4))
        plt.bar(df["algorithm_name"], df[key], color="steelblue")
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"bar_{key}.png", dpi=150)
        plt.close()


def plot_scatters(df: pd.DataFrame):
    pairs = [
        ("nodes_expanded", "total_path_cost", "Nodes Expanded", "Total Path Cost", "scatter_expanded_cost.png"),
        ("execution_time", "total_path_cost", "Execution Time (s)", "Total Path Cost", "scatter_time_cost.png"),
        ("execution_time", "nodes_expanded", "Execution Time (s)", "Nodes Expanded", "scatter_time_expanded.png"),
    ]
    for x, y, xl, yl, fname in pairs:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x], df[y], color="darkorange")
        for _, row in df.iterrows():
            plt.annotate(row["algorithm_name"], (row[x], row[y]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close()


def plot_weighted_astar(df: pd.DataFrame):
    wa = df[df["algorithm_name"] == "Weighted A*"]
    if wa.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(wa["weight"], wa["total_path_cost"], marker="o")
    plt.xlabel("Weight w")
    plt.ylabel("Total Path Cost")
    plt.title("Weighted A*: cost vs w")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wa_cost_vs_w.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(wa["weight"], wa["nodes_expanded"], marker="o", color="seagreen")
    plt.xlabel("Weight w")
    plt.ylabel("Nodes Expanded")
    plt.title("Weighted A*: expansions vs w")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wa_expanded_vs_w.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(wa["weight"], wa["execution_time"], marker="o", color="purple")
    plt.xlabel("Weight w")
    plt.ylabel("Execution Time (s)")
    plt.title("Weighted A*: time vs w")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wa_time_vs_w.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(wa["weight"], wa["path_length"], marker="o", color="brown")
    plt.xlabel("Weight w")
    plt.ylabel("Path Length (nodes)")
    plt.title("Weighted A*: path length vs w")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wa_pathlen_vs_w.png", dpi=150)
    plt.close()


def plot_depth_variants(df: pd.DataFrame):
    dls = df[df["algorithm_name"] == "Depth Limited Search"]
    ids = df[df["algorithm_name"] == "Iterative Deepening Search"]
    if not dls.empty:
        plt.figure(figsize=(5, 3))
        plt.bar(["DLS"], dls["nodes_expanded"], color="gray")
        plt.ylabel("Nodes Expanded")
        plt.title("Depth Limited Search (single run)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "dls_nodes.png", dpi=150)
        plt.close()
    if not ids.empty:
        plt.figure(figsize=(5, 3))
        plt.bar(["IDS"], ids["nodes_expanded"], color="teal")
        plt.ylabel("Nodes Expanded")
        plt.title("Iterative Deepening Search")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ids_nodes.png", dpi=150)
        plt.close()


def plot_routes(G: nx.MultiDiGraph, results: List[SearchResult], start, goal):
    paths = {}
    for res in results:
        if res.path_found and res.path:
            paths[res.algorithm_name] = res.path
            plot_single_route(G, res.path, res.algorithm_name, color=None)
    if paths:
        plot_all_routes(G, paths, start, goal)


def main():
    center = (23.746, 90.376)  # Dhaka example
    G = generate_graph(center, min_nodes=100, max_nodes=140)
    assign_synthetic_features(G)
    apply_cost(G)

    start, goal, G = choose_start_goal(G)
    results = run_all_algorithms(G, start, goal)

    df = results_to_df(results)
    save_csv(df, OUTPUT_DIR / "algorithm_comparison.csv")

    # Prints summary table
    print(df[["algorithm_name", "total_path_cost", "nodes_expanded", "execution_time", "path_length", "max_frontier_size", "weight"]])

    plot_bars(df)
    plot_scatters(df)
    plot_weighted_astar(df)
    plot_depth_variants(df)
    plot_routes(G, results, start, goal)


if __name__ == "__main__":
    main()

