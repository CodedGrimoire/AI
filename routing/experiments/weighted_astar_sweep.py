"""Weighted A* sweep utility.

Runs Weighted A* across a range of weights (at least 10) on the OSM/Dhaka
graph used elsewhere in the repo, and saves PNG charts for cost, speed, and
node expansions. Also reports accuracy relative to the optimal (Dijkstra)
path cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib

# Force non-interactive backend so plotting works in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from routing.data.graph_builder import generate_graph
from routing.data.features_costs import assign_synthetic_features, apply_cost
from routing.heuristics.spatial import euclidean_heuristic, exponential_feature_heuristic
from routing.algorithms import (
    uniform_cost_search,
    weighted_a_star_search,
    compute_path_cost,
)


OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class RunResult:
    w: float
    cost: float
    expanded: int
    time_s: float


def choose_start_goal(G: nx.MultiDiGraph):
    """Same heuristic-based farthest-pair selection as run_experiments."""
    if not nx.is_weakly_connected(G):
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    nodes = list(G.nodes)
    start = nodes[0]
    goal = max(nodes, key=lambda n: euclidean_heuristic(G, start, n, 1.0))
    start = max(nodes, key=lambda n: euclidean_heuristic(G, n, goal, 1.0))
    return start, goal, G


def run_sweep(weights: List[float]) -> None:
    G = generate_graph()
    assign_synthetic_features(G)
    apply_cost(G)

    start, goal, G = choose_start_goal(G)

    # Optimal baseline with Uniform Cost Search (distance-only)
    d_res = uniform_cost_search(G, start, goal)
    d_cost = d_res.total_path_cost

    h_fn = exponential_feature_heuristic(G, goal)

    results: List[RunResult] = []
    for w in weights:
        path, g_cost, expanded, elapsed = weighted_a_star_search(G, start, goal, h_fn, w=w)
        cost = compute_path_cost(G, path)
        results.append(RunResult(w=w, cost=cost, expanded=expanded, time_s=elapsed))

    # --- Plotting ---
    ws = [r.w for r in results]
    costs = [r.cost for r in results]
    expansions = [r.expanded for r in results]
    times = [r.time_s for r in results]
    accuracies = [c / d_cost if d_cost > 0 else float("inf") for c in costs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(ws, costs, marker="o", color="tab:blue")
    axes[0].axhline(d_cost, color="black", linestyle="--", label="Dijkstra cost")
    axes[0].set_xlabel("Weight w")
    axes[0].set_ylabel("Path cost")
    axes[0].set_title("WA* cost vs w")
    axes[0].legend()

    axes[1].plot(ws, expansions, marker="o", color="tab:green")
    axes[1].set_xlabel("Weight w")
    axes[1].set_ylabel("Nodes expanded")
    axes[1].set_title("WA* expansions vs w")

    axes[2].plot(ws, times, marker="o", color="tab:orange")
    axes[2].set_xlabel("Weight w")
    axes[2].set_ylabel("Time (s)")
    axes[2].set_title("WA* runtime vs w")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wa_sweep_metrics.png", dpi=180)
    plt.close(fig)

    # Accuracy bar chart
    plt.figure(figsize=(6, 4))
    plt.bar([str(w) for w in ws], accuracies, color="tab:purple")
    plt.axhline(1.0, color="black", linestyle="--", label="Optimal (Dijkstra)")
    plt.ylabel("Cost / optimal")
    plt.xlabel("Weight w")
    plt.title("WA* accuracy vs optimal cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wa_sweep_accuracy.png", dpi=180)
    plt.close()

    # Text summary for quick CLI view
    print("Baseline (Dijkstra): cost={:.2f}, expanded={}, time={:.4f}s".format(d_cost, d_exp, d_time))
    print("w, cost, cost/opt, expanded, time_s")
    for r, acc in zip(results, accuracies):
        print(f"{r.w:.3g}, {r.cost:.2f}, {acc:.3f}, {r.expanded}, {r.time_s:.4f}")


if __name__ == "__main__":
    # 10+ weights from 0.5 to 5.0 inclusive
    weights = [round(w, 2) for w in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]]
    run_sweep(weights)
