"""Run end-to-end routing experiments on an OSMnx graph.

Steps:
1) Generate graph
2) Assign synthetic features and custom costs
3) Pick start/goal
4) Run Dijkstra, Greedy, A*, Weighted A*
5) Print comparison table
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from route_planning.graph_builder import generate_graph
from route_planning.features_costs import assign_synthetic_features, apply_cost
from route_planning.heuristics import euclidean_heuristic
from route_plotting import plot_all_routes, plot_single_route
from search_algorithms import (
    dijkstra_search,
    greedy_best_first_search,
    a_star_search,
    weighted_a_star_search,
    compute_path_cost,
)


DEFAULT_WEIGHTS = {
    "w_distance": 1.0,
    "w_accident": 35.0,
    "w_traffic": 20.0,
    "w_bump": 12.0,
    "w_safety": 15.0,
}


def choose_start_goal(G: nx.MultiDiGraph):
    # ensure connectivity: use largest weakly connected component
    if not nx.is_weakly_connected(G):
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    nodes = list(G.nodes)
    # choose farthest pair to get opposite ends
    start = nodes[0]
    goal = max(nodes, key=lambda n: euclidean_heuristic(G, start, n, 1.0))
    # refine: pick farthest from that goal for start to maximize separation
    start = max(nodes, key=lambda n: euclidean_heuristic(G, n, goal, 1.0))
    return start, goal, G


def run_all(G: nx.MultiDiGraph, start, goal):
    h_fn = lambda n: euclidean_heuristic(G, n, goal, scale=1.0)

    runs = [
        ("Dijkstra", None, None, lambda: dijkstra_search(G, start, goal, heuristic=lambda n: 0.0)),
        ("Greedy", None, None, lambda: greedy_best_first_search(G, start, goal, h_fn)),
    ]

    # A* with different heuristic scales
    for k in [0.5, 1.0, 1.5]:
        h_k = lambda n, k=k: euclidean_heuristic(G, n, goal, scale=k)
        runs.append(("A*", k, None, lambda h=h_k: a_star_search(G, start, goal, h)))

    # Weighted A* with different weights (k fixed at 1.0)
    h1 = lambda n: euclidean_heuristic(G, n, goal, scale=1.0)
    for w in [1.0, 1.5, 2.0, 3.0]:
        runs.append(("WA*", 1.0, w, lambda w=w: weighted_a_star_search(G, start, goal, h1, w=w)))

    results = []
    for name, k, w, fn in runs:
        path, g_cost, expanded, elapsed = fn()
        total_cost = compute_path_cost(G, path)
        results.append((name, k, w, total_cost, expanded, elapsed, len(path), path))
    return results


def print_table(results):
    print("\nAlgorithm Comparison")
    print("{:<8} {:>4} {:>4} {:>12} {:>12} {:>10} {:>10}".format("Alg", "k", "w", "Path Cost", "Expanded", "Time(s)", "Path len"))
    for name, k, w, cost, expd, t, plen, _ in results:
        print(f"{name:<8} {str(k):>4} {str(w):>4} {cost:>12.2f} {expd:>12d} {t:>10.4f} {plen:>10d}")


def plot_weight_sweeps(results):
    df = pd.DataFrame(results, columns=["alg", "k", "w", "cost", "expanded", "time", "plen", "path"])
    wa = df[df.alg == "WA*"]
    if wa.empty:
        return

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(wa["w"], wa["cost"], marker="o")
    ax[0].set_xlabel("w")
    ax[0].set_ylabel("Path cost")
    ax[0].set_title("WA*: cost vs w")

    ax[1].plot(wa["w"], wa["expanded"], marker="o", color="darkgreen")
    ax[1].set_xlabel("w")
    ax[1].set_ylabel("Nodes expanded")
    ax[1].set_title("WA*: expansions vs w")

    plt.tight_layout()
    plt.savefig("wa_sweep.png", dpi=150)
    plt.close(fig)


def main():
    center = (23.746, 90.376)  # Dhaka example
    G = generate_graph(center, min_nodes=120, max_nodes=180)

    assign_synthetic_features(G)
    apply_cost(G, DEFAULT_WEIGHTS)

    start, goal, G_conn = choose_start_goal(G)
    results = run_all(G_conn, start, goal)
    print_table(results)
    plot_weight_sweeps(results)

    # Route overlays and per-route PNGs (baseline variants)
    base_paths = {
        "Dijkstra": next(p for p in results if p[0] == "Dijkstra")[7],
        "Greedy": next(p for p in results if p[0] == "Greedy")[7],
        "A*": next(p for p in results if p[0] == "A*" and p[1] == 1.0)[7],
        "Weighted A*": next(p for p in results if p[0] == "WA*" and p[2] == 1.5)[7],
    }
    plot_all_routes(G_conn, base_paths, start, goal)
    plot_single_route(G_conn, base_paths["Dijkstra"], "Dijkstra", color="blue", filename="dijkstra.png")
    plot_single_route(G_conn, base_paths["Greedy"], "Greedy", color="orange", filename="greedy.png")
    plot_single_route(G_conn, base_paths["A*"], "A*", color="green", filename="astar.png")
    plot_single_route(G_conn, base_paths["Weighted A*"], "Weighted A*", color="red", filename="weighted_astar.png")

    # Charts: expanded, time, cost vs Dijkstra baseline (using base variants)
    df = pd.DataFrame(results, columns=["alg", "k", "w", "cost", "expanded", "time", "plen", "path"])
    base_cost = df[df.alg == "Dijkstra"].iloc[0]["cost"]

    # Select base variants for bars: Dijkstra, Greedy, A* (k=1), WA* (w=1.5)
    base_rows = df[
        ((df.alg == "Dijkstra"))
        | ((df.alg == "Greedy"))
        | ((df.alg == "A*") & (df.k == 1.0))
        | ((df.alg == "WA*") & (df.w == 1.5))
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].bar(base_rows["alg"], base_rows["expanded"], color="skyblue")
    axes[0].set_title("Nodes Expanded")
    axes[1].bar(base_rows["alg"], base_rows["time"], color="salmon")
    axes[1].set_title("Time (s)")
    axes[2].bar(base_rows["alg"], base_rows["cost"], color="seagreen")
    axes[2].axhline(base_cost, color="black", linestyle="--", label="Dijkstra cost")
    axes[2].set_title("Path Cost vs Dijkstra")
    axes[2].legend()
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("alg_comparison_bars.png", dpi=150)
    plt.close(fig)

    # Accuracy plot: cost ratio vs Dijkstra
    base_costs = base_rows.set_index("alg")["cost"]
    ratios = base_costs / base_cost
    plt.figure(figsize=(4,3))
    ratios.plot(kind="bar", color="purple")
    plt.axhline(1.0, color="black", linestyle="--", label="Dijkstra")
    plt.ylabel("Cost / Dijkstra")
    plt.title("Accuracy vs Dijkstra")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_dijkstra.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
