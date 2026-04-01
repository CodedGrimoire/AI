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

from route_planning.graph_builder import generate_graph
from route_planning.features_costs import assign_synthetic_features, apply_cost
from route_planning.heuristics import euclidean_heuristic
from search_algorithms import (
    dijkstra_search,
    greedy_best_first_search,
    a_star_search,
    weighted_a_star_search,
    compute_path_cost,
)


DEFAULT_WEIGHTS = {
    "w_distance": 1.0,
    "w_accident": 50.0,
    "w_traffic": 40.0,
    "w_bump": 20.0,
    "w_safety": 30.0,
}


def choose_start_goal(G: nx.MultiDiGraph):
    nodes = list(G.nodes)
    start = nodes[0]
    # pick node farthest from start (by Euclidean) for a non-trivial route
    goal = max(nodes, key=lambda n: euclidean_heuristic(G, start, n, 1.0))
    return start, goal


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
        results.append((name, k, w, total_cost, expanded, elapsed, len(path)))
    return results


def print_table(results):
    print("\nAlgorithm Comparison")
    print("{:<8} {:>4} {:>4} {:>12} {:>12} {:>10} {:>10}".format("Alg", "k", "w", "Path Cost", "Expanded", "Time(s)", "Path len"))
    for name, k, w, cost, expd, t, plen in results:
        print(f"{name:<8} {str(k):>4} {str(w):>4} {cost:>12.2f} {expd:>12d} {t:>10.4f} {plen:>10d}")


def plot_weight_sweeps(results):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return

    df = None
    try:
        import pandas as pd
        df = pd.DataFrame(results, columns=["alg", "k", "w", "cost", "expanded", "time", "plen"])
    except Exception:
        return

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
    G = generate_graph(center, min_nodes=50, max_nodes=200)

    assign_synthetic_features(G)
    apply_cost(G, DEFAULT_WEIGHTS)

    start, goal = choose_start_goal(G)
    results = run_all(G, start, goal)
    print_table(results)
    plot_weight_sweeps(results)


if __name__ == "__main__":
    main()
