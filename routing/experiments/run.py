"""Run end-to-end routing experiments on an OSMnx graph.

Steps:
1) Generate graph
2) Assign synthetic features and custom costs
3) Pick start/goal
4) Run Dijkstra, Greedy, A*, Weighted A*
5) Print comparison table
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib

# Force non-interactive backend to prevent GUI hangs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from routing.data.graph_builder import generate_graph
from routing.data.features_costs import assign_synthetic_features, apply_cost
from routing.heuristics.spatial import euclidean_heuristic
from routing.viz.plotting import plot_all_routes, plot_single_route
from routing.algorithms.search import (
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

OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)


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


def plot_cost_vs_expansion(results, filename: str | Path | None = None):
    colors = {
        "Dijkstra": "#1f77b4",
        "Greedy": "#ff7f0e",
        "A*": "#2ca02c",
        "Weighted A*": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, k, w, cost, expd, _, _, _ in results:
        label = name if name != "WA*" else "Weighted A*"
        color = colors.get(label, "gray")
        size = 70 if label == "Greedy" else 50
        ax.scatter(expd, cost, color=color, s=size, zorder=3)
        offset = (6, -10) if label == "Greedy" else (4, 4)
        ax.annotate(label, (expd, cost), textcoords="offset points", xytext=offset, fontsize=9, weight="bold" if label == "Greedy" else "normal")

    ax.set_xlabel("Nodes Expanded")
    ax.set_ylabel("Path Cost")
    ax.set_title("Cost vs Nodes Expanded (Efficiency vs Quality)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=9)
    out_path = Path(filename) if filename is not None else OUTPUT_DIR / "cost_vs_expansion.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _extract_baselines(results):
    """Return ordered dict for key algorithms used in comparisons."""
    order = [
        ("Dijkstra", None, None),
        ("Greedy", None, None),
        ("A*", 1.0, None),
        ("WA*", None, 1.5),
    ]
    out = {}
    for alg, k_req, w_req in order:
        for name, k, w, cost, expd, _, _, _ in results:
            if name != alg:
                continue
            if k_req is not None and k != k_req:
                continue
            if w_req is not None and w != w_req:
                continue
            out["Weighted A*" if alg == "WA*" else alg] = (cost, expd)
            break
    return out


def plot_linear_vs_extended(results_linear, results_extended, filename: str | Path | None = None):
    lin = _extract_baselines(results_linear)
    ext = _extract_baselines(results_extended)

    algorithms = ["Dijkstra", "Greedy", "A*", "Weighted A*"]
    lin_costs = [lin.get(a, (0, 0))[0] for a in algorithms]
    ext_costs = [ext.get(a, (0, 0))[0] for a in algorithms]
    lin_exp = [lin.get(a, (0, 0))[1] for a in algorithms]
    ext_exp = [ext.get(a, (0, 0))[1] for a in algorithms]

    x = range(len(algorithms))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=False)

    bars_lin_cost = axes[0].bar([i - width / 2 for i in x], lin_costs, width, label="Linear", color="#4b8bbe")
    bars_ext_cost = axes[0].bar([i + width / 2 for i in x], ext_costs, width, label="Extended", color="#ff7f0e")
    axes[0].set_title("Path Cost")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(algorithms, rotation=20, ha="right")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # If one model triggers threshold (very large), switch to log to keep bars visible
    max_cost = max(lin_costs + ext_costs)
    mean_cost = sum(lin_costs + ext_costs) / len(lin_costs + ext_costs)
    if max_cost > 5 * mean_cost:
        axes[0].set_yscale("log")

    axes[1].bar([i - width / 2 for i in x], lin_exp, width, label="Linear", color="#4b8bbe")
    axes[1].bar([i + width / 2 for i in x], ext_exp, width, label="Extended", color="#ff7f0e")
    axes[1].set_title("Nodes Expanded")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(algorithms, rotation=20, ha="right")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Linear vs Extended Cost Model Comparison", fontsize=12, y=0.98)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(filename) if filename is not None else OUTPUT_DIR / "linear_vs_extended.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weight_sweeps(results, filename: str | Path | None = None):
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

    out_path = Path(filename) if filename is not None else OUTPUT_DIR / "wa_sweep.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    center = (23.746, 90.376)  # Dhaka example
    # Smaller graph for quicker runs while keeping variability
    G = generate_graph(center, min_nodes=100, max_nodes=140)

    assign_synthetic_features(G)

    # --- Linear model run (legacy default) ---
    apply_cost(G, DEFAULT_WEIGHTS, use_nonlinear=False, use_threshold=False)
    start, goal, G_conn = choose_start_goal(G)
    results_linear = run_all(G_conn, start, goal)
    print_table(results_linear)
    plot_weight_sweeps(results_linear)

    # Scatter efficiency vs quality
    plot_cost_vs_expansion(results_linear)

    # --- Extended model run ---
    apply_cost(G_conn, DEFAULT_WEIGHTS, use_nonlinear=True, use_threshold=True)
    results_extended = run_all(G_conn, start, goal)

    # Comparison chart between linear and extended
    plot_linear_vs_extended(results_linear, results_extended)

    # Route overlays and per-route PNGs (baseline variants)
    base_paths = {
        "Dijkstra": next(p for p in results_linear if p[0] == "Dijkstra")[7],
        "Greedy": next(p for p in results_linear if p[0] == "Greedy")[7],
        "A*": next(p for p in results_linear if p[0] == "A*" and p[1] == 1.0)[7],
        "Weighted A*": next(p for p in results_linear if p[0] == "WA*" and p[2] == 1.5)[7],
    }
    plot_all_routes(G_conn, base_paths, start, goal)
    plot_single_route(G_conn, base_paths["Dijkstra"], "Dijkstra", color="blue", filename=OUTPUT_DIR / "dijkstra.png")
    plot_single_route(G_conn, base_paths["Greedy"], "Greedy", color="orange", filename=OUTPUT_DIR / "greedy.png")
    plot_single_route(G_conn, base_paths["A*"], "A*", color="green", filename=OUTPUT_DIR / "astar.png")
    plot_single_route(G_conn, base_paths["Weighted A*"], "Weighted A*", color="red", filename=OUTPUT_DIR / "weighted_astar.png")

    # Charts: expanded, time, cost vs Dijkstra baseline (using base variants from linear run)
    df_lin = pd.DataFrame(results_linear, columns=["alg", "k", "w", "cost", "expanded", "time", "plen", "path"])
    base_cost = df_lin[df_lin.alg == "Dijkstra"].iloc[0]["cost"]

    # Select base variants for bars: Dijkstra, Greedy, A* (k=1), WA* (w=1.5)
    base_rows = df_lin[
        ((df_lin.alg == "Dijkstra"))
        | ((df_lin.alg == "Greedy"))
        | ((df_lin.alg == "A*") & (df_lin.k == 1.0))
        | ((df_lin.alg == "WA*") & (df_lin.w == 1.5))
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
    plt.savefig(OUTPUT_DIR / "alg_comparison_bars.png", dpi=150)
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
    plt.savefig(OUTPUT_DIR / "accuracy_vs_dijkstra.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
