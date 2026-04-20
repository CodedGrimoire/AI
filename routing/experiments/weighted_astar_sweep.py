"""Weighted A* comparative sweep on Dhaka map (full or node-limited)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from routing.experiments.graph_scope import load_graph_with_costs
from routing.heuristics.spatial import euclidean_heuristic, exponential_feature_heuristic
from routing.algorithms import (
    uniform_cost_search,
    weighted_a_star_search,
    compute_path_cost,
)


@dataclass
class RunResult:
    w: float
    cost: float
    expanded: int
    time_s: float
    path_found: bool


def choose_start_goal(G):
    """Pick far-apart endpoints for stable comparisons."""
    nodes = list(G.nodes)
    start = nodes[0]
    goal = max(nodes, key=lambda n: euclidean_heuristic(G, start, n, 1.0))
    start = max(nodes, key=lambda n: euclidean_heuristic(G, n, goal, 1.0))
    return start, goal


def run_sweep(weights: List[float], *, max_nodes: int | None, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    G = load_graph_with_costs(use_osm=True, max_nodes=max_nodes, seed=seed)
    start, goal = choose_start_goal(G)

    d_res = uniform_cost_search(G, start, goal)
    d_cost = d_res.total_path_cost

    h_fn = exponential_feature_heuristic(G, goal)

    results: List[RunResult] = []
    for w in weights:
        res = weighted_a_star_search(G, start, goal, h_fn, w=w)
        cost = compute_path_cost(G, res.path)
        results.append(
            RunResult(
                w=w,
                cost=cost,
                expanded=res.nodes_expanded,
                time_s=res.execution_time,
                path_found=res.path_found,
            )
        )

    ws = [r.w for r in results]
    costs = [r.cost for r in results]
    expansions = [r.expanded for r in results]
    times = [r.time_s for r in results]
    accuracies = [c / d_cost if d_cost > 0 else float("inf") for c in costs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(ws, costs, marker="o", color="tab:blue")
    axes[0].axhline(d_cost, color="black", linestyle="--", label="UCS baseline")
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
    plt.savefig(output_dir / "wa_sweep_metrics.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(6, 4))
    plt.bar([str(w) for w in ws], accuracies, color="tab:purple")
    plt.axhline(1.0, color="black", linestyle="--", label="Optimal (UCS)")
    plt.ylabel("Cost / optimal")
    plt.xlabel("Weight w")
    plt.title("WA* accuracy vs optimal cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wa_sweep_accuracy.png", dpi=180)
    plt.close()

    df = pd.DataFrame(
        [
            {
                "weight": r.w,
                "cost": r.cost,
                "cost_over_optimal": (r.cost / d_cost if d_cost > 0 else float("inf")),
                "nodes_expanded": r.expanded,
                "time_s": r.time_s,
                "path_found": r.path_found,
            }
            for r in results
        ]
    )
    df.to_csv(output_dir / "wa_sweep_results.csv", index=False)

    scope = "full_map" if max_nodes is None else f"~{max_nodes}_nodes"
    print(f"[info] Scope={scope} | nodes={len(G)} edges={len(G.edges())} start={start} goal={goal}")
    print(
        "Baseline (UCS): cost={:.2f}, expanded={}, time={:.4f}s".format(
            d_res.total_path_cost,
            d_res.nodes_expanded,
            d_res.execution_time,
        )
    )
    print(f"[result] Saved: {output_dir / 'wa_sweep_metrics.png'}")
    print(f"[result] Saved: {output_dir / 'wa_sweep_accuracy.png'}")
    print(f"[result] Saved: {output_dir / 'wa_sweep_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted A* comparative sweep")
    parser.add_argument("--max-nodes", type=int, default=None, help="Limit graph to approximately this many nodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for node-limited sampling")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images/weighted_astar_sweep",
        help="Directory to write plots/results",
    )
    args = parser.parse_args()

    weights = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
    run_sweep(weights, max_nodes=args.max_nodes, seed=args.seed, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
