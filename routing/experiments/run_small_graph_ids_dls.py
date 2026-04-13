"""Small-graph runner focused on Depth-Limited Search and IDS.

Why this exists:
- Uses a much smaller graph so depth-limited behavior is easy to verify.
- Avoids brittle node-id parsing by using literal_eval safely.
- Saves outputs into a dedicated image/csv folder tree.
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Hashable, Optional

import networkx as nx
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from routing.algorithms import (
    breadth_first_search,
    depth_limited_search,
    iterative_deepening_search,
)
from routing.experiments.output_paths import build_output_paths


def make_small_grid(rows: int = 6, cols: int = 6, spacing: float = 30.0) -> nx.MultiDiGraph:
    """Build a small directed grid graph with coordinate attributes."""
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must be >= 2")

    base = nx.grid_2d_graph(rows, cols, create_using=nx.Graph)
    G = nx.MultiDiGraph()
    G.graph["crs"] = "epsg:3857"
    G.graph["graph_label"] = f"Small synthetic grid ({rows}x{cols})"

    for (r, c) in base.nodes:
        G.add_node((r, c), x=float(c * spacing), y=float(r * spacing))
    for u, v in base.edges:
        G.add_edge(u, v, length=spacing, custom_cost=spacing)
        G.add_edge(v, u, length=spacing, custom_cost=spacing)

    return G


def parse_node(raw: Optional[str]) -> Optional[Hashable]:
    """Parse tuple-like node ids safely, e.g. '(0, 0)' or '"(0, 0)"'."""
    if raw is None:
        return None
    try:
        val = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw
    return val


def choose_endpoints(G: nx.MultiDiGraph, start: Optional[Hashable], goal: Optional[Hashable]):
    """Use manual nodes if valid; otherwise pick far-apart corners."""
    if start is not None and goal is not None and start in G and goal in G and start != goal:
        return start, goal

    nodes = list(G.nodes)
    start = nodes[0]
    goal = max(
        nodes,
        key=lambda n: math.hypot(G.nodes[n]["x"] - G.nodes[start]["x"], G.nodes[n]["y"] - G.nodes[start]["y"]),
    )
    return start, goal


def plot_depth_sweep(df: pd.DataFrame, out_png: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(df["depth_limit"], df["nodes_expanded"], marker="o", label="Nodes expanded")
    plt.plot(df["depth_limit"], df["path_found"].astype(int), marker="s", label="Path found (0/1)")
    plt.xlabel("Depth limit")
    plt.title("DLS on Small Graph")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Small graph DLS/IDS runner")
    parser.add_argument("--rows", type=int, default=6, help="Grid rows")
    parser.add_argument("--cols", type=int, default=6, help="Grid cols")
    parser.add_argument("--start", type=str, default=None, help="Start node id, e.g. '(0,0)'")
    parser.add_argument("--goal", type=str, default=None, help="Goal node id, e.g. '(5,5)'")
    parser.add_argument("--dls-max", type=int, default=15, help="Max depth limit to sweep")
    parser.add_argument("--ids-max", type=int, default=20, help="Max depth for IDS")
    args = parser.parse_args()

    paths = build_output_paths("small_graph_ids_dls")

    G = make_small_grid(rows=args.rows, cols=args.cols)
    start, goal = choose_endpoints(G, parse_node(args.start), parse_node(args.goal))

    print(f"[info] Graph={G.graph['graph_label']} | nodes={len(G)} edges={len(G.edges())}")
    print(f"[info] start={start} goal={goal}")

    bfs = breadth_first_search(G, start, goal)
    print(f"[baseline] BFS found={bfs.path_found} | path_length={bfs.path_length} | cost={bfs.total_path_cost}")

    dls_rows = []
    for limit in range(args.dls_max + 1):
        res = depth_limited_search(G, start, goal, limit)
        dls_rows.append(
            {
                "depth_limit": limit,
                "path_found": res.path_found,
                "path_length": res.path_length,
                "nodes_expanded": res.nodes_expanded,
                "execution_time": res.execution_time,
                "cost": res.total_path_cost,
                "cutoff_occurred": res.cutoff_occurred,
            }
        )

    dls_df = pd.DataFrame(dls_rows)
    dls_csv = paths["csv"] / "dls_sweep.csv"
    dls_df.to_csv(dls_csv, index=False)
    plot_depth_sweep(dls_df, paths["plots"] / "dls_depth_sweep.png")

    ids = iterative_deepening_search(G, start, goal, max_depth=args.ids_max)
    ids_df = pd.DataFrame(
        [
            {
                "path_found": ids.path_found,
                "path_length": ids.path_length,
                "nodes_expanded": ids.nodes_expanded,
                "execution_time": ids.execution_time,
                "cost": ids.total_path_cost,
                "depth_reached": ids.depth_reached,
            }
        ]
    )
    ids_csv = paths["csv"] / "ids_summary.csv"
    ids_df.to_csv(ids_csv, index=False)

    print(f"[result] DLS sweep saved: {dls_csv}")
    print(f"[result] IDS summary saved: {ids_csv}")
    if plt is None:
        print("[warn] matplotlib not installed; skipped plot generation")
    else:
        print(f"[result] Plot saved: {paths['plots'] / 'dls_depth_sweep.png'}")


if __name__ == "__main__":
    main()
