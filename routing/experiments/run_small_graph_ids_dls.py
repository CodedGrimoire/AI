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
from routing.data.graph_builder import generate_graph
from routing.data.features_costs import assign_synthetic_features, apply_cost

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from routing.algorithms import (
    SearchResult,
    breadth_first_search,
    depth_limited_search,
    iterative_deepening_search,
)
from routing.experiments.output_paths import build_output_paths


def make_small_grid(rows: int = 10, cols: int = 10, spacing: float = 30.0) -> nx.MultiDiGraph:
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


def make_osm_subgraph(target_nodes: int = 200) -> nx.MultiDiGraph:
    """Load OSM graph and keep a connected subgraph around target_nodes."""
    if target_nodes < 4:
        raise ValueError("target_nodes must be >= 4")

    G = generate_graph(use_osm=True)
    assign_synthetic_features(G)
    apply_cost(G)

    if len(G) <= target_nodes:
        return G

    # Expand from a deterministic seed node and keep the first target_nodes
    # reached by BFS over an undirected view to preserve local connectivity.
    UG = G.to_undirected(as_view=True)
    seed = next(iter(UG.nodes))
    bfs_tree = nx.bfs_tree(UG, seed)
    selected_nodes = list(bfs_tree.nodes())[:target_nodes]
    H = G.subgraph(selected_nodes).copy()

    # Keep one connected component suitable for directed routing.
    if H.is_directed():
        comp = max(nx.strongly_connected_components(H), key=len)
    else:
        comp = max(nx.connected_components(H), key=len)
    H = H.subgraph(comp).copy()
    H.graph["graph_label"] = f"Dhaka OSM subgraph (~{target_nodes} nodes)"
    return H


def dims_for_target_nodes(target_nodes: int) -> tuple[int, int]:
    """Choose near-square grid dimensions for a target node count."""
    if target_nodes < 4:
        raise ValueError("target_nodes must be >= 4")
    rows = max(2, math.isqrt(target_nodes))
    cols = max(2, math.ceil(target_nodes / rows))
    return rows, cols


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


def plot_ids_summary(ids: SearchResult, bfs: SearchResult, out_png: Path) -> None:
    """Plot a compact IDS vs BFS comparison for quick validation."""
    if plt is None:
        return

    labels = ["path_length", "nodes_expanded", "execution_time_s", "cost"]
    ids_vals = [ids.path_length, ids.nodes_expanded, ids.execution_time, ids.total_path_cost]
    bfs_vals = [bfs.path_length, bfs.nodes_expanded, bfs.execution_time, bfs.total_path_cost]

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(8, 4))
    plt.bar([i - width / 2 for i in x], bfs_vals, width=width, label="BFS baseline")
    plt.bar([i + width / 2 for i in x], ids_vals, width=width, label="IDS")
    plt.xticks(x, labels)
    plt.title("IDS Summary (vs BFS baseline)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def plot_map_used(G: nx.MultiDiGraph, start: Hashable, goal: Hashable, out_png: Path) -> None:
    """Render the graph used for the experiment with start/goal markers."""
    if plt is None:
        return
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}

    plt.figure(figsize=(7, 7))
    for u, v, _k, _data in G.edges(keys=True, data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        plt.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=1.0, zorder=1)

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    plt.scatter(xs, ys, c="#6c757d", s=18, zorder=2, label="Nodes")

    sx, sy = pos[start]
    gx, gy = pos[goal]
    plt.scatter([sx], [sy], c="#2f9e44", s=110, marker="o", edgecolors="black", zorder=3, label="Start")
    plt.scatter([gx], [gy], c="#d9480f", s=130, marker="*", edgecolors="black", zorder=3, label="Goal")

    plt.title(f"Map Used: {G.graph.get('graph_label', 'Graph')}")
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def plot_path_on_map(
    G: nx.MultiDiGraph,
    path: list[Hashable],
    start: Hashable,
    goal: Hashable,
    title: str,
    out_png: Path,
) -> None:
    """Render one algorithm path on top of the graph."""
    if plt is None:
        return
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}

    plt.figure(figsize=(7, 7))
    for u, v, _k, _data in G.edges(keys=True, data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        plt.plot([x0, x1], [y0, y1], color="#e5e5e5", linewidth=0.9, zorder=1)

    if path:
        px = [pos[n][0] for n in path if n in pos]
        py = [pos[n][1] for n in path if n in pos]
        plt.plot(px, py, color="#1c7ed6", linewidth=3.0, zorder=3, label="Path")

    sx, sy = pos[start]
    gx, gy = pos[goal]
    plt.scatter([sx], [sy], c="#2f9e44", s=110, marker="o", edgecolors="black", zorder=4, label="Start")
    plt.scatter([gx], [gy], c="#d9480f", s=130, marker="*", edgecolors="black", zorder=4, label="Goal")

    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Small graph DLS/IDS runner")
    parser.add_argument("--use-osm", action="store_true", help="Use OSMnx Dhaka map instead of synthetic grid")
    parser.add_argument("--rows", type=int, default=10, help="Grid rows")
    parser.add_argument("--cols", type=int, default=10, help="Grid cols")
    parser.add_argument("--nodes", type=int, default=100, help="Target node count (overrides rows/cols when set)")
    parser.add_argument("--start", type=str, default=None, help="Start node id, e.g. '(0,0)'")
    parser.add_argument("--goal", type=str, default=None, help="Goal node id, e.g. '(5,5)'")
    parser.add_argument("--dls-max", type=int, default=15, help="Max depth limit to sweep")
    parser.add_argument("--ids-max", type=int, default=20, help="Max depth for IDS")
    args = parser.parse_args()

    paths = build_output_paths("small_graph_ids_dls")

    if args.use_osm:
        G = make_osm_subgraph(target_nodes=args.nodes)
    else:
        if args.nodes is not None:
            rows, cols = dims_for_target_nodes(args.nodes)
        else:
            rows, cols = args.rows, args.cols
        G = make_small_grid(rows=rows, cols=cols)
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
    dls_final = depth_limited_search(G, start, goal, args.dls_max)
    dls_path = dls_final.path if dls_final.path_found else []
    print(f"[path] DLS(limit={args.dls_max}) path: {dls_path if dls_path else 'NOT FOUND'}")

    ids = iterative_deepening_search(G, start, goal, max_depth=args.ids_max)
    print(f"[path] IDS(max_depth={args.ids_max}) path: {ids.path if ids.path_found else 'NOT FOUND'}")
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
    ids_plot = paths["plots"] / "ids_summary.png"
    plot_ids_summary(ids, bfs, ids_plot)
    map_png = paths["plots"] / "map_used.png"
    dls_path_png = paths["plots"] / "dls_path.png"
    ids_path_png = paths["plots"] / "ids_path.png"
    plot_map_used(G, start, goal, map_png)
    plot_path_on_map(G, dls_path, start, goal, f"DLS Path (limit={args.dls_max})", dls_path_png)
    plot_path_on_map(G, ids.path if ids.path_found else [], start, goal, f"IDS Path (max_depth={args.ids_max})", ids_path_png)

    print(f"[result] DLS sweep saved: {dls_csv}")
    print(f"[result] IDS summary saved: {ids_csv}")
    if plt is None:
        print("[warn] matplotlib not installed; skipped plot generation")
    else:
        print(f"[result] Plot saved: {map_png}")
        print(f"[result] Plot saved: {paths['plots'] / 'dls_depth_sweep.png'}")
        print(f"[result] Plot saved: {dls_path_png}")
        print(f"[result] Plot saved: {ids_path_png}")
        print(f"[result] Plot saved: {ids_plot}")


if __name__ == "__main__":
    main()
