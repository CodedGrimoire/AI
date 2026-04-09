"""Route plotting utilities for comparing search algorithms on an OSMnx graph."""

from __future__ import annotations

from pathlib import Path

import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx


COLORS = {
    "Dijkstra": "blue",
    "Greedy": "orange",
    "A*": "green",
    "Weighted A*": "red",
    "Breadth-first search (BFS)": "#1f77b4",
    "Uniform cost search": "#9467bd",
    "Depth-first search (DFS)": "#8c564b",
    "Depth Limited Search": "#e377c2",
    "Iterative Deepening Search": "#7f7f7f",
    "Bidirectional Search": "#bcbd22",
    "Greedy best-first search": "#ff7f0e",
    "A* search": "#2ca02c",
}

OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_single_route(G: nx.MultiDiGraph, path, name: str, color: str | None, filename: str | None = None):
    """Render a single algorithm route with visible nodes, start/goal markers, and a heading."""
    if not path:
        print(f"[warn] {name}: empty path, skipping plot")
        return

    color = color or COLORS.get(name, "black")
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}

    fig, ax = plt.subplots(figsize=(5, 5))

    # Edges (light gray)
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=1, zorder=1)

    # Nodes
    xs, ys = zip(*pos.values())
    ax.scatter(xs, ys, c="#888", s=12, zorder=2, label="Nodes")

    # Path
    path_x = [pos[n][0] for n in path]
    path_y = [pos[n][1] for n in path]
    ax.plot(path_x, path_y, color=color, linewidth=3, zorder=3, label=name)

    # Start/Goal markers
    start, goal = path[0], path[-1]
    ax.scatter(*pos[start], c="green", s=50, zorder=4, label="Start")
    ax.scatter(*pos[goal], c="red", s=50, zorder=4, label="Goal")

    graph_label = G.graph.get("graph_label", "OSM graph")
    ax.set_title(f"{name} on {graph_label} | nodes: {len(G)} | edges: {len(G.edges())}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="best", frameon=False)

    fname = Path(filename) if filename else OUTPUT_DIR / f"{name.lower().replace(' ', '_').replace('*','star')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_routes(G: nx.MultiDiGraph, paths_dict: dict, start, goal, filename: Path | str | None = None):
    """Overlay all algorithm paths on one plot with start/goal markers."""
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}

    fig, ax = plt.subplots(figsize=(6, 6))

    # Base edges
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=1, zorder=1)

    # Overlay each route
    for name, path in paths_dict.items():
        if not path:
            print(f"[warn] {name}: empty path, skipping overlay")
            continue
        color = COLORS.get(name, "black")
        path_x = [pos[n][0] for n in path]
        path_y = [pos[n][1] for n in path]
        ax.plot(path_x, path_y, color=color, linewidth=3, zorder=3, label=name)

    # Mark start/goal
    ax.scatter(*pos[start], c="green", s=50, zorder=4, label="Start")
    ax.scatter(*pos[goal], c="red", s=50, zorder=4, label="Goal")

    ax.set_title(f"Routes overlay | nodes: {len(G)} | edges: {len(G.edges())}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    out_path = Path(filename) if filename is not None else OUTPUT_DIR / "comparison_routes.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_explored_nodes(G: nx.MultiDiGraph, expanded_nodes, path, name: str, filename: str | Path | None = None):
    """Visualize explored nodes and final path."""
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}

    fig, ax = plt.subplots(figsize=(5, 5))
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=0.8, zorder=1)

    if expanded_nodes:
        xs = [pos[n][0] for n in expanded_nodes if n in pos]
        ys = [pos[n][1] for n in expanded_nodes if n in pos]
        ax.scatter(xs, ys, c="#999", s=10, alpha=0.6, label="Expanded", zorder=2)

    if path:
        path_x = [pos[n][0] for n in path]
        path_y = [pos[n][1] for n in path]
        ax.plot(path_x, path_y, color=COLORS.get(name, "blue"), linewidth=2.5, zorder=3, label="Path")
        ax.scatter(path_x[0], path_y[0], c="green", s=40, zorder=4, label="Start")
        ax.scatter(path_x[-1], path_y[-1], c="red", s=40, zorder=4, label="Goal")

    ax.set_title(f"Explored nodes: {name}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="best", frameon=False)

    out_path = Path(filename) if filename is not None else OUTPUT_DIR / f"explored_{name.replace(' ', '_').replace('*','star')}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Example usage placeholder; plug in your own graph and paths.
    print("Run plot_all_routes(G, paths_dict, start, goal) from your experiment code.")
