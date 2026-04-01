"""Route plotting utilities for comparing search algorithms on an OSMnx graph."""

from __future__ import annotations

import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx


COLORS = {
    "Dijkstra": "blue",
    "Greedy": "orange",
    "A*": "green",
    "Weighted A*": "red",
}


def plot_single_route(G: nx.MultiDiGraph, path, name: str, color: str | None):
    if not path:
        print(f"[warn] {name}: empty path, skipping plot")
        return
    color = color or COLORS.get(name, "black")
    fig, ax = ox.plot_graph_route(
        G,
        path,
        route_color=color,
        route_linewidth=3,
        node_size=0,
        bgcolor="white",
        show=False,
        close=False,
    )
    ax.set_title(name)
    plt.savefig(f"{name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_routes(G: nx.MultiDiGraph, paths_dict: dict, start, goal):
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color="#888",
        bgcolor="white",
        show=False,
        close=False,
    )

    # Plot each path
    for name, path in paths_dict.items():
        if not path:
            print(f"[warn] {name}: empty path, skipping overlay")
            continue
        color = COLORS.get(name, "black")
        ox.plot_graph_route(
            G,
            path,
            route_color=color,
            route_linewidth=3,
            node_size=0,
            bgcolor="white",
            ax=ax,
            show=False,
            close=False,
        )

    # Mark start/goal
    ax.scatter(G.nodes[start]["x"], G.nodes[start]["y"], c="green", s=40, label="Start", zorder=5)
    ax.scatter(G.nodes[goal]["x"], G.nodes[goal]["y"], c="red", s=40, label="Goal", zorder=5)

    # Legend
    handles = []
    for name, color in COLORS.items():
        handles.append(plt.Line2D([0], [0], color=color, lw=3, label=name))
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Start"))
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Goal"))
    ax.legend(handles=handles, loc="best")

    plt.savefig("comparison_routes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Example usage placeholder; plug in your own graph and paths.
    print("Run plot_all_routes(G, paths_dict, start, goal) from your experiment code.")
