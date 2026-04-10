"""Render the Dhaka graph with start/end nodes marked and stats overlay.

- Uses the same graph generation and endpoint selection as run_weighted_astar.py
- Saves PNG to images/map_with_terminals.png
- Overlays total node/edge counts on the figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from run_weighted_astar import generate_graph, euclidean


def choose_start_goal(G: nx.MultiDiGraph):
    """Use the same farthest-pair selection as the main codebase.

    - If graph is not weakly connected, keep the largest weakly connected component
    - Pick an initial node, then choose the farthest node as goal
    - Choose the farthest node from that goal as the start
    Returns (start, goal, possibly-trimmed graph)
    """

    if not nx.is_weakly_connected(G):
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    nodes = list(G.nodes)
    start = nodes[0]
    goal = max(nodes, key=lambda n: euclidean(G, start, n))
    start = max(nodes, key=lambda n: euclidean(G, n, goal))
    return start, goal, G

OUTPUT_PATH = Path(__file__).parent / "images" / "map_with_terminals.png"


def main():
    G = generate_graph()
    start, goal, G = choose_start_goal(G)

    print(f"[info] Graph: {G.graph.get('graph_label')} | nodes={len(G)} | edges={len(G.edges())}")
    print(f"[info] Start={start} | Goal={goal}")

    # Plot with osmnx; highlight start/goal
    fig, ax = ox.plot_graph(
        G,
        node_size=6,
        node_color="#4c6ef5",
        edge_color="#888",
        edge_linewidth=0.8,
        show=False,
        close=False,
    )

    # Overlay start and goal as larger scatter points
    start_x, start_y = G.nodes[start]["x"], G.nodes[start]["y"]
    goal_x, goal_y = G.nodes[goal]["x"], G.nodes[goal]["y"]
    ax.scatter([start_x], [start_y], s=40, color="lime", edgecolors="black", zorder=5, label="start")
    ax.scatter([goal_x], [goal_y], s=40, color="red", edgecolors="black", zorder=5, label="goal")

    # Text box with stats
    stats = f"Nodes: {len(G)}\nEdges: {len(G.edges())}"
    ax.text(
        0.02,
        0.98,
        stats,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#333"),
    )

    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"[done] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


def choose_start_goal(G: nx.MultiDiGraph):
    """Use the same farthest-pair selection as the main codebase.

    - If graph is not weakly connected, keep the largest weakly connected component
    - Pick an initial node, then choose the farthest node as goal
    - Choose the farthest node from that goal as the start
    Returns (start, goal, possibly-trimmed graph)
    """

    if not nx.is_weakly_connected(G):
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    nodes = list(G.nodes)
    start = nodes[0]
    goal = max(nodes, key=lambda n: euclidean(G, start, n))
    start = max(nodes, key=lambda n: euclidean(G, n, goal))
    return start, goal, G
