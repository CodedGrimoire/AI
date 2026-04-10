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
import osmnx as ox

from run_weighted_astar import generate_graph, choose_endpoints

OUTPUT_PATH = Path(__file__).parent / "images" / "map_with_terminals.png"


def main():
    G = generate_graph()
    start, goal = choose_endpoints(G)

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
