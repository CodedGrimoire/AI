"""Visualization utilities for informed safe-passage experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx


PALETTE = {
    "Greedy best-first search": "#d95f02",
    "A* search": "#1b9e77",
    "Weighted A*": "#7570b3",
}


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    out = {
        "tables": base_dir / "tables",
        "plots": base_dir / "plots",
        "routes": base_dir / "routes",
        "summary": base_dir / "summary",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def save_bar_chart(df_mean: pd.DataFrame, metric: str, out_path: Path, title: str, higher_better: bool = True) -> None:
    plot_df = df_mean.sort_values(metric, ascending=not higher_better)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [PALETTE.get(a, "#666") for a in plot_df["algorithm_name"]]
    ax.bar(plot_df["algorithm_name"], plot_df[metric], color=colors)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo, g in df.groupby("algorithm_name"):
        ax.scatter(g[x], g[y], label=algo, alpha=0.8, s=25, c=PALETTE.get(algo, "#444"))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_radar_chart(df_mean: pd.DataFrame, metrics: Sequence[str], out_path: Path) -> None:
    if df_mean.empty:
        return

    norm = df_mean.copy()
    for m in metrics:
        lo, hi = norm[m].min(), norm[m].max()
        if hi - lo < 1e-12:
            norm[m] = 0.5
        else:
            # For risk metrics lower is better -> invert for radar desirability
            if m in {"average_traffic", "average_accident_risk", "average_bumpiness"}:
                norm[m] = (hi - norm[m]) / (hi - lo)
            else:
                norm[m] = (norm[m] - lo) / (hi - lo)

    labels = list(metrics)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for _, row in norm.iterrows():
        vals = [float(row[m]) for m in labels]
        vals += vals[:1]
        ax.plot(angles, vals, label=row["algorithm_name"], linewidth=2)
        ax.fill(angles, vals, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Safety-profile radar (normalized desirability)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_route_overlay(
    G: nx.MultiDiGraph,
    routes: dict[str, list],
    start,
    goal,
    out_path: Path,
    title: str,
) -> None:
    route_list = []
    colors = []
    labels = []
    for algo, path in routes.items():
        if path and len(path) >= 2:
            route_list.append(path)
            colors.append(PALETTE.get(algo, "#444"))
            labels.append(algo)

    if not route_list:
        return

    fig, ax = ox.plot_graph_routes(
        G,
        route_list,
        route_colors=colors,
        route_linewidths=2.8,
        route_alpha=0.92,
        orig_dest_size=0,
        bgcolor="white",
        node_size=0,
        edge_color="#d9d9d9",
        edge_linewidth=0.5,
        show=False,
        close=False,
        figsize=(8, 8),
    )

    sx, sy = G.nodes[start]["x"], G.nodes[start]["y"]
    gx, gy = G.nodes[goal]["x"], G.nodes[goal]["y"]
    ax.scatter([sx], [sy], c="#2ca02c", s=50, label="start", zorder=4)
    ax.scatter([gx], [gy], c="#d62728", s=50, label="goal", zorder=4)

    for algo, color in zip(labels, colors):
        ax.plot([], [], color=color, linewidth=2.8, label=algo)

    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
