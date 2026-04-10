"""Weighted A* sweep on the Dhaka road network.

Self-contained experiment:
- Downloads the Dhaka, Bangladesh drive graph with OSMnx
- Runs Weighted A* for a set of heuristic weights
- Compares cost to optimal Dijkstra (Uniform Cost Search)
- Saves charts under ./images

Falls back to a synthetic grid if OSM download fails so the run still completes.
"""
from __future__ import annotations

import heapq
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

# Headless-friendly backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

# Avoid PROJ downloads in restricted environments
os.environ.setdefault("PROJ_NETWORK", "OFF")

OUTPUT_DIR = Path(__file__).parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SearchResult:
    path: List
    total_path_cost: float
    nodes_expanded: int
    time_s: float


# ----------------------- Graph construction ---------------------------------

def generate_graph() -> nx.MultiDiGraph:
    """Build the Dhaka drive network or a synthetic fallback if download fails."""

    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.timeout = 90

    place_query = "Dhaka, Bangladesh"
    try:
        print(f"[info] Downloading OSM graph for: {place_query} (drive)")
        G = ox.graph_from_place(place_query, network_type="drive", simplify=True)
        G = ox.project_graph(G)

        # Keep largest strongly connected component for directed routing
        try:
            largest_strong = max(nx.strongly_connected_components(G), key=len)
            G = G.subgraph(largest_strong).copy()
            G.graph["component_type"] = "strongly_connected"
        except Exception:
            largest_weak = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest_weak).copy()
            G.graph["component_type"] = "weakly_connected"

        G.graph["graph_label"] = "Dhaka road network (OSM)"
        print(
            f"[info] Dhaka graph ready | Nodes: {len(G)} | Edges: {len(G.edges())} | component: {G.graph.get('component_type')}"
        )
        return G
    except Exception as e:
        print(f"[warn] OSM Dhaka load failed ({e}); using synthetic grid")
        return synthetic_grid(target_nodes=400)


def synthetic_grid(target_nodes: int = 400, spacing: float = 30.0) -> nx.MultiDiGraph:
    """Offline fallback: square grid near the requested size."""

    rows = max(2, math.isqrt(target_nodes))
    cols = max(2, math.ceil(target_nodes / rows))
    base = nx.grid_2d_graph(rows, cols, create_using=nx.Graph)

    extra = len(base) - target_nodes
    if extra > 0:
        max_r = rows - 1
        candidates = sorted([(r, c) for r, c in base.nodes if r == max_r], reverse=True)
        for node in candidates:
            if extra <= 0:
                break
            base.remove_node(node)
            extra -= 1

    G = nx.MultiDiGraph()
    G.graph["crs"] = "epsg:3857"
    G.graph["graph_label"] = "Synthetic grid"
    for (r, c) in base.nodes:
        G.add_node((r, c), x=float(c * spacing), y=float(r * spacing))
    for u, v in base.edges:
        G.add_edge(u, v, length=spacing)
        G.add_edge(v, u, length=spacing)
    return G


# ----------------------- Helpers --------------------------------------------

def euclidean(G: nx.MultiDiGraph, a, b) -> float:
    ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
    bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
    return math.hypot(ax - bx, ay - by)


def edge_cost(G: nx.MultiDiGraph, u, v, key) -> float:
    data = G.edges[u, v, key]
    return float(data.get("length", 1.0))


def compute_path_cost(G: nx.MultiDiGraph, path: Iterable) -> float:
    cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        key = min(G[u][v], key=lambda k: edge_cost(G, u, v, k))
        cost += edge_cost(G, u, v, key)
    return cost


# ----------------------- Searches -------------------------------------------

def uniform_cost_search(G: nx.MultiDiGraph, start, goal) -> SearchResult:
    """Dijkstra for baseline optimal cost."""

    start_t = time.perf_counter()
    frontier = [(0.0, start)]
    g: Dict = {start: 0.0}
    parent: Dict = {}
    expanded = 0

    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost != g[node]:
            continue
        expanded += 1
        if node == goal:
            break
        for _, v, k, data in G.out_edges(node, keys=True, data=True):
            new_cost = cost + float(data.get("length", 1.0))
            if v not in g or new_cost < g[v]:
                g[v] = new_cost
                parent[v] = node
                heapq.heappush(frontier, (new_cost, v))

    end_t = time.perf_counter()
    path = reconstruct_path(parent, start, goal)
    return SearchResult(path=path, total_path_cost=g.get(goal, math.inf), nodes_expanded=expanded, time_s=end_t - start_t)


def weighted_a_star(G: nx.MultiDiGraph, start, goal, w: float) -> SearchResult:
    start_t = time.perf_counter()
    h = lambda n: w * euclidean(G, n, goal)

    frontier = [(h(start), 0.0, start)]  # f, g, node
    g: Dict = {start: 0.0}
    parent: Dict = {}
    closed = set()
    expanded = 0

    while frontier:
        f, g_cur, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            break

        for _, v, k, data in G.out_edges(node, keys=True, data=True):
            step = float(data.get("length", 1.0))
            ng = g_cur + step
            if v in closed and ng >= g.get(v, math.inf):
                continue
            if ng < g.get(v, math.inf):
                g[v] = ng
                parent[v] = node
                nf = ng + h(v)
                heapq.heappush(frontier, (nf, ng, v))

    end_t = time.perf_counter()
    path = reconstruct_path(parent, start, goal)
    return SearchResult(path=path, total_path_cost=g.get(goal, math.inf), nodes_expanded=expanded, time_s=end_t - start_t)


# ----------------------- Utilities ------------------------------------------

def reconstruct_path(parent: Dict, start, goal) -> List:
    if goal not in parent and goal != start:
        return []
    node = goal
    path = [node]
    while node != start:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path


def choose_endpoints(G: nx.MultiDiGraph):
    """Pick a far-apart pair using heuristic farthest search."""
    nodes = list(G.nodes)
    start = nodes[0]
    goal = max(nodes, key=lambda n: euclidean(G, start, n))
    start = max(nodes, key=lambda n: euclidean(G, n, goal))
    return start, goal


# ----------------------- Experiment -----------------------------------------

def plot_results(G: nx.MultiDiGraph, weights: List[float], results: List[SearchResult], baseline: SearchResult):
    ws = weights
    costs = [compute_path_cost(G, r.path) if r.path else math.inf for r in results]
    expansions = [r.nodes_expanded for r in results]
    times = [r.time_s for r in results]
    accuracies = [c / baseline.total_path_cost if baseline.total_path_cost > 0 else math.inf for c in costs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(ws, costs, marker="o", color="tab:blue")
    axes[0].axhline(baseline.total_path_cost, color="black", linestyle="--", label="Dijkstra cost")
    axes[0].set_xlabel("Weight w")
    axes[0].set_ylabel("Path cost")
    axes[0].set_title("Weighted A* cost vs w")
    axes[0].legend()

    axes[1].plot(ws, expansions, marker="o", color="tab:green")
    axes[1].set_xlabel("Weight w")
    axes[1].set_ylabel("Nodes expanded")
    axes[1].set_title("Weighted A* expansions vs w")

    axes[2].plot(ws, times, marker="o", color="tab:orange")
    axes[2].set_xlabel("Weight w")
    axes[2].set_ylabel("Time (s)")
    axes[2].set_title("Weighted A* runtime vs w")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wa_sweep_metrics.png", dpi=200)
    plt.close(fig)

    # Accuracy bar chart
    plt.figure(figsize=(6, 4))
    plt.bar([str(w) for w in ws], accuracies, color="tab:purple")
    plt.axhline(1.0, color="black", linestyle="--", label="Optimal (Dijkstra)")
    plt.ylabel("Cost / optimal")
    plt.xlabel("Weight w")
    plt.title("Weighted A* accuracy vs optimal cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wa_sweep_accuracy.png", dpi=200)
    plt.close()


def run(weights: List[float]):
    G = generate_graph()
    start, goal = choose_endpoints(G)
    print(f"[info] Start: {start} | Goal: {goal} | Graph: {G.graph.get('graph_label')}")

    baseline = uniform_cost_search(G, start, goal)
    print(
        f"[baseline] Dijkstra cost={baseline.total_path_cost:.2f} | expanded={baseline.nodes_expanded} | time={baseline.time_s:.3f}s"
    )

    results: List[SearchResult] = []
    for w in weights:
        res = weighted_a_star(G, start, goal, w)
        cost = compute_path_cost(G, res.path) if res.path else math.inf
        ratio = cost / baseline.total_path_cost if baseline.total_path_cost > 0 else math.inf
        print(
            f"w={w:>4}: cost={cost:>10.2f} | cost/opt={ratio:>.3f} | expanded={res.nodes_expanded:>7} | time={res.time_s:>.3f}s"
        )
        results.append(res)

    plot_results(G, weights, results, baseline)


if __name__ == "__main__":
    # At least 10 weights spanning sub- to super-admissible
    weights = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
    run(weights)
