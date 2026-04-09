"""Comprehensive experiment runner and visualization suite.

Generates:
- Combined CSVs (overall / informed / uninformed / weighted A* sweep)
- Extensive bar, scatter, rank, and weighted-A* plots
- Depth-limit/IDS diagnostics
- Path and overlay visualizations (all, informed, uninformed, best-of)
- Explored-nodes visuals for key algorithms

Algorithm names strictly follow lecture notes.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Sequence, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import osmnx as ox

from routing.data.graph_builder import generate_graph
from routing.data.features_costs import assign_synthetic_features, apply_cost
from routing.heuristics.spatial import euclidean_heuristic, exponential_feature_heuristic
from routing.viz.plotting import plot_single_route, plot_all_routes, plot_explored_nodes
from routing.algorithms import (
    SearchResult,
    breadth_first_search,
    uniform_cost_search,
    depth_first_search,
    depth_limited_search,
    iterative_deepening_search,
    bidirectional_search,
    greedy_best_first_search,
    a_star_search,
    weighted_a_star_search,
)


OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

DIRS = {
    "bars": OUTPUT_DIR / "bars",
    "scatter": OUTPUT_DIR / "scatter",
    "weighted_astar": OUTPUT_DIR / "weighted_astar",
    "depth": OUTPUT_DIR / "depth",
    "paths": OUTPUT_DIR / "paths",
    "overlays": OUTPUT_DIR / "overlays",
    "explored": OUTPUT_DIR / "explored",
}
for _p in DIRS.values():
    _p.mkdir(parents=True, exist_ok=True)

WEIGHTS = [0.5, 1.0, 1.5, 2.0, 3.0]
DLS_LIMITS = [2, 4, 6, 8, 10, 15, 20, 25, 30]

UNINFORMED_ORDER = [
    "Breadth-first search (BFS)",
    "Uniform cost search",
    "Depth-first search (DFS)",
    "Depth Limited Search",
    "Iterative Deepening Search",
    "Bidirectional Search",
]

INFORMED_ORDER = [
    "Greedy best-first search",
    "A* search",
    "Weighted A*",
]


def _largest_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if G.is_directed():
        comp = max(nx.strongly_connected_components(G), key=len)
    else:
        comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def choose_start_goal(
    G: nx.MultiDiGraph,
    *,
    start: Optional[Hashable] = None,
    goal: Optional[Hashable] = None,
    random_pair: bool = False,
    far_apart: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[Hashable, Hashable, nx.MultiDiGraph]:
    """Select start/goal nodes with options for manual, random, or far-apart sampling."""

    G = _largest_component(G)
    nodes = list(G.nodes)
    rng = rng or random.Random()

    if start is not None and goal is not None:
        if start not in G or goal not in G:
            raise ValueError("Provided start/goal not in graph")
        return start, goal, G

    if random_pair:
        start = rng.choice(nodes)
        goal = rng.choice(nodes)
        while goal == start:
            goal = rng.choice(nodes)
        return start, goal, G

    # Default: farthest pair heuristic
    start = nodes[0]
    if far_apart:
        goal = max(nodes, key=lambda n: euclidean_heuristic(G, start, n, 1.0))
        start = max(nodes, key=lambda n: euclidean_heuristic(G, n, goal, 1.0))
    else:
        goal = nodes[-1]
    return start, goal, G


def slugify(name: str) -> str:
    return name.lower().replace("*", "star").replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def run_algorithms(
    G: nx.MultiDiGraph,
    start,
    goal,
    *,
    skip_uninformed: bool = False,
    timeout: Optional[int] = None,
) -> List[SearchResult]:
    h_fn = exponential_feature_heuristic(G, goal)

    runs: List[tuple[str, Callable[[], SearchResult]]] = []

    if not skip_uninformed:
        runs.extend(
            [
                ("Breadth-first search (BFS)", lambda: breadth_first_search(G, start, goal)),
                ("Uniform cost search", lambda: uniform_cost_search(G, start, goal)),
                ("Depth-first search (DFS)", lambda: depth_first_search(G, start, goal)),
                ("Depth Limited Search", lambda: depth_limited_search(G, start, goal, limit=20)),
                ("Iterative Deepening Search", lambda: iterative_deepening_search(G, start, goal, max_depth=40)),
                ("Bidirectional Search", lambda: bidirectional_search(G, start, goal)),
            ]
        )

    runs.extend(
        [
            ("Greedy best-first search", lambda: greedy_best_first_search(G, start, goal, h_fn)),
            ("A* search", lambda: a_star_search(G, start, goal, h_fn)),
        ]
    )

    for w in WEIGHTS:
        runs.append(("Weighted A*", lambda w=w: weighted_a_star_search(G, start, goal, h_fn, w=w)))

    results: List[SearchResult] = []
    def _run_with_timeout(label: str, func: Callable[[], SearchResult]) -> SearchResult:
        if timeout is None or timeout <= 0:
            return func()
        import multiprocessing as mp

        def worker(q):
            try:
                q.put(func())
            except Exception as exc:  # noqa
                q.put(exc)

        q: mp.Queue = mp.Queue()
        p = mp.Process(target=worker, args=(q,))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return SearchResult(
                algorithm_name=label,
                path=[],
                path_found=False,
                total_path_cost=float("inf"),
                nodes_expanded=0,
                execution_time=timeout,
                max_frontier_size=0,
                path_length=0,
                visited_count=0,
                start_node=start,
                goal_node=goal,
                expanded_nodes=[],
                cutoff_occurred=True,
            )
        res = q.get()
        if isinstance(res, Exception):
            return SearchResult(
                algorithm_name=label,
                path=[],
                path_found=False,
                total_path_cost=float("inf"),
                nodes_expanded=0,
                execution_time=0.0,
                max_frontier_size=0,
                path_length=0,
                visited_count=0,
                start_node=start,
                goal_node=goal,
                expanded_nodes=[],
            )
        res.algorithm_name = label if label != "Weighted A*" or res.algorithm_name == "Weighted A*" else res.algorithm_name
        return res

    for expected_name, fn in runs:
        results.append(_run_with_timeout(expected_name, fn))
    return results


def results_to_df(results: List[SearchResult]) -> pd.DataFrame:
    def category(name: str) -> str:
        return "uninformed" if name in UNINFORMED_ORDER else "informed"

    rows = []
    for r in results:
        rows.append(
            {
                "algorithm_name": r.algorithm_name,
                "category": category(r.algorithm_name),
                "path_found": r.path_found,
                "total_path_cost": r.total_path_cost,
                "path_length": r.path_length,
                "nodes_expanded": r.nodes_expanded,
                "visited_count": r.visited_count,
                "execution_time": r.execution_time,
                "max_frontier_size": r.max_frontier_size,
                "depth_reached": r.depth_reached,
                "cutoff_occurred": r.cutoff_occurred,
                "meeting_node": r.meeting_node,
                "weight": r.weight,
                "start_node": r.start_node,
                "goal_node": r.goal_node,
            }
        )
    return pd.DataFrame(rows)


def save_group_csvs(df: pd.DataFrame):
    df.to_csv(OUTPUT_DIR / "all_algorithms_results.csv", index=False)
    df[df.category == "uninformed"].to_csv(OUTPUT_DIR / "uninformed_results.csv", index=False)
    df[df.category == "informed"].to_csv(OUTPUT_DIR / "informed_results.csv", index=False)
    wa = df[df.algorithm_name == "Weighted A*"]
    if not wa.empty:
        wa.to_csv(OUTPUT_DIR / "weighted_astar_sweep_results.csv", index=False)


def _out(name: str) -> Path:
    return OUTPUT_DIR / name


def _subdir(name: str) -> Path:
    return DIRS.get(name, OUTPUT_DIR)


def bar_charts(df: pd.DataFrame, metrics: Sequence[str], prefix: str, subdir: str = "bars"):
    for metric in metrics:
        plt.figure(figsize=(9, 4))
        plt.bar(df["algorithm_name"], df[metric], color="steelblue")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{prefix} {metric}")
        plt.tight_layout()
        plt.savefig(_subdir(subdir) / f"{prefix}_{metric}.png", dpi=150)
        plt.close()


def scatter_charts(df: pd.DataFrame, pairs: Sequence[tuple], prefix: str, subdir: str = "scatter"):
    for x, y, fname in pairs:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x], df[y], color="darkorange")
        for _, row in df.iterrows():
            plt.annotate(row["algorithm_name"], (row[x], row[y]), textcoords="offset points", xytext=(3, 3), fontsize=8)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(_subdir(subdir) / f"{prefix}_{fname}.png", dpi=150)
        plt.close()


def rank_plot(df: pd.DataFrame, metric: str, prefix: str, subdir: str = "bars"):
    sorted_df = df.sort_values(metric)
    plt.figure(figsize=(8, 4))
    plt.bar(sorted_df["algorithm_name"], sorted_df[metric], color="seagreen")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Ranking by {metric}")
    plt.tight_layout()
    plt.savefig(_subdir(subdir) / f"{prefix}_rank_{metric}.png", dpi=150)
    plt.close()


def weighted_astar_plots(df: pd.DataFrame):
    wa = df[df.algorithm_name == "Weighted A*"]
    if wa.empty:
        return
    metrics = [
        ("total_path_cost", "weighted_astar_weight_vs_cost.png"),
        ("nodes_expanded", "weighted_astar_weight_vs_expanded.png"),
        ("execution_time", "weighted_astar_weight_vs_time.png"),
        ("path_length", "weighted_astar_weight_vs_path_length.png"),
        ("visited_count", "weighted_astar_weight_vs_visited_count.png"),
        ("max_frontier_size", "weighted_astar_weight_vs_frontier.png"),
    ]
    for metric, fname in metrics:
        plt.figure(figsize=(6, 4))
        plt.plot(wa["weight"], wa[metric], marker="o")
        plt.xlabel("Weight w")
        plt.ylabel(metric)
        plt.title(f"Weighted A* : {metric} vs w")
        plt.tight_layout()
        plt.savefig(_subdir("weighted_astar") / fname, dpi=150)
        plt.close()


def path_visuals(G: nx.MultiDiGraph, results: List[SearchResult], start, goal):
    for res in results:
        fname = _subdir("paths") / f"path_{slugify(res.algorithm_name)}.png"
        plot_single_route(G, res.path, res.algorithm_name, color=None, filename=fname)

    # overlays
    paths_all = {r.algorithm_name: r.path for r in results if r.path_found and r.path}
    if paths_all:
        plot_all_routes(G, paths_all, start, goal, filename=_subdir("overlays") / "overlay_all_algorithms_paths.png")

    paths_uninformed = {r.algorithm_name: r.path for r in results if r.algorithm_name in UNINFORMED_ORDER and r.path_found and r.path}
    if paths_uninformed:
        plot_all_routes(G, paths_uninformed, start, goal, filename=_subdir("overlays") / "overlay_uninformed_paths.png")

    paths_informed = {r.algorithm_name: r.path for r in results if r.algorithm_name in INFORMED_ORDER and r.path_found and r.path and (r.weight in (None, 1.0))}
    if paths_informed:
        plot_all_routes(G, paths_informed, start, goal, filename=_subdir("overlays") / "overlay_informed_paths.png")

    # Best-path highlight plots
    if paths_all:
        df = results_to_df(results)
        best_cost = df.loc[df.total_path_cost.idxmin()].algorithm_name
        best_time = df.loc[df.execution_time.idxmin()].algorithm_name
        best_expanded = df.loc[df.nodes_expanded.idxmin()].algorithm_name
        highlight = {}
        for name in {best_cost, best_time, best_expanded}:
            res = next(r for r in results if r.algorithm_name == name)
            highlight[name] = res.path
        plot_all_routes(G, highlight, start, goal, filename=_subdir("overlays") / "overlay_best_algorithms.png")


def explored_visuals(G: nx.MultiDiGraph, results: List[SearchResult]):
    targets = {"Breadth-first search (BFS)", "Uniform cost search", "Greedy best-first search", "A* search", "Weighted A*"}
    for res in results:
        if res.algorithm_name in targets and res.expanded_nodes is not None:
            fname = _subdir("explored") / f"explored_{slugify(res.algorithm_name)}.png"
            plot_explored_nodes(G, res.expanded_nodes, res.path, res.algorithm_name, filename=fname)

    # comparisons
    pairs = [
        ("Uniform cost search", "A* search", "compare_explored_ucs_vs_astar.png"),
        ("A* search", "Weighted A*", "compare_explored_astar_vs_weighted_astar.png"),
        ("Breadth-first search (BFS)", "Depth-first search (DFS)", "compare_explored_bfs_vs_dfs.png"),
    ]
    for a, b, fname in pairs:
        pa = next((r for r in results if r.algorithm_name == a and r.expanded_nodes), None)
        pb = next((r for r in results if r.algorithm_name == b and r.expanded_nodes), None)
        if not pa or not pb:
            continue
        pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
        plt.figure(figsize=(6, 6))
        xs_a = [pos[n][0] for n in pa.expanded_nodes if n in pos]
        ys_a = [pos[n][1] for n in pa.expanded_nodes if n in pos]
        xs_b = [pos[n][0] for n in pb.expanded_nodes if n in pos]
        ys_b = [pos[n][1] for n in pb.expanded_nodes if n in pos]
        plt.scatter(xs_a, ys_a, c="blue", s=8, alpha=0.4, label=a)
        plt.scatter(xs_b, ys_b, c="red", s=8, alpha=0.4, label=b)
        plt.axis("equal")
        plt.axis("off")
        plt.legend(loc="best", frameon=False)
        plt.title(f"Explored comparison: {a} vs {b}")
        plt.tight_layout()
        plt.savefig(_subdir("explored") / fname, dpi=180, bbox_inches="tight")
        plt.close()


def depth_diagnostics(G: nx.MultiDiGraph, start, goal):
    records = []
    for limit in DLS_LIMITS:
        res = depth_limited_search(G, start, goal, limit)
        records.append({"depth_limit": limit, "path_found": res.path_found, "nodes_expanded": res.nodes_expanded, "execution_time": res.execution_time, "total_path_cost": res.total_path_cost})
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "depth_limited_sweep.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df["depth_limit"], df["nodes_expanded"], marker="o")
    plt.xlabel("Depth limit")
    plt.ylabel("Nodes expanded")
    plt.title("Depth Limited Search sweep")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dls_nodes_vs_depth.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(df["depth_limit"], df["execution_time"], marker="o", color="purple")
    plt.xlabel("Depth limit")
    plt.ylabel("Execution time (s)")
    plt.title("DLS time vs depth")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dls_time_vs_depth.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(df["depth_limit"], df["total_path_cost"], marker="o", color="green")
    plt.xlabel("Depth limit")
    plt.ylabel("Path cost (if found)")
    plt.title("DLS cost vs depth")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dls_cost_vs_depth.png", dpi=150)
    plt.close()

    # IDS approximation by cumulative DLS up to each limit
    ids_records = []
    cumulative_nodes = 0
    cumulative_time = 0.0
    for limit, nodes, t in zip(df["depth_limit"], df["nodes_expanded"], df["execution_time"]):
        cumulative_nodes += nodes
        cumulative_time += t
        ids_records.append({"iteration_limit": limit, "cumulative_nodes": cumulative_nodes, "cumulative_time": cumulative_time})
    ids_df = pd.DataFrame(ids_records)
    ids_df.to_csv(OUTPUT_DIR / "ids_iteration_stats.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(ids_df["iteration_limit"], ids_df["cumulative_nodes"], marker="o")
    plt.xlabel("Iteration depth limit")
    plt.ylabel("Cumulative nodes expanded")
    plt.title("Iterative Deepening Search cumulative nodes")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ids_nodes_vs_limit.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ids_df["iteration_limit"], ids_df["cumulative_time"], marker="o", color="orange")
    plt.xlabel("Iteration depth limit")
    plt.ylabel("Cumulative time (s)")
    plt.title("Iterative Deepening Search cumulative time")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ids_time_vs_limit.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run routing experiments on Dhaka road network")
    parser.add_argument("--skip-uninformed", action="store_true", help="Skip uninformed algorithms on large graphs")
    parser.add_argument("--timeout", type=int, default=None, help="Per-algorithm timeout in seconds")
    parser.add_argument("--start", type=str, default=None, help="Manual start node id")
    parser.add_argument("--goal", type=str, default=None, help="Manual goal node id")
    parser.add_argument("--random-start-goal", action="store_true", help="Pick random start/goal")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--far-apart", action="store_true", default=True, help="Prefer far-apart sampling")
    args = parser.parse_args()

    G = generate_graph(use_osm=True)
    assign_synthetic_features(G)
    apply_cost(G)

    rng = random.Random(args.seed)
    start_node = args.start
    goal_node = args.goal
    if start_node is not None:
        try:
            start_node = eval(start_node)
        except Exception:
            pass
    if goal_node is not None:
        try:
            goal_node = eval(goal_node)
        except Exception:
            pass

    start, goal, G = choose_start_goal(
        G,
        start=start_node,
        goal=goal_node,
        random_pair=args.random_start_goal,
        far_apart=args.far_apart,
        rng=rng,
    )

    print(f"[info] Using start={start}, goal={goal}, nodes={len(G)}, edges={len(G.edges())}")

    # Save a light base map preview
    try:
        fig, ax = ox.plot_graph(
            G,
            node_size=1,
            node_color="#555555",
            edge_color="#cccccc",
            edge_linewidth=0.4,
            bgcolor="white",
            show=False,
            close=False,
        )
        fig.savefig(OUTPUT_DIR / "map.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] map plot failed: {e}")

    results = run_algorithms(G, start, goal, skip_uninformed=args.skip_uninformed, timeout=args.timeout)
    df = results_to_df(results)
    save_group_csvs(df)

    metrics = ["total_path_cost", "path_length", "nodes_expanded", "execution_time", "visited_count", "max_frontier_size"]

    # Overall bar charts
    bar_charts(df, metrics, prefix="overall")
    # Group bars
    bar_charts(df[df.category == "uninformed"], metrics, prefix="uninformed")
    bar_charts(df[df.category == "informed"], metrics, prefix="informed")

    # Scatters
    pairs = [
        ("nodes_expanded", "total_path_cost", "nodes_vs_cost"),
        ("execution_time", "total_path_cost", "time_vs_cost"),
        ("execution_time", "nodes_expanded", "time_vs_expanded"),
        ("visited_count", "total_path_cost", "visited_vs_cost"),
        ("max_frontier_size", "execution_time", "frontier_vs_time"),
    ]
    scatter_charts(df, pairs, prefix="scatter_all")
    scatter_charts(df[df.category == "uninformed"], pairs, prefix="scatter_uninformed")
    scatter_charts(df[df.category == "informed"], pairs, prefix="scatter_informed")

    # Rank plots
    for metric in ["total_path_cost", "execution_time", "nodes_expanded"]:
        rank_plot(df, metric, prefix="overall")

    weighted_astar_plots(df)

    path_visuals(G, results, start, goal)
    explored_visuals(G, results)
    depth_diagnostics(G, start, goal)

    print("Experiment completed. Outputs saved to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
