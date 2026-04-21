"""Streamlit UI for running Dhaka routing experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Callable, Dict, Hashable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
import streamlit as st

# Ensure `routing` package is importable when launched via `streamlit run routing/ui/dashboard.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.algorithms import (
    SearchResult,
    a_star_search,
    bidirectional_search,
    breadth_first_search,
    depth_first_search,
    depth_limited_search,
    greedy_best_first_search,
    iterative_deepening_search,
    uniform_cost_search,
    weighted_a_star_search,
)
from routing.experiments.graph_scope import load_graph_with_costs
from routing.experiments.run import (
    choose_start_goal,
    configure_output_dirs,
    depth_diagnostics,
    explored_visuals,
    path_visuals,
    print_run_summary,
    rank_plot,
    results_to_df,
    run_algorithms,
    save_group_csvs,
    scatter_charts,
    bar_charts,
    weighted_astar_plots,
)
from routing.experiments.weighted_astar_sweep import run_sweep
from routing.experiments.heuristic_verification import (
    admissibility_frame,
    choose_goal_far,
    compute_heuristics,
    consistency_frame,
    write_plots,
)
from routing.heuristics.spatial import exponential_feature_heuristic


DEFAULT_WEIGHTS = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]


@st.cache_resource(show_spinner=False)
def get_graph(seed: int):
    return load_graph_with_costs(use_osm=True, max_nodes=None, seed=seed)


def _algorithm_options() -> Dict[str, str]:
    return {
        "Breadth-first search (BFS)": "bfs",
        "Uniform cost search": "ucs",
        "Depth-first search (DFS)": "dfs",
        "Depth Limited Search": "dls",
        "Iterative Deepening Search": "ids",
        "Bidirectional Search": "bidir",
        "Greedy best-first search": "greedy",
        "A* search": "astar",
        "Weighted A*": "weighted_astar",
    }


def _single_algorithm_run(
    algorithm_code: str,
    G,
    start: Hashable,
    goal: Hashable,
    *,
    dls_limit: int,
    ids_max_depth: int,
    weight: float,
) -> SearchResult:
    h_fn = exponential_feature_heuristic(G, goal)
    runners: Dict[str, Callable[[], SearchResult]] = {
        "bfs": lambda: breadth_first_search(G, start, goal),
        "ucs": lambda: uniform_cost_search(G, start, goal),
        "dfs": lambda: depth_first_search(G, start, goal),
        "dls": lambda: depth_limited_search(G, start, goal, limit=dls_limit),
        "ids": lambda: iterative_deepening_search(G, start, goal, max_depth=ids_max_depth),
        "bidir": lambda: bidirectional_search(G, start, goal),
        "greedy": lambda: greedy_best_first_search(G, start, goal, h_fn),
        "astar": lambda: a_star_search(G, start, goal, h_fn),
        "weighted_astar": lambda: weighted_a_star_search(G, start, goal, h_fn, w=weight),
    }
    return runners[algorithm_code]()


def _display_name(row: pd.Series) -> str:
    if row["algorithm_name"] == "Weighted A*" and pd.notna(row["weight"]):
        return f"Weighted A* (w={row['weight']:g})"
    return str(row["algorithm_name"])


def _route_map_figure(G, path: List[Hashable], title: str):
    if not path:
        return None
    # Use OSMnx native route plotting for robust rendering on full city graphs.
    fig, ax = ox.plot_graph_route(
        G,
        path,
        route_linewidth=3,
        route_alpha=0.95,
        route_color="#d90429",
        orig_dest_size=80,
        node_size=0,
        edge_linewidth=0.4,
        edge_color="#8d99ae",
        bgcolor="white",
        show=False,
        close=False,
    )
    fig.patch.set_facecolor("white")
    ax.set_title(title, color="black")
    return fig


def _comparison_figure(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df["label"] = plot_df.apply(_display_name, axis=1)
    metrics = [
        ("total_path_cost", "Total Path Cost"),
        ("execution_time", "Execution Time (s)"),
        ("nodes_expanded", "Nodes Expanded"),
        ("visited_count", "Visited Count"),
        ("path_length", "Path Length"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = list(axes.flat)
    for i, (metric, title) in enumerate(metrics):
        ax = axes_flat[i]
        ax.bar(plot_df["label"], plot_df[metric], color="#0ea5e9")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    axes_flat[-1].axis("off")
    plt.tight_layout()
    return fig


def _run_all_experiments_whole_dhaka(seed: int, dls_limit: int, ids_max_depth: int, timeout_seconds: int):
    outputs = {
        "all_algorithms": "experiments/all_algorithms/images/full_dhaka",
        "weighted_astar": "experiments/weighted_astar_analysis/images/full_dhaka",
        "heuristic_check": "experiments/heuristic_check/images/full_dhaka",
    }

    all_algorithms_dir = Path(outputs["all_algorithms"])
    configure_output_dirs(all_algorithms_dir)
    G = load_graph_with_costs(use_osm=True, max_nodes=None, seed=seed)
    start, goal, G = choose_start_goal(G, far_apart=True)
    results = run_algorithms(
        G,
        start,
        goal,
        skip_uninformed=False,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
        dls_limit=dls_limit,
        ids_max_depth=ids_max_depth,
    )
    print_run_summary(results)
    df = results_to_df(results)
    save_group_csvs(df)
    metrics = ["total_path_cost", "path_length", "nodes_expanded", "execution_time", "visited_count", "max_frontier_size"]
    bar_charts(df, metrics, prefix="overall")
    bar_charts(df[df.category == "uninformed"], metrics, prefix="uninformed")
    bar_charts(df[df.category == "informed"], metrics, prefix="informed")
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
    for metric in ["total_path_cost", "execution_time", "nodes_expanded"]:
        rank_plot(df, metric, prefix="overall")
    weighted_astar_plots(df)
    path_visuals(G, results, start, goal)
    explored_visuals(G, results)
    depth_diagnostics(G, start, goal)

    run_sweep(
        weights=DEFAULT_WEIGHTS,
        max_nodes=None,
        seed=seed,
        output_dir=Path(outputs["weighted_astar"]),
    )

    heuristic_dir = Path(outputs["heuristic_check"])
    heuristic_dir.mkdir(parents=True, exist_ok=True)
    G2 = load_graph_with_costs(use_osm=True, max_nodes=None, seed=seed)
    goal2 = choose_goal_far(G2)
    h = compute_heuristics(G2, goal2)
    cons_df = consistency_frame(G2, h)
    adm_df = admissibility_frame(G2, goal2, h)
    cons_df.to_csv(heuristic_dir / "consistency_results.csv", index=False)
    adm_df.to_csv(heuristic_dir / "admissibility_results.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "check": "consistency",
                "total_checked": int(len(cons_df)),
                "violations": int(cons_df["is_violation"].sum()),
                "violation_rate": float(cons_df["is_violation"].mean()),
                "max_residual": float(cons_df["residual"].max()),
            },
            {
                "check": "admissibility",
                "total_checked": int(len(adm_df)),
                "violations": int(adm_df["is_violation"].sum()),
                "violation_rate": float(adm_df["is_violation"].mean()),
                "max_residual": float(adm_df["residual"].max()),
            },
        ]
    )
    summary.to_csv(heuristic_dir / "summary.csv", index=False)
    write_plots(cons_df, adm_df, heuristic_dir)
    return outputs, df, results, G, start, goal


def _weighted_astar_custom_weights(
    G,
    start: Hashable,
    goal: Hashable,
    weights: List[float],
) -> pd.DataFrame:
    h_fn = exponential_feature_heuristic(G, goal)
    records = []
    for w in weights:
        res = weighted_a_star_search(G, start, goal, h_fn, w=w)
        records.append(
            {
                "weight": w,
                "path_found": res.path_found,
                "cost": res.total_path_cost,
                "nodes_expanded": res.nodes_expanded,
                "execution_time_s": res.execution_time,
                "path_length": res.path_length,
            }
        )
    return pd.DataFrame(records).sort_values("weight")


def _to_table(result: SearchResult) -> pd.DataFrame:
    row = asdict(result)
    return pd.DataFrame([row])


def main() -> None:
    st.set_page_config(page_title="Dhaka Routing Experiment UI", layout="wide")
    st.title("Dhaka Routing Experiment UI")
    st.caption("Run a single algorithm, the full experiment suite, or Weighted A* with custom weights on the full Dhaka map.")

    with st.sidebar:
        st.header("Controls")
        seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
        dls_limit = st.number_input("DLS limit", min_value=1, max_value=10000, value=120, step=1)
        ids_max_depth = st.number_input("IDS max depth", min_value=1, max_value=10000, value=200, step=1)
        timeout_seconds = st.number_input("Per-algorithm timeout (0 = off)", min_value=0, max_value=7200, value=0, step=1)

    mode = st.selectbox(
        "Menu",
        [
            "Run Single Algorithm (Whole Dhaka Map)",
            "Run Whole Experiment Suite (Whole Dhaka Map)",
            "Run Weighted A* with Different Weights (Whole Dhaka Map)",
        ],
    )

    if mode == "Run Single Algorithm (Whole Dhaka Map)":
        options = _algorithm_options()
        label = st.selectbox("Algorithm", list(options.keys()))
        selected = options[label]
        weight = 1.5
        if selected == "weighted_astar":
            weight = st.number_input("Weighted A* weight (w)", min_value=0.1, max_value=50.0, value=1.5, step=0.1)

        if st.button("Run Selected Algorithm", type="primary"):
            with st.spinner("Loading Dhaka graph and running selected algorithm..."):
                G = get_graph(seed)
                start, goal, G = choose_start_goal(G, far_apart=True)
                result = _single_algorithm_run(
                    selected,
                    G,
                    start,
                    goal,
                    dls_limit=int(dls_limit),
                    ids_max_depth=int(ids_max_depth),
                    weight=float(weight),
                )
            st.success("Run completed.")
            st.write(f"Start: `{start}` | Goal: `{goal}`")
            st.dataframe(_to_table(result), use_container_width=True)
            if result.path_found and result.path:
                st.subheader("Route Map")
                fig = _route_map_figure(G, result.path, f"{label} Route on Full Dhaka Map")
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            else:
                st.warning("No path found, so no route map is available.")

    if mode == "Run Whole Experiment Suite (Whole Dhaka Map)":
        if st.button("Run All Experiments", type="primary"):
            with st.spinner("Running all experiments on full Dhaka map. This may take a while..."):
                outputs, all_df, all_results, run_graph, run_start, run_goal = _run_all_experiments_whole_dhaka(
                    int(seed),
                    int(dls_limit),
                    int(ids_max_depth),
                    int(timeout_seconds),
                )
            st.success("All experiments completed.")
            st.write(f"Start: `{run_start}` | Goal: `{run_goal}`")
            table_df = all_df.copy()
            table_df["algorithm"] = table_df.apply(_display_name, axis=1)
            st.dataframe(
                table_df[
                    [
                        "algorithm",
                        "path_found",
                        "total_path_cost",
                        "execution_time",
                        "nodes_expanded",
                        "visited_count",
                        "path_length",
                        "max_frontier_size",
                    ]
                ],
                use_container_width=True,
            )
            st.subheader("All-Algorithm Comparison Graphs")
            comp_fig = _comparison_figure(all_df)
            st.pyplot(comp_fig, use_container_width=True)
            plt.close(comp_fig)
            best = min((r for r in all_results if r.path_found and r.path), key=lambda r: r.total_path_cost, default=None)
            if best is not None:
                st.subheader("Best Path Map (Lowest Cost)")
                best_label = best.algorithm_name if best.weight is None else f"{best.algorithm_name} (w={best.weight:g})"
                best_fig = _route_map_figure(run_graph, best.path, f"{best_label} on Full Dhaka Map")
                if best_fig is not None:
                    st.pyplot(best_fig, use_container_width=True)
                    plt.close(best_fig)
            st.write("Outputs saved to:")
            st.code("\n".join(f"{k}: {v}" for k, v in outputs.items()))

    if mode == "Run Weighted A* with Different Weights (Whole Dhaka Map)":
        default_text = ", ".join(str(w) for w in DEFAULT_WEIGHTS)
        weights_text = st.text_input("Weights (comma-separated)", value=default_text)
        if st.button("Run Weighted A* Weight Sweep", type="primary"):
            with st.spinner("Loading Dhaka graph and running Weighted A* for all weights..."):
                weights = [float(w.strip()) for w in weights_text.split(",") if w.strip()]
                G = get_graph(seed)
                start, goal, G = choose_start_goal(G, far_apart=True)
                df = _weighted_astar_custom_weights(G, start, goal, weights)
            st.success("Weighted A* sweep completed.")
            st.write(f"Start: `{start}` | Goal: `{goal}`")
            st.dataframe(df, use_container_width=True)
            st.line_chart(df.set_index("weight")[["cost", "nodes_expanded", "execution_time_s"]], use_container_width=True)


if __name__ == "__main__":
    main()
