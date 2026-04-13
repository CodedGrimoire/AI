"""Whole-Dhaka informed-search safe-passage experiment pipeline.

Run example:
python -m dhaka_safe_passage.experiment_runner --pairs 15 --weighted-w 1.5 --include-ucs
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
from typing import Hashable

import matplotlib

matplotlib.use("Agg")
import networkx as nx
import pandas as pd

from .algorithms import (
    a_star_search,
    greedy_best_first_search,
    uniform_cost_search,
    weighted_a_star_search,
)
from .contextual_features import FeatureConfig, assign_contextual_features
from .cost_functions import CostWeights, apply_contextual_cost, apply_distance_cost
from .graph_builder import load_or_build_dhaka_graph
from .heuristics import HeuristicWeights, contextual_heuristic, euclidean_distance
from .metrics import SafeScoreWeights, compute_route_quality
from .visualization import (
    ensure_dirs,
    save_bar_chart,
    save_radar_chart,
    save_route_overlay,
    save_scatter,
)


def _largest_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    try:
        comp = max(nx.strongly_connected_components(G), key=len)
    except Exception:
        comp = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(comp).copy()


def sample_practical_subgraph(G: nx.MultiDiGraph, target_nodes: int, seed: int) -> nx.MultiDiGraph:
    if target_nodes <= 0 or len(G) <= target_nodes:
        return G

    rng = random.Random(seed)
    UG = G.to_undirected()
    picked: set[Hashable] = set()

    while len(picked) < target_nodes:
        anchor = rng.choice(list(G.nodes))
        # radius in hops, growing if needed
        for cutoff in (6, 10, 14, 18):
            around = set(nx.single_source_shortest_path_length(UG, anchor, cutoff=cutoff).keys())
            picked |= around
            if len(picked) >= target_nodes:
                break
        if len(picked) >= target_nodes:
            break

    if len(picked) > target_nodes:
        picked = set(rng.sample(list(picked), target_nodes))

    return _largest_component(G.subgraph(picked).copy())


def sample_start_goal_pairs(
    G: nx.MultiDiGraph,
    n_pairs: int,
    seed: int,
    min_distance_m: float,
    max_distance_m: float,
) -> list[tuple[Hashable, Hashable]]:
    rng = random.Random(seed)
    nodes = list(G.nodes)
    pairs: list[tuple[Hashable, Hashable]] = []
    attempts = 0
    max_attempts = max(2000, n_pairs * 300)

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        s = rng.choice(nodes)
        g = rng.choice(nodes)
        if s == g:
            continue

        d = euclidean_distance(G, s, g)
        if d < min_distance_m or d > max_distance_m:
            continue

        pair = (s, g)
        rev = (g, s)
        if pair in pairs or rev in pairs:
            continue

        pairs.append(pair)

    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"Could not sample enough start-goal pairs in distance range [{min_distance_m}, {max_distance_m}] m. "
            f"Requested={n_pairs}, sampled={len(pairs)}"
        )

    return pairs


def _run_for_pair(
    G: nx.MultiDiGraph,
    pair_id: int,
    start,
    goal,
    weighted_w: float,
    include_ucs: bool,
    score_weights: SafeScoreWeights,
    h_weights: HeuristicWeights,
) -> tuple[list[dict], dict[str, list]]:
    rows: list[dict] = []
    routes: dict[str, list] = {}

    h_fn = contextual_heuristic(G, goal, h_weights)

    algos = [
        ("Greedy best-first search", lambda: greedy_best_first_search(G, start, goal, h_fn)),
        ("A* search", lambda: a_star_search(G, start, goal, h_fn)),
        ("Weighted A*", lambda: weighted_a_star_search(G, start, goal, h_fn, w=weighted_w)),
    ]
    if include_ucs:
        algos.append(("Uniform cost search", lambda: uniform_cost_search(G, start, goal)))

    for label, fn in algos:
        res = fn()
        metrics = compute_route_quality(G, res.path, score_weights) if res.path_found else {
            "total_path_length": float("nan"),
            "average_traffic": float("nan"),
            "average_accident_risk": float("nan"),
            "average_bumpiness": float("nan"),
            "average_safety": float("nan"),
            "cumulative_safety": float("nan"),
            "cumulative_contextual_penalty": float("nan"),
            "safe_passage_score": float("nan"),
        }

        routes[label] = res.path
        rows.append(
            {
                "pair_id": pair_id,
                "start_node": start,
                "goal_node": goal,
                "algorithm_name": label,
                "weight": res.weight,
                "path_found": res.path_found,
                "total_path_cost": res.total_path_cost,
                "nodes_expanded": res.nodes_expanded,
                "execution_time": res.execution_time,
                **metrics,
            }
        )

    return rows, routes


def _write_formula_notes(out_dir: Path, c_w: CostWeights, h_w: HeuristicWeights, s_w: SafeScoreWeights) -> None:
    text = f"""Contextual edge cost used for informed search:

cost(u,v) = length(u,v) * max({c_w.min_multiplier:.3f}, 1 + alpha*T + beta*A + gamma*B - lambda*S)
alpha={c_w.alpha_traffic}, beta={c_w.beta_accident}, gamma={c_w.gamma_bumpiness}, lambda={c_w.lambda_safety}

Contextual heuristic used for Greedy, A*, Weighted A*:

h(n) = d(n,g) * max({h_w.min_factor:.3f}, 1 + alpha*T(n) + beta*A(n) + gamma*B(n) - lambda*S(n))
alpha={h_w.alpha_traffic}, beta={h_w.beta_accident}, gamma={h_w.gamma_bumpiness}, lambda={h_w.lambda_safety}

Safe passage score (report metric):

safe_passage_score = w1*avg_safety - w2*avg_traffic - w3*avg_accident_risk - w4*avg_bumpiness
w1={s_w.w_safety}, w2={s_w.w_traffic}, w3={s_w.w_accident}, w4={s_w.w_bumpiness}
"""
    (out_dir / "summary" / "formulas.txt").write_text(text, encoding="utf-8")


def _interpretation(df_mean: pd.DataFrame) -> str:
    if df_mean.empty:
        return "No results to interpret."

    def best(metric: str, maximize: bool) -> str:
        row = df_mean.sort_values(metric, ascending=not maximize).iloc[0]
        return str(row["algorithm_name"])

    safest = best("safe_passage_score", True)
    traffic_best = best("average_traffic", False)
    accident_best = best("average_accident_risk", False)
    bump_best = best("average_bumpiness", False)
    fastest = best("execution_time", False)
    balanced = best("cumulative_contextual_penalty", False)

    weighted = df_mean[df_mean.algorithm_name == "Weighted A*"]
    astar = df_mean[df_mean.algorithm_name == "A* search"]
    tradeoff_line = "Weighted A* trade-off vs A* could not be computed."
    if not weighted.empty and not astar.empty:
        wr = weighted.iloc[0]
        ar = astar.iloc[0]
        speed_delta = ar.execution_time - wr.execution_time
        safety_delta = wr.safe_passage_score - ar.safe_passage_score
        tradeoff_line = (
            f"Weighted A* vs A*: time delta={speed_delta:.4f}s (positive means Weighted A* faster), "
            f"safe-passage delta={safety_delta:.4f} (positive means Weighted A* safer)."
        )

    return "\n".join(
        [
            f"Safest overall route quality: {safest}",
            f"Best at avoiding traffic exposure: {traffic_best}",
            f"Best at minimizing accident exposure: {accident_best}",
            f"Best at minimizing bumpiness: {bump_best}",
            f"Fastest runtime: {fastest}",
            f"Best balanced contextual penalty: {balanced}",
            tradeoff_line,
        ]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dhaka informed search safe-passage experiments")
    p.add_argument("--graph-dir", type=Path, default=Path("graph_cache"), help="Folder for cached Dhaka graph files")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/safe_passage_dhaka"), help="Folder for tables/plots/summary")
    p.add_argument("--pairs", type=int, default=12, help="Number of start-goal pairs")
    p.add_argument("--representative-pairs", type=int, default=3, help="Number of route overlays to save")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--weighted-w", type=float, default=1.5)
    p.add_argument("--include-ucs", action="store_true", help="Include UCS baseline in informed comparison")
    p.add_argument("--min-distance-m", type=float, default=3500.0)
    p.add_argument("--max-distance-m", type=float, default=18000.0)
    p.add_argument("--practical-subgraph-nodes", type=int, default=0, help="Optional manageable subgraph size sampled from whole Dhaka context")
    p.add_argument("--dry-run", action="store_true", help="Parse args and print config without executing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(vars(args))
        return

    out = ensure_dirs(args.output_dir)

    # Configurable weights (explicit for report clarity)
    feature_cfg = FeatureConfig(seed=args.seed)
    cost_weights = CostWeights(alpha_traffic=0.9, beta_accident=1.1, gamma_bumpiness=0.6, lambda_safety=0.8)
    heuristic_weights = HeuristicWeights(alpha_traffic=0.7, beta_accident=0.9, gamma_bumpiness=0.5, lambda_safety=0.8)
    score_weights = SafeScoreWeights(w_safety=1.0, w_traffic=0.9, w_accident=1.1, w_bumpiness=0.6)

    G = load_or_build_dhaka_graph(args.graph_dir)
    G = sample_practical_subgraph(G, args.practical_subgraph_nodes, args.seed)
    G = _largest_component(G)

    assign_contextual_features(G, feature_cfg)

    # Uninformed behavior remains distance-only if you run uninformed externally.
    apply_distance_cost(G)

    # Informed safe-routing objective uses contextual edge costs.
    apply_contextual_cost(G, cost_weights)

    pairs = sample_start_goal_pairs(G, args.pairs, args.seed, args.min_distance_m, args.max_distance_m)

    all_rows: list[dict] = []
    representative: list[tuple[int, Hashable, Hashable, dict[str, list]]] = []

    for pair_id, (s, g) in enumerate(pairs, start=1):
        rows, routes = _run_for_pair(
            G,
            pair_id,
            s,
            g,
            args.weighted_w,
            args.include_ucs,
            score_weights,
            heuristic_weights,
        )
        all_rows.extend(rows)
        if len(representative) < args.representative_pairs:
            representative.append((pair_id, s, g, routes))

    df = pd.DataFrame(all_rows)
    informed_df = df[df["algorithm_name"].isin(["Greedy best-first search", "A* search", "Weighted A*", "Uniform cost search"])].copy()

    mean_cols = [
        "total_path_cost",
        "total_path_length",
        "nodes_expanded",
        "execution_time",
        "average_traffic",
        "average_accident_risk",
        "average_bumpiness",
        "average_safety",
        "cumulative_safety",
        "cumulative_contextual_penalty",
        "safe_passage_score",
    ]

    df_mean = informed_df.groupby("algorithm_name", as_index=False)[mean_cols].mean(numeric_only=True)

    df.to_csv(out["tables"] / "pair_level_results.csv", index=False)
    df_mean.to_csv(out["tables"] / "average_metrics_by_algorithm.csv", index=False)

    save_bar_chart(df_mean, "safe_passage_score", out["plots"] / "bar_safe_passage_score.png", "Average safe-passage score by algorithm", higher_better=True)
    save_bar_chart(df_mean, "average_traffic", out["plots"] / "bar_avg_traffic.png", "Average traffic exposure by algorithm", higher_better=False)
    save_bar_chart(df_mean, "average_accident_risk", out["plots"] / "bar_avg_accident_risk.png", "Average accident-risk exposure by algorithm", higher_better=False)
    save_bar_chart(df_mean, "average_safety", out["plots"] / "bar_avg_safety.png", "Average safety score by algorithm", higher_better=True)
    save_bar_chart(df_mean, "average_bumpiness", out["plots"] / "bar_avg_bumpiness.png", "Average bumpiness by algorithm", higher_better=False)

    save_scatter(informed_df, "execution_time", "safe_passage_score", out["plots"] / "scatter_safe_passage_vs_runtime.png", "Safe-passage score vs runtime")
    save_scatter(informed_df, "nodes_expanded", "total_path_cost", out["plots"] / "scatter_cost_vs_nodes_expanded.png", "Path cost vs nodes expanded")

    save_radar_chart(
        df_mean,
        metrics=["average_safety", "average_traffic", "average_accident_risk", "average_bumpiness", "safe_passage_score"],
        out_path=out["plots"] / "radar_safety_profile.png",
    )

    for pair_id, s, g, routes in representative:
        save_route_overlay(
            G,
            {k: v for k, v in routes.items() if k in {"Greedy best-first search", "A* search", "Weighted A*", "Uniform cost search"}},
            s,
            g,
            out["routes"] / f"route_overlay_pair_{pair_id:02d}.png",
            title=f"Pair {pair_id}: informed route comparison",
        )

    interpretation = _interpretation(df_mean)
    graph_info = f"Graph nodes={len(G)}, edges={len(G.edges())}, component={G.graph.get('component_type', 'unknown')}"
    run_info = f"pairs={args.pairs}, weighted_w={args.weighted_w}, include_ucs={args.include_ucs}, seed={args.seed}"
    summary_text = "\n".join([graph_info, run_info, "", interpretation])
    (out["summary"] / "experiment_summary.txt").write_text(summary_text, encoding="utf-8")

    _write_formula_notes(args.output_dir, cost_weights, heuristic_weights, score_weights)

    # Also store reproducibility assumptions.
    assumptions = {
        "feature_config": asdict(feature_cfg),
        "cost_weights": asdict(cost_weights),
        "heuristic_weights": asdict(heuristic_weights),
        "safe_score_weights": asdict(score_weights),
        "min_distance_m": args.min_distance_m,
        "max_distance_m": args.max_distance_m,
    }
    pd.Series(assumptions, dtype="object").to_json(out["summary"] / "assumptions.json", indent=2)

    print(summary_text)
    print(f"\nSaved pair-level table: {out['tables'] / 'pair_level_results.csv'}")
    print(f"Saved average table: {out['tables'] / 'average_metrics_by_algorithm.csv'}")
    print(f"Saved plots under: {out['plots']}")
    print(f"Saved route overlays under: {out['routes']}")


if __name__ == "__main__":
    main()
