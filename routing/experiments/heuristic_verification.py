"""Empirical admissibility/consistency checks for exponential heuristic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Hashable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from routing.experiments.graph_scope import load_graph_with_costs
from routing.heuristics.spatial import euclidean_heuristic, exponential_feature_heuristic


def choose_goal_far(G) -> Hashable:
    nodes = list(G.nodes)
    start = nodes[0]
    goal = max(nodes, key=lambda n: euclidean_heuristic(G, start, n, 1.0))
    return goal


def edge_min_cost(G, u: Hashable, v: Hashable) -> float:
    datas = G.get_edge_data(u, v)
    if not datas:
        return float("inf")
    return min(float(d.get("custom_cost", 1.0)) for d in datas.values())


def compute_heuristics(G, goal: Hashable) -> Dict[Hashable, float]:
    h_fn = exponential_feature_heuristic(G, goal)
    return {n: h_fn(n) for n in G.nodes}


def consistency_frame(G, h: Dict[Hashable, float]) -> pd.DataFrame:
    rows = []
    for u, v in G.edges():
        c = edge_min_cost(G, u, v)
        residual = h[u] - (c + h[v])
        rows.append(
            {
                "u": u,
                "v": v,
                "h_u": h[u],
                "edge_cost": c,
                "h_v": h[v],
                "residual": residual,
                "is_violation": residual > 1e-9,
            }
        )
    return pd.DataFrame(rows)


def admissibility_frame(G, goal: Hashable, h: Dict[Hashable, float]) -> pd.DataFrame:
    # true_cost_to_goal(n) via Dijkstra on reversed graph from goal
    rev = G.reverse(copy=False)
    true_costs = dict(nx.single_source_dijkstra_path_length(rev, goal, weight="custom_cost"))
    rows = []
    for n in G.nodes:
        tc = float(true_costs.get(n, float("inf")))
        residual = h[n] - tc
        rows.append(
            {
                "node": n,
                "h_n": h[n],
                "true_cost_to_goal": tc,
                "residual": residual,
                "is_violation": residual > 1e-9,
            }
        )
    return pd.DataFrame(rows)


def write_plots(cons_df: pd.DataFrame, adm_df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(cons_df["residual"], bins=60, color="#1f77b4", alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="--")
    plt.title("Consistency residuals: h(u) - (c(u,v)+h(v))")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "consistency_residual_histogram.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    finite = adm_df[adm_df["true_cost_to_goal"] < float("inf")]
    plt.hist(finite["residual"], bins=60, color="#2ca02c", alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="--")
    plt.title("Admissibility residuals: h(n) - true_cost(n,goal)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "admissibility_delta_histogram.png", dpi=180)
    plt.close()

    rates = pd.DataFrame(
        [
            {
                "check": "consistency",
                "total": int(len(cons_df)),
                "violations": int(cons_df["is_violation"].sum()),
            },
            {
                "check": "admissibility",
                "total": int(len(adm_df)),
                "violations": int(adm_df["is_violation"].sum()),
            },
        ]
    )
    rates["violation_rate"] = rates["violations"] / rates["total"].replace(0, 1)
    plt.figure(figsize=(6, 4))
    plt.bar(rates["check"], rates["violation_rate"], color=["#1f77b4", "#2ca02c"])
    plt.ylim(0, 1)
    plt.ylabel("Violation rate")
    plt.title("Heuristic violation rates")
    plt.tight_layout()
    plt.savefig(out_dir / "heuristic_violation_rates.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic admissibility/consistency verification")
    parser.add_argument("--max-nodes", type=int, default=None, help="Limit graph to approximately this many nodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for node-limited sampling")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images/heuristic_verification",
        help="Directory to write reports/results",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = load_graph_with_costs(use_osm=True, max_nodes=args.max_nodes, seed=args.seed)
    goal = choose_goal_far(G)
    h = compute_heuristics(G, goal)

    cons_df = consistency_frame(G, h)
    adm_df = admissibility_frame(G, goal, h)

    cons_df.to_csv(out_dir / "consistency_results.csv", index=False)
    adm_df.to_csv(out_dir / "admissibility_results.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "check": "consistency",
                "total_checked": int(len(cons_df)),
                "violations": int(cons_df["is_violation"].sum()),
                "violation_rate": float(cons_df["is_violation"].mean()),
                "max_residual": float(cons_df["residual"].max()),
                "avg_positive_residual_violating": float(cons_df.loc[cons_df["is_violation"], "residual"].mean())
                if cons_df["is_violation"].any()
                else 0.0,
            },
            {
                "check": "admissibility",
                "total_checked": int(len(adm_df)),
                "violations": int(adm_df["is_violation"].sum()),
                "violation_rate": float(adm_df["is_violation"].mean()),
                "max_residual": float(adm_df["residual"].max()),
                "avg_positive_residual_violating": float(adm_df.loc[adm_df["is_violation"], "residual"].mean())
                if adm_df["is_violation"].any()
                else 0.0,
            },
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    write_plots(cons_df, adm_df, out_dir)

    scope = "full_map" if args.max_nodes is None else f"~{args.max_nodes}_nodes"
    print(f"[info] Scope={scope} | nodes={len(G)} edges={len(G.edges())} goal={goal}")
    print(f"[result] Saved: {out_dir / 'consistency_results.csv'}")
    print(f"[result] Saved: {out_dir / 'admissibility_results.csv'}")
    print(f"[result] Saved: {out_dir / 'summary.csv'}")
    print(f"[result] Saved: {out_dir / 'consistency_residual_histogram.png'}")
    print(f"[result] Saved: {out_dir / 'admissibility_delta_histogram.png'}")
    print(f"[result] Saved: {out_dir / 'heuristic_violation_rates.png'}")


if __name__ == "__main__":
    main()
