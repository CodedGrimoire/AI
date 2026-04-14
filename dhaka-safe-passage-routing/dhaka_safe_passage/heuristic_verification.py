"""Empirical admissibility and consistency checks for informed-search heuristic.

This module reuses the same contextual feature assignment, edge-cost model,
and contextual heuristic used by informed search in this repository.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
from typing import Callable, Hashable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .contextual_features import FeatureConfig, assign_contextual_features
from .cost_functions import CostWeights, apply_contextual_cost
from .experiment_runner import sample_practical_subgraph
from .graph_builder import load_or_build_dhaka_graph
from .heuristics import HeuristicWeights, contextual_heuristic


def _largest_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    try:
        comp = max(nx.strongly_connected_components(G), key=len)
    except Exception:
        comp = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(comp).copy()


def _edge_cost_from_data(data: dict, edge_cost_key: str) -> float:
    return max(float(data.get(edge_cost_key, data.get("length", 1.0))), 1e-9)


def build_weight_function(edge_cost_key: str) -> Callable:
    """Return a NetworkX weight function compatible with (Multi)DiGraph."""

    def weight(u, v, data):
        if isinstance(data, dict) and data and all(isinstance(x, dict) for x in data.values()):
            # MultiGraph adjacency case: dict[key] -> edge attrs
            return min(_edge_cost_from_data(attrs, edge_cost_key) for attrs in data.values())
        return _edge_cost_from_data(data, edge_cost_key)

    return weight


def compute_true_remaining_costs(
    G: nx.MultiDiGraph,
    goal: Hashable,
    edge_cost_key: str = "custom_cost",
) -> dict[Hashable, float]:
    """Compute h*(n) to one goal for all reachable nodes under current cost model."""

    GR = G.reverse(copy=False)
    weight_fn = build_weight_function(edge_cost_key)
    return nx.single_source_dijkstra_path_length(GR, goal, weight=weight_fn)


def evaluate_admissibility_for_goal(
    G: nx.MultiDiGraph,
    goal: Hashable,
    heuristic: Callable[[Hashable], float],
    *,
    edge_cost_key: str = "custom_cost",
    sampled_nodes: int = 3000,
    seed: int = 2026,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed)
    nodes = list(G.nodes)
    if not nodes:
        raise RuntimeError("Graph has no nodes.")

    k = min(sampled_nodes, len(nodes))
    sampled = rng.sample(nodes, k)
    true_costs = compute_true_remaining_costs(G, goal, edge_cost_key=edge_cost_key)

    rows: list[dict] = []
    skipped_unreachable = 0

    for n in sampled:
        if n not in true_costs:
            skipped_unreachable += 1
            continue

        h_n = float(heuristic(n))
        h_star = float(true_costs[n])
        delta = h_n - h_star
        admissible = delta <= tolerance

        rows.append(
            {
                "node_id": n,
                "goal_id": goal,
                "heuristic_value": h_n,
                "true_remaining_cost": h_star,
                "delta": delta,
                "admissible": admissible,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "goal_id": goal,
            "total_nodes_sampled": k,
            "total_tested": 0,
            "skipped_unreachable": skipped_unreachable,
            "admissible_cases": 0,
            "violations": 0,
            "violation_rate": float("nan"),
            "maximum_overestimation": float("nan"),
            "average_overestimation_violating": float("nan"),
            "minimum_delta": float("nan"),
            "maximum_delta": float("nan"),
        }
        return df, summary

    viol = df[df["admissible"] == False]
    summary = {
        "goal_id": goal,
        "total_nodes_sampled": k,
        "total_tested": int(len(df)),
        "skipped_unreachable": skipped_unreachable,
        "admissible_cases": int((df["admissible"] == True).sum()),
        "violations": int(len(viol)),
        "violation_rate": float(len(viol) / len(df)),
        "maximum_overestimation": float(viol["delta"].max()) if not viol.empty else 0.0,
        "average_overestimation_violating": float(viol["delta"].mean()) if not viol.empty else 0.0,
        "minimum_delta": float(df["delta"].min()),
        "maximum_delta": float(df["delta"].max()),
    }
    return df, summary


def evaluate_admissibility_multi_goal(
    G: nx.MultiDiGraph,
    goals: list[Hashable],
    heuristic_builder: Callable[[Hashable], Callable[[Hashable], float]],
    *,
    edge_cost_key: str = "custom_cost",
    sampled_nodes_per_goal: int = 3000,
    seed: int = 2026,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    all_rows: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for i, goal in enumerate(goals):
        h_fn = heuristic_builder(goal)
        df_goal, summary_goal = evaluate_admissibility_for_goal(
            G,
            goal,
            h_fn,
            edge_cost_key=edge_cost_key,
            sampled_nodes=sampled_nodes_per_goal,
            seed=seed + i,
            tolerance=tolerance,
        )
        all_rows.append(df_goal)
        summaries.append(summary_goal)

    adm_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_by_goal_df = pd.DataFrame(summaries)

    if adm_df.empty:
        aggregate = {
            "total_node_goal_pairs_tested": 0,
            "skipped_unreachable_pairs": int(summary_by_goal_df.get("skipped_unreachable", pd.Series(dtype=int)).sum()) if not summary_by_goal_df.empty else 0,
            "admissibility_violations": 0,
            "admissibility_violation_rate": float("nan"),
            "max_overestimation": float("nan"),
            "avg_overestimation_violating": float("nan"),
            "min_delta": float("nan"),
            "max_delta": float("nan"),
        }
        return adm_df, summary_by_goal_df, aggregate

    viol = adm_df[adm_df["admissible"] == False]
    aggregate = {
        "total_node_goal_pairs_tested": int(len(adm_df)),
        "skipped_unreachable_pairs": int(summary_by_goal_df["skipped_unreachable"].sum()) if not summary_by_goal_df.empty else 0,
        "admissibility_violations": int(len(viol)),
        "admissibility_violation_rate": float(len(viol) / len(adm_df)),
        "max_overestimation": float(viol["delta"].max()) if not viol.empty else 0.0,
        "avg_overestimation_violating": float(viol["delta"].mean()) if not viol.empty else 0.0,
        "min_delta": float(adm_df["delta"].min()),
        "max_delta": float(adm_df["delta"].max()),
    }
    return adm_df, summary_by_goal_df, aggregate


def evaluate_consistency(
    G: nx.MultiDiGraph,
    goal: Hashable,
    heuristic: Callable[[Hashable], float],
    *,
    edge_cost_key: str = "custom_cost",
    sampled_edges: int = 8000,
    seed: int = 2026,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed)
    edges = list(G.edges(keys=True, data=True))
    if not edges:
        raise RuntimeError("Graph has no edges.")

    k = min(sampled_edges, len(edges))
    sample = rng.sample(edges, k)

    rows: list[dict] = []
    for u, v, _, data in sample:
        h_u = float(heuristic(u))
        h_v = float(heuristic(v))
        c_uv = _edge_cost_from_data(data, edge_cost_key)
        residual = h_u - (c_uv + h_v)
        consistent = residual <= tolerance

        rows.append(
            {
                "u": u,
                "v": v,
                "goal_id": goal,
                "h_u": h_u,
                "h_v": h_v,
                "edge_cost": c_uv,
                "residual": residual,
                "consistent": consistent,
            }
        )

    df = pd.DataFrame(rows)
    viol = df[df["consistent"] == False]
    summary = {
        "goal_id": goal,
        "total_edges_checked": int(len(df)),
        "consistent_cases": int((df["consistent"] == True).sum()),
        "violations": int(len(viol)),
        "violation_rate": float(len(viol) / len(df)) if len(df) else float("nan"),
        "max_positive_residual": float(viol["residual"].max()) if not viol.empty else 0.0,
        "avg_positive_residual_violating": float(viol["residual"].mean()) if not viol.empty else 0.0,
        "minimum_residual": float(df["residual"].min()) if len(df) else float("nan"),
        "maximum_residual": float(df["residual"].max()) if len(df) else float("nan"),
    }
    return df, summary


def _plot_histograms(adm_df: pd.DataFrame, cons_df: pd.DataFrame, out_dir: Path) -> list[str]:
    generated: list[str] = []

    if not adm_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(adm_df["delta"], bins=50, color="#1f77b4", alpha=0.85)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title("Admissibility delta histogram (delta = h(n)-h*(n))")
        ax.set_xlabel("delta")
        ax.set_ylabel("count")
        plt.tight_layout()
        p = out_dir / "admissibility_delta_histogram.png"
        plt.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(p.name)

    if not cons_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(cons_df["residual"], bins=50, color="#ff7f0e", alpha=0.85)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title("Consistency residual histogram (h(u) - (c(u,v)+h(v)))")
        ax.set_xlabel("residual")
        ax.set_ylabel("count")
        plt.tight_layout()
        p = out_dir / "consistency_residual_histogram.png"
        plt.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(p.name)

    if not adm_df.empty and not cons_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        rates = [
            float((adm_df["admissible"] == False).mean()),
            float((cons_df["consistent"] == False).mean()),
        ]
        ax.bar(["Admissibility", "Consistency"], rates, color=["#1f77b4", "#ff7f0e"])
        ax.set_ylabel("Violation rate")
        ax.set_ylim(0, max(0.01, max(rates) * 1.2))
        ax.set_title("Heuristic violation rates")
        plt.tight_layout()
        p = out_dir / "heuristic_violation_rates.png"
        plt.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(p.name)

    return generated


def _format_top_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    if df.empty:
        return "No violations found."
    top = df[cols].head(n).copy()

    # Build markdown table without optional tabulate dependency.
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in top.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *rows])


def generate_markdown_report(
    *,
    graph_label: str,
    graph_nodes: int,
    graph_edges: int,
    sampled_nodes_per_goal: int,
    sampled_edges: int,
    goals: list[Hashable],
    random_seed: int,
    edge_cost_formula: str,
    heuristic_formula: str,
    admissibility_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    admissibility_summary: dict,
    consistency_summary: dict,
    cost_weights: CostWeights,
    heuristic_weights: HeuristicWeights,
    plot_files: list[str],
) -> str:
    adm_viol = admissibility_df[admissibility_df["admissible"] == False].sort_values("delta", ascending=False)
    cons_viol = consistency_df[consistency_df["consistent"] == False].sort_values("residual", ascending=False)

    if np.isnan(admissibility_summary.get("admissibility_violation_rate", float("nan"))):
        adm_interp = "Admissibility could not be established empirically because no reachable node-goal pairs were tested."
    elif admissibility_summary["admissibility_violations"] == 0:
        adm_interp = "The heuristic appears empirically admissible on the tested samples."
    else:
        adm_interp = "The heuristic is not empirically admissible on the tested samples."

    if np.isnan(consistency_summary.get("violation_rate", float("nan"))):
        cons_interp = "Consistency could not be established empirically because no edges were tested."
    elif consistency_summary["violations"] == 0:
        cons_interp = "The heuristic appears empirically consistent on the tested edges."
    else:
        cons_interp = "The heuristic is not empirically consistent on the tested edges."

    if admissibility_summary.get("admissibility_violations", 0) == 0 and consistency_summary.get("violations", 0) == 0:
        conclusion = "Empirical checks show no sampled violations; the heuristic behaved like an admissible and consistent heuristic under this setup."
    else:
        conclusion = "Empirical checks found violations; the heuristic behaves as a non-admissible and/or non-consistent heuristic in this setup, so strict optimality guarantees are not supported."

    plots_section = "\n".join([f"- `{p}`" for p in plot_files]) if plot_files else "- None generated"

    return f"""# Heuristic Admissibility and Consistency Evaluation

## Experimental Setup
- Graph: {graph_label}
- Graph size: {graph_nodes} nodes, {graph_edges} edges
- Sampled nodes per goal: {sampled_nodes_per_goal}
- Sampled directed edges: {sampled_edges}
- Number of goal nodes: {len(goals)}
- Goal node IDs: {', '.join(str(g) for g in goals)}
- Random seed: {random_seed}
- Edge-cost model used:
  - `{edge_cost_formula}`
  - Weights: `{asdict(cost_weights)}`
- Heuristic formula used:
  - `{heuristic_formula}`
  - Weights: `{asdict(heuristic_weights)}`

## Admissibility Results
- Total tested node-goal pairs: {admissibility_summary.get('total_node_goal_pairs_tested', 0)}
- Skipped unreachable pairs: {admissibility_summary.get('skipped_unreachable_pairs', 0)}
- Number of violations: {admissibility_summary.get('admissibility_violations', 0)}
- Violation rate: {admissibility_summary.get('admissibility_violation_rate', float('nan')):.6f}
- Max overestimation: {admissibility_summary.get('max_overestimation', float('nan')):.6f}
- Avg overestimation among violations: {admissibility_summary.get('avg_overestimation_violating', float('nan')):.6f}

## Consistency Results
- Total tested edges: {consistency_summary.get('total_edges_checked', 0)}
- Number of violations: {consistency_summary.get('violations', 0)}
- Violation rate: {consistency_summary.get('violation_rate', float('nan')):.6f}
- Max positive residual: {consistency_summary.get('max_positive_residual', float('nan')):.6f}
- Avg positive residual among violations: {consistency_summary.get('avg_positive_residual_violating', float('nan')):.6f}

## Interpretation
- {adm_interp}
- {cons_interp}

## Example Violating Cases
### Top 10 Admissibility Violations
{_format_top_table(adm_viol, ['node_id', 'goal_id', 'heuristic_value', 'true_remaining_cost', 'delta'])}

### Top 10 Consistency Violations
{_format_top_table(cons_viol, ['u', 'v', 'goal_id', 'h_u', 'edge_cost', 'h_v', 'residual'])}

## Generated Artifacts
- `admissibility_results.csv`
- `consistency_results.csv`
- `admissibility_summary_by_goal.csv`
- `heuristic_verification_report.md`
{plots_section}

## Conclusion
{conclusion}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Empirical heuristic admissibility and consistency verification")
    p.add_argument("--graph-dir", type=Path, default=Path("graph_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/heuristic_verification"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sampled-nodes", type=int, default=3000)
    p.add_argument("--sampled-edges", type=int, default=8000)
    p.add_argument("--num-goals", type=int, default=5, help="Number of sampled goal nodes for admissibility")
    p.add_argument("--goal-node", type=int, default=None, help="Optional fixed goal node")
    p.add_argument("--practical-subgraph-nodes", type=int, default=0, help="Optional manageable subgraph size while retaining Dhaka context")
    p.add_argument("--edge-cost-key", type=str, default="custom_cost")
    p.add_argument("--tolerance", type=float, default=1e-9)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(vars(args))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_cfg = FeatureConfig(seed=args.seed)
    cost_weights = CostWeights(alpha_traffic=0.9, beta_accident=1.1, gamma_bumpiness=0.6, lambda_safety=0.8)
    heuristic_weights = HeuristicWeights(alpha_traffic=0.7, beta_accident=0.9, gamma_bumpiness=0.5, lambda_safety=0.8)

    G = load_or_build_dhaka_graph(args.graph_dir)
    G = sample_practical_subgraph(G, args.practical_subgraph_nodes, args.seed)
    G = _largest_component(G)

    assign_contextual_features(G, feature_cfg)
    apply_contextual_cost(G, cost_weights)

    rng = random.Random(args.seed)
    nodes = list(G.nodes)
    if not nodes:
        raise RuntimeError("No nodes in graph after preprocessing.")

    if args.goal_node is not None:
        if args.goal_node not in G:
            raise RuntimeError(f"Provided --goal-node {args.goal_node} is not in the graph component.")
        goals = [args.goal_node]
    else:
        k_goals = min(max(1, args.num_goals), len(nodes))
        goals = rng.sample(nodes, k_goals)

    heuristic_builder = lambda goal: contextual_heuristic(G, goal, heuristic_weights)

    adm_df, adm_goal_summary_df, adm_summary = evaluate_admissibility_multi_goal(
        G,
        goals,
        heuristic_builder,
        edge_cost_key=args.edge_cost_key,
        sampled_nodes_per_goal=args.sampled_nodes,
        seed=args.seed,
        tolerance=args.tolerance,
    )

    # Consistency is goal-agnostic for this heuristic family's inequality form;
    # evaluate on first goal for a concrete h(·) instance and report it.
    consistency_goal = goals[0]
    h_cons = heuristic_builder(consistency_goal)
    cons_df, cons_summary = evaluate_consistency(
        G,
        consistency_goal,
        h_cons,
        edge_cost_key=args.edge_cost_key,
        sampled_edges=args.sampled_edges,
        seed=args.seed,
        tolerance=args.tolerance,
    )

    adm_path = args.output_dir / "admissibility_results.csv"
    cons_path = args.output_dir / "consistency_results.csv"
    adm_goal_summary_path = args.output_dir / "admissibility_summary_by_goal.csv"
    report_path = args.output_dir / "heuristic_verification_report.md"

    adm_df.to_csv(adm_path, index=False)
    cons_df.to_csv(cons_path, index=False)
    adm_goal_summary_df.to_csv(adm_goal_summary_path, index=False)

    plot_files = []
    if not args.skip_plots:
        plot_files = _plot_histograms(adm_df, cons_df, args.output_dir)

    edge_cost_formula = "cost(u,v)=length(u,v)*max(min_multiplier,1+alpha*T+beta*A+gamma*B-lambda*S)"
    heuristic_formula = "h(n)=d(n,g)*max(min_factor,1+alpha*T(n)+beta*A(n)+gamma*B(n)-lambda*S(n))"

    report_md = generate_markdown_report(
        graph_label=G.graph.get("graph_label", "Dhaka road network"),
        graph_nodes=len(G),
        graph_edges=len(G.edges()),
        sampled_nodes_per_goal=args.sampled_nodes,
        sampled_edges=args.sampled_edges,
        goals=goals,
        random_seed=args.seed,
        edge_cost_formula=edge_cost_formula,
        heuristic_formula=heuristic_formula,
        admissibility_df=adm_df,
        consistency_df=cons_df,
        admissibility_summary=adm_summary,
        consistency_summary=cons_summary,
        cost_weights=cost_weights,
        heuristic_weights=heuristic_weights,
        plot_files=plot_files,
    )
    report_path.write_text(report_md, encoding="utf-8")

    print(f"Saved: {adm_path}")
    print(f"Saved: {cons_path}")
    print(f"Saved: {adm_goal_summary_path}")
    print(f"Saved: {report_path}")
    if plot_files:
        print("Saved plots:")
        for p in plot_files:
            print(f"- {args.output_dir / p}")

    print("\nAdmissibility summary:")
    print(pd.Series(adm_summary, dtype='object').to_string())
    print("\nConsistency summary:")
    print(pd.Series(cons_summary, dtype='object').to_string())


if __name__ == "__main__":
    main()
