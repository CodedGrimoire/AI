"""Empirically verify exponential heuristic for admissibility and consistency.

Usage:
    python heuristic_verification.py

Graph + costs come from the existing pipeline:
- route_planning.graph_builder.generate_graph
- route_planning.features_costs.assign_synthetic_features / apply_cost

Heuristic: exponential feature heuristic to the chosen goal node.

Checks:
- Consistency: h(u) <= c(u,v) + h(v) for every directed edge (uses min custom_cost per multi-edge).
- Admissibility: h(n) <= true shortest-path cost(n -> goal) via Dijkstra.

Outputs a concise summary and a few example violations if any.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple

import networkx as nx

from routing.data.graph_builder import generate_graph
from routing.data.features_costs import assign_synthetic_features, apply_cost
from routing.heuristics.spatial import euclidean_heuristic, exponential_feature_heuristic
from routing.algorithms import uniform_cost_search
from routing.experiments.run import choose_start_goal


def edge_min_cost(G: nx.MultiDiGraph, u: Hashable, v: Hashable) -> float:
    """Return the minimum custom_cost over parallel edges u->v (default 1.0)."""
    datas = G.get_edge_data(u, v)
    if not datas:
        return float("inf")
    return min(float(d.get("custom_cost", 1.0)) for d in datas.values())


def compute_heuristics(G: nx.MultiDiGraph, goal: Hashable) -> Dict[Hashable, float]:
    h_fn = exponential_feature_heuristic(G, goal)
    return {n: h_fn(n) for n in G.nodes}


def check_consistency(G: nx.MultiDiGraph, h: Dict[Hashable, float]):
    total = 0
    violations: List[Tuple[Hashable, Hashable, float, float, float]] = []
    for u, v in G.edges():
        total += 1
        c = edge_min_cost(G, u, v)
        if h[u] > c + h[v] + 1e-9:  # tiny tolerance
            if len(violations) < 5:
                violations.append((u, v, h[u], c, h[v]))
    return total, violations


def dijkstra_true_cost(G: nx.MultiDiGraph, start: Hashable, goal: Hashable) -> float:
    # Uniform cost search equals Dijkstra on positive edge distances.
    res = uniform_cost_search(G, start, goal)
    return res.total_path_cost if res.path_found else float("inf")


def check_admissibility(G: nx.MultiDiGraph, goal: Hashable, h: Dict[Hashable, float]):
    total = 0
    violations: List[Tuple[Hashable, float, float]] = []
    for n in G.nodes:
        total += 1
        true_cost = dijkstra_true_cost(G, n, goal)
        if h[n] > true_cost + 1e-9:
            if len(violations) < 5:
                violations.append((n, h[n], true_cost))
    return total, violations


def build_graph():
    center = (23.746, 90.376)  # Dhaka example
    G = generate_graph(center, min_nodes=100, max_nodes=140)
    assign_synthetic_features(G)
    apply_cost(G)
    start, goal, G_conn = choose_start_goal(G)
    return G_conn, goal


def main():
    G, goal = build_graph()
    h = compute_heuristics(G, goal)

    # Consistency
    edges_checked, cons_violations = check_consistency(G, h)
    cons_status = "CONSISTENT" if not cons_violations else "NOT CONSISTENT"

    # Admissibility
    nodes_checked, adm_violations = check_admissibility(G, goal, h)
    adm_status = "ADMISSIBLE" if not adm_violations else "NOT ADMISSIBLE"

    print("=== HEURISTIC VERIFICATION ===")
    print()
    print("[Consistency]")
    print(f"Edges checked: {edges_checked}")
    print(f"Violations: {len(cons_violations)}")
    print(f"Status: {cons_status}")
    if cons_violations:
        print("Examples (u, v, h(u), cost, h(v)):")
        for u, v, hu, c, hv in cons_violations:
            print(f"  {u} -> {v} | h(u)={hu:.4f}, cost={c:.4f}, h(v)={hv:.4f}")

    print()
    print("[Admissibility]")
    print(f"Nodes checked: {nodes_checked}")
    print(f"Violations: {len(adm_violations)}")
    print(f"Status: {adm_status}")
    if adm_violations:
        print("Examples (n, h(n), true_cost):")
        for n, hn, tc in adm_violations:
            print(f"  {n} | h={hn:.4f}, true_cost={tc:.4f}")


if __name__ == "__main__":
    main()
