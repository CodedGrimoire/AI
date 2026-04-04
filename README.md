# Routing Experiments

Modular layout for road-network search experiments, heuristics checks, and plotting.

## Environment
- Python 3.11+ recommended.
- Activate your venv if present: `source venv/bin/activate`.
- Optional: set `MPLCONFIGDIR=/tmp/mpl` to silence matplotlib cache warnings in headless runs.

## Key package
`routing/` is a package with submodules:
- `routing.data` – graph generation, synthetic features, and cost model.
- `routing.heuristics` – heuristic functions (e.g., Euclidean).
- `routing.algorithms` – Dijkstra, Greedy, A*, Weighted A*.
- `routing.viz` – plotting helpers.
- `routing.experiments` – runnable scripts.
- `routing.utils` – OSMnx helpers.

### Code structure
```
routing/
  __init__.py              # public API re-exports
  data/
    graph_builder.py       # graph generation
    features_costs.py      # synthetic features + edge cost model
  heuristics/
    __init__.py
    spatial.py             # Euclidean heuristic
  algorithms/
    __init__.py
    search.py              # Dijkstra, Greedy, A*, Weighted A*
  viz/
    plotting.py            # single/overlay route plots
  experiments/
    run.py                 # end-to-end experiments & charts
    weighted_astar_sweep.py# WA* weight sweep plots
    heuristic_verification.py # heuristic checks
  utils/
    osmnx_map.py           # OSMnx map utilities
images/                    # generated figures
cache/                     # OSMnx graph cache
README.md                  # this guide
```

## How to run

| Purpose | Command (from repo root) | Outputs |
| --- | --- | --- |
| Full experiment suite (linear + extended cost models, charts, route PNGs) | `python -m routing.experiments.run` | PNGs in `images/` (cost_vs_expansion, linear_vs_extended, routes, bars, accuracy) |
| Weighted A* sweep (multiple weights, metrics + accuracy) | `python -m routing.experiments.weighted_astar_sweep` | `images/wa_sweep_metrics.png`, `images/wa_sweep_accuracy.png` |
| Heuristic admissibility/consistency check | `python -m routing.experiments.heuristic_verification` | Prints summary to stdout |

All scripts rebuild a small OSM graph around Dhaka, attach synthetic features, apply the cost model, run the algorithms, and save figures to `images/`.

## Public API shortcuts
Import helpers directly from the package:
```python
from routing import (
    generate_graph,
    assign_synthetic_features,
    apply_cost,
    euclidean_heuristic,
    dijkstra_search,
    a_star_search,
    weighted_a_star_search,
    greedy_best_first_search,
    compute_path_cost,
    plot_single_route,
    plot_all_routes,
)
```

## Notes
- Cached OSM graphs live in `cache/` (OSMnx default behavior).
- Generated plots are collected in `images/` so the repo stays tidy.
