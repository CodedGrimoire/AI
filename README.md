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

## Organized experiment folders

Use the three dedicated folders under `experiments/`:

- `experiments/all_algorithms`
- `experiments/weighted_astar_analysis`
- `experiments/heuristic_check`

Each folder has:

- its own `images/` directory
- two run commands in its local `README.md`:
  - full Dhaka map
  - Dhaka map restricted to 1000 nodes

## Quick commands (repo root)

Activate env first:

```bash
source venv/bin/activate
```

All algorithms:

```bash
python -m routing.experiments.run \
  --dls-limit 300 \
  --ids-max-depth 500 \
  --output-dir experiments/all_algorithms/images/full_dhaka
python -m routing.experiments.run \
  --max-nodes 1000 \
  --dls-limit 300 \
  --ids-max-depth 500 \
  --output-dir experiments/all_algorithms/images/dhaka_1000_nodes
```

Weighted A* comparative sweep:

```bash
python -m routing.experiments.weighted_astar_sweep --output-dir experiments/weighted_astar_analysis/images/full_dhaka
python -m routing.experiments.weighted_astar_sweep --max-nodes 1000 --output-dir experiments/weighted_astar_analysis/images/dhaka_1000_nodes
```

Heuristic verification:

```bash
python -m routing.experiments.heuristic_verification --output-dir experiments/heuristic_check/images/full_dhaka
python -m routing.experiments.heuristic_verification --max-nodes 1000 --output-dir experiments/heuristic_check/images/dhaka_1000_nodes
```

## UI Dashboard

Run an interactive UI with a menu for:
- running one selected algorithm on the full Dhaka map
- running the whole experiment suite on the full Dhaka map
- running Weighted A* with custom weight values

```bash
source venv/bin/activate
streamlit run routing/ui/dashboard.py
```

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
