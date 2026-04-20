# Weighted A* Comparative Analysis

This folder tracks weighted A* sweeps across multiple `w` values.

## Commands

From repo root:

```bash
source venv/bin/activate && python -m routing.experiments.weighted_astar_sweep \
  --output-dir experiments/weighted_astar_analysis/images/full_dhaka
```

```bash
source venv/bin/activate && python -m routing.experiments.weighted_astar_sweep \
  --max-nodes 500 \
  --output-dir experiments/weighted_astar_analysis/images/dhaka_500_nodes
```
