# Heuristic Verification

This folder tracks admissibility and consistency verification outputs.

## Commands

From repo root:

```bash
source venv/bin/activate && python -m routing.experiments.heuristic_verification \
  --output-dir experiments/heuristic_check/images/full_dhaka
```

```bash
source venv/bin/activate && python -m routing.experiments.heuristic_verification \
  --max-nodes 1000 \
  --output-dir experiments/heuristic_check/images/dhaka_1000_nodes
```
