# All Algorithms Experiment

This folder tracks runs where all search algorithms are executed and compared.

## Commands

From repo root:

```bash
source venv/bin/activate && python -m routing.experiments.run \
  --output-dir experiments/all_algorithms/images/full_dhaka
```

```bash
source venv/bin/activate && python -m routing.experiments.run \
  --max-nodes 1000 \
  --output-dir experiments/all_algorithms/images/dhaka_1000_nodes
```
