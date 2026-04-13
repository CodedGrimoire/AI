# Dhaka Safe-Passage Routing Experiments

Standalone extension repo for informed-search safe-passage evaluation on Dhaka road networks.

## What This Adds
- Deterministic Dhaka-wide contextual edge attributes:
  - `traffic_level`
  - `accident_risk`
  - `bumpiness`
  - `safety_score`
- Contextual informed-routing cost model:
  - `cost(u,v) = length(u,v) * max(min_multiplier, 1 + alpha*T + beta*A + gamma*B - lambda*S)`
- Contextual heuristic for informed search:
  - `h(n) = d(n,g) * max(min_factor, 1 + alpha*T(n) + beta*A(n) + gamma*B(n) - lambda*S(n))`
- Path quality metrics + safe passage score:
  - `safe_passage_score = w1*avg_safety - w2*avg_traffic - w3*avg_accident_risk - w4*avg_bumpiness`
- Whole-Dhaka multi-pair experiments and report-ready plots.

## Project Structure
- `dhaka_safe_passage/graph_builder.py`
- `dhaka_safe_passage/contextual_features.py`
- `dhaka_safe_passage/cost_functions.py`
- `dhaka_safe_passage/heuristics.py`
- `dhaka_safe_passage/metrics.py`
- `dhaka_safe_passage/visualization.py`
- `dhaka_safe_passage/experiment_runner.py`
- `dhaka_safe_passage/algorithms/*` (search implementations)

## Important Scope Rule
- Uninformed search implementations are copied as-is from the base project.
- Safe-passage objective is applied to informed algorithms (Greedy, A*, Weighted A*) with optional UCS reference.

## Graph Cache and Outputs
- Graph cache folder (separate from your old repo runs): `graph_cache/`
- Experiment outputs: `outputs/safe_passage_dhaka/`

## Setup
```bash
cd dhaka-safe-passage-routing
source ../venv/bin/activate
pip install -r requirements.txt
```

## Run (Default)
```bash
python -m dhaka_safe_passage.experiment_runner --pairs 12 --weighted-w 1.5 --include-ucs
```

## Run (Practical Mode For Large Graph)
```bash
python -m dhaka_safe_passage.experiment_runner \
  --pairs 15 \
  --weighted-w 1.5 \
  --include-ucs \
  --practical-subgraph-nodes 18000
```

## Main Outputs
- `outputs/safe_passage_dhaka/tables/pair_level_results.csv`
- `outputs/safe_passage_dhaka/tables/average_metrics_by_algorithm.csv`
- `outputs/safe_passage_dhaka/plots/*.png`
- `outputs/safe_passage_dhaka/routes/route_overlay_pair_*.png`
- `outputs/safe_passage_dhaka/summary/experiment_summary.txt`
- `outputs/safe_passage_dhaka/summary/formulas.txt`
- `outputs/safe_passage_dhaka/summary/assumptions.json`

## Heuristic Verification (Admissibility + Consistency)
Run with module:
```bash
MPLCONFIGDIR=/tmp/mpl python -m dhaka_safe_passage.heuristic_verification \
  --sampled-nodes 3000 \
  --sampled-edges 8000 \
  --num-goals 5
```

Or with convenience script:
```bash
MPLCONFIGDIR=/tmp/mpl ./run_heuristic_verification.py \
  --sampled-nodes 3000 \
  --sampled-edges 8000 \
  --num-goals 5
```

Same fixed goal node as your previous experiment:
```bash
MPLCONFIGDIR=/tmp/mpl python -m dhaka_safe_passage.heuristic_verification \
  --goal-node 4594175828 \
  --sampled-nodes 3000 \
  --sampled-edges 8000
```

Heuristic verification outputs:
- `outputs/heuristic_verification/admissibility_results.csv`
- `outputs/heuristic_verification/consistency_results.csv`
- `outputs/heuristic_verification/admissibility_summary_by_goal.csv`
- `outputs/heuristic_verification/heuristic_verification_report.md`
- `outputs/heuristic_verification/admissibility_delta_histogram.png`
- `outputs/heuristic_verification/consistency_residual_histogram.png`
- `outputs/heuristic_verification/heuristic_violation_rates.png`
