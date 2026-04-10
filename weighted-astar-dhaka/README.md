# Weighted A* sweep on Dhaka (standalone)

Run a weighted A* sweep on the full Dhaka, Bangladesh road network and compare
against optimal Dijkstra cost. Results are saved as PNG charts in `images/`.

## Quick start
```bash
cd weighted-astar-dhaka
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_weighted_astar.py
```

What you get
- `images/wa_sweep_metrics.png` – cost, expansions, and runtime vs weight.
- `images/wa_sweep_accuracy.png` – cost ratio vs optimal.
- Console log shows per-weight stats and the start/goal nodes chosen.

Offline fallback
If OSM download fails (no network), the script automatically switches to a
synthetic grid so the experiment still runs, though the numbers won’t reflect
Dhaka.
