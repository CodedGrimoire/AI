"""Shared output-directory helpers for experiment scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def build_output_paths(experiment_name: str) -> Dict[str, Path]:
    """Create and return standard output folders under images/<experiment_name>."""
    root = Path("images") / experiment_name
    dirs = {
        "root": root,
        "plots": root / "plots",
        "csv": root / "csv",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
