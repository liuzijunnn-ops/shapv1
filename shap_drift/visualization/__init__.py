"""Visualization utilities — shared across all plotting scripts."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from shap_drift.config import GEN_COLORS, OUTPUT_DIR
from shap_drift.datasets import DATASETS, DATASET_ORDER, DS_LABELS, DS_COLORS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Correction method color palette
# ---------------------------------------------------------------------------
METHOD_COLORS: Dict[str, str] = {
    "Original": "#9E9E9E",
    "SDC-Corr": "#1E88E5",
    "SDC-Guarded": "#43A047",
    "FSC-Corr": "#FB8C00",
    "FSC-Guarded": "#E53935",
}

# ---------------------------------------------------------------------------
# Baseline method color palette
# ---------------------------------------------------------------------------
BASELINE_COLORS: Dict[str, str] = {
    "SHAP Distillation": "#00897B",
    "Fine-tuning": "#5C6BC0",
    "CORAL": "#FF7043",
}

# ---------------------------------------------------------------------------
# Ablation method color palette
# ---------------------------------------------------------------------------
ABLATION_COLORS: Dict[str, str] = {
    "Fixed α": "#78909C",
    "CV α": "#E53935",
    "Adaptive α": "#43A047",
    "Correlation prior": "#1E88E5",
    "Uniform prior": "#9E9E9E",
    "Random prior": "#FF7043",
    "MI prior": "#7B1FA2",
    "Random sampling": "#9E9E9E",
    "Density sampling": "#1E88E5",
    "Boundary sampling": "#E53935",
}

# ---------------------------------------------------------------------------
# Standard publication-quality plot settings
# ---------------------------------------------------------------------------
PLT_RCPARAMS: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


def setup_plot_style() -> None:
    """Apply standard plot style to matplotlib rcParams."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLT_RCPARAMS)


def _load_json(filename: str, output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    """Load a JSON result file, returning empty dict on failure."""
    fpath = output_dir / f"{filename}.json"
    try:
        with open(fpath) as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("  Missing data file: %s", fpath)
        return {}
    except json.JSONDecodeError as exc:
        log.error("  Invalid JSON in %s: %s", fpath, exc)
        return {}
