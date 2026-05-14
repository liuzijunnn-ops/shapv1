"""Baseline comparison visualization plots."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from shap_drift.config import OUTPUT_DIR, FIGURE_DIR
from shap_drift.datasets import DATASET_ORDER, DS_LABELS
from shap_drift.visualization import setup_plot_style, BASELINE_COLORS, METHOD_COLORS

log = logging.getLogger(__name__)


def plot_fair_baseline(
    results_csv: Path = OUTPUT_DIR / "fair_baseline_comparison.csv",
    output_dir: Path = FIGURE_DIR,
) -> None:
    """Plot fair baseline comparison at multiple budget levels."""
    import matplotlib.pyplot as plt
    setup_plot_style()

    if not results_csv.exists():
        log.warning("  No fair baseline data found")
        return

    df = pd.read_csv(results_csv)
    methods = [
        ("fsc_delta", "FSC-Corr", METHOD_COLORS.get("FSC-Corr", "#FB8C00")),
        ("distill_delta", "SHAP Distill", BASELINE_COLORS["SHAP Distillation"]),
        ("ft_delta", "Fine-tuning", BASELINE_COLORS["Fine-tuning"]),
        ("sdc_delta", "SDC-Corr (100%)", METHOD_COLORS.get("SDC-Corr", "#1E88E5")),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fractions = sorted(df["fraction"].unique())

    for col, label, color in methods:
        means = [df[df["fraction"] == f][col].mean() for f in fractions]
        ax.plot([f * 100 for f in fractions], means, "o-", label=label, color=color, linewidth=2)

    ax.set_xlabel("Real Data Budget (%)")
    ax.set_ylabel("Mean Δρ")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title("Fair Baseline Comparison at Equal Data Budget")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "fair_baseline_comparison.pdf", dpi=200)
    plt.close(fig)


def plot_ablation(
    output_dir: Path = FIGURE_DIR,
) -> None:
    """Plot ablation experiment results."""
    import matplotlib.pyplot as plt
    from shap_drift.visualization import ABLATION_COLORS

    setup_plot_style()

    for ablation_type, filename in [("alpha", "ablation_alpha.csv"),
                                     ("prior", "ablation_prior.csv"),
                                     ("sample", "ablation_sample.csv")]:
        csv_path = OUTPUT_DIR / filename
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        delta_cols = [c for c in df.columns if c.endswith("_delta")]
        if not delta_cols:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        means = [df[c].mean() for c in delta_cols]
        stds = [df[c].std() for c in delta_cols]
        labels = [c.replace(f"{ablation_type}_", "").replace("_delta", "") for c in delta_cols]

        ax.barh(range(len(labels)), means, xerr=stds, color="#3498db", alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Δρ")
        ax.set_title(f"{ablation_type.title()} Selection Ablation")
        fig.tight_layout()
        fig.savefig(output_dir / f"ablation_{ablation_type}.pdf", dpi=200)
        plt.close(fig)
