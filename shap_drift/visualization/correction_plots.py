"""Correction visualization plots."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from shap_drift.config import OUTPUT_DIR, FIGURE_DIR
from shap_drift.datasets import DATASET_ORDER, DS_LABELS
from shap_drift.visualization import setup_plot_style, METHOD_COLORS

log = logging.getLogger(__name__)


def plot_correction_comparison(
    correction_data: Dict[str, Any],
    output_dir: Path = FIGURE_DIR,
) -> None:
    """Plot correction method comparison (bar chart)."""
    import matplotlib.pyplot as plt
    setup_plot_style()

    # Extract from significance JSON or aggregate CSV
    csv_path = OUTPUT_DIR / "multiseed_correction_aggregate.csv"
    if not csv_path.exists():
        log.warning("  No correction aggregate data found")
        return

    df = pd.read_csv(csv_path)
    methods = [("SDC-Corr_mean", "SDC-Corr"), ("A4:CV_mean", "A4:CV")]

    fig, ax = plt.subplots(figsize=(10, 5))
    datasets = df["dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.3

    for i, (col, label) in enumerate(methods):
        if col not in df.columns:
            continue
        vals = [df[df["dataset"] == ds][col].mean() - df[df["dataset"] == ds]["Original_mean"].mean()
                for ds in datasets]
        ax.bar(x + i * width, vals, width, label=label,
               color=METHOD_COLORS.get(label, "#999"))

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([DS_LABELS.get(d, d) for d in datasets], rotation=30, ha="right")
    ax.set_ylabel("Δρ (improvement over original)")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title("Correction Method Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "correction_comparison.pdf", dpi=200)
    plt.close(fig)


def plot_guarded_vs_standard(
    results_csv: Path = OUTPUT_DIR / "guarded_correction_results.csv",
    output_dir: Path = FIGURE_DIR,
) -> None:
    """Plot SDC-Corr vs SDC-Guarded comparison."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_plot_style()

    if not results_csv.exists():
        log.warning("  No guarded correction data found")
        return

    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter of orig_rho vs delta
    ax = axes[0]
    for method, col, color in [("SDC-Corr", "sdc_delta", METHOD_COLORS["SDC-Corr"]),
                                ("SDC-Guarded", "guarded_delta", METHOD_COLORS["SDC-Guarded"])]:
        ax.scatter(df["orig_rho"], df[col], alpha=0.3, s=20, label=method, color=color)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Original ρ")
    ax.set_ylabel("Δρ")
    ax.set_title("Correction Effect vs Drift Severity")
    ax.legend()

    # Right: boxplot by dataset
    ax = axes[1]
    plot_df = df.melt(id_vars=["dataset"], value_vars=["sdc_delta", "guarded_delta"],
                      var_name="Method", value_name="Δρ")
    plot_df["Method"] = plot_df["Method"].map({"sdc_delta": "SDC-Corr", "guarded_delta": "SDC-Guarded"})
    sns.boxplot(data=plot_df, x="dataset", y="Δρ", hue="Method",
                palette=METHOD_COLORS, ax=ax)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xticklabels([DS_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                       rotation=30, ha="right")
    ax.set_title("Guarded vs Standard by Dataset")

    fig.tight_layout()
    fig.savefig(output_dir / "guarded_vs_standard.pdf", dpi=200)
    plt.close(fig)
