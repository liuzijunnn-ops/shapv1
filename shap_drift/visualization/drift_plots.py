"""Drift visualization plots — SHAP drift heatmaps, bar charts, per-sample consistency."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from shap_drift.config import OUTPUT_DIR, FIGURE_DIR, GEN_COLORS
from shap_drift.datasets import DATASET_ORDER, DS_LABELS, DS_COLORS
from shap_drift.visualization import setup_plot_style, _load_json

log = logging.getLogger(__name__)


def plot_drift_heatmap(
    drift_data: Dict[str, Any],
    output_dir: Path = FIGURE_DIR,
) -> None:
    """Plot SHAP drift ρ heatmap: datasets × generators."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_plot_style()

    # Extract rho means for XGBoost (most representative model)
    rows = []
    for ds_name in DATASET_ORDER:
        ds_drift = drift_data.get(ds_name, {})
        xgb_drift = ds_drift.get("XGBoost", {})
        for gen_name in GEN_COLORS:
            md = xgb_drift.get(gen_name, {})
            ms = md.get("multi_seed", {})
            rho_mean = ms.get("rho", {}).get("mean", md.get("rho", 0))
            rows.append({"Dataset": DS_LABELS.get(ds_name, ds_name), "Generator": gen_name, "ρ": rho_mean})

    if not rows:
        log.warning("  No drift data for heatmap")
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Dataset", columns="Generator", values="ρ")
    pivot = pivot.reindex(index=[DS_LABELS.get(d, d) for d in DATASET_ORDER if DS_LABELS.get(d, d) in pivot.index])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                center=0.5, ax=ax, linewidths=0.5)
    ax.set_title("SHAP Explanation Drift (Spearman ρ)")
    fig.tight_layout()
    fig.savefig(output_dir / "drift_heatmap.pdf", dpi=200)
    plt.close(fig)


def plot_drift_bar(
    drift_data: Dict[str, Any],
    output_dir: Path = FIGURE_DIR,
) -> None:
    """Plot per-dataset drift ρ bar chart."""
    import matplotlib.pyplot as plt
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DATASET_ORDER))
    width = 0.18
    gen_names = list(GEN_COLORS.keys())

    for i, gen_name in enumerate(gen_names):
        rhos = []
        for ds_name in DATASET_ORDER:
            ds_drift = drift_data.get(ds_name, {})
            xgb_drift = ds_drift.get("XGBoost", {})
            md = xgb_drift.get(gen_name, {})
            ms = md.get("multi_seed", {})
            rhos.append(ms.get("rho", {}).get("mean", md.get("rho", 0)))
        ax.bar(x + i * width, rhos, width, label=gen_name, color=GEN_COLORS[gen_name])

    ax.set_xticks(x + width * (len(gen_names) - 1) / 2)
    ax.set_xticklabels([DS_LABELS.get(d, d) for d in DATASET_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("SHAP Drift by Dataset × Generator")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "drift_bar.pdf", dpi=200)
    plt.close(fig)


def plot_per_sample_consistency(
    persample_data: Dict[str, Any],
    output_dir: Path = FIGURE_DIR,
) -> None:
    """Plot per-sample cosine similarity and rank correlation."""
    import matplotlib.pyplot as plt
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rows_cos, rows_rho = [], []
    for ds_name in DATASET_ORDER:
        ds_ps = persample_data.get(ds_name, {})
        xgb_ps = ds_ps.get("XGBoost", {})
        for gen_name in GEN_COLORS:
            ps = xgb_ps.get(gen_name, {})
            if ps:
                rows_cos.append({"Dataset": DS_LABELS.get(ds_name, ds_name), "Generator": gen_name,
                                 "value": ps.get("cos_sim_mean", 0)})
                rows_rho.append({"Dataset": DS_LABELS.get(ds_name, ds_name), "Generator": gen_name,
                                 "value": ps.get("per_rho_mean", 0)})

    if not rows_cos:
        plt.close(fig)
        return

    df_cos = pd.DataFrame(rows_cos)
    df_rho = pd.DataFrame(rows_rho)

    for ax, df, title, ylabel in [(ax1, df_cos, "Cosine Similarity", "Mean Cos Sim"),
                                   (ax2, df_rho, "Rank Correlation", "Mean ρ")]:
        pivot = df.pivot(index="Dataset", columns="Generator", values="value")
        pivot = pivot.reindex(index=[DS_LABELS.get(d, d) for d in DATASET_ORDER
                                     if DS_LABELS.get(d, d) in pivot.index])
        pivot.plot(kind="bar", ax=ax, color=[GEN_COLORS.get(g, "#999") for g in pivot.columns])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_dir / "per_sample_consistency.pdf", dpi=200)
    plt.close(fig)
