"""Forest Cover Type dataset loader (multi-class → binary).

Original task is 7-class forest cover prediction (581K samples).  For the
SHAP-drift study we binarize against the second-most-common class
(``Cover_Type == 2``, ≈49% prevalence — near-balanced, makes statistical
tests well-powered) and keep the 10 numeric topographic / hydrographic
features (dropping the 44 one-hot Wilderness/Soil flags which are
near-binary and would dominate KS distance estimates).

Down-sampled to ~30k rows by stratified sampling to keep generator
training tractable on commodity hardware.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CT_FEATURES: List[str] = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
CT_TARGET: str = "high_cover"

_CT_PATH = Path("dataset/CovType/covtype.csv")
_POSITIVE_CLASS = 2  # Most common (≈49%) — balanced binary target.
_TARGET_SIZE = 30_000


def load_covtype(random_state: int = 42, target_size: int = _TARGET_SIZE) -> pd.DataFrame:
    """Load Forest Cover Type as binarized (class 2 vs others).

    Args:
        random_state: seed for stratified downsampling.
        target_size: total samples after stratified subsample.
    """
    if not _CT_PATH.exists():
        raise FileNotFoundError(f"CovType file not found: {_CT_PATH}")

    df = pd.read_csv(_CT_PATH)
    df[CT_TARGET] = (df["Cover_Type"] == _POSITIVE_CLASS).astype(int)

    keep = CT_FEATURES + [CT_TARGET]
    df = df[keep].copy()

    # Stratified downsample by binary target.
    if target_size and len(df) > target_size:
        rng = np.random.RandomState(int(random_state))
        pos_idx = df.index[df[CT_TARGET] == 1].tolist()
        neg_idx = df.index[df[CT_TARGET] == 0].tolist()
        # Preserve original class proportions.
        n_pos = int(round(target_size * len(pos_idx) / len(df)))
        n_neg = target_size - n_pos
        pick_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        pick_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        df = df.loc[np.concatenate([pick_pos, pick_neg])].reset_index(drop=True)

    for col in CT_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=CT_FEATURES + [CT_TARGET]).reset_index(drop=True)
    log.info(
        "  CovType: %d samples, %d features, positive rate=%.1f%%",
        len(df), len(CT_FEATURES), df[CT_TARGET].mean() * 100,
    )
    return df
