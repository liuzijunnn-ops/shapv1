"""Higgs Boson dataset loader (high-dimensional near-balanced binary).

The Higgs challenge data has 28 high-energy-physics features and a near
50/50 class balance.  This loader downsamples to 20k stratified rows to
keep generator wall-clock manageable while still exercising a high-dim
feature space (≈3× larger than CreditCardFraud's 11).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HG_FEATURES: List[str] = [f"f{i}" for i in range(28)]
HG_TARGET: str = "signal"

_HG_PATH = Path("dataset/Higgs/train.csv")
_TARGET_SIZE = 20_000


def load_higgs(random_state: int = 42, target_size: int = _TARGET_SIZE) -> pd.DataFrame:
    """Load Higgs Boson detection (28 numeric features, binary)."""
    if not _HG_PATH.exists():
        raise FileNotFoundError(f"Higgs file not found: {_HG_PATH}")

    # Read only needed columns (header has 'label' + f0..f27).
    df = pd.read_csv(_HG_PATH)
    df = df.rename(columns={"label": HG_TARGET})

    df[HG_TARGET] = df[HG_TARGET].astype(float).round().astype(int).clip(0, 1)
    keep = HG_FEATURES + [HG_TARGET]
    df = df[keep].copy()

    if target_size and len(df) > target_size:
        rng = np.random.RandomState(int(random_state))
        pos_idx = df.index[df[HG_TARGET] == 1].tolist()
        neg_idx = df.index[df[HG_TARGET] == 0].tolist()
        n_pos = int(round(target_size * len(pos_idx) / len(df)))
        n_neg = target_size - n_pos
        pick_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        pick_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        df = df.loc[np.concatenate([pick_pos, pick_neg])].reset_index(drop=True)

    for col in HG_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=HG_FEATURES + [HG_TARGET]).reset_index(drop=True)
    log.info(
        "  Higgs: %d samples, %d features, positive rate=%.1f%%",
        len(df), len(HG_FEATURES), df[HG_TARGET].mean() * 100,
    )
    return df
