"""Thyroid Disease dataset loader (heavily imbalanced binary).

The original ThyroidDF has 20+ diagnostic-code targets (K, G, I, F, …) plus
a "-" code for healthy patients (73.8%).  We binarize the target as
``healthy (target == '-') vs. any-diagnosis`` to study explanation drift
under heavy class imbalance (≈26% positive rate, with several minority
diagnoses < 2%).

Feature set: 6 numeric thyroid panel measurements (age, TSH, T3, TT4, T4U,
FTI) + 2 boolean clinical flags (sex, on_thyroxine).  ``TBG`` is dropped
because 96% of patients are missing the measurement.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TH_FEATURES: List[str] = [
    "age",            # numeric
    "TSH", "T3", "TT4", "T4U", "FTI",  # 5 thyroid panel measurements
    "sex",            # F=0, M=1
    "on_thyroxine",   # t=1, f=0
]
TH_TARGET: str = "diseased"

_TH_PATH = Path("dataset/Thyroid/thyroidDF.csv")


def _encode_bool(s: pd.Series) -> pd.Series:
    """Map 't'/'f' → 1/0; 'M'/'F' → 1/0; coerce others to NaN."""
    s = s.astype(str).str.lower().str.strip()
    return s.map({"t": 1, "f": 0, "m": 1, "f.": 0}).fillna(
        s.map({"true": 1, "false": 0})
    ).astype(float)


def load_thyroid() -> pd.DataFrame:
    """Load Thyroid Disease dataset.

    Pipeline:
      1. Binarize target: ``-`` (healthy) → 0, anything else → 1.
      2. Encode ``sex`` (F=0, M=1) and ``on_thyroxine`` (t/f → 0/1).
      3. Convert numeric thyroid panels; median-impute missing values.
      4. Drop ``patient_id`` and 96%-missing ``TBG``.

    Edge case: when a numeric column has *all* missing values for a row,
    median imputation still produces a finite value globally — but we
    drop any row left with NaN after imputation to be safe.
    """
    if not _TH_PATH.exists():
        raise FileNotFoundError(f"Thyroid file not found: {_TH_PATH}")

    df = pd.read_csv(_TH_PATH)

    # Binary target.
    df[TH_TARGET] = (df["target"].astype(str) != "-").astype(int)

    # Encode binary categoricals.
    sex_map = {"F": 0, "M": 1}
    df["sex"] = df["sex"].map(sex_map).astype(float)
    df["on_thyroxine"] = df["on_thyroxine"].map({"f": 0, "t": 1}).astype(float)

    # Median-impute the 5 thyroid panel measurements + age.
    numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # sex may still be NaN if there are unknown codes — fall back to mode.
    if df["sex"].isna().any():
        df["sex"] = df["sex"].fillna(df["sex"].mode().iloc[0] if not df["sex"].mode().empty else 0)
    if df["on_thyroxine"].isna().any():
        df["on_thyroxine"] = df["on_thyroxine"].fillna(0)

    # Age sanity: ThyroidDF has stray values up to 65000+ (data-entry errors).
    df["age"] = df["age"].clip(0, 110)

    df = df[TH_FEATURES + [TH_TARGET]].dropna().reset_index(drop=True)
    log.info(
        "  Thyroid: %d samples, %d features, positive rate=%.1f%%",
        len(df), len(TH_FEATURES), df[TH_TARGET].mean() * 100,
    )
    return df
