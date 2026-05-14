"""Pima Indians Diabetes dataset loader (clinical binary classification)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DB_FEATURES: List[str] = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
DB_TARGET: str = "Outcome"

# Columns where 0 is a sentinel for "missing measurement" (per UCI docs).
_ZERO_AS_NA: List[str] = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_diabetes() -> pd.DataFrame:
    """Load Pima Indians Diabetes (768 samples, 8 numeric features, binary).

    Cleaning:
      * Zero values in physiological columns are treated as missing and
        imputed with the column median (a well-known UCI quirk).
      * Returns ``DB_FEATURES + [DB_TARGET]`` only.
    """
    fp = Path("dataset/Diabetes /diabetes.csv")
    if not fp.exists():
        # Fallback to a possibly trimmed folder name without trailing space.
        fp = Path("dataset/Diabetes/diabetes.csv")
    df = pd.read_csv(fp)

    for col in _ZERO_AS_NA:
        df.loc[df[col] == 0, col] = np.nan
        df[col] = df[col].fillna(df[col].median())

    for col in DB_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=DB_FEATURES + [DB_TARGET]).reset_index(drop=True)
    df[DB_TARGET] = df[DB_TARGET].astype(int).clip(0, 1)
    return df[DB_FEATURES + [DB_TARGET]].copy()
