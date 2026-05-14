"""Heart Disease dataset loader."""
from __future__ import annotations

from typing import List

import pandas as pd

HD_FEATURES: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
HD_TARGET: str = "heart_disease"


def load_heart_disease() -> pd.DataFrame:
    """Load Heart Disease dataset from UCI (Cleveland, 303 samples, 13 features).

    Target: num > 0 → 1 (heart disease), num = 0 → 0 (healthy).
    """
    feature_names = HD_FEATURES + [HD_TARGET]
    df = pd.read_csv(
        "dataset/heart+disease/processed.cleveland.data",
        header=None,
        names=feature_names,
        na_values="?",
    )
    for col in feature_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df[HD_TARGET] = (df[HD_TARGET] > 0).astype(int)
    return df
