"""California Housing dataset loader (regression → binary classification)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

CH_FEATURES: List[str] = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude",
]
CH_TARGET: str = "HighPrice"


def load_california_housing() -> pd.DataFrame:
    """Load California Housing as binary classification (20640 samples).

    Target: median_house_value > median → HighPrice.
    """
    ch_path = Path("dataset/CaliforniaHousing/california_housing.csv")
    if ch_path.exists():
        df = pd.read_csv(ch_path)
        df = df.rename(columns={c: CH_FEATURES[i] for i, c in enumerate(df.columns[:8]) if i < len(CH_FEATURES)})
        df = df[CH_FEATURES + ["MedHouseVal"]].copy()
        df[CH_TARGET] = (df["MedHouseVal"] >= df["MedHouseVal"].median()).astype(int)
        df = df.drop(columns=["MedHouseVal"])
        return df
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=CH_FEATURES)
    df[CH_TARGET] = (data.target >= np.median(data.target)).astype(int)
    return df
