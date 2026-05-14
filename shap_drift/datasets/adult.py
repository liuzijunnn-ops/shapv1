"""Adult Income dataset loader."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

log = logging.getLogger(__name__)

ADULT_FEATURES: List[str] = [
    "age", "education_num", "capital_gain", "capital_loss",
    "hours_per_week", "fnlwgt",
]
ADULT_TARGET: str = "high_income"


def load_adult() -> pd.DataFrame:
    """Load UCI Adult Income dataset from local file (~48k samples, 6 numeric features)."""
    adult_path = Path("dataset/adult/adult.data")
    if not adult_path.exists():
        alt = Path("dataset/adult (1)/adult.data")
        if alt.exists():
            adult_path = alt

    if adult_path.exists():
        col_names = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "class",
        ]
        df = pd.read_csv(adult_path, header=None, names=col_names,
                         skipinitialspace=True, na_values="?")
        df = df.dropna().reset_index(drop=True)
        select = ["age", "education_num", "capital_gain", "capital_loss",
                  "hours_per_week", "fnlwgt", "class"]
        df = df[select].copy()
        df[ADULT_TARGET] = df["class"].str.strip().str.rstrip(".").apply(
            lambda x: 1 if x == ">50K" else 0
        )
        df = df.drop(columns=["class"])
        for col in ADULT_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        return df

    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
        df = data.frame.copy()
        rename = {
            "age": "age", "education-num": "education_num",
            "capital-gain": "capital_gain", "capital-loss": "capital_loss",
            "hours-per-week": "hours_per_week", "fnlwgt": "fnlwgt",
        }
        df = df[list(rename.keys()) + ["class"]].copy()
        df = df.rename(columns=rename)
        df[ADULT_TARGET] = (df["class"] == ">50K").astype(int)
        df = df.drop(columns=["class"])
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception:
        log.warning("Adult dataset download failed, using make_classification fallback")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=30000, n_features=6, n_informative=4,
                                    n_redundant=1, random_state=42)
        df = pd.DataFrame(X, columns=ADULT_FEATURES)
        df[ADULT_TARGET] = y
        return df
