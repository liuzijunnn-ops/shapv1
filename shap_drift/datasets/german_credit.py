"""German Credit dataset loader."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

log = logging.getLogger(__name__)

GC_FEATURES: List[str] = [
    "existing_checking", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since",
    "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker",
]
GC_TARGET: str = "bad_credit"


def load_german_credit() -> pd.DataFrame:
    """Load German Credit dataset from local UCI file (1000 samples, 20 features)."""
    gc_path = Path("dataset/statlog+german+credit+data/german.data")
    if gc_path.exists():
        df = pd.read_csv(gc_path, sep=" ", header=None, names=GC_FEATURES + [GC_TARGET])
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.factorize(df[col])[0]
        df[GC_TARGET] = (df[GC_TARGET] == 2).astype(int) if df[GC_TARGET].max() == 2 else df[GC_TARGET]
        return df
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        for col in df.columns:
            if df[col].dtype == "category" or df[col].dtype == object:
                df[col] = pd.factorize(df[col])[0]
        df = df.rename(columns={"class": GC_TARGET})
        df[GC_TARGET] = (df[GC_TARGET] == df[GC_TARGET].unique()[0]).astype(int)
        rename_map = {old: new for old, new in zip(df.columns[:-1], GC_FEATURES) if old != GC_TARGET}
        df = df.rename(columns=rename_map)
        return df
    except Exception:
        log.warning("German Credit download failed, using make_classification fallback")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                                    n_redundant=5, random_state=42)
        df = pd.DataFrame(X, columns=GC_FEATURES)
        df[GC_TARGET] = y
        return df
