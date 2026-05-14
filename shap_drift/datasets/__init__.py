"""Dataset registry and loaders."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable configuration for a single benchmark dataset."""
    name: str
    loader: Callable[[], pd.DataFrame]
    features: Union[List[str], Callable[[], List[str]]]
    target: str

    def get_features(self) -> List[str]:
        """Return resolved feature list (lazy for callable)."""
        if callable(self.features):
            return list(self.features())
        return list(self.features)


def prepare_dataset(
    df: pd.DataFrame, features: List[str], target: str
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Clean, validate, and split a DataFrame into (df_clean, X, y).

    Raises:
        KeyError: if ``target`` or any ``features`` column is missing.
        ValueError: if the cleaned DataFrame is empty or the target column
            contains a single class only.
    """
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=list(features) + [target]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("prepare_dataset: no rows survived NA filtering")

    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(int)

    if len(np.unique(y)) < 2:
        # Not fatal but downstream stratified splits will fail — surface a
        # clear error here rather than waiting for the cryptic sklearn one.
        raise ValueError(
            "prepare_dataset: target column has only one unique class after cleaning"
        )

    return df, X, y


# -- lazy imports to avoid circular deps --
from shap_drift.datasets.credit_card_fraud import load_creditcard_fraud, get_ccf_features, CCF_TARGET
from shap_drift.datasets.heart_disease import load_heart_disease, HD_FEATURES, HD_TARGET
from shap_drift.datasets.injury_1798 import load_injury_1798, INJ_FEATURES, INJ_TARGET
from shap_drift.datasets.california_housing import load_california_housing, CH_FEATURES, CH_TARGET
from shap_drift.datasets.german_credit import load_german_credit, GC_FEATURES, GC_TARGET
from shap_drift.datasets.adult import load_adult, ADULT_FEATURES, ADULT_TARGET
# v0.3 additions — broader coverage along 4 axes:
#   * Diabetes  — clinical, small N, near-balanced (P0 case-study target)
#   * CovType   — multi-class binarized, large N, mostly continuous
#   * Higgs     — high-dim (28 features), near-balanced, physics
#   * Thyroid   — heavily imbalanced (~26% positive), mixed types
from shap_drift.datasets.diabetes import load_diabetes, DB_FEATURES, DB_TARGET
from shap_drift.datasets.covtype import load_covtype, CT_FEATURES, CT_TARGET
from shap_drift.datasets.higgs import load_higgs, HG_FEATURES, HG_TARGET
from shap_drift.datasets.thyroid import load_thyroid, TH_FEATURES, TH_TARGET


DATASETS: Dict[str, DatasetConfig] = {
    "CreditCardFraud": DatasetConfig(
        name="CreditCardFraud",
        loader=load_creditcard_fraud,
        features=get_ccf_features,
        target=CCF_TARGET,
    ),
    "HeartDisease": DatasetConfig(
        name="HeartDisease",
        loader=load_heart_disease,
        features=HD_FEATURES,
        target=HD_TARGET,
    ),
    "Injury1798": DatasetConfig(
        name="Injury1798",
        loader=load_injury_1798,
        features=INJ_FEATURES,
        target=INJ_TARGET,
    ),
    "CaliforniaHousing": DatasetConfig(
        name="CaliforniaHousing",
        loader=load_california_housing,
        features=CH_FEATURES,
        target=CH_TARGET,
    ),
    "GermanCredit": DatasetConfig(
        name="GermanCredit",
        loader=load_german_credit,
        features=GC_FEATURES,
        target=GC_TARGET,
    ),
    "Adult": DatasetConfig(
        name="Adult",
        loader=load_adult,
        features=ADULT_FEATURES,
        target=ADULT_TARGET,
    ),
    "Diabetes": DatasetConfig(
        name="Diabetes",
        loader=load_diabetes,
        features=DB_FEATURES,
        target=DB_TARGET,
    ),
    "CovType": DatasetConfig(
        name="CovType",
        loader=load_covtype,
        features=CT_FEATURES,
        target=CT_TARGET,
    ),
    "Higgs": DatasetConfig(
        name="Higgs",
        loader=load_higgs,
        features=HG_FEATURES,
        target=HG_TARGET,
    ),
    "Thyroid": DatasetConfig(
        name="Thyroid",
        loader=load_thyroid,
        features=TH_FEATURES,
        target=TH_TARGET,
    ),
}

DATASET_ORDER: List[str] = list(DATASETS.keys())

DS_LABELS: Dict[str, str] = {
    "CreditCardFraud": "Credit Card Fraud",
    "HeartDisease": "Heart Disease",
    "Injury1798": "Injury (1798)",
    "CaliforniaHousing": "California Housing",
    "GermanCredit": "German Credit",
    "Adult": "Adult Income",
    "Diabetes": "Pima Diabetes",
    "CovType": "Forest Cover Type",
    "Higgs": "Higgs Boson",
    "Thyroid": "Thyroid Disease",
}

DS_COLORS: Dict[str, str] = {
    "CreditCardFraud": "#D32F2F",
    "HeartDisease": "#1976D2",
    "Injury1798": "#E53935",
    "CaliforniaHousing": "#43A047",
    "GermanCredit": "#FB8C00",
    "Adult": "#8E24AA",
    "Diabetes": "#00ACC1",
    "CovType": "#558B2F",
    "Higgs": "#5E35B1",
    "Thyroid": "#EF6C00",
}
