"""Credit Card Fraud dataset loader.

Feature selection used to run at *import time*, which had three downsides:

  1. Importing :pymod:`shap_drift.datasets` would trigger a full XGBoost
     training pass + a multi-hundred-MB CSV read, breaking ``pytest -s``
     collection and any non-data workflow (e.g. linting, doc builds).
  2. The selector could fail when the dataset was missing, but the wrapping
     ``try/except`` swallowed the error silently and substituted a fallback
     feature list — making the failure invisible.
  3. The selection used a hard-coded seed and was effectively a form of
     in-sample feature snooping on the entire dataset.

We now defer feature selection until the dataset is actually loaded, cache
the result on the first call, and expose ``CCF_FEATURES`` as a property-like
attribute via :func:`get_ccf_features` (kept as a module attribute for
backward compatibility).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

CCF_TARGET: str = "Class"
_CCF_PATH = Path("dataset/CreditCardFraud/creditcard.csv")
_DEFAULT_CCF_FEATURES: List[str] = [f"V{i}" for i in range(1, 11)] + ["Amount"]

# Mutable cache populated on first call to ``load_creditcard_fraud``.  Kept as
# a list (not a tuple) so legacy code that may mutate it does not crash —
# though such code is discouraged.
CCF_FEATURES: List[str] = list(_DEFAULT_CCF_FEATURES)

_FEATURE_CACHE: Optional[List[str]] = None


def _select_ccf_features(random_state: int = 42) -> List[str]:
    """Run XGBoost feature selection on CreditCardFraud to determine top V features.

    Returns the 10 most-important V columns (sorted by their numeric index for
    deterministic ordering) plus ``Amount``.  Falls back to ``V1..V10``+
    ``Amount`` if XGBoost is unavailable or the dataset is missing.
    """
    if not _CCF_PATH.exists():
        log.warning(
            "CreditCardFraud dataset not found at %s — using default features",
            _CCF_PATH,
        )
        return list(_DEFAULT_CCF_FEATURES)

    try:
        import xgboost as xgb
        df_raw = pd.read_csv(_CCF_PATH)
        all_v_features = [f"V{i}" for i in range(1, 29)]
        feature_cols_all = all_v_features + ["Amount"]
        X_all = df_raw[feature_cols_all].values
        y_all = df_raw["Class"].values

        if len(df_raw) > 50_000:
            idx, _, _, _ = train_test_split(
                np.arange(len(df_raw)), y_all,
                test_size=0.8, random_state=random_state, stratify=y_all,
            )
            X_sel, y_sel = X_all[idx], y_all[idx]
        else:
            X_sel, y_sel = X_all, y_all

        fs_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=random_state,
        )
        fs_model.fit(X_sel, y_sel)
        importances = fs_model.feature_importances_
        v_importance = sorted(
            zip(all_v_features, importances), key=lambda x: x[1], reverse=True,
        )
        top_v = [name for name, _ in v_importance[:10]]
        top_v.sort(key=lambda x: int(x[1:]))
        return top_v + ["Amount"]
    except Exception as exc:
        log.warning("CCF feature selection failed (%s) — using defaults", exc)
        return list(_DEFAULT_CCF_FEATURES)


def get_ccf_features(refresh: bool = False) -> List[str]:
    """Return the CCF feature list, computing it lazily on first use.

    Set ``refresh=True`` to force re-computation (e.g. when the dataset
    changes on disk).
    """
    global _FEATURE_CACHE, CCF_FEATURES
    if _FEATURE_CACHE is None or refresh:
        _FEATURE_CACHE = _select_ccf_features()
        CCF_FEATURES = list(_FEATURE_CACHE)
    return list(_FEATURE_CACHE)


def load_creditcard_fraud(random_state: int = 42) -> pd.DataFrame:
    """Load Credit Card Fraud dataset (284K rows, 0.17% fraud rate).

    Pipeline:
      1. Lazily run XGBoost feature selection (top-10 V features + ``Amount``).
      2. Downsample the majority class to a 5:1 ratio against the minority
         class for training efficiency.

    Args:
        random_state: controls the down-sampling RNG.
    """
    from sklearn.utils import resample

    get_ccf_features()  # populate cache & CCF_FEATURES (kept for backward compat)
    df_raw = pd.read_csv(_CCF_PATH)
    df = df_raw.drop(columns=["Time"]).copy()

    fraud_idx = df[df["Class"] == 1].index
    normal_idx = df[df["Class"] == 0].index
    n_normal_target = min(len(fraud_idx) * 5, len(normal_idx))
    if n_normal_target < len(normal_idx):
        normal_downsampled = resample(
            df.loc[normal_idx], replace=False,
            n_samples=n_normal_target, random_state=int(random_state),
        )
        df = pd.concat([df.loc[fraud_idx], normal_downsampled]).reset_index(drop=True)

    return df
