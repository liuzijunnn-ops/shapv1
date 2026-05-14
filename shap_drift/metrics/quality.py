"""Synthetic data quality evaluation metrics."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from shap_drift.config import safe_corr

_EPS = 1e-10


def evaluate_quality(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate synthetic-data quality: marginal, correlation, downstream utility.

    All blocks are individually wrapped so a failure in one (e.g. an empty
    upper-triangle for n_features = 1) does not invalidate the rest.
    """
    metrics: Dict[str, Any] = {}

    # --- Marginal distribution comparison ---
    # ``dtype in (...)`` previously missed np.int32, np.uint8, np.float16, …
    # ``pd.api.types.is_numeric_dtype`` is the canonical test.
    ks_vals, w1_vals = [], []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_real[col]):
            continue
        real_col = df_real[col].dropna().to_numpy()
        synth_col = df_synth[col].dropna().to_numpy()
        if real_col.size == 0 or synth_col.size == 0:
            continue
        ks_stat, _ = stats.ks_2samp(real_col, synth_col)
        w1_dist = stats.wasserstein_distance(real_col, synth_col)
        real_std = float(np.std(real_col))
        ks_vals.append(float(ks_stat) if np.isfinite(ks_stat) else 0.0)
        if np.isfinite(w1_dist):
            w1_vals.append(float(w1_dist / (real_std + _EPS)))
    metrics["marginal"] = {
        "mean_ks": float(np.mean(ks_vals)) if ks_vals else 0.0,
        "mean_w1_norm": float(np.mean(w1_vals)) if w1_vals else 0.0,
        "n_features_evaluated": len(ks_vals),
    }

    # --- Correlation structure comparison ---
    real_corr = safe_corr(df_real, feature_cols)
    synth_corr = safe_corr(df_synth, feature_cols)
    diff = (real_corr - synth_corr).abs().values
    if diff.shape[0] >= 2:
        triu_i, triu_j = np.triu_indices_from(diff, k=1)
        triu_vals = diff[triu_i, triu_j]
        metrics["correlation"] = {
            "mean_abs_diff": float(triu_vals.mean()),
            "max_abs_diff": float(triu_vals.max()),
            "n_pairs": int(triu_vals.size),
        }
    else:
        # 0- or 1-feature case: no off-diagonal pair → metric undefined.
        metrics["correlation"] = {
            "mean_abs_diff": 0.0, "max_abs_diff": 0.0, "n_pairs": 0,
        }

    # --- Downstream utility (TSTR) ---
    try:
        import xgboost as xgb
        X_synth = df_synth[feature_cols].values.astype(np.float32)
        y_synth = pd.to_numeric(
            df_synth[target_col], errors="coerce"
        ).fillna(0).astype(int).clip(0, 1).to_numpy()
        X_real = df_real[feature_cols].values.astype(np.float32)
        y_real = pd.to_numeric(
            df_real[target_col], errors="coerce"
        ).fillna(0).astype(int).clip(0, 1).to_numpy()

        if len(np.unique(y_synth)) < 2:
            metrics["downstream"] = {"auroc": 0.0, "error": "single_class"}
        elif len(np.unique(y_real)) < 2:
            metrics["downstream"] = {"auroc": 0.0, "error": "real_single_class"}
        else:
            clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=int(random_state), eval_metric="logloss",
            )
            clf.fit(X_synth, y_synth)
            y_pred = clf.predict_proba(X_real)[:, 1]
            metrics["downstream"] = {"auroc": float(roc_auc_score(y_real, y_pred))}
    except Exception as exc:
        metrics["downstream"] = {"auroc": 0.0, "error": str(exc)}

    return metrics
