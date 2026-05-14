"""Ablation 2: Prior weight strategy (Correlation vs Uniform vs Random vs MI)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from shap_drift.correction.sdc import (
    _EPS, _RATIO_LO, _RATIO_HI, _compute_synth_prior, _ensure_2d, sdc_corr,
)
from shap_drift.models.explainers import eval_rho

_EPS_PRIOR = _EPS


def _compute_uniform_prior(n_features: int) -> np.ndarray:
    return np.ones(n_features) / max(n_features, 1)


def _compute_random_prior(n_features: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.dirichlet(np.ones(n_features))


def _compute_mi_prior(
    df_real: pd.DataFrame, features: List[str], target: str, n_bins: int = 10,
) -> np.ndarray:
    """Mutual-information prior with robust binning.

    ``pd.qcut`` can return NaN codes for ties beyond the requested quantiles;
    those rows are dropped per-feature before passing to ``mutual_info_score``.
    """
    mi_vals = []
    target_vals = pd.to_numeric(df_real[target], errors="coerce")
    for feat in features:
        try:
            x = pd.to_numeric(df_real[feat], errors="coerce")
            x_binned = pd.qcut(x, q=n_bins, duplicates="drop")
            codes = x_binned.cat.codes
            mask = (codes >= 0) & target_vals.notna()
            if mask.sum() < 2:
                mi_vals.append(0.0)
                continue
            mi = mutual_info_score(codes[mask], target_vals[mask])
            mi_vals.append(float(mi) if np.isfinite(mi) else 0.0)
        except (ValueError, TypeError):
            mi_vals.append(0.0)
    mi_arr = np.array(mi_vals, dtype=np.float64)
    total = float(mi_arr.sum())
    if total < _EPS_PRIOR:
        return np.ones_like(mi_arr) / max(len(mi_arr), 1)
    return mi_arr / total


def ablate_prior_weight(
    shap_synth: np.ndarray,
    shap_real: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Ablation: compare prior weight strategies (correlation / uniform / random / MI)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    shap_synth_2d = _ensure_2d(shap_synth)
    orig_rho = eval_rho(shap_real, shap_synth_2d)
    results: Dict[str, Any] = {"original_rho": orig_rho}

    n_features = len(features)
    synth_prior = _compute_synth_prior(shap_synth_2d)

    # Correlation prior (default) — single SDC-Corr call, single eval_rho.
    sv_corr, _ = sdc_corr(shap_synth_2d, df_real, features, target, alpha=alpha)
    corr_rho = eval_rho(shap_real, sv_corr)
    results["correlation_prior"] = {"rho": corr_rho, "delta_rho": corr_rho - orig_rho}

    for name, prior_fn in [
        ("uniform_prior", lambda: _compute_uniform_prior(n_features)),
        ("random_prior", lambda: _compute_random_prior(n_features)),
        ("mi_prior", lambda: _compute_mi_prior(df_real, features, target)),
    ]:
        prior = prior_fn()
        ratio = np.clip(prior / (synth_prior + _EPS_PRIOR), _RATIO_LO, _RATIO_HI)
        scale = alpha * ratio + (1.0 - alpha)
        sv = shap_synth_2d * scale[np.newaxis, :]
        rho = eval_rho(shap_real, sv)
        results[name] = {"rho": rho, "delta_rho": rho - orig_rho}

    return results
