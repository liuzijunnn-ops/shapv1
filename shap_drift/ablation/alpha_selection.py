"""Ablation 1: Alpha selection strategy (Fixed vs CV vs Adaptive)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from scipy import stats

from shap_drift.correction.baselines import _safe_stratified_split, _kernel_shap
from shap_drift.correction.sdc import _compute_ratio, _ensure_2d, sdc_corr
from shap_drift.models import is_tree_model
from shap_drift.models.explainers import _safe_spearman, eval_rho, extract_binary_shap


def sdc_corr_adaptive(
    shap_synth: np.ndarray,
    shap_real: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
    scale: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """A1: Per-feature adaptive alpha based on local KS drift."""
    shap_synth = _ensure_2d(shap_synth)
    shap_real = _ensure_2d(shap_real)

    n_features = shap_synth.shape[1]
    local_ks = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        a = shap_real[:, j].ravel()
        b = shap_synth[:, j].ravel()
        if a.size == 0 or b.size == 0:
            continue
        ks, _ = stats.ks_2samp(a, b)
        local_ks[j] = float(ks) if np.isfinite(ks) else 0.0

    alpha_per_feature = np.clip(local_ks * scale, 0.1, 1.0)
    ratio = _compute_ratio(shap_synth, df_real, features, target)
    scale_factor = alpha_per_feature * ratio + (1.0 - alpha_per_feature)
    return shap_synth * scale_factor[np.newaxis, :], alpha_per_feature


def sdc_corr_cv(
    shap_synth: np.ndarray,
    shap_real: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
    X_real: np.ndarray,
    y_real: np.ndarray,
    model_class: type,
    model_kwargs: Dict[str, Any],
    alpha_grid: Optional[List[float]] = None,
    model_name: str = "",
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """A4: SDC-Corr-CV — cross-validated alpha selection."""
    if alpha_grid is None:
        alpha_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    X_tr, X_val, y_tr, y_val = _safe_stratified_split(
        X_real, y_real, test_size=0.2, random_state=random_state,
    )

    real_model = model_class(**model_kwargs)
    real_model.fit(X_tr, y_tr)

    n_features = X_real.shape[1] if X_real.ndim == 2 else None
    if is_tree_model(model_name):
        sv_val_real = shap.TreeExplainer(real_model).shap_values(X_val)
        sv_val_real = extract_binary_shap(sv_val_real, n_features=n_features)
    else:
        sv_val_real = _kernel_shap(
            real_model, X_tr, X_val,
            n_features=n_features, random_state=random_state,
        )

    gi_val_real = np.mean(np.abs(sv_val_real), axis=0).flatten()

    best_alpha, best_rho = 0.5, -2.0
    alpha_results: Dict[str, float] = {}

    for alpha in alpha_grid:
        sv_corrected, _ = sdc_corr(shap_synth, df_real, features, target, alpha=alpha)
        gi_corrected = np.mean(np.abs(sv_corrected), axis=0).flatten()
        if len(gi_corrected) != len(gi_val_real):
            rho = 0.0
        else:
            rho, _ = _safe_spearman(gi_corrected, gi_val_real)
        alpha_results[f"alpha={alpha}"] = float(rho)
        if rho > best_rho:
            best_rho, best_alpha = rho, alpha

    corrected, ratio_map = sdc_corr(shap_synth, df_real, features, target, alpha=best_alpha)
    info = {
        "best_alpha": float(best_alpha), "best_rho": float(best_rho),
        "alpha_results": alpha_results, "ratio": ratio_map,
    }
    return corrected, info


def ablate_alpha_selection(
    shap_synth: np.ndarray,
    shap_real: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
    X_real: np.ndarray,
    y_real: np.ndarray,
    model_class: type,
    model_kwargs: Dict[str, Any],
    model_name: str = "",
    fixed_alphas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Ablation: compare alpha selection strategies."""
    if fixed_alphas is None:
        fixed_alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    orig_rho = eval_rho(shap_real, shap_synth)
    results: Dict[str, Any] = {"original_rho": orig_rho}

    for alpha in fixed_alphas:
        sv_corr, info = sdc_corr(shap_synth, df_real, features, target, alpha=alpha)
        rho = eval_rho(shap_real, sv_corr)
        results[f"fixed_alpha_{alpha}"] = {"rho": rho, "delta_rho": rho - orig_rho, "alpha": alpha}

    sv_cv, cv_info = sdc_corr_cv(
        shap_synth, shap_real, df_real, features, target,
        X_real, y_real, model_class, model_kwargs, model_name=model_name,
    )
    cv_rho = eval_rho(shap_real, sv_cv)
    results["cv_alpha"] = {"rho": cv_rho, "delta_rho": cv_rho - orig_rho, "best_alpha": cv_info["best_alpha"]}

    sv_adapt, alpha_per_feat = sdc_corr_adaptive(shap_synth, shap_real, df_real, features, target)
    adapt_rho = eval_rho(shap_real, sv_adapt)
    results["adaptive_alpha"] = {"rho": adapt_rho, "delta_rho": adapt_rho - orig_rho, "mean_alpha": float(np.mean(alpha_per_feat))}

    return results
