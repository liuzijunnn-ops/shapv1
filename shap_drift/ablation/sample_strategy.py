"""Ablation 3: Sample selection strategy (Random vs Density vs Boundary)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from shap_drift.correction.fsc import fsc_corr
from shap_drift.models.explainers import eval_rho


def _density_sampling(X: np.ndarray, n_samples: int) -> np.ndarray:
    """Select the n_samples observations closest to the data centroid."""
    n_samples = min(int(n_samples), len(X))
    centroid = X.mean(axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    return np.argsort(distances)[:n_samples]


def _boundary_sampling(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    model_class: type,
    model_kwargs: dict,
) -> np.ndarray:
    """Select the n_samples lowest-confidence observations under a fitted model."""
    n_samples = min(int(n_samples), len(X))
    model = model_class(**model_kwargs)
    model.fit(X, y)
    proba = model.predict_proba(X)
    confidence = np.max(proba, axis=1)
    return np.argsort(confidence)[:n_samples]


def _build_subset_df(
    X: np.ndarray, y: np.ndarray, idx: np.ndarray,
    features: List[str], target: str,
) -> pd.DataFrame:
    df = pd.DataFrame(X[idx], columns=list(features))
    df[target] = y[idx]
    return df


def ablate_sample_strategy(
    X_real: np.ndarray,
    y_real: np.ndarray,
    shap_synth: np.ndarray,
    shap_real: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
    model_class: type,
    model_kwargs: dict,
    calibration_fraction: float = 0.2,
    alpha: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Ablation: compare calibration sample selection strategies."""
    if not 0.0 < calibration_fraction <= 1.0:
        raise ValueError(
            f"calibration_fraction must be in (0, 1], got {calibration_fraction}"
        )
    orig_rho = eval_rho(shap_real, shap_synth)
    results: Dict[str, Any] = {"original_rho": orig_rho}
    n_cal = max(5, int(len(X_real) * calibration_fraction))
    n_cal = min(n_cal, len(X_real))

    # Random sampling
    rng = np.random.RandomState(int(random_state))
    random_idx = rng.choice(len(X_real), n_cal, replace=False)
    df_random = _build_subset_df(X_real, y_real, random_idx, features, target)
    sv_random = fsc_corr(shap_synth, df_random, features, target, alpha=alpha)
    rho_random = eval_rho(shap_real, sv_random)
    results["random_sampling"] = {
        "rho": rho_random, "delta_rho": rho_random - orig_rho, "n_cal": n_cal,
    }

    # Density sampling
    density_idx = _density_sampling(X_real, n_cal)
    df_density = _build_subset_df(X_real, y_real, density_idx, features, target)
    sv_density = fsc_corr(shap_synth, df_density, features, target, alpha=alpha)
    rho_density = eval_rho(shap_real, sv_density)
    results["density_sampling"] = {
        "rho": rho_density, "delta_rho": rho_density - orig_rho, "n_cal": n_cal,
    }

    # Boundary sampling
    try:
        boundary_idx = _boundary_sampling(
            X_real, y_real, n_cal, model_class, model_kwargs,
        )
        df_boundary = _build_subset_df(X_real, y_real, boundary_idx, features, target)
        sv_boundary = fsc_corr(shap_synth, df_boundary, features, target, alpha=alpha)
        rho_boundary = eval_rho(shap_real, sv_boundary)
        results["boundary_sampling"] = {
            "rho": rho_boundary, "delta_rho": rho_boundary - orig_rho, "n_cal": n_cal,
        }
    except Exception as exc:
        results["boundary_sampling"] = {
            "rho": orig_rho, "delta_rho": 0.0, "n_cal": n_cal, "error": str(exc),
        }

    return results
