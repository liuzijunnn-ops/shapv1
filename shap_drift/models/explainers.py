"""SHAP computation and evaluation utilities."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shap
from scipy import stats
from sklearn.cluster import KMeans

from shap_drift.models import is_tree_model

log = logging.getLogger(__name__)

# Maximum plausible number of classes for SHAP outputs.  Used to disambiguate
# the class axis from the samples / features axes when SHAP returns a 3-D
# array on edge-case datasets (e.g., only 2 test samples or 2 features).
_MAX_CLASSES = 16


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman ρ with NaN / zero-variance guards.

    Returns (rho, p_value).  Falls back to (0.0, 1.0) when either input has
    fewer than 2 valid observations or zero variance — preventing NaN
    propagation through downstream aggregations.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return 0.0, 1.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0, 1.0
    rho, p_val = stats.spearmanr(a, b)
    rho = float(rho) if np.isfinite(rho) else 0.0
    p_val = float(p_val) if np.isfinite(p_val) else 1.0
    return rho, p_val


def extract_binary_shap(shap_values: Any, n_features: Optional[int] = None) -> np.ndarray:
    """Extract SHAP values for binary classification (auto-handle 3D / list format).

    Robust to all common SHAP layouts:
      * list of length-2 arrays   →  positive-class array
      * 3-D `(samples, features, classes)`  (SHAP ≥ 0.40 convention)
      * 3-D `(classes, samples, features)`  (legacy SHAP < 0.40 convention)
      * already 2-D `(samples, features)`   →  returned as-is

    The class axis is disambiguated by (a) preferring the last axis (modern
    convention) and (b) requiring the candidate axis to have ≤ `_MAX_CLASSES`
    entries — so it cannot be confused with the samples axis when there
    happen to be only 2 test samples or 2 features.

    Args:
        shap_values: raw SHAP output (list or ndarray)
        n_features: optional, used for a final sanity check on the returned shape

    Returns:
        2-D ndarray `(n_samples, n_features)`.
    """
    if isinstance(shap_values, list):
        # Multi-output list — for binary classifiers use the positive class.
        sv = np.array(shap_values[-1], dtype=np.float64)
    else:
        sv = np.array(shap_values, dtype=np.float64)

    if sv.ndim == 2:
        return sv

    if sv.ndim == 3:
        # Modern convention: (samples, features, classes).  The class axis is
        # the LAST dim if it is small enough to be plausibly a class axis.
        if sv.shape[-1] <= _MAX_CLASSES and (n_features is None or sv.shape[1] == n_features):
            return sv[..., -1]
        # Legacy convention: (classes, samples, features).
        if sv.shape[0] <= _MAX_CLASSES and (n_features is None or sv.shape[-1] == n_features):
            return sv[-1]
        # Last-resort fallback — take the smallest plausible class axis.
        for axis in range(sv.ndim):
            if sv.shape[axis] <= _MAX_CLASSES:
                return np.take(sv, -1, axis=axis)

    raise ValueError(
        f"Could not extract 2-D SHAP values from array of shape {sv.shape}"
    )


def extract_binary_interaction(interaction_values: Any) -> np.ndarray:
    """Extract SHAP interaction values for binary classification.

    Handles:
      * list of length-2 arrays   →  positive-class array
      * 4-D `(samples, features, features, classes)`  (modern)
      * 4-D `(classes, samples, features, features)`  (legacy)
      * already 3-D                                   →  returned as-is
    """
    if isinstance(interaction_values, list):
        iv = np.array(interaction_values[-1], dtype=np.float64)
    else:
        iv = np.array(interaction_values, dtype=np.float64)

    if iv.ndim == 3:
        return iv
    if iv.ndim == 4:
        if iv.shape[-1] <= _MAX_CLASSES:
            return iv[..., -1]
        if iv.shape[0] <= _MAX_CLASSES:
            return iv[-1]
    raise ValueError(
        f"Could not extract 3-D SHAP interaction values from array of shape {iv.shape}"
    )


def _summarize_background(
    X: np.ndarray, n_samples: int = 100, random_state: int = 42,
) -> np.ndarray:
    """Summarize background data using k-means for KernelExplainer.

    Args:
        X: training data to summarize.
        n_samples: requested number of background centroids.
        random_state: seed for KMeans (controls reproducibility).
    """
    if len(X) <= n_samples:
        return X
    # Guard: n_clusters must be at least 2 and at most n_samples in X.
    k = max(2, min(int(n_samples), len(X)))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=3)
    kmeans.fit(X)
    return kmeans.cluster_centers_


def compute_shap(
    model_class: type,
    model_kwargs: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_name: str = "",
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Fit a model and compute SHAP values on X_test.

    Automatically selects TreeExplainer or KernelExplainer based on model type.

    Args:
        random_state: seed for kernel-explainer background subsampling and
            test-subsampling.  Defaults to ``model_kwargs.get("random_state", 42)``
            so per-seed runs remain reproducible.
    """
    if random_state is None:
        random_state = int(model_kwargs.get("random_state", 42))

    clf = model_class(**model_kwargs)
    clf.fit(X_train, y_train)
    n_features = X_train.shape[1] if X_train.ndim == 2 else None

    if is_tree_model(model_name):
        sv = shap.TreeExplainer(clf).shap_values(X_test)
        sv = extract_binary_shap(sv, n_features=n_features)
    else:
        n_train = len(X_train)
        n_test = len(X_test)

        n_background = min(50, max(20, n_train // 2))
        background = _summarize_background(
            X_train, n_samples=n_background, random_state=random_state,
        )
        explainer = shap.KernelExplainer(clf.predict_proba, background)

        max_explain = min(n_test, 200)
        if n_test <= max_explain:
            X_explain = X_test
        else:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n_test, size=max_explain, replace=False)
            X_explain = X_test[idx]

        nsamples = max(100, min(500, n_train))
        sv = explainer.shap_values(X_explain, nsamples=nsamples)
        sv = extract_binary_shap(sv, n_features=n_features)

    # Clean NaN / inf (KernelExplainer can produce them on poorly-calibrated models)
    n_nan = int(np.isnan(sv).sum())
    n_inf = int(np.isinf(sv).sum())
    if n_nan > 0 or n_inf > 0:
        log.warning(
            "  compute_shap(%s): %d NaN, %d inf in SHAP values — replacing with 0",
            model_name, n_nan, n_inf,
        )
        sv = np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0)

    global_imp = np.mean(np.abs(sv), axis=0).flatten()
    return sv, global_imp, clf


def eval_rho(shap_real: np.ndarray, shap_corrected: np.ndarray) -> float:
    """Compute Spearman ρ between real and corrected global SHAP importances.

    Returns 0.0 on degenerate inputs (NaN / constant / mismatched length).
    """
    gi_real = np.nan_to_num(
        np.mean(np.abs(shap_real), axis=0).flatten(),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    gi_corr = np.nan_to_num(
        np.mean(np.abs(shap_corrected), axis=0).flatten(),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    rho, _ = _safe_spearman(gi_real, gi_corr)
    return rho


def eval_shap(
    shap_real: np.ndarray,
    global_real: np.ndarray,
    shap_corrected: np.ndarray,
    features: List[str],
) -> Dict[str, Any]:
    """Compute Spearman ρ, sign agreement, and per-sample cosine similarity."""
    global_corrected = np.mean(np.abs(shap_corrected), axis=0).flatten()
    rho, p_val = _safe_spearman(global_real, global_corrected)
    sign_agree = float(np.mean(
        np.sign(np.mean(shap_real, axis=0)) == np.sign(np.mean(shap_corrected, axis=0)),
    ))

    n = min(shap_real.shape[0], shap_corrected.shape[0])
    if n == 0:
        return {"rho": rho, "p": p_val, "sign_agree": sign_agree, "cos_sim": 0.0}

    vr = shap_real[:n].reshape(n, -1)
    vc = shap_corrected[:n].reshape(n, -1)
    norm_r = np.linalg.norm(vr, axis=1)
    norm_c = np.linalg.norm(vc, axis=1)
    valid = (norm_r > 1e-10) & (norm_c > 1e-10)
    cos_sims = np.full(n, np.nan)
    cos_sims[valid] = np.sum(vr[valid] * vc[valid], axis=1) / (norm_r[valid] * norm_c[valid])
    cos_mean = float(np.nanmean(cos_sims)) if valid.any() else 0.0

    return {
        "rho": rho,
        "p": p_val,
        "sign_agree": sign_agree,
        "cos_sim": cos_mean,
    }
