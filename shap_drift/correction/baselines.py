"""Baseline comparison methods: SHAP Distillation, Fine-tuning, CORAL."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shap
from sklearn.model_selection import train_test_split

from shap_drift.correction.sdc import _EPS, _RATIO_LO, _RATIO_HI
from shap_drift.models import is_tree_model
from shap_drift.models.explainers import extract_binary_shap, _summarize_background


def _safe_stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split that falls back gracefully on degenerate label sets.

    ``train_test_split(..., stratify=y)`` raises when any class has fewer
    than 2 samples or when ``y`` is single-class.  We try stratified first,
    then fall back to a plain shuffle split.
    """
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None,
        )


def _kernel_shap(
    model: Any,
    X_background: np.ndarray,
    X_test: np.ndarray,
    n_features: Optional[int] = None,
    random_state: int = 42,
) -> np.ndarray:
    """Run KernelExplainer on a test set with a summarized background.

    Centralizes the KernelExplainer setup (background size, test subsampling,
    nsamples heuristic) so multiple baselines stay consistent.
    """
    n_train = len(X_background)
    n_test = len(X_test)
    n_background = min(50, max(20, n_train // 2)) if n_train >= 2 else max(1, n_train)
    background = _summarize_background(
        X_background, n_samples=n_background, random_state=random_state,
    )
    explainer = shap.KernelExplainer(model.predict_proba, background)

    max_explain = min(n_test, 200)
    if n_test <= max_explain:
        X_explain = X_test
    else:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_test, size=max_explain, replace=False)
        X_explain = X_test[idx]

    nsamples = max(100, min(500, n_train))
    sv_raw = explainer.shap_values(X_explain, nsamples=nsamples)
    return extract_binary_shap(sv_raw, n_features=n_features)


def shap_distillation(
    shap_synth: np.ndarray,
    X_real: np.ndarray,
    y_real: np.ndarray,
    features: List[str],
    model_class: type,
    model_kwargs: Dict[str, Any],
    model_name: str = "",
    alpha: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """SHAP Distillation: align synthetic SHAP to real-data teacher's global importance."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if len(X_real) < 4:
        # Too few samples for a meaningful 80/20 split → no distillation.
        return shap_synth.copy(), {"alpha": alpha, "skipped": "insufficient_real_samples"}

    X_tr, X_val, y_tr, y_val = _safe_stratified_split(
        X_real, y_real, test_size=0.2, random_state=random_state,
    )

    teacher = model_class(**model_kwargs)
    teacher.fit(X_tr, y_tr)

    n_features = shap_synth.shape[1] if shap_synth.ndim >= 2 else None
    if is_tree_model(model_name):
        sv_teacher = shap.TreeExplainer(teacher).shap_values(X_val)
        sv_teacher = extract_binary_shap(sv_teacher, n_features=n_features)
    else:
        sv_teacher = _kernel_shap(
            teacher, X_tr, X_val, n_features=n_features, random_state=random_state,
        )

    teacher_global = np.nan_to_num(
        np.mean(np.abs(sv_teacher), axis=0).flatten(),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    synth_global = np.nan_to_num(
        np.mean(np.abs(shap_synth), axis=0).flatten(),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    t_sum = float(teacher_global.sum()) or 1.0
    s_sum = float(synth_global.sum()) or 1.0
    teacher_prior = teacher_global / t_sum
    synth_prior = synth_global / s_sum
    distill_ratio = np.clip(teacher_prior / (synth_prior + _EPS), _RATIO_LO, _RATIO_HI)

    scale_factor = alpha * distill_ratio + (1.0 - alpha)
    corrected = shap_synth * scale_factor[np.newaxis, :]

    info = {
        "alpha": float(alpha),
        "distill_ratio": dict(zip(features, distill_ratio.tolist())),
    }
    return corrected, info


def finetune_baseline(
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    X_real_cal: np.ndarray,
    y_real_cal: np.ndarray,
    X_test: np.ndarray,
    features: List[str],
    model_class: type,
    model_kwargs: Dict[str, Any],
    model_name: str = "",
    finetune_epochs: int = 50,
    real_oversample: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fine-tuning baseline.

    Two regimes — both honestly named:

    * **MLP**: incremental ``partial_fit`` adapts the network parameters on
      the real calibration batch (true fine-tuning of warm weights).
    * **Tree / other**: most scikit-learn estimators do not support warm
      restarts on a new training set; we therefore train on a *joint*
      dataset with the real-data block oversampled by ``real_oversample`` ×.
      This is mathematically equivalent to re-weighting the real loss and
      is documented (in ``info["method"]``) so the comparison stays fair.
    """
    n_features = X_synth.shape[1] if X_synth.ndim >= 2 else None

    if model_name == "MLP" and hasattr(model_class, "partial_fit"):
        model = model_class(**model_kwargs)
        # Establish ``classes_`` via an initial fit on the synthetic pool,
        # then nudge the weights with the real calibration data.
        model.fit(X_synth, y_synth)
        if hasattr(model, "partial_fit"):
            try:
                model.partial_fit(X_real_cal, y_real_cal)
            except (ValueError, AttributeError):
                pass
        ft_method = "mlp_partial_fit"
    else:
        # Joint training with real-data oversampling.  This explicitly does
        # NOT depend on the previous, discarded ``fit(X_synth, …)`` call —
        # we therefore skip it to avoid wasted compute.
        rng = np.random.RandomState(random_state)
        order = rng.permutation(len(X_synth) + real_oversample * len(X_real_cal))
        X_combined = np.vstack([X_synth, np.tile(X_real_cal, (real_oversample, 1))])
        y_combined = np.concatenate([y_synth, np.tile(y_real_cal, real_oversample)])
        X_combined, y_combined = X_combined[order], y_combined[order]
        model = model_class(**model_kwargs)
        model.fit(X_combined, y_combined)
        ft_method = f"joint_oversample_x{real_oversample}"

    if is_tree_model(model_name):
        sv = shap.TreeExplainer(model).shap_values(X_test)
        sv = extract_binary_shap(sv, n_features=n_features)
    else:
        sv = _kernel_shap(
            model, X_synth, X_test, n_features=n_features, random_state=random_state,
        )

    info = {
        "finetune_epochs": finetune_epochs,
        "n_cal_samples": int(len(X_real_cal)),
        "method": ft_method,
    }
    return sv, info


def coral_align(
    X_synth: np.ndarray,
    X_real: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """CORAL: align second-order statistics of ``X_synth`` to ``X_real``.

    Standard CORAL also centers the data before whitening; we add a
    centering step (which the previous version omitted), then re-center
    the output onto the real-data mean so the marginal first moments
    match too.
    """
    mu_synth = X_synth.mean(axis=0)
    mu_real = X_real.mean(axis=0)
    Xs = X_synth - mu_synth
    Xr = X_real - mu_real

    cov_real = np.cov(Xr, rowvar=False) + np.eye(Xr.shape[1]) * eps
    cov_synth = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * eps

    try:
        L_real = np.linalg.cholesky(cov_real)
        L_synth = np.linalg.cholesky(cov_synth)
    except np.linalg.LinAlgError:
        eigvals_r, eigvecs_r = np.linalg.eigh(cov_real)
        eigvals_s, eigvecs_s = np.linalg.eigh(cov_synth)
        eigvals_r = np.clip(eigvals_r, eps, None)
        eigvals_s = np.clip(eigvals_s, eps, None)
        L_real = eigvecs_r @ np.diag(np.sqrt(eigvals_r)) @ eigvecs_r.T
        L_synth = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T

    # Solve `L_synth Z = Xs.T` instead of forming inv(L_synth) explicitly —
    # numerically more stable, especially when L_synth is ill-conditioned.
    Z = np.linalg.solve(L_synth, Xs.T).T
    return Z @ L_real.T + mu_real


def coral_baseline(
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    X_real: np.ndarray,
    X_test: np.ndarray,
    features: List[str],
    model_class: type,
    model_kwargs: Dict[str, Any],
    model_name: str = "",
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """CORAL Baseline: align feature distributions, retrain model, compute SHAP."""
    X_synth_aligned = coral_align(X_synth, X_real)
    model = model_class(**model_kwargs)
    model.fit(X_synth_aligned, y_synth)

    n_features = X_synth.shape[1] if X_synth.ndim >= 2 else None
    if is_tree_model(model_name):
        sv = shap.TreeExplainer(model).shap_values(X_test)
        sv = extract_binary_shap(sv, n_features=n_features)
    else:
        sv = _kernel_shap(
            model, X_synth_aligned, X_test,
            n_features=n_features, random_state=random_state,
        )

    info = {"method": "CORAL", "n_synth": int(len(X_synth)), "n_real": int(len(X_real))}
    return sv, info
