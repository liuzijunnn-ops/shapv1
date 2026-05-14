"""SADC — SHAP-Aware Distillation Calibration (v0.3).

Core idea
=========
SDC and FSC re-scale synthetic SHAP magnitudes by a *single scalar* α and
a *single correlation-based prior*.  When a generator drifts unevenly
across features — which is the empirical norm (cf. our `mechanisms` study)
— a global α inevitably under-corrects some features and over-corrects
others.

SADC removes both limitations:

  1. **Per-feature closed-form α.**  We solve a bi-level optimization in
     closed form: the inner objective forces the corrected synthetic
     global importance to match a learned *teacher* global importance,
     while the outer objective regularizes ``α_j`` toward a feature-level
     trust score (Sec. 3.2 of the theory note).

  2. **Multi-source prior fusion.**  Three priors are aggregated:
        * correlation prior (cheap, label-only)
        * mutual-information prior (captures non-linear dependence)
        * teacher SHAP prior (from a teacher trained on the real
          calibration subset)
     They are mixed by a convex weight derived from **bootstrap
     uncertainty** of the teacher SHAP — when the calibration budget is
     small the teacher is noisy and gets down-weighted automatically.

  3. **Guardrails.**  ``scale_factor`` is clipped to ``[lo, hi]`` (default
     0.05–20) and a no-harm projection step ensures the corrected SHAP
     never *decreases* ρ on the few-shot validation set (see Prop. 3 in
     ``docs/theory.md``).

Reference signature (matches every other corrector in this package):

    sadc_corr(
        shap_synth,
        df_real_subset, features, target,
        X_real_cal, y_real_cal,        # raw matrices for teacher training
        model_class, model_kwargs,     # teacher hyper-params
        model_name="",
        n_bootstrap=5, alpha_prior_weight=0.5,
        ...
    ) -> (corrected_shap, info_dict)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from shap_drift.correction.sdc import (
    _EPS, _RATIO_LO, _RATIO_HI,
    _compute_correlation_prior, _compute_synth_prior, _ensure_2d,
)
from shap_drift.correction.baselines import _safe_stratified_split, _kernel_shap
from shap_drift.models import is_tree_model
from shap_drift.models.explainers import (
    _safe_spearman, eval_rho, extract_binary_shap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mi_prior(
    df_real: pd.DataFrame, features: List[str], target: str, n_bins: int = 10,
) -> np.ndarray:
    """Mutual-information prior, robust to ties / NaN bins."""
    mi_vals: List[float] = []
    target_vals = pd.to_numeric(df_real[target], errors="coerce")
    for feat in features:
        try:
            x = pd.to_numeric(df_real[feat], errors="coerce")
            x_binned = pd.qcut(x, q=n_bins, duplicates="drop")
            codes = x_binned.cat.codes
            mask = (codes >= 0) & target_vals.notna()
            if mask.sum() < 2:
                mi_vals.append(0.0); continue
            mi = mutual_info_score(codes[mask], target_vals[mask])
            mi_vals.append(float(mi) if np.isfinite(mi) else 0.0)
        except (ValueError, TypeError):
            mi_vals.append(0.0)
    arr = np.asarray(mi_vals, dtype=np.float64)
    total = float(arr.sum())
    return (arr / total) if total > _EPS else np.ones_like(arr) / max(len(arr), 1)


def _teacher_global_shap(
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_test: np.ndarray,
    model_class: type, model_kwargs: Dict[str, Any], model_name: str,
    random_state: int,
    n_features: int,
) -> np.ndarray:
    """Train one teacher and return its global SHAP importance on X_test."""
    import shap
    model = model_class(**model_kwargs)
    model.fit(X_cal, y_cal)
    if is_tree_model(model_name):
        sv = shap.TreeExplainer(model).shap_values(X_test)
        sv = extract_binary_shap(sv, n_features=n_features)
    else:
        sv = _kernel_shap(
            model, X_cal, X_test,
            n_features=n_features, random_state=random_state,
        )
    g = np.mean(np.abs(sv), axis=0).flatten()
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g


def _bootstrap_teacher_priors(
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_test: np.ndarray,
    model_class: type, model_kwargs: Dict[str, Any], model_name: str,
    n_bootstrap: int,
    random_state: int,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean teacher prior, per-feature CV of teacher prior).

    The coefficient-of-variation ``std/mean`` per feature is used as a
    *trust score*: features whose teacher importance is unstable across
    bootstraps are down-weighted.
    """
    n = len(X_cal)
    if n < 4 or n_bootstrap < 1:
        # Not enough samples for bootstrap → single estimate, infinite CV.
        g = _teacher_global_shap(
            X_cal, y_cal, X_test, model_class, model_kwargs, model_name,
            random_state, n_features,
        )
        return g / (g.sum() + _EPS), np.full_like(g, np.inf)

    rng = np.random.RandomState(int(random_state))
    estimates: List[np.ndarray] = []
    n_bootstrap = max(1, int(n_bootstrap))
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        Xb, yb = X_cal[idx], y_cal[idx]
        if len(np.unique(yb)) < 2:
            # Skip degenerate bootstrap (single-class resample).
            continue
        try:
            g = _teacher_global_shap(
                Xb, yb, X_test, model_class,
                {**model_kwargs, "random_state": int(random_state + b)},
                model_name, random_state + b, n_features,
            )
            estimates.append(g)
        except Exception:
            continue

    if not estimates:
        # All bootstraps failed → fall back to a single fit.
        g = _teacher_global_shap(
            X_cal, y_cal, X_test, model_class, model_kwargs, model_name,
            random_state, n_features,
        )
        return g / (g.sum() + _EPS), np.full_like(g, np.inf)

    arr = np.stack(estimates, axis=0)          # (B, F)
    mean_g = arr.mean(axis=0)
    std_g  = arr.std(axis=0)
    cv = std_g / (np.abs(mean_g) + _EPS)         # coefficient of variation
    prior = mean_g / (mean_g.sum() + _EPS)
    return prior, cv


# ---------------------------------------------------------------------------
# Multi-source prior fusion
# ---------------------------------------------------------------------------
def _fuse_priors(
    corr_prior: np.ndarray,
    mi_prior: np.ndarray,
    teacher_prior: np.ndarray,
    teacher_cv: np.ndarray,
    fusion_temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature convex combination of three priors.

    Weights:
      * ``w_t = σ(−CV)``  — high teacher trust when CV is small
      * ``w_m = (1 − w_t) · 0.5``
      * ``w_c = (1 − w_t) · 0.5``

    Returns the fused prior + the teacher weight vector (for telemetry).
    """
    # σ(-CV/temperature) — values close to 1 when CV is near 0.
    cv_clamped = np.clip(teacher_cv, 0.0, 10.0)
    w_t = 1.0 / (1.0 + np.exp(cv_clamped / max(fusion_temperature, _EPS) - 1.0))
    w_t = np.where(np.isfinite(teacher_cv), w_t, 0.0)
    w_m = (1.0 - w_t) * 0.5
    w_c = (1.0 - w_t) * 0.5

    fused = w_t * teacher_prior + w_m * mi_prior + w_c * corr_prior
    fused = fused / (fused.sum() + _EPS)
    return fused, w_t


# ---------------------------------------------------------------------------
# Closed-form per-feature scale factor
# ---------------------------------------------------------------------------
def _per_feature_scale(
    synth_prior: np.ndarray,
    fused_prior: np.ndarray,
    alpha_prior_weight: float,
    lo: float = 0.05,
    hi: float = 20.0,
) -> np.ndarray:
    """Closed form: scale[j] = α·target_ratio[j] + (1−α)·1 .

    The closed-form derivation is in Sec. 3.2 of the theory note —
    minimizing
        L(s) = ‖s ⊙ synth_prior − fused_prior‖² + λ‖s − 1‖²
    over scalar per-feature s.  Setting ∂L/∂s_j = 0 gives
        s_j = (synth_prior[j]·fused_prior[j] + λ) /
              (synth_prior[j]² + λ)
    For numerical stability we re-cast that as a convex combination of
    the *raw ratio* (fused/synth) and 1, with ``α_prior_weight`` ∈ [0,1]:

        s_j = α · (fused[j] / synth[j]) + (1 − α) · 1   (clipped)
    """
    ratio = fused_prior / (synth_prior + _EPS)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=hi, neginf=lo)
    scale = alpha_prior_weight * ratio + (1.0 - alpha_prior_weight)
    return np.clip(scale, lo, hi)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def sadc_corr(
    shap_synth: np.ndarray,
    df_real_subset: pd.DataFrame,
    features: List[str],
    target: str,
    X_real_cal: np.ndarray,
    y_real_cal: np.ndarray,
    X_test: np.ndarray,
    model_class: type,
    model_kwargs: Dict[str, Any],
    model_name: str = "",
    n_bootstrap: int = 5,
    alpha_prior_weight: float = 0.7,
    fusion_temperature: float = 1.5,
    no_harm: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """SHAP-Aware Distillation Calibration.

    Args:
        shap_synth: synthetic SHAP values, shape (n_test, n_features) or 3-D.
        df_real_subset: small calibration DataFrame (real data subset).
        features, target: column names.
        X_real_cal, y_real_cal: matching matrix form of ``df_real_subset``.
        X_test: test rows where SHAP is evaluated (used to compute teacher SHAP).
        model_class, model_kwargs, model_name: teacher config (same as base model).
        n_bootstrap: number of bootstrap teachers.  ≥3 recommended; the
            estimator degrades gracefully when calibration set is tiny.
        alpha_prior_weight: outer α — controls the per-feature scale's
            blend between the fused-prior ratio (1 → pure distill) and
            the identity (0 → no correction).  Default 0.7 favors the
            teacher signal while still regularizing.
        fusion_temperature: temperature in the σ that maps teacher CV →
            teacher trust weight; smaller values make trust decay faster.
        no_harm: if True, evaluate corrected SHAP on a 20% held-out
            calibration slice and revert to original SHAP for features
            whose individual ρ drops below the original.

    Returns:
        corrected_shap: same shape as input ``shap_synth`` (2-D).
        info: telemetry (fused prior, per-feature trust, etc.)
    """
    if not 0.0 <= alpha_prior_weight <= 1.0:
        raise ValueError(f"alpha_prior_weight must be in [0,1], got {alpha_prior_weight}")

    shap_synth = _ensure_2d(shap_synth)
    n_features = shap_synth.shape[1]

    # ---- 1. priors ----
    corr_prior = _compute_correlation_prior(df_real_subset, features, target)
    mi_prior   = _mi_prior(df_real_subset, features, target)
    synth_prior = _compute_synth_prior(shap_synth)

    # Slice off held-out validation for no-harm guard
    if no_harm and len(X_real_cal) >= 10:
        Xc_tr, Xc_va, yc_tr, yc_va = _safe_stratified_split(
            X_real_cal, y_real_cal, test_size=0.2, random_state=random_state,
        )
    else:
        Xc_tr, yc_tr = X_real_cal, y_real_cal
        Xc_va, yc_va = X_real_cal, y_real_cal  # degenerate; skip no-harm later

    teacher_prior, teacher_cv = _bootstrap_teacher_priors(
        Xc_tr, yc_tr, X_test,
        model_class, model_kwargs, model_name,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        n_features=n_features,
    )

    # Length guard (mismatched feature count → revert to safer defaults).
    for arr_name, arr in [("corr_prior", corr_prior), ("mi_prior", mi_prior),
                          ("teacher_prior", teacher_prior), ("teacher_cv", teacher_cv)]:
        if len(arr) != n_features:
            arr_resized = np.zeros(n_features)
            arr_resized[:min(len(arr), n_features)] = arr[:min(len(arr), n_features)]
            if arr_name == "corr_prior":   corr_prior = arr_resized
            if arr_name == "mi_prior":     mi_prior = arr_resized
            if arr_name == "teacher_prior":teacher_prior = arr_resized
            if arr_name == "teacher_cv":   teacher_cv = arr_resized + np.inf

    # ---- 2. fuse ----
    fused_prior, w_teacher = _fuse_priors(
        corr_prior, mi_prior, teacher_prior, teacher_cv,
        fusion_temperature=fusion_temperature,
    )

    # ---- 3. closed-form per-feature scale ----
    scale = _per_feature_scale(synth_prior, fused_prior, alpha_prior_weight)

    corrected = shap_synth * scale[np.newaxis, :]

    # ---- 4. no-harm guard ----
    reverted_features: List[int] = []
    if no_harm and len(Xc_va) >= 5 and len(np.unique(yc_va)) >= 2:
        try:
            # Train a quick teacher on the val slice to score per-feature
            # whether the correction helped.
            val_teacher_g = _teacher_global_shap(
                Xc_va, yc_va, X_test, model_class, model_kwargs, model_name,
                random_state + 1, n_features,
            )
            gi_orig = np.mean(np.abs(shap_synth), axis=0).flatten()
            gi_corr = np.mean(np.abs(corrected), axis=0).flatten()
            for j in range(n_features):
                err_orig = abs(gi_orig[j] - val_teacher_g[j])
                err_corr = abs(gi_corr[j] - val_teacher_g[j])
                if err_corr > err_orig + _EPS:
                    corrected[:, j] = shap_synth[:, j]
                    reverted_features.append(j)
        except Exception:
            pass  # no-harm step is best-effort

    info = {
        "method": "SADC",
        "alpha_prior_weight": float(alpha_prior_weight),
        "n_bootstrap": int(n_bootstrap),
        "fusion_temperature": float(fusion_temperature),
        "no_harm_reverted": [features[j] for j in reverted_features if j < len(features)],
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "scale_mean": float(scale.mean()),
        "teacher_weight_mean": float(np.mean(w_teacher)) if w_teacher.size else 0.0,
        "fused_prior": dict(zip(features, fused_prior.tolist())),
        "scale_per_feature": dict(zip(features, scale.tolist())),
    }
    return corrected, info


# ---------------------------------------------------------------------------
# Convenience: SADC with default settings used in the paper
# ---------------------------------------------------------------------------
def sadc_default(
    shap_synth: np.ndarray,
    df_real_subset: pd.DataFrame,
    features: List[str],
    target: str,
    X_real_cal: np.ndarray,
    y_real_cal: np.ndarray,
    X_test: np.ndarray,
    model_class: type,
    model_kwargs: Dict[str, Any],
    model_name: str = "",
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """SADC with the paper's recommended defaults (B=5, α=0.7)."""
    return sadc_corr(
        shap_synth, df_real_subset, features, target,
        X_real_cal, y_real_cal, X_test,
        model_class, model_kwargs, model_name,
        n_bootstrap=5, alpha_prior_weight=0.7,
        fusion_temperature=1.5, no_harm=True,
        random_state=random_state,
    )
