"""SDC (SHAP Drift Correction) methods: SDC-Corr and SDC-Guarded."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from shap_drift.models.explainers import _safe_spearman, extract_binary_shap

# Numerical constants — centralized so the same epsilon is reused throughout
# the correction stack (avoids divergent ε values in sister files).
_EPS = 1e-10
_RATIO_LO, _RATIO_HI = 0.1, 10.0


def _ensure_2d(shap_values: np.ndarray) -> np.ndarray:
    """Ensure SHAP values are 2-D (n_samples, n_features).

    Delegates to :func:`extract_binary_shap` so the shape-detection logic
    (last-dim is class, ≤ _MAX_CLASSES) lives in a single place.  This avoids
    the historical bug where ``sv.shape[0] == 2`` mis-fired on a 2-sample
    test set.
    """
    sv = extract_binary_shap(shap_values)
    if sv.ndim != 2:
        raise ValueError(f"Expected 2-D SHAP array, got shape {sv.shape}")
    return sv


def _compute_correlation_prior(
    df_real: pd.DataFrame, features: List[str], target: str,
) -> np.ndarray:
    """Compute |corr(feature, target)| normalized to a probability simplex.

    Robust to constant features / target (which yield NaN from pandas'
    ``corrwith``) and to all-zero correlation vectors (falls back to a
    uniform prior so the downstream ratio remains well-defined).
    """
    real_corr = df_real[features].corrwith(df_real[target]).abs().values
    real_corr = np.nan_to_num(real_corr, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(real_corr.sum())
    if total < _EPS:
        # Degenerate case: no information in correlations → uniform prior.
        return np.ones_like(real_corr) / max(len(real_corr), 1)
    return real_corr / total


def _compute_synth_prior(shap_synth: np.ndarray) -> np.ndarray:
    """Compute synthetic global importance prior on the probability simplex."""
    synth_imp = np.nan_to_num(
        np.mean(np.abs(shap_synth), axis=0).flatten(),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    total = float(synth_imp.sum())
    if total < _EPS:
        return np.ones_like(synth_imp) / max(len(synth_imp), 1)
    return synth_imp / total


def _compute_ratio(
    shap_synth: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
) -> np.ndarray:
    """Compute the real/synth importance ratio (shared by SDC and FSC)."""
    shap_synth = _ensure_2d(shap_synth)
    real_prior = _compute_correlation_prior(df_real, features, target)
    synth_prior = _compute_synth_prior(shap_synth)
    return np.clip(real_prior / (synth_prior + _EPS), _RATIO_LO, _RATIO_HI)


def sdc_corr(
    shap_synth: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """SDC-Corr: Correlation-transfer correction.

    Uses real data's feature-target correlation as importance prior to reweight
    synthetic SHAP magnitudes.

    Formula:
        corrected[i,j] = SHAP[i,j] * (α * ratio[j] + 1 - α)
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    shap_synth = _ensure_2d(shap_synth)
    ratio = _compute_ratio(shap_synth, df_real, features, target)
    scale_factor = alpha * ratio + (1.0 - alpha)
    # `sign(x) * |x|` is identical to `x`; the previous version did extra work.
    corrected = shap_synth * scale_factor[np.newaxis, :]
    return corrected, dict(zip(features, ratio.tolist()))


def estimate_prior_reliability(
    shap_synth: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
) -> Dict[str, float]:
    """Estimate the reliability of the correlation-based prior.

    Returns:
        Dict with rank_agreement, ratio_spread, should_correct, confidence.

    Notes:
        * ``rank_agreement`` is Spearman ρ between the real-correlation prior
          and the synthetic global-importance prior; high ρ → priors agree.
        * ``confidence`` reflects the *strength* of that agreement signal —
          i.e. how far ρ is from 0.  Previously this was inverted (peaked
          at ρ = 0.5), which contradicted the docstring.
    """
    shap_synth = _ensure_2d(shap_synth)
    real_prior = _compute_correlation_prior(df_real, features, target)
    synth_prior = _compute_synth_prior(shap_synth)

    rank_agreement, _ = _safe_spearman(real_prior, synth_prior)
    ratio = real_prior / (synth_prior + _EPS)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=_RATIO_HI, neginf=_RATIO_LO)
    ratio_spread = float(np.std(ratio))

    # Confidence = |ρ| ∈ [0, 1]; high agreement (positive *or* negative) is
    # more informative than agreement near 0.
    confidence = float(min(1.0, max(0.0, abs(rank_agreement))))
    should_correct = (rank_agreement < 0.5) or (ratio_spread > 0.8)

    return {
        "rank_agreement": float(rank_agreement),
        "ratio_spread": ratio_spread,
        "should_correct": bool(should_correct),
        "confidence": confidence,
    }


def _dampen_alpha(
    rank_agreement: float,
    alpha: float,
    rho_threshold: float,
    fallback_alpha: float,
) -> float:
    """Linearly dampen alpha toward ``fallback_alpha`` once rank-agreement
    exceeds ``rho_threshold``.  At ρ = threshold the dampening factor is 1
    (full alpha); at ρ = 1 it is 0 (alpha = fallback_alpha).
    """
    if rank_agreement <= rho_threshold:
        return float(alpha)
    span = max(1.0 - rho_threshold, _EPS)
    dampening = 1.0 - (rank_agreement - rho_threshold) / span
    dampening = max(0.0, min(1.0, dampening))
    return float(fallback_alpha + (alpha - fallback_alpha) * dampening)


def sdc_corr_guarded(
    shap_synth: np.ndarray,
    df_real: pd.DataFrame,
    features: List[str],
    target: str,
    alpha: float = 0.5,
    rho_threshold: float = 0.7,
    fallback_alpha: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """SDC-Corr-Guarded: Adaptive correction with prior reliability guard.

    Prevents over-correction when the synth/real rank-agreement is already
    high (priors already aligned) by dampening alpha toward
    ``fallback_alpha``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= fallback_alpha <= 1.0:
        raise ValueError(f"fallback_alpha must be in [0, 1], got {fallback_alpha}")

    shap_synth = _ensure_2d(shap_synth)
    reliability = estimate_prior_reliability(shap_synth, df_real, features, target)
    effective_alpha = _dampen_alpha(
        reliability["rank_agreement"], alpha, rho_threshold, fallback_alpha,
    )

    ratio = _compute_ratio(shap_synth, df_real, features, target)
    scale_factor = effective_alpha * ratio + (1.0 - effective_alpha)
    corrected = shap_synth * scale_factor[np.newaxis, :]

    info = {
        "effective_alpha": effective_alpha,
        "requested_alpha": float(alpha),
        "dampened": bool(effective_alpha < alpha - _EPS),
        "reliability": reliability,
        "ratio": dict(zip(features, ratio.tolist())),
    }
    return corrected, info
