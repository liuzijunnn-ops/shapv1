"""FSC (Few-Shot Calibration) methods: FSC-Corr and FSC-Guarded.

These are functionally SDC-Corr / SDC-Corr-Guarded restricted to a small
calibration subset of the real data.  We deliberately delegate to the SDC
internals to keep the formula in *one* place (eliminates a long-standing
drift bug where the two implementations diverged).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from shap_drift.correction.sdc import (
    _EPS,
    _compute_ratio,
    _dampen_alpha,
    _ensure_2d,
    estimate_prior_reliability,
)


def fsc_corr(
    shap_synth: np.ndarray,
    df_real_subset: pd.DataFrame,
    features: List[str],
    target: str,
    alpha: float = 0.5,
) -> np.ndarray:
    """FSC-Corr: SDC-Corr with a few-shot real-data prior.

    The only difference vs. SDC-Corr is the size of ``df_real_subset`` —
    here we expect a small calibration set drawn from the real distribution.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    shap_synth = _ensure_2d(shap_synth)
    ratio = _compute_ratio(shap_synth, df_real_subset, features, target)
    scale_factor = alpha * ratio + (1.0 - alpha)
    return shap_synth * scale_factor[np.newaxis, :]


def fsc_corr_guarded(
    shap_synth: np.ndarray,
    df_real_subset: pd.DataFrame,
    features: List[str],
    target: str,
    alpha: float = 0.5,
    rho_threshold: float = 0.7,
    fallback_alpha: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """FSC-Corr-Guarded: Few-shot calibration with prior reliability guard."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= fallback_alpha <= 1.0:
        raise ValueError(f"fallback_alpha must be in [0, 1], got {fallback_alpha}")

    shap_synth = _ensure_2d(shap_synth)
    reliability = estimate_prior_reliability(shap_synth, df_real_subset, features, target)
    effective_alpha = _dampen_alpha(
        reliability["rank_agreement"], alpha, rho_threshold, fallback_alpha,
    )

    ratio = _compute_ratio(shap_synth, df_real_subset, features, target)
    scale_factor = effective_alpha * ratio + (1.0 - effective_alpha)
    corrected = shap_synth * scale_factor[np.newaxis, :]

    info = {
        "effective_alpha": effective_alpha,
        "requested_alpha": float(alpha),
        "dampened": bool(effective_alpha < alpha - _EPS),
        "reliability": reliability,
    }
    return corrected, info
