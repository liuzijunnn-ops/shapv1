"""SHAP drift metrics — core drift measurement functions."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from shap_drift.models.explainers import _safe_spearman

log = logging.getLogger(__name__)

_EPS = 1e-10


def drift_metrics(
    shap_real: np.ndarray,
    global_real: np.ndarray,
    shap_synth: np.ndarray,
    global_synth: np.ndarray,
    features: List[str],
) -> Dict[str, Any]:
    """Compute SHAP drift metrics between real and synthetic explanations."""
    # Defensive: replace NaN/inf
    shap_real = np.nan_to_num(shap_real, nan=0.0, posinf=0.0, neginf=0.0)
    shap_synth = np.nan_to_num(shap_synth, nan=0.0, posinf=0.0, neginf=0.0)
    global_real = np.nan_to_num(global_real, nan=0.0, posinf=0.0, neginf=0.0)
    global_synth = np.nan_to_num(global_synth, nan=0.0, posinf=0.0, neginf=0.0)

    # Length consistency: drift_metrics can be called with global vectors
    # whose length disagrees with ``features`` (e.g. after a partial slice).
    # Truncate to a common safe length to avoid index-out-of-range when
    # building the top-k feature lists.
    n_features = min(len(features), len(global_real), len(global_synth))
    if n_features == 0:
        return {
            "rho": 0.0, "p": 1.0, "attr_dist_norm": 0.0, "sign_agree": 0.0,
            "top": {}, "mean_ks": 0.0, "local_ks": {},
        }
    features = features[:n_features]
    global_real = global_real[:n_features]
    global_synth = global_synth[:n_features]
    shap_real = shap_real[:, :n_features]
    shap_synth = shap_synth[:, :n_features]

    mean_real = np.mean(shap_real, axis=0)
    mean_synth = np.mean(shap_synth, axis=0)

    attr_dist = float(
        np.linalg.norm(mean_real - mean_synth)
        / (np.linalg.norm(mean_real) + _EPS)
    )
    rho, p_val = _safe_spearman(global_real, global_synth)

    top_overlap: Dict[str, Any] = {}
    for k in (3, 5):
        k_clipped = min(k, n_features)
        if k_clipped == 0:
            continue
        top_real = set(int(x) for x in np.argsort(np.abs(global_real))[-k_clipped:])
        top_synth = set(int(x) for x in np.argsort(np.abs(global_synth))[-k_clipped:])
        top_overlap[f"top{k}"] = {
            "ratio": len(top_real & top_synth) / k_clipped,
            "k_effective": k_clipped,
            "real": [features[i] for i in sorted(top_real)],
            "synth": [features[i] for i in sorted(top_synth)],
        }

    sign_agree = float(np.mean(np.sign(mean_real) == np.sign(mean_synth)))

    local_ks: Dict[str, float] = {}
    for i, feat_name in enumerate(features):
        a = shap_real[:, i].ravel()
        b = shap_synth[:, i].ravel()
        if a.size == 0 or b.size == 0:
            local_ks[feat_name] = 0.0
            continue
        ks_stat, _ = stats.ks_2samp(a, b)
        local_ks[feat_name] = float(ks_stat) if np.isfinite(ks_stat) else 0.0

    return {
        "rho": rho,
        "p": p_val,
        "attr_dist_norm": attr_dist,
        "sign_agree": sign_agree,
        "top": top_overlap,
        "mean_ks": float(np.mean(list(local_ks.values()))) if local_ks else 0.0,
        "local_ks": local_ks,
    }


def per_sample_consistency(
    shap_real: np.ndarray,
    shap_synth: np.ndarray,
) -> Dict[str, Any]:
    """Per-sample SHAP cosine similarity and rank correlation (vectorized)."""
    n = min(shap_real.shape[0], shap_synth.shape[0])
    if n == 0:
        return {
            "cos_sim_mean": 0.0, "cos_sim_std": 0.0,
            "per_rho_mean": 0.0, "per_rho_std": 0.0,
            "per_sign_mean": 0.0, "n_samples": 0,
        }
    sr = shap_real[:n]
    ss = shap_synth[:n]

    vr = sr.reshape(n, -1)
    vs = ss.reshape(n, -1)

    norm_r = np.linalg.norm(vr, axis=1)
    norm_s = np.linalg.norm(vs, axis=1)
    valid_cos = (norm_r > _EPS) & (norm_s > _EPS)
    cos_sims = np.full(n, np.nan)
    cos_sims[valid_cos] = (
        np.sum(vr[valid_cos] * vs[valid_cos], axis=1)
        / (norm_r[valid_cos] * norm_s[valid_cos])
    )
    valid_cos_vals = cos_sims[~np.isnan(cos_sims)]

    std_r = np.std(vr, axis=1)
    std_s = np.std(vs, axis=1)
    valid_rho = (std_r > _EPS) & (std_s > _EPS)
    rho_vals: List[float] = []
    for i in range(n):
        if not valid_rho[i] or vr.shape[1] < 2:
            continue
        rho_i, _ = _safe_spearman(vr[i], vs[i])
        rho_vals.append(rho_i)
    rho_arr = np.array(rho_vals) if rho_vals else np.empty(0)

    sign_agree_vals = np.mean(np.sign(vr) == np.sign(vs), axis=1)

    return {
        "cos_sim_mean": float(np.mean(valid_cos_vals)) if valid_cos_vals.size else 0.0,
        "cos_sim_std": float(np.std(valid_cos_vals)) if valid_cos_vals.size else 0.0,
        "per_rho_mean": float(np.mean(rho_arr)) if rho_arr.size else 0.0,
        "per_rho_std": float(np.std(rho_arr)) if rho_arr.size else 0.0,
        "per_sign_mean": float(np.mean(sign_agree_vals)),
        "n_samples": int(n),
    }


def _compute_kl_divergence(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    features: List[str],
    n_bins: int = 20,
) -> float:
    """Mean KL(real ‖ synth) across features.

    Correctness notes:
      1. The two histograms **must share bin edges** for KL to be defined.
         Previously each side used independent edges → ill-defined output.
      2. KL is defined for probability mass functions, so we normalize the
         histograms to sum to 1 (with Laplace smoothing) instead of using
         density-scaled counts.
    """
    kl_vals: List[float] = []
    for col in features:
        try:
            real_vals = pd.to_numeric(df_real[col], errors="coerce").dropna().to_numpy()
            synth_vals = pd.to_numeric(df_synth[col], errors="coerce").dropna().to_numpy()
            if real_vals.size == 0 or synth_vals.size == 0:
                continue
            combined = np.concatenate([real_vals, synth_vals])
            lo, hi = float(np.min(combined)), float(np.max(combined))
            if hi <= lo:
                # All values identical → KL is 0 in the trivial sense.
                continue
            bins = np.linspace(lo, hi, n_bins + 1)
            real_hist, _ = np.histogram(real_vals, bins=bins)
            synth_hist, _ = np.histogram(synth_vals, bins=bins)
            real_p = real_hist.astype(np.float64) + _EPS
            synth_p = synth_hist.astype(np.float64) + _EPS
            real_p /= real_p.sum()
            synth_p /= synth_p.sum()
            kl = float(stats.entropy(real_p, synth_p))
            if np.isfinite(kl):
                kl_vals.append(kl)
        except (ValueError, TypeError) as exc:
            log.debug("  KL divergence skipped for %s: %s", col, exc)
    return float(np.mean(kl_vals)) if kl_vals else 0.0
