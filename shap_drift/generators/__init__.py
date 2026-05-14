"""Synthetic data generators — unified via table-synthesizers library.

3 paradigms, each with 1 representative:
  - Statistical: GaussianCopula (core, no extra deps)
  - GAN:         CTGAN (core, no extra deps)
  - VAE:         TVAE  (core, no extra deps)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from shap_drift.config import GENERATOR_SUFFIX, GENERATORS, OUTPUT_DIR, detect_cuda

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
_SMALL_DATASET_THRESHOLD = 500


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _get_default_config(gen_name: str, n_samples: int) -> Dict[str, Any]:
    """Return default training config for each generator, adjusted for size."""
    cuda_available, _ = detect_cuda()
    configs: Dict[str, Dict[str, Any]] = {
        "GaussianCopula": {},
        "CTGAN": {
            "epochs": 10 if n_samples < _SMALL_DATASET_THRESHOLD else 50,
            "cuda": cuda_available,
            "verbose": False,
        },
        "TVAE": {
            "epochs": 10 if n_samples < _SMALL_DATASET_THRESHOLD else 50,
        },
    }
    if n_samples < _SMALL_DATASET_THRESHOLD:
        if gen_name == "CTGAN":
            configs["CTGAN"]["generator_dim"] = [64, 64]
            configs["CTGAN"]["discriminator_dim"] = [64, 64]
        elif gen_name == "TVAE":
            configs["TVAE"]["compress_dims"] = [64, 64]
            configs["TVAE"]["decompress_dims"] = [64, 64]
    return configs.get(gen_name, {})


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def _sample(synth: Any, n: int) -> pd.DataFrame:
    """Sample from GaussianCopula / CTGAN / TVAE."""
    result = synth.sample(n, return_dataframe=True)
    if isinstance(result, pd.DataFrame):
        return result
    # Fallback: raw tensor → DataFrame
    raw = result
    if hasattr(raw, "cpu"):
        raw = raw.cpu().detach().numpy()
    elif hasattr(raw, "numpy"):
        raw = raw.numpy()
    return pd.DataFrame(np.array(raw))


# ---------------------------------------------------------------------------
# Main generation entry points
# ---------------------------------------------------------------------------
def generate_one(
    df_work: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    gen_name: str,
    ds_name: str,
    config_override: Optional[Dict[str, Any]] = None,
) -> bool:
    """Generate one synthetic dataset.  Returns True on success."""
    from stg.tableSynthesizer import TableSynthesizer

    n = len(df_work)
    suffix = GENERATOR_SUFFIX.get(gen_name, gen_name.lower()[:4])
    out_path = OUTPUT_DIR / f"{ds_name}_{suffix}.csv"

    if out_path.exists():
        log.info("  %s: already exists", gen_name)
        return True

    try:
        default_config = _get_default_config(gen_name, n)
        if config_override:
            default_config.update(config_override)

        synth = TableSynthesizer(model=gen_name, config=default_config)
        synth.fit(df_work)
        df_synth = _sample(synth, n)

        # Ensure columns match
        if list(df_synth.columns) != list(df_work.columns):
            df_synth.columns = df_work.columns

        # Clip numeric features to real-data range
        for col in feature_cols:
            lo, hi = df_work[col].min(), df_work[col].max()
            df_synth[col] = pd.to_numeric(df_synth[col], errors="coerce").clip(lo, hi)

        # Binarize target — coerce non-numeric values to NaN first so the
        # subsequent ``round().astype(int)`` cannot raise on stray strings.
        df_synth[target_col] = (
            pd.to_numeric(df_synth[target_col], errors="coerce")
              .fillna(0)
              .round()
              .astype(int)
              .clip(0, 1)
        )

        # Drop NaN rows
        df_synth = df_synth.dropna(subset=feature_cols).reset_index(drop=True)

        # Final check: target must have ≥ 2 classes
        real_pos_rate = float(df_work[target_col].mean())
        n_classes = df_synth[target_col].nunique()
        if n_classes < 2:
            log.warning(
                "  %s: target has %d class(es) after generation — "
                "forcing real_pos_rate=%.2f%%",
                gen_name, n_classes, real_pos_rate * 100,
            )
            n_pos = int(len(df_synth) * real_pos_rate)
            labels = np.zeros(len(df_synth), dtype=int)
            labels[:n_pos] = 1
            # Use a seeded RNG so the rebalance is deterministic w.r.t.
            # dataset+generator combinations.
            rng = np.random.RandomState(abs(hash((ds_name, gen_name))) % (2**32))
            rng.shuffle(labels)
            df_synth[target_col] = labels

        df_synth.to_csv(out_path, index=False)
        log.info("  %s: %d samples generated", gen_name, len(df_synth))
        return True

    except ImportError as exc:
        log.error("  %s: MISSING DEPENDENCY — %s", gen_name, exc)
        return False
    except Exception as exc:
        log.error("  %s: FAILED — %s", gen_name, exc)
        return False


def generate_all(
    datasets: Dict[str, Any],
) -> None:
    """Generate synthetic data for all dataset × generator combinations."""
    from shap_drift.datasets import prepare_dataset

    log.info("=" * 70)
    log.info("  SYNTHETIC DATA GENERATION")
    log.info("=" * 70)

    for ds_name, ds_cfg in datasets.items():
        log.info("  %s:", ds_name)
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        df_clean, X, y = prepare_dataset(df, features, target)
        df_work = df_clean[features + [target]].copy()

        for gen_name, _ in GENERATORS:
            generate_one(df_work, features, target, gen_name, ds_name)

    log.info("  ✓ Generation complete.")
