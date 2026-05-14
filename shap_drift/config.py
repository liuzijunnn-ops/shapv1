"""Centralized configuration for SHAP Explanation Drift Study.

Single source of truth for paths, seeds, generator registry, colors, and
utility functions.  Visualization scripts should import from here rather
than hard-coding values.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("results")
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Generator registry — 3 paradigms, each with 1 representative
# All implemented via table-synthesizers (stg) library.
# ---------------------------------------------------------------------------
GENERATOR_SUFFIX: Dict[str, str] = {
    "GaussianCopula": "gc",
    "CTGAN": "ctgan",
    "TVAE": "tvae",
}
GENERATORS: List[Tuple[str, str]] = list(GENERATOR_SUFFIX.items())

# ---------------------------------------------------------------------------
# Seeds (10 for higher statistical power)
# ---------------------------------------------------------------------------
SEEDS: List[int] = [42, 123, 456, 789, 2024, 314, 271, 828, 159, 653]
N_SEEDS: int = len(SEEDS)

# ---------------------------------------------------------------------------
# Generator color palette
# ---------------------------------------------------------------------------
GEN_COLORS: Dict[str, str] = {
    "GaussianCopula": "#2ecc71",
    "CTGAN": "#e74c3c",
    "TVAE": "#3498db",
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_synth_datasets(
    ds_name: str, features: List[str], target: str
) -> Dict[str, pd.DataFrame]:
    """Load all available synthetic datasets for a given dataset name."""
    synth_dfs: Dict[str, pd.DataFrame] = {}
    for gen_name, suffix in GENERATORS:
        fpath = OUTPUT_DIR / f"{ds_name}_{suffix}.csv"
        if not fpath.exists():
            continue
        df_s = pd.read_csv(fpath)
        for col in features:
            df_s[col] = pd.to_numeric(df_s[col], errors="coerce")
        df_s[target] = df_s[target].clip(0, 1).round().astype(int)
        synth_dfs[gen_name] = df_s
    return synth_dfs


def safe_corr(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute correlation matrix, replacing inf/nan with 0."""
    corr = df[features].corr()
    return corr.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_serializable(obj: object) -> object:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def set_global_seed(seed: int = 42) -> None:
    """Set random seed across all known RNGs for reproducibility.

    Covers (in best-effort order):
      * Python's built-in ``random``
      * NumPy's legacy global RNG
      * PyTorch (CPU + CUDA, both per-device and all-device seeds)
      * ``PYTHONHASHSEED`` for hash-based randomness in child processes

    Reproducibility caveats: thread-parallel reductions and certain CUDA
    kernels remain non-deterministic even with a fixed seed; callers that
    need bit-exact reproducibility should additionally configure
    ``torch.use_deterministic_algorithms(True)``.
    """
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def detect_cuda() -> Tuple[bool, str]:
    """Detect CUDA availability and return (is_available, info_string)."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        else:
            info = "CPU only"
        return available, info
    except ImportError:
        return False, "PyTorch not installed"


def setup_logging(name: str = __name__) -> logging.Logger:
    """Configure standard logging for pipeline scripts.

    Idempotent: a second call does NOT add duplicate handlers (the previous
    implementation re-ran ``basicConfig`` on every call, which is a no-op
    only if the root logger already had handlers — fragile behaviour).
    """
    import warnings
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    warnings.filterwarnings("ignore")
    return logging.getLogger(name)
