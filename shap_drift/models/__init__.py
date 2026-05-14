"""Model definitions — single source of truth for MODEL_CONFIGS and EXPLAINER_MAP.

All baseline estimators carry a fixed default ``random_state`` so that
direct ``model_class(**model_kwargs)`` invocations from baselines / ablation
paths remain reproducible even when the caller does not pass an explicit
seed.  Per-seed experiments in ``run.py`` still override ``random_state``
on a per-call basis (see ``{**mk_base, "random_state": seed}``).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Model specification type
ModelSpec = Tuple[str, type, Dict[str, Any]]

DEFAULT_RANDOM_STATE: int = 42

# Model configurations.  We deliberately freeze each kwargs dict via a
# defensive ``copy.deepcopy`` accessor (``get_model_configs``) to prevent
# accidental cross-experiment mutation when callers do ``mk_base["x"] = …``.
_MODEL_CONFIGS_TEMPLATE: List[ModelSpec] = [
    (
        "XGBoost",
        xgb.XGBClassifier,
        dict(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=DEFAULT_RANDOM_STATE,
        ),
    ),
    (
        "RandomForest",
        RandomForestClassifier,
        dict(
            n_estimators=200, max_depth=8, n_jobs=-1,
            random_state=DEFAULT_RANDOM_STATE,
        ),
    ),
    (
        "MLP",
        MLPClassifier,
        dict(
            hidden_layer_sizes=(128, 64), max_iter=300,
            early_stopping=True,
            random_state=DEFAULT_RANDOM_STATE,
        ),
    ),
]

# Public, mutation-safe accessor.
MODEL_CONFIGS: List[ModelSpec] = [
    (name, cls, copy.deepcopy(cfg)) for name, cls, cfg in _MODEL_CONFIGS_TEMPLATE
]


def get_model_configs() -> List[ModelSpec]:
    """Return a fresh deep copy of MODEL_CONFIGS so callers cannot mutate
    the shared template across experiments."""
    return [(name, cls, copy.deepcopy(cfg)) for name, cls, cfg in _MODEL_CONFIGS_TEMPLATE]


MODEL_ORDER: List[str] = [name for name, _, _ in _MODEL_CONFIGS_TEMPLATE]

# Explainer mapping
EXPLAINER_MAP: Dict[str, str] = {
    "XGBoost": "tree",
    "RandomForest": "tree",
    "MLP": "kernel",
}


def is_tree_model(model_name: str) -> bool:
    """Check if a model uses TreeExplainer."""
    return EXPLAINER_MAP.get(model_name, "tree") == "tree"
