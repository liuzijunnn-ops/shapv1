"""Injury 1798 dataset loader — biomechanical gait analysis for injury prediction."""
from __future__ import annotations

import json
import logging
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

log = logging.getLogger(__name__)

INJ_TARGET: str = "injured"

# Selected features: top 12 biomechanical features by MI + 3 metadata = 15
# Determined via MI-based feature selection on the full dataset.
INJ_FEATURES: List[str] = [
    # Biomechanical (12) — selected by mutual information with injury status
    "left_HIP_ROT_PEAK_VEL",
    "left_STANCE_TIME",
    "right_HIP_ADD_PEAK_ANGLE",
    "right_KNEE_ADD_PEAK_ANGLE",
    "right_FOOT_ANG_at_HS",
    "right_HIP_EXT_percent_STANCE",
    "left_FOOT_PROG_ANGLE",
    "left_KNEE_FLEX_PEAK_ANGLE",
    "left_HIP_ABD_PEAK_VEL",
    "right_HIP_ROT_percent_STANCE",
    "left_HIP_EXT_PEAK_ANGLE",
    "right_ANKLE_DF_PEAK_ANGLE",
    # Metadata (3)
    "age",
    "speed_r",
    "gender",
]


def load_injury_1798() -> pd.DataFrame:
    """Load Injury 1798 dataset from JSON files.

    Target: InjDefn → binary (No injury=0, any injury=1).
    Features: 12 biomechanical gait variables + 3 metadata = 15 features.

    The 12 biomechanical features were selected via mutual information
    ranking from 152 candidate features (76 per side × 2 sides).
    These capture hip, knee, ankle, and temporal gait characteristics
    most predictive of running injury.
    """
    base_dir = "dataset/1798/reformat_data"
    meta_path = "dataset/1798/run_data_meta.csv"

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = pd.read_csv(meta_path)
    rows = []

    for _, mrow in meta.iterrows():
        subdir = str(mrow["sub_id"])
        fname = mrow["filename"]
        fpath = os.path.join(base_dir, subdir, fname)

        if not os.path.exists(fpath):
            continue

        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Binary target: injured (1) vs healthy (0)
        row = {"injured": 0 if mrow["InjDefn"] == "No injury" else 1}

        # Extract biomechanical features from dv_r
        dv_r = data.get("dv_r", {})
        bio_features = [f for f in INJ_FEATURES if f not in ("age", "speed_r", "gender")]
        for feat in bio_features:
            side, name = feat.split("_", 1)
            try:
                row[feat] = float(dv_r.get(side, {}).get(name, np.nan))
            except (TypeError, ValueError):
                row[feat] = np.nan

        # Metadata
        row["age"] = mrow["age"]
        row["speed_r"] = mrow["speed_r"]
        row["gender"] = 0 if mrow["Gender"] == "Female" else (1 if mrow["Gender"] == "Male" else np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Clean: drop rows with too many missing biomechanical features
    bio_cols = [f for f in INJ_FEATURES if f not in ("age", "speed_r", "gender")]
    df = df.dropna(subset=bio_cols, thresh=len(bio_cols) - 2).reset_index(drop=True)

    # Fill remaining NaN with median
    for col in INJ_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Clip age outliers (255 → likely data entry error)
    df["age"] = df["age"].clip(10, 100)

    log.info("  Injury1798: %d samples, %d features, class balance=%.1f%% positive",
             len(df), len(INJ_FEATURES), df[INJ_TARGET].mean() * 100)

    return df
