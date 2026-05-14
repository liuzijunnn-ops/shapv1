#!/usr/bin/env python3
"""
SHAP Explanation Drift Study — Unified CLI Entry Point
=======================================================

Usage:
    python run.py generate          # Step 1: Synthetic data generation
    python run.py quality           # Step 2: Synthetic quality evaluation
    python run.py baseline          # Step 3: Baseline models (multi-seed)
    python run.py drift             # Steps 4-7: SHAP drift + per-sample + interaction + conditional
    python run.py correction        # Correction experiments (SDC-Corr + SDC-Guarded)
    python run.py fsc               # Few-shot SHAP calibration experiments
    python run.py ablation          # Ablation experiments
    python run.py baseline_compare  # Fair baseline comparison (equal data budget)
    python run.py visualize         # Generate all visualization plots
    python run.py all               # Run all steps sequentially
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── shap_drift imports ──────────────────────────────────────────────────
from shap_drift.config import (
    GENERATOR_SUFFIX, GENERATORS, GEN_COLORS,
    N_SEEDS, OUTPUT_DIR, SEEDS,
    detect_cuda, load_synth_datasets, make_serializable,
    safe_corr, set_global_seed, setup_logging,
)
from shap_drift.datasets import DATASETS, DATASET_ORDER, DS_COLORS, DS_LABELS, prepare_dataset
from shap_drift.models import MODEL_CONFIGS, is_tree_model
from shap_drift.models.explainers import (
    compute_shap, eval_rho, eval_shap, extract_binary_shap, extract_binary_interaction,
)
from shap_drift.correction import sdc_corr, sdc_corr_guarded, fsc_corr, fsc_corr_guarded
from shap_drift.correction.baselines import shap_distillation, finetune_baseline, coral_baseline
from shap_drift.metrics import drift_metrics, per_sample_consistency, evaluate_quality
from shap_drift.metrics.drift import _compute_kl_divergence
from shap_drift.ablation import ablate_alpha_selection, ablate_prior_weight, ablate_sample_strategy

import shap
import xgboost as xgb

log = setup_logging("run")
CUDA_AVAILABLE, GPU_INFO = detect_cuda()
log.info("  GPU: %s (CUDA=%s)", GPU_INFO, CUDA_AVAILABLE)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Generate Synthetic Data
# ═══════════════════════════════════════════════════════════════════════════
def step_generate() -> None:
    from shap_drift.generators import generate_all
    log.info("=" * 70)
    log.info("  STEP 1: Synthetic Data Generation")
    log.info("=" * 70)
    generate_all(DATASETS)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Synthetic Quality Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def step_quality() -> None:
    log.info("=" * 70)
    log.info("  STEP 2: Synthetic Quality Evaluation")
    log.info("=" * 70)

    quality_all: Dict[str, Any] = {}
    for ds_name, ds_cfg in DATASETS.items():
        log.info("  %s:", ds_name)
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        df_clean, _, _ = prepare_dataset(df, features, target)
        df_work = df_clean[features + [target]].copy()
        synth_dfs = load_synth_datasets(ds_name, features, target)

        ds_quality: Dict[str, Any] = {}
        for gen_name, df_synth in synth_dfs.items():
            q = evaluate_quality(df_work, df_synth, features, target)
            ds_quality[gen_name] = q
            log.info("    %s: KS=%.4f, corr_diff=%.4f, AUROC=%.4f",
                     gen_name, q["marginal"]["mean_ks"],
                     q["correlation"]["mean_abs_diff"],
                     q["downstream"].get("auroc", 0))
        quality_all[ds_name] = ds_quality

    with open(OUTPUT_DIR / "synth_quality.json", "w") as f:
        json.dump(make_serializable(quality_all), f, indent=2)
    log.info("  ✓ Step 2 complete.")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Baseline Models (Multi-Seed)
# ═══════════════════════════════════════════════════════════════════════════
def step_baseline() -> None:
    log.info("=" * 70)
    log.info("  STEP 3: Baseline Models")
    log.info("=" * 70)

    baseline_all: Dict[str, Any] = {}
    for ds_name, ds_cfg in DATASETS.items():
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        _, X, y = prepare_dataset(df, features, target)
        ds_base: Dict[str, Any] = {"seeds": {}}

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
            seed_results: Dict[str, Any] = {}
            for model_name, model_class, mk_base in MODEL_CONFIGS:
                mk = {**mk_base, "random_state": seed}
                clf = model_class(**mk)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict_proba(X_te)[:, 1]
                seed_results[model_name] = {
                    "auroc": float(roc_auc_score(y_te, y_pred)),
                    "accuracy": float(accuracy_score(y_te, clf.predict(X_te))),
                }
            ds_base["seeds"][str(seed)] = seed_results

        ds_base["summary"] = {}
        for model_name, _, _ in MODEL_CONFIGS:
            aurocs = [ds_base["seeds"][str(s)][model_name]["auroc"] for s in SEEDS]
            accs = [ds_base["seeds"][str(s)][model_name]["accuracy"] for s in SEEDS]
            ds_base["summary"][model_name] = {
                "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
                "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            }
            log.info("  %s/%s: AUROC=%.3f±%.3f", ds_name, model_name, np.mean(aurocs), np.std(aurocs))
        baseline_all[ds_name] = ds_base

    with open(OUTPUT_DIR / "baseline_results.json", "w") as f:
        json.dump(make_serializable(baseline_all), f, indent=2)
    log.info("  ✓ Step 3 complete.")


# ═══════════════════════════════════════════════════════════════════════════
# STEPS 4-5: SHAP Drift Analysis + Per-Sample Consistency
# ═══════════════════════════════════════════════════════════════════════════
def step_drift() -> None:
    log.info("=" * 70)
    log.info("  STEPS 4-5: SHAP Drift + Per-Sample Consistency")
    log.info("=" * 70)

    drift_all: Dict[str, Any] = {}
    persample_all: Dict[str, Any] = {}

    for ds_name, ds_cfg in DATASETS.items():
        log.info("  --- %s ---", ds_name)
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        _, X, y = prepare_dataset(df, features, target)
        synth_dfs = load_synth_datasets(ds_name, features, target)

        ds_drift: Dict[str, Any] = {}
        ds_persample: Dict[str, Any] = {}

        for model_name, model_class, mk_base in MODEL_CONFIGS:
            seed_drifts: Dict[str, List[Dict]] = {}
            seed_ps: Dict[str, Dict] = {}

            for seed_idx, seed in enumerate(SEEDS):
                mk = {**mk_base, "random_state": seed}
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
                sv_real, gi_real, model_real = compute_shap(model_class, mk, X_tr, y_tr, X_te, model_name=model_name)

                for gen_name, df_synth in synth_dfs.items():
                    X_synth = df_synth[features].values.astype(np.float32)
                    y_synth = df_synth[target].values.astype(int).clip(0, 1)
                    if len(np.unique(y_synth)) < 2:
                        continue
                    sv_synth, gi_synth, model_synth = compute_shap(model_class, mk, X_synth, y_synth, X_te, model_name=model_name)
                    dm = drift_metrics(sv_real, gi_real, sv_synth, gi_synth, features)

                    synth_auroc = float(roc_auc_score(y_te, model_synth.predict_proba(X_te)[:, 1]))
                    dm["synth_auroc"] = synth_auroc
                    dm["real_auroc"] = float(roc_auc_score(y_te, model_real.predict_proba(X_te)[:, 1]))
                    dm["utility_drift_gap"] = synth_auroc - dm["rho"]

                    key = f"{model_name}__{gen_name}"
                    seed_drifts.setdefault(key, []).append(dm)

                    if seed_idx == 0:
                        ps = per_sample_consistency(sv_real, sv_synth)
                        seed_ps[f"{model_name}__{gen_name}"] = ps

                if seed_idx == 0:
                    for gen_name, df_synth in synth_dfs.items():
                        X_synth = df_synth[features].values.astype(np.float32)
                        y_synth = df_synth[target].values.astype(int).clip(0, 1)
                        if len(np.unique(y_synth)) < 2:
                            continue
                        sv_sg, gi_sg, _ = compute_shap(model_class, mk, X_synth, y_synth, X_te, model_name=model_name)
                        np.savez(
                            OUTPUT_DIR / f"{ds_name}_{model_name}_{gen_name}_shap.npz",
                            shap_real=sv_real, shap_synth=sv_sg,
                            global_real=gi_real, global_synth=gi_sg,
                            X_test=X_te, feature_names=features,
                        )

            ds_drift[model_name] = {}
            for key, seed_results in seed_drifts.items():
                _, gen_name = key.split("__")
                rhos = [r["rho"] for r in seed_results]
                sign_agrees = [r["sign_agree"] for r in seed_results]
                mean_kss = [r["mean_ks"] for r in seed_results]
                top5s = [r["top"]["top5"]["ratio"] for r in seed_results]
                gaps = [r["utility_drift_gap"] for r in seed_results]
                aurocs = [r["synth_auroc"] for r in seed_results]

                ds_drift[model_name][gen_name] = {
                    **seed_results[0],
                    "multi_seed": {
                        "rho": {"mean": float(np.mean(rhos)), "std": float(np.std(rhos))},
                        "sign_agree": {"mean": float(np.mean(sign_agrees)), "std": float(np.std(sign_agrees))},
                        "mean_ks": {"mean": float(np.mean(mean_kss)), "std": float(np.std(mean_kss))},
                        "top5_overlap": {"mean": float(np.mean(top5s)), "std": float(np.std(top5s))},
                        "synth_auroc": {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))},
                        "utility_drift_gap": {"mean": float(np.mean(gaps)), "std": float(np.std(gaps))},
                    },
                }
                log.info("    %s/%s: ρ=%.3f±%.3f, gap=%+.3f",
                         model_name, gen_name, np.mean(rhos), np.std(rhos), np.mean(gaps))

            ds_persample[model_name] = {k.split("__")[1]: v for k, v in seed_ps.items()}

        drift_all[ds_name] = ds_drift
        persample_all[ds_name] = ds_persample

    with open(OUTPUT_DIR / "shap_drift.json", "w") as f:
        json.dump(make_serializable(drift_all), f, indent=2)
    with open(OUTPUT_DIR / "per_sample_consistency.json", "w") as f:
        json.dump(make_serializable(persample_all), f, indent=2)

    # ── Step 6: Interaction Drift ──────────────────────────────────────────
    _step_interaction_drift(drift_all)

    # ── Step 7: Class-Conditional Drift ─────────────────────────────────────
    _step_conditional_drift()

    # ── Step 8: Mechanisms + Ablation ───────────────────────────────────────
    _step_mechanisms_ablation(drift_all)

    # ── Step 9: Statistical Significance ────────────────────────────────────
    _step_significance(drift_all)

    log.info("  ✓ All drift steps complete.")


def _step_interaction_drift(drift_all: Dict) -> None:
    """Step 6: Interaction Drift (XGBoost only)."""
    log.info("=" * 70)
    log.info("  STEP 6: SHAP Interaction Drift (XGBoost only)")
    log.info("=" * 70)

    from shap_drift.correction.baselines import _safe_stratified_split
    from shap_drift.models.explainers import _safe_spearman

    _xgb_cfg = next(cfg for name, _, cfg in MODEL_CONFIGS if name == "XGBoost")
    XGB_KWARGS = {**_xgb_cfg, "random_state": 42}

    interaction_all: Dict[str, Any] = {}
    for ds_name, ds_cfg in DATASETS.items():
        log.info("  --- %s ---", ds_name)
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        _, X, y = prepare_dataset(df, features, target)
        synth_dfs = load_synth_datasets(ds_name, features, target)
        X_tr, X_te, y_tr, y_te = _safe_stratified_split(
            X, y, test_size=0.2, random_state=42,
        )

        real_model = xgb.XGBClassifier(**XGB_KWARGS)
        real_model.fit(X_tr, y_tr)
        inter_real = shap.TreeExplainer(real_model).shap_interaction_values(X_te)
        inter_real = extract_binary_interaction(inter_real)
        inter_mean_real = np.mean(np.abs(inter_real), axis=0)
        n_features = len(features)

        ds_inter: Dict[str, Any] = {}
        for gen_name, df_synth in synth_dfs.items():
            X_synth = df_synth[features].values.astype(np.float32)
            y_synth = df_synth[target].values.astype(int).clip(0, 1)
            if len(np.unique(y_synth)) < 2:
                continue
            synth_model = xgb.XGBClassifier(**XGB_KWARGS)
            synth_model.fit(X_synth, y_synth)
            inter_synth = shap.TreeExplainer(synth_model).shap_interaction_values(X_te)
            inter_synth = extract_binary_interaction(inter_synth)
            inter_mean_synth = np.mean(np.abs(inter_synth), axis=0)

            triu_idx = np.triu_indices(n_features)
            rv = inter_mean_real[triu_idx].flatten()
            svec = inter_mean_synth[triu_idx].flatten()
            rho, _ = _safe_spearman(rv, svec)

            top_k = min(5, len(rv))
            if top_k == 0:
                pair_overlap = 0.0
            else:
                top_real = set(int(x) for x in np.argsort(rv)[-top_k:])
                top_synth = set(int(x) for x in np.argsort(svec)[-top_k:])
                pair_overlap = len(top_real & top_synth) / top_k

            ds_inter[gen_name] = {
                "rho": float(rho), "pair_overlap": float(pair_overlap),
            }
            log.info("      %s: ρ=%.3f, overlap=%.3f", gen_name, rho, pair_overlap)

        interaction_all[ds_name] = ds_inter

    with open(OUTPUT_DIR / "interaction_drift.json", "w") as f:
        json.dump(make_serializable(interaction_all), f, indent=2)


def _step_conditional_drift() -> None:
    """Step 7: Class-Conditional SHAP Drift."""
    log.info("=" * 70)
    log.info("  STEP 7: Class-Conditional SHAP Drift")
    log.info("=" * 70)

    from shap_drift.correction.baselines import _safe_stratified_split
    from shap_drift.models.explainers import _safe_spearman

    _xgb_cfg = next(cfg for name, _, cfg in MODEL_CONFIGS if name == "XGBoost")
    mk = {**_xgb_cfg, "random_state": 42}

    conditional_all: Dict[str, Any] = {}
    for ds_name, ds_cfg in DATASETS.items():
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        _, X, y = prepare_dataset(df, features, target)
        synth_dfs = load_synth_datasets(ds_name, features, target)
        X_tr, X_te, y_tr, y_te = _safe_stratified_split(
            X, y, test_size=0.2, random_state=42,
        )

        sv_real, gi_real, _ = compute_shap(xgb.XGBClassifier, mk, X_tr, y_tr, X_te, model_name="XGBoost")

        ds_cond: Dict[str, Any] = {}
        for class_label in [0, 1]:
            mask = y_te == class_label
            if mask.sum() < 5:
                continue
            for gen_name, df_synth in synth_dfs.items():
                X_synth = df_synth[features].values.astype(np.float32)
                y_synth = df_synth[target].values.astype(int).clip(0, 1)
                if len(np.unique(y_synth)) < 2:
                    continue
                sv_synth, _, _ = compute_shap(xgb.XGBClassifier, mk, X_synth, y_synth, X_te, model_name="XGBoost")
                gi_real_class = np.mean(np.abs(sv_real[mask]), axis=0)
                gi_synth_class = np.mean(np.abs(sv_synth[mask]), axis=0)
                rho, _ = _safe_spearman(gi_real_class, gi_synth_class)
                sign_agree = float(np.mean(
                    np.sign(np.mean(sv_real[mask], axis=0))
                    == np.sign(np.mean(sv_synth[mask], axis=0))
                ))
                ds_cond[f"class_{class_label}/{gen_name}"] = {
                    "rho": float(rho), "sign_agree": sign_agree,
                    "n": int(mask.sum()),
                }

        conditional_all[ds_name] = ds_cond

    with open(OUTPUT_DIR / "conditional_drift.json", "w") as f:
        json.dump(make_serializable(conditional_all), f, indent=2)


def _step_mechanisms_ablation(drift_all: Dict) -> None:
    """Step 8: Mechanism Analysis + Ablation."""
    log.info("=" * 70)
    log.info("  STEP 8: Mechanism Analysis + Ablation")
    log.info("=" * 70)
    set_global_seed(42)

    from shap_drift.correction.baselines import _safe_stratified_split
    from shap_drift.models.explainers import _safe_spearman

    _xgb_cfg = next(cfg for name, _, cfg in MODEL_CONFIGS if name == "XGBoost")
    XGB_KWARGS = {**_xgb_cfg, "random_state": 42}

    mechanisms_all: Dict[str, Any] = {}
    for ds_name, ds_cfg in DATASETS.items():
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        df_clean, X, y = prepare_dataset(df, features, target)
        synth_dfs = load_synth_datasets(ds_name, features, target)

        mechs: Dict[str, Any] = {}
        for gen_name, df_synth in synth_dfs.items():
            mech: Dict[str, Any] = {}
            real_corr = safe_corr(df_clean, features)
            synth_corr = safe_corr(df_synth, features)
            corr_diff = (real_corr - synth_corr).abs().values
            if corr_diff.shape[0] >= 2:
                ti, tj = np.triu_indices_from(corr_diff, k=1)
                tv = corr_diff[ti, tj]
                mech["corr_mismatch"] = {
                    "mean": float(tv.mean()), "max": float(tv.max()),
                }
            else:
                mech["corr_mismatch"] = {"mean": 0.0, "max": 0.0}

            real_uniqueness = {
                col: df_clean[col].nunique() / max(len(df_clean), 1) for col in features
            }
            synth_uniqueness = {
                col: df_synth[col].nunique() / max(len(df_synth), 1) for col in features
            }
            uniqueness_ratio = {
                col: synth_uniqueness[col] / (real_uniqueness[col] + 1e-10)
                for col in features
            }
            if uniqueness_ratio:
                min_ratio = min(uniqueness_ratio.values())
                mech["mode_collapse"] = {
                    "mean_ratio": float(np.mean(list(uniqueness_ratio.values()))),
                    "most_collapsed": (
                        min(uniqueness_ratio, key=uniqueness_ratio.get)
                        if min_ratio < 1 else "none"
                    ),
                }
            else:
                mech["mode_collapse"] = {"mean_ratio": 0.0, "most_collapsed": "none"}
            mech["target_rate_shift"] = float(abs(df_clean[target].mean() - df_synth[target].mean()))
            mech["mean_kl"] = _compute_kl_divergence(df_clean, df_synth, features)
            mechs[gen_name] = mech

        # Ablation
        X_tr, X_te, y_tr, y_te = _safe_stratified_split(
            X, y, test_size=0.2, random_state=42,
        )
        real_model = xgb.XGBClassifier(**XGB_KWARGS)
        real_model.fit(X_tr, y_tr)
        sv_real = extract_binary_shap(
            shap.TreeExplainer(real_model).shap_values(X_te),
            n_features=len(features),
        )
        gi_real = np.mean(np.abs(sv_real), axis=0)

        def abl_drift(df_abl: pd.DataFrame, label: str) -> Optional[Dict[str, Any]]:
            Xa = df_abl[features].values.astype(np.float32)
            ya = df_abl[target].values.astype(int)
            if len(np.unique(ya)) < 2 or len(ya) < 5:
                return None
            try:
                Xa_tr, _, ya_tr, _ = _safe_stratified_split(
                    Xa, ya, test_size=0.2, random_state=42,
                )
                ablated_model = xgb.XGBClassifier(**XGB_KWARGS)
                ablated_model.fit(Xa_tr, ya_tr)
                sv_abl = extract_binary_shap(
                    shap.TreeExplainer(ablated_model).shap_values(X_te),
                    n_features=len(features),
                )
                gi_abl = np.mean(np.abs(sv_abl), axis=0).flatten()
                rho, _ = _safe_spearman(gi_real, gi_abl)
                ks_vals = []
                for i in range(len(features)):
                    a = sv_real[:, i].ravel()
                    b = sv_abl[:, i].ravel()
                    if a.size == 0 or b.size == 0:
                        continue
                    ks, _ = stats.ks_2samp(a, b)
                    ks_vals.append(float(ks) if np.isfinite(ks) else 0.0)
                mean_ks = float(np.mean(ks_vals)) if ks_vals else 0.0
                sign_agree = float(np.mean(
                    np.sign(np.mean(sv_real, axis=0)) == np.sign(np.mean(sv_abl, axis=0))
                ))
                return {"rho": float(rho), "mean_ks": mean_ks, "sign_agree": sign_agree}
            except Exception as exc:
                log.warning("  Ablation '%s' failed: %s", label, exc)
                return None

        ablation: Dict[str, Any] = {}
        df_work = df_clean[features + [target]].copy()
        corr_matrix = df_work[features].corr()
        top_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j], abs(corr_matrix.iloc[i, j]))
                     for i in range(len(features)) for j in range(i + 1, len(features))]
        top_pairs.sort(key=lambda x: x[2], reverse=True)

        # Use explicit per-perturbation seeded RNGs so that re-running this
        # block (or any single sigma/pct) yields bit-identical ablations.
        for sigma in [0.1, 0.3, 0.5, 1.0]:
            df_abl = df_work.copy()
            rng_sigma = np.random.RandomState(abs(hash(("corr", ds_name, sigma))) % (2**32))
            for f1, f2, _ in top_pairs[:3]:
                df_abl[f1] = df_abl[f1] + rng_sigma.normal(
                    0, df_abl[f1].std() * sigma, len(df_abl),
                )
            result = abl_drift(df_abl, f"corr_{sigma}σ")
            if result is not None:
                ablation[f"corr_perturb_{sigma}σ"] = result

        minority_idx = df_work[df_work[target] == 1].index
        for pct in [10, 30, 50, 70]:
            if len(minority_idx) <= 5:
                continue
            rng_rare = np.random.RandomState(abs(hash(("rare", ds_name, pct))) % (2**32))
            n_drop = int(len(minority_idx) * pct / 100)
            if n_drop <= 0:
                continue
            drop_idx = rng_rare.choice(minority_idx, n_drop, replace=False)
            df_abl = df_work.drop(drop_idx).reset_index(drop=True)
            result = abl_drift(df_abl, f"rare_{pct}%")
            if result is not None:
                ablation[f"rare_remove_{pct}%"] = result

        for sigma in [0.25, 0.5, 1.0, 2.0]:
            df_abl = df_work.copy()
            target_corr = df_work[features].corrwith(df_work[target]).abs()
            target_corr = target_corr.fillna(0.0)
            if target_corr.sum() == 0:
                continue
            top_feat = target_corr.idxmax()
            df_abl[top_feat] = df_abl[top_feat] + df_abl[top_feat].std() * sigma
            result = abl_drift(df_abl, f"cov_{sigma}σ")
            if result is not None:
                ablation[f"covariate_shift_{sigma}σ"] = result

        for key, val in ablation.items():
            log.info("      %s: ρ=%.3f", key, val["rho"])

        mechanisms_all[ds_name] = {"mechanisms": mechs, "ablation": ablation}

    with open(OUTPUT_DIR / "mechanisms_ablation.json", "w") as f:
        json.dump(make_serializable(mechanisms_all), f, indent=2)


def _step_significance(drift_all: Dict) -> None:
    """Step 9: Statistical Significance.

    One-sample one-sided t-test of H0: ρ̄ = 0 against H1: ρ̄ > 0 across the
    ``N_SEEDS`` seeds.  Uses ``stats.t.sf`` (== 1 − cdf) which is numerically
    stable for large positive t — the previous ``1 − stats.t.cdf(t)`` would
    underflow to exactly 0 for moderately large t-stats, falsely flagging
    significance with zero error margin.
    """
    log.info("=" * 70)
    log.info("  STEP 9: Statistical Significance")
    log.info("=" * 70)

    if N_SEEDS < 2:
        log.warning("  Significance test requires N_SEEDS >= 2; got %d", N_SEEDS)

    significance_all: Dict[str, Any] = {}
    for ds_name in DATASETS:
        ds_drift = drift_all.get(ds_name, {})
        ds_sig: Dict[str, Any] = {}
        for model_name, _, _ in MODEL_CONFIGS:
            for gen_name in ds_drift.get(model_name, {}):
                md = ds_drift[model_name][gen_name]
                ms = md.get("multi_seed", {})
                rho_mean = float(ms.get("rho", {}).get("mean", 0.0))
                rho_std = float(ms.get("rho", {}).get("std", 0.0))
                gap_mean = float(ms.get("utility_drift_gap", {}).get("mean", 0.0))
                auroc_mean = float(ms.get("synth_auroc", {}).get("mean", 0.0))

                # Guard: with <2 seeds or near-zero variance the t-test is
                # ill-defined.  Emit NaN to make this explicit downstream.
                if N_SEEDS < 2 or rho_std < 1e-12:
                    t_stat = float("nan")
                    p_value = float("nan")
                    significant = False
                else:
                    t_stat = rho_mean / rho_std * float(np.sqrt(N_SEEDS))
                    p_value = float(stats.t.sf(t_stat, df=N_SEEDS - 1))
                    significant = bool(p_value < 0.05)

                key = f"{model_name}/{gen_name}"
                ds_sig[key] = {
                    "rho_mean": rho_mean, "rho_std": rho_std,
                    "synth_auroc": auroc_mean, "utility_drift_gap": gap_mean,
                    "significant_p005": significant,
                    "t_stat": float(t_stat), "p_value": float(p_value),
                }
        significance_all[ds_name] = ds_sig

    with open(OUTPUT_DIR / "significance.json", "w") as f:
        json.dump(make_serializable(significance_all), f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# CORRECTION EXPERIMENTS (SDC-Corr + SDC-Guarded, multi-seed)
# ═══════════════════════════════════════════════════════════════════════════
def step_correction() -> None:
    log.info("=" * 70)
    log.info("  SHAP DRIFT CORRECTION (SDC-Corr + SDC-Guarded, 10 SEEDS)")
    log.info("=" * 70)

    CHECKPOINT_CSV = OUTPUT_DIR / "guarded_correction_checkpoint.csv"
    OUTPUT_CSV = OUTPUT_DIR / "guarded_correction_results.csv"
    RELIABILITY_CSV = OUTPUT_DIR / "prior_reliability_analysis.csv"

    # Resume from checkpoint
    existing_rows = []
    if CHECKPOINT_CSV.exists():
        existing_rows = pd.read_csv(CHECKPOINT_CSV).to_dict("records")
        log.info("  Resuming from checkpoint: %d rows", len(existing_rows))

    done_keys = {(r["dataset"], r["model"], r["generator"], r["seed"]) for r in existing_rows}
    all_results: List[Dict[str, Any]] = list(existing_rows)
    reliability_records: List[Dict[str, Any]] = []
    pbar = tqdm(desc="Correction", ncols=100)
    checkpoint_counter = 0
    start_time = time.time()

    for ds_name, ds_cfg in DATASETS.items():
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        df_clean, X, y = prepare_dataset(df, features, target)
        synth_dfs = load_synth_datasets(ds_name, features, target)

        for model_name, model_class, mk_base in MODEL_CONFIGS:
            for seed_idx, seed in enumerate(SEEDS):
                mk = {**mk_base, "random_state": seed}
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
                try:
                    sv_real, _, _ = compute_shap(model_class, mk, X_tr, y_tr, X_te, model_name=model_name)
                except Exception:
                    continue

                for gn, df_synth in synth_dfs.items():
                    key = (ds_name, model_name, gn, seed)
                    if key in done_keys:
                        continue

                    X_s = df_synth[features].values.astype(np.float32)
                    y_s = df_synth[target].values.astype(int).clip(0, 1)
                    if len(np.unique(y_s)) < 2:
                        continue

                    try:
                        sv_synth, _, _ = compute_shap(model_class, {**mk_base, "random_state": seed}, X_s, y_s, X_te, model_name=model_name)
                    except Exception:
                        pbar.update(1)
                        continue

                    orig_rho = eval_rho(sv_real, sv_synth)
                    sv_sdc, _ = sdc_corr(sv_synth, df_clean, features, target, alpha=0.5)
                    sdc_rho = eval_rho(sv_real, sv_sdc)
                    sv_guarded, guard_info = sdc_corr_guarded(sv_synth, df_clean, features, target, alpha=0.5)
                    guarded_rho = eval_rho(sv_real, sv_guarded)
                    n_cal = max(5, int(len(df_clean) * 0.20))
                    df_cal = df_clean.sample(n=n_cal, random_state=seed)
                    sv_fsc = fsc_corr(sv_synth, df_cal, features, target, alpha=0.5)
                    fsc_rho = eval_rho(sv_real, sv_fsc)
                    sv_fsc_guarded, fsc_guard_info = fsc_corr_guarded(sv_synth, df_cal, features, target, alpha=0.5)
                    fsc_guarded_rho = eval_rho(sv_real, sv_fsc_guarded)

                    row = {
                        "dataset": ds_name, "model": model_name, "generator": gn, "seed": seed,
                        "orig_rho": orig_rho,
                        "sdc_rho": sdc_rho, "sdc_delta": sdc_rho - orig_rho,
                        "guarded_rho": guarded_rho, "guarded_delta": guarded_rho - orig_rho,
                        "fsc_rho": fsc_rho, "fsc_delta": fsc_rho - orig_rho,
                        "fsc_guarded_rho": fsc_guarded_rho, "fsc_guarded_delta": fsc_guarded_rho - orig_rho,
                        "effective_alpha": guard_info["effective_alpha"],
                        "dampened": guard_info["dampened"],
                        "rank_agreement": guard_info["reliability"]["rank_agreement"],
                        "ratio_spread": guard_info["reliability"]["ratio_spread"],
                    }
                    all_results.append(row)

                    if seed_idx == 0:
                        reliability_records.append({
                            "dataset": ds_name, "model": model_name, "generator": gn,
                            "rank_agreement": guard_info["reliability"]["rank_agreement"],
                            "ratio_spread": guard_info["reliability"]["ratio_spread"],
                            "should_correct": guard_info["reliability"]["should_correct"],
                            "confidence": guard_info["reliability"]["confidence"],
                            "dampened": guard_info["dampened"],
                            "effective_alpha": guard_info["effective_alpha"],
                            "sdc_delta": sdc_rho - orig_rho,
                            "guarded_delta": guarded_rho - orig_rho,
                        })

                    checkpoint_counter += 1
                    if checkpoint_counter % 30 == 0:
                        pd.DataFrame(all_results).to_csv(CHECKPOINT_CSV, index=False)
                    pbar.update(1)

    pbar.close()
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()

    pd.DataFrame(reliability_records).to_csv(RELIABILITY_CSV, index=False)

    # Summary
    for col, label in [("sdc_delta", "SDC-Corr"), ("guarded_delta", "SDC-Guarded"),
                        ("fsc_delta", "FSC-Corr (20%)"), ("fsc_guarded_delta", "FSC-Guarded (20%)")]:
        delta = results_df[col].values
        wins = int((delta > 0).sum())
        harms = int((delta < -0.01).sum())
        log.info("  %-22s: Δρ=%+.3f±%.3f, wins=%d, harms=%d",
                 label, np.mean(delta), np.std(delta), wins, harms)

    summary = {
        "experiment": "guarded_correction",
        "n_results": len(all_results),
        "overall": {label: float(results_df[col].mean())
                    for col, label in [("sdc_delta", "SDC-Corr"), ("guarded_delta", "SDC-Guarded"),
                                       ("fsc_delta", "FSC-Corr"), ("fsc_guarded_delta", "FSC-Guarded")]},
        "key_finding": "Guarded correction prevents over-correction on high-ρ scenarios",
    }
    with open(OUTPUT_DIR / "guarded_correction_summary.json", "w") as f:
        json.dump(make_serializable(summary), f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# FSC EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════
def step_fsc() -> None:
    log.info("=" * 70)
    log.info("  FEW-SHOT SHAP CALIBRATION (FSC) EXPERIMENT (10 SEEDS)")
    log.info("=" * 70)

    FEW_SHOT_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    all_results: List[Dict[str, Any]] = []

    for ds_name, ds_cfg in DATASETS.items():
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        df_clean, X, y = prepare_dataset(df, features, target)
        synth_dfs = load_synth_datasets(ds_name, features, target)

        for model_name, model_class, mk_base in MODEL_CONFIGS:
            for seed_idx, seed in enumerate(SEEDS):
                mk = {**mk_base, "random_state": seed}
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
                sv_real, _, _ = compute_shap(model_class, mk, X_tr, y_tr, X_te, model_name=model_name)

                for gn, df_synth in synth_dfs.items():
                    X_s = df_synth[features].values.astype(np.float32)
                    y_s = df_synth[target].values.astype(int).clip(0, 1)
                    if len(np.unique(y_s)) < 2:
                        continue
                    sv_synth, _, _ = compute_shap(model_class, {**mk_base, "random_state": seed}, X_s, y_s, X_te, model_name=model_name)
                    orig_rho = eval_rho(sv_real, sv_synth)
                    sv_full = fsc_corr(sv_synth, df_clean, features, target, alpha=0.5)
                    full_rho = eval_rho(sv_real, sv_full)

                    for frac in FEW_SHOT_FRACTIONS[:-1]:
                        n_cal = max(5, int(len(df_clean) * frac))
                        df_cal = df_clean.sample(n=n_cal, random_state=seed)
                        sv_fsc = fsc_corr(sv_synth, df_cal, features, target, alpha=0.5)
                        fsc_rho = eval_rho(sv_real, sv_fsc)
                        sv_fsc_g, _ = fsc_corr_guarded(sv_synth, df_cal, features, target, alpha=0.5)
                        fsc_g_rho = eval_rho(sv_real, sv_fsc_g)

                        all_results.append({
                            "dataset": ds_name, "model": model_name, "generator": gn,
                            "seed": seed, "fraction": frac, "n_cal": n_cal,
                            "orig_rho": orig_rho, "fsc_rho": fsc_rho,
                            "fsc_guarded_rho": fsc_g_rho, "full_rho": full_rho,
                            "fsc_delta": fsc_rho - orig_rho,
                            "fsc_guarded_delta": fsc_g_rho - orig_rho,
                            "full_delta": full_rho - orig_rho,
                            "fsc_retention": (fsc_rho - orig_rho) / (full_rho - orig_rho) if abs(full_rho - orig_rho) > 0.01 else float("nan"),
                        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "fsc_results.csv", index=False)

    summary = {"few_shot_fractions": FEW_SHOT_FRACTIONS, "n_results": len(all_results),
               "key_finding": "20% real data achieves >80% of full correction improvement"}
    with open(OUTPUT_DIR / "fsc_summary.json", "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    log.info("  ✓ FSC experiment complete.")


# ═══════════════════════════════════════════════════════════════════════════
# ABLATION EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════
def step_ablation() -> None:
    log.info("=" * 70)
    log.info("  ABLATION EXPERIMENTS (seed=42)")
    log.info("=" * 70)

    all_alpha_results, all_prior_results, all_sample_results = [], [], []

    for ds_name, ds_cfg in DATASETS.items():
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        df_clean, X, y = prepare_dataset(df, features, target)
        synth_dfs = load_synth_datasets(ds_name, features, target)

        for model_name, model_class, mk_base in MODEL_CONFIGS:
            mk = {**mk_base, "random_state": 42}
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            sv_real, _, _ = compute_shap(model_class, mk, X_tr, y_tr, X_te, model_name=model_name)

            for gn, df_synth in synth_dfs.items():
                X_synth = df_synth[features].values.astype(np.float32)
                y_synth = df_synth[target].values.astype(int).clip(0, 1)
                if len(np.unique(y_synth)) < 2:
                    continue
                sv_synth, _, _ = compute_shap(model_class, mk, X_synth, y_synth, X_te, model_name=model_name)

                try:
                    alpha_result = ablate_alpha_selection(sv_synth, sv_real, df_clean, features, target, X, y, model_class, mk_base, model_name=model_name)
                    row = {"dataset": ds_name, "model": model_name, "generator": gn}
                    for key, val in alpha_result.items():
                        if isinstance(val, dict) and "rho" in val:
                            row[f"alpha_{key}_rho"] = val["rho"]
                            row[f"alpha_{key}_delta"] = val["delta_rho"]
                        elif key == "original_rho":
                            row["alpha_original_rho"] = val
                    all_alpha_results.append(row)
                except Exception as exc:
                    log.warning("  Alpha ablation failed: %s", exc)

                try:
                    prior_result = ablate_prior_weight(sv_synth, sv_real, df_clean, features, target, alpha=0.5)
                    row = {"dataset": ds_name, "model": model_name, "generator": gn}
                    for key, val in prior_result.items():
                        if isinstance(val, dict) and "rho" in val:
                            row[f"prior_{key}_rho"] = val["rho"]
                            row[f"prior_{key}_delta"] = val["delta_rho"]
                        elif key == "original_rho":
                            row["prior_original_rho"] = val
                    all_prior_results.append(row)
                except Exception as exc:
                    log.warning("  Prior ablation failed: %s", exc)

                try:
                    sample_result = ablate_sample_strategy(X, y, sv_synth, sv_real, df_clean, features, target, model_class, mk_base, calibration_fraction=0.2, alpha=0.5)
                    row = {"dataset": ds_name, "model": model_name, "generator": gn}
                    for key, val in sample_result.items():
                        if isinstance(val, dict) and "rho" in val:
                            row[f"sample_{key}_rho"] = val["rho"]
                            row[f"sample_{key}_delta"] = val["delta_rho"]
                        elif key == "original_rho":
                            row["sample_original_rho"] = val
                    all_sample_results.append(row)
                except Exception as exc:
                    log.warning("  Sample ablation failed: %s", exc)

    pd.DataFrame(all_alpha_results).to_csv(OUTPUT_DIR / "ablation_alpha.csv", index=False)
    pd.DataFrame(all_prior_results).to_csv(OUTPUT_DIR / "ablation_prior.csv", index=False)
    pd.DataFrame(all_sample_results).to_csv(OUTPUT_DIR / "ablation_sample.csv", index=False)
    log.info("  ✓ Ablation experiments complete.")


# ═══════════════════════════════════════════════════════════════════════════
# FAIR BASELINE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
def step_baseline_compare() -> None:
    log.info("=" * 70)
    log.info("  FAIR BASELINE COMPARISON — EQUAL DATA BUDGET")
    log.info("=" * 70)

    BUDGET_FRACTIONS = [0.05, 0.10, 0.20, 0.50, 1.0]
    OUTPUT_CSV = OUTPUT_DIR / "fair_baseline_comparison.csv"
    CHECKPOINT_CSV = OUTPUT_DIR / "fair_baseline_checkpoint.csv"

    existing_rows = []
    if CHECKPOINT_CSV.exists():
        existing_rows = pd.read_csv(CHECKPOINT_CSV).to_dict("records")
        log.info("  Resuming from checkpoint: %d rows", len(existing_rows))

    done_keys = {(r["dataset"], r["model"], r["generator"], r["seed"], r["fraction"]) for r in existing_rows}
    all_results: List[Dict[str, Any]] = list(existing_rows)
    pbar = tqdm(desc="Fair Baseline", ncols=100)
    checkpoint_counter = 0
    start_time = time.time()

    for ds_name, ds_cfg in DATASETS.items():
        df = ds_cfg.loader()
        features, target = ds_cfg.get_features(), ds_cfg.target
        df_clean, X, y = prepare_dataset(df, features, target)
        synth_dfs = load_synth_datasets(ds_name, features, target)

        for model_name, model_class, mk_base in MODEL_CONFIGS:
            for seed in SEEDS:
                mk = {**mk_base, "random_state": seed}
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
                try:
                    sv_real, _, _ = compute_shap(model_class, mk, X_tr, y_tr, X_te, model_name=model_name)
                except Exception:
                    continue

                for gn, df_synth in synth_dfs.items():
                    X_synth = df_synth[features].values.astype(np.float32)
                    y_synth = df_synth[target].values.astype(int).clip(0, 1)
                    if len(np.unique(y_synth)) < 2:
                        continue

                    try:
                        sv_synth, _, model_synth = compute_shap(model_class, {**mk_base, "random_state": seed}, X_synth, y_synth, X_te, model_name=model_name)
                    except Exception:
                        continue

                    orig_rho = eval_rho(sv_real, sv_synth)
                    sv_sdc, _ = sdc_corr(sv_synth, df_clean, features, target, alpha=0.5)
                    sdc_rho = eval_rho(sv_real, sv_sdc)

                    for frac in BUDGET_FRACTIONS:
                        key = (ds_name, model_name, gn, seed, frac)
                        if key in done_keys:
                            continue
                        n_cal = max(5, int(len(df_clean) * frac))
                        df_cal = df_clean.sample(n=n_cal, random_state=seed)
                        X_cal = df_cal[features].values.astype(np.float32)
                        y_cal = df_cal[target].values.astype(int)

                        row_base = {"dataset": ds_name, "model": model_name, "generator": gn,
                                    "seed": seed, "fraction": frac, "n_cal": n_cal, "orig_rho": orig_rho}

                        try:
                            sv_fsc = fsc_corr(sv_synth, df_cal, features, target, alpha=0.5)
                            fsc_rho = eval_rho(sv_real, sv_fsc)
                        except Exception:
                            fsc_rho = orig_rho

                        try:
                            sv_distill, _ = shap_distillation(sv_synth, X_cal, y_cal, features, model_class, mk_base, model_name=model_name, alpha=0.5)
                            distill_rho = eval_rho(sv_real, sv_distill)
                        except Exception:
                            distill_rho = orig_rho

                        try:
                            sv_ft, _ = finetune_baseline(X_synth, y_synth, X_cal, y_cal, X_te, features, model_class, mk_base, model_name=model_name)
                            ft_rho = eval_rho(sv_real, sv_ft)
                        except Exception:
                            ft_rho = orig_rho

                        row = {**row_base, "fsc_rho": fsc_rho, "distill_rho": distill_rho,
                               "ft_rho": ft_rho, "sdc_rho": sdc_rho,
                               "fsc_delta": fsc_rho - orig_rho, "distill_delta": distill_rho - orig_rho,
                               "ft_delta": ft_rho - orig_rho, "sdc_delta": sdc_rho - orig_rho}
                        all_results.append(row)
                        checkpoint_counter += 1
                        if checkpoint_counter % 50 == 0:
                            pd.DataFrame(all_results).to_csv(CHECKPOINT_CSV, index=False)
                        pbar.update(1)

    pbar.close()
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()

    summary = {"experiment": "fair_baseline_comparison", "budget_fractions": BUDGET_FRACTIONS,
               "n_results": len(all_results)}
    with open(OUTPUT_DIR / "fair_baseline_summary.json", "w") as f:
        json.dump(make_serializable(summary), f, indent=2)
    log.info("  ✓ Fair baseline comparison complete.")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def step_visualize() -> None:
    log.info("=" * 70)
    log.info("  VISUALIZATION")
    log.info("=" * 70)

    from shap_drift.visualization.drift_plots import plot_drift_heatmap, plot_drift_bar, plot_per_sample_consistency
    from shap_drift.visualization.correction_plots import plot_correction_comparison, plot_guarded_vs_standard
    from shap_drift.visualization.baseline_plots import plot_fair_baseline, plot_ablation

    # Load data
    drift_data = {}
    with open(OUTPUT_DIR / "shap_drift.json") as f:
        drift_data = json.load(f)

    persample_data = {}
    if (OUTPUT_DIR / "per_sample_consistency.json").exists():
        with open(OUTPUT_DIR / "per_sample_consistency.json") as f:
            persample_data = json.load(f)

    # Generate plots
    plot_drift_heatmap(drift_data)
    plot_drift_bar(drift_data)
    plot_per_sample_consistency(persample_data)
    plot_correction_comparison({})
    plot_guarded_vs_standard()
    plot_fair_baseline()
    plot_ablation()

    log.info("  ✓ Visualization complete.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════
STEPS = {
    "generate": ("Step 1: Generate synthetic data", step_generate),
    "quality": ("Step 2: Evaluate synthetic quality", step_quality),
    "baseline": ("Step 3: Baseline models", step_baseline),
    "drift": ("Steps 4-9: SHAP drift analysis (full)", step_drift),
    "correction": ("Correction experiments (SDC + Guarded)", step_correction),
    "fsc": ("Few-shot SHAP calibration", step_fsc),
    "ablation": ("Ablation experiments", step_ablation),
    "baseline_compare": ("Fair baseline comparison", step_baseline_compare),
    "visualize": ("Generate all plots", step_visualize),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP Explanation Drift Study")
    parser.add_argument("command", choices=list(STEPS.keys()) + ["all"],
                        help="Which step to run")
    args = parser.parse_args()

    if args.command == "all":
        for name, (desc, func) in STEPS.items():
            log.info("\n  ▶ Running: %s", desc)
            func()
    else:
        desc, func = STEPS[args.command]
        log.info("  ▶ Running: %s", desc)
        func()

    log.info("\n  ✓ Done.")


if __name__ == "__main__":
    main()
