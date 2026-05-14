"""Correction methods for SHAP drift.

Core methods (4):
  - SDC-Corr: Correlation-transfer correction (baseline)
  - SDC-Guarded: Adaptive correction with prior reliability guard
  - FSC-Corr: Few-shot SHAP calibration
  - FSC-Guarded: Few-shot calibration with reliability guard
"""
from __future__ import annotations

from shap_drift.correction.sdc import sdc_corr, sdc_corr_guarded, estimate_prior_reliability
from shap_drift.correction.fsc import fsc_corr, fsc_corr_guarded
from shap_drift.correction.baselines import shap_distillation, finetune_baseline, coral_baseline
from shap_drift.correction.sadc import sadc_corr, sadc_default

__all__ = [
    "sdc_corr", "sdc_corr_guarded", "estimate_prior_reliability",
    "fsc_corr", "fsc_corr_guarded",
    "shap_distillation", "finetune_baseline", "coral_baseline",
    "sadc_corr", "sadc_default",
]
