"""Case studies — domain-grounded illustrations of SHAP drift correction.

Two scenarios are implemented:

  * **Medical**: Pima Diabetes & Heart Disease.  Simulates a physician
    inspecting top SHAP risk factors before deciding on intervention.
    Drifted explanations could mislead clinical decisions; SADC's
    corrected ranking is shown side-by-side.

  * **Financial**: German Credit.  Simulates a loan officer who must
    explain a credit denial to a customer.  Drifted feature attributions
    can lead to discriminatory rationale; we show how SADC realigns the
    top factors with the real-data rationale.

Both scenarios emit per-sample explanation cards (see ``explanation_card``)
and aggregate ranking deltas (``ranking_delta``).  Output is an HTML
dashboard at ``results/case_study.html``.
"""
from __future__ import annotations

from shap_drift.case_study.runner import run_case_studies

__all__ = ["run_case_studies"]
