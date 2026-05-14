#!/usr/bin/env bash
# ===========================================================================
# SHAP Explanation Drift Study — Full Experiment Pipeline (v8)
# ===========================================================================
# Unified CLI via run.py
#
# Usage:
#   bash run_all.sh              # Run everything
#   bash run_all.sh --skip-gen   # Skip synthetic data generation (if already done)
#   bash run_all.sh --drift-only # Only run SHAP drift analysis
#   bash run_all.sh --viz-only   # Only run visualization scripts
# ===========================================================================

set -euo pipefail

# Clear Python bytecode cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1

# Auto-activate conda environment if needed
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "shap-drift" ]; then
    echo "Activating conda environment 'shap-drift'..."
    eval "$(conda shell.bash hook)"
    conda activate shap-drift 2>/dev/null || {
        echo "ERROR: conda environment 'shap-drift' not found."
        echo "Run 'bash setup_server.sh' first to create it."
        exit 1
    }
fi
echo "Using conda env: $CONDA_DEFAULT_ENV (Python $(python3 --version 2>&1 | cut -d' ' -f2))"

SKIP_GEN=false
VIZ_ONLY=false
DRIFT_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-gen)    SKIP_GEN=true ;;
        --viz-only)    VIZ_ONLY=true ;;
        --drift-only)  DRIFT_ONLY=true ;;
        *)             echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  SHAP Explanation Drift Study — Experiment Pipeline (v8)"
echo "  Started at: $(date)"
echo "============================================================"

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi failed)"
else
    echo "  WARNING: nvidia-smi not found — CUDA may not be available"
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 1: Synthetic Data Generation
# ---------------------------------------------------------------------------
if [ "$VIZ_ONLY" = false ]; then
    if [ "$SKIP_GEN" = false ]; then
        echo ">>> Phase 1: Generating synthetic data..."
        time python3 run.py generate 2>&1 | tee results/generate.log
    else
        echo ">>> Phase 1: SKIPPED (using existing synthetic data)"
    fi

    # ---------------------------------------------------------------------------
    # Phase 2: Quality + Baseline + Drift (Steps 2-9)
    # ---------------------------------------------------------------------------
    echo ""
    echo ">>> Phase 2: Quality evaluation + baseline models + SHAP drift..."
    time python3 run.py quality 2>&1 | tee -a results/pipeline.log
    time python3 run.py baseline 2>&1 | tee -a results/pipeline.log
    time python3 run.py drift 2>&1 | tee -a results/pipeline.log
    echo "  Pipeline complete."

    if [ "$DRIFT_ONLY" = true ]; then
        echo "  (--drift-only specified, skipping correction experiments)"
        exit 0
    fi

    # ---------------------------------------------------------------------------
    # Phase 3: Correction Experiments (SDC-Corr + SDC-Guarded)
    # ---------------------------------------------------------------------------
    echo ""
    echo ">>> Phase 3: Correction experiments (SDC-Corr + SDC-Guarded)..."
    time python3 run.py correction 2>&1 | tee results/correction.log

    # ---------------------------------------------------------------------------
    # Phase 4: FSC Experiments
    # ---------------------------------------------------------------------------
    echo ""
    echo ">>> Phase 4: FSC (few-shot calibration) experiments..."
    time python3 run.py fsc 2>&1 | tee results/fsc.log

    # ---------------------------------------------------------------------------
    # Phase 5: Ablation Experiments
    # ---------------------------------------------------------------------------
    echo ""
    echo ">>> Phase 5: Ablation experiments..."
    time python3 run.py ablation 2>&1 | tee results/ablation.log

    # ---------------------------------------------------------------------------
    # Phase 6: Fair Baseline Comparison
    # ---------------------------------------------------------------------------
    echo ""
    echo ">>> Phase 6: Fair baseline comparison (equal data budget)..."
    time python3 run.py baseline_compare 2>&1 | tee results/fair_baseline.log
fi

# ---------------------------------------------------------------------------
# Phase 7: Visualizations
# ---------------------------------------------------------------------------
echo ""
echo ">>> Phase 7: Generating visualizations..."
time python3 run.py visualize 2>&1 | tee results/viz.log
echo "  Visualizations complete."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished at: $(date)"
echo "============================================================"
echo ""
echo "  Results directory: results/"
echo "  Figures directory: results/figures/"
echo ""
echo "  Key output files:"
echo "    - results/guarded_correction_results.csv"
echo "    - results/fair_baseline_comparison.csv"
echo "    - results/prior_reliability_analysis.csv"
echo "    - results/fsc_results.csv"
echo "    - results/shap_drift.json"
