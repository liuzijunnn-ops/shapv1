#!/usr/bin/env bash
# ===========================================================================
# 服务器部署脚本 — 一键安装并验证 shap-drift 项目
# ===========================================================================
#
# 用法:
#   bash deploy_server.sh              # 完整安装（conda + 依赖 + 验证）
#   bash deploy_server.sh --skip-conda # 跳过 conda 环境创建
#
# 最低要求: conda 或 pip, Python 3.10+
# ===========================================================================

set -euo pipefail

SKIP_CONDA=false
for arg in "$@"; do
    case $arg in
        --skip-conda) SKIP_CONDA=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  SHAP-Drift 服务器部署"
echo "  $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. 创建 conda 环境
# ---------------------------------------------------------------------------
if [ "$SKIP_CONDA" = false ]; then
    echo ""
    echo ">>> 创建 conda 环境 'shap-drift'..."
    if conda env list | grep -q "shap-drift"; then
        echo "  环境 'shap-drift' 已存在，跳过创建"
    else
        conda env create -f environment.yml
        echo "  环境创建完成"
    fi
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate shap-drift 2>/dev/null || {
    echo "ERROR: 无法激活 conda 环境 'shap-drift'"
    exit 1
}
echo "  Python: $(python3 --version)"

# ---------------------------------------------------------------------------
# 2. 安装项目包 (editable mode)
# ---------------------------------------------------------------------------
echo ""
echo ">>> 安装 shap-drift 包..."
pip install -e . --no-deps 2>/dev/null || {
    # 如果没有 pyproject.toml 支持，手动添加到 PYTHONPATH
    echo "  pyproject.toml 安装跳过，使用 PYTHONPATH"
}
echo "  安装完成"

# ---------------------------------------------------------------------------
# 3. 验证导入
# ---------------------------------------------------------------------------
echo ""
echo ">>> 验证模块导入..."
python3 -c "
from shap_drift.config import SEEDS, GENERATORS, detect_cuda
from shap_drift.datasets import DATASETS, prepare_dataset
from shap_drift.models import MODEL_CONFIGS
from shap_drift.correction import sdc_corr, sdc_corr_guarded, fsc_corr, fsc_corr_guarded
from shap_drift.metrics import drift_metrics, evaluate_quality
from shap_drift.ablation import ablate_alpha_selection

print(f'  数据集: {list(DATASETS.keys())}')
print(f'  生成器: {[g[0] for g in GENERATORS]}')
print(f'  模型: {[m[0] for m in MODEL_CONFIGS]}')
print(f'  种子数: {len(SEEDS)}')

# 验证 Injury1798
df = DATASETS['Injury1798'].loader()
print(f'  Injury1798: {len(df)} samples OK')

cuda, info = detect_cuda()
print(f'  CUDA: {info}')
print('  ✓ 所有模块导入成功')
"

# ---------------------------------------------------------------------------
# 4. 验证数据文件
# ---------------------------------------------------------------------------
echo ""
echo ">>> 验证数据文件..."
for ds_dir in CreditCardFraud heart+disease 1798 CaliforniaHousing "statlog+german+credit+data" adult; do
    if [ -d "dataset/$ds_dir" ]; then
        echo "  ✓ dataset/$ds_dir"
    else
        echo "  ✗ dataset/$ds_dir — MISSING"
    fi
done

# ---------------------------------------------------------------------------
# 5. 验证 run.py
# ---------------------------------------------------------------------------
echo ""
echo ">>> 验证 run.py CLI..."
python3 run.py --help > /dev/null 2>&1 && echo "  ✓ run.py CLI 正常" || echo "  ✗ run.py CLI 失败"

# ---------------------------------------------------------------------------
# 完成
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  部署完成！"
echo ""
echo "  运行实验:"
echo "    bash run_all.sh              # 全部实验"
echo "    python3 run.py generate      # 单步运行"
echo "    python3 run.py drift         # SHAP 漂移分析"
echo "    python3 run.py --help        # 查看所有命令"
echo "============================================================"
