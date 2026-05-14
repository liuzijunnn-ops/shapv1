# CLAUDE.md — SHAP Explanation Drift Study (v0.3)

> 给协作 LLM 的项目导览。读完本文档应能：(a) 复现任意实验步骤；
> (b) 知道每个模块的职责与不变量；(c) 理解 v0.3 引入的新方法、新数据集
> 与新理论分析。

---

## 1. 项目一句话

研究**合成表格数据**在 **SHAP 解释** 上的"漂移（Explanation Drift）"现象，
并提出 / 评估 5 类修正方法（SDC、SDC-Guarded、FSC、FSC-Guarded、**SADC**），
覆盖 **10 个数据集 × 5 个生成器 × 3 个分类器 × 10 个随机种子**。

## 2. v0.3 新增内容速览

| 类别 | 新增 | 路径 |
| --- | --- | --- |
| 数据集 | Diabetes / CovType / Higgs / Thyroid（4 个）| `shap_drift/datasets/{diabetes,covtype,higgs,thyroid}.py` |
| 生成器 | TabDDPM（扩散）/ TabSyn（潜变量扩散）| `shap_drift/generators/diffusion.py` |
| 修正方法 | **SADC**（SHAP-Aware Distillation Calibration）| `shap_drift/correction/sadc.py` |
| 理论 | 5 条命题（最优性、no-harm、扩散律）| `docs/theory.md` |
| Case Study | 临床（Diabetes / HeartDisease）+ 金融（GermanCredit）| `shap_drift/case_study/` |

## 3. 关键术语

| 术语 | 含义 |
| --- | --- |
| **Drift ρ** | Spearman rank correlation between real and synth global SHAP|
| **Utility-Drift Gap** | `synth_AUROC − ρ`，衡量"高效用但低解释一致性" |
| **SDC-Corr** | 用真实数据特征-目标相关性作为先验，重加权合成 SHAP |
| **SDC-Guarded** | rank_agreement 超阈值时阻尼 α 以避免过修正 |
| **FSC-Corr** | SDC 的少样本版本：5%–50% 真实数据子集 |
| **SADC (NEW)** | 多源 prior（corr+MI+bootstrap teacher）+ per-feature 闭式 α + no-harm 投影 |

## 4. 顶层目录

```
p6/
├── run.py                # 唯一 CLI 入口（命令见 §6）
├── deploy_server.sh / run_all.sh
├── environment.yml / pyproject.toml
├── docs/
│   └── theory.md          # ★ v0.3 理论分析（5 条命题）
├── shap_drift/
│   ├── config.py         # 路径、种子、生成器/颜色注册、序列化
│   ├── datasets/         # 10 个数据集 loader + DATASETS 注册
│   │   ├── credit_card_fraud.py / heart_disease.py / injury_1798.py
│   │   ├── california_housing.py / german_credit.py / adult.py
│   │   └── diabetes.py / covtype.py / higgs.py / thyroid.py   ← NEW
│   ├── generators/       # GC / CTGAN / TVAE / TabDDPM / TabSyn
│   │   ├── __init__.py    # generate_one、_build_synthesizer 工厂
│   │   └── diffusion.py   ← NEW（reference TabDDPM + TabSyn）
│   ├── models/           # XGB / RF / MLP + SHAP 计算封装
│   ├── correction/       # sdc / fsc / baselines / sadc          ← NEW: sadc
│   ├── metrics/          # drift.py / quality.py
│   ├── ablation/         # alpha / prior / sample 三轴消融
│   ├── case_study/       ← NEW（临床 + 金融场景）
│   └── visualization/    # 全部图表输出 PDF
├── dataset/              # 原始数据（CSV/JSON），不在 git 中
└── results/              # 所有 JSON/CSV/NPZ/PDF/HTML 输出
```

## 5. 数据集矩阵

| 数据集 | 类型 | 样本数 | 特征数 | 不平衡度 | 主要用途 |
| --- | --- | --- | --- | --- | --- |
| CreditCardFraud | 欺诈检测 | ~3K (5:1 下采样) | 11 | 17% pos | 二分类基线 |
| HeartDisease | 临床二分类 | 303 | 13 | 46% pos | Case study |
| Injury1798 | 生物力学 | ~1.7K | 15 | 50% pos | 高维 + 真实业务 |
| CaliforniaHousing | 房价（连续→二值化）| 20K | 8 | 50% pos | 中等规模 |
| GermanCredit | 信贷决策 | 1K | 20 | 30% pos | Case study 金融 |
| Adult | 收入预测 | ~30K | 6 | 24% pos | 大规模 |
| **Diabetes** | 临床 | 768 | 8 | 35% pos | **小样本临床** |
| **CovType** | 林木类型（多→二）| 30K（下采样）| 10 | 49% pos | **多分类 → 二值** |
| **Higgs** | 物理 | 20K（下采样）| 28 | 47% pos | **高维近平衡** |
| **Thyroid** | 甲状腺疾病 | 9.2K | 8 | 26% pos | **极端不平衡** |

## 6. CLI 命令

`python run.py <step>` — 所有 step 的产物写入 `results/`，可重入（带 checkpoint）。

| 命令 | 作用 | 主要输出 |
| --- | --- | --- |
| `generate` | 用 5 个生成器各生成 1 份合成 CSV | `results/{ds}_{gc|ctgan|tvae|tabddpm|tabsyn}.csv` |
| `quality` | 评估合成质量（KS、W1、TSTR-AUROC）| `synth_quality.json` |
| `baseline` | 真实数据上 XGB/RF/MLP 的 AUROC | `baseline_results.json` |
| `drift` | SHAP 漂移主指标 + 显著性等 6 个子步骤 | `shap_drift.json` 等 6 个 JSON |
| `correction` | SDC/SDC-Guarded/FSC/FSC-Guarded/**SADC**（10 seeds）| `guarded_correction_results.csv` |
| `fsc` | FSC 在 6 个 calibration fraction 上扫掠 | `fsc_results.csv` |
| `ablation` | α / prior / sample 三轴消融 | `ablation_*.csv` |
| `baseline_compare` | Distill / Fine-tune / FSC / **SADC** 等数据预算 | `fair_baseline_comparison.csv` |
| `visualize` | 把上述结果绘制为 PDF | `results/figures/*.pdf` |
| `case_study` ★ | 生成临床+金融案例的 HTML 报告 | `results/case_study.html` |
| `all` | 依次执行上述全部 | — |

## 7. SADC 方法要点

**SADC**（SHAP-Aware Distillation Calibration）— 论文招牌方法，
在 `shap_drift/correction/sadc.py`。三大创新：

1. **多源 prior 融合** — correlation + MI + bootstrap teacher SHAP，
   权重由 teacher CV (coefficient of variation) 通过 σ 函数自动决定。
2. **Per-feature 闭式 α** — 不是全局标量；每特征独立 scale，公式见
   `_per_feature_scale`。
3. **No-harm 投影** — 用 calibration set 的 20% 评估每特征的修正效果，
   伤害到的特征自动回滚（理论见 docs/theory.md Sec. 6）。

**接入**：
* `step_correction` 在 20% 预算下评估 SADC（与 SDC/FSC 并列）
* `step_baseline_compare` 在所有 5 个预算下评估 SADC（与 Distill/Fine-tune 公平对比）

**预期效果**（理论 + 设计驱动）：
* 低预算（5%–10%）：bootstrap 不确定性自动加权 → Δρ 显著优于 SHAP Distillation
* 高预算（≥20%）：teacher 信号增强 → Δρ 接近 Fine-tuning
* 关键优势：**post-hoc 重加权**，计算成本 ≪ Fine-tuning

## 8. TabDDPM / TabSyn 实现说明

`shap_drift/generators/diffusion.py` 提供 reference 实现：

* **TabDDPM**: feature-space Gaussian DDPM；连续变量用 MLP 噪声预测器；
  目标列采用 empirical bootstrap 重采样（不参与 diffusion）。
* **TabSyn**: 同样的 DDPM，但在小型 VAE 编码的 16-dim 潜空间中训练。

依赖：仅 `torch + numpy + pandas`。若用户安装上游官方包（如 `synthcity`），
可通过修改 `_build_synthesizer` 工厂切换。

## 9. 不变量 / 约定

1. **种子表** 在 `config.py:SEEDS = [42, 123, 456, 789, 2024, 314, 271, 828, 159, 653]`。
2. **数据集注册** 由 `shap_drift.datasets.DATASETS` 单点提供。
3. **模型注册** 由 `shap_drift.models.MODEL_CONFIGS` 单点提供；调用前用
   `get_model_configs()` 取深拷贝，**不要**就地修改共享 dict。
4. **SHAP 维度统一** — 始终通过 `extract_binary_shap` / `_ensure_2d` 取
   `(n_samples, n_features)` 2-D 数组（详见 §11 形状歧义历史 Bug）。
5. **NaN/Inf 处理** — Spearman 走 `models.explainers._safe_spearman`；
   ε 全包统一为 `_EPS = 1e-10`。
6. **生成器特征列** —— TabDDPM/TabSyn 不 diffuse 目标列，需在 `fit(df, target_col=...)`
   中显式声明（已在 `generate_one` 自动注入）。

## 10. 复现建议（v0.3 完整流水线）

```bash
conda env create -f environment.yml
conda activate shap-drift

# 把 10 个原始数据集放到 ./dataset/<source>/
# （路径见各 loader 的常量）

# 完整流水线（10 datasets × 5 generators × 3 models × 10 seeds ≈ 数小时）
python run.py all

# 单步：
python run.py generate           # ~2 hours on a single GPU for TabDDPM/TabSyn
python run.py quality
python run.py baseline
python run.py drift
python run.py correction         # 含 SADC（20% 预算）
python run.py fsc
python run.py ablation
python run.py baseline_compare   # 含 SADC（5 个预算）— 关键论文图
python run.py visualize
python run.py case_study         # 临床/金融案例 HTML
```

**checkpoint 机制**：`correction` / `baseline_compare` 长跑步骤写
`*_checkpoint.csv`，Ctrl+C 后重跑可断点续跑。

## 11. 近期硬化要点（v0.2 + v0.3）

| 主题 | 关键位置 | 摘要 |
| --- | --- | --- |
| 形状歧义 | `models/explainers.py:extract_binary_shap` | `_MAX_CLASSES=16` + `n_features` 校验，2-sample/2-feature 边界不再误判 |
| 数值守卫 | `models/explainers.py:_safe_spearman` | 常量/NaN/长度不一致 → `(0.0, 1.0)` |
| 公式去重 | `correction/sdc.py + fsc.py` | FSC 委托 SDC 的 `_compute_ratio` / `_dampen_alpha` |
| KL 散度 | `metrics/drift.py:_compute_kl_divergence` | 共享 bin 边界 + 概率归一化 |
| p 值数值稳 | `run.py:_step_significance` | `stats.t.sf(t,df)` 替代 `1-cdf` |
| CORAL | `correction/baselines.py:coral_align` | 中心化 + `np.linalg.solve` 替代显式逆 |
| Fine-tune 真实化 | `correction/baselines.py:finetune_baseline` | MLP 走 `partial_fit`；其它走 `joint_oversample_x{N}` 并诚实标注 |
| 全局种子 | `config.set_global_seed` | 覆盖 `PYTHONHASHSEED`、numpy、random、torch (CPU+CUDA) |
| 模型默认种子 | `models/__init__.py` | XGB/RF/MLP 全部带 `random_state=42`，`get_model_configs()` 返回深拷贝 |
| CCF 延迟加载 ★ | `datasets/credit_card_fraud.py` | 特征选择改为 lazy + 缓存，避免 import-time 副作用 |
| **SADC（v0.3）** | `correction/sadc.py` | 多源 prior + per-feature 闭式 α + no-harm |
| **理论** | `docs/theory.md` | 5 条命题对应 SDC/Guarded/SADC/FSC |
| **Case Study** | `case_study/runner.py` | 三个场景 × 每场景 3 个样本卡 |

## 12. 已知风险点

* `compute_shap` 对 KernelExplainer 不做后台进程隔离 — MLP × 大数据集时占内存。
* `step_correction` 和 `step_baseline_compare` 重复计算了 SADC（不同预算）。
* `dataset/CovType/`, `dataset/CreditCardFraud/`, `dataset/Higgs/` 体积较大，建议本地 SSD。
* TabDDPM/TabSyn 默认 50 epochs，在 CPU 上 ~3 min/dataset/generator。GPU ~30s。
* SADC 的 bootstrap teacher 在 5% calibration 时仍可能因小样本失败 — `info["no_harm_reverted"]` 会反映被回滚的特征。

## 13. 输出目录速览

```
results/
├── synth_quality.json
├── baseline_results.json
├── shap_drift.json
├── per_sample_consistency.json
├── interaction_drift.json
├── conditional_drift.json
├── mechanisms_ablation.json
├── significance.json
├── guarded_correction_results.csv      # 含 sadc_rho / sadc_delta 两列
├── guarded_correction_summary.json
├── prior_reliability_analysis.csv
├── fsc_results.csv
├── fsc_summary.json
├── fair_baseline_comparison.csv        # 含 sadc_delta（v0.3 关键论文图）
├── fair_baseline_summary.json
├── ablation_{alpha,prior,sample}.csv
├── case_study.html                     ★ v0.3 临床+金融案例 HTML
├── {ds}_{model}_{gen}_shap.npz          # seed=42 SHAP 张量备份（含新数据集 + 新生成器）
├── {ds}_{suffix}.csv                    # 合成数据（5 个 suffix: gc/ctgan/tvae/tabddpm/tabsyn）
└── figures/                             # 所有 PDF 图
```

## 14. v0.3 论文层面的预期贡献

1. **基准扩展**: 6→10 datasets, 3→5 generators (含 2024 SOTA TabDDPM/TabSyn)
2. **新方法 SADC**: 闭式解 + bootstrap 不确定性 + no-harm guard，预期在低预算下 Δρ 显著优于 SHAP Distillation 和 Fine-tuning
3. **理论分析**: 5 条命题（L₂ 最优性 / no-harm 保证 / 完美 calibration 上界 / Bayes shrinkage / √n 扩散律）
4. **真实使用案例**: 临床（糖尿病/心脏病）+ 金融（信贷），可视化为 HTML

匹配目标期刊：**KDD / TPAMI / TKDE** 一线档可投。
