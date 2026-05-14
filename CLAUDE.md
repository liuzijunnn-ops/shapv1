# CLAUDE.md — SHAP Explanation Drift Study

> 给协作 LLM 的项目导览。读完本文档应能：(a) 复现任意实验步骤；
> (b) 知道每个模块的职责与不变量；(c) 理解最新一轮 Bug 修复的关注点。

---

## 1. 项目一句话

研究**合成表格数据**与真实数据在**SHAP 解释**上的"漂移（Explanation Drift）"
现象，并提出/评估两类修正方法（SDC、FSC）及其 Guarded 变体，覆盖 6 个数据集 ×
3 个生成器范式 × 3 个分类器 × 10 个随机种子。

## 2. 关键术语

| 术语 | 含义 |
| --- | --- |
| **Drift ρ** | Spearman rank correlation between real-data global SHAP and synth-data global SHAP（越高越一致）|
| **Utility-Drift Gap** | `synth_AUROC − ρ`，即"下游表现 vs 解释一致性"的差距；衡量"高 AUROC 但解释偏离"的隐患 |
| **SDC-Corr** | 用真实数据特征-目标相关性作为先验，重加权合成 SHAP（公式见 `correction/sdc.py`）|
| **SDC-Guarded** | 当先验本身可靠性低（rank_agreement > 阈值）时阻尼 α，避免"过修正" |
| **FSC-Corr** | SDC 的少样本版本：仅用 5%–50% 的真实数据子集计算先验 |

## 3. 顶层目录

```
p6/
├── run.py                # 唯一 CLI 入口（命令子集见 §5）
├── deploy_server.sh      # GPU 服务器部署脚本
├── run_all.sh            # 串行完整流水线
├── environment.yml       # conda env spec
├── pyproject.toml        # PEP 621 元数据
├── shap_drift/           # 主包
│   ├── config.py         # 路径、种子表、生成器/颜色注册表、可序列化工具
│   ├── datasets/         # 6 个真实数据集 loader + DatasetConfig 注册表
│   ├── generators/       # 调用 stg.TableSynthesizer 包装 GC / CTGAN / TVAE
│   ├── models/           # XGBoost / RF / MLP 默认配置 + SHAP 计算封装
│   ├── correction/       # sdc.py / fsc.py / baselines.py（distill+finetune+CORAL）
│   ├── metrics/          # drift.py（漂移）+ quality.py（合成数据 TSTR/KS/W1）
│   ├── ablation/         # alpha / prior / sample 三个轴的消融
│   └── visualization/    # 全部图表输出 PDF
├── dataset/              # 原始数据（CSV/JSON），不在 git 中
└── results/              # 所有 JSON/CSV/NPZ/PDF 输出（每次 run 增量写入）
```

## 4. 数据流（一图概览）

```
DATASETS
  └─ ds_cfg.loader() → DataFrame
       └─ prepare_dataset() → df_clean, X, y          ┐
                                                       │
GENERATORS (stg)                                       │
  └─ generate_one() → results/{ds}_{suffix}.csv ─┐    │
                                                  │    │
                  load_synth_datasets(ds_name) ───┼────┤
                                                  ▼    ▼
                                          run.py per-step pipelines
                                            │
                                            ├─ compute_shap()        → sv_real, gi_real
                                            ├─ drift_metrics()       → ρ, sign_agree, top-k, KS
                                            ├─ sdc_corr / fsc_corr   → sv_corrected
                                            └─ eval_rho/eval_shap    → improvement Δρ
```

## 5. CLI 命令

`python run.py <step>` — 所有 step 的产物写入 `results/`，可重入（带 checkpoint）。

| 命令 | 作用 | 主要输出 |
| --- | --- | --- |
| `generate` | 用 3 个生成器各生成 1 份合成 CSV | `results/{ds}_{gc|ctgan|tvae}.csv` |
| `quality` | 评估合成质量（KS、W1、TSTR-AUROC、相关性差异）| `synth_quality.json` |
| `baseline` | 真实数据上 XGB/RF/MLP 的 AUROC/Acc（10 seeds）| `baseline_results.json` |
| `drift` | 步骤 4–9：drift 指标 + per-sample + 交互 + 类条件 + 机制 + 显著性 | `shap_drift.json`, `per_sample_consistency.json`, `interaction_drift.json`, `conditional_drift.json`, `mechanisms_ablation.json`, `significance.json` |
| `correction` | SDC-Corr / SDC-Guarded / FSC-Corr / FSC-Guarded（10 seeds）| `guarded_correction_results.csv`, `guarded_correction_summary.json`, `prior_reliability_analysis.csv` |
| `fsc` | FSC 在 6 个 calibration fraction 上的扫掠 | `fsc_results.csv`, `fsc_summary.json` |
| `ablation` | α 选择 / 先验类型 / 样本策略 三轴消融 | `ablation_{alpha,prior,sample}.csv` |
| `baseline_compare` | SHAP-Distill / Fine-tune / CORAL vs FSC/SDC，等数据预算 | `fair_baseline_comparison.csv`, `fair_baseline_summary.json` |
| `visualize` | 把上述结果绘制为 PDF | `results/figures/*.pdf` |
| `all` | 依次执行上述全部 | — |

## 6. 不变量 / 约定

1. **种子表** 在 `config.py:SEEDS = [42, 123, 456, 789, 2024, 314, 271, 828, 159, 653]`
   — 任何"多种子"实验都遍历这 10 个 seed；新增 step 应遵循这一约定，不要额外注入随机性。
2. **数据集注册** 由 `shap_drift.datasets.DATASETS` 单点提供，新增数据集只需追加一个
   `DatasetConfig(name, loader, features, target)`。
3. **模型注册** 由 `shap_drift.models.MODEL_CONFIGS` 单点提供；调用前请用
   `get_model_configs()` 取得深拷贝，**不要**就地修改共享 dict。
4. **SHAP 维度统一** — 始终通过 `extract_binary_shap` / `_ensure_2d` 取出
   `(n_samples, n_features)` 2D 数组；不要自己手写 `if sv.shape[0] == 2` 形状检测
   （历史 Bug，详见 §8）。
5. **NaN/Inf 处理** — Spearman 调用统一走 `models.explainers._safe_spearman`；
   分母 ε 全包统一为 `_EPS = 1e-10`（定义在 `correction/sdc.py`）。
6. **目录写入** — 所有持久化路径来自 `config.OUTPUT_DIR`（= `results/`）；图表写入
   `config.FIGURE_DIR`（= `results/figures/`）。

## 7. 复现建议

```bash
# 准备环境
conda env create -f environment.yml
conda activate shap-drift

# 把原始数据放到 ./dataset/<source>/ 下（详见各 loader 的常量）

# 完整流水线（约几小时，10 seeds × 6 datasets × 3 generators × 3 models）
python run.py all

# 或者按需单步：
python run.py generate && python run.py quality
python run.py baseline
python run.py drift
python run.py correction
python run.py fsc
python run.py ablation
python run.py baseline_compare
python run.py visualize
```

完整复现需要：

* CCF 特征选择已改为**延迟**（不在 import 时跑 XGBoost），首次 `load_creditcard_fraud()`
  时缓存（详见 `datasets/credit_card_fraud.py`）。
* 长跑步骤（`correction`、`fsc`、`baseline_compare`）写 `*_checkpoint.csv`，
  中途 Ctrl+C 后再次执行可断点续跑。
* 所有 ablation 中的"扰动种子"使用 `hash((kind, ds_name, sigma/pct)) % 2**32`
  生成的 `RandomState`，单档扰动可独立重放。

## 8. 近期硬化要点（v0.2 重构）

读改后代码时请注意：

| 主题 | 关键位置 | 摘要 |
| --- | --- | --- |
| 形状歧义 | `models/explainers.py:extract_binary_shap` | 引入 `_MAX_CLASSES = 16` 与 `n_features` 一致性校验，2-sample/2-feature 边界不再误判类别轴 |
| 数值守卫 | `models/explainers.py:_safe_spearman` | 常量/全 NaN/长度不一致输入统一返回 `(0.0, 1.0)`，避免聚合污染 |
| 公式去重 | `correction/sdc.py + fsc.py` | FSC 委托 SDC 的 `_compute_ratio` / `_dampen_alpha`，避免双份实现漂移 |
| KL 散度 | `metrics/drift.py:_compute_kl_divergence` | 共享 bin 边界 + 概率归一化（之前两端独立分箱+density 是错的）|
| p 值数值稳 | `run.py:_step_significance` | `stats.t.sf(t, df)` 替代 `1 - cdf`，避免大 t 下溢报"绝对显著" |
| CORAL | `correction/baselines.py:coral_align` | 加中心化、用 `np.linalg.solve` 替代显式逆，提升数值稳定性 |
| Fine-tune 真实化 | `correction/baselines.py:finetune_baseline` | 非 MLP 走 `joint_oversample_x{N}` 并诚实标注；MLP 真正用 `partial_fit` |
| 可视化导入 | `visualization/*.py` | `DATASET_ORDER/DS_LABELS/DS_COLORS` 从 `shap_drift.datasets` 正确导入（之前误用 config，会 ImportError）|
| 全局种子 | `config.set_global_seed` | 覆盖 `PYTHONHASHSEED`、numpy、random、torch (CPU+CUDA) |
| 模型默认种子 | `models/__init__.py` | XGB/RF/MLP 全部带 `random_state=42` 默认值，`get_model_configs()` 返回深拷贝 |
| 边界条件 | `metrics/quality.py`, `prepare_dataset` | 单特征 triu、单类别 target、空 DataFrame 等均显式守卫 |

## 9. 常见任务速查

* **新增一个数据集**
  → 在 `shap_drift/datasets/` 下加 `<ds>.py`，导出 `<DS>_FEATURES / <DS>_TARGET / load_<ds>()`；
  在 `datasets/__init__.py:DATASETS` 注册 `DatasetConfig`。
* **新增一个生成器**
  → 在 `config.py:GENERATOR_SUFFIX` 添加映射，在 `generators/__init__.py:_get_default_config`
  补一个分支，确保生成的 CSV 路径符合 `{ds}_{suffix}.csv` 模式。
* **新增一个修正方法**
  → 在 `correction/` 新建一个文件，签名遵循
  `(shap_synth, df_real, features, target, **kwargs) -> (corrected, info)`；
  在 `correction/__init__.py` 导出；在 `run.py:step_correction` 加入流水线。
* **替换/扩展模型** → 修改 `models/__init__.py:_MODEL_CONFIGS_TEMPLATE`，
  注意更新 `EXPLAINER_MAP`（是否 tree-based）。

## 10. 已知风险点

* `compute_shap` 对 KernelExplainer 不做后台进程隔离 — MLP × 大数据集时占内存。
* `step_fsc` 与 `step_correction` 重复了部分计算（都各自重新 fit 模型 + 计算 SHAP）；
  如需大规模实验可抽取共享 cache。
* `dataset/1798/` 与 `dataset/CreditCardFraud/` 体积较大，建议挂载本地 SSD。
* SHAP 库 ≥ 0.45 与 < 0.40 的输出形状语义不同；我们的解析逻辑兼容两者，
  但仍建议固定一个版本以避免 corner case。

## 11. 输出目录速览

```
results/
├── synth_quality.json             # KS / W1 / TSTR-AUROC / 相关性差异
├── baseline_results.json          # 真实数据上 3 模型 × 10 seed × 6 ds 的性能
├── shap_drift.json                # 主指标：ρ / sign_agree / top-k / KS（每个 model×gen×ds）
├── per_sample_consistency.json    # 样本级 cosine / spearman / sign
├── interaction_drift.json         # 二阶交互的 drift（XGBoost-only）
├── conditional_drift.json         # 类条件 drift（class=0/1 分别看）
├── mechanisms_ablation.json       # mode_collapse / corr_mismatch / KL / target_rate_shift / 数据扰动消融
├── significance.json              # t-test（10 seeds）H0: ρ̄=0
├── guarded_correction_results.csv # 主修正实验明细（540 行）
├── guarded_correction_summary.json
├── prior_reliability_analysis.csv # rank_agreement vs Δρ 散点数据
├── fsc_results.csv                # 6 fraction × 多设置（2700 行）
├── fsc_summary.json
├── fair_baseline_comparison.csv   # SHAP-Distill / Fine-tune / CORAL vs FSC/SDC
├── fair_baseline_summary.json
├── ablation_{alpha,prior,sample}.csv
├── {ds}_{model}_{gen}_shap.npz    # seed=42 的 SHAP 原始张量备份
├── {ds}_{gc|ctgan|tvae}.csv       # 生成的合成数据
└── figures/                       # 所有 PDF 图
```
