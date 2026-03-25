# cuts

`cuts/` 是围绕 `icr_probe_repro/` 展开的分阶段实验目录，用来把原始 ICR 特征、额外的 entropy 特征，以及三组 cut 实验组织成一个可维护的对比流水线。

当前目标不是替代 `icr_probe_repro/`，而是在其已有数据和训练脚本之上，补齐：

- Cut A: 置信度轨迹特征实验
- Cut B: shallow overconfidence hypothesis 分析与特征实验
- Cut C: adaptive gated probe / confidence weighting 实验

## 依赖关系

本项目默认与 `icr_probe_repro/` 并列放置：

```text
trajectory/
├── cuts/
└── icr_probe_repro/
```

`shared/paths.py` 中的关键依赖路径默认写死为：

- `../icr_probe_repro/data/input/icr_halu_eval_random_qwen2.5.jsonl`
- `../icr_probe_repro/data/input/qa_data.json`
- `../icr_probe_repro/src`
- `../icr_probe_repro/scripts`
- `../icr_probe_repro/results`

如果 `icr_probe_repro/` 不在上述相对位置，当前脚本会直接失败，需要先调整目录结构或修改 [shared/paths.py](/Volumes/LMY/code/trajectory/cuts/shared/paths.py)。

## 目录结构

```text
cuts/
├── cut_a/                  # Cut A 特征构造、训练、错误分析
├── cut_b/                  # Cut B 统计分析、可视化、训练
├── cut_c/                  # Cut C gating 训练与比较分析
├── data/
│   ├── entropy/            # entropy_scores.jsonl
│   ├── combined/           # Cut A/C 依赖的联合 span 数据集
│   └── results/
│       ├── baseline/
│       ├── cut_a/
│       ├── cut_b/
│       └── cut_c/
├── scripts/
│   ├── run_entropy_extraction.py
│   ├── run_baseline.py
│   ├── run_cut_a.py
│   ├── run_cut_b.py
│   ├── run_cut_c.py
│   └── run_all.py
└── shared/                 # 共享路径、数据装载、entropy/inference/eval 工具
```

说明：

- `cut_*` 目录放实验逻辑，不放跨项目共享基础设施。
- `shared/` 负责路径、数据合并、entropy 抽取、评估工具等公共层。
- `data/` 放本项目自产出的数据和结果；上游 ICR 中间数据仍写回 `icr_probe_repro/`。

## 数据流

推荐把整个项目理解为以下数据流：

```text
ICR precomputed data
-> entropy extraction
-> baseline
-> combined dataset
-> cut experiments
```

展开后对应为：

1. `icr_probe_repro/data/input/icr_halu_eval_random_qwen2.5.jsonl`
   已包含 ICR 预计算结果，是本项目的起点。
2. `scripts/run_entropy_extraction.py`
   对同一批样本重新做 forward pass，提取逐层 entropy，输出到 `data/entropy/entropy_scores.jsonl`。
3. `scripts/run_baseline.py`
   调用 `icr_probe_repro/scripts/*` 构造 span-ready 数据、tokenizer windows、silver labels、span dataset，并训练 ICR-only baseline / discrepancy 模型。
4. `scripts/run_cut_a.py`
   读取 ICR + entropy + silver span labels，构造 `data/combined/*.jsonl` 联合数据集，再跑 Cut A 训练和错误分析。
5. `scripts/run_cut_b.py`
   做 shallow/deep entropy 统计分析与作图；如果联合数据集已存在，再继续训练 mismatch 特征模型。
6. `scripts/run_cut_c.py`
   读取联合数据集，训练 adaptive gated probe，并做 gate behavior comparison。

## 推荐执行顺序

推荐顺序：

1. 先确认 `icr_probe_repro/` 可用，且原始 ICR 输入数据存在。
2. 运行 entropy extraction。
3. 运行 baseline，生成上游 span labels / dataset / baseline 结果。
4. 运行 Cut A，生成 combined dataset 并完成 trajectory 特征实验。
5. 运行 Cut B，完成 shallow/deep 分析和 mismatch 特征训练。
6. 运行 Cut C。
7. 如果你只想先验证 shallow/deep 假设，也可以在 Cut A 之前先单独运行一次 `scripts/run_cut_b.py --skip_training`。
8. 如需一键串起来，可以直接用 `scripts/run_all.py`；默认顺序会先跑 Cut A，再跑 Cut B，避免因为缺少 combined dataset 而跳过训练。

如果只想走一遍默认顺序：

```bash
python3 scripts/run_all.py --device cpu
```

## 各阶段关键命令

### 1. Entropy extraction

完整运行：

```bash
python3 scripts/run_entropy_extraction.py --device cuda
```

快速抽样：

```bash
python3 scripts/run_entropy_extraction.py --device cpu --max_samples 32
```

常见输出：

- `data/entropy/entropy_scores.jsonl`

### 2. Baseline

完整运行：

```bash
python3 scripts/run_baseline.py --device cpu --python python3
```

快速抽样：

```bash
python3 scripts/run_baseline.py --device cpu --python python3 --max_samples 128
```

已有产物时尽量复用：

```bash
python3 scripts/run_baseline.py --device cpu --python python3 --skip_existing
```

常见输出：

- `icr_probe_repro/data/intermediate/icr_halu_eval_span_ready.jsonl`
- `icr_probe_repro/data/span_candidates/tokenizer_windows.jsonl`
- `icr_probe_repro/data/span_labels/tokenizer_windows_silver_labels.jsonl`
- `icr_probe_repro/data/datasets/tokenizer_windows_dataset.jsonl`
- `data/results/baseline/baseline_mlp/*.metrics.json`
- `data/results/baseline/baseline_mlp/*.oof_predictions.jsonl`
- `data/results/baseline/discrepancy/*.metrics.json`

### 3. Cut B

先只做分析：

```bash
python3 scripts/run_cut_b.py --device cpu --skip_training
```

如果联合数据集已存在，运行完整版本：

```bash
python3 scripts/run_cut_b.py --device cpu
```

常见输出：

- `data/results/cut_b/analysis_results.json`
- `data/results/cut_b/analysis_report.txt`
- `data/results/cut_b/figures/*.png`
- `data/results/cut_b/training/training_summary.json`（仅训练时）

### 4. Cut A

完整运行：

```bash
python3 scripts/run_cut_a.py --device cpu
```

仅构造 combined dataset 并跳过训练：

```bash
python3 scripts/run_cut_a.py --device cpu --skip_training --skip_error_analysis
```

常见输出：

- `data/combined/tokenizer_windows_mean.jsonl` 或其他 `*.jsonl`
- `data/combined/*.summary.json`
- `data/results/cut_a/training/training_summary.json`
- `data/results/cut_a/training/comparison_table.txt`
- `data/results/cut_a/error_analysis/error_analysis.json`（若 baseline/cut A 预测文件都存在）
- `data/results/cut_a/run_summary.json`

### 5. Cut C

完整运行：

```bash
python3 scripts/run_cut_c.py --device cpu
```

跳过 gate comparison：

```bash
python3 scripts/run_cut_c.py --device cpu --skip_compare
```

常见输出：

- `data/results/cut_c/training/training_summary.json`
- `data/results/cut_c/training/*/torch/*.oof_predictions.jsonl`
- `data/results/cut_c/comparison/comparison_summary.json`
- `data/results/cut_c/comparison/*/gate_analysis_summary.json`
- `data/results/cut_c/run_summary.json`

### 6. 一键串行运行

```bash
python3 scripts/run_all.py --device cpu --python python3
```

例如跳过 baseline 和 Cut C：

```bash
python3 scripts/run_all.py --skip_baseline --skip_cut_c --device cpu --python python3
```

## 每个脚本做什么

### [scripts/run_entropy_extraction.py](/Volumes/LMY/code/trajectory/cuts/scripts/run_entropy_extraction.py)

- 调用 [shared/inference.py](/Volumes/LMY/code/trajectory/cuts/shared/inference.py) 的 CLI。
- 对 ICR 输入样本重新做模型 forward pass。
- 计算逐层 logit entropy。
- 在结束后对输出文件做简单统计检查。

### [scripts/run_baseline.py](/Volumes/LMY/code/trajectory/cuts/scripts/run_baseline.py)

- 调用 `icr_probe_repro/scripts` 下的上游预处理与训练脚本。
- 依次构造 `span_ready -> tokenizer_windows -> silver_labels -> span_dataset`。
- 训练 baseline MLP 和 discrepancy 模型。
- 把关键结果复制到本仓库的 `data/results/baseline/` 下，方便 cut 实验统一读取。

### [scripts/run_cut_b.py](/Volumes/LMY/code/trajectory/cuts/scripts/run_cut_b.py)

- 读取 ICR + entropy 合并后的 sample 级记录。
- 先做 shallow/deep entropy 假设检验和可视化。
- 如果 `data/combined/*.jsonl` 已存在，再训练 mismatch 特征模型。

### [scripts/run_cut_a.py](/Volumes/LMY/code/trajectory/cuts/scripts/run_cut_a.py)

- 读取 ICR 输入、entropy 输出、silver span labels。
- 构造 pooled span-level combined dataset。
- 训练多组 trajectory 特征模型。
- 如果 baseline 和 Cut A 的 OOF 预测文件都存在，再做错误分析。

### [scripts/run_cut_c.py](/Volumes/LMY/code/trajectory/cuts/scripts/run_cut_c.py)

- 读取 combined dataset。
- 训练 `icr_only`、`icr_entropy_concat`、`gated_joint`、`gated_staged` 等变体。
- 对 gated 模型做 gate distribution / entropy relationship / subgroup performance 分析。

### [scripts/run_all.py](/Volumes/LMY/code/trajectory/cuts/scripts/run_all.py)

- 负责按固定顺序串行调度：
  `entropy -> baseline -> cut_b -> cut_a -> cut_c`
- 前一阶段失败则停止，不继续后续阶段。
- 某个脚本文件如果暂时不存在，会记录为 skipped/missing，而不是直接崩溃。

## 预期产物

一个完整实验跑完后，维护者通常应能看到：

- `data/entropy/entropy_scores.jsonl`
- `data/results/baseline/`
- `data/combined/*.jsonl`
- `data/results/cut_a/`
- `data/results/cut_b/`
- `data/results/cut_c/`

其中最关键的总结文件通常是：

- `data/results/cut_a/training/training_summary.json`
- `data/results/cut_b/analysis_results.json`
- `data/results/cut_b/training/training_summary.json`（若已训练）
- `data/results/cut_c/training/training_summary.json`
- `data/results/cut_c/comparison/comparison_summary.json`

## 注意事项

- Entropy extraction 不是读取缓存 ICR 的副产品，而是需要对模型重新做一次新的 forward pass，时间和显存开销都单独存在。
- Baseline 所依赖的上游中间数据一开始可能并不存在；`scripts/run_baseline.py` 的职责之一就是把这些中间文件补齐。
- Cut B 可以先以 analysis-only 模式运行，即使 combined dataset 还没有准备好，统计分析与作图仍然可以先完成。
- Cut C 明确依赖 combined dataset；如果还没有 `data/combined/*.jsonl`，应先运行 Cut A 或自行生成联合数据集。
- 当前路径假设 `cuts/` 与 `icr_probe_repro/` 是 sibling 目录，迁移仓库位置时最容易在这里踩坑。
- `scripts/run_all.py` 会按固定顺序调度，且 Cut B 在 Cut A 之前；这意味着第一次全跑时，Cut B 可能只完成分析、训练部分被其自身脚本自动跳过，这是预期行为。
- 如果未来 Cut C 脚本或目录在并行开发中暂时缺失，README 中的结构和命令仍代表预期接口；`scripts/run_all.py` 也会把缺失脚本记录为跳过。
