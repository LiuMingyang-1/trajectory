# How To Run

这份说明只讲怎么运行，不讲实现细节。

默认假设你已经把整个 [icr_probe_repro](/Volumes/LMY/code/icr_probe_modify/icr_probe_repro) 目录复制到一台有 GPU 的机器上。

## 1. 进入目录

```bash
cd /path/to/icr_probe_repro
```

## 2. 安装依赖

推荐 Python 3.10 或 3.11。

```bash
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

## 3. 最常用的运行方式

### 方式 A：直接复用现有 ICR JSONL，跑完整实验

这是最推荐的方式，因为目录里已经自带了：

- `data/input/icr_halu_eval_random_qwen2.5.jsonl`
- `data/input/qa_data.json`

直接执行：

```bash
bash scripts/run_pipeline.sh full
```

这条命令会做这些事：

1. 根据现有 ICR JSONL 恢复 token 对齐信息
2. 构建 tokenizer window spans
3. 构建 spaCy spans
4. 构建 silver labels
5. 导出两个可训练数据集
6. 跑 5 类方法
7. 汇总结果
8. 画默认图表

## 4. 小规模 smoke test

如果你想先确认链路能跑通，可以先只跑小样本。

```bash
MAX_SAMPLES=1000 DEVICE=cpu bash scripts/run_pipeline.sh minimal
```

这条命令只跑：

- tokenizer window
- Baseline MLP

注意：

- 默认是 GPU 优先
- 这里显式写 `DEVICE=cpu` 只是为了快速试链路

如果你想用 GPU 跑小样本，也可以：

```bash
MAX_SAMPLES=1000 DEVICE=cuda bash scripts/run_pipeline.sh minimal
```

## 5. 从头重新计算 ICR 再跑完整实验

如果你不想复用现有 `jsonl`，而是要从原始 QA 数据重新算 ICR，再继续后面的全部流程，执行：

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct DEVICE=cuda \
bash scripts/run_pipeline.sh from-scratch
```

这时会先调用：

- [compute_icr_halueval.py](/Volumes/LMY/code/icr_probe_modify/icr_probe_repro/scripts/compute_icr_halueval.py)

然后再自动接完整下游流程。

## 6. 只重新汇总结果

如果模型已经跑完，只想重新产出总表：

```bash
bash scripts/run_pipeline.sh summary-only
```

会输出：

- `results/summary.json`
- `results/summary.csv`
- `results/summary.md`

## 7. 只重新画图

如果结果文件已经在 `results/` 里，只想重新生成图：

```bash
bash scripts/run_pipeline.sh figures-only
```

## 8. 常用环境变量

### `DEVICE`

默认值：

```bash
cuda
```

如果你要强制 CPU：

```bash
DEVICE=cpu bash scripts/run_pipeline.sh minimal
```

### `MAX_SAMPLES`

限制样本数，适合 smoke test：

```bash
MAX_SAMPLES=500 bash scripts/run_pipeline.sh minimal
```

### `SPACY_MODEL`

默认：

```bash
en_core_web_sm
```

### `ICR_INPUT_PATH`

如果你要换成别的预计算 ICR 文件：

```bash
ICR_INPUT_PATH=/path/to/your_icr.jsonl bash scripts/run_pipeline.sh full
```

### `QA_DATA_PATH`

如果你要换成别的 QA 数据文件：

```bash
QA_DATA_PATH=/path/to/qa_data.json bash scripts/run_pipeline.sh full
```

### `MODEL_NAME_OR_PATH`

只在 `from-scratch` 模式下需要，用于重新计算 ICR：

```bash
MODEL_NAME_OR_PATH=/path/to/local/model DEVICE=cuda \
bash scripts/run_pipeline.sh from-scratch
```

## 9. 输出结果看哪里

主要看两个目录：

- `results/`
- `figures/`

重点文件：

- `results/summary.md`
- `results/summary.csv`
- `figures/method_summary.png`
- `figures/sample_aggregation_summary.png`

## 10. 推荐执行顺序

如果你第一次到 GPU 机器上跑，建议按这个顺序：

### 第一步：小样本试链路

```bash
MAX_SAMPLES=1000 DEVICE=cpu bash scripts/run_pipeline.sh minimal
```

### 第二步：复用现有 ICR，跑完整实验

```bash
bash scripts/run_pipeline.sh full
```

### 第三步：如果需要，再从头重算 ICR

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct DEVICE=cuda \
bash scripts/run_pipeline.sh from-scratch
```

## 11. 最短版本

如果你只想记一条命令：

```bash
cd /path/to/icr_probe_repro
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
bash scripts/run_pipeline.sh full
```
