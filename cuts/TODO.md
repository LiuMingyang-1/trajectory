# TODO

## infra/shared

- [x] 建立 `shared/paths.py` 统一管理 `cuts/` 与 `icr_probe_repro/` 路径。
- [x] 提供 `shared/data_loader.py` 以加载 ICR / entropy / combined dataset。
- [x] 提供 `shared/entropy.py` 与 `shared/inference.py` 做 entropy 抽取。
- [x] 提供 `shared/eval_utils.py` 作为训练与评估公共工具。
- [x] 提供 `scripts/run_entropy_extraction.py` 作为 entropy CLI 入口。
- [x] 提供 `scripts/run_all.py` 作为串行 orchestration 入口。
- [ ] 在目标机器上验证 entropy extraction 依赖可用性（`transformers` / `torch` / 模型权重）。
- [ ] 用小样本跑通 `run_all.py` 的默认顺序并检查阶段间衔接。

## baseline

- [x] 提供 `scripts/run_baseline.py` 统一调用上游 ICR 预处理和 baseline 训练脚本。
- [x] 支持 span-ready、tokenizer windows、silver labels、span dataset 的串行构造。
- [x] 支持 baseline MLP 与 discrepancy 结果回收至 `data/results/baseline/`。
- [ ] 首次完整运行 baseline，确认上游中间文件会被正确生成。
- [ ] 校验 `data/results/baseline/` 中 metrics 与 OOF prediction 文件齐全。
- [ ] 记录一版 baseline 的核心指标结论。

## cut A

- [x] 实现 Cut A 特征构造逻辑。
- [x] 实现 Cut A 训练入口与多特征集实验。
- [x] 支持按需构造 `data/combined/*.jsonl`。
- [x] 实现 Cut A error analysis。
- [x] 提供 `scripts/run_cut_a.py` 统一入口。
- [ ] 运行 Cut A 训练并检查 `training_summary.json` 与 `comparison_table.txt`。
- [ ] 校验 combined dataset 的样本数、缺失样本统计与 pooling 设置。
- [ ] 对照 baseline 完成一次错误分析结果复核。

## cut B

- [x] 实现 shallow/deep entropy 统计分析。
- [x] 实现 Cut B 可视化输出。
- [x] 实现 mismatch 特征训练逻辑。
- [x] 提供 `scripts/run_cut_b.py` 统一入口。
- [ ] 先执行一次 analysis-only 运行并检查图表与统计报告。
- [ ] 在 combined dataset 准备好后补跑 Cut B 训练。
- [ ] 复核 mismatch 特征相对 ICR-only 的增益是否稳定。

## cut C

- [x] 实现 Cut C gating / adaptive probe 训练代码。
- [x] 实现 Cut C gate comparison 分析。
- [x] 提供 `scripts/run_cut_c.py` 统一入口。
- [ ] 在 combined dataset 准备好后运行 Cut C 全流程。
- [ ] 检查 gated 模型的 `training_summary.json` 与 `comparison_summary.json`。
- [ ] 复核 gate distribution、entropy relationship、subgroup performance 结论。

## docs/results

- [x] 编写仓库级 README。
- [x] 补充项目 TODO 清单。
- [ ] 整理一份简短实验记录，说明每个阶段的输入、输出、耗时与设备。
- [ ] 将 baseline / Cut A / Cut B / Cut C 的核心结果汇总到统一位置。
- [ ] 标注最终推荐复现实验命令与默认参数组合。
