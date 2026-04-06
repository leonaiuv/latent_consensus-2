# 研究交接产物索引

## 核心文档

- 执行手册：`/Users/diannao/Desktop/latent_consensus/latent_consensus_engineering_execution_manual_v1_2_3.md`
- 本目录总览：`/Users/diannao/Desktop/latent_consensus/research_handoffs/2026-04-06_phase1_status/README.md`
- 结构化快照：`/Users/diannao/Desktop/latent_consensus/research_handoffs/2026-04-06_phase1_status/status_snapshot.json`

## Gate 0 产物

- `Gate 0` 汇总：`/Users/diannao/Desktop/latent_consensus/results/gate0/gate0_summary.json`
- profiling 报告：`/Users/diannao/Desktop/latent_consensus/results/gate0/profile_memory_report.json`
- 数据校验报告：`/Users/diannao/Desktop/latent_consensus/results/gate0/dataset_validation_report.json`
- 参数/FLOP 账本：`/Users/diannao/Desktop/latent_consensus/results/accounting/model_accounting_report.json`

## Phase 1 真实预跑

- 预跑汇总：`/Users/diannao/Desktop/latent_consensus/results/arithmetic_debug_real/phase1_preflight_report.json`
- `EXP-A01`：`/Users/diannao/Desktop/latent_consensus/results/arithmetic_debug_real/EXP-A01`
- `EXP-A02`：`/Users/diannao/Desktop/latent_consensus/results/arithmetic_debug_real/EXP-A02`
- `EXP-A03`：`/Users/diannao/Desktop/latent_consensus/results/arithmetic_debug_real/EXP-A03`
- `EXP-A04`：`/Users/diannao/Desktop/latent_consensus/results/arithmetic_debug_real/EXP-A04`
- `EXP-A05`：`/Users/diannao/Desktop/latent_consensus/results/arithmetic_debug_real/EXP-A05`
- `EXP-A06`：`/Users/diannao/Desktop/latent_consensus/results/arithmetic_debug_real/EXP-A06`

## 问题记录

- 问题目录：`/Users/diannao/Desktop/latent_consensus/issue_records`
- `001 BRS 模板容量`：`/Users/diannao/Desktop/latent_consensus/issue_records/2026-04-06-001-brs-default-template-capacity.md`
- `002 Arithmetic 答案边界`：`/Users/diannao/Desktop/latent_consensus/issue_records/2026-04-06-002-arithmetic-answer-boundary.md`
- `003 Phase 1 难度梯度不稳`：`/Users/diannao/Desktop/latent_consensus/issue_records/2026-04-06-003-phase1-preflight-noisy-difficulty-gradient.md`

## 关键实现文件

- Arithmetic 真实 runner：`/Users/diannao/Desktop/latent_consensus/src/latent_consensus/training/arithmetic_runner.py`
- 文本任务编码：`/Users/diannao/Desktop/latent_consensus/src/latent_consensus/training/text_tasks.py`
- 真实 LC 模型：`/Users/diannao/Desktop/latent_consensus/src/latent_consensus/models/latent_consensus_causal_lm.py`
- 真实 trainer：`/Users/diannao/Desktop/latent_consensus/src/latent_consensus/training/causal_lm_trainer.py`
- CLI 入口：`/Users/diannao/Desktop/latent_consensus/scripts/run_arithmetic_debug.py`

