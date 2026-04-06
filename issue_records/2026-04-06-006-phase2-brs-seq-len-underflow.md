# 问题记录 006：Phase 2 BRS 正式规模下默认 `seq_len=192` 导致答案区被截断

## 所属阶段

- `Phase 2` BRS 主线首次真实启动

## 触发步骤

- 运行：

```bash
source .venv/bin/activate && \
export HF_ENDPOINT=https://hf-mirror.com && \
PYTHONPATH=src python scripts/run_local_core_ladder.py \
  --mode brs \
  --runtime-mode real \
  --data-dir /Users/diannao/Desktop/latent_consensus/data/processed/brs_phase2_expanded_probe \
  --model-name gpt2 \
  --device mps \
  --gate1-report-path /Users/diannao/Desktop/latent_consensus/results/phase1_gate1_real/gate1_report_phase2_waived.json \
  --output-root /Users/diannao/Desktop/latent_consensus/results/phase2_mainline_real \
  --resume
```

## 现象

首个实验在训练前的数据编码阶段直接失败，错误为：

```text
ValueError: train-step4-0 的 answer 被完全截断
```

触发位置在 `tokenize_lm_examples()`，说明输入序列在截断后已经完全丢失了答案 token。

## 原因判断

根因是 `Phase 2` 配置基线和正式规模 BRS 文本长度不匹配：

1. `local_base.yaml` 中默认 `training.seq_len = 192`
2. `EXP-B05 ~ EXP-B12` 当前仍通过 `lc2_shared.yaml / lc3_shared.yaml` 进入，继承到的是 `local_base` 的 `seq_len`
3. 正式规模 BRS 的完整文本长度经实测已达到：
   - `train/val/test` 最大约 `219` token
   - `ood` 最大约 `247` token

因此在 `seq_len=192` 下，`step4/step6` 样本的 `answer_prefix` 甚至完整答案都会被挤出窗口。

## 影响范围

- `Phase 2` 主线无法真正开始训练，阻塞全部 `12` 个 B 组实验。
- 如果只在命令行临时加大 `--seq-len` 而不修正配置来源，后续复现和 resume 极易再次踩坑。

## 已实施修复

本次已落下两处最小修复：

1. 将 `configs/brs_main.yaml` 的 `training.seq_len` 提升到 `256`
2. 将 `EXP-B05 ~ EXP-B12` 的 registry 配置基线统一切到 `brs_main.yaml`
   - 保留各自的 `name / n_processors / observe / weight_mode`
   - 但不再让它们悄悄继承 `local_base.phase0 + seq_len=192`

同时补了回归测试，要求 `EXP-B05` 与 `EXP-B11` 必须解析到：

- `experiment.phase = phase2`
- `training.seq_len = 256`

## 临时处理方向

1. 将 BRS 组实验的配置基线对齐到 `brs_main.yaml`
2. 在 `brs_main.yaml` 中为正式规模 BRS 提供足够的 `seq_len`
3. 为 BRS 配置增加测试，确保 `Phase 2` 不再回退到 `local_base=192`

## 当前状态

- 状态：已缓解，等待真实主线继续验证
- 优先级：高
