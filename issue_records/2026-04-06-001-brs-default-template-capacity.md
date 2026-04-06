# 问题记录 001：BRS 默认生成参数在 `step_count=6` 下模板容量不足

## 所属阶段

- `Phase 0` 数据生成与校验

## 触发步骤

- 运行 `data/generate_brs.py`
- 参数：`--step-counts 2 4 6 --train-count 3 --val-count 2 --test-count 2 --ood-count 1`
- 默认 ID 配置：`entity_count=8`、`distractor_count=2`

## 现象

生成器抛出以下错误：

```text
ValueError: BRS 数据生成未能找到足够多的跨 split 唯一模板
```

## 原因判断

- 当前生成器对跨 split 模板去重是严格开启的。
- 当 `step_count=6` 且 ID 侧实体数、干扰分支数过小，模板空间不足以同时覆盖 `train/val/test` 的唯一性要求。
- 这不是去重逻辑失效，而是默认生成参数与高步数模板容量不匹配。

## 影响范围

- 若继续沿用默认 `id_entity_count=8` / `id_distractor_count=2`，`Phase 0` 与后续 `Phase 2` 数据重建可能不稳定。
- 会导致“脚本能运行但换一组 split 数量就失败”的隐性风险。

## 临时处理

本次已通过扩大模板空间解除阻塞：

- `id_entity_count` 提升到 `12`
- `ood_entity_count` 提升到 `16`
- `id_distractor_count` 提升到 `3`
- `ood_distractor_count` 提升到 `5`

在该配置下，`data/processed/brs` 成功重建，且校验报告显示：

- `duplicates = 0`
- `ood_leaks = 0`
- `template_overlaps = 0`

## 后续动作

1. 在正式进入 `Phase 2` 前，把 BRS 生成参数与手册中的目标样本规模重新对齐。
2. 为 `generate_brs.py` 增加“模板容量不足时的更明确报错与建议参数”。
3. 为高步数配置补一条测试，避免默认参数回归到不足状态。

## 当前状态

- 状态：已缓解，未根治
- 解除方式：扩大 BRS 模板空间参数并重新生成、重新校验
