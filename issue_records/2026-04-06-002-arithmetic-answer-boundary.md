# 问题记录 002：Arithmetic 真实 runner 的答案边界在 tokenizer 下被吞并

## 所属阶段

- `Phase 1` Arithmetic-Debug 真实训练链路预跑

## 触发步骤

- 运行 `scripts/run_arithmetic_debug.py`
- 实验：`EXP-A01`
- 运行模式：`runtime_mode=real`
- 设备：`MPS`
- 模型：`gpt2`

## 现象

真实 runner 在样本编码阶段抛出错误：

```text
ValueError: train-step2-0 的 answer 被完全截断
```

## 根因分析

问题不在数据本身，而在“答案起点定位”这一步：

1. 旧实现用 `tokenized(answer_prefix)` 与 `tokenized(full_text)` 的长度差来猜答案起点。
2. `GPT-2` 使用 byte-level BPE，答案前缀与数字可能发生跨边界合并。
3. 一旦发生合并，`answer_prefix` 的 token 长度就不再等于“答案开始前的真实 token 位置”。
4. 结果是 `answer_mask` 被错误算成全空，训练链路直接中断。

## 影响范围

- Arithmetic 真实训练不可启动。
- 后续 BRS 真实训练如果继续沿用同一套“按 token 长度猜边界”的逻辑，也会有同类风险。

## 修复动作

本次已经完成修复：

1. `text_tasks.py` 改为优先使用 tokenizer 的 `offset_mapping` 按字符区间定位答案开始位置。
2. 对不支持 `offset_mapping` 的测试 tokenizer，保留 token-length fallback。
3. 对 `answer_prefix + answer` 的拼接增加显式边界处理，避免空白缺失导致答案与前缀粘连。
4. 为该问题补充并跑通了相关单测与 runner 集成测试。

## 验证方式

- `tests/training/test_text_tasks.py`
- `tests/training/test_arithmetic_runner.py`

两者已通过。

在后续正式 `Phase 1` 6-run 完成后，又做了一轮结果侧复核：

- 扫描目录：`results/phase1_gate1_real/arithmetic_debug/EXP-A*/{val,test,ood}_predictions.jsonl`
- 重点检查：
  - `predicted_answer` 明显短于 `gold_answer`
  - `predicted_answer` 只是 `gold_answer` 的数字前缀
  - `predicted_answer` 含有 `+ - * / =` 等明显残缺符号
- 结果：`suspicious_count = 0`

这说明在正式 6-run 路径下，之前出现过的 `"505" -> "5"`、`"660" -> "5"`、`"4 *"` 这类截断残留，本轮没有再出现。

## 当前状态

- 状态：已解决
- 后续建议：BRS 真实文本任务接入时，直接复用当前 offset-based 边界定位逻辑
