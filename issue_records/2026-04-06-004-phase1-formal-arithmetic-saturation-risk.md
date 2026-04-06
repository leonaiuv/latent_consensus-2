# 问题记录 004：Phase 1 正式规模下 Arithmetic-Debug 出现整体饱和风险

## 所属阶段

- `Phase 1` Arithmetic-Debug

## 背景

按照 `v1.2.3` 工程执行手册，`Arithmetic-Debug` 的职责不是给出研究 headline，而是作为 `Gate 1` 的链路验证关：

1. 验证 `CoT / LC-1 / LC-2-S / Ind-2-S / LC-3-S / Ind-3-S` 六条真实训练链路都能完整跑通。
2. 验证 `2-step` 相比 `6-step` 存在清晰难度梯度。
3. 在此基础上，才允许进入 `BRS` 主线。

本次正式运行使用的参数比预跑明显放大：

- `train_limit_per_step = 256`
- `val_limit_per_step = 64`
- `test_limit_per_step = 64`
- `ood_limit_per_step = 64`
- `max_epochs = 5`
- `batch_size = 2`
- `gradient_accumulation_steps = 32`
- `runtime = GPT-2 124M + PyTorch + MPS`

## 最终现象

6-run 正式矩阵现已全部完成，结果如下：

- `EXP-A01 / CoT`
  - `test_exact_match = 1.0`
  - `ood_exact_match = 1.0`
  - `step_accuracy = {2: 1.0, 4: 1.0, 6: 1.0}`
- `EXP-A02 / LC-1`
  - `test_exact_match = 1.0`
  - `ood_exact_match = 1.0`
  - `step_accuracy = {2: 1.0, 4: 1.0, 6: 1.0}`
- `EXP-A03 / LC-2-S`
  - `test_exact_match = 0.9947916666666666`
  - `ood_exact_match = 1.0`
  - `step_accuracy = {2: 1.0, 4: 1.0, 6: 0.984375}`
- `EXP-A04 / Ind-2-S`
  - `test_exact_match = 0.9947916666666666`
  - `ood_exact_match = 1.0`
  - `step_accuracy = {2: 1.0, 4: 1.0, 6: 0.984375}`
- `EXP-A05 / LC-3-S`
  - `test_exact_match = 1.0`
  - `ood_exact_match = 1.0`
  - `step_accuracy = {2: 1.0, 4: 1.0, 6: 1.0}`
- `EXP-A06 / Ind-3-S`
  - `test_exact_match = 1.0`
  - `ood_exact_match = 1.0`
  - `step_accuracy = {2: 1.0, 4: 1.0, 6: 1.0}`

对应的正式 `Gate 1` 报告为：

- `cot_lc1_stable = true`
- `shared_runs_complete = true`
- `artifacts_complete = true`
- `gradient_pass_count = 2 / 6`
- `gradient_mean_delta = 0.005208333333333333`
- `gate1_passed = false`

这说明在当前正式规模下，Arithmetic 已经从“难度梯度验证任务”退化成“几乎所有模型都能学满的链路 smoke”。

## 影响

- `Phase 1` 的“训练链路稳定”条件大概率可以满足。
- 但 `Gate 1` 中的关键条件 “`2-step` 明显易于 `6-step`” 反而可能因为整体饱和而失去可判别性。
- 如果 6-run 最终普遍接近满分，则不能把这次正式规模结果解释为“难度梯度成立”；相反，它更可能意味着当前 Arithmetic 任务过于容易，已经不再适合作为有效 gate。

## 初步原因判断

1. 当前 `Arithmetic-Debug` 数据在真实 GPT-2 训练路径下明显过于容易。
2. `teacher_steps + 答案监督` 的文本格式很可能带来了较强模板对齐信号。
3. 当前 `test/ood` 规模不足以在近乎满分时恢复可判别的难度分层。
4. `Arithmetic-Debug` 本来就被定义为 debug 数据集；当真实链路稳定后，它可能天然失去区分 `Gate 1` 所需的判别力。

## 后续动作

1. 将 `Phase 1` 的正式结论固定为：“真实训练链路通过，但 `Gate 1` 因难度梯度不清晰而未通过”。
2. 在进入 `BRS` 前，先回到任务定义层处理 gate 失效问题：
   - 提高 Arithmetic 任务难度
   - 调整监督格式，避免 teacher trace 过强泄漏
   - 或重新评估是否需要把 `Gate 1` 的验证重心前移到更贴近 `BRS` 的任务
3. 在新的 gate 方案明确前，不应把当前正式 `Phase 1` 结果当作进入 `Phase 2` 的许可。

## 当前状态

- 状态：未解决
- 建议优先级：高
