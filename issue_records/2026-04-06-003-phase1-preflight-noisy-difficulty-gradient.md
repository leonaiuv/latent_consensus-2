# 问题记录 003：Phase 1 小样本预跑尚未稳定呈现 `2-step > 6-step` 难度梯度

## 所属阶段

- `Phase 1` Arithmetic-Debug

## 触发步骤

- 使用真实 `GPT-2 + MPS` 完成 `EXP-A01` 到 `EXP-A06` 的小样本预跑
- 运行参数：
  - `train_limit_per_step = 16`
  - `val_limit_per_step = 8`
  - `test_limit_per_step = 8`
  - `ood_limit_per_step = 8`
  - `max_epochs = 1`
  - `batch_size = 2`

## 现象

6 个实验都已成功完成训练、验证、测试、OOD 评估、checkpoint 与预测文件导出，但按步数的准确率并未稳定体现“2-step 明显易于 6-step”：

- `EXP-A01`：`2-step=0.875`，`6-step=0.625`
- `EXP-A02`：`2-step=0.5`，`6-step=0.625`
- `EXP-A03`：`2-step=1.0`，`6-step=1.0`
- `EXP-A04`：`2-step=0.875`，`6-step=0.75`
- `EXP-A05`：`2-step=0.875`，`6-step=0.875`
- `EXP-A06`：`2-step=0.75`，`6-step=0.875`

## 原因判断

当前更像“链路验收跑”，还不是足以支持 Gate 1 判断的正式训练：

1. 每个 step bucket 的测试样本只有 `8` 条，方差过大。
2. 仅训练 `1 epoch`，不足以稳定体现课程难度差异。
3. `LC-N/Ind-N` 真实实现刚切换到 GPT-2 路径，还需要更长训练观察。

## 影响范围

- 不能用当前结果宣称 `Phase 1 Gate 1` 已通过。
- 但可以确认真实训练链路、shared observe on/off 路径、checkpoint 与预测导出都已打通。

## 临时结论

- `P1-01` 到 `P1-03` 的真实实现链路已具备可运行性。
- 当前结果只可视为 `Phase 1` 真实预跑，不可视为正式 gate 判定结果。

## 后续动作

1. 优先提高 `max_epochs` 与每个 step bucket 的样本量。
2. 继续保留 `EXP-A01` 到 `EXP-A06` 的完整真实矩阵，不要只看单个实验。
3. 在样本规模放大后再次检查 `2-step` 与 `6-step` 的难度梯度。

## 当前状态

- 状态：未解决
- 建议优先级：高
