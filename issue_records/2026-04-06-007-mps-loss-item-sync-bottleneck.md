# 问题记录 007：`CausalLMTrainer` 在 `MPS` 上逐 batch 调用 `loss.item()` 造成严重同步瓶颈

## 所属阶段

- `Phase 2` BRS 主线重新启动后的性能诊断

## 触发步骤

- 修复 `seq_len` 问题后，重新启动 `Phase 2` 真实主线
- 首个实验 `EXP-B01` 持续运行数分钟，但仍未写出首个 checkpoint 或 `history.json`

## 现象

- 训练进程没有退出，也没有新的 Python 异常
- `torch + MPS + Metal` 相关库均已加载
- 但主线程长时间没有进入首次写盘

对运行中的 Python 进程做 `sample` 后，主线程热点集中在：

```text
output.loss.item()
-> at::native::_local_scalar_dense_mps
-> at::mps::MPSStream::copy_and_sync
-> waitUntilCompleted
```

同时，手动中断时栈也落在：

```text
total_loss += float(output.loss.item())
```

## 原因判断

- 当前 `CausalLMTrainer._train_epoch()` 会在每个 micro-batch 上调用一次 `output.loss.item()`
- `CausalLMTrainer.evaluate()` 也会在每个 batch 上调用一次 `output.loss.item()`
- 在 `MPS` 后端，这会强制 CPU 等待 GPU/Metal command buffer 完成，形成高频同步
- 对正式规模 `BRS` 来说，这会把训练速度拖到非常慢，虽然不一定报错，但会显著影响主线推进

## 影响范围

- 不是数值正确性 bug，但会严重拖慢 `Phase 2` 全部 `12` 个实验
- 会让“首轮 epoch 很慢”变成“每个 micro-batch 都在强制同步”，整体吞吐明显低于应有水平
- 如果不处理，主线实验虽然理论上能跑完，但时间成本会被放大

## 已实施修复

本次已实施最小修复：

1. 不再在 batch 内对 `loss` 调用 `.item()`
2. 训练与评估阶段改为先在设备侧累计 `loss.detach()`，再在 epoch / split 末尾统一同步取值
3. 补了回归测试，验证聚合后的 loss 仍保持原来的均值语义
4. 训练相关回归与全量回归已重新通过

## 当前状态

- 状态：已缓解，等待主线重新验证吞吐改善
- 优先级：高
