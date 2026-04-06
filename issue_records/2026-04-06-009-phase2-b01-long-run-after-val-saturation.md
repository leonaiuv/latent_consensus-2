# 问题记录 009：`EXP-B01` 在验证集已满分后仍长时间继续运行

## 所属阶段

- `Phase 2` BRS 主线在线诊断

## 现象

`EXP-B01` 在首次验证后很早就得到：

- `val_exact_match = 1.0`
- `step_accuracy = {2: 1.0, 4: 1.0, 6: 1.0}`

随后继续运行超过 1 小时，但目录中长期没有新的最终产物：

- 无 `test_metrics.json`
- 无 `ood_metrics.json`
- 无 `summary.json`

当前可见文件长期维持为：

- `best_checkpoint.pt`
- `val_metrics.json`
- `val_predictions.jsonl`

## 关键证据

从当前 `best_checkpoint.pt` 读取到的训练历史为：

```python
{
  "best_epoch": 1,
  "history_len": 2,
  "history": [
    {"epoch": 0, "train_loss": 0.09038780629634857, "val_loss": 0.02311609499156475, "val_exact_match": 1.0},
    {"epoch": 1, "train_loss": 0.026413576677441597, "val_loss": 0.01830288954079151, "val_exact_match": 1.0}
  ]
}
```

这说明：

1. 至少已经完整跑完 `epoch 0` 和 `epoch 1`
2. `epoch 1` 被保存为 best checkpoint 的原因不是 `exact_match` 变好，而是 `val_loss` 继续下降

## 原因判断

当前长跑主要由两个因素叠加造成：

1. 单个 epoch 很长
   - 正式规模下每个 epoch 大约包含：
     - 训练 micro-batch：`12000 / 2 = 6000`
     - 验证 micro-batch：`1200 / 2 = 600`
   - 因此一个 epoch 本身就是 `30-40` 分钟级别

2. 早停准则仍基于 `val_loss`
   - 即使 `val_exact_match` 已经达到 `1.0`
   - 只要 `val_loss` 还在微幅下降，训练就会继续被视为“在改进”
   - 这会让“研究上已经接近无增益”的 run 继续消耗大量时间

## 影响

- 这不是死锁或崩溃，更像是“工程仍在跑，但研究价值已提前饱和”
- 在当前 BRS 天花板风险较高的背景下，这会拖慢对 `test/ood` 的关键判断
- 如果放任这种策略扩展到 `B02 ~ B12`，整体时间成本会被显著放大

## 结论

当前更合理的判断是：

1. `EXP-B01` 没有卡死
2. 它之所以长时间未结束，是因为：
   - epoch 粒度很大
   - 且早停依据是 `val_loss`，不是更贴近研究判别力的指标

## 当前状态

- 状态：已确认
- 优先级：高
