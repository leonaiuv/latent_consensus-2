# Latent Consensus 研究交接摘要

## 文档用途

本目录用于向 AI 研究员或协作评审者同步 `latent_consensus` 项目截至 `2026-04-06` 的真实工程状态。  
目标不是重复工程日志，而是回答以下问题：

1. 这个项目当前在研究上究竟想验证什么。
2. 真实实验路径已经推进到哪一步。
3. 遇到了什么问题，哪些已经解决，哪些仍然阻塞结论。
4. 目前拿到了哪些数据，以及这些数据能和不能支持什么判断。

---

## 研究背景

### v1.2.3 的核心目标

本项目当前遵循 `v1.2.3` 执行手册，研究问题被收缩为一个本地可完成的最小因果闭环：

- 硬件边界：`M4 Mac + 16GB 统一内存`
- 框架边界：`PyTorch + MPS`
- 基础模型：`GPT-2 124M`
- 主线架构：`LC-1 / LC-2-S / LC-3-S / Ind-2-S / Ind-3-S`
- 结论优先级：先验证 `observe_gain`，再讨论更大 `N`、独立权重或论文级扩展

### 当前版本为什么先做 Arithmetic

在 `v1.2.3` 中，`Arithmetic-Debug` 不是主结论任务，而是链路校验任务。  
它主要回答三件事：

1. 真实 `GPT-2 + MPS` 训练链路是否已打通。
2. `LC-N / Ind-N` 的 shared observe on/off 路径是否真实可运行。
3. 是否能在按步数分桶结果里看到基础难度梯度，为后续 `BRS` 主线提供准入证据。

所以，Arithmetic 当前是 `Gate 1` 的工程验证集，而不是研究 headline。

---

## 当前推进状态

### 已完成部分

#### 1. Gate 0 已通过

真实环境已经打通，不再是 `numpy smoke`：

- `torch 2.11.0`
- `transformers 5.5.0`
- `datasets 4.8.4`
- `MPS available = true`
- `GPT-2` 权重已真实下载并可加载到 `mps:0`

真实 `10-step` profiling 结果：

- `runtime_status = ok`
- `oom = false`
- `fallback_triggered = false`
- 平均 step 时间约 `890.42 ms`
- 去掉首步预热后的平均 step 时间约 `425.28 ms`
- 峰值当前分配内存约 `2.05 GB`
- 推荐上限约 `12.71 GB`

#### 2. Phase 1 的真实训练路径已实现

已从 `numpy smoke` 升级到真实 `GPT-2` 路径，包括：

- 文本任务编码
- GPT-2 上的 `LatentConsensusCausalLM`
- Causal LM trainer
- Arithmetic 真实 runner
- CLI 入口

#### 3. Phase 1 已完成一轮真实 6-run 小样本预跑

已真实执行：

- `EXP-A01` CoT
- `EXP-A02` LC-1
- `EXP-A03` LC-2-S
- `EXP-A04` Ind-2-S
- `EXP-A05` LC-3-S
- `EXP-A06` Ind-3-S

所有实验均已成功产出：

- checkpoint
- history
- val/test/ood metrics
- val/test/ood predictions

---

## 当前不能宣告成功的原因

### Gate 1 还不能判定通过

虽然真实 6-run 预跑已经完整跑通，但当前只能称为“真实预跑”，还不能称为“正式 Gate 1 通过”。  
关键原因是：

- 当前使用的是小样本预跑参数：
  - `train_limit_per_step = 16`
  - `val_limit_per_step = 8`
  - `test_limit_per_step = 8`
  - `ood_limit_per_step = 8`
  - `max_epochs = 1`
  - `batch_size = 2`
- 在这个规模下，按步数的难度梯度不稳定，尚不能稳定支持“`2-step` 明显易于 `6-step`”。

换句话说，当前我们已经证明：

- 真实训练链路可运行
- 真实 shared 架构可运行
- observe on/off 路径可运行

但还没有证明：

- Arithmetic 已经在正式训练规模下稳定呈现预期难度结构
- 因而也还不能用它作为 `BRS` 主线前的正式准入依据

---

## 当前真实数据

### Gate 0 核心数据

- `Gate 0 passed = true`
- Arithmetic 数据校验：`duplicates = 0`、`ood_leaks = 0`、`template_overlaps = 0`
- BRS 数据校验：`duplicates = 0`、`ood_leaks = 0`、`template_overlaps = 0`
- GPT-2 实例化参数量：`124,439,808`

### Phase 1 真实 6-run 预跑结果

|实验|模型|test exact match|test step accuracy|ood exact match|
|---|---|---:|---|---:|
|`EXP-A01`|CoT|`0.8333`|`2:0.875, 4:1.0, 6:0.625`|`0.9583`|
|`EXP-A02`|LC-1|`0.5833`|`2:0.5, 4:0.625, 6:0.625`|`0.5833`|
|`EXP-A03`|LC-2-S|`0.9583`|`2:1.0, 4:0.875, 6:1.0`|`0.8333`|
|`EXP-A04`|Ind-2-S|`0.7917`|`2:0.875, 4:0.75, 6:0.75`|`0.7917`|
|`EXP-A05`|LC-3-S|`0.8333`|`2:0.875, 4:0.75, 6:0.875`|`0.8750`|
|`EXP-A06`|Ind-3-S|`0.7917`|`2:0.75, 4:0.75, 6:0.875`|`0.7500`|

### 对这些数值的谨慎解释

当前数值只能说明：

1. 实验入口和训练/评估/落盘链路都真的跑起来了。
2. `LC-2-S / Ind-2-S` 与 `LC-3-S / Ind-3-S` 在真实 GPT-2 路径上已经能够分开执行。
3. 预跑下确实出现了 `LC-N` 与 `Ind-N` 的数值差异，不再是完全相同的假结果。

当前数值不能说明：

1. `observe_gain` 已经稳健成立。
2. `LC-N` 在 Arithmetic 上已经可靠优于 matched `Ind-N`。
3. 可以依据当前 Arithmetic 结果直接推进到 `Phase 2 BRS` 正式主线。

---

## 已遇到的问题与状态

### 已解决问题

#### 问题 001：BRS 默认模板容量不足

- 现象：`step_count=6` 下，默认 ID 模板空间不足，导致跨 split 唯一模板约束无法满足
- 当前处理：扩大 `entity_count` 与 `distractor_count`
- 状态：已缓解，未根治

#### 问题 002：Arithmetic 答案边界被 tokenizer 吞并

- 现象：真实 GPT-2 byte-level tokenizer 下，答案起点定位失败，训练直接报错
- 根因：原实现靠 token 长度差推测答案边界，不适合 byte-level BPE
- 修复：改为优先使用 `offset_mapping` 按字符位置定位答案区间
- 状态：已解决

### 未解决问题

#### 问题 003：Phase 1 小样本预跑的难度梯度不稳定

- 现象：`2-step > 6-step` 并未在 6 个实验中稳定呈现
- 最可能原因：
  - 样本量太小
  - epoch 太少
  - 当前仍处于链路验收规模，不是正式训练规模
- 状态：未解决，当前最高优先级阻塞

---

## 当前最可能的研究判断

### 工程层面

可以明确判断：

- 工程已经从“文档 + smoke 骨架”推进到“真实 GPT-2 实验基础设施就绪”
- Gate 0 已过
- Phase 1 真实训练链路已打通

### 研究层面

当前最合理的判断是：

- 现在拿到的是“真实预跑证据”，不是“正式 Phase 1 结论”
- 研究上最需要评估的，不是是否立刻进入 BRS，而是：
  - 先不先把 Arithmetic 训练规模放大，把 `Gate 1` 做稳
  - 还是接受 Arithmetic 仅作为链路验证，直接转向 BRS 小规模真实 pilot

### 我更偏向的解释

如果研究目标是严格按 `v1.2.3` 执行手册推进，那么当前更稳妥的路径仍然是：

1. 继续保留 Arithmetic 作为 Gate 1 验证集
2. 放大样本规模与训练轮数
3. 再检查 `2-step` 与 `6-step` 的难度梯度是否稳定
4. 只有在 Gate 1 通过后，再进入 BRS 主线

原因是：当前最大的剩余不确定性不是“BRS 有没有希望”，而是“Phase 1 的正式 gate 到底是不是已经满足”。

---

## 建议研究员重点评估的问题

1. 在当前 `v1.2.3` 目标下，Arithmetic 是否必须严格承担 `Gate 1` 准入责任，还是只要链路通即可放行 BRS。
2. 当前 `LC-N / Ind-N` 的真实实现形式，是否足够贴近手册中的 shared observe on/off 设计，还是仍然偏工程最小实现。
3. `Gate 1` 的正式放行标准里，“`2-step > 6-step`”应该要求到什么稳定程度。
4. 下一轮预算应优先投给：
   - Arithmetic 扩样本和 epoch
   - 还是 BRS 的小规模真实 pilot

---

## 一句话总结

项目已经不再停留在 smoke 骨架阶段，真实 `GPT-2 + MPS` 路径已经跑通；当前最核心的问题不再是“能不能跑”，而是“Phase 1 的正式 gate 是否已经足够稳，能否放行 BRS 主线”。

