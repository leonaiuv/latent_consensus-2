# Latent Consensus 工程执行手册 v1.2.3

> 对应主手册：`latent_consensus_experiment_proposal_v_1_2_3.md`  
> 参考摘要：`latent_consensus_v1_2_3_changelog.md`  
> 文档目的：把 v1.2.3 提案收敛成可直接交给 AI/工程代理执行的工程手册。  
> 当前仓库状态：仅有文档，尚未初始化代码工程。  
> 默认受众：AI/工程代理，而不是研究综述读者。  
> 文档原则：中文、TDD、Gate 驱动、先证据链后扩范围。

---

## 0. 文档定位

这不是对 `v1.2.2` 任务清单的补丁，而是一份独立的 `v1.2.3` 工程执行手册。  
它同时承担三件事：

1. 作为工程 backlog，定义按阶段推进的任务卡。
2. 作为执行约束，限定本地实验边界、结论边界和停止条件。
3. 作为 AI 代理的决策手册，明确每一步的输入、允许动作、禁止动作、交付物、Gate 和失败回退。

---

## 1. 固定边界与默认值

### 1.1 本版固定边界

- 硬件：M4 Mac，16GB 统一内存。
- 框架：PyTorch + MPS。
- 基础模型：GPT-2 124M。
- 本地主线：`N=2/3`，`shared weights`。
- 数据集优先级：`Arithmetic-Debug` 仅用于 debug，`BRS` 为主数据集。
- 主结论对象：`observe_gain` 及其通信证据链。
- 结果上限：本地最多支持到 `L3` 结论，不允许在本地默认写 `L4` 结论。

### 1.2 本版明确不做

1. 不把 Translator Agent / Verbalizer Agent 放进主线。
2. 不把 `N>3` 作为本地默认目标。
3. 不把独立权重、Hybrid、大 `N`、容量公平性、公共 benchmark 放进主线必做。
4. 不在没有容量匹配单模型基线时声称“架构优于单模型”。
5. 不把 Arithmetic 写成 headline。
6. 不把 attention consensus 设成默认方案。
7. 不在 `LC-N > LC-1` 且 `LC-N ≈ Ind-N` 时声称“互观察有效”。

### 1.3 本版默认参数

|项|默认值|
|---|---|
|`weight_mode`|`shared`|
|`N`|`2, 3`|
|`K`|`5`|
|`alpha_init`|`0.1`，可学习|
|`dropout`|`0.1`|
|训练噪声 `std`|`0.005`|
|推理默认|关闭噪声与 dropout|
|`seq_len`|`192`|
|有效 batch size|`64`|
|micro-batch|`2` 或 `4`|
|梯度累积|`16` 或 `32`|
|最大 epoch|`30`|
|早停|`patience = 5`|
|统一内存警戒线|`13GB`|

---

## 2. 执行规则

### 2.1 TDD 强制流程

所有实现任务必须按以下顺序执行：

1. 先写测试：覆盖正常、边界、异常、性能四类场景。
2. 再写最小实现：只为当前测试通过服务，不提前实现条件菜单和云端阶段。
3. 最后重构：收敛接口、补注释、整理结果 schema 和命令入口。

默认测试主轴是 `pytest + CLI + 报告校验`。  
`Playwright Interactive` 只在后续出现 Web 报告界面或可视化页面时用于 smoke，不作为当前离线实验工程的主验证手段。

### 2.2 完成定义

|项目|完成标准|
|---|---|
|代码|存在明确入口文件，能独立运行|
|测试|存在单测或集成测试，且覆盖当前任务目标|
|配置|参数不写死，进入 `configs/`|
|日志|关键指标进入本地 `results/`，如接入 wandb 需字段一致|
|报告|每个 Gate 有独立决策报告或结果摘要|
|文档|任何影响结论口径的改动，都要同步更新本手册或结果 schema|

### 2.3 协作策略

- 中等及以上任务默认拆成三条并行工作流：
  - `数据/校验`
  - `模型/训练`
  - `分析/统计`
- 三条工作流必须是非重叠写集；共享约束只通过配置、schema 和结果目录对齐。
- 如果某条工作流的前置 Gate 未通过，其余工作流不得自行扩 scope。

### 2.4 结论顺序

结果章节必须按证据链排序，禁止先画 `N` 曲线：

1. `observe_gain`
2. `observe_off_delta` / `scramble_delta`
3. `synergy_rate` + `single_processor_accuracy`
4. `repr_diversity_eval` + `processor_disagreement`
5. ID / OOD 对比
6. adaptive halting 停止步数分布

---

## 3. 三层看板

### 3.1 主线必做

- `Phase 0` 环境、账本、数据、骨架
- `Phase 1` Arithmetic-Debug
- `Phase 2` BRS 主线
- `Phase 3` intervention 与低成本推理分析

### 3.2 条件菜单

- `Phase 4` 第 3 个 seed、独立权重 readiness、promotion runner

### 3.3 云端阶段

- `Phase 5` 容量公平性、大 `N`、公共 benchmark

> 规则：只有上一个阶段 Gate 通过，才能进入下一个阶段。  
> 规则：条件菜单必须满足准入条件，不能因为“看起来有意思”提前执行。  
> 规则：云端阶段不允许反向污染本地主结论措辞。

---

## 4. 固定接口与目录约束

### 4.1 固定配置入口

- `configs/local_base.yaml`
- `configs/arithmetic_debug.yaml`
- `configs/brs_main.yaml`
- `configs/lc2_shared.yaml`
- `configs/lc3_shared.yaml`
- `configs/optional_independent.yaml`

### 4.2 固定脚本入口

- `data/generate_arithmetic_debug.py`
- `data/generate_brs.py`
- `scripts/validate_datasets.py`
- `scripts/profile_memory.py`
- `scripts/model_accounting.py`
- `scripts/run_local_core_ladder.py`
- `scripts/run_arithmetic_debug.py`
- `scripts/run_brs_main.py`
- `scripts/run_brs_promotion.py`
- `src/analysis/intervention_tests.py`
- `src/analysis/synergy.py`
- `src/analysis/halting.py`

### 4.3 固定结果字段

以下字段必须进入统一结果 schema：

- `macro_acc_id`
- `macro_acc_ood`
- `observe_gain_N`
- `observe_off_delta`
- `scramble_delta`
- `synergy_rate`
- `processor_disagreement`
- `leave_one_out_delta`
- `latency_per_sample`
- `accuracy_per_param`
- `accuracy_per_flop`

### 4.4 固定目录骨架

```text
latent-consensus/
├── configs/
├── data/
│   ├── generate_arithmetic_debug.py
│   ├── generate_brs.py
│   └── processed/
├── src/
│   ├── models/
│   ├── training/
│   ├── analysis/
│   └── utils/
├── scripts/
├── tests/
├── results/
├── checkpoints/
└── logs/
```

---

## 5. 主实验矩阵

### 5.1 Phase 1：Arithmetic-Debug（6 runs，seed=42）

|ID|数据集|模型|
|---|---|---|
|EXP-A01|Arithmetic|CoT|
|EXP-A02|Arithmetic|LC-1|
|EXP-A03|Arithmetic|LC-2-S|
|EXP-A04|Arithmetic|Ind-2-S|
|EXP-A05|Arithmetic|LC-3-S|
|EXP-A06|Arithmetic|Ind-3-S|

### 5.2 Phase 2：BRS 主线（12 runs，seeds=42,123）

|ID|数据集|模型|seed|
|---|---|---|---|
|EXP-B01|BRS|CoT|42|
|EXP-B02|BRS|CoT|123|
|EXP-B03|BRS|LC-1|42|
|EXP-B04|BRS|LC-1|123|
|EXP-B05|BRS|LC-2-S|42|
|EXP-B06|BRS|LC-2-S|123|
|EXP-B07|BRS|Ind-2-S|42|
|EXP-B08|BRS|Ind-2-S|123|
|EXP-B09|BRS|LC-3-S|42|
|EXP-B10|BRS|LC-3-S|123|
|EXP-B11|BRS|Ind-3-S|42|
|EXP-B12|BRS|Ind-3-S|123|

### 5.3 Phase 4：条件晋级（4 runs，seed=456）

> 只有 `LC-2-S` 或 `LC-3-S` 对 matched `Ind-N` 出现 Positive 时，才能进入本阶段。

|ID|数据集|模型|seed|
|---|---|---|---|
|EXP-C01|BRS|CoT|456|
|EXP-C02|BRS|LC-1|456|
|EXP-C03|BRS|winning `LC-N`|456|
|EXP-C04|BRS|matched `Ind-N`|456|

---

## 6. Phase 0：环境、账本、数据、骨架

### 6.1 阶段定义

|项|内容|
|---|---|
|输入|`latent_consensus_experiment_proposal_v_1_2_3.md`、`latent_consensus_v1_2_3_changelog.md`|
|允许动作|建工程骨架、写配置、写数据生成器、写 schema、写本地 profiling 与 accounting 脚本|
|禁止动作|提前写独立权重、提前写大 `N`、提前跑公共 benchmark|
|阶段交付物|10-step 压测报告、参数/FLOP 账本、两套数据与校验报告、目录骨架、统一结果 schema|

### 6.2 测试场景

|类型|必须覆盖|
|---|---|
|正常|MPS 可用、数据可生成、schema 可导出、账本可产出|
|边界|`seq_len=192`、micro-batch 最小配置、step=10 边界不崩|
|异常|OOM、MPS fallback、重复样本、OOD 模板泄漏、字段缺失时报错|
|性能|10-step 内存/时延报告必须真实落盘，禁止只跑 1 step|

### 6.3 任务卡

#### `P0-01` 工程骨架与配置约束

|字段|内容|
|---|---|
|目标|建立可安装、可测试、可运行的研究工程骨架|
|测试先行|目录存在性测试；配置名与固定入口一致性测试|
|最小实现|创建目录、`pyproject.toml`、基础配置文件和空测试入口|
|重构点|统一配置加载接口，避免多个脚本各自解析参数|
|验收产物|目录骨架、基础配置、空 `pytest` 通过记录|
|阻塞条件|目录命名与固定接口不一致时不得进入后续任务|

#### `P0-02` MPS smoke 与 10-step 真配置压测

|字段|内容|
|---|---|
|目标|落实真实训练准入，禁止用 1 step 替代正式压测|
|测试先行|profile 报告 schema 测试，必须包含内存、时延、设备字段|
|最小实现|`profile_memory.py` 输出 JSON 报告，连续 10 step 记录内存与 step time|
|重构点|统一采样接口，供后续 Gate 和报告复用|
|验收产物|10-step JSON 报告、失败日志、推荐配置摘要|
|阻塞条件|出现 OOM、fallback、梯度异常或报告缺字段时不得进入 Gate 0|

#### `P0-03` 参数量 / FLOP 账本

|字段|内容|
|---|---|
|目标|建立弱公平性前置账本|
|测试先行|账本 JSON/CSV schema 测试，校验 `params`、`train_flops_per_step`、`infer_flops_per_sample`|
|最小实现|对 `LC-1`、`LC-2-S`、`LC-3-S` 输出账本|
|重构点|统一模型签名，避免脚本按模型名写分支|
|验收产物|本地账本文件、生成命令、版本说明|
|阻塞条件|账本不可复现或字段缺失时，不得写任何效率结论|

#### `P0-04` Arithmetic-Debug 数据生成与验收

|字段|内容|
|---|---|
|目标|生成仅用于 debug 的算术数据|
|测试先行|覆盖正常样本、结果范围、ID/OOD 非重叠、步数分桶正确|
|最小实现|`generate_arithmetic_debug.py` 输出 train/val/test/ood 四个 split|
|重构点|抽象公共 split 逻辑，供 BRS 复用|
|验收产物|数据文件、统计摘要、抽样检查记录|
|阻塞条件|若 Arithmetic 无法稳定生成 2/4/6 step 样本，不得进入 Phase 1|

#### `P0-05` BRS 数据生成与验收

|字段|内容|
|---|---|
|目标|生成 v1.2.3 主数据集 BRS|
|测试先行|唯一正确链路校验、死路分支校验、无环校验、OOD 复杂度提升校验|
|最小实现|`generate_brs.py` 输出 train/val/test/ood 四个 split 与 teacher steps|
|重构点|把“唯一正确链路”和“干扰分支”校验封装成可复用断言|
|验收产物|BRS 数据文件、统计摘要、样本人工抽查记录|
|阻塞条件|若 CoT/LC-1 在后续验证里都低于可用水平，必须回到本任务重做数据定义|

#### `P0-06` 去重 / OOD 校验与结果 schema

|字段|内容|
|---|---|
|目标|统一数据验证和结果输出口径|
|测试先行|构造重复样本、OOD 泄漏样本、缺字段结果文件，确认脚本报错|
|最小实现|`validate_datasets.py` 和统一 metrics/result schema|
|重构点|把通用校验规则抽成 schema 层，避免散落在各脚本里|
|验收产物|去重报告、OOD 报告、结果 schema 文档|
|阻塞条件|未覆盖 `observe_off_delta`、`scramble_delta`、`synergy_rate` 等字段时不得进入 Phase 1|

### 6.4 Gate 0

以下条件必须全部满足：

- MPS 可用。
- 10-step 真配置压测无 OOM。
- 参数量 / FLOP 账本可导出。
- CoT / LC-1 单个 batch 可回传梯度。
- 数据去重与 OOD 报告通过。
- 结果 schema 可覆盖固定字段。

### 6.5 失败回退

- 若 10-step 压测失败：优先降 micro-batch、再降 `seq_len`，不得直接改结论目标。
- 若 BRS 校验失败：先改 teacher steps 或任务复杂度，不得直接加架构复杂度。

---

## 7. Phase 1：Arithmetic-Debug

### 7.1 阶段定义

|项|内容|
|---|---|
|输入|Gate 0 全部通过；Arithmetic-Debug 数据已落盘|
|允许动作|实现 `LC-1`、`LC-2-S`、`Ind-2-S`、`LC-3-S`、`Ind-3-S` 的 shared 路径与训练链路|
|禁止动作|独立权重、Hybrid、大 `N`、公共 benchmark、容量公平性|
|阶段交付物|6 个 debug runs、loss 曲线、按步数准确率、检查点、样本预测文件|

### 7.2 测试场景

|类型|必须覆盖|
|---|---|
|正常|CoT、LC-1、LC-2-S、Ind-2-S、LC-3-S、Ind-3-S 都能启动并完成训练/评估|
|边界|`N=1 == LC-1` 等价；`K=1/5` 行为差异可观测|
|异常|shape 错误、梯度不回传、日志字段缺失、checkpoint 不可读时报错|
|性能|6 个 Arithmetic runs 可在本地预算内完成并产出完整报告|

### 7.3 任务卡

#### `P1-01` `LC-1` 与 shared 模型骨架

|字段|内容|
|---|---|
|目标|把 `LC-1`、`LC-2-S`、`LC-3-S` 与 `Ind-2-S`、`Ind-3-S` 的 shared 实现前移到主线|
|测试先行|`N=1 == LC-1` 等价测试；输出 shape 测试；梯度回传测试|
|最小实现|实现 shared 路径、observe on/off 切换、latent consensus 聚合|
|重构点|把 `LC-N` 与 `Ind-N` 统一在同一模型家族下，区别仅体现在 observe 开关|
|验收产物|模型模块、单测、最小 smoke checkpoint|
|阻塞条件|若 `Ind-N` 通过删处理器实现而不是 observe=0 实现，视为不合格|

#### `P1-02` trainer、curriculum、评估入口

|字段|内容|
|---|---|
|目标|统一训练、验证、checkpoint、日志与按步数评估|
|测试先行|fake model + fake dataset smoke；配置加载；CLI 覆盖测试|
|最小实现|trainer、curriculum、metrics、CoT baseline 入口|
|重构点|统一训练与评估结果写入路径，避免 Arithmetic/BRS 各自维护 schema|
|验收产物|训练入口、评估入口、metrics 文件、CLI 示例|
|阻塞条件|若日志字段与固定结果字段不一致，不得进入下一任务|

#### `P1-03` Arithmetic 6-run runner

|字段|内容|
|---|---|
|目标|按固定矩阵执行 `EXP-A01` 到 `EXP-A06`|
|测试先行|矩阵展开测试，确保 6 组实验不缺不重；结果聚合 schema 测试|
|最小实现|`run_arithmetic_debug.py` 按实验 ID 启动训练、评估和结果落盘|
|重构点|提取实验 ID 到配置映射，供 Phase 2 复用|
|验收产物|6 个实验结果、loss 曲线、按步数准确率表、运行日志|
|阻塞条件|任何一个实验无法完整产出检查点与预测文件时，Gate 1 不通过|

#### `P1-04` 18-run 主梯子 orchestrator

|字段|内容|
|---|---|
|目标|提供一个统一入口编排 Phase 1 和 Phase 2 的 18-run 本地主梯子|
|测试先行|实验编排顺序测试；只跑 Arithmetic / 只跑 BRS / 全量运行三种模式测试|
|最小实现|`run_local_core_ladder.py` 串联 Arithmetic 6-run 与 BRS 12-run，支持断点续跑|
|重构点|把实验依赖关系抽成状态机，避免手动维护 shell 顺序|
|验收产物|18-run 执行日志、断点续跑记录、阶段汇总索引|
|阻塞条件|若不能区分 Gate 1 前后允许执行的实验范围，则不得作为主入口使用|

### 7.4 Gate 1

满足以下条件即可：

1. CoT 与 `LC-1` 都能稳定收敛。
2. `LC-2-S` 与 `Ind-2-S` 训练过程无崩溃。
3. `2-step` 明显易于 `6-step`。
4. 指标、检查点、样本预测文件都能导出。

> Arithmetic-Debug 不要求 `observe_gain` 为正。  
> 若 Arithmetic 无法完成链路验证，不得进入 BRS 主线。

### 7.5 失败回退

- 若 `LC-N` 与 `Ind-N` 结果完全相同：先查 observe 实现是否生效，再查日志是否写错字段。
- 若全部准确率接近 0：先查数据格式与课程训练，再查模型实现。

---

## 8. Phase 2：BRS 主线

### 8.1 阶段定义

|项|内容|
|---|---|
|输入|Gate 1 通过；BRS 数据与 shared 模型入口稳定|
|允许动作|只运行 `EXP-B01` 到 `EXP-B12`，并做 matched `LC-N/Ind-N` 配对统计|
|禁止动作|独立权重、Hybrid、大 `N`、第 3 个 seed、公共 benchmark|
|阶段交付物|12 个主线 runs、paired bootstrap `95% CI`、McNemar 辅助统计、Gate 2 决策报告|

### 8.2 测试场景

|类型|必须覆盖|
|---|---|
|正常|12 个 BRS runs 可执行并可聚合成 matched pair|
|边界|ID 与 OOD 都要有 `2/4/6` 步分桶结果|
|异常|配对实验缺 seed、缺预测文件、缺 matched baseline 时直接报错|
|性能|统计脚本能在本地样本规模内完成 bootstrap 与 McNemar|

### 8.3 任务卡

#### `P2-01` BRS 12-run runner

|字段|内容|
|---|---|
|目标|执行 `EXP-B01` 到 `EXP-B12` 的固定矩阵|
|测试先行|矩阵展开测试；实验 ID 与模型/seed 映射测试|
|最小实现|`run_brs_main.py` 按实验 ID 启动训练、评估、预测导出|
|重构点|统一 Arithmetic 与 BRS 的 runner 参数格式|
|验收产物|12 个 BRS runs、结果 CSV/JSON、预测文件、日志|
|阻塞条件|若任何 matched pair 缺失，不得进入统计与 Gate 2|

#### `P2-02` 配对统计与 Gate 2 分类

|字段|内容|
|---|---|
|目标|对 `observe_gain_N` 产出 pooled bootstrap `95% CI`，并给出 Positive / Weak / Negative 判定|
|测试先行|bootstrap 可重复性测试；McNemar 输入合法性测试；缺 pair 时失败测试|
|最小实现|输出 `LC-2-S vs Ind-2-S`、`LC-3-S vs Ind-3-S` 的 ID/OOD 配对统计|
|重构点|将统计核心抽象到分析层，供 Phase 3 复用|
|验收产物|配对统计报告、Gate 2 决策文件、seed 级摘要|
|阻塞条件|未先产出 pooled bootstrap `95% CI`，不得写主结论|

### 8.4 Gate 2

定义：

- `observe_gain_N = Acc(LC-N) - Acc(Ind-N)`

判定规则：

|结果|条件|后续动作|
|---|---|---|
|Positive|`LC-2-S` 或 `LC-3-S` 的 `observe_gain` 在 BRS-ID 上 pooled bootstrap `95% CI` 不跨 0，且 OOD 方向不为负|进入 Phase 3，并允许 Phase 4 条件晋级|
|Weak|点估计为正但 CI 跨 0；或 ID 为正、OOD 为负；或不同 seed 方向不一致|进入 Phase 3，但不允许直接进入独立权重|
|Negative|`N=2,3` 两组都 `observe_gain <= 0`，且 intervention 也看不到通信依赖|停止扩 `N`，停止独立权重，回到任务定义或课程训练|

### 8.5 失败回退

- 若 CoT 与 `LC-1` 在 BRS 上都明显低于可用水平：回退到 `P0-05`，重做 BRS 数据定义或 teacher steps。
- 若 `LC-N > LC-1` 但 `LC-N ≈ Ind-N`：不得写“互观察有效”，直接进入 Phase 3 查通信依赖。

---

## 9. Phase 3：intervention 与低成本推理分析

### 9.1 阶段定义

|项|内容|
|---|---|
|输入|Gate 2 为 Positive 或 Weak；主线 checkpoint 可用|
|允许动作|只做不重训优先的推理期干预、读出分析、聚合分析与 halting 分析|
|禁止动作|未经准入直接上独立权重、直接扩大 `N`、跳过 communication evidence 先写结论|
|阶段交付物|通信证据链报告、synergy 报告、halting 报告、聚合方式对比报告|

### 9.2 测试场景

|类型|必须覆盖|
|---|---|
|正常|observe-off、scramble、single-processor readout、halting、聚合对比都能运行|
|边界|单处理器全部错误但共识正确时能正确计入 `synergy_rate`|
|异常|无 checkpoint、无预测文件、无单处理器 logits 时直接报错|
|性能|intervention 默认不重训，能基于现有 checkpoint 完成分析|

### 9.3 任务卡

#### `P3-01` `intervention_tests.py`

|字段|内容|
|---|---|
|目标|建立通信内容证据链|
|测试先行|observe mask 行为测试；message scramble 保持形状/统计量测试；single-processor readout 输出测试|
|最小实现|支持 `observe-off`、`message scramble`、`single-processor readout`|
|重构点|统一推理期开关，避免每种 intervention 单独写脚本|
|验收产物|`observe_off_delta`、`scramble_delta`、单处理器准确率报告|
|阻塞条件|若 intervention 依赖重训但未明确说明，视为未完成|

#### `P3-02` `synergy.py`

|字段|内容|
|---|---|
|目标|量化“协作而不只是平均”|
|测试先行|`synergy_rate` 样本计数测试；`leave_one_out_delta` 计算测试；处理器预测分歧测试|
|最小实现|输出 `synergy_rate`、`leave_one_out_delta`、`processor_disagreement`|
|重构点|统一样本级预测读取逻辑，避免统计脚本各读一份文件|
|验收产物|synergy 报告、处理器分歧摘要、样本级案例导出|
|阻塞条件|若没有 per-sample 预测与单处理器读出文件，不得声称存在协作纠错|

#### `P3-03` `halting.py` 与聚合分析

|字段|内容|
|---|---|
|目标|补齐低成本推理分析：adaptive halting 与最终聚合方式|
|测试先行|停止条件测试；`latent mean / logit mean / logit PoE` 输出一致性测试|
|最小实现|实现 adaptive halting，比较 `latent mean`、`logit mean`、`logit PoE`|
|重构点|将聚合策略注册化，避免写死在推理主循环|
|验收产物|halting 报告、停止步数分布、聚合方式对比表|
|阻塞条件|若没有按样本记录停止步数，不得解读 recurrence 深度|

### 9.4 Phase 3 结论约束

- 若 `observe_gain > 0` 但 `observe_off_delta ≈ 0`：只能写“可能存在训练期正则化效应”，不能写“推理时使用了通信内容”。
- 若 `observe_gain > 0` 但 `scramble_delta ≈ 0`：只能写“消息语义依赖未被证明”。
- 若 `synergy_rate = 0`：不得宣称 specialization 或协作纠错。

---

## 10. Phase 4：条件晋级

### 10.1 阶段定义

|项|内容|
|---|---|
|输入|Gate 2 为 Positive；Phase 3 至少有一条通信证据为正|
|允许动作|追加第 3 个 seed；准备独立权重准入检查；写 promotion runner|
|禁止动作|未满足准入时直接训练独立权重或大 `N`|
|阶段交付物|4-run promotion 结果、seed=456 报告、独立权重 readiness 结论|

### 10.2 准入条件

必须同时满足：

1. shared 版本在 BRS 上已有 Positive 信号。
2. 10-step 压测通过。
3. `observe_off_delta` 或 `scramble_delta` 至少一个为正。
4. 接受“更大参数量使归因更难”的风险，并在报告里显式声明。

### 10.3 任务卡

#### `P4-01` 4-run promotion runner

|字段|内容|
|---|---|
|目标|执行 `EXP-C01` 到 `EXP-C04`|
|测试先行|winning `LC-N` 与 matched `Ind-N` 映射测试；seed=456 唯一性测试|
|最小实现|`run_brs_promotion.py` 执行第 3 个 seed 的配对实验|
|重构点|把 runner 对 strongest candidate 的选择逻辑抽到配置层|
|验收产物|4 个 promotion runs、更新后的配对统计|
|阻塞条件|若 strongest candidate 未明确，不得运行 promotion|

#### `P4-02` 独立权重 readiness

|字段|内容|
|---|---|
|目标|只做 readiness 判断，不默认落地训练|
|测试先行|参数量差异报告测试；readiness 判定逻辑测试|
|最小实现|生成 independent 准入检查报告|
|重构点|把独立权重的风险声明写成固定模板|
|验收产物|准入/不准入结论、风险摘要、下一阶段建议|
|阻塞条件|未通过准入条件时，不得创建独立权重实验任务|

---

## 11. Phase 5：云端阶段

### 11.1 阶段定义

|项|内容|
|---|---|
|输入|Phase 4 完成；本地 shared 结果已清楚为正|
|允许动作|容量公平性、大 `N`、公共 benchmark、独立权重云端验证|
|禁止动作|用云端实验结果反向改写本地 Phase 0-3 的措辞边界|
|阶段交付物|容量匹配单模型报告、大 `N` 报告、公共 benchmark 结果|

### 11.2 推荐顺序

1. 容量公平性
2. `LC-2-I / Ind-2-I`
3. `LC-3-I / Ind-3-I`
4. `N=5`
5. 公共 benchmark

### 11.3 云端必做约束

- 如果要写“优于单模型”，必须补容量匹配单模型基线。
- 如果要写“可扩展到更大 `N`”，必须至少完成 `N=5`。
- 如果要把结果写成论文级结论，必须追加更多 seeds。

---

## 12. 固定测试矩阵

每个任务组都必须覆盖以下四类测试：

|类别|说明|本版至少要出现的实例|
|---|---|---|
|正常|主路径是否可用|BRS 数据能生成；12-run 主线能执行；intervention 能运行|
|边界|边界输入是否稳定|`N=1 == LC-1`；`K=1/5`；10-step profiling；最小 batch 配置|
|异常|错误输入是否可诊断|OOD 泄漏、重复样本、缺预测文件、缺 matched pair、无 checkpoint|
|性能|是否满足本地预算与真实资源约束|10-step 内存/时延报告、bootstrap 运行时间、intervention 默认不重训|

---

## 13. 结果与报告编排

### 13.1 Figure / Table 顺序

1. `observe_gain` 柱状图
2. `observe_off_delta` / `scramble_delta`
3. `synergy_rate` 与 `single_processor_accuracy`
4. `repr_diversity_eval` 与 `processor_disagreement`
5. ID / OOD 对比图
6. adaptive halting 停止步数分布

### 13.2 必备报告文件

- `results/gate0/`
- `results/arithmetic_debug/`
- `results/brs_main/`
- `results/interventions/`
- `results/promotion/`
- `results/accounting/`

### 13.3 不允许的结果表达

- 先画 `N` 曲线再补证据链。
- 仅凭 `LC-N > LC-1` 就宣称互观察有效。
- 在未做容量公平性前写“架构优于单模型”。

---

## 14. 停止条件与升级条件

### 14.1 必须停止

1. Gate 2 为 Negative。
2. CoT 与 `LC-1` 在 BRS 上都不可用。
3. `observe_gain > 0` 但通信证据全部无响应，仍想继续扩 `N`。
4. 本地内存与时延约束长期不满足，仍想继续叠加复杂度。

### 14.2 可以升级

1. Gate 2 为 Positive。
2. `observe_off_delta` 或 `scramble_delta` 至少一个为正。
3. `synergy_rate > 0` 或 `leave_one_out_delta` / `processor_disagreement` 非零。
4. promotion 第 3 个 seed 不翻转方向。

---

## 15. 附录 A：最小结论模板

### A.1 正结果模板

> 在 GPT-2 124M、本地 M4 16GB、BRS 分支逻辑任务上，带互观察的 `LC-N` 相比 matched `Ind-N` 出现正的 `observe_gain`。进一步地，在推理期关闭互观察或打乱消息后，性能下降，说明模型确实使用了通信内容。部分样本上，共识结果正确而所有单处理器读出错误，提示存在协作纠错现象。

### A.2 弱正结果模板

> `LC-N` 相比 `Ind-N` 有弱正信号，但当前 `95% CI` 仍跨 0，或 OOD 方向不稳。现阶段只能说 mutual observation 值得继续探查，不能声称其已被证明有效。

### A.3 负结果模板

> 在当前 shared-weight 设计与本地预算下，`LC-N` 未显著优于 matched `Ind-N`。这说明当前版本的增益更可能来自并行/集成或训练扰动，而不是互观察协作。后续不应继续盲目扩 `N`，而应优先修改任务设计、课程训练或通信机制。

---

## 16. 附录 B：执行提醒

1. 先证明 `observe_gain`，再决定是否扩 `N`。
2. 先证明通信内容被用到，再决定是否做独立权重。
3. 先完成本地最小因果闭环，再决定是否进入云端。
4. 主线只服务一个问题：互观察本身有没有带来可检验的因果增益。
