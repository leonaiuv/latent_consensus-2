# Latent Consensus 实验手册 v1.2.3

## 多隐空间处理器互观察推理架构——本地优先、逻辑主线、因果收缩版

> 版本：v1.2.3  
> 日期：2026-04-06  
> 更新说明：基于 v1.2.2 与后续文献审查，重新定义本地实验边界，目标从“全量扫描”调整为“笔电可完成的最小因果闭环”。  
> 适用硬件：M4 Mac，16GB 统一内存；如有云端算力，可进入扩展阶段。  
> 核心原则：先证明 `observe_gain`，再决定是否扩 `N`、独立权重、Hybrid、Translator Agent 或论文级公共基准。  

---

## 一、v1.2.3 相比 v1.2.2 的核心变化

|变更项|v1.2.2|v1.2.3|为什么这样改|
|---|---|---|---|
|主目标|完整主线 + 菜单扩展|**本地最小因果闭环**|M4 16GB 适合做 pilot，不适合一开始跑完整矩阵|
|主数据集|算术主线，逻辑菜单|**算术降为 debug；分支逻辑/搜索升为主线**|latent reasoning 更可能在探索/分支型任务体现增益|
|本地必做实验|30 组主实验|**18-run 本地核心梯子 + 4-run 条件晋级**|先缩 scope，避免“实验做完了但结论不干净”|
|独立权重|N=2,3 主线必做|**退到条件菜单**|先确认互观察本身是否有因果增益，再加参数量和分化自由度|
|大 N 扫描|N=5,8 菜单|**云端扩展优先，不列入本地必做**|本地先看 N=2/3 是否已有稳定信号|
|容量公平性|本地强制 spot check|**改为 paper-ready/云端必做，本地只允许弱表述**|16GB 本地不适合强行做 355M 训练再得出不稳结论|
|协作证据|表示分化 + 功能分化|**新增 observe-off、message scramble、synergy rate**|仅 `LC-N > Ind-N` 还不够，需要证明“通信内容真的被用到了”|
|推理阶段分析|ABL-06 噪声策略|**保留，并新增 adaptive halting / PoE 聚合**|低成本、信息量高，适合本地先做|
|架构边界|可继续外扩|**明确不把 Translator Agent 放进 v1.2.3 主线**|现在最值钱的不是多一个组件，而是把 mutual observation 的因果效应做干净|

---

## 二、两分钟读完：v1.2.3 到底在验证什么

### 一句话

验证：**在相同训练预算、相同初始化、相同处理器数量下，带互观察的 `LC-N` 是否比无互观察的同构 `Ind-N` 在分支逻辑任务上更准确，并且这种提升是否依赖真实通信而不是纯并行/集成效应。**

### 本版本的四级结论梯子

|级别|允许得出的结论|需要满足的条件|
|---|---|---|
|L0|本地可以稳定复现 LC-1 / LC-2-S / LC-3-S|训练稳定，无 NaN/OOM，指标链路完整|
|L1|互观察对逻辑任务有**局部因果增益**|`observe_gain = Acc(LC-N) - Acc(Ind-N)` 在主测试集上为正，且 CI 不跨 0|
|L2|模型**真的使用了通信内容**|`observe_off_delta` 或 `scramble_delta` 为正，且方向稳定|
|L3|出现了**协作而不只是平均**|`synergy_rate > 0`，并伴随 `leave_one_out_delta` / `processor_disagreement` 非零|
|L4|架构优于更大单模型或大 N 可扩展|需要云端扩展、容量匹配单模型、更多 seeds，**不属于本地 v1.2.3 的默认结论**|

### 成功标准

- **最小成功**：在逻辑主数据集上，`LC-2-S` 或 `LC-3-S` 对同构 `Ind-N` 的 `observe_gain` 为正。
- **强成功**：除了 `observe_gain > 0`，还出现 `observe_off_delta > 0` 或 `scramble_delta > 0`，并且 `synergy_rate > 0`。
- **负结果也有价值**：如果 `LC-N ≈ Ind-N`，则可以比较干净地说明当前设计更像“并行轨迹/集成”，而不是“互观察协作”。这仍然是有信息量的研究结果。

### v1.2.3 明确不做什么

1. 不把显式 Translator Agent / Verbalizer Agent 放进主线。
2. 不把 `N>3` 作为本地默认目标。
3. 不在没有参数匹配单模型基线时声称“架构优于单模型”。
4. 不把算术结果写成 headline。
5. 不把注意力 Consensus 设成默认方案。
6. 不在 `LC-N > LC-1` 且 `LC-N ≈ Ind-N` 时声称“互观察有效”。

---

## 三、研究原则

### 3.1 本地优先原则

M4 16GB 的正确使用方式不是“硬拼完整论文矩阵”，而是：

1. 用最小配置建立训练、日志、评估、干预四条链路；
2. 用同构 `LC-N / Ind-N` 做因果对照；
3. 在正信号出现后再扩参数、扩 N、扩数据集、扩公共 benchmark。

### 3.2 因果配对原则

任何进入结论的 `LC-N` 都必须配一个：

- 相同 `N`
- 相同 `weight_mode`
- 相同初始化
- 相同训练预算
- 相同课程训练
- 相同 `K`
- 相同 seed

的 `Ind-N` 对照。  
**只有 Observe 打开/关闭这一处不同。**

### 3.3 任务优先级原则

- **算术**：用于 debug、确认课程训练和误差累积现象。
- **分支逻辑/搜索**：用于主结论。
- **公共 benchmark**：放到第二阶段，且优先选择更像探索/搜索的任务。

### 3.4 结论措辞原则

只有满足下面条件时，才能使用对应表述：

|表述|硬条件|
|---|---|
|“互观察有效”|`observe_gain` CI 不跨 0|
|“模型用到了通信内容”|`observe_off_delta > 0` 或 `scramble_delta > 0`|
|“出现功能分化/协作”|`synergy_rate > 0` 且 `leave_one_out_delta` 稳定非零|
|“优于单模型”|容量匹配单模型基线已完成|
|“可扩展到更大 N”|N=5 或 Hybrid 结果已完成|

---

## 四、硬件与框架决策

### 4.1 本地硬件假设

- 设备：M4 Mac，16GB 统一内存
- 本地实验目标：GPT-2 124M 级别，`N=2/3`，shared weights 为主
- 大模型 / 大 N / 参数匹配单模型：**默认视为第二阶段**

### 4.2 框架策略

**默认：PyTorch + MPS**

原因：

1. COCONUT 官方代码直接可复用；
2. 调试成本最低；
3. v1.2.3 首要目标是因果 pilot，不是框架迁移。

**MLX 仅在以下条件同时满足时才进入：**

- PyTorch MPS 单个逻辑主实验明显过慢；
- Phase 1 与 Phase 2 的 PyTorch 结果已跑通；
- 你愿意把迁移本身视为独立工程任务。

### 4.3 本地护栏参数

|项|本地默认值|备注|
|---|---|---|
|基础模型|GPT-2 124M|与 COCONUT 同级别起步|
|精度|float32|优先稳定性|
|`seq_len`|192|本地优先；云端可回到 256|
|有效 batch size|64|先保速度与内存；云端再升 128|
|micro-batch|2 或 4|以 E-09 真实峰值为准|
|梯度累积|16 或 32|维持有效 batch|
|max epoch|30|本地 pilot；验证早停|
|早停|patience = 5|验证集主指标不提升则停|
|默认 `K`|5|不在第一轮同时扫 K|
|默认 `N`|2, 3|本地主线只做这两个|
|统一内存警戒线|13GB|超过则先降 batch/seq_len|

### 4.4 E-09 真实训练准入

任何配置进入正式训练前，必须通过连续 10 step 的真实配置压测，记录：

- `torch.mps.current_allocated_memory()`
- `torch.mps.driver_allocated_memory()`
- `torch.mps.recommended_max_memory()`
- step time
- 是否触发 fallback / OOM / 梯度异常

**不要再用“1 step 不 OOM”代替可训练。**

---

## 五、数据集设计：主次关系重排

## 5.1 Arithmetic-Debug（调试集，不写 headline）

### 目标

只做三件事：

1. 确认 CoT / LC-1 / LC-N / Ind-N 的训练链路正常；
2. 观察是否存在“步数越深准确率越低”的误差累积现象；
3. 快速暴露课程训练、评估、日志记录的 bug。

### 配置

|项|值|
|---|---|
|步数|2, 4, 6|
|训练集|2,000 / step|
|验证集|200 / step|
|测试集|200 / step|
|OOD 测试集|200 / step|
|数字范围（ID）|1-99|
|数字范围（OOD）|100-199|
|运算|加、减、乘|
|最大结果|0-9999|
|总训练量|6,000|

### 成功阈值（仅用于 Gate，不用于论文主结论）

- `LC-1` 的 2-step ID 准确率应明显高于随机；
- 6-step 通常应低于 2-step；
- `LC-N` 与 `Ind-N` 都能稳定训练，不应因为实现错误导致结果完全相同或全部归零。

---

## 5.2 BRS：Branching Relational Search（主数据集）

> v1.2.3 的核心变化：**把逻辑/搜索任务升级为主线。**

### 为什么选它

你的架构最值得验证的不是“会不会把 38×2 算成 76”，而是：

- 不同处理器会不会走出不同候选路径；
- 互相看见对方状态后，能不能减少死路、漏条件、误分支；
- 这种提升是不是来自通信内容，而不是单纯多跑几个分支。

### 任务定义

每个样本给出：

- 一个目标实体对；
- 一组关系事实；
- 若干干扰分支；
- 一条唯一有效推理链。

模型需要输出最终关系或最终结论。

### 推荐关系模板

每条样本只用一种关系族，避免语言表面变化掩盖结构问题：

- `>` / `<`
- `before` / `after`
- `left_of` / `right_of`
- `contains` / `in`

### 样本示例

```text
Facts:
A > B
B > C
A > E
E > F
C > D
Query:
A ? D

Teacher steps:
A > C
A > D

Answer:
A > D
```

### 本地主线配置

|项|值|
|---|---|
|步数|2, 4, 6|
|训练集|4,000 / step|
|验证集|400 / step|
|测试集|400 / step|
|OOD 测试集|400 / step|
|总训练量|12,000|
|ID 实体数|4-10|
|OOD 实体数|11-14|
|ID 干扰项比例|30%|
|OOD 干扰项比例|50%|
|分支因子（ID）|2-3|
|分支因子（OOD）|3-4|

### 数据生成要求

1. 每条样本只存在一个正确结论；
2. 至少存在一个“看似可走但最后死路”的干扰分支；
3. 训练/验证/测试/OOD 模板严格去重；
4. OOD 不能只换数字或实体名，必须同时提高分支复杂度或关系组合难度；
5. 使用与算术相同的 `[STEP]` 样式，兼容 CoT / COCONUT 课程训练。

### Gate 判断

- 如果 CoT 和 LC-1 在 BRS 上都明显低于可用水平，则说明 **任务定义过难或 teacher steps 不合适**，先调数据，不要直接加架构复杂度。
- 如果 BRS 可用，则它取代算术成为 v1.2.3 的主汇报数据集。

---

## 六、模型、对照与主线范围

## 6.1 v1.2.3 主线只保留这四类模型

|模型|说明|是否本地必做|
|---|---|---|
|CoT|显式思维链基线|是|
|LC-1|单处理器 latent 基线|是|
|LC-2-S / LC-3-S|共享权重 + 互观察|是|
|Ind-2-S / Ind-3-S|共享权重 + 无互观察同构对照|是|

### 为什么先不把独立权重列入主线

因为独立权重会同时引入：

- 更大参数量；
- 更大显存/统一内存压力；
- 更自由的分化空间；
- 更难归因的提升来源。

v1.2.3 先回答更基础的问题：**共享权重条件下，互观察是否已经有因果增益。**

---

## 6.2 LC-N 默认实现（主线）

```python
# h_i: processor i 的 latent state
# f: shared transformer block / latent recurrence operator

for t in range(K):
    for i in range(N):
        h_i = mutate(h_i)                    # 独立 dropout + 小噪声
    for i in range(N):
        obs_i = mean(h_j for j != i)        # observe on
        h_i = f(h_i + alpha * obs_i)

h_star = mean(h_i for i in range(N))        # latent consensus
logits = lm_head(h_star)

# diagnostics
single_logits = [lm_head(h_i) for i in range(N)]
```

### 主线固定项

|项|值|
|---|---|
|`weight_mode`|shared|
|`observe`|LC-N: on；Ind-N: off|
|`K`|5|
|`alpha_init`|0.1，可学习|
|dropout|0.1|
|noise std|0.005（训练）|
|推理时默认|关闭噪声和 dropout|

### 关键约束

- `LC-N` 与 `Ind-N` 共享完全相同的初始化和训练脚本；
- `Ind-N` 不删掉处理器，不改 `N`，只把 Observe 置零；
- `N=1` 不当作 `Ind-1`，而是单独的 `LC-1` 基线。

---

## 6.3 暂不进入主线，但保留的扩展

|扩展|状态|触发条件|
|---|---|---|
|LC-2-I / Ind-2-I|条件菜单|shared 版本已出现正信号，且内存压测通过|
|LC-3-I / Ind-3-I|条件菜单|LC-2-I 稳定且有增益|
|N=5|云端优先|本地 N=2/3 已清楚为正|
|Hybrid / N=8|云端优先|N=5 或独立权重显示趋势明确|
|Translator Agent|延后到 v1.3|本版本先不做，避免混杂因子|

---

## 七、v1.2.3 实验矩阵：本地 18-run 核心梯子

## 7.1 本地必做（18 runs）

### A 组：Arithmetic-Debug，seed = 42（6 runs）

|ID|数据集|模型|
|---|---|---|
|EXP-A01|Arithmetic|CoT|
|EXP-A02|Arithmetic|LC-1|
|EXP-A03|Arithmetic|LC-2-S|
|EXP-A04|Arithmetic|Ind-2-S|
|EXP-A05|Arithmetic|LC-3-S|
|EXP-A06|Arithmetic|Ind-3-S|

### B 组：BRS 主线，seeds = 42, 123（12 runs）

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

## 7.2 条件晋级（4 runs）

> 只有当 `LC-2-S` 或 `LC-3-S` 在 BRS 上对 matched `Ind-N` 出现正信号时，才进入本阶段。

|ID|数据集|模型|seed|
|---|---|---|---|
|EXP-C01|BRS|CoT|456|
|EXP-C02|BRS|LC-1|456|
|EXP-C03|BRS|winning `LC-N`|456|
|EXP-C04|BRS|matched `Ind-N`|456|

### 为什么这样设计

- Arithmetic 只用 1 个 seed 做链路确认；
- BRS 主线先用 2 seeds 看方向；
- 只有出现正信号的组合才值得追加第 3 个 seed。

---

## 八、Gate 设计：什么时候继续，什么时候停

## Gate 0：环境准入

必须全部通过：

- MPS 可用；
- 10-step 真实配置压测无 OOM；
- 参数量 / FLOP 账本可导出；
- CoT / LC-1 代码和评估脚本都能在一个 batch 上回传梯度；
- wandb 或本地 CSV 日志正常记录。

## Gate 1：Arithmetic-Debug 通过标准

满足以下条件即可：

1. CoT 与 LC-1 都能稳定收敛；
2. `LC-2-S` 与 `Ind-2-S` 的训练过程无崩溃；
3. 2-step 明显容易于 6-step；
4. 指标、检查点、样本预测文件都能导出。

**Arithmetic-Debug 不要求 observe_gain 必须为正。**

## Gate 2：BRS 是否值得晋级

定义：

- `observe_gain_N = Acc(LC-N) - Acc(Ind-N)`

### Positive
满足任一：

- `LC-2-S` 的 `observe_gain` 在 BRS-ID 上 pooled bootstrap 95% CI 不跨 0，且 OOD 方向不为负；
- `LC-3-S` 满足同样条件。

### Weak
- 点估计为正，但 CI 跨 0；
- 或 ID 为正、OOD 为负；
- 或不同 seeds 方向不一致。

### Negative
- `N=2,3` 两组都 `observe_gain <= 0`；
- 且 intervention 诊断也看不到通信内容依赖。

### 后续动作

|结果|动作|
|---|---|
|Positive|进入条件晋级 + intervention 诊断|
|Weak|不加复杂度，先做低成本 intervention / 推理策略分析|
|Negative|停止扩 N / 停止独立权重，先回到任务定义或课程训练问题|

---

## 九、评估指标：v1.2.3 新增“通信内容证据链”

## 9.1 主指标

|指标|定义|用途|
|---|---|---|
|`macro_acc_id`|BRS 2/4/6 step ID 平均准确率|本地主指标|
|`macro_acc_ood`|BRS 2/4/6 step OOD 平均准确率|泛化稳健性|
|`observe_gain_N`|`Acc(LC-N) - Acc(Ind-N)`|互观察因果增益|
|`latency_per_sample`|推理时延|效率对比|
|`accuracy_per_param`|准确率 / 参数量|弱公平性指标|
|`accuracy_per_flop`|准确率 / FLOP|弱公平性指标|

---

## 9.2 表示 / 功能分化指标（保留）

|指标|说明|
|---|---|
|`repr_diversity_eval`|eval/no-noise 下处理器 hidden states 的平均非相似度|
|`processor_disagreement`|各处理器单独解码时的预测分歧率|
|`leave_one_out_delta`|去掉某处理器后的性能变化|

---

## 9.3 v1.2.3 新增的三条强证据

### A. `observe_off_delta`

> 同一个训练好的 `LC-N`，推理时直接把 Observe 置零，再测准确率。

定义：

```text
observe_off_delta
= Acc(LC-N, normal inference)
- Acc(LC-N, observe masked to zero at inference)
```

含义：

- 如果它明显大于 0，说明模型在推理时确实依赖互观察；
- 如果接近 0，而 `observe_gain` 却为正，说明增益可能更像训练期正则化，而不是通信内容本身。

### B. `scramble_delta`

> 破坏“消息内容”，但保留张量形状和数值范围。

做法：

- 在每个 observe step，把“其他处理器状态”按 batch 维随机打乱后再喂给当前样本；
- 这样维持统计量，但破坏样本级语义对齐。

定义：

```text
scramble_delta
= Acc(LC-N, normal inference)
- Acc(LC-N, scrambled observe messages)
```

含义：

- 如果为正，说明模型用到的是**有语义的消息**，而不是“反正加一点别的向量就好”。

### C. `synergy_rate`

> 共识结果正确，但所有单处理器单独读出都错。

定义：

```text
synergy_rate
= P( LC-N correct
     AND forall i, single_processor_i wrong )
```

含义：

- 这是最接近“协作纠错”的样本级证据；
- 比只看平均准确率更有说服力。

---

## 9.4 统计方案

### 主统计

- **样本级 paired bootstrap 95% CI**
- 对象：`observe_gain_N`、`observe_off_delta`、`scramble_delta`

### 辅助统计

- **McNemar test**：对同一测试集上的 paired predictions 做离散配对检验
- seeds 只用于稳健性，不把“3 个 seed 的 t-test”当主结论

### seed 规则

|阶段|seed|
|---|---|
|本地 pilot|42, 123|
|条件晋级|456|
|paper-ready / 云端|再加 789, 1024|

---

## 十、训练流程

## Phase 0：环境、账本、数据

交付物：

- 10-step 峰值内存记录
- 参数 / FLOP 账本
- Arithmetic-Debug 与 BRS 数据集
- CoT / LC-1 / LC-N / Ind-N 代码骨架
- intervention 脚本骨架（observe-off / scramble / single-processor readout）

## Phase 1：Arithmetic-Debug

目标：

- 把链路跑通，不追求 headline 结果。

输出：

- 6 个 debug runs
- loss 曲线
- 2/4/6-step accuracy
- 统一内存和 step time 记录

## Phase 2：BRS 主线（2 seeds）

目标：

- 只回答一个问题：`LC-N` 相比 matched `Ind-N` 是否有 `observe_gain`。

输出：

- `LC-2-S vs Ind-2-S`
- `LC-3-S vs Ind-3-S`
- bootstrap CI
- OOD 方向

## Phase 3：intervention 诊断（不需重训优先）

对 Positive 或 Weak 结果执行：

1. `observe_off_delta`
2. `scramble_delta`
3. `single_processor_i` accuracy
4. `synergy_rate`
5. `ABL-06` 推理策略
6. `ABL-09` adaptive halting
7. `ABL-11` final logit aggregation（mean / PoE）

## Phase 4：条件晋级 / 扩展

只有 Positive 才允许进入：

- 第 3 个 seed
- `LC-2-I / Ind-2-I`
- `LC-3-I / Ind-3-I`
- 公共 benchmark
- 云端容量公平性

---

## 十一、推理期低成本分析：本版本的高性价比菜单

## ABL-06：推理期扰动策略（保留）

|子实验|策略|是否重训|
|---|---|---|
|ABL-06a|无噪声、无 dropout|否|
|ABL-06b|轻量噪声 `std=0.001`|否|
|ABL-06c|5 次推理多数票|否|
|ABL-06d|轻量噪声 + 多次投票|否|

## ABL-09：Adaptive Halting（新增）

> 不改训练，只改推理停止条件。

### 默认规则

- `K_max = 8`
- 若连续两步满足  
  `mean_cosine_delta(h(t), h(t-1)) < tau`  
  则提前停止。
- 初始 `tau = 0.001`

### 记录

- 每个样本实际停止步数
- 简单样本与复杂样本的平均步数
- 提前停止后准确率是否保持

### 解读

- 若复杂样本自然停得更晚，说明 recurrence 深度可能携带真实计算意义；
- 若几乎全部样本都在 1-2 步停止，说明模型没真正学会用 recurrence。

## ABL-10：通信内容干预（新增，优先级高）

|子实验|说明|是否重训|
|---|---|---|
|ABL-10a|Observe masked to zero|否|
|ABL-10b|Scrambled observe messages|否|
|ABL-10c|只保留 1 个可见邻居（sparse observe）|否，若写成推理期开关|
|ABL-10d|禁止最后 1 步通信|否|

## ABL-11：最终聚合方式（新增，经典算法借鉴）

> 不改训练，先只改最终答案聚合。

|子实验|方式|含义|
|---|---|---|
|ABL-11a|latent mean（默认）|当前主线|
|ABL-11b|single best processor|看 consensus 是否必要|
|ABL-11c|logit mean|更接近集成|
|ABL-11d|logit Product-of-Experts|经典 PoE 思路，强化一致结论|

### 推荐顺序

先做：

1. latent mean
2. logit mean
3. logit PoE

如果 PoE 在 BRS 上更稳，而 Arithmetic 上无提升，反而是有趣结果：说明**共识方式而不是处理器数量本身**可能是关键瓶颈。

---

## 十二、条件菜单：什么时候才做独立权重

## 12.1 LC-2-I / Ind-2-I 的准入条件

必须同时满足：

1. shared 版本在 BRS 上已有 Positive 信号；
2. 10-step 压测通过；
3. `observe_off_delta` 或 `scramble_delta` 不为 0；
4. 你准备接受“更大参数量导致的归因更难”。

## 12.2 LC-3-I / Ind-3-I 的准入条件

必须先完成 LC-2-I / Ind-2-I，且：

- 方向稳定；
- 没有明显 OOM 风险；
- 你要验证的是“分化自由度是否进一步放大互观察增益”，而不是“单纯更大模型是否更强”。

---

## 十三、云端扩展：本地不默认做，但论文级必须考虑

## 13.1 容量公平性

如果你要写“多处理器架构优于单模型”，至少补其中一条：

- `LC-1-Medium`（参数量匹配）
- 或 `CoT-Medium`（若前者训练不稳）

没有这个对照，v1.2.3 只能做出**局部因果 pilot**结论，不能做“架构全局更优”结论。

## 13.2 大 N 扫描

只有本地 N=2/3 已清晰为正，才值得做：

- `LC-5-S / Ind-5-S`
- `LC-5-I / Ind-5-I`
- `LC-8-H / Ind-8-H`

## 13.3 公共 benchmark 顺序

v1.2.3 建议优先级：

1. **ProsQA / ProntoQA**
2. 自造 BRS 的更大版
3. GSM8K（仅当你明确要验证算术执行能力）

理由：如果本地 BRS 已经说明 mutual observation 更像“探索/搜索”增益，那么先去更相似的公共任务更合理，不要一上来就拿算术 stress test 充当唯一外部验证。

---

## 十四、结果图表：本版本最应该画的不是 N 曲线，而是证据链

### Figure 1
`observe_gain` 柱状图：`N=2,3`

### Figure 2
`observe_off_delta` 与 `scramble_delta`

### Figure 3
`synergy_rate` 与 single-processor accuracy

### Figure 4
`repr_diversity_eval` + `processor_disagreement`

### Figure 5
ID / OOD 对比图

### Figure 6
Adaptive halting 的停止步数分布

### Figure 7（扩展后）
shared vs independent

### Figure 8（云端）
容量公平性对照

---

## 十五、故障排除：v1.2.3 新增判断

|症状|可能原因|先做什么|
|---|---|---|
|`LC-N > LC-1` 但 `LC-N ≈ Ind-N`|更像并行/集成，不像互观察|停止扩 N，先做 `observe_off_delta` 与 `scramble_delta`|
|`observe_gain > 0` 但 `observe_off_delta ≈ 0`|训练期正则化效应，推理期不依赖通信|缩减结论，不要说“通信内容有效”|
|`observe_gain > 0` 但 `scramble_delta ≈ 0`|模型可能不在乎消息语义，只在乎额外扰动|优先做 PoE / sparse observe，不要急着上独立权重|
|`synergy_rate = 0`|没有真实协作，或单处理器已足够|不要宣称 specialization|
|Arithmetic 有提升，BRS 没提升|更可能是执行/容量效应|不要把算术写成核心成功|
|BRS 正，Arithmetic 弱|反而支持“互观察偏探索”这一解释|可继续扩逻辑类任务|
|Adaptive halting 总在 1-2 步停|模型没用 recurrence|先查课程训练，再决定是否改架构|
|Adaptive halting 总跑满 `K_max`|收敛判据过严或 latent 不稳定|调 `tau`，再看 hidden delta 分布|
|shared 全同质化|纯对称架构塌缩|先开 ABL-00：role embedding / sparse observe|
|独立权重好很多|可能是真分化，也可能只是参数更大|一定补容量公平性|

---

## 十六、v1.2.3 的“最小结论模板”

### 若结果为正

> 在 GPT-2 124M、本地 M4 16GB、分支逻辑任务 BRS 上，带互观察的 `LC-N` 相比 matched `Ind-N` 出现正的 `observe_gain`。进一步地，在推理期关闭互观察或打乱消息后，性能下降，说明模型确实使用了通信内容。部分样本上，最终共识正确而所有单处理器读出错误，提示存在协作纠错现象。

### 若结果为弱正

> `LC-N` 相比 `Ind-N` 有弱正信号，但当前 CI 仍跨 0，或 OOD 方向不稳。现阶段只能说 mutual observation 值得继续探查，不能声称其已被证明有效。

### 若结果为负

> 在当前 shared-weight 设计与本地预算下，`LC-N` 未显著优于 matched `Ind-N`。这说明当前版本的增益更可能来自并行/集成或训练扰动，而不是互观察协作。本结果直接指导后续不应继续盲目扩 N，而应优先修改任务设计、课程训练或通信机制。

---

## 十七、工程实现建议

### 目录结构

```text
latent-consensus/
├── configs/
│   ├── local_base.yaml
│   ├── arithmetic_debug.yaml
│   ├── brs_main.yaml
│   ├── lc2_shared.yaml
│   ├── lc3_shared.yaml
│   └── optional_independent.yaml
├── data/
│   ├── generate_arithmetic_debug.py
│   ├── generate_brs.py
│   └── processed/
├── src/
│   ├── models/
│   │   ├── lc1.py
│   │   ├── lcn_shared.py
│   │   └── ind_n.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── curriculum.py
│   │   └── metrics.py
│   ├── analysis/
│   │   ├── bootstrap.py
│   │   ├── intervention_tests.py
│   │   ├── synergy.py
│   │   └── halting.py
│   └── utils/
│       ├── memory_probe.py
│       └── flops_accounting.py
├── results/
├── checkpoints/
└── logs/
```

### 必须额外实现的三个脚本

1. `intervention_tests.py`
   - observe-off
   - message scramble
   - single-processor readout

2. `synergy.py`
   - `synergy_rate`
   - `leave_one_out_delta`

3. `halting.py`
   - adaptive halting
   - 停止步数统计

---

## 十八、经典算法与老论文：给 v1.3 预留的真正高价值方向

> 你提到“很多前沿研究，可能要和几十年前的算法耦合才会发生化学反应”。这句话是对的。  
> v1.2.3 不把它们全部塞进主线，但明确给出下一步优先级。

|经典思路|老工作|可对应到 Latent Consensus 的位置|
|---|---|---|
|Query by Committee|Seung, Opper, Sompolinsky (1992)|用处理器间分歧决定是否继续 latent recurrence|
|Belief Propagation / Message Passing|Pearl (1988), Kschischang et al. (2001)|把 `observe = mean(others)` 换成图结构消息传递|
|Product of Experts|Hinton (2002)|最终答案聚合，不再只做 latent mean|
|Adaptive Computation Time|Graves (2016)|让简单样本少想几步，复杂样本多想几步|
|Sparsely-Gated MoE|Shazeer et al. (2017)|不是每个处理器每步都互看，做稀疏通信|
|Hopfield / Attractor Dynamics|Hopfield (1982)|把 consensus 视为收敛过程而不是一次平均|

### v1.3 值得优先尝试的三个方向

1. **Disagreement-Gated Recurrence**  
   处理器越分歧，越继续思考；越收敛，越早停。

2. **Graph Observe**  
   从 `mean(all others)` 升级为可学习或规则化的稀疏通信图。

3. **Translator / Verbalizer Agent**  
   当前版本先不做；一旦 mutual observation 的因果效应站住，再把“隐式思考与显式表达分离”纳入第二阶段。

---

## 十九、参考文献（v1.2.3 推荐阅读顺序）

### 地基

1. Hao et al., 2024. **COCONUT**. arXiv:2412.06769  
2. Shen et al., 2025. **CODI**. arXiv:2502.21074  
3. Geiping et al., 2025. **Recurrent Depth**. arXiv:2502.05171  

### 跟你最接近的邻域

4. Du et al., 2025. **Interlat**. arXiv:2511.09149  
5. Zou et al., 2025. **LatentMAS**. arXiv:2511.20639  
6. Wang et al., 2026. **PLaT**. arXiv:2601.21358  
7. Knupp et al., 2026. **Dreamer**. arXiv:2601.21582  
8. Amos et al., 2026. **Thinking States**. arXiv:2602.08332  

### 机制与风险

9. Li et al., 2026. **Dynamics Within Latent Chain-of-Thought**. arXiv:2602.08783  
10. Liang & Pan, 2026. **Do Latent-CoT Models Think Step-by-Step?** arXiv:2602.00449  
11. Cui et al., 2026. **Weak and Strong Supervision in Latent Reasoning**. arXiv:2602.22441  
12. Zou et al., 2026. **Capabilities and Fundamental Limits of Latent CoT**. arXiv:2602.01148  

---

## 二十、最终执行建议

### 你现在真正该做的顺序

1. 先把本地 18-run 核心梯子跑通；
2. 只把 BRS 的 `observe_gain` 当主结论；
3. 必做 intervention：`observe_off_delta`、`scramble_delta`、`synergy_rate`；
4. 只在 Positive 情况下追加第 3 个 seed；
5. 只在 Positive 且 intervention 有响应时，才开独立权重；
6. 只在本地信号明确后，才用云端做容量公平性和大 N。

### v1.2.3 的判断标准

这不是“缩小野心”，而是把研究问题压到**最容易得到真答案**的形状。  
你现在最值钱的不是做出一个更复杂的图，而是拿到一句足够干净的话：

> “在本地可复现实验里，互观察本身有没有带来可检验的因果增益？”

只要这句话站住，你的方向就活了。  
如果这句话站不住，v1.2.3 也能明确告诉你：问题出在任务、训练、通信还是归因，而不是把时间继续耗在更大的矩阵上。

---

## 附录 A：v1.2.3 更新日志

- **[scope 收缩]** 从“本地尽量完整跑”改为“本地最小因果闭环”。
- **[主次重排]** Arithmetic 降为 debug；BRS（分支逻辑/搜索）升为主线。
- **[主实验重构]** 本地固定为 18-run 核心梯子 + 4-run 条件晋级。
- **[独立权重降级]** LC-2-I / LC-3-I 从主线改为条件菜单。
- **[新增强证据]** `observe_off_delta`、`scramble_delta`、`synergy_rate`。
- **[新增低成本分析]** Adaptive Halting、PoE 聚合。
- **[结论边界更严格]** 无容量匹配单模型时，不得声称“架构优于单模型”。
- **[版本边界]** Translator Agent 明确推迟到 v1.3，不纳入本版主线。