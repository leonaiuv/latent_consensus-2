# Latent Consensus 实验手册 v1.2.2

# 多隐空间处理器互观察推理架构——完整实验执行方案

> 版本：v1.2.2（审查修订版：补齐因果对照 + 统计方案 + 资源校准）
> 日期：2026-04-06
> 更新说明：基于 v1.2.1，修正因果识别、参数公平性、统计口径、分化度量、对称性风险与 Gate 设计，确保“结果可跑、结论也可信”
> 角色分工：研究方向由提出者定义，本手册供 AI 工程师独立执行

-----

## 更新摘要（全版本变更追踪）

### v1.0 → v1.1

|变更项    |v1.0                   |v1.1                     |原因              |
|-------|-----------------------|-------------------------|----------------|
|处理器差异来源|仅初始化噪声 ε~N(0, 0.01)，一次性|持续突变：每步迭代独立 dropout + 小扰动|一次性噪声在共享权重下被快速抹平|
|独立权重实验 |消融实验（条件触发）             |升级为主线实验                  |独立权重是分化涌现的更强条件  |
|核心类比   |多人讨论                   |进化论：共享DNA + 持续突变 + 自然选择  |更精确描述设计意图       |

### v1.1 → v1.2

|变更项      |v1.1  |v1.2                  |原因          |
|---------|------|----------------------|------------|
|数据集      |仅算术   |新增逻辑推理                |算术错误模式太单一   |
|Consensus|简单平均  |新增注意力加权 (ABL-07)      |简单平均抹平分化信息  |
|推理阶段     |标准关闭突变|新增推理突变策略 (ABL-06)     |推理时可能同质化回归  |
|早期预警     |无     |diversity_index 阈值自动停止|避免浪费计算      |
|N=8 内存   |未估算   |分层共享 (Hybrid) + 内存预算表 |独立权重 N=8 OOM|

### v1.2 → v1.2.1（本次更新）

|变更项              |v1.2               |v1.2.1                                             |原因                                    |
|-----------------|-------------------|---------------------------------------------------|--------------------------------------|
|**实验优先级**        |42 组主线 + 消融全部列为可能执行|**必做 18 组 + 菜单按结果触发**                              |复杂度膨胀，变量太多难定位因果                       |
|**逻辑推理数据集**      |Phase 2 结束后跑 D 组   |**Phase 1 加 Gate 1b 快速验证 CoT 基线**                  |如果 GPT-2 在逻辑推理上 <30%，D 组无意义           |
|**注意力 Consensus**|ABL-07 消融          |**新增明确降级规则：训练-验证 gap >10% → 回退 mean**              |20K 训练样本下过拟合风险高                       |
|**Hybrid 分割**    |固定 6/6             |**明确标注为”v1 固定值”，后续消融为 ABL-08**                     |承认拍脑袋，但 v1 先跑通                        |
|**训练框架**         |PyTorch MPS（隐含）    |**明确框架选择指南：Phase 1 PyTorch MPS → Phase 2 可选迁移 MLX**|M4 Mac 上 MLX 性能更优但 COCONUT 代码是 PyTorch|

### v1.2.1 → v1.2.2（本次更新）

|变更项|v1.2.1|v1.2.2|原因|
|---|---|---|---|
|**因果对照**|`Ind-N` 仅在必要时触发|**所有进入结论的 `LC-N` 配置必须配对同构 `Ind-N` 对照；N=2,3 升级为必做**|否则无法区分“互观察增益”和“多轨迹/集成增益”|
|**参数公平性**|默认直接比较不同参数量模型|**新增容量公平性验证：最佳多处理器配置必须对比参数匹配单模型基线，并统一报告 `accuracy/param`、`accuracy/FLOP`**|避免把“模型更大”误判成“架构更好”|
|**统计方案**|3 种子 + `paired t-test`|**主报告改为样本级 paired bootstrap 95% CI；关键结论补到 5 seeds**|3 seeds 不足以支撑稳定显著性结论|
|**分化度量**|训练态 `diversity_index` 直接硬早停|**改为 `eval/no-noise` 的表示分化 + 功能分化双指标；预警默认只报警不硬停**|训练噪声会污染分化判断|
|**资源校准**|前向+反向 1 步不 OOM 即视为可行|**新增全配置 10 step 峰值统一内存/FLOP 基准，覆盖 `K=5 + seq_len + grad accum`**|参数量估算无法代表真实训练峰值|
|**对称性风险**|完全对称主架构，仅靠噪声打破同质化|**新增 ABL-00 轻度非对称救援：processor role embedding / sparse observe**|双重 mean 容易把差异抹平|
|**Gate 1b / 数据集**|仅 CoT、仅 2 步、1 seed 判逻辑数据集是否可行|**改为 CoT + LC-1 双基线，在 2/4/6 步的 ID/OOD 上联合判断**|单点 Gate 太粗糙，容易误砍整条实验线|

-----

## 第一部分：项目速览（2 分钟读完）

### 一句话描述

验证”多个隐空间推理模块互相观察对方的隐藏状态，是否比单个模块独自推理更准确”——以及是否会涌现自然分化。

### 核心思路

当前的隐空间推理（如 Meta 的 COCONUT）使用单个模块在连续向量空间中迭代推理，不生成中间 token。我们的假设是：如果让 N 个这样的模块同时推理，并且互相能看到对方的状态，它们可以互相纠错。更进一步：如果在每步迭代中持续注入随机差异（类似生物进化中的基因突变），训练梯度作为”自然选择”，处理器之间可能自发涌现功能分化。

### 进化论类比

```
生物进化                          Latent Consensus
───────                          ─────────────────
共享的基础 DNA        ←→          共享的 Transformer 权重（或底层共享）
DNA 复制时的随机突变   ←→          每步迭代的独立 dropout + 噪声
环境压力              ←→          互观察信号（看到别人的状态）
自然选择              ←→          训练梯度（保留有用差异，淘汰无用差异）
物种分化              ←→          处理器功能分化（如果涌现的话）
```

### 你要做的事

扫描处理器数量 N = 1, 2, 3, 5, 8，画出”N vs. 推理准确率”曲线和分化热力图。同时跑”无互观察”和”独立权重”对照组。

### [v1.2.2] 实验分层：必做 vs 菜单

```
必做主实验（30 组，约 42-48 小时）：
  A 组基线 (6 组)
  + B 组共享互观察 N=2,3 (6 组)
  + C 组独立互观察 N=2,3 (6 组)
  + D 组共享无互观察匹配对照 N=2,3 (6 组)
  + E 组独立无互观察匹配对照 N=2,3 (6 组)

必做验证（3-6 组，约 4-10 小时）：
  PM 组容量公平性 spot check
  ├── 优先：LC-1-Medium (GPT-2 medium, 355M) × 3 seeds
  └── 若 M4 16GB 无法稳定训练：降级为 CoT-Medium × 3 seeds，并强制汇报 accuracy/param 与 accuracy/FLOP

菜单（按中间检查点结果触发）：
  大 N 扫描 (每个 LC-N 必须配对同构 Ind-N) → 趋势明确时触发
  D 组逻辑推理 → Gate 1b 通过 + 最优配置确定后触发
  ABL-00~08 消融 → 各自触发条件满足时触发
```

-----

## 第二部分：实验前准备（Checklist）

> ⚠️ **以下所有准备项必须全部完成后，才能进入实验执行阶段。**

### 2.1 前置知识准备

|编号  |任务                                                   |交付物      |预计时间     |
|----|-----------------------------------------------------|---------|---------|
|P-01|精读 COCONUT 论文 (arxiv: 2412.06769)                    |1 页核心机制笔记|2-3 天    |
|P-02|通读 COCONUT 官方代码 (github.com/facebookresearch/coconut)|代码结构注释文档 |2-3 天    |
|P-03|理解多阶段课程训练（Curriculum Training）的具体实现                  |训练调度伪代码  |含在 P-02 中|

**P-01 重点关注：**

- COCONUT 如何把 token 空间推理”搬到”连续隐空间：最后一层隐藏状态直接作为下一轮输入
- 课程训练的阶段切换逻辑：逐步把 CoT 中的 token 推理步替换为隐空间迭代步
- GSM8K 上退化的原因分析（论文中是否有讨论）

### 2.2 环境搭建与框架选择

#### 2.2.1 框架选择指南 [v1.2.1 新增]

|框架             |优势                                                |劣势                               |推荐阶段                            |
|---------------|--------------------------------------------------|---------------------------------|--------------------------------|
|**PyTorch MPS**|COCONUT 官方代码直接可用；HuggingFace 生态完整；调试资源丰富          |M4 上比 MLX 慢约 30-50%；MPS 后端部分算子不支持|**Phase 1（复现 COCONUT）**         |
|**MLX**        |原生 Apple Silicon 优化；统一内存无拷贝开销；你在 kill-k-line 中已有经验|COCONUT 代码需要重写；生态较小，自定义架构可能缺算子   |**Phase 2（如果 Phase 1 训练速度成为瓶颈）**|

**推荐策略：**

```
Phase 1: PyTorch MPS
  ├── 直接基于 COCONUT 官方 PyTorch 代码改写
  ├── 减少复现风险（不引入框架迁移的 bug）
  └── Phase 1 训练量不大（6 组实验 × 50 epoch），速度不是瓶颈

Phase 1 结束后评估：
  ├── 如果单个实验 < 2 小时 → 继续用 PyTorch MPS，速度可接受
  └── 如果单个实验 > 3 小时 → 考虑将核心模块迁移到 MLX
      ├── 迁移范围：仅迁移前向传播 + 训练循环
      ├── 保留 PyTorch 的数据加载和评估管线
      └── 预计迁移时间：3-5 天（你有 MLX 经验）

Phase 2: PyTorch MPS 或 MLX（取决于 Phase 1 评估）
```

**如果选择 MLX，关键注意事项：**

- MLX 的 `nn.Module` 接口与 PyTorch 类似但不完全一致
- GPT-2 权重需要从 HuggingFace 格式转换为 MLX 格式（`mlx-community/gpt2` 可能已有）
- 自定义的 Observe、Mutation、Consensus 模块需要用 MLX 重写
- wandb 集成方式相同（wandb 不依赖框架）
- kill-k-line 中用的 MLX LoRA 经验可以复用，但 LC-N 的自定义架构比 LoRA 复杂

**[决策点] 在 Phase 0 中完成框架选择，记录在实验日志中。**

#### 2.2.2 环境搭建清单

|编号      |任务                                      |验收标准                                          |
|--------|----------------------------------------|----------------------------------------------|
|E-01    |安装 Python 3.10+, PyTorch (MPS 后端)       |`torch.backends.mps.is_available()` 返回 `True` |
|E-02    |安装 transformers, datasets, wandb        |`import` 无报错                                  |
|E-03    |下载 GPT-2 124M 预训练权重                     |`GPT2LMHeadModel.from_pretrained("gpt2")` 成功加载|
|E-04    |Smoke test: GPT-2 在 MPS 上完成 10 步前向+反向传播 |无 NaN, 无 OOM, loss 在正常范围                      |
|E-05    |配置 Weights & Biases (wandb) 项目          |`wandb.init(project="latent-consensus")` 成功   |
|E-06    |创建项目目录结构（见下方）                           |目录就绪                                          |
|E-07    |独立权重内存预算测试                              |见 2.2.3 内存预算表                                 |
|**E-08**|**[v1.2.2] 记录 Smoke Test 耗时，评估是否需要 MLX**|**单步前向+反向耗时记录**                               |
|**E-09**|**[v1.2.2] 全配置峰值内存基准**|**`seq_len=256, K=5, micro-batch, grad accum` 下连续 10 step 无 OOM，并记录峰值统一内存**|
|**E-10**|**[v1.2.2] 参数量 / FLOP 账本**|**每个配置都有 `params`, `train_flops_per_step`, `infer_flops_per_sample` 记录**|
|**E-11**|**[v1.2.2] 统计预注册**|**主指标、CI 算法、关键结论所需 seeds 数量写入实验日志**|

**Smoke Test 脚本（E-04 验收用）：**

```python
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("mps")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 10 步前向 + 反向
times = []
for i in range(10):
    start = time.time()
    inputs = tokenizer("2 + 3 = 5, 5 * 4 =", return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Step {i+1}: loss={loss.item():.4f}, time={elapsed:.3f}s")
    model.zero_grad()
    torch.mps.empty_cache()

print(f"\nSmoke test passed!")
print(f"Average step time: {sum(times)/len(times):.3f}s")
print(f"→ 如果 avg > 0.5s，Phase 2 考虑迁移 MLX")
```

#### 2.2.3 独立权重内存预算

|配置               |参数量                   |预估内存 (float32, 含梯度+优化器)|M4 16GB 可行性       |
|-----------------|----------------------|-----------------------|------------------|
|LC-1 (基线)        |124M                  |~1.5 GB                |✅ 宽裕              |
|LC-3-S (共享权重)    |124M + α              |~1.8 GB                |✅ 宽裕              |
|LC-3-I (独立权重)    |124M × 3 = 372M       |~4.5 GB                |✅ 可行              |
|LC-5-I (独立权重)    |124M × 5 = 620M       |~7.5 GB                |⚠️ micro-batch 降为 2|
|LC-8-I (独立权重)    |124M × 8 = 992M       |~12 GB                 |❌ 大概率 OOM         |
|**LC-8-H (分层共享)**|**底层共享 + 顶层独立 ≈ 400M**|**~5 GB**              |**✅ 可行**          |

**⚠️ 注意：** 上表只覆盖参数、梯度和优化器的粗估，不覆盖 `K=5` 隐空间迭代带来的激活峰值，也不覆盖 checkpoint / 日志缓存等额外开销。

**v1.2.2 验收要求：**

- E-07 只证明“理论上接近可行”，**不能**作为正式训练准入
- E-09 必须在真实训练配置下连续跑满 10 step，记录峰值统一内存和 step time
- 只有 E-07 + E-09 都通过，才能把该配置列入 Phase 2 主实验

**结论：** 独立权重 N 扫到 5，N=8 用 Hybrid 模式；但是否真正可训，以 E-09 的真实峰值基准为准。

#### 2.2.4 因果识别与容量公平性原则 [v1.2.2 新增]

> **这部分不是建议，是结论有效性的硬约束。**

1. 任何要写进结论的 `LC-N` 配置，必须配对一个同 `N`、同 `weight_mode`、同训练预算的 `Ind-N` 对照。
2. 任何要宣称“架构优于单模型”的结果，必须同时报告原始准确率、`observe_gain = LC-N - Ind-N`、`accuracy/param`、`accuracy/FLOP`。
3. 最优多处理器配置必须补跑一个参数匹配单模型基线。优先 `GPT-2 medium` 的 `LC-1-Medium`；如果硬件不允许，则降级为 `CoT-Medium`，并在结论中明确写出限制。
4. **禁止**仅根据 `LC-N > LC-1` 就下结论“互观察有效”。

**项目目录结构（E-06）：**

```
latent-consensus/
├── configs/
│   ├── base.yaml               # 共享基础配置
│   ├── lc1.yaml                # LC-1 配置
│   ├── lcn_shared.yaml         # LC-N 共享权重
│   ├── lcn_independent.yaml    # LC-N 独立权重
│   └── lcn_hybrid.yaml         # LC-N 分层共享
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generate_arithmetic.py  # 算术数据生成
│   └── generate_logic.py       # 逻辑推理数据生成
├── src/
│   ├── models/
│   │   ├── lc1.py              # LC-1 (COCONUT 复现)
│   │   ├── lcn.py              # LC-N (支持 shared/independent/hybrid)
│   │   ├── ind_n.py            # Ind-N (独立多模块对照)
│   │   ├── observe.py          # Observe 函数
│   │   ├── mutation.py         # 持续突变模块
│   │   └── consensus.py        # Consensus 函数（mean/attention）
│   ├── training/
│   │   ├── trainer.py          # 统一训练循环
│   │   ├── curriculum.py       # 多阶段课程调度
│   │   ├── early_stop.py       # 分化监控 + 预警
│   │   └── metrics.py          # 评估指标计算
│   └── analysis/
│       ├── plot_n_curve.py
│       ├── divergence.py
│       └── consensus_speed.py
├── scripts/
│   ├── run_sweep.sh
│   └── run_single.sh
├── checkpoints/
├── logs/
├── results/
└── README.md
```

### 2.3 数据集构造

#### 2.3.1 多步算术链数据集（主数据集）

|参数        |值                              |
|----------|-------------------------------|
|推理步数      |2, 4, 6, 8                     |
|每个步数的训练集大小|5,000 条                        |
|每个步数的验证集大小|500 条                          |
|每个步数的测试集大小|500 条                          |
|每个步数的 OOD 测试集大小|500 条                      |
|数字范围      |1-99                           |
|运算类型      |加、减、乘（避免除法产生小数）                |
|总数据量      |训练 20,000 + 验证 2,000 + 测试 2,000 + OOD 测试 2,000|

**数据格式示例（4 步）：**

```
输入: "23 + 15 = [STEP] 38 * 2 = [STEP] 76 - 14 = [STEP] 62 + 7 = [STEP]"
标签: "38 [STEP] 76 [STEP] 62 [STEP] 69"
最终答案: 69
```

**数据生成规则：**

- 每步结果必须在 0-9999 范围内
- 减法结果不允许为负数
- ID 训练/验证/测试集使用不同的随机种子，确保无重叠
- **[v1.2.2] OOD 测试集：** 数字范围改为 `100-199`，并额外保留一部分训练中未出现的运算模板顺序，用于检查组合泛化
- ID / OOD 的答案格式完全一致，避免把分词差异误当成泛化差异

#### 2.3.2 多步逻辑推理数据集（菜单数据集，Gate 1b 通过后启用）

> **为什么需要这个数据集：** 算术错误是系统性的——模型要么会算 38×2，要么不会，5 个处理器大概率犯同一个错。逻辑推理的错误模式更多样（走错推理路径、遗漏条件、混淆传递方向），更可能让不同处理器走不同路径从而互相纠错。

|参数        |值                              |
|----------|-------------------------------|
|推理步数      |2, 4, 6, 8                     |
|每个步数的训练集大小|5,000 条                        |
|每个步数的验证集大小|500 条                          |
|每个步数的测试集大小|500 条                          |
|每个步数的 OOD 测试集大小|500 条                      |
|实体数量      |3-10 个（随步数增加）                  |
|关系类型      |大于、包含、位于…之上/之前                 |
|总数据量      |训练 20,000 + 验证 2,000 + 测试 2,000 + OOD 测试 2,000|

**数据格式示例（4 步传递推理）：**

```
输入: "A > B [STEP] B > C [STEP] C > D [STEP] A ? D [STEP]"
标签: "A > B [STEP] A > C [STEP] A > D [STEP] A > D"
最终答案: A > D
```

**数据生成规则：**

- 每步引入一个新的关系事实或要求一步推理
- 确保每条数据有且仅有一个正确答案
- 包含干扰项：30% 的样本添加与最终问题无关的事实
- 不允许出现循环关系
- ID 训练/验证/测试集不同种子，无重叠
- **[v1.2.2] OOD 测试集：** 实体数提升到 `11-14`，并保留部分训练中未出现的关系组合与更高干扰项比例（50%）
- Gate 1b 与最终逻辑实验都要同时报告 ID 与 OOD 结果

**⚠️ [v1.2.2] 这个数据集在 Phase 0 就生成；仅在 Gate 1b（CoT + LC-1，2/4/6 步，ID/OOD 联合判定）通过后才用于训练。**

**验收标准（两个数据集共用）：**

- [ ] 随机抽查 50 条，人工验算全部正确
- [ ] 训练/验证/测试集无重复样本
- [ ] OOD 测试集与训练模板无重叠
- [ ] 各步数的答案分布大致均匀

### 2.4 超参数总表

|超参数           |值                                      |来源/理由                |
|--------------|---------------------------------------|---------------------|
|基础模型          |GPT-2 124M (HuggingFace “gpt2”)        |与 COCONUT 论文一致       |
|学习率           |1e-4                                   |COCONUT 论文配置         |
|有效 batch size |128                                    |COCONUT 论文配置         |
|micro-batch   |4（独立权重 N≥5 时降为 2）                      |M4 16GB 内存限制         |
|梯度累积步数        |32（micro-batch=2 时为 64）                |维持有效 batch size = 128|
|优化器           |Adam (β₁=0.9, β₂=0.999)                |标准配置                 |
|最大 epoch      |50                                     |COCONUT 论文配置         |
|模型选择标准        |验证集 `macro_acc_id` 最佳对应的 epoch              |避免只挑某一步数最好结果         |
|隐空间迭代次数 K     |5                                      |固定值，后续消融再扫描          |
|互观察机制         |加法融合: h_i’ = f(h_i + α·mean(h_{-i}))   |最简单的基线方案             |
|共识机制          |简单平均 mean（默认）；注意力加权（ABL-07）            |默认用最简方案              |
|α 初始值         |0.1（可学习参数）                             |初始小值                 |
|持续突变 — dropout|每个处理器独立 mask, p=0.1                    |维持多样性                |
|持续突变 — 噪声     |ε_i(t) ~ N(0, 0.005), 每步独立             |模拟 DNA 复制突变          |
|轻度非对称救援        |processor role embedding dim=16（默认关闭，ABL-00 打开）|对称架构 collapse 时救援手段|
|主报告指标          |`macro_acc_id = mean(acc_2, acc_4, acc_6, acc_8)`|防止 cherry-pick       |
|稳健性指标          |`macro_acc_ood`、`observe_gain`、`accuracy/param`、`accuracy/FLOP`|主结果必须配套报告|
|分化预警阈值         |`repr_diversity_eval < 0.02` @ epoch 5 连续 2 次 → 触发报警，不默认硬停|训练噪声会污染分化判断|
|随机种子          |pilot：42, 123, 456；关键结论复验：再加 789, 1024 → 共 5 seeds|兼顾成本与可信度|
|精度            |float32                                |MPS fp16 不够稳定        |
|梯度裁剪          |max_norm=1.0                           |防止梯度爆炸               |
|序列最大长度        |256 tokens                             |自造数据足够               |

### 2.5 实验运行矩阵 [v1.2.2 重新组织]

> **核心原则：** 主实验先建立可解释的因果对照，再谈扩展；所有菜单实验沿用“同构 `LC-N` / `Ind-N` 成对执行”的规则。

#### 2.5.1 必做主实验（30 组，约 42-48 小时）

**A 组：基线**

|实验 ID      |架构  |N|互观察|权重|种子        |预计时间   |
|-----------|----|-|---|--|----------|-------|
|EXP-01a/b/c|CoT |—|—  |— |42/123/456|3×40min|
|EXP-02a/b/c|LC-1|1|—  |— |42/123/456|3×50min|

**B 组：共享权重 + 互观察（N=2,3）**

|实验 ID      |架构    |N|互观察|权重|种子        |预计时间    |
|-----------|------|-|---|--|----------|--------|
|EXP-03a/b/c|LC-2-S|2|有  |共享|42/123/456|3×75min |
|EXP-04a/b/c|LC-3-S|3|有  |共享|42/123/456|3×100min|

**C 组：独立权重 + 互观察（N=2,3）**

|实验 ID      |架构    |N|互观察|权重|种子        |预计时间    |
|-----------|------|-|---|--|----------|--------|
|EXP-11a/b/c|LC-2-I|2|有  |独立|42/123/456|3×80min |
|EXP-12a/b/c|LC-3-I|3|有  |独立|42/123/456|3×110min|

**D 组：共享权重 + 无互观察匹配对照（N=2,3）**

|实验 ID      |架构     |N|互观察|权重|种子        |预计时间   |
|-----------|-------|-|---|--|----------|-------|
|EXP-07a/b/c|Ind-2-S|2|无  |共享|42/123/456|3×70min|
|EXP-08a/b/c|Ind-3-S|3|无  |共享|42/123/456|3×90min|

**E 组：独立权重 + 无互观察匹配对照（N=2,3）**

|实验 ID      |架构     |N|互观察|权重|种子        |预计时间   |
|-----------|-------|-|---|--|----------|-------|
|EXP-15a/b/c|Ind-2-I|2|无  |独立|42/123/456|3×75min|
|EXP-16a/b/c|Ind-3-I|3|无  |独立|42/123/456|3×105min|

**执行顺序：** A → (B vs D) → (C vs E) → 中间检查点

#### 2.5.2 必做验证：容量公平性 spot check（3-6 组，约 4-10 小时）

|实验 ID       |架构          |参数量 |种子        |触发规则|预计时间|
|------------|-------------|-----|----------|------|------|
|EXP-PM1a/b/c|LC-1-Medium  |~355M|42/123/456|优先执行；若能稳定训练，则作为参数匹配单模型基线|3×120min|
|EXP-PM2a/b/c|CoT-Medium   |~355M|42/123/456|仅当 `LC-1-Medium` 在 M4 16GB 上无法稳定训练时启用|3×90min|

**硬规则：** `PM1` 和 `PM2` 至少完成其中一条 3-seed 基线，才能在论文或结论中宣称“多处理器架构优于同等容量单模型”。

#### 2.5.3 菜单实验（按触发条件执行）

**菜单 1：大 N 扫描（中间检查点结果为”趋势明确”时触发）**

|实验 ID      |架构     |N|互观察|权重  |种子        |预计时间    |
|-----------|-------|-|---|----|----------|--------|
|EXP-05a/b/c|LC-5-S |5|有  |共享  |42/123/456|3×150min|
|EXP-25a/b/c|Ind-5-S|5|无  |共享  |42/123/456|3×145min|
|EXP-06a/b/c|LC-8-S |8|有  |共享  |42/123/456|3×250min|
|EXP-26a/b/c|Ind-8-S|8|无  |共享  |42/123/456|3×240min|
|EXP-13a/b/c|LC-5-I |5|有  |独立  |42/123/456|3×160min|
|EXP-27a/b/c|Ind-5-I|5|无  |独立  |42/123/456|3×155min|
|EXP-14a/b/c|LC-8-H |8|有  |分层共享|42/123/456|3×180min|
|EXP-28a/b/c|Ind-8-H|8|无  |分层共享|42/123/456|3×175min|

**菜单 2：逻辑推理复验（Gate 1b 通过 + 最优配置确定后触发）**

|实验 ID      |架构    |N |互观察|权重|数据集|种子        |预计时间   |
|-----------|------|--|---|--|---|----------|-------|
|EXP-21a/b/c|CoT   |— |—  |— |逻辑 |42/123/456|3×40min|
|EXP-22a/b/c|LC-1  |1 |—  |— |逻辑 |42/123/456|3×50min|
|EXP-23a/b/c|LC-N* |N*|有  |最优|逻辑 |42/123/456|视 N*   |
|EXP-24a/b/c|Ind-N*|N*|无  |最优|逻辑 |42/123/456|视 N*   |

**菜单 3：消融 / 诊断实验**

|实验 ID     |变量                                     |触发条件              |
|----------|---------------------------------------|------------------|
|**ABL-00**|**轻度非对称救援：processor role embedding / sparse observe**|**共享与独立都同质化，且 `repr_diversity_eval` 持续偏低时**|
|ABL-01    |K 扫描 (K=3,5,8)                         |主实验确定 N* 后        |
|ABL-02    |无突变 vs 有突变                             |验证持续突变必要性         |
|ABL-03    |交叉注意力 vs 加法融合                          |加法融合效果不明显时        |
|ABL-04    |稀疏观察 vs 全连接                            |N=8 完成后           |
|ABL-05    |dropout_p 扫描 (0.05, 0.1, 0.2)          |分化对 dropout 敏感时   |
|ABL-06    |推理阶段突变策略（4 种）                          |**必做，成本极低不需重训**   |
|ABL-07    |注意力 Consensus vs 简单平均                  |分化涌现但性能提升有限时      |
|**ABL-08**|**[v1.2.2] Hybrid 分割点 (3/9, 6/6, 9/3)**|**Hybrid 模式效果显著时**|

#### 2.5.4 ABL-06：推理阶段突变策略

> 不需要重新训练，只改推理脚本。主实验完成后立即执行。

|子实验    |策略     |说明                         |
|-------|-------|---------------------------|
|ABL-06a|标准关闭   |推理时无 dropout 无噪声           |
|ABL-06b|轻量噪声   |ε_i ~ N(0, 0.001)，无 dropout|
|ABL-06c|多次投票   |5 次推理不同噪声，多数票              |
|ABL-06d|噪声 + 投票|结合 b 和 c                   |

#### 2.5.5 ABL-07：注意力加权 Consensus [含降级规则]

**注意力 Consensus 设计：**

```python
class AttentionConsensus(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, processor_states):
        """
        processor_states: [N, batch, seq_len, hidden_dim]
        """
        q = self.query(processor_states.mean(dim=0))
        k = self.key(processor_states)
        v = self.value(processor_states)

        attn = (q.unsqueeze(0) * k).sum(-1)
        attn = F.softmax(attn / (self.hidden_dim ** 0.5), dim=0)

        h_star = (attn.unsqueeze(-1) * v).sum(dim=0)
        return h_star, attn
```

**⚠️ [v1.2.2] 降级规则：**

```
ABL-07 训练过程中每 5 epoch 检查：
  如果 (训练准确率 - 验证准确率) > 10%：
    → 立刻停止 ABL-07
    → 结论："注意力 Consensus 在 20K 数据量下过拟合，mean 更稳健"
    → 不要调参抢救，这不是主线问题
    → 记录过拟合出现的 epoch 和 gap 值
```

#### 2.5.6 ABL-08：Hybrid 分割点消融 [v1.2.2 延续]

> **前提：** Hybrid 模式效果明显优于 shared 模式。否则不做。

|分割     |共享层      |独立层      |类比         |
|-------|---------|---------|-----------|
|3/9    |底 3 层    |顶 9 层    |只共享最基础的特征提取|
|**6/6**|**底 6 层**|**顶 6 层**|**v1 默认值** |
|9/3    |底 9 层    |顶 3 层    |只在最高层分化    |

**⚠️ [v1.2.2] 6/6 仍然是 v1 固定值。只有当 `Hybrid > shared` 且优势在 3 seeds 上方向一致时，才值得做 ABL-08。**

-----

## 第三部分：实验执行手册

### 3.1 Phase 0：环境与数据准备

**时间：1-2 周 | 前置依赖：无**

```
执行顺序：
  P-01（读论文）和 E-01~E-11（环境搭建 + 框架评估 + 资源校准 + 统计预注册）→ 并行
         ↓ 两者都完成后
  P-02（读代码）和 2.3（构造两个数据集 + OOD split）→ 并行
         ↓ 两者都完成后
  Phase 0 完成 ✓
```

**Phase 0 交付物：**

- [ ] 环境验收：Smoke Test 通过 + 耗时记录
- [ ] **[v1.2.2] 框架决策记录：Phase 2 用 PyTorch MPS 还是 MLX**
- [ ] 全配置内存验收：候选配置在真实训练参数下连续 10 step 无 OOM，峰值统一内存与 step time 已记录
- [ ] 参数量 / FLOP 账本：每个配置都有 `params`、`train_flops_per_step`、`infer_flops_per_sample`
- [ ] 数据验收：两个数据集的统计报告 + 各 50 条人工抽查 + OOD split 去重报告
- [ ] 知识验收：COCONUT 核心机制笔记
- [ ] 统计预注册：主指标、CI 算法、关键结论所需 seeds 写入实验日志
- [ ] 代码骨架：目录结构 + 配置模板 + mutation.py + consensus.py + early_stop.py 骨架

**🚫 Gate 0 检查点：以上全部打勾才能进入 Phase 1**

-----

### 3.2 Phase 1：复现 LC-1 基线 + Gate 1b

**时间：1-2 周 | 前置依赖：Phase 0 全部通过**

**目标：** 在算术数据集上复现 COCONUT 的 LC-1，建立基准线。同时用更稳的双基线 Gate 快速验证逻辑推理数据集是否值得进入 Phase 2/3。

**执行步骤：**

```
步骤 1: 实现 LC-1 模型
  ├── 基于 COCONUT 代码改写
  ├── 核心：GPT-2 最后一层隐藏状态重新输入 Transformer 做 K 次迭代
  └── 单元测试：输出 shape 正确，梯度可回传

步骤 2: 实现训练循环
  ├── 课程训练调度器
  ├── 梯度累积（micro-batch=4, 累积 32 步）
  ├── wandb 日志 + 早期预警框架接入
  └── 模型检查点保存

步骤 3: 跑 EXP-01（CoT 基线）和 EXP-02（LC-1）
  ├── 算术数据集，各 3 个种子
  └── 测试集评估

步骤 4: [v1.2.2] Gate 1b — 逻辑推理可行性验证
  ├── 用 CoT 和 LC-1 两个基线在逻辑推理数据集上跑 5-10 epoch
  ├── pilot seeds: 42, 123（2 个种子即可）
  ├── 同时看 2/4/6 步的 ID 与 OOD 结果
  └── 判断标准见下方

步骤 5: 验收
```

**Phase 1 验收标准：**

|检查项            |通过条件                   |不通过怎么办               |
|---------------|-----------------------|---------------------|
|LC-1 训练 loss 收敛|loss 稳定下降，最终 < 初始值的 1/3|检查学习率、梯度裁剪、数据格式      |
|LC-1 2 步准确率    |≥ 60%                  |隐空间迭代实现可能有误          |
|误差累积现象         |8 步准确率明显低于 2 步         |全部一样高→任务太简单；全部一样低→bug|
|3 个种子结果稳定      |标准差 < 5%               |增加 epoch 或检查数据       |

**[v1.2.2] Gate 1b：逻辑推理可行性判断**

|判定条件（至少一个模型满足）|判断|后续行动|
|---|---|---|
|`2-step ID >= 50%` 且 `4-step ID >= 35%`，并且 `2/4/6-step macro_acc_id >= 40%`、`OOD gap <= 15pp`|✅ 可行|逻辑推理列入菜单，等最优配置确定后执行|
|`2-step ID` 在 `35-50%`，或 `4-step ID` 在 `20-35%`，或 `OOD gap > 15pp`|⚠️ 重做数据|减少实体数 / 关系类型 / 干扰项密度，重新生成后再验证|
|CoT 与 LC-1 都满足 `2-step ID < 35%` 且 `2/4/6-step macro_acc_id < 30%`|❌ 砍掉|GPT-2 124M 对该逻辑任务不足，D 组整组取消，仅保留算术主线|

**Phase 1 交付物：**

- [ ] LC-1 模型代码 + 单元测试
- [ ] CoT 和 LC-1 基线结果（算术，3 种子均值 ± 标准差）
- [ ] 训练损失曲线
- [ ] 按步数分组准确率表格
- [ ] **[v1.2.2] Gate 1b 结果记录：CoT + LC-1 在逻辑推理 2/4/6 步的 ID/OOD 准确率 + D 组 Go/No-Go 决策**

**🚫 Gate 1 检查点：LC-1 基线稳定 + Gate 1b 决策完成，才能进入 Phase 2**

-----

### 3.3 Phase 2：LC-N 实现 + N 扫描

**时间：3-4 周 | 前置依赖：Phase 1 Gate 1 通过**

**目标：** 实现 LC-N 和 Ind-N（含持续突变 + 分化监控），先跑必做主实验 30 组，建立“`LC-N` vs 同构 `Ind-N`”的因果对照，再按结果触发菜单。

**执行步骤：**

```
步骤 1: 实现持续突变模块 mutation.py
  ├── ProcessorMutation 类（见第六部分）
  └── 单元测试：不同 processor_id 输出不同；σ=0 且 dropout=0 时为恒等

步骤 2: 实现 LC-N 架构（3-5 天）
  ├── 参数化：N, weight_mode (shared/independent/hybrid)
  ├── Observe 函数 + Consensus 函数（默认 mean）
  ├── 每步调用 mutation.mutate()
  ├── 预留 `processor_role_embedding` 钩子（默认关闭，仅 ABL-00 打开）
  ├── Hybrid 模式：底 L_shared 层共享，顶 L_indep 层独立
  │   └── [v1.2.2] v1 固定 L_shared=6, L_indep=6
  └── 单元测试：N=1 且 mutation 关闭 = LC-1（关键！）

步骤 3: 实现 Ind-N 架构（1-2 天）
  ├── 复用 LC-N，Observe 返回零向量
  └── 保留持续突变

步骤 4: 分化监控接入
  ├── 表示分化：`repr_diversity_eval`（eval/no-noise hidden states）
  ├── 功能分化：`processor_disagreement` + `leave_one_out_delta`
  ├── 预警默认只报警，不自动停止
  └── 单元测试确认监控值在可预期范围内变化

步骤 5: 快速验证（1 天）
  ├── LC-2-S vs Ind-2-S、LC-2-I vs Ind-2-I 各跑 3-5 epoch，确认不崩溃
  ├── 确认 `observe_gain`、`repr_diversity_eval`、`processor_disagreement` 都能正确记录
  └── [v1.2.2] 如果选择了 MLX 框架，此时完成迁移验证

步骤 6: 必做实验执行
  ├── B vs D：LC-2-S → Ind-2-S → LC-3-S → Ind-3-S
  ├── C vs E：LC-2-I → Ind-2-I → LC-3-I → Ind-3-I
  └── 每个 3 种子

步骤 7: 中间检查点 [v1.2.2 核心决策点]
  ├── 画初步 N-性能曲线、`observe_gain` 柱状图、分化热力图
  └── 决定菜单哪些项要执行（见下方决策表）

步骤 8: 记录 strongest candidate
  ├── 为 Phase 3 的容量公平性 spot check 选出最强 `LC-N`
  └── 菜单实验按中间检查点决策执行
```

**⚡ [v1.2.2] 中间检查点决策表：**

|结果模式                   |触发菜单                 |不触发                |
|-----------------------|---------------------|-------------------|
|`LC-N > Ind-N` 且 `LC-N > LC-1`，趋势明确|菜单 1（大 N 扫描）|—|
|`LC-N > LC-1` 但 `LC-N ≈ Ind-N`|—|不要宣称“互观察有效”；优先做容量公平性验证|
|`repr_diversity_eval` 高，但 `processor_disagreement` 很低|ABL-02 / ABL-05|ABL-07|
|共享与独立都同质化|ABL-00（轻度非对称救援）|全部大 N；先不要盲目扩 N|
|独立权重优于共享，且 `observe_gain` 随 N 增大|菜单 1（只跑独立 / Hybrid 大 N）|共享权重大 N|
|全部低于 LC-1|—|全部菜单；先诊断实现、预算或课程调度问题|
|Gate 1b 通过 + 最优配置确定|菜单 2（逻辑推理复验）|—|

**Phase 2 交付物：**

- [ ] LC-N / Ind-N 代码 + mutation.py + consensus.py + early_stop.py + 全部单元测试
- [ ] 必做主实验 30 组结果 CSV
- [ ] N-性能曲线初版（N=1,2,3，共享 vs 独立，含误差线）
- [ ] `observe_gain = LC-N - Ind-N` 柱状图（N=2,3，shared / independent）
- [ ] 分化热力图（N=2,3 × shared/independent，`eval/no-noise`）
- [ ] strongest candidate 记录：进入 Phase 3 做容量公平性验证的配置
- [ ] 中间检查点决策记录：触发了哪些菜单项，为什么
- [ ] 菜单实验结果（如果触发了的话）

**🚫 Gate 2 检查点：主实验 30 组完成，中间检查点决策已做出**

-----

### 3.4 Phase 3：深度分析 + 消融实验

**时间：2-3 周 | 前置依赖：Phase 2 Gate 2 通过**

**3.4.1 必做分析：**

|分析                |方法                      |输出             |
|------------------|------------------------|---------------|
|主效应统计检验          |**paired bootstrap 95% CI（样本级）**；关键结论补到 5 seeds|带 CI 的 N-性能曲线|
|LC-N vs. Ind-N 差异 |每个 N 的 `observe_gain = LC-N - Ind-N`|“互观察增益”柱状图|
|**表示分化分析**        |**`repr_diversity_eval` + 处理器间成对余弦相似度（eval/no-noise）**|**分化热力图** ← 最重要|
|**功能分化分析**        |**`processor_disagreement` + `leave_one_out_delta`**|功能分化条形图|
|共享 vs 独立 vs 分层分化对比|同 N 下不同权重模式的表示/功能分化双指标     |分化程度柱状图        |
|按步数分组曲线族          |分别画 2/4/6/8 步           |4 条曲线叠加图       |
|共识速度              |余弦相似度随迭代步 t 变化          |收敛曲线图          |
|α 参数分析            |训练后 α 值                 |α 值表           |
|突变效应分析            |有/无突变的差异                |多样性维持曲线        |
|容量公平性验证          |最佳 `LC-N` vs `LC-1-Medium / CoT-Medium` + `accuracy/param` + `accuracy/FLOP`|容量公平性对照表|
|计算效率              |各 N 训练/推理时间 + FLOP      |帕累托曲线          |
|错误案例分析            |LC-N 对 LC-1 错（反之亦然）     |各 10 个案例       |

**如果 D 组（逻辑推理）已执行，额外分析：**

|分析            |方法                    |输出          |
|--------------|----------------------|------------|
|算术 vs 逻辑推理增益对比|同配置两数据集的 `observe_gain = LC-N - Ind-N`|任务类型 × 增益交互图|
|错误类型分析        |LC-N 纠正的错误分类          |错误类型分布饼图    |

**3.4.2 必做验证：容量公平性**

|验证项|触发条件|通过标准|
|---|---|---|
|`LC-1-Medium` 或 `CoT-Medium` 3-seed 基线|Phase 2 strongest candidate 已确定|最佳 `LC-N` 至少在一个容量匹配单模型基线上保持方向一致优势，或明确承认容量混杂风险|
|关键结论 5-seed 复验|准备写入摘要/结论的 H1/H3/H6/H7|新增 2 seeds 后方向不翻转，bootstrap CI 与 seed-level 结果一致|

**3.4.3 消融实验（按触发条件）：**

|实验                   |触发条件          |备注                              |
|---------------------|--------------|--------------------------------|
|ABL-00: 轻度非对称救援   |共享与独立都同质化     |processor role embedding / sparse observe|
|ABL-01: K 扫描         |确定 N* 后       |—                               |
|ABL-02: 无突变 vs 有突变   |验证突变必要性       |—                               |
|ABL-03: 交叉注意力        |加法融合效果不明显     |—                               |
|ABL-04: 稀疏观察         |N=8 完成后       |—                               |
|ABL-05: dropout_p 扫描 |分化对 dropout 敏感|—                               |
|**ABL-06: 推理突变策略**   |**必做（成本极低）**  |**不需重训**                        |
|ABL-07: 注意力 Consensus|分化但性能有限       |**降级规则：train-val gap >10% → 停止**|
|ABL-08: Hybrid 分割点   |Hybrid 效果显著   |3/9, 6/6, 9/3                   |

**3.4.4 阶段二复验（可选）：**

|触发条件             |任务                     |硬件       |
|-----------------|-----------------------|---------|
|N-性能曲线有明确趋势或分化被确认|GSM8K 上复验 LC-1 vs LC-N*|Colab Pro|
|同上               |或 ProsQA 上复验           |M4 Mac   |

**Phase 3 交付物：**

- [ ] 完整统计分析报告
- [ ] 容量公平性验证报告（必做）
- [ ] 关键结论 5-seed 复验记录（如果要写入摘要/标题）
- [ ] ABL-06 推理策略对比报告（必做）
- [ ] 其他消融结果（如果触发）
- [ ] 分化涌现报告（如果观察到）
- [ ] 核心发现总结（3-5 条）

-----

### 3.5 Phase 4：论文撰写

**时间：2-3 周 | 前置依赖：Phase 3 完成**

|章节          |内容                                            |页数 |
|------------|----------------------------------------------|---|
|Abstract    |动机 + 方法 + 核心结果 + 意义                           |0.3|
|Introduction|隐空间推理现状 → COCONUT 局限 → 互观察假设 → 持续突变 → 贡献      |1  |
|Related Work|COCONUT, Quiet-STaR, iCLP, JEPA, MoE, Ensemble|1  |
|Method      |LC-N + Observe/Consensus + 持续突变 + 分层共享        |2  |
|Experiments |设置 + N-曲线 + 分化分析 + 消融 + (逻辑推理对比)              |3.5|
|Discussion  |解读 + 局限 + 进化论类比反思 + 未来方向                      |1  |
|Conclusion  |总结                                            |0.5|

**核心图表：**

|图表          |内容                             |
|------------|-------------------------------|
|Figure 1    |LC-N 架构示意图（含持续突变 + 分层共享）       |
|**Figure 2**|**分化热力图** ← 可能全文最重要            |
|Figure 3    |N-性能曲线（共享 vs 独立 vs 分层）         |
|Figure 4    |按步数分组的 N-性能曲线族                 |
|Figure 5    |LC-N vs. Ind-N 互观察增益           |
|Figure 6    |余弦相似度随迭代步变化（有/无突变）             |
|Figure 7    |算术 vs 逻辑推理增益对比（如果 D 组执行了）      |
|Figure 8    |推理阶段突变策略对比 (ABL-06)            |
|Table 1     |完整实验结果（含 bootstrap CI 与 seeds 数）   |
|Table 2     |消融结果                           |
|Table 3     |容量公平性与注意力 Consensus 结果（如执行）|

-----

## 第四部分：数据记录规范

### 4.1 训练过程（每个 epoch，wandb 自动记录）

|指标                        |频率     |说明      |
|--------------------------|-------|--------|
|train/loss                |每步     |训练损失    |
|train/lr                  |每步     |学习率     |
|val/loss                  |每 epoch|验证损失    |
|val/macro_acc_id          |每 epoch|ID 主指标  |
|val/accuracy_{2,4,6,8}step|每 epoch|按步数准确率  |
|val/macro_acc_ood         |每 epoch|OOD 主指标 |
|meta/epoch                |每 epoch|当前 epoch|
|meta/curriculum_stage     |每 epoch|课程阶段    |
|meta/unified_memory_peak_mb|每 epoch|统一内存峰值 |
|meta/epoch_time_sec       |每 epoch|训练时间    |
|meta/diversity_warning_triggered|每 epoch|是否触发分化预警|

### 4.2 LC-N 特有指标（N > 1）

|指标                                   |频率       |说明                |
|-------------------------------------|---------|------------------|
|observe/alpha                        |每 epoch  |α 值               |
|eval/repr_diversity                  |每 5 epoch|`eval/no-noise` 表示分化指标|
|observe/processor_cosine_sim         |每 5 epoch|`eval/no-noise` N×N 余弦相似度矩阵|
|observe/consensus_cosine_sim_per_step|每 5 epoch|各迭代步的平均余弦相似度      |
|observe/processor_norm_std           |每 5 epoch|范数标准差             |
|functional/processor_disagreement    |每 5 epoch|处理器预测分歧率          |
|functional/leave_one_out_delta       |每 5 epoch|去掉单处理器后的性能变化      |
|mutation/diversity_index_train       |每 epoch  |训练态平均成对距离（仅辅助观察）|
|mutation/dropout_effect              |每 5 epoch|突变前后 L2 范数        |
|consensus/attention_weights          |每 5 epoch|仅 ABL-07：各处理器注意力权重|
|consensus/weight_entropy             |每 5 epoch|仅 ABL-07：注意力权重熵   |

### 4.3 检查点策略

|事件        |内容         |命名                     |
|----------|-----------|-----------------------|
|验证准确率新高   |完整模型 + 优化器 |`{exp_id}_best.pt`     |
|训练结束      |完整模型 + 优化器 |`{exp_id}_final.pt`    |
|每 10 epoch|仅权重        |`{exp_id}_epoch{n}.pt` |
|分化预警快照    |完整模型 + 预警原因|`{exp_id}_warning.pt`|

### 4.4 测试集评估

|数据       |格式                                                   |用途    |
|---------|-----------------------------------------------------|------|
|预测结果     |CSV: sample_id, split, dataset, n_steps, pred, gold, correct|错误分析  |
|推理过程     |JSON: 每步隐藏状态统计量                                      |深度分析  |
|处理器状态    |JSON: 各处理器各步余弦相似度 + disagreement + leave-one-out     |分化案例分析|
|ABL-06 结果|CSV: sample_id, strategy, pred, correct              |推理策略对比|
|整体指标     |JSON: `macro_acc_id`, `macro_acc_ood`, `observe_gain`, `latency`, `accuracy/param`, `accuracy/FLOP`|汇总|
|统计摘要     |JSON: bootstrap CI, seeds_used, strongest_candidate  |主结论支撑|

### 4.5 实验日志

```markdown
## EXP-03a: LC-2-S, seed=42, 算术
- 开始: 2026-04-20 14:30
- 结束: 2026-04-20 16:15
- 状态: ✅ / ❌ / ⚠️ 触发分化预警 / 🛑 人工停止
- repr_diversity_eval @ epoch 5: [值]
- functional_disagreement @ epoch 5: [值]
- observe_gain（相对 Ind-2-S）: [值]
- 分化观察: [表示分化 / 功能分化趋势]
- wandb: [链接]
- 备注: [非预期观察]
```

-----

## 第五部分：实验后复盘手册

### 5.1 结果汇总 CSV

```csv
exp_id,architecture,N,observe,weight_mode,mutation,dataset,seed,acc_2,acc_4,acc_6,acc_8,macro_acc_id,macro_acc_ood,observe_gain,repr_div_eval,func_disagreement,capacity_matched,warning_triggered,time_min
EXP-02a,LC,1,no,shared,no,arith,42,85.2,72.4,58.1,43.2,64.7,51.3,N/A,N/A,N/A,no,no,48
EXP-03a,LC,2,yes,shared,yes,arith,42,87.1,74.3,61.2,46.8,67.4,54.8,2.7,0.15,0.11,no,no,73
EXP-11a,LC,2,yes,indep,yes,arith,42,88.5,76.1,63.9,49.2,69.4,57.2,4.1,0.32,0.19,no,no,78
...
```

### 5.2 假设检验

|假设                      |检验                       |标准                    |
|------------------------|-------------------------|----------------------|
|H1: ∃ N>1 使 `LC-N > matched Ind-N`|paired bootstrap 95% CI + 关键结论 5 seeds|CI 不跨 0，且新增 2 seeds 后方向不翻转|
|H1b: 最优 `LC-N` 优于参数匹配单模型|容量公平性 spot check|方向一致，或明确标注容量混杂|
|H2: 步数越多互观察增益越大            |`observe_gain` 与步数相关性          |Pearson r > 0.7 或单调上升趋势|
|H3: 表示分化涌现              |epoch 1 vs 50 的 `repr_diversity_eval`|显著上升                  |
|H3b: 功能分化真实存在           |`leave_one_out_delta` 非零且稳定       |至少一个处理器的边际贡献方向稳定|
|H4: 非单调性                |N-曲线是否有下降段               |目视 + 二次拟合             |
|H5: 突变维持多样性             |有/无突变的表示/功能分化衰减对比        |有突变衰减更慢               |
|H6: 逻辑增益 > 算术增益         |同配置两数据集 `observe_gain` 对比     |bootstrap CI 不跨 0（仅 D 组执行时）|
|H7: 注意力 Consensus > mean|ABL-07 对比                |bootstrap CI 不跨 0（仅 ABL-07 执行时）|

### 5.3 结论决策树

```
N-性能曲线呈什么形态？
│
├─ 明确上升（H1 成立）
│   ├─ LC-N > Ind-N 且容量匹配基线也落后 → "互观察有效，且不只是容量效应"
│   │   ├─ 独立 > 共享 → "分化是关键"
│   │   │   ├─ 逻辑增益 > 算术增益 → "多样错误模式下更有效"
│   │   │   └─ 逻辑 ≈ 算术 → 效果与任务无关
│   │   └─ 共享 ≈ 独立 → "互观察本身有效，不依赖分化"
│   ├─ LC-N > Ind-N 但容量匹配基线更强 → "结果可能主要来自参数量"
│   └─ LC-N ≈ Ind-N → "更像集成效应，不足以宣称互观察有效"
│       └─ 先做容量公平性验证，再决定是否跑 ABL-07
│
├─ U 型 → 找 N_min，非常有趣
│
├─ 平坦
│   ├─ 有表示分化无功能分化 → 噪声大于 specialization，先做 ABL-02 / ABL-05
│   ├─ 有功能分化无提升 → ABL-07（mean 是否浪费分化信息？）
│   └─ 无分化 → ABL-00 或加大突变重试
│
└─ 下降
    └─ 诊断：同质化/互确认偏误/噪声过大
        └─ 跑 ABL-06 看推理时关闭噪声是否改善
```

### 5.4 复盘报告模板

1. 实验是否按计划执行？调整了什么？
1. 意外发现
1. 分化涌现观察：表示分化 vs 功能分化，条件、时间、程度
1. 持续突变效果
1. [v1.2.2] 因果对照是否充分：每个结论是否都有匹配 `Ind-N`
1. [v1.2.2] 容量公平性结果：参数匹配单模型对照给出的约束是什么
1. [v1.2.2] 必做 vs 菜单执行情况：菜单触发了哪些？决策是否正确？
1. [v1.2.2] 框架选择回顾：PyTorch MPS vs MLX，实际体验如何？
1. 任务类型影响（如果 D 组执行了）
1. Consensus 方式影响（如果 ABL-07 执行了）
1. 推理阶段策略（ABL-06 结果）
1. 失败与分化预警记录
1. 资源消耗：实际 vs 预估
1. 如果重做会改什么？

-----

## 第六部分：架构技术规格

### 6.1 LC-1（基线，复现 COCONUT）

```
输入 token 序列
    ↓
[嵌入层]
    ↓
[Transformer 编码] → 隐藏状态 h
    ↓
[隐空间迭代]：h → f(h) → f(f(h)) → ... → h*  （K 次）
    ↓
[解码层] → 输出 token
```

### 6.2 LC-N（N 处理器互观察架构）

```
输入 token 序列
    ↓
[嵌入层]
    ↓
[Transformer 编码] → h₀
    ↓
    ├─→ [处理器 1]：h₁ = f(h₀) + ε₁(0) [+ r₁，仅 ABL-00]
    ├─→ [处理器 2]：h₂ = f(h₀) + ε₂(0) [+ r₂，仅 ABL-00]
    └─→ [处理器 N]：hN = f(h₀) + εN(0) [+ rN，仅 ABL-00]
    ↓
[互观察迭代 × K 次]：
    每步 t，每个处理器 i：

        A. 持续突变
           h_i(t) = dropout_i(h_i(t)) + ε_i(t)

        B. 互观察
           observed = α · mean(h_j(t), j ≠ i)

        C. 迭代
           h_i(t+1) = f_mode(h_i(t) + observed)
           │ shared:      f（全部共享）
           │ independent: f_i（各自独立）
           │ hybrid:      底层 f_shared + 顶层 f_i_top
    ↓
[共识层]：
    │ mean:      h* = mean(h₁(K),...,hN(K))
    │ attention: h* = AttentionConsensus(...)  [ABL-07]
    ↓
[解码层] → 输出 token
```

**v1.2.2 说明：**

- 主线实验默认保持对称架构，`r_i` 关闭，用来测试“仅靠持续突变是否足以产生分化”
- 如果 shared / independent 都出现持续同质化，才开启 ABL-00 的轻度非对称救援
- 分化结论以 `eval/no-noise` 的表示分化和功能分化共同判断，不以训练态噪声直接下结论

**权重模式：**

|模式         |底层|顶层|参数量 (N=8)  |适用               |
|-----------|--|--|-----------|-----------------|
|shared     |共享|共享|124M       |基础方案             |
|independent|独立|独立|124M × N   |N≤5              |
|hybrid     |共享|独立|~400M (N=8)|N=8，分割点 v1 固定 6/6|

**mutation.py：**

```python
class ProcessorMutation(nn.Module):
    def __init__(self, n_processors, hidden_dim, dropout_p=0.1, noise_std=0.005):
        super().__init__()
        self.n_processors = n_processors
        self.dropout_p = dropout_p
        self.noise_std = noise_std
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=dropout_p) for _ in range(n_processors)
        ])

    def forward(self, h_i, processor_id, inference_noise_std=0.0):
        if self.training:
            h_mutated = self.dropouts[processor_id](h_i)
            noise = torch.randn_like(h_mutated) * self.noise_std
            h_mutated = h_mutated + noise
        else:
            h_mutated = h_i
            if inference_noise_std > 0:
                noise = torch.randn_like(h_i) * inference_noise_std
                h_mutated = h_mutated + noise
        return h_mutated
```

**consensus.py：**

```python
class MeanConsensus(nn.Module):
    def forward(self, processor_states):
        return processor_states.mean(dim=0)


class AttentionConsensus(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, processor_states):
        q = self.query(processor_states.mean(dim=0))
        k = self.key(processor_states)
        v = self.value(processor_states)
        attn = (q.unsqueeze(0) * k).sum(-1)
        attn = F.softmax(attn / (self.hidden_dim ** 0.5), dim=0)
        h_star = (attn.unsqueeze(-1) * v).sum(dim=0)
        return h_star, attn


def build_consensus(mode, hidden_dim):
    if mode == "mean":
        return MeanConsensus()
    elif mode == "attention":
        return AttentionConsensus(hidden_dim)
    else:
        raise ValueError(f"Unknown consensus mode: {mode}")
```

**early_stop.py：**

```python
class DiversityMonitor:
    def __init__(self, warn_threshold=0.02, check_epoch=5, patience=2):
        self.warn_threshold = warn_threshold
        self.check_epoch = check_epoch
        self.patience = patience
        self.warn_count = 0

    def compute_repr_diversity(self, processor_states):
        N = processor_states.shape[0]
        flat = processor_states.flatten(start_dim=1)
        total_sim, count = 0.0, 0
        for i in range(N):
            for j in range(i + 1, N):
                sim = F.cosine_similarity(flat[i], flat[j], dim=0).mean()
                total_sim += sim.item()
                count += 1
        return 1.0 - (total_sim / count if count > 0 else 1.0)

    def should_warn(self, epoch, repr_diversity):
        if epoch >= self.check_epoch and repr_diversity < self.warn_threshold:
            self.warn_count += 1
        else:
            self.warn_count = 0
        if self.warn_count >= self.patience:
            return True, (
                f"DIVERSITY_WARNING: repr_diversity={repr_diversity:.4f} "
                f"< {self.warn_threshold} for {self.warn_count} checks"
            )
        return False, None
```

### 6.3 Ind-N（消融对照）

与 LC-N 完全相同，Observe 返回零向量。持续突变保留。用 `observe=True/False` 开关控制。

**v1.2.2 硬规则：** 任何要进入结论的 `LC-N`，都必须有同构 `Ind-N` 成对结果。

-----

## 第七部分：故障排除手册

|症状               |原因              |排查               |解决                             |
|-----------------|----------------|-----------------|-------------------------------|
|loss 不下降         |lr 太大/太小        |loss 曲线形态        |震荡→降 lr；平坦→升 lr                |
|loss NaN         |梯度爆炸            |打印梯度范数           |降裁剪到 0.5                       |
|OOM              |micro-batch/N 太大|内存监控             |micro-batch→2→1，或用 hybrid      |
|10 step 基准通过但正式训练仍 OOM|激活峰值或日志缓存被低估|对比 E-09 与正式训练配置|启用 activation checkpoint / 降日志频率 / 降 seq_len|
|LC-N = LC-1      |α 没生效           |打印 α 和 Observe 输出|检查代码接入                         |
|LC-N > LC-1 但 ≈ Ind-N|更像集成增益而非互观察|看 `observe_gain` CI|不要宣称互观察有效；先做容量公平性验证|
|处理器全同            |突变太弱或对称性过强      |`repr_diversity_eval`、`processor_disagreement`|先做 ABL-00；再考虑 dropout→0.2, noise→0.01|
|突变导致不稳定          |噪声太大            |loss 波动          |noise→0.001                    |
|独立权重 N≥5 OOM     |参数量太大           |总参数量             |hybrid 或降 micro-batch          |
|表示分化很高但功能分化很低|噪声制造了差异但没有 specialization|对比 `repr_diversity_eval` 与 `leave_one_out_delta`|优先做 ABL-02 / ABL-05，不要直接上 ABL-07|
|容量匹配单模型 ≥ 最优 LC-N|主收益可能来自参数量      |看 PM 组结果         |缩减结论，只保留“多处理器值得探索”而非“架构更优”|
|注意力 Consensus 过拟合|参数多数据少          |train-val gap    |**gap >10% → 停止，回退 mean**      |
|逻辑推理 ID 可用但 OOD 很差|数据集存在捷径        |ID/OOD gap         |降干扰项或改关系分布，重做 Gate 1b|
|大量分化预警出现         |同质化严重或监控阈值过严    |预警比例             |先人工抽样，再决定提高突变或放宽阈值|
|MPS 算子不支持        |MPS 限制          |报错信息             |`PYTORCH_ENABLE_MPS_FALLBACK=1`|
|训练太慢             |数据瓶颈            |profiling        |num_workers=0, 预 tokenize      |
|课程切换 loss 飙升     |切换太激进           |恢复速度             |减慢切换                           |
|验证准确率归零          |评估 bug          |手动检查 5 条         |修复评估代码                         |

-----

## 第八部分：参考文献与资源

### 必读

1. **COCONUT** — Hao et al., 2024. “Training Large Language Models to Reason in a Continuous Latent Space.” arXiv:2412.06769

### 参考代码

1. COCONUT 官方: github.com/facebookresearch/coconut
1. 第三方复现: github.com/lucidrains/coconut-pytorch
1. GPT-2: HuggingFace `gpt2` (124M)
1. **[v1.2.2] MLX GPT-2: mlx-community/gpt2（如果迁移 MLX 时使用）**

### 相关论文

- Quiet-STaR (Zelikman et al., 2024)
- SoftCoT (2025)
- JEPA (LeCun, 2022)
- “Reasoning by Superposition” (2025)

-----

## 附录 A：版本更新日志

### v0.1 初版

- LC-2 为核心验证目标，固定 N=2

### v0.2 N 作为扫描变量

- N 扫描 (1,2,3,5,8) + Ind-N 对照 + H4 非单调假设

### v0.3 模型选择 + 硬件适配

- GPT-2 124M + M4 Mac 16GB + 两阶段策略

### v1.0 工程师可执行版

- 30 组实验矩阵、Gate 检查点、数据记录规范、故障排除

### v1.1 持续突变架构修正

- 持续突变（独立 dropout + 每步噪声）替代一次性噪声
- 独立权重升级为主线
- mutation.py + H3b/H5 + diversity_index

### v1.2 全面优化版

- 逻辑推理数据集 + 注意力 Consensus (ABL-07) + 推理突变策略 (ABL-06)
- 早期预警 + Hybrid 分层共享 + 内存预算表
- 42 组实验 + H6/H7

### v1.2.1 工程师反馈整合版

- **[复杂度控制] 实验分层：必做 18 组 + 菜单按结果触发。** 解决 v1.2 的复杂度膨胀问题，变量越多越难定位因果
- **[Gate 1b] Phase 1 快速验证逻辑推理数据集可行性。** CoT 基线 <30% → 砍掉 D 组，不等到 Phase 2 结束才发现
- **[降级规则] 注意力 Consensus 过拟合时的明确退出条件。** train-val gap >10% → 停止 ABL-07，回退 mean，不调参抢救
- **[Hybrid 分割] 明确标注 6/6 为”v1 固定值”。** 新增 ABL-08 用于后续确认最优分割点（仅 Hybrid 效果显著时触发）
- **[框架选择] 新增 PyTorch MPS vs MLX 选择指南。** Phase 1 用 PyTorch（COCONUT 代码直接可用），Phase 2 按速度评估是否迁移 MLX

### v1.2.2 审查修订版 ← 当前版本

- **[因果对照] `Ind-N` 从菜单升级为主线配对对照。** 所有进入结论的 `LC-N` 都必须有同构 `Ind-N`
- **[容量公平性] 新增参数匹配单模型基线与 `accuracy/param`、`accuracy/FLOP` 汇报要求。** 不再把参数量差异混进架构结论
- **[统计口径] 主检验改为 paired bootstrap 95% CI，关键结论补到 5 seeds。** 避免 3 seeds 上过度宣称显著性
- **[分化度量] 从训练态 `diversity_index` 硬早停，改为 `eval/no-noise` 表示分化 + 功能分化双指标。**
- **[资源校准] 新增 E-09 真实训练配置 10 step 峰值内存基准。** 不再用“1 step 不 OOM”代替正式可训
- **[对称性风险] 新增 ABL-00 轻度非对称救援。** shared / independent 都同质化时先救架构，再决定是否扩 N
- **[Gate 1b] 改为 CoT + LC-1 双基线，在 2/4/6 步 ID/OOD 上联合判断。** 避免用单点阈值误砍整条逻辑线
