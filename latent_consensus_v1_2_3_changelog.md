# Latent Consensus v1.2.3 变更摘要

## 核心变化

1. 从“全量矩阵”改成“本地最小因果闭环”。
2. 算术从主线降级为 debug；分支逻辑/搜索升级为主线。
3. 本地必做从大矩阵收缩为 18-run 核心梯子。
4. 独立权重、大 N、容量公平性单模型基线，全部退到条件菜单或云端阶段。
5. 新增三条通信证据链：
   - `observe_off_delta`
   - `scramble_delta`
   - `synergy_rate`
6. 新增两类低成本分析：
   - Adaptive Halting
   - Product-of-Experts / logit mean 聚合
7. 明确把 Translator Agent 推迟到 v1.3，不混入 v1.2.3 主线。

## 本地执行顺序

### Phase 0
环境、10-step 真实峰值内存基准、参数/FLOP 账本、数据生成。

### Phase 1
Arithmetic-Debug：
- CoT
- LC-1
- LC-2-S
- Ind-2-S
- LC-3-S
- Ind-3-S

只跑 1 个 seed，用于调试链路。

### Phase 2
BRS 主线：
- CoT
- LC-1
- LC-2-S
- Ind-2-S
- LC-3-S
- Ind-3-S

先跑 2 个 seeds（42, 123）。

### Phase 3
若 `LC-2-S` 或 `LC-3-S` 对 matched `Ind-N` 出现正信号，则追加：
- 第 3 个 seed（456）
- observe-off
- message scramble
- synergy 分析
- adaptive halting
- final logit 聚合对比

### Phase 4
只有正信号明确时才做：
- 独立权重
- 更大 N
- 公共 benchmark
- 容量公平性云端验证

## 本版最重要的一句话

先回答：
“互观察本身，有没有比 matched no-observe baseline 更强？”

不要在这个问题没回答之前，继续扩 N、扩模块、扩参数。