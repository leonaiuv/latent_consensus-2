# Latent Consensus 工程任务清单 v1.2.2

> 对应主手册：`latent_consensus_experiment_proposal_v_1_2_2.md`
> 目的：把实验手册收敛成可以直接开工的工程 backlog
> 当前仓库状态：仅有文档，尚未初始化代码工程

-----

## 0. 使用规则

1. 全程按 TDD 执行：先写测试，再写最小实现，再重构。
2. 每个任务完成后，必须补齐对应测试、命令入口、结果落盘路径。
3. 未完成当前阶段 Gate，不得进入下一阶段任务。
4. 所有结论性实验任务，必须同时维护 `LC-N` 与同构 `Ind-N` 对照。
5. 任何会影响结论口径的实现，必须同步更新日志字段和结果 schema。

## 1. 完成定义

|项目|完成标准|
|---|---|
|代码|存在明确入口文件，能独立运行|
|测试|新增代码有对应单测或集成测试|
|配置|参数不写死，进入 `configs/`|
|日志|关键指标进入 wandb 和本地 `results/`|
|文档|README 或任务清单状态有更新|

## 2. 开工顺序

1. `P0` 工程初始化与数据准备
2. `P1` LC-1 基线复现与 Gate 1b
3. `P2` LC-N / Ind-N 实现与主实验矩阵
4. `P3` 统计分析、容量公平性与消融

## 3. 今日即可开工的前 8 个任务

- [ ] `P0-01` 初始化 Python 工程与依赖管理
- [ ] `P0-02` 建立目录骨架与基础配置文件
- [ ] `P0-03` 写 MPS smoke test 与 10 step 资源基准脚本
- [ ] `P0-04` 建立参数量 / FLOP 账本脚本
- [ ] `P0-05` 写算术数据生成器与验收测试
- [ ] `P0-06` 写逻辑数据生成器与验收测试
- [ ] `P0-07` 写数据去重 / OOD 检查脚本
- [ ] `P0-08` 写结果 schema 与实验日志模板

-----

## 4. P0 工程初始化与数据准备

### `P0-01` 初始化 Python 工程与依赖管理

|字段|内容|
|---|---|
|目标|建立可安装、可测试、可运行的研究工程基础|
|输出|`pyproject.toml`、`.python-version`、`.gitignore`|
|测试先行|依赖安装后 `python -c "import torch, transformers, datasets, wandb"` 通过|
|完成标准|本地可创建虚拟环境，`pytest` 可运行空测试集|
|依赖|无|

### `P0-02` 建立目录骨架与基础配置文件

|字段|内容|
|---|---|
|目标|把主手册中的目录结构真正落盘|
|输出|`configs/`、`data/`、`src/`、`scripts/`、`tests/`、`results/`|
|测试先行|写一个目录存在性测试，校验关键路径和配置文件名|
|完成标准|目录与主手册一致，`base.yaml` 至少包含数据、模型、训练、日志四段|
|依赖|`P0-01`|

### `P0-03` 写 MPS smoke test 与 10 step 资源基准脚本

|字段|内容|
|---|---|
|目标|落实 `E-04`、`E-08`、`E-09`|
|输出|`scripts/smoke_test.py`、`scripts/profile_memory.py`|
|测试先行|为 profile 输出 schema 写测试，校验包含 `step_time`、`peak_memory_mb`、`device`|
|完成标准|能在 `seq_len=256, K=5` 的真实配置下连续 10 step，落盘 JSON 报告|
|依赖|`P0-01`、`P0-02`|

### `P0-04` 建立参数量 / FLOP 账本脚本

|字段|内容|
|---|---|
|目标|落实容量公平性前置条件|
|输出|`scripts/model_accounting.py`、`results/model_accounting/`|
|测试先行|为账本 JSON/CSV schema 写测试，校验 `params`、`train_flops_per_step`、`infer_flops_per_sample`|
|完成标准|能对 `LC-1`、`LC-2-S`、`LC-2-I` 产出账本|
|依赖|`P0-02`|

### `P0-05` 写算术数据生成器与验收测试

|字段|内容|
|---|---|
|目标|生成 ID/OOD 算术数据集|
|输出|`data/generate_arithmetic.py`、`tests/data/test_generate_arithmetic.py`|
|测试先行|覆盖正常样本、负数保护、结果范围、OOD 模板不重叠|
|完成标准|可生成 train/val/test/ood 四个 split，且通过随机抽样校验|
|依赖|`P0-01`、`P0-02`|

### `P0-06` 写逻辑数据生成器与验收测试

|字段|内容|
|---|---|
|目标|生成 ID/OOD 逻辑推理数据集|
|输出|`data/generate_logic.py`、`tests/data/test_generate_logic.py`|
|测试先行|覆盖唯一答案、无环、干扰项比例、OOD 关系组合差异|
|完成标准|可生成 2/4/6/8 步全量数据，并输出统计摘要|
|依赖|`P0-01`、`P0-02`|

### `P0-07` 写数据去重 / OOD 检查脚本

|字段|内容|
|---|---|
|目标|把“训练无重复、OOD 无模板泄漏”自动化|
|输出|`scripts/validate_datasets.py`|
|测试先行|构造重复样本与 OOD 泄漏假数据，确保脚本能报错|
|完成标准|对算术和逻辑两套数据都能生成验证报告|
|依赖|`P0-05`、`P0-06`|

### `P0-08` 写结果 schema 与实验日志模板

|字段|内容|
|---|---|
|目标|统一训练、评估、统计产物格式|
|输出|`src/training/metrics_schema.py`、`results/README.md`、`logs/experiment_template.md`|
|测试先行|为 metrics/result JSON schema 写测试|
|完成标准|至少覆盖 `macro_acc_id`、`macro_acc_ood`、`observe_gain`、`repr_diversity_eval`|
|依赖|`P0-02`|

### `P0-Gate`

- [ ] `P0-01` 到 `P0-08` 全部完成
- [ ] smoke test 报告已落盘
- [ ] 数据验收报告已落盘
- [ ] 参数量 / FLOP 账本已落盘

-----

## 5. P1 LC-1 基线复现与 Gate 1b

### `P1-01` LC-1 核心模块与 shape 测试

|字段|内容|
|---|---|
|目标|实现最小可用 `LC-1` 前向|
|输出|`src/models/lc1.py`、`tests/models/test_lc1.py`|
|测试先行|shape、梯度回传、`K=1/5` 行为差异|
|完成标准|`N=1` 模式前向稳定，能接 trainer|
|依赖|`P0-Gate`|

### `P1-02` 课程训练调度器

|字段|内容|
|---|---|
|目标|实现 curriculum stage 切换|
|输出|`src/training/curriculum.py`、`tests/training/test_curriculum.py`|
|测试先行|覆盖阶段推进、边界 epoch、非法配置|
|完成标准|可被 trainer 按 epoch 调用|
|依赖|`P1-01`|

### `P1-03` 统一 trainer 骨架

|字段|内容|
|---|---|
|目标|把训练、验证、checkpoint、wandb 统一起来|
|输出|`src/training/trainer.py`、`tests/training/test_trainer_smoke.py`|
|测试先行|最小 fake model + fake dataset 跑通 1 epoch|
|完成标准|能输出 best/final checkpoint 和基本日志|
|依赖|`P1-01`、`P1-02`、`P0-08`|

### `P1-04` 评估指标与 CoT baseline 入口

|字段|内容|
|---|---|
|目标|统一 CoT / LC-1 的评估口径|
|输出|`src/training/metrics.py`、`scripts/run_cot_baseline.py`|
|测试先行|`macro_acc_id`、`macro_acc_ood`、按步数统计测试|
|完成标准|CoT baseline 可单独运行并产出结果 JSON/CSV|
|依赖|`P1-03`|

### `P1-05` LC-1 训练配置与单次实验脚本

|字段|内容|
|---|---|
|目标|跑通 `EXP-02` 所需最小入口|
|输出|`configs/lc1.yaml`、`scripts/run_single.sh`|
|测试先行|配置加载测试、命令行参数覆盖测试|
|完成标准|单次训练可以用配置启动并完成评估|
|依赖|`P1-03`、`P1-04`|

### `P1-06` Gate 1b runner 与报告生成

|字段|内容|
|---|---|
|目标|自动执行 CoT + LC-1 的逻辑数据集 pilot|
|输出|`scripts/run_gate1b.py`、`results/gate1b/`|
|测试先行|报告判定逻辑测试，覆盖 `可行 / 重做数据 / 砍掉` 三条路径|
|完成标准|自动输出 Go/No-Go 决策和原因|
|依赖|`P1-04`、`P1-05`、`P0-06`|

### `P1-Gate`

- [ ] `EXP-01` 和 `EXP-02` 入口已跑通
- [ ] `Gate 1b` 报告已落盘
- [ ] `LC-1` 单测与 trainer smoke test 通过

-----

## 6. P2 LC-N / Ind-N 与主实验矩阵

### `P2-01` mutation 模块

|字段|内容|
|---|---|
|目标|实现持续突变|
|输出|`src/models/mutation.py`、`tests/models/test_mutation.py`|
|测试先行|不同 processor 输出不同；`noise=0 && dropout=0` 为恒等|
|完成标准|训练态 / 推理态行为分离|
|依赖|`P1-Gate`|

### `P2-02` observe 与 consensus 模块

|字段|内容|
|---|---|
|目标|实现观察与共识基线|
|输出|`src/models/observe.py`、`src/models/consensus.py`、对应测试|
|测试先行|`mean observe`、`mean consensus`、非法 mode 报错|
|完成标准|`shared` 和 `independent` 路径都可复用|
|依赖|`P2-01`|

### `P2-03` LC-N shared 版本

|字段|内容|
|---|---|
|目标|最先跑通 `LC-2-S`|
|输出|`src/models/lcn.py`、`tests/models/test_lcn_shared.py`|
|测试先行|`N=1 && mutation off == LC-1`，`N=2` 输出 shape 正确|
|完成标准|可接 trainer 跑 1 epoch smoke|
|依赖|`P2-02`|

### `P2-04` Ind-N 对照版本

|字段|内容|
|---|---|
|目标|建立同构无互观察对照|
|输出|`src/models/ind_n.py`、`tests/models/test_ind_n.py`|
|测试先行|确认 observe 为零向量，但 mutation 仍生效|
|完成标准|可直接替换 `LC-N` 进入同一 trainer|
|依赖|`P2-03`|

### `P2-05` LC-N independent / hybrid 版本

|字段|内容|
|---|---|
|目标|补齐 `LC-2-I`、`LC-3-I`、`LC-8-H` 路径|
|输出|`src/models/lcn.py` 扩展、`tests/models/test_lcn_modes.py`|
|测试先行|参数量差异、共享层/独立层分割、非法分割报错|
|完成标准|三种 `weight_mode` 都能初始化并训练|
|依赖|`P2-03`|

### `P2-06` 分化监控与 warning 机制

|字段|内容|
|---|---|
|目标|实现 `repr_diversity_eval`、`processor_disagreement`、`leave_one_out_delta`|
|输出|`src/training/early_stop.py`、`tests/training/test_diversity_monitor.py`|
|测试先行|高/低分化样本的预警逻辑测试|
|完成标准|warning 只报警不默认停训|
|依赖|`P2-03`、`P2-04`|

### `P2-07` 主实验配置矩阵与 sweep 脚本

|字段|内容|
|---|---|
|目标|把 `A/B/C/D/E` 组真正参数化|
|输出|`configs/*.yaml`、`scripts/run_sweep.sh`|
|测试先行|配置矩阵展开测试，确保 30 组不缺不重|
|完成标准|能按实验 ID 启动单次或批量实验|
|依赖|`P2-05`、`P2-06`|

### `P2-08` 主实验 smoke 与结果聚合

|字段|内容|
|---|---|
|目标|先跑小样本确认日志链路完整|
|输出|`scripts/smoke_lcn_pair.py`、`results/experiments/`|
|测试先行|结果聚合 schema 测试|
|完成标准|至少跑通 `LC-2-S vs Ind-2-S`、`LC-2-I vs Ind-2-I` 各 1 次|
|依赖|`P2-07`|

### `P2-Gate`

- [ ] `LC-N`、`Ind-N`、`mutation`、`consensus`、`monitor` 的核心测试通过
- [ ] 30 组实验配置已生成
- [ ] 配对 smoke run 已完成

-----

## 7. P3 统计分析、容量公平性与消融

### `P3-01` 结果汇总与 bootstrap 统计

|字段|内容|
|---|---|
|目标|把实验结果聚合成主结论输入|
|输出|`src/analysis/bootstrap_stats.py`、`tests/analysis/test_bootstrap_stats.py`|
|测试先行|固定随机种子的 bootstrap 可重复性测试|
|完成标准|输出 CI、effect size、seeds_used|
|依赖|`P2-Gate`|

### `P3-02` 图表与 strongest candidate 选择

|字段|内容|
|---|---|
|目标|生成 N 曲线、observe_gain、分化热力图|
|输出|`src/analysis/plot_n_curve.py`、`src/analysis/divergence.py`|
|测试先行|图表输入 schema 测试|
|完成标准|能自动选出 strongest candidate 并落盘|
|依赖|`P3-01`|

### `P3-03` 容量公平性 spot check

|字段|内容|
|---|---|
|目标|落实 `PM1 / PM2`|
|输出|`configs/lc1_medium.yaml`、`configs/cot_medium.yaml`、报告文件|
|测试先行|容量对照报告 schema 测试|
|完成标准|至少完成 `LC-1-Medium` 或 `CoT-Medium` 的 3-seed 对照|
|依赖|`P3-02`|

### `P3-04` ABL-06 推理策略脚本

|字段|内容|
|---|---|
|目标|完成最低成本必做消融|
|输出|`scripts/run_ablation_06.py`|
|测试先行|四种策略输出格式一致性测试|
|完成标准|可在 strongest candidate checkpoint 上直接运行|
|依赖|`P3-02`|

### `P3-05` 条件消融入口

|字段|内容|
|---|---|
|目标|为 `ABL-00/01/02/05/07/08` 预留统一入口|
|输出|`scripts/run_ablation.py`|
|测试先行|消融参数路由测试|
|完成标准|至少支持 `ABL-00`、`ABL-02`、`ABL-07`|
|依赖|`P3-04`|

### `P3-Gate`

- [ ] bootstrap CI 结果可复现
- [ ] strongest candidate 已确定
- [ ] 容量公平性对照完成
- [ ] ABL-06 已可运行

-----

## 8. 建议实施节奏

### 第 1 天

- 完成 `P0-01` 到 `P0-03`
- 先把工程跑起来，不要急着写模型

### 第 2-3 天

- 完成 `P0-04` 到 `P0-08`
- 数据和 schema 先定死，后面改动成本最低

### 第 4-6 天

- 完成 `P1-01` 到 `P1-03`
- 让 `LC-1` 与 trainer 至少 smoke 跑通

### 第 7-9 天

- 完成 `P1-04` 到 `P1-06`
- 做出第一份 `Gate 1b` 决策

### 第 10 天以后

- 进入 `P2`，先跑 shared，再补 Ind-N，再做 independent

-----

## 9. 当前建议的首个开发批次

- 批次 A：`P0-01`、`P0-02`、`P0-03`
- 批次 B：`P0-05`、`P0-06`、`P0-07`
- 批次 C：`P1-01`、`P1-02`、`P1-03`

> 建议真实开工时一次只推进一个批次；批次内严格按测试先行执行。
