# 问题记录 005：CLI 临时执行时存在解释器与导入路径偏差

## 所属阶段

- `Phase 2` 进入前的运行编排与回归校验

## 触发步骤

1. 直接运行 `pytest -q`
2. 直接通过 heredoc 执行临时 Python 脚本以落盘 `Gate 1` 研究豁免报告

## 现象

出现了两类命令层面的偏差：

1. `pytest -q` 命中了系统 `pyenv` 解释器，而不是项目 `.venv`，导致收集阶段报错：

```text
ModuleNotFoundError: No module named 'torch'
```

2. 临时 heredoc 脚本虽然在 `.venv` 下执行，但没有显式带上 `PYTHONPATH=src`，因此本地包导入失败：

```text
ModuleNotFoundError: No module named 'latent_consensus'
```

## 原因判断

- 当前项目采用 `src/` 布局，本地包默认依赖 `PYTHONPATH=src` 或可编辑安装。
- shell 会话未强制锁定到项目 `.venv`，直接运行命令时可能落到系统 Python。
- 这不是模型实现、数据生成或训练逻辑的问题，而是 CLI 入口约定没有被每次显式声明。

## 影响范围

- 会导致“代码本身可运行，但临时命令无法复现”的假性失败。
- 若后续在不同 shell、自动化脚本或新终端中复用命令，容易再次触发相同问题。
- 不影响已经在项目 `.venv` 中完成的真实训练与数据结果。

## 临时处理

本次已通过以下方式解除阻塞：

1. 回归测试统一改为：

```bash
source .venv/bin/activate && pytest -q
```

2. 临时脚本统一改为：

```bash
source .venv/bin/activate && PYTHONPATH=src python ...
```

## 后续动作

1. 后续所有实验与校验命令都显式写出 `.venv` 激活方式。
2. 如需进一步降低误用概率，可补一个统一入口脚本，固定 `.venv + PYTHONPATH=src`。
3. 在研究交接或运行文档中写清楚这一约定，避免新成员复现时踩坑。

## 当前状态

- 状态：已缓解
- 性质：运行入口约定问题，不是实验逻辑缺陷
