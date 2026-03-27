# EvoKernel Prototype

EvoKernel Prototype 是一个可运行的 Python 原型，实现了论文
*Toward Cold-Start Drafting and Continual Refining: A Value-Driven Memory
Approach with Application to NPU Kernel Synthesis* 中 EvoKernel 工作流的核心
结构。

这个仓库不是论文结果复现工程，而是一个偏工程化的原型，重点验证以下能力：

- 基于任务描述生成 kernel 候选代码
- 可插拔后端执行
- 多阶段验证
- 记忆持久化与复用
- 价值驱动的 drafting / refining 闭环

当前第一个落地后端是 `cpu_simd`，候选 kernel 会被编译成共享库，并通过
`ctypes` 加载执行。

## 当前状态

当前原型已经实现，并且可以在本地完整验证。

已完成的主要模块：

- 基于 `uv` 的 Python 项目骨架
- 带校验的配置加载
- memory store、seed knowledge 注入、state signature、recall policy、Q-value 更新
- OpenAI-compatible 生成器客户端与 prompt builder
- CPU SIMD backend 协议与 benchmark task 注册表
- anti-hack、correctness、compile/runtime error、latency 聚合与 bottleneck 诊断
  verifier
- drafting / refining orchestrator 与 run report
- CLI 入口与本地 deterministic 验证路径
- CPU SIMD 端到端集成测试与共享 memory 复用测试

v1 里的 benchmark 任务：

- `vector_add`
- `reduce_sum`
- `matmul_tiled`
- `layernorm`

## 目录结构

```text
src/evokernel/
  backend/        后端协议、CPU SIMD backend、toolchain 封装
  benchmarks/     benchmark task 模型与 CPU SIMD 任务定义
  domain/         枚举、错误类型、核心 Pydantic 模型
  generator/      生成请求/结果类型、prompt builder、OpenAI 客户端
  memory/         SQLite memory bank、embedding 检索与 compact state signature
  orchestrator/   drafting/refining episode loop 与 run-report 模型
  retrieval/      recall、epsilon-greedy policy、reward update、Q-value store
  verifier/       anti-hack、correctness、profiling、顶层 verifier

configs/
  default.toml
  cpu_simd.toml

prompts/
  drafting_system.md
  refining_system.md

tests/
  单元测试、orchestrator 测试、verifier 测试、backend 测试、integration 测试
```

## 架构概览

运行时被拆成了职责清晰的几个模块：

1. `generator`
   根据任务摘要、backend 约束、检索到的上下文和 verifier 反馈生成候选代码。

2. `backend`
   负责候选代码落盘、编译、执行、测量延迟，以及抽取 backend 特定的错误信息。

3. `verifier`
   负责 anti-hack 检查、compile/runtime 处理、正确性对比、latency 聚合，并
   输出 `VerificationOutcome`。

4. `memory` 与 `retrieval`
   负责持久化历史尝试、注入 backend seed knowledge、构造紧凑 state signature、
   召回候选记忆，并更新 `Q1` / `Q2`。

5. `orchestrator`
   驱动完整的 episode：
   - drafting：先找到第一个可行解；经验记忆走 value-driven 检索，API knowledge
     走单独的 backend-aware 通道
   - refining：从可行起点继续优化，并使用 profiler/bottleneck 诊断、observable
     child variant 和补充高性能变体来组织上下文
   - reward 更新与 run-report 生成

6. `cli`
   负责把 config、backend、generator、memory、artifact 路径和 episode loop
   组装成一个可运行命令。

## 快速开始

### 依赖要求

- Python 3.12+
- `uv`
- `clang` 或 `gcc`

### 安装依赖

```bash
uv sync
```

### 运行测试

```bash
uv run pytest
```

### 代码检查

```bash
uv run ruff check .
```

## CLI 使用方式

### 本地 deterministic 验证路径

为了方便本地开发和验证，仓库提供了一个 deterministic 的 dev/test 路径：

```bash
uv run python -m evokernel.cli \
  --config configs/cpu_simd.toml \
  --task vector_add \
  --generator deterministic-test
```

这条路径是为本地验证和 CI 风格检查准备的，不用于真实模型推理。只有当你显式
请求 `deterministic-test` 时，它才会从 `tests/conftest.py` 里加载对应的
override。

### OpenAI-compatible provider 路径

```bash
export OPENAI_API_KEY=...

uv run python -m evokernel.cli \
  --config configs/cpu_simd.toml \
  --task vector_add
```

生成器配置来自 TOML：

```toml
[generator]
provider = "openai_compatible"
model = "gpt-5.4"
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1" # 可选
```

### 跨运行复用 shared memory

```bash
uv run python -m evokernel.cli \
  --config configs/cpu_simd.toml \
  --task reduce_sum \
  --generator deterministic-test \
  --reuse-memory
```

启用 `--reuse-memory` 后，CLI 会从指定 work root 下的 `shared_memory.sqlite3`
中读取已有 memory，并在 run report 中记录真正被复用的 memory ID。

无论是否启用 `--reuse-memory`，CLI 启动时都会把当前 backend 的 seed knowledge
写入 shared memory，并把这些 seed 项纳入当前运行可见的检索范围；`--reuse-memory`
只影响历史运行产生的共享经验是否参与当前检索。

## 运行产物

默认情况下，CLI 会把运行结果写到 `.evokernel/` 下；也可以通过 `--work-root`
指定输出目录。

典型目录结构如下：

```text
<work-root>/
  shared_memory.sqlite3
  artifacts/
    vector_add/
      run_report.json
      vector_add-1/
        candidate.cpp
        harness.cpp
        candidate.so
        toolchain.json
        cases/
```

`run_report.json` 里会包含：

- task 和 backend 标识
- 每次 attempt 的摘要
- best candidate 信息
- 已加载 memory 数量
- 实际被复用的 memory ID

## 配置文件

`configs/default.toml` 与 `configs/cpu_simd.toml` 当前主要定义：

- retrieval 参数：
  - `final_context_count`
  - `over_retrieval_lambda`
  - `epsilon`
  - `alpha`
- embedding 配置：
  - `provider`
  - `model`
  - `dimensions`
- generator 配置
- runtime 输出目录与 `attempt_budget`
- benchmark 任务列表

当前默认 benchmark 集合：

- `vector_add`
- `reduce_sum`
- `matmul_tiled`
- `layernorm`

## 验证覆盖

当前测试主要覆盖：

- package import 与 CLI smoke 行为
- config 解析与校验
- memory 持久化、Q-value 持久化与原子导出
- dense retrieval、hybrid drafting retrieval 与 epsilon-greedy 选择
- prompt 构建与 OpenAI-compatible HTTP 请求
- CPU SIMD 编译、执行与 latency 测量
- anti-hack 规则检查
- correctness mismatch 摘要与 profiling 聚合
- profiler diagnosis、bottleneck-conditioned refining retrieval 与 child-variant
  context
- drafting 到 refining 的状态切换
- CPU SIMD CLI 路径与 shared-memory reuse 的端到端验证

常用定点命令：

```bash
uv run pytest tests/integration/test_cpu_simd_pipeline.py -v
uv run pytest tests/orchestrator/test_episode.py -v
uv run pytest tests/verifier/test_correctness.py -v
uv run pytest tests/backend/test_cpu_simd_backend.py -v
```

## 当前限制

这是一个原型，不是完整产品。当前还有一些明确限制：

- shared memory 现在持久化在 SQLite 中；只有传 `--reuse-memory` 才会把历史共享
  memory 纳入当前检索视野，但每次运行产生的新 memory 和 `Q1/Q2` 都会写回共享库。
- 当前已经有 `M0` 风格的 typed seed knowledge 与 drafting 阶段的 hybrid retrieval，
  但仍然是轻量实现：API knowledge 主要来自仓库内置 seed bundle，而不是外部文档管线。
- refining 已经接入 profiler/bottleneck 条件化检索，但 bottleneck 诊断仍然是基于
  benchmark baseline 和规则的轻量推断，不是真实硬件 profiler 事件分析。
- `deterministic-test` 生成器路径本质上是仓库内的 dev/test 路径，不是通用生产
  generator 实现。
- 当前只实现了 CPU SIMD backend，还没有 Ascend/CUDA 风格后端。
- 默认 embedding provider 仍然是本地 hashing 方案，便于离线测试；如果要接近
  论文里的真实 dense retrieval 质量，应该在配置里切到外部 embedding 服务。

## 开发说明

- 项目使用 `uv` 做环境和命令管理。
- backend 会保留每次 attempt 的构建目录，方便检查失败产物。
- 编译器优先选择 `clang` / `clang++`，找不到时回退到 `gcc` / `g++`。
- 每次构建实际使用的编译器信息都会写入对应 attempt 的 `toolchain.json`。

## 参考文档

- 论文笔记与算法整理：
  [`EvoKernel_算法流程整理.md`](./EvoKernel_%E7%AE%97%E6%B3%95%E6%B5%81%E7%A8%8B%E6%95%B4%E7%90%86.md)
- 论文正文提取：
  [`doc/paper.txt`](./doc/paper.txt)
- 设计规格：
  [`docs/superpowers/specs/2026-03-26-evokernel-prototype-design.md`](./docs/superpowers/specs/2026-03-26-evokernel-prototype-design.md)
- 实现计划：
  [`docs/superpowers/plans/2026-03-26-evokernel-prototype.md`](./docs/superpowers/plans/2026-03-26-evokernel-prototype.md)

## 许可证

本项目使用 GNU General Public License v3.0。见 [LICENSE](./LICENSE)。
