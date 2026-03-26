# EvoKernel 算法流程整理

本文整理对象为论文 **Toward Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis**。下面仅围绕论文核心算法 `EvoKernel` 的执行流程展开，不重复实验结果。

## 1. 算法目标

EvoKernel 的目标是：在 **缺少领域训练数据**、**不能依赖额外微调** 的条件下，自动完成面向特定硬件后端的 kernel 合成，并把整个过程拆成两个连续阶段：

1. **Cold-Start Drafting**：先找到一个“可行”的初始 kernel。
2. **Continual Refining**：在已有可行 kernel 的基础上，继续做性能优化，主要目标是降低延迟。

它的核心思想不是直接训练生成模型参数，而是维护一个会持续增长的 **记忆库（memory bank）**，再通过 **价值驱动检索（value-driven retrieval）** 从记忆中挑选最有帮助的上下文给 LLM 生成器使用。

## 2. 核心对象

### 2.1 任务定义

每个 kernel 合成任务 `x` 包含：

- PyTorch 参考算子
- 输入 shape
- 算子超参数

生成器在给定任务 `x` 和上下文 `c_t` 后，生成候选 kernel 代码 `y_t`。

### 2.2 状态

论文把过程建模为一个 memory-based MDP，状态写成：

`s_t = (x, ξ_t)`

其中：

- `x` 是静态任务本身
- `ξ_t` 是动态状态，例如当前是否已有可行解、当前最优延迟等

### 2.3 动作

动作 `a_t` 就是本轮生成出来的 kernel 代码。

### 2.4 记忆库 M

记忆库是一个会不断扩展的异构知识库，包含：

- 后端 API 模板或接口知识
- 历史成功/失败经验总结
- 生成轨迹，包括 draft 和 refined 版本
- kernel 优化 best practices

每轮交互后，都会把 `(state, action, reward)` 及验证结果写回 memory。

### 2.5 检索价值函数 Q

对于候选记忆项 `m`，定义阶段相关的价值函数：

- `Q1(s, m)`：在 Drafting 阶段，这条记忆是否有助于生成功能正确的 kernel
- `Q2(s, m)`：在 Refining 阶段，这条记忆是否有助于降低延迟

EvoKernel 先做一次稠密检索，得到 top-K 候选池 `C(x)`，其中：

`K = λN`

再用 Q 值从候选池里筛出最终的 `N` 条上下文。

## 3. 总体算法流程

EvoKernel 的主循环可以概括为下面 8 步：

1. 读取任务 `x`，初始化动态状态 `ξ_0`、记忆库 `M_0`。
2. 从 memory 中检索与当前任务相关的候选上下文。
3. 在 **Drafting 阶段**，基于 `Q1` 选择上下文，驱动 LLM 生成候选 kernel。
4. 用多重验证器检查候选 kernel 是否通过反作弊、编译、正确性测试。
5. 若通过，则获得第一个可行 kernel，进入 **Refining 阶段**；若失败，则根据奖励更新 Q 值并继续 drafting。
6. 在 **Refining 阶段**，从已有可行 kernel 集合中选一个优化起点，再检索优化轨迹、best practices、子节点信息等上下文。
7. 生成改进后的 kernel，验证其是否仍可行，并根据延迟变化给奖励，更新 `Q2`。
8. 把新经验写回 memory，直到预算耗尽，输出当前最佳 kernel。

## 4. 详细流程拆解

### 4.1 初始化

给定任务 `x` 后，系统初始化：

- 记忆库 `M_0`
- 轮次预算 `T`
- Drafting 阶段的 Q 值 `Q1`
- Refining 阶段的 Q 值 `Q2`
- 当前最优延迟 `b_t`
- 优化起点集合 `P(x)`，初始为空

其中 `M_0` 可以带有后端 API 模板等种子知识，不要求已有大量专家示例。

### 4.2 阶段一：Cold-Start Drafting

这一阶段的目标不是追求最优性能，而是尽快得到 **第一个可行 kernel**。

#### 步骤 1：候选记忆检索

针对当前任务 `x`，先从 memory 中做稠密检索，得到候选池 `C(x)`。

#### 步骤 2：基于 Q1 的上下文选择

在候选池基础上，用基于 `Q1` 的 `ε-greedy` 策略选出当前轮真正注入到 prompt 里的上下文 `c_t`。

论文说明：

- 经验类记忆、代码轨迹主要由价值驱动选择
- API 知识在升级版系统中采用后端感知的混合检索方式

#### 步骤 3：生成 draft kernel

生成器按下面的复合策略工作：

`π(y_t | s_t, M_t) = G_θ(a_t | s_t, c_t) · μ(c_t | s_t, M_t)`

即：

- 检索策略 `μ` 决定拿哪些记忆作为上下文
- 生成器 `G_θ` 在这个上下文条件下输出 kernel 代码

#### 步骤 4：可行性验证

把候选代码送入验证器 `V(x, y_t)`，得到：

`o_t = (g_hack, g_comp, g_corr, ℓ_lat)`

含义分别是：

- `g_hack`：是否通过反作弊检查
- `g_comp`：是否成功编译
- `g_corr`：是否通过正确性验证
- `ℓ_lat`：若可运行，则测得延迟

可行性定义为：

`g_feas(o_t) = g_hack ∧ g_comp ∧ g_corr`

#### 步骤 5：给 Drafting 阶段奖励

Drafting 使用二值奖励：

`r_{1,t} = +1`，如果当前 kernel 可行  
`r_{1,t} = -1`，否则

它的作用很直接：只奖励“找到可行解”这件事，不关心此时性能好坏。

#### 步骤 6：更新 Q1 并写回 memory

对本轮被检索到的上下文项 `m ∈ c_t`，用统一 Monte-Carlo 规则更新：

`Q(s, m) ← Q(s, m) + α (r - Q(s, m))`

这里在 Drafting 阶段取 `r = r_{1,t}`。

同时把：

- 本轮生成代码
- 验证结果
- 成功/失败经验

写回记忆库 `M`。

#### 步骤 7：是否结束 Drafting

- 如果已找到可行 kernel，则把它加入 `P(x)`，转入 Refining
- 如果还没找到，但预算未耗尽，则继续下一轮 Drafting
- 如果预算耗尽，则任务结束

### 4.3 阶段二：Continual Refining

当系统已经有至少一个可行 kernel 后，目标就从“能不能做出来”切换为“能不能做得更快”。

#### 步骤 1：维护优化起点集合 P(x)

`P(x)` 初始包含阶段一得到的首个可行 kernel。之后每发现新的可行变体，也加入 `P(x)`，作为未来 refinement 的候选起点。

#### 步骤 2：选择本轮优化起点

在当前状态 `s_t` 下，从 memory 中取出可用的优化起点，再根据 `Q2` 选择一个起点 `p_t`。

这一步的意义是：不是每次都从当前最好 kernel 出发，而是允许从历史上“值得再挖”的版本继续优化。

#### 步骤 3：检索 refinement 上下文

在确定起点后，继续检索辅助上下文 `c_t`，内容包括：

- 历史优化轨迹
- kernel 优化 best practices
- 当前起点的可观察子节点信息
- 升级版系统里还会加入 profiler 诊断出的瓶颈匹配案例

#### 步骤 4：生成 refined kernel

LLM 以“优化起点 + refinement 上下文”为条件，生成新的候选 kernel。

#### 步骤 5：验证 refined kernel

再次经过与 Drafting 相同的多重验证器：

- 先检查是否作弊
- 再检查能否编译
- 再检查输出是否正确
- 若可行，则测量延迟

#### 步骤 6：计算性能优化奖励

Refining 阶段奖励和当前最优延迟 `b_t` 有关。

若当前 kernel 不可行，则：

`r_{2,t} = -1`

若当前 kernel 可行，则：

`r_{2,t} = tanh(log b_t - log ℓ_lat(o_t))`

这个奖励的含义是：

- 若新 kernel 比当前最优更快，则奖励为正
- 若更慢，则奖励为负
- 通过 `tanh` 和对数差，把极端数值压到稳定范围内

论文随后又对该奖励做在线标准化：

`r̂_{2,t} = (r_{2,t} - μ_2) / σ_2`

这里 `(μ_2, σ_2)` 是运行中的均值和标准差估计，类似 PopArt 风格归一化。

#### 步骤 7：更新 Q2

对下列对象做价值更新：

- 本轮选中的优化起点 `p_t`
- 本轮 refinement 上下文中的检索项 `z ∈ c_t`

更新规则依然是：

`Q(s, m) ← Q(s, m) + α (r - Q(s, m))`

这里只是把 `r` 换成标准化后的 `r̂_{2,t}`。

#### 步骤 8：扩展 memory 和起点集合

若 refined kernel 仍然可行，则：

- 把代码和验证结果写回 memory
- 把这个新可行版本加入 `P(x)`
- 如果其延迟优于当前最优，则同步更新 `b_t`

之后继续下一轮 refinement，直到预算用完。

## 5. 多重验证器的内部逻辑

验证器 `V` 是 EvoKernel 的环境接口，负责把代码执行结果转成 RL 可用反馈。

### 5.1 反作弊门

先做规则过滤，拒绝明显作弊行为，例如：

- 调用高层 `torch` API 偷渡答案
- 常量折叠等捷径

然后再做模型辅助检查，识别更隐蔽的 harness 操作或规避方式。

### 5.2 编译门

检查代码是否能在目标后端工具链下成功编译。

### 5.3 正确性门

将生成 kernel 的输出与 PyTorch 参考实现对比，判断：

`||out_y(x) - ref(x)|| ≤ τ`

只有通过正确性门，才算功能正确。

### 5.4 性能测量

只有在可行时才测量延迟 `ℓ_lat`，作为 Refining 阶段的奖励依据。

## 6. 统一伪代码

```text
Input:
  task x
  initial memory M0
  drafting value table Q1
  refining value table Q2
  total budget T

Initialize:
  t = 0
  M <- M0
  P(x) <- empty
  best_latency <- +inf
  feasible_found <- false

While t < T:
  if feasible_found == false:
    1. dense retrieve candidate pool C(x) from M
    2. select drafting context c_t from C(x) by epsilon-greedy over Q1
    3. generate kernel y_t using LLM conditioned on (x, c_t)
    4. run verifier o_t = V(x, y_t)
    5. assign feasibility reward:
         r = +1 if feasible else -1
    6. update Q1 for all retrieved items in c_t
    7. write (y_t, o_t, r) into M
    8. if feasible:
         feasible_found <- true
         add y_t into P(x)
         best_latency <- measured latency

  else:
    1. choose start point p_t from P(x) using Q2
    2. retrieve refinement context c_t from M
    3. generate refined kernel y_t conditioned on (x, p_t, c_t)
    4. run verifier o_t = V(x, y_t)
    5. compute refinement reward:
         r = -1 if infeasible
         r = tanh(log(best_latency) - log(current_latency)) if feasible
    6. normalize reward to r_hat
    7. update Q2 for p_t and retrieved items in c_t
    8. write (y_t, o_t, r_hat) into M
    9. if feasible:
         add y_t into P(x)
         if current_latency < best_latency:
           best_latency <- current_latency

  t <- t + 1

Output:
  best feasible kernel found within budget
```

## 7. 这套算法真正解决的问题

EvoKernel 的关键不在“让 LLM 一次写对”，而在于构造一个能 **跨轮次积累经验、跨任务迁移经验、并根据目标动态调整检索重点** 的闭环：

- 在 Drafting 阶段，系统优先学习“哪些记忆有助于把代码写对”
- 在 Refining 阶段，系统优先学习“哪些起点和哪些经验有助于继续降延迟”
- 每次验证反馈都会反过来更新检索价值，因此 memory 不是静态知识库，而是一个不断演化的策略载体

## 8. 一句话总结

EvoKernel 可以看成一个“**基于记忆检索的两阶段 kernel 合成代理**”：先用价值驱动检索帮助 LLM 找到可行 draft，再利用同一套 memory 和新的奖励目标，持续把可行 kernel 往更低延迟方向推进。
