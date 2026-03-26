# EvoKernel Prototype Design

**Date:** 2026-03-26
**Status:** Approved
**Scope:** `src/`, `configs/`, `benchmarks/`, `prompts/`, `tests/`

## Problem

We need a runnable EvoKernel framework prototype, not a paper-result reproduction. The system should preserve the paper's core structure:

- two-stage `drafting -> refining` optimization
- a shared memory bank
- value-driven retrieval with stage-specific `Q1/Q2`
- multi-gate verification
- cross-task experience reuse

The prototype must run on the current machine without an Ascend toolchain. At the same time, it must be straightforward to migrate to an Ascend environment later by replacing backend-specific plugins instead of rewriting the orchestrator.

## Goals

- Build a real end-to-end EvoKernel-style loop in `Python`, managed with `uv`
- Use a pluggable backend architecture so CPU SIMD is the default backend and Ascend can be added later
- Support real LLM providers through a generator abstraction instead of hard-coding one model API
- Run a small benchmark of `3-5` CPU SIMD operator tasks
- Preserve enough traceability to inspect retrieval decisions, rewards, and kernel evolution across iterations

## Non-Goals

- Reproducing the paper's Ascend C metrics or exact benchmark setup
- Full KernelBench compatibility in the first version
- Distributed search or multi-node execution
- Heavyweight retrieval infrastructure in v1
- Ascend-specific engineering beyond the extension points required for later migration

## Design

### Core Architecture

The prototype is organized around six stable boundaries:

1. `orchestrator`
   Owns the EvoKernel episode loop for one task: state initialization, stage transitions, budget accounting, memory updates, and final result selection.

2. `generator`
   Encapsulates real LLM providers. Inputs include task description, stage, retrieved context, backend constraints, and previous verifier feedback. Outputs include candidate kernel code and generation metadata. The interface must support OpenAI-compatible endpoints first and leave room for additional providers later.

3. `memory + retrieval`
   Stores heterogeneous memories:
   - API/backend knowledge
   - success and failure experience summaries
   - generation traces
   - refinement best practices

   Retrieval is two-step:
   - candidate recall
   - stage-specific filtering by `Q1` or `Q2`

4. `backend`
   Defines how a candidate is materialized and executed. The first implementation is `CpuSimdBackend`, responsible for project scaffolding, compilation, execution, and profiling. Future backends, including Ascend, implement the same interface.

5. `verifier`
   Produces a structured multi-gate outcome:
   - anti-hack
   - compile
   - correctness
   - latency

   The verifier is backend-aware but returns a backend-neutral schema to the orchestrator.

6. `benchmark/task spec`
   Defines operator tasks with a standard format: reference implementation, input generator, tolerance settings, prompt metadata, and performance baseline data. The v1 benchmark contains `3-5` CPU SIMD tasks but is shaped so future Ascend tasks fit the same contract.

### Episode State and Data Flow

A task runs as one episode. Memory may be shared across episodes.

State is modeled as:

`s_t = (task, dynamic_state)`

`dynamic_state` must at least contain:

- current stage: `drafting` or `refining`
- iteration count and remaining budget
- current best feasible candidate
- best-so-far latency
- feasible start set `P(x)`

The runtime loop is:

1. initialize episode state from task definition and backend config
2. enter `drafting`
3. retrieve candidate memories and select final context with `Q1` under an `epsilon-greedy` policy
4. call the generator with task, backend constraints, retrieved context, and recent feedback
5. execute verification through backend and verifier
6. assign reward, update memory values, and persist the attempt
7. if the first feasible kernel is found, move it into `P(x)` and switch to `refining`
8. during `refining`, choose a start point from `P(x)`, retrieve optimization context, generate a variant, verify it, and update `Q2`
9. continue until budget exhaustion or explicit stop conditions
10. return the best feasible candidate and a structured run report

### Drafting Stage

The drafting objective is feasibility only.

- candidate memories are recalled from the shared memory bank
- final prompt context is selected using `Q1`
- generated candidates are scored with binary reward:
  - `+1` if feasible
  - `-1` otherwise
- once feasibility is reached, the first valid kernel becomes the initial member of `P(x)`

This stage should not optimize for latency beyond recording it as metadata.

### Refining Stage

The refining objective is latency reduction while preserving feasibility.

- a start point is chosen from `P(x)` using `Q2`
- additional refinement context is retrieved from memory
- the generator produces a modified candidate relative to the chosen start point
- infeasible candidates receive `-1`
- feasible candidates receive:

`r2 = tanh(log(best_latency) - log(new_latency))`

- `r2` is normalized online before updating `Q2`
- every feasible refinement is added back into `P(x)` so the search space expands over time

This preserves the paper's notion that refinement can branch from historical feasible candidates rather than only the current best one.

### Memory Model

Every attempt is written back into memory with enough structure for later retrieval and analysis. Each memory item should include:

- task fingerprint
- backend identifier
- stage
- candidate code
- prompt/context summary
- verifier outcome
- reward
- parent candidate reference when present
- whether the item is feasible
- whether the item became a refinement start point

The implementation should support at least these memory categories:

- backend/API knowledge
- summarized failures
- summarized successes
- generation traces
- refinement hints

The first version may use lightweight feature-based or text-based recall before `Q`-based filtering. It does not need embeddings or a dedicated vector database.

### Value Update

The first version uses the paper's simple Monte-Carlo update:

`Q(s, m) <- Q(s, m) + alpha * (r - Q(s, m))`

To keep the prototype implementable, `s` is not the full raw runtime state. Instead, the system uses a compact state signature derived from stable features such as:

- backend
- operator family
- stage
- shape bucket
- error category

This gives a practical key space for `Q1/Q2` while preserving the intended stage-aware retrieval behavior.

### Backend Plugin Contract

The backend interface must be explicit enough that `CpuSimdBackend` and a future `AscendBackend` can both satisfy it without changing orchestrator logic. The contract should cover:

- task materialization into a buildable runnable artifact
- build/compile execution
- correctness execution against the task reference
- latency measurement
- backend-specific prompt constraints or API snippets
- structured error extraction

The CPU implementation should compile generated C/C++ kernels with SIMD intrinsics through the system toolchain, then invoke them from the Python harness for validation and profiling.

### Verifier Semantics

The verifier returns a structured outcome aligned with the paper:

- `ghack`
- `gcomp`
- `gcorr`
- `latency`

Feasibility is defined as all three boolean gates succeeding.

#### Anti-Hack

The first version uses static rule-based screening to reject trivial shortcuts, for example:

- calling the high-level reference implementation directly
- using disallowed high-level array/vector libraries instead of the requested kernel path
- bypassing the expected generated entrypoint

#### Compile

Compilation errors must be captured as structured categories, not only raw logs, so they can feed memory and retrieval.

#### Correctness

Each task includes:

- a reference implementation
- randomized test input generation
- selected edge cases
- configurable `atol/rtol`

Verifier output should include concise mismatch summaries rather than unbounded raw traces.

#### Latency

Latency is measured only for feasible candidates. The profiler should support warmup and repeated runs, and report a stable aggregate such as median or trimmed mean. The interface should leave space for richer profiler metadata later.

## Initial Benchmark

The first benchmark includes `3-5` CPU SIMD tasks chosen to provide both feasibility and optimization pressure. The target set is:

- `vector_add`
- `reduce_sum`
- `matmul_tiled`
- `layernorm` or `softmax`

The benchmark format should make task definitions easy to extend and should preserve enough metadata for future mapping to Ascend-oriented task suites.

## Configuration and Runtime Expectations

The project uses `uv` for environment and dependency management.

Configuration should allow independent selection of:

- backend plugin
- generator provider and model
- iteration budget
- retrieval parameters such as `N`, `K`, `lambda`, `epsilon`, and `alpha`
- benchmark subset
- logging and artifact directories

The runtime should emit structured artifacts for each run, including selected contexts, rewards, verifier outcomes, and best-result summaries.

## Testing Strategy

Testing must cover both algorithm logic and execution plumbing.

1. unit tests
   - state transition logic
   - reward calculation
   - `Q1/Q2` update behavior
   - retrieval selection
   - memory item serialization

2. backend contract tests
   - build success and failure paths
   - correctness harness wiring
   - latency measurement interface

3. integration tests
   - one small task completing `drafting -> refining`
   - cross-task memory reuse smoke test
   - deterministic mock generator path for CI where needed

The presence of a mock or deterministic generator path for tests is acceptable, but production runtime remains centered on real provider-backed generation.

## Success Criteria

The prototype is considered successful when:

- `uv` can create and run the project cleanly
- at least one benchmark task completes the full `drafting -> refining` lifecycle
- memory is reused across multiple tasks
- run artifacts make retrieval, rewards, and candidate evolution inspectable
- adding a future `AscendBackend` requires only a new backend implementation plus task definitions, not orchestrator rewrites

## Deferred Work

These are explicitly deferred beyond the initial prototype:

- exact Ascend C toolchain integration
- full KernelBench compatibility
- profiler-driven bottleneck retrieval
- embedding-based dense retrieval infrastructure
- large-scale benchmark automation
- distributed or parallel candidate search
