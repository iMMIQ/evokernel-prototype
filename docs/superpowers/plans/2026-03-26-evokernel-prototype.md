# EvoKernel Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable EvoKernel-style prototype in Python with `uv`, a pluggable backend contract, a real LLM generator abstraction, and a CPU SIMD backend that supports drafting, refining, memory reuse, correctness checks, and latency profiling.

**Architecture:** The system is split into small, stable units: domain models and config, memory and retrieval, generator providers, backend and verifier contracts, the episode orchestrator, and benchmark/CLI entrypoints. The first backend is CPU SIMD, but all orchestration logic stays backend-neutral so a future `AscendBackend` only needs to implement the same plugin interfaces and task definitions.

**Tech Stack:** Python 3.12, `uv`, `pytest`, `pydantic`, `httpx`, `numpy`, system `clang` or `gcc`, CPU SIMD C/C++ kernels via `ctypes`

---

## Planned File Structure

### Core Package

- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/evokernel/__init__.py`
- Create: `src/evokernel/config.py`
- Create: `src/evokernel/domain/models.py`
- Create: `src/evokernel/domain/enums.py`
- Create: `src/evokernel/domain/errors.py`

### Memory and Retrieval

- Create: `src/evokernel/memory/store.py`
- Create: `src/evokernel/memory/state_signature.py`
- Create: `src/evokernel/retrieval/recall.py`
- Create: `src/evokernel/retrieval/policy.py`
- Create: `src/evokernel/retrieval/reward.py`
- Create: `src/evokernel/retrieval/q_store.py`

### Generator Providers

- Create: `src/evokernel/generator/base.py`
- Create: `src/evokernel/generator/openai_compatible.py`
- Create: `src/evokernel/generator/prompt_builder.py`
- Create: `prompts/drafting_system.md`
- Create: `prompts/refining_system.md`

### Backend and Verification

- Create: `src/evokernel/backend/base.py`
- Create: `src/evokernel/backend/cpu_simd.py`
- Create: `src/evokernel/backend/toolchain.py`
- Create: `src/evokernel/verifier/core.py`
- Create: `src/evokernel/verifier/anti_hack.py`
- Create: `src/evokernel/verifier/correctness.py`
- Create: `src/evokernel/verifier/profiling.py`

### Benchmark and Runtime

- Create: `src/evokernel/benchmarks/task_registry.py`
- Create: `src/evokernel/benchmarks/cpu_simd_tasks.py`
- Create: `src/evokernel/benchmarks/models.py`
- Create: `src/evokernel/orchestrator/episode.py`
- Create: `src/evokernel/orchestrator/run_report.py`
- Create: `src/evokernel/cli.py`
- Create: `configs/default.toml`
- Create: `configs/cpu_simd.toml`

### Tests

- Create: `tests/conftest.py`
- Create: `tests/test_smoke_import.py`
- Create: `tests/domain/test_models.py`
- Create: `tests/memory/test_store.py`
- Create: `tests/retrieval/test_policy.py`
- Create: `tests/generator/test_prompt_builder.py`
- Create: `tests/generator/test_openai_compatible.py`
- Create: `tests/backend/test_cpu_simd_backend.py`
- Create: `tests/verifier/test_anti_hack.py`
- Create: `tests/verifier/test_correctness.py`
- Create: `tests/verifier/test_profiling.py`
- Create: `tests/orchestrator/test_episode.py`
- Create: `tests/integration/test_cpu_simd_pipeline.py`

## Ground Rules For Implementation

- Use TDD for every new module: write one failing test, run it, implement the minimum code, rerun, then commit.
- Keep files focused. Do not combine generator, backend, verifier, and orchestrator logic in one file.
- Prefer pure-Python coordination and use `ctypes` for loading compiled CPU SIMD kernels. Do not introduce `pybind11` or a custom C extension build in v1.
- Make budget exhaustion the only required stop condition in v1. Early stopping is optional and must not be required to get a complete run.
- Define retrieval `lambda` explicitly as the over-retrieval multiplier used to compute `K = lambda * N`.
- Lock the initial benchmark set to `vector_add`, `reduce_sum`, `matmul_tiled`, and `layernorm`.

### Task 1: Bootstrap The `uv` Project And Package Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/evokernel/__init__.py`
- Create: `tests/test_smoke_import.py`

- [ ] **Step 1: Write the failing import smoke test**

```python
from evokernel import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_smoke_import.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evokernel'`

- [ ] **Step 3: Write minimal package bootstrap**

```python
__version__ = "0.1.0"
```

Add a `pyproject.toml` with:
- package name `evokernel`
- Python `>=3.12`
- runtime deps: `pydantic`, `httpx`, `numpy`, `typing-extensions`
- dev deps: `pytest`, `pytest-cov`, `pytest-httpx`, `ruff`
- console script: `evokernel = evokernel.cli:main`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_smoke_import.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version src/evokernel/__init__.py tests/test_smoke_import.py
git commit -m "chore: bootstrap evokernel uv project"
```

### Task 2: Add Domain Models And Configuration Parsing

**Files:**
- Create: `src/evokernel/config.py`
- Create: `src/evokernel/domain/models.py`
- Create: `src/evokernel/domain/enums.py`
- Create: `src/evokernel/domain/errors.py`
- Create: `configs/default.toml`
- Create: `configs/cpu_simd.toml`
- Test: `tests/domain/test_models.py`

- [ ] **Step 1: Write failing tests for runtime config and episode models**

```python
from evokernel.config import load_runtime_config
from evokernel.domain.models import EpisodeState, VerificationOutcome


def test_load_runtime_config_reads_lambda_multiplier(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[retrieval]\nfinal_context_count=4\nover_retrieval_lambda=3\n",
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)
    assert config.retrieval.over_retrieval_lambda == 3


def test_episode_state_starts_in_drafting():
    state = EpisodeState.initial(task_id="vector_add", budget=30)
    assert state.stage.value == "drafting"
    assert state.remaining_budget == 30
    assert state.start_points == []


def test_verification_outcome_requires_all_gates_for_feasibility():
    outcome = VerificationOutcome(
        anti_hack_passed=True,
        compile_passed=True,
        correctness_passed=False,
        latency_ms=None,
        error_category="wrong_answer",
        feedback_summary="mismatch at output[3]",
    )
    assert outcome.is_feasible is False


def test_load_runtime_config_reads_generator_provider_fields(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[generator]\nprovider=\"openai_compatible\"\nmodel=\"gpt-5.4\"\nbase_url=\"https://api.example.test/v1\"\napi_key_env=\"OPENAI_API_KEY\"\n",
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)
    assert config.generator.provider == "openai_compatible"
    assert config.generator.model == "gpt-5.4"


def test_load_runtime_config_reads_backend_benchmark_and_artifact_paths(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[runtime]\nbackend=\"cpu_simd\"\nartifact_dir=\"artifacts\"\nlog_dir=\"logs\"\n[benchmark]\ntasks=[\"vector_add\", \"reduce_sum\"]\n",
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)
    assert config.runtime.backend == "cpu_simd"
    assert config.benchmark.tasks == ["vector_add", "reduce_sum"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/domain/test_models.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement the minimal config and domain model layer**

```python
class Stage(str, Enum):
    DRAFTING = "drafting"
    REFINING = "refining"


class RetrievalConfig(BaseModel):
    final_context_count: int = 4
    over_retrieval_lambda: int = 3
    epsilon: float = 0.1
    alpha: float = 0.2


class GeneratorConfig(BaseModel):
    provider: str = "openai_compatible"
    model: str = "gpt-5.4"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


class RuntimeConfig(BaseModel):
    backend: str = "cpu_simd"
    artifact_dir: str = "artifacts"
    log_dir: str = "logs"
```

Implement:
- `EpisodeState.initial(...)`
- `VerificationOutcome.is_feasible`
- `load_runtime_config(path)` using standard library `tomllib`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/domain/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evokernel/config.py src/evokernel/domain configs/default.toml configs/cpu_simd.toml tests/domain/test_models.py
git commit -m "feat: add runtime config and domain models"
```

### Task 3: Implement Memory Store, State Signatures, And Reward Updates

**Files:**
- Create: `src/evokernel/memory/store.py`
- Create: `src/evokernel/memory/state_signature.py`
- Create: `src/evokernel/retrieval/reward.py`
- Create: `src/evokernel/retrieval/q_store.py`
- Test: `tests/memory/test_store.py`

- [ ] **Step 1: Write failing tests for state signatures, memory writes, and Q updates**

```python
from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem, VerificationOutcome
from evokernel.memory.state_signature import build_state_signature
from evokernel.memory.store import InMemoryStore
from evokernel.retrieval.q_store import QValueStore
from evokernel.retrieval.reward import update_q_value


def test_build_state_signature_uses_backend_task_stage_and_error():
    signature = build_state_signature(
        backend_id="cpu_simd",
        operator_family="reduction",
        stage=Stage.DRAFTING,
        shape_bucket="1d_small",
        error_category="compile_error",
    )
    assert signature == "cpu_simd|reduction|drafting|1d_small|compile_error"


def test_memory_store_persists_attempts_and_start_points():
    store = InMemoryStore()
    item = MemoryItem(
        task_id="reduce_sum",
        backend_id="cpu_simd",
        operator_family="reduction",
        stage=Stage.REFINING,
        code="void kernel();",
        summary="first feasible refinement",
        reward=0.5,
        is_feasible=True,
        became_start_point=True,
        verifier_outcome=VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=True,
            latency_ms=1.2,
            error_category=None,
            feedback_summary="ok",
        ),
    )
    store.add(item)
    assert store.list_start_points("reduce_sum")[0].summary == "first feasible refinement"


def test_update_q_value_uses_monte_carlo_rule():
    assert update_q_value(current=0.25, reward=1.0, alpha=0.2) == 0.4


def test_q_value_store_tracks_q1_and_q2_independently():
    store = QValueStore()
    key = "cpu_simd|elementwise|drafting|1d_small|compile_error"
    store.update(stage=Stage.DRAFTING, state_signature=key, memory_id="m1", reward=1.0, alpha=0.5)
    store.update(stage=Stage.REFINING, state_signature=key, memory_id="m1", reward=-1.0, alpha=0.5)
    assert store.get(stage=Stage.DRAFTING, state_signature=key, memory_id="m1") == 0.5
    assert store.get(stage=Stage.REFINING, state_signature=key, memory_id="m1") == -0.5


def test_memory_item_serialization_round_trip():
    item = MemoryItem(
        task_id="vector_add",
        backend_id="cpu_simd",
        operator_family="elementwise",
        stage=Stage.DRAFTING,
        code="void evokernel_entry() {}",
        summary="compile fix",
        context_summary="api: include immintrin",
        memory_kind="failure_summary",
        reward=-1.0,
        is_feasible=False,
        became_start_point=False,
        verifier_outcome=VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=False,
            correctness_passed=False,
            latency_ms=None,
            error_category="compile_error",
            feedback_summary="missing include",
        ),
    )
    payload = item.model_dump()
    restored = MemoryItem.model_validate(payload)
    assert restored.context_summary == "api: include immintrin"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/memory/test_store.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement the minimal memory model**

```python
def update_q_value(current: float, reward: float, alpha: float) -> float:
    return current + alpha * (reward - current)
```

Implement:
- `MemoryItem` model if not already present
- `InMemoryStore.add`, `recall`, `list_start_points`
- `InMemoryStore.save_jsonl(path)` and `InMemoryStore.load_jsonl(path)` for persistence and reuse across runs
- deterministic `build_state_signature`
- explicit `MemoryItem` fields for `parent_attempt_id`, `context_summary`, and `memory_kind`
- memory kinds covering `backend_knowledge`, `failure_summary`, `success_summary`, `generation_trace`, and `refinement_hint`
- `QValueStore` keyed by `(stage, state_signature, memory_id)` with explicit independent `Q1/Q2` storage

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/memory/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evokernel/memory src/evokernel/retrieval/reward.py tests/memory/test_store.py
git commit -m "feat: add memory store and q-value updates"
```

### Task 4: Implement Candidate Recall And Stage-Specific Retrieval Policy

**Files:**
- Create: `src/evokernel/retrieval/recall.py`
- Create: `src/evokernel/retrieval/policy.py`
- Test: `tests/retrieval/test_policy.py`

- [ ] **Step 1: Write failing tests for candidate recall and `epsilon-greedy` selection**

```python
from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem
from evokernel.retrieval.policy import select_context_items
from evokernel.retrieval.recall import recall_candidates
from evokernel.retrieval.q_store import QValueStore


def test_recall_candidates_limits_pool_by_lambda_times_n(memory_items):
    recalled = recall_candidates(
        items=memory_items,
        operator_family="elementwise",
        final_context_count=2,
        over_retrieval_lambda=3,
    )
    assert len(recalled) == 6


def test_select_context_items_prefers_high_q_items_when_epsilon_zero(memory_items):
    q_store = QValueStore()
    q_store.set(stage=Stage.DRAFTING, state_signature="sig", memory_id="best", value=0.9)
    q_store.set(stage=Stage.DRAFTING, state_signature="sig", memory_id="second", value=0.7)
    selected = select_context_items(
        candidates=memory_items,
        stage=Stage.DRAFTING,
        state_signature="sig",
        q_store=q_store,
        final_context_count=2,
        epsilon=0.0,
    )
    assert [item.summary for item in selected] == ["best", "second"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_policy.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement minimal recall and selection**

```python
def recall_candidates(...):
    limit = final_context_count * over_retrieval_lambda
    return ranked_items[:limit]


def select_context_items(...):
    if epsilon == 0:
        return sorted(
            candidates,
            key=lambda item: q_store.get(stage=stage, state_signature=state_signature, memory_id=item.memory_id),
            reverse=True,
        )[:final_context_count]
```

Implement:
- lightweight ranking by backend/operator/stage relevance
- deterministic branch for `epsilon=0`
- injectable randomness for later tests
- selection keyed through `QValueStore`, never by ad hoc fields on `MemoryItem`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_policy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evokernel/retrieval tests/retrieval/test_policy.py
git commit -m "feat: add memory recall and retrieval policy"
```

### Task 5: Add Prompt Builder And OpenAI-Compatible Generator Provider

**Files:**
- Create: `src/evokernel/generator/base.py`
- Create: `src/evokernel/generator/openai_compatible.py`
- Create: `src/evokernel/generator/prompt_builder.py`
- Create: `prompts/drafting_system.md`
- Create: `prompts/refining_system.md`
- Test: `tests/generator/test_prompt_builder.py`
- Test: `tests/generator/test_openai_compatible.py`

- [ ] **Step 1: Write failing tests for prompt assembly and provider payload shape**

```python
from evokernel.generator.prompt_builder import build_generation_prompt
from evokernel.generator.openai_compatible import OpenAICompatibleGenerator


def test_build_generation_prompt_includes_stage_constraints_and_feedback():
    prompt = build_generation_prompt(
        stage="drafting",
        task_summary="vector add on float32 arrays",
        backend_constraints=["emit a C entrypoint", "use SIMD intrinsics"],
        retrieved_context=["failure: missing include", "api: use __m256"],
        feedback_summary="previous compile error: unknown intrinsic",
    )
    assert "vector add on float32 arrays" in prompt
    assert "previous compile error" in prompt
    assert "use SIMD intrinsics" in prompt


def test_openai_compatible_generator_builds_responses_payload():
    generator = OpenAICompatibleGenerator(
        model="gpt-5.4",
        base_url="https://example.invalid/v1",
        api_key="test",
    )
    payload = generator.build_payload(system_prompt="sys", user_prompt="usr")
    assert payload["model"] == "gpt-5.4"
    assert payload["input"][0]["role"] == "system"


def test_openai_compatible_generator_generate_uses_http_client(httpx_mock):
    httpx_mock.add_response(
        json={"output": [{"content": [{"type": "output_text", "text": "void evokernel_entry() {}"}]}]}
    )
    generator = OpenAICompatibleGenerator(
        model="gpt-5.4",
        base_url="https://example.invalid/v1",
        api_key="test",
    )
    result = generator.generate_from_prompts(system_prompt="sys", user_prompt="usr")
    assert "evokernel_entry" in result.code
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/generator/test_prompt_builder.py tests/generator/test_openai_compatible.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement the minimal generator layer**

```python
class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
```

Implement:
- `GenerationRequest` and `GenerationResult`
- prompt assembly from task, context, and verifier feedback
- `OpenAICompatibleGenerator.build_payload(...)`
- real provider-backed `generate()` path using `httpx`
- config-driven provider selection fields for provider, model, base URL, and API key env lookup
- keep HTTP execution thin and separately testable

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generator/test_prompt_builder.py tests/generator/test_openai_compatible.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evokernel/generator prompts tests/generator
git commit -m "feat: add prompt builder and openai-compatible generator"
```

### Task 6: Implement The CPU SIMD Backend Toolchain Contract

**Files:**
- Create: `src/evokernel/backend/base.py`
- Create: `src/evokernel/backend/toolchain.py`
- Create: `src/evokernel/backend/cpu_simd.py`
- Create: `src/evokernel/benchmarks/models.py`
- Create: `src/evokernel/benchmarks/task_registry.py`
- Create: `src/evokernel/benchmarks/cpu_simd_tasks.py`
- Test: `tests/backend/test_cpu_simd_backend.py`

- [ ] **Step 1: Write failing tests for backend constraints and build artifact generation**

```python
from evokernel.backend.cpu_simd import CpuSimdBackend
from evokernel.benchmarks.models import BenchmarkTask
from evokernel.benchmarks.cpu_simd_tasks import build_vector_add_task


def test_cpu_backend_exposes_prompt_constraints():
    backend = CpuSimdBackend()
    constraints = backend.prompt_constraints()
    assert "Generate C or C++ kernel code" in constraints[0]


def test_cpu_backend_materializes_build_files(tmp_path):
    backend = CpuSimdBackend(work_root=tmp_path)
    task = build_vector_add_task()
    artifact = backend.materialize_candidate(
        task=task,
        candidate_code="void evokernel_entry(...) {}",
        attempt_id="attempt-1",
    )
    assert artifact.source_path.exists()
    assert artifact.harness_path.exists()


def test_vector_add_task_exposes_reference_inputs_tolerances_and_prompt_metadata():
    task = build_vector_add_task()
    assert isinstance(task, BenchmarkTask)
    assert callable(task.reference_impl)
    assert task.tolerances.atol > 0
    assert "simd" in task.prompt_metadata["keywords"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backend/test_cpu_simd_backend.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement minimal backend and task registry**

```python
class Backend(Protocol):
    def prompt_constraints(self) -> list[str]:
        ...

    def materialize_candidate(self, task, candidate_code, attempt_id):
        ...

    def compile(self, artifact):
        ...

    def load_callable(self, artifact):
        ...

    def run_reference_case(self, artifact, case):
        ...

    def measure_latency(self, artifact, case, warmup_runs, timed_runs):
        ...

    def extract_structured_error(self, stderr: str):
        ...
```

Implement:
- `CpuSimdBackend`
- `BenchmarkTask` schema covering reference implementation, randomized inputs, edge-case inputs, tolerances, prompt metadata, operator family, and baseline data
- benchmark task builders for `vector_add`, `reduce_sum`, `matmul_tiled`, `layernorm`
- simple harness file emission into a per-attempt working directory
- toolchain wrapper that prefers `clang`, then `gcc`
- explicit execution hooks used later by verifier code so the orchestrator never shells into artifacts directly

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backend/test_cpu_simd_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evokernel/backend src/evokernel/benchmarks tests/backend/test_cpu_simd_backend.py
git commit -m "feat: add cpu simd backend contract and task registry"
```

### Task 7: Implement Anti-Hack, Correctness, And Profiling Verifiers

**Files:**
- Create: `src/evokernel/verifier/core.py`
- Create: `src/evokernel/verifier/anti_hack.py`
- Create: `src/evokernel/verifier/correctness.py`
- Create: `src/evokernel/verifier/profiling.py`
- Test: `tests/verifier/test_anti_hack.py`
- Test: `tests/verifier/test_correctness.py`
- Test: `tests/verifier/test_profiling.py`

- [ ] **Step 1: Write failing tests for each verification gate**

```python
from evokernel.verifier.anti_hack import check_for_disallowed_patterns
from evokernel.verifier.correctness import compare_outputs
from evokernel.verifier.profiling import aggregate_latency_measurements


def test_anti_hack_rejects_numpy_shortcuts():
    result = check_for_disallowed_patterns("import numpy as np\nnp.add(a, b)")
    assert result.passed is False
    assert result.error_category == "anti_hack"


def test_compare_outputs_uses_atol_rtol():
    passed, summary = compare_outputs(
        actual=[1.0, 2.001],
        expected=[1.0, 2.0],
        atol=1e-2,
        rtol=1e-2,
    )
    assert passed is True


def test_aggregate_latency_measurements_returns_median():
    assert aggregate_latency_measurements([3.0, 1.0, 2.0]) == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/verifier/test_anti_hack.py tests/verifier/test_correctness.py tests/verifier/test_profiling.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement the verifier components**

```python
def aggregate_latency_measurements(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
```

Implement:
- rule-based anti-hack pattern checks
- correctness comparator with concise mismatch summaries
- profiling aggregation helper
- a top-level verifier coordinator that combines gate results into `VerificationOutcome`
- verifier integration through backend hooks for compile errors, runtime execution, latency measurement, and structured error extraction

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/verifier/test_anti_hack.py tests/verifier/test_correctness.py tests/verifier/test_profiling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evokernel/verifier tests/verifier
git commit -m "feat: add verifier gates for anti-hack correctness and profiling"
```

### Task 8: Implement The Drafting And Refining Episode Orchestrator

**Files:**
- Create: `src/evokernel/orchestrator/episode.py`
- Create: `src/evokernel/orchestrator/run_report.py`
- Test: `tests/orchestrator/test_episode.py`

- [ ] **Step 1: Write failing tests for stage transition and memory reuse**

```python
from evokernel.orchestrator.episode import run_episode


def test_run_episode_switches_to_refining_after_first_feasible_attempt(fake_runtime):
    report = run_episode(fake_runtime, task_id="vector_add")
    assert report.attempts[0].stage == "drafting"
    assert any(attempt.stage == "refining" for attempt in report.attempts)


def test_run_episode_updates_best_latency_from_feasible_refinements(fake_runtime):
    report = run_episode(fake_runtime, task_id="vector_add")
    assert report.best_candidate is not None
    assert report.best_candidate.verifier_outcome.latency_ms == 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/orchestrator/test_episode.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement the minimal orchestrator**

```python
while state.remaining_budget > 0:
    if state.stage is Stage.DRAFTING:
        ...
    else:
        ...
    state = state.after_attempt(result)
```

Implement:
- drafting loop with `Q1`
- refining loop with `Q2`
- explicit start-point selection from `P(x)`
- reward calculation including normalized `r2`
- run report object that captures attempts, selected context, and best candidate

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/orchestrator/test_episode.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evokernel/orchestrator tests/orchestrator/test_episode.py
git commit -m "feat: add evokernel drafting and refining orchestrator"
```

### Task 9: Add CLI Entry Point And End-To-End CPU SIMD Integration Test

**Files:**
- Create: `src/evokernel/cli.py`
- Create: `tests/integration/test_cpu_simd_pipeline.py`
- Modify: `src/evokernel/backend/cpu_simd.py`
- Modify: `src/evokernel/verifier/core.py`
- Modify: `src/evokernel/orchestrator/episode.py`

- [ ] **Step 1: Write a failing end-to-end integration test with a deterministic generator**

```python
from evokernel.cli import main


def test_cpu_simd_pipeline_runs_vector_add(tmp_path, monkeypatch):
    exit_code = main(
        [
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "vector_add",
            "--generator",
            "deterministic-test",
            "--work-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 0


def test_cpu_simd_pipeline_reuses_memory_across_two_tasks(tmp_path, monkeypatch):
    first_exit = main(
        [
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "vector_add",
            "--generator",
            "deterministic-test",
            "--work-root",
            str(tmp_path),
        ]
    )
    second_exit = main(
        [
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "reduce_sum",
            "--generator",
            "deterministic-test",
            "--work-root",
            str(tmp_path),
            "--reuse-memory",
        ]
    )
    assert first_exit == 0
    assert second_exit == 0
    second_report = json.loads((tmp_path / "artifacts" / "reduce_sum" / "run_report.json").read_text(encoding="utf-8"))
    assert second_report["memory"]["loaded_item_count"] > 0
    assert second_report["memory"]["reused_memory_ids"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_cpu_simd_pipeline.py -v`
Expected: FAIL with import errors or missing CLI

- [ ] **Step 3: Implement the CLI and deterministic integration path**

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = build_runtime(args)
    report = run_episode(runtime, task_id=args.task)
    return 0 if report.best_candidate is not None else 1
```

Implement:
- CLI argument parsing
- runtime builder that wires config, backend, memory, generator, and task registry
- deterministic generator fixture for tests only
- real provider runtime path that instantiates `OpenAICompatibleGenerator` from config when `--generator deterministic-test` is not selected
- artifact directory output for run reports
- shared-memory load/save path so a second task can consume memories written by the first task
- run-report evidence of memory reuse, including loaded-item count and reused memory IDs for each task

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_cpu_simd_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evokernel/cli.py src/evokernel/backend/cpu_simd.py src/evokernel/verifier/core.py src/evokernel/orchestrator/episode.py tests/integration/test_cpu_simd_pipeline.py
git commit -m "feat: add cli and end-to-end cpu simd pipeline"
```

### Task 10: Run Full Verification And Tighten Developer Ergonomics

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Modify: `configs/default.toml`
- Modify: `configs/cpu_simd.toml`

- [ ] **Step 1: Add a failing regression test for config defaults or fixtures discovered during full-suite runs**

```python
def test_default_config_uses_layernorm_in_initial_benchmark():
    config = load_runtime_config(Path("configs/default.toml"))
    assert "layernorm" in config.benchmark.tasks
```

- [ ] **Step 2: Run the targeted regression test to verify it fails**

Run: `uv run pytest tests/domain/test_models.py::test_default_config_uses_layernorm_in_initial_benchmark -v`
Expected: FAIL until defaults are aligned

- [ ] **Step 3: Implement the minimal fix and add final quality commands**

Update defaults, fixtures, and `pyproject.toml` so the following commands work:

```bash
uv run pytest
uv run ruff check .
uv run python -m evokernel.cli --config configs/cpu_simd.toml --task vector_add --generator deterministic-test
```

- [ ] **Step 4: Run the full verification suite**

Run: `uv run pytest`
Expected: PASS

Run: `uv run ruff check .`
Expected: PASS

Run: `uv run python -m evokernel.cli --config configs/cpu_simd.toml --task vector_add --generator deterministic-test`
Expected: exits `0` and writes a run report under the configured artifact directory

Run: `uv run pytest tests/integration/test_cpu_simd_pipeline.py::test_cpu_simd_pipeline_reuses_memory_across_two_tasks -v`
Expected: PASS

Run: `uv run pytest tests/generator/test_openai_compatible.py::test_openai_compatible_generator_generate_uses_http_client -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/conftest.py configs/default.toml configs/cpu_simd.toml
git commit -m "chore: finalize evokernel prototype verification path"
```

## Implementation Notes

- Use `layernorm` as the fourth benchmark task in v1.
- Keep generated C/C++ harnesses in per-attempt directories under a configurable work root so failed builds remain inspectable.
- Keep provider-specific HTTP logic isolated to `src/evokernel/generator/openai_compatible.py`. Do not let raw HTTP requests leak into the orchestrator.
- Do not block implementation on a real hosted model. The deterministic test generator exists only to make CI and local verification repeatable.
- If `clang` is unavailable, the toolchain wrapper may fall back to `gcc`, but the backend must surface the chosen compiler in run artifacts.
