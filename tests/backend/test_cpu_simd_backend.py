import json

import numpy as np

from evokernel.backend.cpu_simd import CpuSimdBackend
from evokernel.backend.toolchain import CompilerSpec, CpuSimdToolchain
from evokernel.benchmarks.cpu_simd_tasks import (
    build_layernorm_task,
    build_matmul_tiled_task,
    build_vector_add_task,
)
from evokernel.benchmarks.models import BenchmarkTask


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
    assert artifact.compiler_info_path.exists()

    compiler_info = json.loads(
        artifact.compiler_info_path.read_text(encoding="utf-8")
    )

    assert compiler_info["compiler"] == backend.toolchain.compiler.executable


def test_vector_add_task_exposes_reference_inputs_tolerances_and_prompt_metadata():
    task = build_vector_add_task()

    assert isinstance(task, BenchmarkTask)
    assert callable(task.reference_impl)
    assert task.tolerances.atol > 0
    assert "simd" in task.prompt_metadata["keywords"]


def test_cpu_toolchain_prefers_clang_over_gcc(monkeypatch):
    available = {
        "clang": "/usr/bin/clang",
        "clang++": "/usr/bin/clang++",
        "gcc": "/usr/bin/gcc",
        "g++": "/usr/bin/g++",
    }

    monkeypatch.setattr(
        "evokernel.backend.toolchain.which",
        lambda executable: available.get(executable),
    )

    toolchain = CpuSimdToolchain()

    assert CpuSimdToolchain.COMPILER_PREFERENCE == (
        "clang",
        "clang++",
        "gcc",
        "g++",
    )
    assert toolchain.compiler.executable == "clang"


def test_cpu_toolchain_build_command_includes_harness(tmp_path):
    backend = CpuSimdBackend(
        work_root=tmp_path,
        toolchain=CpuSimdToolchain(compiler=CompilerSpec(executable="clang++")),
    )
    task = build_vector_add_task()

    artifact = backend.materialize_candidate(
        task=task,
        candidate_code='extern "C" void evokernel_entry() {}',
        attempt_id="attempt-compile",
    )
    command = backend.toolchain.build_command(artifact)

    assert str(artifact.source_path) in command
    assert str(artifact.harness_path) in command


def test_run_reference_case_executes_compiled_vector_add_candidate(
    tmp_path,
):
    backend = CpuSimdBackend(
        work_root=tmp_path,
        toolchain=CpuSimdToolchain(compiler=CompilerSpec(executable="clang")),
    )
    task = build_vector_add_task()
    case = task.randomized_inputs[0]

    artifact = backend.materialize_candidate(
        task=task,
        candidate_code="""
#include <cstddef>
extern "C" void evokernel_entry(
    float* out,
    const float* a,
    const float* b,
    std::size_t n
) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}
""",
        attempt_id="attempt-reference",
    )
    compilation = backend.compile(artifact)
    assert compilation.returncode == 0, compilation.stderr

    result = backend.run_reference_case(artifact=artifact, case=case)

    assert result.case_path.exists()
    assert result.output.shape == (32,)
    serialized_case = json.loads(result.case_path.read_text(encoding="utf-8"))
    assert serialized_case["inputs"]["a"][0] == 0.0
    assert serialized_case["task_id"] == task.task_id
    np.testing.assert_allclose(result.output, case["a"] - case["b"])


def test_measure_latency_passes_real_case_buffers_into_candidate_entrypoint(
    tmp_path, monkeypatch
):
    backend = CpuSimdBackend(
        work_root=tmp_path,
        toolchain=CpuSimdToolchain(compiler=CompilerSpec(executable="clang++")),
    )
    task = build_vector_add_task()
    case = task.edge_case_inputs[0]

    artifact = backend.materialize_candidate(
        task=task,
        candidate_code='extern "C" void evokernel_entry() {}',
        attempt_id="attempt-latency",
    )

    recorded_paths: list[str] = []

    class FakeCallable:
        def evokernel_entry(self, out, a, b, n):
            recorded_paths.append(int(n))
            np.testing.assert_allclose(a[:n], case["a"])
            np.testing.assert_allclose(b[:n], case["b"])
            out[:n] = a[:n] + b[:n]

    monkeypatch.setattr(backend, "load_callable", lambda _: FakeCallable())

    latency_ms = backend.measure_latency(
        artifact=artifact,
        case=case,
        warmup_runs=1,
        timed_runs=2,
    )

    assert latency_ms >= 0.0
    assert len(recorded_paths) == 3
    assert recorded_paths == [8, 8, 8]
    serialized_case = json.loads(
        artifact.last_case_path.read_text(encoding="utf-8")
    )
    assert serialized_case["inputs"]["b"] == [1.0] * 8


def test_measure_latency_uses_no_copy_runner_path(tmp_path, monkeypatch):
    backend = CpuSimdBackend(
        work_root=tmp_path,
        toolchain=CpuSimdToolchain(compiler=CompilerSpec(executable="clang++")),
    )
    task = build_vector_add_task()
    case = task.edge_case_inputs[0]
    artifact = backend.materialize_candidate(
        task=task,
        candidate_code='extern "C" void evokernel_entry() {}',
        attempt_id="attempt-latency-nocopy",
    )

    calls: list[str] = []

    def fake_build_candidate_runner(callable_obj, task_id, case, materialize_output):
        calls.append("materialize" if materialize_output else "in_place")

        def run():
            return np.zeros_like(case["a"])

        return run

    monkeypatch.setattr(backend, "load_callable", lambda _: object())
    monkeypatch.setattr(
        backend,
        "_build_candidate_runner",
        fake_build_candidate_runner,
    )

    latency_ms = backend.measure_latency(
        artifact=artifact,
        case=case,
        warmup_runs=1,
        timed_runs=2,
    )

    assert latency_ms >= 0.0
    assert calls == ["in_place"]


def test_matmul_task_cases_do_not_allow_returning_lhs():
    task = build_matmul_tiled_task()
    case = task.randomized_inputs[0]

    assert not np.allclose(case["a"], task.reference_impl(**case))


def test_layernorm_task_cases_require_gamma_and_beta():
    task = build_layernorm_task()
    case = task.randomized_inputs[0]
    reference = task.reference_impl(**case)
    normalized_only = (
        (case["x"] - np.mean(case["x"], axis=-1, keepdims=True))
        / np.sqrt(np.var(case["x"], axis=-1, keepdims=True) + case["eps"])
    )

    assert not np.allclose(reference, normalized_only)


def test_extract_structured_error_prefers_linker_classification():
    backend = CpuSimdBackend(
        toolchain=CpuSimdToolchain(compiler=CompilerSpec(executable="clang"))
    )

    error = backend.extract_structured_error(
        "ld: undefined reference to evokernel_entry\nclang: error: linker command failed"
    )

    assert error is not None
    assert error.category == "link_error"
