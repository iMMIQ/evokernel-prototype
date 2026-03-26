import json

from evokernel.backend.cpu_simd import CpuSimdBackend
from evokernel.backend.toolchain import CompilerSpec, CpuSimdToolchain
from evokernel.benchmarks.cpu_simd_tasks import build_vector_add_task
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
        "clang++": "/usr/bin/clang++",
        "g++": "/usr/bin/g++",
    }

    monkeypatch.setattr(
        "evokernel.backend.toolchain.which",
        lambda executable: available.get(executable),
    )

    toolchain = CpuSimdToolchain()

    assert toolchain.compiler.executable == "clang++"


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


def test_run_reference_case_serializes_case_into_attempt_artifact(tmp_path):
    backend = CpuSimdBackend(
        work_root=tmp_path,
        toolchain=CpuSimdToolchain(compiler=CompilerSpec(executable="clang++")),
    )
    task = build_vector_add_task()
    case = task.randomized_inputs[0]

    artifact = backend.materialize_candidate(
        task=task,
        candidate_code='extern "C" void evokernel_entry() {}',
        attempt_id="attempt-reference",
    )
    result = backend.run_reference_case(artifact=artifact, case=case)

    assert result.case_path.exists()
    assert result.output.shape == (32,)
    serialized_case = json.loads(result.case_path.read_text(encoding="utf-8"))
    assert serialized_case["inputs"]["a"][0] == 0.0
    assert serialized_case["task_id"] == task.task_id


def test_measure_latency_uses_serialized_case_and_harness_entrypoint(
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
        def evokernel_run_case(self, case_path):
            recorded_paths.append(case_path.decode("utf-8"))
            return len(recorded_paths)

    monkeypatch.setattr(backend, "load_callable", lambda _: FakeCallable())

    latency_ms = backend.measure_latency(
        artifact=artifact,
        case=case,
        warmup_runs=1,
        timed_runs=2,
    )

    assert latency_ms >= 0.0
    assert len(recorded_paths) == 3
    assert all(path == recorded_paths[0] for path in recorded_paths)
    serialized_case = json.loads(
        artifact.last_case_path.read_text(encoding="utf-8")
    )
    assert serialized_case["inputs"]["b"] == [1.0] * 8
