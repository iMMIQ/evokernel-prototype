from evokernel.backend.cpu_simd import CpuSimdBackend
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


def test_vector_add_task_exposes_reference_inputs_tolerances_and_prompt_metadata():
    task = build_vector_add_task()

    assert isinstance(task, BenchmarkTask)
    assert callable(task.reference_impl)
    assert task.tolerances.atol > 0
    assert "simd" in task.prompt_metadata["keywords"]
