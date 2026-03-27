from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evokernel.backend.base import CompilationResult, ReferenceExecutionResult
from evokernel.benchmarks.cpu_simd_tasks import build_vector_add_task
from evokernel.verifier.core import verify_candidate
from evokernel.verifier.profiling import aggregate_latency_measurements


def test_aggregate_latency_measurements_returns_median():
    assert aggregate_latency_measurements([3.0, 1.0, 2.0]) == 2.0


def test_aggregate_latency_measurements_averages_even_length_middle_pair():
    assert aggregate_latency_measurements([1.0, 2.0, 10.0, 12.0]) == 6.0


@dataclass(slots=True)
class _ProfilingBackend:
    measured_case: dict | None = None

    def materialize_candidate(self, task, candidate_code, attempt_id):
        return type("Artifact", (), {"task": task})()

    def compile(self, artifact):
        return CompilationResult(
            command=["clang", "candidate.cpp"],
            returncode=0,
            stdout="",
            stderr="",
            binary_path=Path("candidate.so"),
        )

    def run_reference_case(self, artifact, case):
        return ReferenceExecutionResult(
            case_path=Path("case.json"),
            output=np.asarray(case["a"] + case["b"], dtype=np.float32),
        )

    def measure_latency(self, artifact, case, warmup_runs, timed_runs):
        self.measured_case = case
        return [3.0, 1.0, 2.0][timed_runs - 1]

    def extract_structured_error(self, stderr):
        return None


def test_verify_candidate_profiles_first_randomized_case():
    backend = _ProfilingBackend()
    task = build_vector_add_task()

    outcome = verify_candidate(
        backend=backend,
        task=task,
        candidate_code='extern "C" void evokernel_entry() {}',
        attempt_id="attempt-profile",
        warmup_runs=1,
        timed_runs=3,
    )

    assert outcome.anti_hack_passed is True
    assert outcome.compile_passed is True
    assert outcome.correctness_passed is True
    assert outcome.latency_ms == 2.0
    assert backend.measured_case == task.randomized_inputs[0]
