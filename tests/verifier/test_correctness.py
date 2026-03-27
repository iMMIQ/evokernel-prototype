from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evokernel.backend.base import CompilationResult, ReferenceExecutionResult
from evokernel.benchmarks.models import BenchmarkTask, BenchmarkTolerances
from evokernel.benchmarks.cpu_simd_tasks import build_vector_add_task
from evokernel.verifier.correctness import compare_outputs
from evokernel.verifier.core import verify_candidate


def test_compare_outputs_uses_atol_rtol():
    passed, summary = compare_outputs(
        actual=[1.0, 2.001],
        expected=[1.0, 2.0],
        atol=1e-2,
        rtol=1e-2,
    )

    assert passed is True
    assert summary is None


def test_compare_outputs_reports_shape_mismatch():
    passed, summary = compare_outputs(
        actual=[1.0, 2.0],
        expected=[[1.0, 2.0]],
        atol=1e-6,
        rtol=1e-6,
    )

    assert passed is False
    assert "shape mismatch" in summary


@dataclass(slots=True)
class _CompileFailBackend:
    stderr: str

    def materialize_candidate(self, task, candidate_code, attempt_id):
        return attempt_id

    def compile(self, artifact):
        return CompilationResult(
            command=["clang", "candidate.cpp"],
            returncode=1,
            stdout="",
            stderr=self.stderr,
            binary_path=Path("candidate.so"),
        )

    def extract_structured_error(self, stderr):
        return type(
            "Structured",
            (),
            {"category": "compile_error", "message": stderr.strip()},
        )()


def test_verify_candidate_uses_structured_compile_error_summary():
    outcome = verify_candidate(
        backend=_CompileFailBackend(stderr="candidate.cpp:3: error: bad code"),
        task=build_vector_add_task(),
        candidate_code='extern "C" void evokernel_entry() {}',
        attempt_id="attempt-compile-failure",
    )

    assert outcome.anti_hack_passed is True
    assert outcome.compile_passed is False
    assert outcome.correctness_passed is False
    assert outcome.error_category == "compile_error"
    assert "bad code" in outcome.feedback_summary


@dataclass(slots=True)
class _CorrectnessBackend:
    compile_called: int = 0
    cases_run: list[dict] | None = None

    def __post_init__(self):
        if self.cases_run is None:
            self.cases_run = []

    def materialize_candidate(self, task, candidate_code, attempt_id):
        return type("Artifact", (), {"task": task})()

    def compile(self, artifact):
        self.compile_called += 1
        return CompilationResult(
            command=["clang", "candidate.cpp"],
            returncode=0,
            stdout="",
            stderr="",
            binary_path=Path("candidate.so"),
        )

    def run_reference_case(self, artifact, case):
        self.cases_run.append(case)
        if len(self.cases_run) == 1:
            output = np.asarray(case["a"] + case["b"], dtype=np.float32)
        else:
            output = np.zeros_like(case["a"])
        return ReferenceExecutionResult(case_path=Path("case.json"), output=output)

    def measure_latency(self, artifact, case, warmup_runs, timed_runs):
        raise AssertionError("profiling should not run after correctness fails")

    def extract_structured_error(self, stderr):
        return None


def test_verify_candidate_checks_all_cases_and_fails_on_first_mismatch():
    backend = _CorrectnessBackend()
    task = build_vector_add_task()

    outcome = verify_candidate(
        backend=backend,
        task=task,
        candidate_code='extern "C" void evokernel_entry() {}',
        attempt_id="attempt-wrong-answer",
    )

    assert outcome.anti_hack_passed is True
    assert outcome.compile_passed is True
    assert outcome.correctness_passed is False
    assert outcome.error_category == "wrong_answer"
    assert len(backend.cases_run) == 2
    assert "case 2" in outcome.feedback_summary


@dataclass(slots=True)
class _NoCaseBackend:
    measured_latency: bool = False

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
        raise AssertionError("no-case tasks must not execute correctness runs")

    def measure_latency(self, artifact, case, warmup_runs, timed_runs):
        self.measured_latency = True
        raise AssertionError("no-case tasks must not be profiled")

    def extract_structured_error(self, stderr):
        return None


def test_verify_candidate_fails_when_task_has_no_correctness_cases():
    backend = _NoCaseBackend()
    task = BenchmarkTask(
        task_id="no_cases",
        operator_family="test",
        summary="Task without correctness cases.",
        reference_impl=lambda: np.asarray(0.0, dtype=np.float32),
        randomized_inputs=[],
        edge_case_inputs=[],
        tolerances=BenchmarkTolerances(atol=1e-6, rtol=1e-6),
    )

    outcome = verify_candidate(
        backend=backend,
        task=task,
        candidate_code='extern "C" void evokernel_entry() {}',
        attempt_id="attempt-no-cases",
    )

    assert outcome.anti_hack_passed is True
    assert outcome.compile_passed is True
    assert outcome.correctness_passed is False
    assert outcome.error_category == "missing_correctness_cases"
    assert "no correctness cases" in outcome.feedback_summary
    assert backend.measured_latency is False
