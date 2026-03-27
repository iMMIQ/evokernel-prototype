from __future__ import annotations

from typing import Any

from evokernel.domain.models import VerificationOutcome
from evokernel.verifier.anti_hack import check_for_disallowed_patterns
from evokernel.verifier.correctness import compare_outputs
from evokernel.verifier.profiling import aggregate_latency_measurements


def verify_candidate(
    backend: Any,
    task: Any,
    candidate_code: str,
    attempt_id: str,
    *,
    warmup_runs: int = 1,
    timed_runs: int = 5,
    profiling_samples: int = 3,
) -> VerificationOutcome:
    anti_hack = check_for_disallowed_patterns(candidate_code)
    if not anti_hack.passed:
        return VerificationOutcome(
            anti_hack_passed=False,
            compile_passed=False,
            correctness_passed=False,
            latency_ms=None,
            error_category=anti_hack.error_category,
            feedback_summary=anti_hack.feedback_summary,
        )

    artifact = backend.materialize_candidate(
        task=task,
        candidate_code=candidate_code,
        attempt_id=attempt_id,
    )
    compilation = backend.compile(artifact)
    if compilation.returncode != 0:
        category, summary = _extract_backend_error(
            backend=backend,
            stderr=compilation.stderr,
            fallback_category="compile_error",
        )
        return VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=False,
            correctness_passed=False,
            latency_ms=None,
            error_category=category,
            feedback_summary=summary,
        )

    all_cases = [*task.randomized_inputs, *task.edge_case_inputs]
    if not all_cases:
        return VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=False,
            latency_ms=None,
            error_category="missing_correctness_cases",
            feedback_summary="task defines no correctness cases",
        )

    for case_index, case in enumerate(all_cases, start=1):
        try:
            result = backend.run_reference_case(artifact=artifact, case=case)
        except Exception as exc:  # pragma: no cover - defensive path
            category, summary = _extract_backend_error(
                backend=backend,
                stderr=str(exc),
                fallback_category="runtime_error",
            )
            return VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=False,
                latency_ms=None,
                error_category=category,
                feedback_summary=summary,
            )

        expected = task.reference_impl(**case)
        passed, summary = compare_outputs(
            actual=result.output,
            expected=expected,
            atol=task.tolerances.atol,
            rtol=task.tolerances.rtol,
        )
        if passed:
            continue
        return VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=False,
            latency_ms=None,
            error_category="wrong_answer",
            feedback_summary=f"case {case_index}: {summary}",
        )

    profiling_case = _select_profiling_case(task)
    latency_ms = None
    if profiling_case is not None:
        try:
            samples = [
                backend.measure_latency(
                    artifact=artifact,
                    case=profiling_case,
                    warmup_runs=warmup_runs,
                    timed_runs=timed_runs,
                )
                for _ in range(max(profiling_samples, 1))
            ]
        except Exception as exc:  # pragma: no cover - defensive path
            category, summary = _extract_backend_error(
                backend=backend,
                stderr=str(exc),
                fallback_category="runtime_error",
            )
            return VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=False,
                latency_ms=None,
                error_category=category,
                feedback_summary=summary,
            )
        latency_ms = aggregate_latency_measurements(samples)

    return VerificationOutcome(
        anti_hack_passed=True,
        compile_passed=True,
        correctness_passed=True,
        latency_ms=latency_ms,
        error_category=None,
        feedback_summary=None,
    )


def _select_profiling_case(task: Any) -> dict[str, Any] | None:
    if task.randomized_inputs:
        return task.randomized_inputs[0]
    if task.edge_case_inputs:
        return task.edge_case_inputs[0]
    return None


def _extract_backend_error(
    backend: Any,
    stderr: str,
    fallback_category: str,
) -> tuple[str, str]:
    structured = backend.extract_structured_error(stderr)
    if structured is not None:
        return structured.category, structured.message
    summary = stderr.strip() or fallback_category
    return fallback_category, summary
