from __future__ import annotations

from types import SimpleNamespace

import pytest

from evokernel.domain.models import VerificationOutcome
from evokernel.memory.store import InMemoryStore
from evokernel.retrieval.q_store import QValueStore
from evokernel.orchestrator.episode import run_episode


@pytest.fixture
def fake_runtime():
    responses = {
        "draft-1": VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=False,
            latency_ms=None,
            error_category="wrong_answer",
            feedback_summary="tail mismatch",
        ),
        "draft-2": VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=True,
            latency_ms=2.0,
            error_category=None,
            feedback_summary=None,
        ),
        "refine-1": VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=True,
            latency_ms=1.5,
            error_category=None,
            feedback_summary=None,
        ),
    }

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def generate(self, request):
            self.calls.append(request)
            if request.stage == "drafting":
                return SimpleNamespace(code=f"draft-{len(self.calls)}")
            return SimpleNamespace(code=f"refine-{len(self.calls) - 2}")

    def verifier(*, candidate_code: str, **_kwargs):
        return responses[candidate_code]

    return SimpleNamespace(
        backend=object(),
        backend_id="cpu_simd",
        generator=FakeGenerator(),
        memory_store=InMemoryStore(),
        q_store=QValueStore(),
        verifier=verifier,
        config=SimpleNamespace(
            retrieval=SimpleNamespace(
                final_context_count=2,
                over_retrieval_lambda=2,
                epsilon=0.0,
                alpha=0.5,
            ),
            runtime=SimpleNamespace(attempt_budget=3),
        ),
    )


def test_run_episode_switches_to_refining_after_first_feasible_attempt(
    fake_runtime,
):
    report = run_episode(fake_runtime, task_id="vector_add")

    assert report.attempts[0].stage == "drafting"
    assert any(attempt.stage == "refining" for attempt in report.attempts)


def test_run_episode_reuses_feasible_draft_as_refining_start_point(
    fake_runtime,
):
    report = run_episode(fake_runtime, task_id="vector_add")

    first_feasible_draft = next(
        attempt
        for attempt in report.attempts
        if attempt.stage == "drafting" and attempt.verifier_outcome.is_feasible
    )
    refining_attempt = next(
        attempt for attempt in report.attempts if attempt.stage == "refining"
    )

    assert refining_attempt.start_point_id == first_feasible_draft.memory_id


def test_run_episode_updates_best_latency_from_feasible_refinements(
    fake_runtime,
):
    report = run_episode(fake_runtime, task_id="vector_add")

    assert report.best_candidate is not None
    assert report.best_candidate.verifier_outcome.latency_ms == 1.5
