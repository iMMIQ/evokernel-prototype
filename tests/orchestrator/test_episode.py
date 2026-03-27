from __future__ import annotations

from types import SimpleNamespace

import pytest

from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem, VerificationOutcome
from evokernel.memory.embedding import HashingTextEmbedder
from evokernel.memory.seeds import ingest_seed_memory
from evokernel.memory.store import InMemoryStore
from evokernel.retrieval.q_store import QValueStore
from evokernel.orchestrator.episode import run_episode


def _build_fake_runtime(
    responses: dict[str, VerificationOutcome],
    *,
    attempt_budget: int,
):
    class FakeGenerator:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def generate(self, request):
            self.calls.append(request)
            if request.stage == "drafting":
                draft_number = sum(
                    1 for call in self.calls if call.stage == "drafting"
                )
                return SimpleNamespace(code=f"draft-{draft_number}")

            refine_number = sum(
                1 for call in self.calls if call.stage == "refining"
            )
            return SimpleNamespace(code=f"refine-{refine_number}")

    def verifier(*, candidate_code: str, **_kwargs):
        return responses[candidate_code]

    return SimpleNamespace(
        backend=object(),
        backend_id="cpu_simd",
        generator=FakeGenerator(),
        memory_store=InMemoryStore(),
        q_store=QValueStore(),
        embedder=HashingTextEmbedder(dimensions=32),
        verifier=verifier,
        config=SimpleNamespace(
            retrieval=SimpleNamespace(
                policy="value_driven",
                final_context_count=2,
                over_retrieval_lambda=2,
                epsilon=0.0,
                alpha=0.5,
            ),
            runtime=SimpleNamespace(attempt_budget=attempt_budget),
        ),
    )


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
    return _build_fake_runtime(responses, attempt_budget=3)


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


def test_run_episode_prefers_new_feasible_refinement_as_next_start_point():
    runtime = _build_fake_runtime(
        {
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
            "refine-2": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=1.0,
                error_category=None,
                feedback_summary=None,
            ),
        },
        attempt_budget=4,
    )

    report = run_episode(runtime, task_id="vector_add")

    assert report.attempts[2].stage == "refining"
    assert report.attempts[3].stage == "refining"
    assert report.attempts[3].start_point_id == report.attempts[2].memory_id


def test_run_episode_handles_zero_latency_refinement_rewards():
    runtime = _build_fake_runtime(
        {
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
                latency_ms=0.0,
                error_category=None,
                feedback_summary=None,
            ),
            "refine-1": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=0.0,
                error_category=None,
                feedback_summary=None,
            ),
        },
        attempt_budget=3,
    )

    report = run_episode(runtime, task_id="vector_add")

    assert report.best_candidate is not None
    assert report.best_candidate.verifier_outcome.latency_ms == 0.0


def test_run_episode_separates_api_knowledge_from_experiential_context():
    runtime = _build_fake_runtime(
        {
            "draft-1": VerificationOutcome(
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
                latency_ms=1.0,
                error_category=None,
                feedback_summary=None,
            ),
        },
        attempt_budget=2,
    )
    ingest_seed_memory(
        runtime.memory_store,
        backend_id="cpu_simd",
        backend_constraints=["Expose evokernel_entry."],
    )
    runtime.memory_store.add(_build_runtime_memory_item("history"))

    report = run_episode(runtime, task_id="vector_add")

    first_request = runtime.generator.calls[0]

    assert report.attempts[0].stage == "drafting"
    assert first_request.api_knowledge_context
    assert first_request.retrieved_context
    assert "API Knowledge" not in "\n".join(first_request.retrieved_context)
    assert "api_knowledge" in report.attempts[0].context_role_ids


def test_run_episode_passes_profiler_summary_into_refining_request():
    runtime = _build_fake_runtime(
        {
            "draft-1": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=2.0,
                bottleneck_label="vectorization_gap",
                profiler_summary="Profiler diagnosis: likely bottleneck=vectorization_gap.",
                latency_ratio_to_target=40.0,
                error_category=None,
                feedback_summary=None,
            ),
            "refine-1": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=1.0,
                bottleneck_label="vectorization_gap",
                profiler_summary="Profiler diagnosis: likely bottleneck=vectorization_gap.",
                latency_ratio_to_target=20.0,
                error_category=None,
                feedback_summary=None,
            ),
        },
        attempt_budget=2,
    )

    run_episode(runtime, task_id="vector_add")

    refining_request = runtime.generator.calls[1]

    assert refining_request.stage == "refining"
    assert "vectorization_gap" in (refining_request.profiler_summary or "")


def test_run_episode_surfaces_observable_child_variants_in_refining_context():
    runtime = _build_fake_runtime(
        {
            "draft-1": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=2.0,
                bottleneck_label="vectorization_gap",
                profiler_summary="Profiler diagnosis: likely bottleneck=vectorization_gap.",
                latency_ratio_to_target=40.0,
                error_category=None,
                feedback_summary=None,
            ),
            "refine-1": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=False,
                latency_ms=None,
                bottleneck_label=None,
                profiler_summary=None,
                latency_ratio_to_target=None,
                error_category="wrong_answer",
                feedback_summary="tail cleanup regressed",
            ),
            "refine-2": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=1.8,
                bottleneck_label="vectorization_gap",
                profiler_summary="Profiler diagnosis: likely bottleneck=vectorization_gap.",
                latency_ratio_to_target=36.0,
                error_category=None,
                feedback_summary=None,
            ),
        },
        attempt_budget=3,
    )

    report = run_episode(runtime, task_id="vector_add")

    third_request = runtime.generator.calls[2]
    third_attempt = report.attempts[2]

    assert third_request.stage == "refining"
    assert any(
        "Observable Child Variant" in entry
        for entry in third_request.retrieved_context
    )
    assert "observable_child" in third_attempt.context_role_ids


def test_run_episode_heuristic_policy_chooses_best_latency_start_point():
    runtime = _build_fake_runtime(
        {
            "draft-1": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=2.0,
                bottleneck_label="vectorization_gap",
                profiler_summary="Profiler diagnosis: likely bottleneck=vectorization_gap.",
                latency_ratio_to_target=40.0,
                error_category=None,
                feedback_summary=None,
            ),
            "refine-1": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=1.6,
                bottleneck_label="vectorization_gap",
                profiler_summary="Profiler diagnosis: likely bottleneck=vectorization_gap.",
                latency_ratio_to_target=32.0,
                error_category=None,
                feedback_summary=None,
            ),
        },
        attempt_budget=2,
    )
    runtime.config.retrieval.policy = "heuristic"

    start_point_a = _build_runtime_memory_item("start-a").model_copy(
        update={
            "task_id": "vector_add",
            "is_feasible": True,
            "became_start_point": True,
            "verifier_outcome": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=1.8,
                bottleneck_label="vectorization_gap",
                profiler_summary="Profiler diagnosis: likely bottleneck=vectorization_gap.",
                latency_ratio_to_target=36.0,
                error_category=None,
                feedback_summary=None,
            ),
        }
    )
    start_point_b = _build_runtime_memory_item("start-b").model_copy(
        update={
            "task_id": "vector_add",
            "is_feasible": True,
            "became_start_point": True,
            "verifier_outcome": VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=True,
                latency_ms=1.2,
                bottleneck_label="vectorization_gap",
                profiler_summary="Profiler diagnosis: likely bottleneck=vectorization_gap.",
                latency_ratio_to_target=24.0,
                error_category=None,
                feedback_summary=None,
            ),
        }
    )
    runtime.memory_store.add(start_point_a)
    runtime.memory_store.add(start_point_b)

    report = run_episode(runtime, task_id="vector_add")

    refining_attempt = next(
        attempt for attempt in report.attempts if attempt.stage == "refining"
    )
    assert refining_attempt.start_point_id == "start-b"


def _build_runtime_memory_item(memory_id: str):
    return MemoryItem(
        memory_id=memory_id,
        task_id="vector_add",
        backend_id="cpu_simd",
        operator_family="elementwise",
        stage=Stage.DRAFTING,
        code="#include <immintrin.h>\nvoid evokernel_entry() {}",
        summary="historical compile fix with __m256 and tail cleanup",
        memory_kind="failure_summary",
        reward=0.25,
        is_feasible=False,
        became_start_point=False,
        verifier_outcome=VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=False,
            latency_ms=None,
            error_category="wrong_answer",
            feedback_summary="tail cleanup missing",
        ),
    )
