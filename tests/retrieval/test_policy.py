import pytest

from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem, VerificationOutcome
from evokernel.retrieval.policy import select_context_items
from evokernel.retrieval.q_store import QValueStore
from evokernel.retrieval.recall import recall_candidates


@pytest.fixture
def memory_items() -> list[MemoryItem]:
    outcome = VerificationOutcome(
        anti_hack_passed=True,
        compile_passed=True,
        correctness_passed=True,
        latency_ms=1.0,
        error_category=None,
        feedback_summary="ok",
    )
    return [
        MemoryItem(
            memory_id="best-id",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void best();",
            summary="best",
            reward=0.9,
            is_feasible=True,
            became_start_point=True,
            verifier_outcome=outcome,
        ),
        MemoryItem(
            memory_id="second-id",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void second();",
            summary="second",
            reward=0.7,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
        ),
        MemoryItem(
            memory_id="third-id",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.REFINING,
            code="void third();",
            summary="third",
            reward=0.3,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
        ),
        MemoryItem(
            memory_id="fourth-id",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void fourth();",
            summary="fourth",
            reward=0.2,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
        ),
        MemoryItem(
            memory_id="fifth-id",
            task_id="vector_add",
            backend_id="cuda",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void fifth();",
            summary="fifth",
            reward=0.1,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
        ),
        MemoryItem(
            memory_id="sixth-id",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="reduction",
            stage=Stage.DRAFTING,
            code="void sixth();",
            summary="sixth",
            reward=-0.1,
            is_feasible=False,
            became_start_point=False,
            verifier_outcome=VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=True,
                correctness_passed=False,
                latency_ms=None,
                error_category="wrong_answer",
                feedback_summary="bad",
            ),
        ),
    ]


def test_recall_candidates_limits_pool_by_lambda_times_n(memory_items):
    recalled = recall_candidates(
        items=memory_items,
        operator_family="elementwise",
        backend_id="cpu_simd",
        stage=Stage.DRAFTING,
        final_context_count=2,
        over_retrieval_lambda=3,
    )
    assert len(recalled) == 6


def test_recall_candidates_prefers_backend_and_stage_matches_before_reward():
    outcome = VerificationOutcome(
        anti_hack_passed=True,
        compile_passed=True,
        correctness_passed=True,
        latency_ms=1.0,
        error_category=None,
        feedback_summary="ok",
    )
    items = [
        MemoryItem(
            memory_id="target-match",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void target_match();",
            summary="target-match",
            reward=0.1,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
        ),
        MemoryItem(
            memory_id="wrong-stage",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.REFINING,
            code="void wrong_stage();",
            summary="wrong-stage",
            reward=0.8,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
        ),
        MemoryItem(
            memory_id="wrong-backend",
            task_id="vector_add",
            backend_id="cuda",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void wrong_backend();",
            summary="wrong-backend",
            reward=0.9,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
        ),
    ]

    recalled = recall_candidates(
        items=items,
        backend_id="cpu_simd",
        operator_family="elementwise",
        stage=Stage.DRAFTING,
        final_context_count=2,
        over_retrieval_lambda=2,
    )

    assert [item.summary for item in recalled[:3]] == [
        "target-match",
        "wrong-stage",
        "wrong-backend",
    ]


def test_select_context_items_prefers_high_q_items_when_epsilon_zero(memory_items):
    q_store = QValueStore()
    q_store.set(stage=Stage.DRAFTING, state_signature="sig", memory_id="best-id", value=0.9)
    q_store.set(stage=Stage.DRAFTING, state_signature="sig", memory_id="second-id", value=0.7)
    selected = select_context_items(
        candidates=memory_items,
        stage=Stage.DRAFTING,
        state_signature="sig",
        q_store=q_store,
        final_context_count=2,
        epsilon=0.0,
    )
    assert [item.summary for item in selected] == ["best", "second"]
