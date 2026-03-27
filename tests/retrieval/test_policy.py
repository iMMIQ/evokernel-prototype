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
            embedding=[1.0, 0.0],
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
            embedding=[0.8, 0.2],
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
            embedding=[0.2, 0.8],
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
            embedding=[0.0, 1.0],
        ),
    ]


def test_recall_candidates_limits_pool_by_lambda_times_n(memory_items):
    recalled = recall_candidates(
        items=memory_items,
        query_embedding=[1.0, 0.0],
        final_context_count=2,
        over_retrieval_lambda=2,
    )
    assert [item.memory_id for item in recalled] == [
        "best-id",
        "second-id",
        "third-id",
        "fourth-id",
    ]


def test_recall_candidates_uses_reward_and_memory_id_as_tie_breakers():
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
            memory_id="z-id",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void z();",
            summary="z",
            reward=0.5,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
            embedding=[1.0, 0.0],
        ),
        MemoryItem(
            memory_id="a-id",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void a();",
            summary="a",
            reward=0.5,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=outcome,
            embedding=[1.0, 0.0],
        ),
    ]

    recalled = recall_candidates(
        items=items,
        query_embedding=[1.0, 0.0],
        final_context_count=2,
        over_retrieval_lambda=1,
    )

    assert [item.memory_id for item in recalled] == ["z-id", "a-id"]


def test_select_context_items_prefers_high_q_items_when_epsilon_zero(memory_items):
    q_store = QValueStore()
    q_store.set(stage=Stage.DRAFTING, state_signature="sig", memory_id="best-id", value=0.9)
    q_store.set(stage=Stage.DRAFTING, state_signature="sig", memory_id="second-id", value=0.7)
    candidates = [memory_items[1], memory_items[3], memory_items[0], memory_items[2]]
    selected = select_context_items(
        candidates=candidates,
        stage=Stage.DRAFTING,
        state_signature="sig",
        q_store=q_store,
        final_context_count=2,
        epsilon=0.0,
    )
    assert [item.summary for item in selected] == ["best", "second"]
