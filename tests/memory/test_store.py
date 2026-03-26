from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem, VerificationOutcome
from evokernel.memory.state_signature import build_state_signature
from evokernel.memory.store import InMemoryStore
from evokernel.retrieval.q_store import QValueStore
from evokernel.retrieval.reward import update_q_value


def test_build_state_signature_uses_backend_task_stage_and_error():
    signature = build_state_signature(
        backend_id="cpu_simd",
        operator_family="reduction",
        stage=Stage.DRAFTING,
        shape_bucket="1d_small",
        error_category="compile_error",
    )
    assert signature == "cpu_simd|reduction|drafting|1d_small|compile_error"


def test_memory_store_persists_attempts_and_start_points():
    store = InMemoryStore()
    item = MemoryItem(
        task_id="reduce_sum",
        backend_id="cpu_simd",
        operator_family="reduction",
        stage=Stage.REFINING,
        code="void kernel();",
        summary="first feasible refinement",
        reward=0.5,
        is_feasible=True,
        became_start_point=True,
        verifier_outcome=VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=True,
            latency_ms=1.2,
            error_category=None,
            feedback_summary="ok",
        ),
    )
    store.add(item)
    assert store.list_start_points("reduce_sum")[0].summary == "first feasible refinement"


def test_update_q_value_uses_monte_carlo_rule():
    assert update_q_value(current=0.25, reward=1.0, alpha=0.2) == 0.4


def test_q_value_store_tracks_q1_and_q2_independently():
    store = QValueStore()
    key = "cpu_simd|elementwise|drafting|1d_small|compile_error"
    store.update(stage=Stage.DRAFTING, state_signature=key, memory_id="m1", reward=1.0, alpha=0.5)
    store.update(stage=Stage.REFINING, state_signature=key, memory_id="m1", reward=-1.0, alpha=0.5)
    assert store.get(stage=Stage.DRAFTING, state_signature=key, memory_id="m1") == 0.5
    assert store.get(stage=Stage.REFINING, state_signature=key, memory_id="m1") == -0.5


def test_memory_item_serialization_round_trip():
    item = MemoryItem(
        task_id="vector_add",
        backend_id="cpu_simd",
        operator_family="elementwise",
        stage=Stage.DRAFTING,
        code="void evokernel_entry() {}",
        summary="compile fix",
        context_summary="api: include immintrin",
        memory_kind="failure_summary",
        reward=-1.0,
        is_feasible=False,
        became_start_point=False,
        verifier_outcome=VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=False,
            correctness_passed=False,
            latency_ms=None,
            error_category="compile_error",
            feedback_summary="missing include",
        ),
    )
    payload = item.model_dump()
    restored = MemoryItem.model_validate(payload)
    assert restored.context_summary == "api: include immintrin"


def test_memory_store_jsonl_round_trip(tmp_path):
    store = InMemoryStore()
    item = MemoryItem(
        task_id="vector_add",
        backend_id="cpu_simd",
        operator_family="elementwise",
        stage=Stage.DRAFTING,
        code="void evokernel_entry() {}",
        summary="first draft",
        memory_kind="generation_trace",
        reward=0.25,
        is_feasible=False,
        became_start_point=False,
        verifier_outcome=VerificationOutcome(
            anti_hack_passed=True,
            compile_passed=True,
            correctness_passed=False,
            latency_ms=None,
            error_category="wrong_answer",
            feedback_summary="edge cases fail",
        ),
    )
    store.add(item)

    path = tmp_path / "memory.jsonl"
    store.save_jsonl(path)
    restored = InMemoryStore.load_jsonl(path)

    recalled = restored.recall("vector_add")
    assert len(recalled) == 1
    assert recalled[0].summary == "first draft"
