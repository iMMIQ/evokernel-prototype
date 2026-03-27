import pytest

from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem, VerificationOutcome
from evokernel.memory.seeds import ingest_seed_memory
from evokernel.memory.state_signature import build_state_signature
from evokernel.memory.store import InMemoryStore
from evokernel.retrieval.q_store import QValueStore
from evokernel.retrieval.reward import update_q_value


def _build_memory_item(*, memory_id: str, task_id: str = "vector_add") -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        task_id=task_id,
        backend_id="cpu_simd",
        operator_family="elementwise",
        stage=Stage.DRAFTING,
        code="void evokernel_entry() {}",
        summary=f"summary-{memory_id}",
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


def test_q_value_store_persists_to_sqlite(tmp_path):
    path = tmp_path / "memory.sqlite3"
    first = QValueStore(db_path=path)
    first.set(
        stage=Stage.DRAFTING,
        state_signature="sig",
        memory_id="memory-1",
        value=0.8,
    )
    first.close()

    second = QValueStore(db_path=path)
    assert second.get(
        stage=Stage.DRAFTING,
        state_signature="sig",
        memory_id="memory-1",
    ) == 0.8
    second.close()


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


def test_memory_item_rejects_inconsistent_feasibility():
    with pytest.raises(ValueError, match="is_feasible"):
        MemoryItem(
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void evokernel_entry() {}",
            summary="compile fix",
            reward=1.0,
            is_feasible=True,
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


def test_memory_item_rejects_infeasible_start_point():
    with pytest.raises(ValueError, match="became_start_point"):
        MemoryItem(
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void evokernel_entry() {}",
            summary="compile fix",
            reward=-1.0,
            is_feasible=False,
            became_start_point=True,
            verifier_outcome=VerificationOutcome(
                anti_hack_passed=True,
                compile_passed=False,
                correctness_passed=False,
                latency_ms=None,
                error_category="compile_error",
                feedback_summary="missing include",
            ),
        )


def test_memory_store_sqlite_round_trip(tmp_path):
    path = tmp_path / "memory.sqlite3"
    writer = InMemoryStore(path)
    writer.add(_build_memory_item(memory_id="persisted"))
    writer.close()

    reader = InMemoryStore(path, reuse_existing=True)
    recalled = reader.recall(task_id="vector_add")

    assert len(recalled) == 1
    assert recalled[0].summary == "summary-persisted"
    reader.close()


def test_memory_store_hides_historical_rows_without_reuse(tmp_path):
    path = tmp_path / "memory.sqlite3"
    first = InMemoryStore(path)
    first.add(_build_memory_item(memory_id="history"))
    first.close()

    second = InMemoryStore(path, reuse_existing=False)
    assert second.loaded_memory_ids == []
    assert second.recall(task_id="vector_add") == []

    second.add(_build_memory_item(memory_id="current"))
    assert [item.memory_id for item in second.recall(task_id="vector_add")] == [
        "current"
    ]
    second.close()


def test_memory_store_save_jsonl_is_atomic(tmp_path, monkeypatch):
    store = InMemoryStore()
    store.add(_build_memory_item(memory_id="new-data"))

    path = tmp_path / "memory.jsonl"
    path.write_text("old data\n", encoding="utf-8")

    def fail_replace(self, target):
        assert self != target
        assert path.read_text(encoding="utf-8") == "old data\n"
        raise RuntimeError("replace failed")

    monkeypatch.setattr(type(path), "replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        store.save_jsonl(path)

    assert path.read_text(encoding="utf-8") == "old data\n"


def test_q_value_store_rejects_unsupported_stage():
    store = QValueStore()

    with pytest.raises(ValueError, match="Unsupported stage"):
        store.get(stage="unsupported", state_signature="state", memory_id="m1")


def test_ingest_seed_memory_persists_backend_knowledge_and_hints(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = InMemoryStore(path, reuse_existing=False)

    memory_ids = ingest_seed_memory(
        store,
        backend_id="cpu_simd",
        backend_constraints=["Expose evokernel_entry."],
    )

    visible_items = store.recall(backend_id="cpu_simd")

    assert memory_ids
    assert any(item.memory_kind == "backend_knowledge" for item in visible_items)
    assert any(item.memory_kind == "refinement_hint" for item in visible_items)
    store.close()


def test_ingest_seed_memory_is_idempotent_for_stable_seed_ids(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = InMemoryStore(path)

    first_ids = ingest_seed_memory(
        store,
        backend_id="cpu_simd",
        backend_constraints=["Expose evokernel_entry."],
    )
    second_ids = ingest_seed_memory(
        store,
        backend_id="cpu_simd",
        backend_constraints=["Expose evokernel_entry."],
    )

    persisted = store.recall(task_id="__seed__", backend_id="cpu_simd")

    assert first_ids == second_ids
    assert len(persisted) == len(first_ids)
    store.close()
