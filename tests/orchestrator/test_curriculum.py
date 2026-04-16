from __future__ import annotations

import tempfile
from pathlib import Path

from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem, VerificationOutcome
from evokernel.orchestrator.curriculum import _order_by_difficulty, _order_mixed
from evokernel.orchestrator.curriculum_report import CurriculumReport


def _feasible_outcome() -> VerificationOutcome:
    return VerificationOutcome(
        anti_hack_passed=True,
        compile_passed=True,
        correctness_passed=True,
        latency_ms=None,
        error_category=None,
        feedback_summary=None,
    )


def _failed_outcome(error_category: str = "compile_error") -> VerificationOutcome:
    return VerificationOutcome(
        anti_hack_passed=True,
        compile_passed=False,
        correctness_passed=False,
        latency_ms=None,
        error_category=error_category,
        feedback_summary="test failure",
    )


def test_order_by_difficulty_places_l1_first():
    task_ids = ["layernorm", "vector_add", "matmul_tiled", "reduce_sum"]
    ordered = _order_by_difficulty(None, task_ids)

    assert ordered[:2] == ["vector_add", "reduce_sum"]
    assert set(ordered[2:]) == {"layernorm", "matmul_tiled"}


def test_order_mixed_interleaves_l1_and_l2():
    task_ids = ["layernorm", "vector_add", "matmul_tiled", "reduce_sum"]
    ordered = _order_mixed(None, task_ids)

    assert len(ordered) == 4
    l1_indices = [ordered.index(tid) for tid in ["vector_add", "reduce_sum"]]
    l2_indices = [ordered.index(tid) for tid in ["layernorm", "matmul_tiled"]]
    assert min(l1_indices) < max(l2_indices)
    assert min(l2_indices) < max(l1_indices)


def test_curriculum_report_properties():
    from evokernel.orchestrator.run_report import RunReport

    report = CurriculumReport(
        task_reports=[
            ("vector_add", RunReport(task_id="vector_add", backend_id="cpu_simd")),
            ("layernorm", RunReport(task_id="layernorm", backend_id="cpu_simd")),
        ],
        strategy="l1_then_l2",
    )

    assert report.task_ids == ["vector_add", "layernorm"]
    assert report.total_count == 2
    assert report.solved_count == 0


def test_import_from_store_transfers_items():
    from evokernel.memory.embedding import HashingTextEmbedder
    from evokernel.memory.store import InMemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.sqlite3"
        source_store = InMemoryStore(source_path, embedder=HashingTextEmbedder())

        item = MemoryItem(
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void foo() {}",
            summary="test item from source",
            context_summary="test",
            memory_kind="generation_trace",
            reward=1.0,
            is_feasible=True,
            became_start_point=True,
            verifier_outcome=_feasible_outcome(),
        )
        source_store.add(item)
        source_store.close()

        target_store = InMemoryStore(embedder=HashingTextEmbedder())
        count = target_store.import_from_store(source_path)

        assert count == 1
        recalled = target_store.recall(task_id="vector_add")
        assert len(recalled) == 1
        assert recalled[0].summary == "test item from source"


def test_import_from_store_excludes_task_ids():
    from evokernel.memory.embedding import HashingTextEmbedder
    from evokernel.memory.store import InMemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.sqlite3"
        source_store = InMemoryStore(source_path, embedder=HashingTextEmbedder())

        item_a = MemoryItem(
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void a() {}",
            summary="item A",
            context_summary="test",
            memory_kind="generation_trace",
            reward=1.0,
            is_feasible=True,
            became_start_point=False,
            verifier_outcome=_feasible_outcome(),
        )
        item_b = MemoryItem(
            task_id="reduce_sum",
            backend_id="cpu_simd",
            operator_family="reduction",
            stage=Stage.DRAFTING,
            code="void b() {}",
            summary="item B",
            context_summary="test",
            memory_kind="generation_trace",
            reward=-1.0,
            is_feasible=False,
            became_start_point=False,
            verifier_outcome=_failed_outcome(),
        )
        source_store.add(item_a)
        source_store.add(item_b)
        source_store.close()

        target_store = InMemoryStore(embedder=HashingTextEmbedder())
        count = target_store.import_from_store(
            source_path,
            exclude_task_ids=["reduce_sum"],
        )

        assert count == 1
        recalled = target_store.recall()
        assert len(recalled) == 1
        assert recalled[0].task_id == "vector_add"


def test_import_from_store_skips_duplicates():
    from evokernel.memory.embedding import HashingTextEmbedder
    from evokernel.memory.store import InMemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.sqlite3"
        source_store = InMemoryStore(source_path, embedder=HashingTextEmbedder())

        item = MemoryItem(
            memory_id="fixed-id-123",
            task_id="vector_add",
            backend_id="cpu_simd",
            operator_family="elementwise",
            stage=Stage.DRAFTING,
            code="void foo() {}",
            summary="duplicate item",
            context_summary="test",
            memory_kind="generation_trace",
            reward=0.0,
            is_feasible=False,
            became_start_point=False,
            verifier_outcome=_failed_outcome(),
        )
        source_store.add(item)
        source_store.close()

        target_store = InMemoryStore(embedder=HashingTextEmbedder())
        target_store.add(item)
        count = target_store.import_from_store(source_path)

        assert count == 0


def test_import_from_store_nonexistent_path():
    from evokernel.memory.embedding import HashingTextEmbedder
    from evokernel.memory.store import InMemoryStore

    store = InMemoryStore(embedder=HashingTextEmbedder())
    count = store.import_from_store("/nonexistent/path.sqlite3")
    assert count == 0
