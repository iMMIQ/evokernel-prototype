from __future__ import annotations

from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem


def build_memory_document(item: MemoryItem) -> str:
    parts = [
        f"backend={item.backend_id}",
        f"task_id={item.task_id}",
        f"operator_family={item.operator_family}",
        f"stage={item.stage.value}",
        f"memory_kind={item.memory_kind}",
        f"summary={item.summary}",
    ]
    if item.context_summary:
        parts.append(f"context={item.context_summary}")
    if item.verifier_outcome.feedback_summary:
        parts.append(f"feedback={item.verifier_outcome.feedback_summary}")
    if item.verifier_outcome.error_category:
        parts.append(f"error={item.verifier_outcome.error_category}")
    parts.append("code:")
    parts.append(item.code)
    return "\n".join(parts)


def build_retrieval_query(
    *,
    backend_id: str,
    task_id: str,
    operator_family: str,
    task_summary: str,
    stage: Stage,
    shape_bucket: str,
    keywords: list[str] | None = None,
    error_category: str | None = None,
    feedback_summary: str | None = None,
    start_point: MemoryItem | None = None,
) -> str:
    parts = [
        f"backend={backend_id}",
        f"task_id={task_id}",
        f"operator_family={operator_family}",
        f"stage={stage.value}",
        f"shape_bucket={shape_bucket}",
        f"task_summary={task_summary}",
    ]
    if keywords:
        parts.append("keywords=" + ",".join(keywords))
    if error_category:
        parts.append(f"error={error_category}")
    if feedback_summary:
        parts.append(f"feedback={feedback_summary}")
    if start_point is not None:
        parts.append(f"start_point_summary={start_point.summary}")
        parts.append(
            "start_point_feedback="
            f"{start_point.verifier_outcome.feedback_summary or 'none'}"
        )
        parts.append("start_point_code:")
        parts.append(start_point.code)
    return "\n".join(parts)
