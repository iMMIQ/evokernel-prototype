from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from evokernel.domain.models import MemoryItem


def recall_candidates(
    items: list[MemoryItem],
    query_embedding: Sequence[float],
    *,
    final_context_count: int,
    over_retrieval_lambda: int,
) -> list[MemoryItem]:
    limit = max(final_context_count * over_retrieval_lambda, 0)
    if limit == 0:
        return []

    ranked_items = sorted(
        items,
        key=lambda item: (
            _cosine_similarity(query_embedding, item.embedding),
            item.is_feasible,
            item.became_start_point,
            item.reward,
            item.memory_id,
        ),
        reverse=True,
    )
    return ranked_items[:limit]


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(lhs * rhs for lhs, rhs in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
