from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem


def recall_candidates(
    items: list[MemoryItem],
    operator_family: str,
    *,
    backend_id: str | None = None,
    stage: Stage | None = None,
    final_context_count: int,
    over_retrieval_lambda: int,
) -> list[MemoryItem]:
    limit = max(final_context_count * over_retrieval_lambda, 0)
    ranked_items = sorted(
        items,
        key=lambda item: (
            item.backend_id == backend_id if backend_id is not None else True,
            item.operator_family == operator_family,
            item.stage == stage if stage is not None else True,
            item.is_feasible,
            item.became_start_point,
            item.reward,
            item.summary,
            item.memory_id,
        ),
        reverse=True,
    )
    return ranked_items[:limit]
