from collections.abc import Sequence
from random import Random

from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem
from evokernel.retrieval.q_store import QValueStore


def select_context_items(
    candidates: Sequence[MemoryItem],
    stage: Stage,
    state_signature: str,
    q_store: QValueStore,
    final_context_count: int,
    epsilon: float,
    random_source: Random | None = None,
) -> list[MemoryItem]:
    indexed_candidates = list(enumerate(candidates))
    if epsilon == 0:
        ranked = sorted(
            indexed_candidates,
            key=lambda pair: (
                q_store.get(
                    stage=stage,
                    state_signature=state_signature,
                    memory_id=pair[1].memory_id,
                ),
                -pair[0],
            ),
            reverse=True,
        )
        return [item for _, item in ranked[:final_context_count]]

    rng = random_source or Random()
    remaining = indexed_candidates[:]
    selected: list[MemoryItem] = []

    while remaining and len(selected) < final_context_count:
        if rng.random() < epsilon:
            index = rng.randrange(len(remaining))
            _, item = remaining.pop(index)
            selected.append(item)
            continue

        best_index = max(
            range(len(remaining)),
            key=lambda idx: (
                q_store.get(
                    stage=stage,
                    state_signature=state_signature,
                    memory_id=remaining[idx][1].memory_id,
                ),
                -remaining[idx][0],
            ),
        )
        _, item = remaining.pop(best_index)
        selected.append(item)

    return selected
