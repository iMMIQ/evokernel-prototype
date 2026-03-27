from collections.abc import Sequence
from math import inf
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


def select_context_items_by_policy(
    candidates: Sequence[MemoryItem],
    *,
    policy: str,
    stage: Stage,
    state_signature: str,
    q_store: QValueStore,
    final_context_count: int,
    epsilon: float,
    random_source: Random | None = None,
) -> list[MemoryItem]:
    if policy == "value_driven":
        return select_context_items(
            candidates=candidates,
            stage=stage,
            state_signature=state_signature,
            q_store=q_store,
            final_context_count=final_context_count,
            epsilon=epsilon,
            random_source=random_source,
        )
    if policy == "heuristic":
        return list(candidates[:final_context_count])
    raise ValueError(f"Unsupported retrieval policy: {policy}")


def select_start_point_by_policy(
    candidates: Sequence[MemoryItem],
    *,
    policy: str,
    stage: Stage,
    state_signature: str,
    q_store: QValueStore,
    epsilon: float,
    random_source: Random | None = None,
) -> MemoryItem | None:
    if not candidates:
        return None
    if policy == "value_driven":
        selected = select_context_items(
            candidates=candidates,
            stage=stage,
            state_signature=state_signature,
            q_store=q_store,
            final_context_count=1,
            epsilon=epsilon,
            random_source=random_source,
        )
        if not selected:
            return None
        return selected[0]
    if policy == "heuristic":
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.verifier_outcome.latency_ms is None,
                (
                    item.verifier_outcome.latency_ms
                    if item.verifier_outcome.latency_ms is not None
                    else inf
                ),
                -item.reward,
                item.memory_id,
            ),
        )
        return ranked[0] if ranked else None
    raise ValueError(f"Unsupported retrieval policy: {policy}")
