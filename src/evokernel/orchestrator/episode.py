from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt, tanh
import re

from evokernel.benchmarks.task_registry import get_benchmark_task
from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem
from evokernel.generator.base import GenerationRequest
from evokernel.memory.document import build_retrieval_query
from evokernel.memory.state_signature import build_state_signature
from evokernel.orchestrator.run_report import AttemptRecord, RunReport
from evokernel.retrieval.policy import select_context_items
from evokernel.retrieval.recall import recall_candidates
from evokernel.verifier.core import verify_candidate


@dataclass(slots=True)
class _OnlineRewardNormalizer:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def normalize(self, value: float) -> float:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if self.count < 2:
            return value

        variance = self.m2 / (self.count - 1)
        if variance <= 0:
            return value

        return (value - self.mean) / sqrt(variance)


@dataclass(slots=True)
class _ContextSelection:
    experiential_items: list[MemoryItem]
    api_knowledge_items: list[MemoryItem]


def run_episode(runtime, task_id: str) -> RunReport:
    task = get_benchmark_task(task_id)
    attempt_budget = runtime.config.runtime.attempt_budget
    report = RunReport(task_id=task_id, backend_id=runtime.backend_id)

    stage = Stage.DRAFTING
    best_candidate: MemoryItem | None = None
    last_error_category: str | None = None
    last_feedback: str | None = None
    reward_normalizer = _OnlineRewardNormalizer()
    memory_to_attempt: dict[str, str] = {}
    shape_bucket = _build_shape_bucket(task)

    for attempt_index in range(1, attempt_budget + 1):
        state_signature = build_state_signature(
            backend_id=runtime.backend_id,
            operator_family=task.operator_family,
            stage=stage,
            shape_bucket=shape_bucket,
            error_category=last_error_category,
        )
        start_point = _select_start_point(
            runtime=runtime,
            task=task,
            state_signature=state_signature,
        ) if stage == Stage.REFINING else None
        context_selection = _select_context(
            runtime=runtime,
            task=task,
            stage=stage,
            state_signature=state_signature,
            shape_bucket=shape_bucket,
            error_category=last_error_category,
            feedback_summary=last_feedback,
            start_point=start_point,
        )

        retrieved_context = _build_retrieved_context(
            selected_context=context_selection.experiential_items,
            start_point=start_point,
        )
        request = GenerationRequest(
            stage=stage.value,
            task_summary=task.summary,
            backend_constraints=_get_backend_constraints(runtime, task),
            retrieved_context=retrieved_context,
            api_knowledge_context=_build_api_knowledge_context(
                context_selection.api_knowledge_items
            ),
            profiler_summary=_resolve_profiler_summary(start_point),
            feedback_summary=last_feedback,
        )
        generation = runtime.generator.generate(request)
        attempt_id = f"{task_id}-{attempt_index}"
        verifier_outcome = _verify(
            runtime=runtime,
            task=task,
            candidate_code=generation.code,
            attempt_id=attempt_id,
        )
        reward = _calculate_reward(
            stage=stage,
            verifier_outcome=verifier_outcome,
            best_candidate=best_candidate,
            reward_normalizer=reward_normalizer,
        )

        memory_item = runtime.memory_store.add(
            MemoryItem(
                task_id=task_id,
                backend_id=runtime.backend_id,
                operator_family=task.operator_family,
                stage=stage,
                code=generation.code,
                summary=_build_attempt_summary(
                    stage=stage,
                    attempt_index=attempt_index,
                    reward=reward,
                    latency_ms=verifier_outcome.latency_ms,
                ),
                context_summary=_build_context_summary(
                    experiential_context=context_selection.experiential_items,
                    api_knowledge_context=context_selection.api_knowledge_items,
                    start_point=start_point,
                ),
                reward=reward,
                is_feasible=verifier_outcome.is_feasible,
                became_start_point=verifier_outcome.is_feasible,
                verifier_outcome=verifier_outcome,
                parent_attempt_id=(
                    memory_to_attempt.get(start_point.memory_id)
                    if start_point is not None
                    else None
                ),
                parent_memory_id=(
                    start_point.memory_id if start_point is not None else None
                ),
            )
        )
        memory_to_attempt[memory_item.memory_id] = attempt_id

        _update_q_values(
            runtime=runtime,
            stage=stage,
            state_signature=state_signature,
            reward=reward,
            selected_context=context_selection.experiential_items,
            start_point=start_point,
            produced_item=memory_item,
        )

        report.attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                memory_id=memory_item.memory_id,
                stage=stage,
                reward=reward,
                verifier_outcome=verifier_outcome,
                selected_context_ids=[
                    item.memory_id
                    for item in (
                        context_selection.experiential_items
                        + context_selection.api_knowledge_items
                    )
                ],
                start_point_id=(
                    start_point.memory_id if start_point is not None else None
                ),
            )
        )

        if _is_better_candidate(memory_item, best_candidate):
            best_candidate = memory_item
            report.best_candidate = memory_item

        if stage == Stage.DRAFTING and verifier_outcome.is_feasible:
            stage = Stage.REFINING

        last_error_category = verifier_outcome.error_category
        last_feedback = verifier_outcome.feedback_summary

    return report


def _select_context(
    runtime,
    task,
    stage: Stage,
    state_signature: str,
    shape_bucket: str,
    error_category: str | None,
    feedback_summary: str | None,
    start_point: MemoryItem | None,
) -> _ContextSelection:
    experiential_items = _select_experiential_context(
        runtime=runtime,
        task=task,
        stage=stage,
        state_signature=state_signature,
        shape_bucket=shape_bucket,
        error_category=error_category,
        feedback_summary=feedback_summary,
        start_point=start_point,
    )
    api_knowledge_items: list[MemoryItem] = []
    if stage == Stage.DRAFTING:
        api_knowledge_items = _select_api_knowledge_context(
            runtime=runtime,
            task=task,
            stage=stage,
            shape_bucket=shape_bucket,
            error_category=error_category,
            feedback_summary=feedback_summary,
            experiential_items=experiential_items,
        )
    return _ContextSelection(
        experiential_items=experiential_items,
        api_knowledge_items=api_knowledge_items,
    )


def _select_experiential_context(
    runtime,
    task,
    stage: Stage,
    state_signature: str,
    shape_bucket: str,
    error_category: str | None,
    feedback_summary: str | None,
    start_point: MemoryItem | None,
) -> list[MemoryItem]:
    retrieval_config = runtime.config.retrieval
    query_text = build_retrieval_query(
        backend_id=runtime.backend_id,
        task_id=task.task_id,
        operator_family=task.operator_family,
        task_summary=task.summary,
        stage=stage,
        shape_bucket=shape_bucket,
        keywords=list(task.prompt_metadata.get("keywords", [])),
        error_category=error_category,
        feedback_summary=feedback_summary,
        bottleneck_label=_resolve_bottleneck_label(start_point),
        profiler_summary=_resolve_profiler_summary(start_point),
        start_point=start_point,
    )
    query_embedding = runtime.embedder.embed_texts([query_text])[0]
    recalled_items = [
        item
        for item in runtime.memory_store.recall(
            backend_id=runtime.backend_id,
            exclude_memory_ids=(
                {start_point.memory_id}
                if start_point is not None
                else None
            ),
        )
        if item.memory_kind != "backend_knowledge"
    ]
    recalled = recall_candidates(
        items=recalled_items,
        query_embedding=query_embedding,
        final_context_count=retrieval_config.final_context_count,
        over_retrieval_lambda=retrieval_config.over_retrieval_lambda,
    )
    if stage == Stage.REFINING and start_point is not None:
        recalled = _rerank_refinement_candidates(
            recalled,
            start_point=start_point,
        )
    return select_context_items(
        candidates=recalled,
        stage=stage,
        state_signature=state_signature,
        q_store=runtime.q_store,
        final_context_count=retrieval_config.final_context_count,
        epsilon=retrieval_config.epsilon,
    )


def _select_api_knowledge_context(
    runtime,
    task,
    stage: Stage,
    shape_bucket: str,
    error_category: str | None,
    feedback_summary: str | None,
    experiential_items: list[MemoryItem],
) -> list[MemoryItem]:
    retrieval_config = runtime.config.retrieval
    api_budget = max(1, min(2, retrieval_config.final_context_count))
    query_text = build_retrieval_query(
        backend_id=runtime.backend_id,
        task_id=task.task_id,
        operator_family=task.operator_family,
        task_summary=task.summary,
        stage=stage,
        shape_bucket=shape_bucket,
        keywords=list(task.prompt_metadata.get("keywords", [])),
        error_category=error_category,
        feedback_summary=feedback_summary,
        start_point=None,
    )
    query_embedding = runtime.embedder.embed_texts([query_text])[0]
    candidates = runtime.memory_store.recall(
        backend_id=runtime.backend_id,
        memory_kind="backend_knowledge",
    )
    recalled = recall_candidates(
        items=candidates,
        query_embedding=query_embedding,
        final_context_count=api_budget,
        over_retrieval_lambda=retrieval_config.over_retrieval_lambda,
    )
    exact_names = _extract_api_terms(
        task=task,
        experiential_items=experiential_items,
    )
    indexed_recalled = {item.memory_id: index for index, item in enumerate(recalled)}
    ranked = sorted(
        recalled,
        key=lambda item: (
            _is_static_infrastructure_bundle(item),
            _count_exact_name_hits(item, exact_names),
            _count_task_keyword_hits(item, task),
            -indexed_recalled[item.memory_id],
        ),
        reverse=True,
    )
    return ranked[:api_budget]


def _select_start_point(runtime, task, state_signature: str) -> MemoryItem | None:
    start_points = runtime.memory_store.list_start_points(task.task_id)
    if not start_points:
        return None

    selected = select_context_items(
        candidates=start_points,
        stage=Stage.REFINING,
        state_signature=state_signature,
        q_store=runtime.q_store,
        final_context_count=1,
        epsilon=runtime.config.retrieval.epsilon,
    )
    if not selected:
        return None
    return selected[0]


def _verify(runtime, task, candidate_code: str, attempt_id: str):
    if hasattr(runtime, "verifier"):
        return runtime.verifier(
            backend=runtime.backend,
            task=task,
            candidate_code=candidate_code,
            attempt_id=attempt_id,
        )
    return verify_candidate(
        backend=runtime.backend,
        task=task,
        candidate_code=candidate_code,
        attempt_id=attempt_id,
    )


def _calculate_reward(
    stage: Stage,
    verifier_outcome,
    best_candidate: MemoryItem | None,
    reward_normalizer: _OnlineRewardNormalizer,
) -> float:
    if stage == Stage.DRAFTING:
        return 1.0 if verifier_outcome.is_feasible else -1.0

    if not verifier_outcome.is_feasible:
        return -1.0

    best_latency = (
        best_candidate.verifier_outcome.latency_ms
        if best_candidate is not None
        else None
    )
    new_latency = verifier_outcome.latency_ms
    if best_latency is None or new_latency is None:
        return 0.0
    if best_latency <= 0 or new_latency <= 0:
        return 0.0

    raw_reward = tanh(log(best_latency) - log(new_latency))
    return reward_normalizer.normalize(raw_reward)


def _update_q_values(
    runtime,
    stage: Stage,
    state_signature: str,
    reward: float,
    selected_context: list[MemoryItem],
    start_point: MemoryItem | None,
    produced_item: MemoryItem,
) -> None:
    alpha = runtime.config.retrieval.alpha
    for item in selected_context:
        runtime.q_store.update(
            stage=stage,
            state_signature=state_signature,
            memory_id=item.memory_id,
            reward=reward,
            alpha=alpha,
        )

    if start_point is not None:
        already_updated = any(
            item.memory_id == start_point.memory_id
            for item in selected_context
        )
        if not already_updated:
            runtime.q_store.update(
                stage=stage,
                state_signature=state_signature,
                memory_id=start_point.memory_id,
                reward=reward,
                alpha=alpha,
            )

    if stage == Stage.REFINING and produced_item.is_feasible:
        runtime.q_store.update(
            stage=stage,
            state_signature=state_signature,
            memory_id=produced_item.memory_id,
            reward=reward,
            alpha=alpha,
        )


def _get_backend_constraints(runtime, task) -> list[str]:
    constraints = getattr(runtime, "backend_constraints", None)
    if constraints is not None:
        return list(constraints)

    metadata_keywords = task.prompt_metadata.get("keywords", [])
    return [f"keyword: {keyword}" for keyword in metadata_keywords]


def _build_retrieved_context(
    selected_context: list[MemoryItem],
    start_point: MemoryItem | None,
) -> list[str]:
    retrieved_context: list[str] = []
    if start_point is not None:
        profiler_summary = (
            start_point.verifier_outcome.profiler_summary or "none"
        )
        bottleneck_label = (
            start_point.verifier_outcome.bottleneck_label or "none"
        )
        retrieved_context.append(
            "Start point:\n"
            f"memory_id={start_point.memory_id}\n"
            f"summary={start_point.summary}\n"
            f"bottleneck={bottleneck_label}\n"
            f"profile={profiler_summary}\n"
            f"code={start_point.code}"
        )

    seen_memory_ids = {start_point.memory_id} if start_point is not None else set()
    for item in selected_context:
        if item.memory_id in seen_memory_ids:
            continue
        label = _label_refinement_item(
            item=item,
            start_point=start_point,
        )
        retrieved_context.append(
            f"{label}:\n"
            f"memory_id={item.memory_id}\n"
            f"kind={item.memory_kind}\n"
            f"summary={item.summary}\n"
            f"bottleneck={item.verifier_outcome.bottleneck_label or 'none'}\n"
            f"feedback={item.verifier_outcome.feedback_summary or 'none'}"
        )
        seen_memory_ids.add(item.memory_id)
    return retrieved_context


def _build_api_knowledge_context(items: list[MemoryItem]) -> list[str]:
    return [
        (
            "Knowledge:\n"
            f"memory_id={item.memory_id}\n"
            f"summary={item.summary}\n"
            f"snippet={item.code}"
        )
        for item in items
    ]


def _build_context_summary(
    experiential_context: list[MemoryItem],
    api_knowledge_context: list[MemoryItem],
    start_point: MemoryItem | None,
) -> str | None:
    parts: list[str] = []
    if start_point is not None:
        parts.append(f"start_point={start_point.memory_id}")
    if experiential_context:
        parts.append(
            "context=" + ",".join(item.memory_id for item in experiential_context)
        )
    if api_knowledge_context:
        parts.append(
            "api_knowledge="
            + ",".join(item.memory_id for item in api_knowledge_context)
        )
    if not parts:
        return None
    return " | ".join(parts)


def _build_attempt_summary(
    stage: Stage,
    attempt_index: int,
    reward: float,
    latency_ms: float | None,
) -> str:
    summary = f"{stage.value} attempt {attempt_index} reward={reward:.3f}"
    if latency_ms is not None:
        return f"{summary} latency_ms={latency_ms:.3f}"
    return summary


def _is_better_candidate(
    candidate: MemoryItem,
    current_best: MemoryItem | None,
) -> bool:
    if not candidate.is_feasible:
        return False
    if current_best is None or not current_best.is_feasible:
        return True

    current_latency = current_best.verifier_outcome.latency_ms
    candidate_latency = candidate.verifier_outcome.latency_ms
    if current_latency is None:
        return candidate_latency is not None
    if candidate_latency is None:
        return False
    return candidate_latency < current_latency


def _resolve_bottleneck_label(start_point: MemoryItem | None) -> str | None:
    if start_point is None:
        return None
    return start_point.verifier_outcome.bottleneck_label


def _resolve_profiler_summary(start_point: MemoryItem | None) -> str | None:
    if start_point is None:
        return None
    return start_point.verifier_outcome.profiler_summary


def _rerank_refinement_candidates(
    candidates: list[MemoryItem],
    *,
    start_point: MemoryItem,
) -> list[MemoryItem]:
    target_bottleneck = start_point.verifier_outcome.bottleneck_label
    start_latency = start_point.verifier_outcome.latency_ms
    indexed = {item.memory_id: index for index, item in enumerate(candidates)}
    ranked = sorted(
        candidates,
        key=lambda item: (
            _is_observable_child(item, start_point=start_point),
            _matches_bottleneck(item, target_bottleneck),
            _is_complementary_high_performing_variant(
                item,
                start_point=start_point,
                start_latency=start_latency,
            ),
            item.is_feasible,
            -indexed[item.memory_id],
        ),
        reverse=True,
    )
    return ranked


def _is_observable_child(item: MemoryItem, *, start_point: MemoryItem) -> bool:
    return item.parent_memory_id == start_point.memory_id


def _matches_bottleneck(
    item: MemoryItem,
    target_bottleneck: str | None,
) -> bool:
    if target_bottleneck is None:
        return False
    if item.verifier_outcome.bottleneck_label == target_bottleneck:
        return True
    haystack = f"{item.summary}\n{item.retrieval_text or ''}".lower()
    return target_bottleneck.lower() in haystack


def _is_complementary_high_performing_variant(
    item: MemoryItem,
    *,
    start_point: MemoryItem,
    start_latency: float | None,
) -> bool:
    if not item.is_feasible:
        return False
    if item.task_id != start_point.task_id:
        return False
    latency = item.verifier_outcome.latency_ms
    if start_latency is None or latency is None:
        return False
    return latency < start_latency


def _label_refinement_item(
    *,
    item: MemoryItem,
    start_point: MemoryItem | None,
) -> str:
    if start_point is not None and item.parent_memory_id == start_point.memory_id:
        return "Observable Child Variant"
    if item.memory_kind == "refinement_hint":
        return "Refinement Hint"
    if (
        start_point is not None
        and item.is_feasible
        and item.task_id == start_point.task_id
    ):
        return "Complementary Variant"
    return "Context"


def _extract_api_terms(task, experiential_items: list[MemoryItem]) -> set[str]:
    terms = set(task.prompt_metadata.get("preferred_intrinsics", []))
    terms.update(task.prompt_metadata.get("keywords", []))
    for item in experiential_items:
        terms.update(
            re.findall(
                r"__\w+|_mm\w+|immintrin\.h|xmmintrin\.h|emmintrin\.h|vld1q_f32",
                item.code,
            )
        )
    return {term.lower() for term in terms if term}


def _is_static_infrastructure_bundle(item: MemoryItem) -> bool:
    return ":core:" in item.memory_id


def _count_exact_name_hits(item: MemoryItem, exact_names: set[str]) -> int:
    haystack = f"{item.summary}\n{item.code}\n{item.retrieval_text or ''}".lower()
    return sum(1 for term in exact_names if term in haystack)


def _count_task_keyword_hits(item: MemoryItem, task) -> int:
    haystack = f"{item.summary}\n{item.retrieval_text or ''}".lower()
    keywords = [
        task.operator_family,
        *task.prompt_metadata.get("keywords", []),
    ]
    return sum(1 for keyword in keywords if keyword.lower() in haystack)


def _build_shape_bucket(task) -> str:
    exemplar_inputs = task.randomized_inputs or task.edge_case_inputs
    if not exemplar_inputs:
        return "unknown"

    shape_parts: list[str] = []
    total_elements = 0
    for value in exemplar_inputs[0].values():
        shape = getattr(value, "shape", None)
        if not shape:
            continue
        shape_parts.append("x".join(str(dimension) for dimension in shape))
        size = getattr(value, "size", None)
        if isinstance(size, int):
            total_elements += size

    if total_elements <= 64:
        size_bucket = "small"
    elif total_elements <= 1024:
        size_bucket = "medium"
    else:
        size_bucket = "large"

    if not shape_parts:
        return size_bucket
    return f"{'-'.join(shape_parts)}:{size_bucket}"
