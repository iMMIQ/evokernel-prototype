from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt, tanh

from evokernel.benchmarks.task_registry import get_benchmark_task
from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem
from evokernel.generator.base import GenerationRequest
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


def run_episode(runtime, task_id: str) -> RunReport:
    task = get_benchmark_task(task_id)
    retrieval_config = runtime.config.retrieval
    attempt_budget = getattr(runtime.config.runtime, "attempt_budget", 1)
    report = RunReport(task_id=task_id, backend_id=runtime.backend_id)

    stage = Stage.DRAFTING
    best_candidate: MemoryItem | None = None
    last_error_category: str | None = None
    last_feedback: str | None = None
    reward_normalizer = _OnlineRewardNormalizer()
    memory_to_attempt: dict[str, str] = {}

    for attempt_index in range(1, attempt_budget + 1):
        state_signature = build_state_signature(
            backend_id=runtime.backend_id,
            operator_family=task.operator_family,
            stage=stage,
            shape_bucket=_build_shape_bucket(task),
            error_category=last_error_category,
        )
        selected_context = _select_context(
            runtime=runtime,
            task=task,
            stage=stage,
            state_signature=state_signature,
        )
        start_point = _select_start_point(
            runtime=runtime,
            task=task,
            state_signature=state_signature,
        ) if stage == Stage.REFINING else None

        retrieved_context = _build_retrieved_context(
            selected_context=selected_context,
            start_point=start_point,
        )
        request = GenerationRequest(
            stage=stage.value,
            task_summary=task.summary,
            backend_constraints=_get_backend_constraints(runtime, task),
            retrieved_context=retrieved_context,
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
                context_summary=_build_context_summary(selected_context, start_point),
                reward=reward,
                is_feasible=verifier_outcome.is_feasible,
                became_start_point=verifier_outcome.is_feasible,
                verifier_outcome=verifier_outcome,
                parent_attempt_id=(
                    memory_to_attempt.get(start_point.memory_id)
                    if start_point is not None
                    else None
                ),
            )
        )
        memory_to_attempt[memory_item.memory_id] = attempt_id

        _update_q_values(
            runtime=runtime,
            stage=stage,
            state_signature=state_signature,
            reward=reward,
            selected_context=selected_context,
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
                selected_context_ids=[item.memory_id for item in selected_context],
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


def _select_context(runtime, task, stage: Stage, state_signature: str) -> list[MemoryItem]:
    retrieval_config = runtime.config.retrieval
    recalled = recall_candidates(
        items=runtime.memory_store.recall(task.task_id),
        operator_family=task.operator_family,
        backend_id=runtime.backend_id,
        stage=Stage.DRAFTING if stage == Stage.DRAFTING else None,
        final_context_count=retrieval_config.final_context_count,
        over_retrieval_lambda=retrieval_config.over_retrieval_lambda,
    )
    return select_context_items(
        candidates=recalled,
        stage=stage,
        state_signature=state_signature,
        q_store=runtime.q_store,
        final_context_count=retrieval_config.final_context_count,
        epsilon=retrieval_config.epsilon,
    )


def _select_start_point(runtime, task, state_signature: str) -> MemoryItem | None:
    start_points = list(reversed(runtime.memory_store.list_start_points(task.task_id)))
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
        retrieved_context.append(
            "Start point:\n"
            f"memory_id={start_point.memory_id}\n"
            f"summary={start_point.summary}\n"
            f"code={start_point.code}"
        )

    seen_memory_ids = {start_point.memory_id} if start_point is not None else set()
    for item in selected_context:
        if item.memory_id in seen_memory_ids:
            continue
        retrieved_context.append(
            "Context:\n"
            f"memory_id={item.memory_id}\n"
            f"summary={item.summary}\n"
            f"feedback={item.verifier_outcome.feedback_summary or 'none'}"
        )
        seen_memory_ids.add(item.memory_id)
    return retrieved_context


def _build_context_summary(
    selected_context: list[MemoryItem],
    start_point: MemoryItem | None,
) -> str | None:
    parts: list[str] = []
    if start_point is not None:
        parts.append(f"start_point={start_point.memory_id}")
    if selected_context:
        parts.append(
            "context=" + ",".join(item.memory_id for item in selected_context)
        )
    if not parts:
        return None
    return " ".join(parts)


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
