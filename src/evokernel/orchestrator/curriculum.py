from __future__ import annotations

from evokernel.benchmarks.task_registry import get_benchmark_task
from evokernel.orchestrator.curriculum_report import CurriculumReport
from evokernel.orchestrator.episode import run_episode


def run_curriculum(runtime, task_ids: list[str]) -> CurriculumReport:
    strategy = runtime.config.curriculum.strategy
    ordered_ids = _order_tasks(runtime, task_ids, strategy)

    report = CurriculumReport(strategy=strategy)
    for task_id in ordered_ids:
        episode_report = run_episode(runtime, task_id=task_id)
        report.task_reports.append((task_id, episode_report))

    return report


def _order_tasks(runtime, task_ids: list[str], strategy: str) -> list[str]:
    if strategy == "l1_then_l2":
        return _order_by_difficulty(runtime, task_ids)
    if strategy == "mixed":
        return _order_mixed(runtime, task_ids)
    return list(task_ids)


def _order_by_difficulty(runtime, task_ids: list[str]) -> list[str]:
    task_difficulties: dict[str, int] = {}
    for task_id in task_ids:
        task = get_benchmark_task(task_id)
        task_difficulties[task_id] = task.difficulty

    return sorted(task_ids, key=lambda tid: task_difficulties[tid])


def _order_mixed(runtime, task_ids: list[str]) -> list[str]:
    task_difficulties: dict[str, int] = {}
    for task_id in task_ids:
        task = get_benchmark_task(task_id)
        task_difficulties[task_id] = task.difficulty

    l1 = [tid for tid in task_ids if task_difficulties[tid] == 1]
    l2 = [tid for tid in task_ids if task_difficulties[tid] >= 2]

    mixed: list[str] = []
    i, j = 0, 0
    while i < len(l1) or j < len(l2):
        if i < len(l1):
            mixed.append(l1[i])
            i += 1
        if j < len(l2):
            mixed.append(l2[j])
            j += 1
    return mixed
