from __future__ import annotations

from evokernel.benchmarks.cpu_simd_tasks import (
    build_layernorm_task,
    build_matmul_tiled_task,
    build_reduce_sum_task,
    build_vector_add_task,
)
from evokernel.benchmarks.models import BenchmarkTask


def get_benchmark_task(task_id: str) -> BenchmarkTask:
    registry = {
        "vector_add": build_vector_add_task,
        "reduce_sum": build_reduce_sum_task,
        "matmul_tiled": build_matmul_tiled_task,
        "layernorm": build_layernorm_task,
    }
    try:
        return registry[task_id]()
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark task: {task_id}") from exc


def list_benchmark_tasks() -> list[BenchmarkTask]:
    return [
        build_vector_add_task(),
        build_reduce_sum_task(),
        build_matmul_tiled_task(),
        build_layernorm_task(),
    ]


def list_tasks_by_difficulty(difficulty: int | None = None) -> list[BenchmarkTask]:
    all_tasks = list_benchmark_tasks()
    if difficulty is None:
        return all_tasks
    return [t for t in all_tasks if t.difficulty == difficulty]


def get_all_task_ids() -> list[str]:
    return [t.task_id for t in list_benchmark_tasks()]
