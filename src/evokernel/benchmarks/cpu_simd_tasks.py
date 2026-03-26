from __future__ import annotations

import numpy as np

from evokernel.benchmarks.models import BenchmarkTask, BenchmarkTolerances


def _vector_add_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a + b, dtype=np.float32)


def _reduce_sum_reference(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.sum(x, dtype=np.float32), dtype=np.float32)


def _matmul_tiled_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a @ b, dtype=np.float32)


def _layernorm_reference(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + eps)
    return np.asarray(normalized * gamma + beta, dtype=np.float32)


def build_vector_add_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="vector_add",
        operator_family="elementwise",
        summary="Add two float32 vectors with CPU SIMD-friendly contiguous inputs.",
        reference_impl=_vector_add_reference,
        randomized_inputs=[
            {
                "a": np.linspace(0.0, 31.0, 32, dtype=np.float32),
                "b": np.linspace(31.0, 0.0, 32, dtype=np.float32),
            }
        ],
        edge_case_inputs=[
            {
                "a": np.zeros(8, dtype=np.float32),
                "b": np.ones(8, dtype=np.float32),
            }
        ],
        tolerances=BenchmarkTolerances(atol=1e-6, rtol=1e-6),
        prompt_metadata={
            "keywords": ["cpu", "simd", "vector_add", "float32"],
            "preferred_intrinsics": ["sse", "avx", "neon"],
        },
        baseline_data={"target_latency_ms": 0.05},
    )


def build_reduce_sum_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="reduce_sum",
        operator_family="reduction",
        summary="Reduce a float32 vector to a scalar sum.",
        reference_impl=_reduce_sum_reference,
        randomized_inputs=[
            {"x": np.arange(128, dtype=np.float32) / np.float32(7.0)}
        ],
        edge_case_inputs=[{"x": np.array([], dtype=np.float32)}],
        tolerances=BenchmarkTolerances(atol=1e-5, rtol=1e-5),
        prompt_metadata={
            "keywords": ["cpu", "simd", "reduce_sum", "horizontal-reduction"]
        },
        baseline_data={"target_latency_ms": 0.03},
    )


def build_matmul_tiled_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="matmul_tiled",
        operator_family="matmul",
        summary="Multiply two float32 matrices using cache-friendly tiling.",
        reference_impl=_matmul_tiled_reference,
        randomized_inputs=[
            {
                "a": np.arange(64, dtype=np.float32).reshape(8, 8),
                "b": np.flip(
                    np.arange(64, dtype=np.float32).reshape(8, 8) / 11.0,
                    axis=1,
                ),
            }
        ],
        edge_case_inputs=[
            {
                "a": np.array(
                    [
                        [1.0, 2.0, 0.0, -1.0],
                        [0.5, -3.0, 4.0, 2.0],
                        [7.0, 1.5, -2.0, 0.25],
                        [3.0, 3.0, 3.0, 3.0],
                    ],
                    dtype=np.float32,
                ),
                "b": np.array(
                    [
                        [2.0, 1.0, 0.0, -1.0],
                        [1.0, 0.0, 1.0, 2.0],
                        [0.0, -2.0, 3.0, 1.0],
                        [4.0, 1.0, -1.0, 0.5],
                    ],
                    dtype=np.float32,
                ),
            }
        ],
        tolerances=BenchmarkTolerances(atol=1e-4, rtol=1e-4),
        prompt_metadata={
            "keywords": ["cpu", "simd", "matmul", "tiling", "float32"]
        },
        baseline_data={"tile_sizes": [4, 8]},
    )


def build_layernorm_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="layernorm",
        operator_family="normalization",
        summary="Apply float32 layer normalization over the trailing axis.",
        reference_impl=_layernorm_reference,
        randomized_inputs=[
            {
                "x": np.arange(24, dtype=np.float32).reshape(3, 8),
                "gamma": np.array(
                    [0.5, 1.0, 1.5, -0.5, 2.0, 0.25, -1.0, 0.75],
                    dtype=np.float32,
                ),
                "beta": np.array(
                    [1.0, -1.0, 0.25, 0.0, 2.0, -0.5, 1.5, -2.0],
                    dtype=np.float32,
                ),
                "eps": 1e-5,
            }
        ],
        edge_case_inputs=[
            {
                "x": np.array(
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, -2.0, 4.0]],
                    dtype=np.float32,
                ),
                "gamma": np.array([2.0, -1.0, 0.5, 1.5], dtype=np.float32),
                "beta": np.array([-0.5, 1.0, 0.25, 2.0], dtype=np.float32),
                "eps": 1e-5,
            }
        ],
        tolerances=BenchmarkTolerances(atol=1e-4, rtol=1e-4),
        prompt_metadata={
            "keywords": ["cpu", "simd", "layernorm", "float32"]
        },
        baseline_data={"target_latency_ms": 0.08},
    )
