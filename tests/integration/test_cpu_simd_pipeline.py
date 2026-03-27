from __future__ import annotations

import json

import evokernel.cli as cli_module
from evokernel.cli import main
from evokernel.generator.base import GenerationRequest, GenerationResult


class _DeterministicTestGenerator:
    _VECTOR_ADD_CODE = """
#include <cstddef>

// fixture deterministic vector_add
extern "C" void evokernel_entry(
    float* out,
    const float* a,
    const float* b,
    std::size_t n
) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
""".strip()

    _REDUCE_SUM_CODE = """
#include <cstddef>

// fixture deterministic reduce_sum
extern "C" void evokernel_entry(
    float* out,
    const float* x,
    std::size_t n
) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    out[0] = sum;
}
""".strip()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        task_summary = request.task_summary.lower()
        if "add two float32 vectors" in task_summary:
            return GenerationResult(code=self._VECTOR_ADD_CODE)
        if "reduce a float32 vector to a scalar sum" in task_summary:
            return GenerationResult(code=self._REDUCE_SUM_CODE)
        raise ValueError(request.task_summary)


def _install_deterministic_generator_override(monkeypatch) -> None:
    monkeypatch.setitem(
        cli_module.GENERATOR_OVERRIDES,
        "deterministic-test",
        lambda _config: _DeterministicTestGenerator(),
    )


def test_cpu_simd_pipeline_runs_vector_add(tmp_path, monkeypatch):
    _install_deterministic_generator_override(monkeypatch)

    exit_code = main(
        [
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "vector_add",
            "--generator",
            "deterministic-test",
            "--work-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    report = json.loads(
        (
            tmp_path / "artifacts" / "vector_add" / "run_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["task_id"] == "vector_add"
    assert report["best_candidate"] is not None
    candidate_code = (
        tmp_path
        / "artifacts"
        / "vector_add"
        / "vector_add-1"
        / "candidate.cpp"
    ).read_text(encoding="utf-8")
    assert "fixture deterministic vector_add" in candidate_code


def test_cpu_simd_pipeline_reuses_memory_across_two_tasks(tmp_path, monkeypatch):
    _install_deterministic_generator_override(monkeypatch)

    first_exit = main(
        [
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "vector_add",
            "--generator",
            "deterministic-test",
            "--work-root",
            str(tmp_path),
        ]
    )
    second_exit = main(
        [
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "reduce_sum",
            "--generator",
            "deterministic-test",
            "--work-root",
            str(tmp_path),
            "--reuse-memory",
        ]
    )

    assert first_exit == 0
    assert second_exit == 0

    second_report = json.loads(
        (
            tmp_path / "artifacts" / "reduce_sum" / "run_report.json"
        ).read_text(encoding="utf-8")
    )
    assert second_report["memory"]["loaded_item_count"] > 0
    assert second_report["memory"]["reused_memory_ids"]


def test_cpu_simd_pipeline_rejects_unsupported_generator_name(
    tmp_path, capsys
):
    exit_code = main(
        [
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "vector_add",
            "--generator",
            "unsupported-generator",
            "--work-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert "unsupported generator" in capsys.readouterr().err.lower()
