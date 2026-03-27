from __future__ import annotations

import json

from evokernel.cli import main


def test_cpu_simd_pipeline_runs_vector_add(tmp_path, monkeypatch):
    _ = monkeypatch

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


def test_cpu_simd_pipeline_reuses_memory_across_two_tasks(tmp_path, monkeypatch):
    _ = monkeypatch

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
