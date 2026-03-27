from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from evokernel.cli import main


def test_cpu_simd_pipeline_runs_vector_add(
    tmp_path, deterministic_test_generator_override
):

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
    assert report["retrieval_policy"] == "value_driven"
    assert "context_role_ids" in report["attempts"][0]
    candidate_code = (
        tmp_path
        / "artifacts"
        / "vector_add"
        / "vector_add-1"
        / "candidate.cpp"
    ).read_text(encoding="utf-8")
    assert "fixture deterministic vector_add" in candidate_code


def test_cli_subprocess_supports_deterministic_test_generator_for_local_verification(
    tmp_path,
):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evokernel.cli",
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "vector_add",
            "--generator",
            "deterministic-test",
            "--work-root",
            str(tmp_path),
        ],
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        tmp_path / "artifacts" / "vector_add" / "run_report.json"
    ).exists()


def test_cpu_simd_pipeline_reuses_memory_across_two_tasks(
    tmp_path, deterministic_test_generator_override
):

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


def test_cpu_simd_pipeline_preserves_shared_memory_when_reuse_is_disabled(
    tmp_path, deterministic_test_generator_override
):
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
        ]
    )
    third_exit = main(
        [
            "--config",
            "configs/cpu_simd.toml",
            "--task",
            "vector_add",
            "--generator",
            "deterministic-test",
            "--work-root",
            str(tmp_path),
            "--reuse-memory",
        ]
    )

    assert first_exit == 0
    assert second_exit == 0
    assert third_exit == 0

    second_report = json.loads(
        (
            tmp_path / "artifacts" / "reduce_sum" / "run_report.json"
        ).read_text(encoding="utf-8")
    )
    third_report = json.loads(
        (
            tmp_path / "artifacts" / "vector_add" / "run_report.json"
        ).read_text(encoding="utf-8")
    )

    assert second_report["memory"]["loaded_item_count"] == 0
    assert third_report["memory"]["loaded_item_count"] >= 2


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
