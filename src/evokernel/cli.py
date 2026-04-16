from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from evokernel.backend.cpu_simd import CpuSimdBackend
from evokernel.config import AppConfig, load_runtime_config
from evokernel.generator.openai_compatible import OpenAICompatibleGenerator
from evokernel.memory.embedding import build_text_embedder
from evokernel.memory.seeds import ingest_seed_memory
from evokernel.memory.store import InMemoryStore
from evokernel.orchestrator.curriculum import run_curriculum
from evokernel.orchestrator.episode import run_episode
from evokernel.retrieval.q_store import QValueStore

GENERATOR_OVERRIDES: dict[str, Callable[[AppConfig], object] | object] = {}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    runtime = None

    try:
        config = load_runtime_config(args.config)
        task_ids = _resolve_task_ids(args, config)

        if len(task_ids) == 1:
            runtime, work_root = _build_runtime(args, config, task_ids[0])
            _apply_memory_transfer(runtime, config)
            report = run_episode(runtime, task_id=task_ids[0])
            artifact_dir = work_root / config.runtime.artifact_dir / task_ids[0]
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_run_report(artifact_dir=artifact_dir, report=report, runtime=runtime)
            return 0 if report.best_candidate is not None else 1

        runtime, work_root = _build_runtime(args, config, None)
        _apply_memory_transfer(runtime, config)
        curriculum_report = run_curriculum(runtime, task_ids=task_ids)
        for task_id, ep_report in curriculum_report.task_reports:
            task_artifact = work_root / config.runtime.artifact_dir / task_id
            task_artifact.mkdir(parents=True, exist_ok=True)
            _write_run_report(artifact_dir=task_artifact, report=ep_report, runtime=runtime)
        _write_curriculum_report(
            work_root=work_root,
            config=config,
            report=curriculum_report,
        )
        return 0 if curriculum_report.solved_count > 0 else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            runtime.memory_store.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evokernel")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--generator")
    parser.add_argument("--work-root")
    parser.add_argument("--reuse-memory", action="store_true")
    return parser


def _resolve_task_ids(args: argparse.Namespace, config: AppConfig) -> list[str]:
    if args.task:
        return [args.task]
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",") if t.strip()]
    if config.benchmark.tasks:
        return list(config.benchmark.tasks)
    raise ValueError("specify --task, --tasks, or set [benchmark] tasks in config")


def _apply_memory_transfer(runtime, config: AppConfig) -> None:
    curriculum_config = config.curriculum
    if curriculum_config.source_memory_path is None:
        return
    count = runtime.memory_store.import_from_store(
        curriculum_config.source_memory_path,
        exclude_task_ids=curriculum_config.exclude_task_ids,
    )
    print(f"imported {count} memory items from {curriculum_config.source_memory_path}")


def _build_runtime(
    args: argparse.Namespace,
    config: AppConfig,
    task_id: str | None,
) -> tuple[SimpleNamespace, Path]:
    work_root = (
        Path(args.work_root).resolve()
        if args.work_root is not None
        else (Path.cwd() / ".evokernel").resolve()
    )
    task_dir = work_root / config.runtime.artifact_dir / (task_id or "_curriculum")
    task_dir.mkdir(parents=True, exist_ok=True)
    memory_path = work_root / "shared_memory.sqlite3"
    embedder = build_text_embedder(config.embedding)
    memory_store = InMemoryStore(
        memory_path,
        embedder=embedder,
        reuse_existing=args.reuse_memory,
    )
    backend = CpuSimdBackend(work_root=task_dir)
    ingest_seed_memory(
        memory_store,
        backend_id=config.runtime.backend,
        backend_constraints=backend.prompt_constraints(),
    )

    runtime = SimpleNamespace(
        backend=backend,
        backend_id=config.runtime.backend,
        backend_constraints=backend.prompt_constraints(),
        generator=_build_generator(args.generator, config),
        inspector=_build_inspector(config),
        embedder=embedder,
        memory_store=memory_store,
        q_store=QValueStore(connection=memory_store.connection),
        config=config,
        loaded_memory_ids=memory_store.loaded_memory_ids,
    )
    return runtime, work_root


def _build_generator(
    generator_name: str | None,
    config: AppConfig,
):
    resolved_generator = generator_name or config.generator.provider

    if resolved_generator == "deterministic-test":
        _load_dev_generator_override()
        try:
            override = GENERATOR_OVERRIDES[resolved_generator]
        except KeyError as exc:
            raise ValueError(
                "generator override required for deterministic-test"
            ) from exc
        return override(config) if callable(override) else override
    if resolved_generator == "openai_compatible":
        return OpenAICompatibleGenerator.from_config(config.generator)
    raise ValueError(f"Unsupported generator: {resolved_generator}")


def _build_inspector(config: AppConfig) -> dict | None:
    verifier_config = config.verifier
    if not verifier_config.inspector_enabled:
        return None
    api_key = config.generator.api_key or None
    base_url = config.generator.base_url or "https://api.openai.com/v1"
    model = verifier_config.inspector_model or config.generator.model
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "timeout": verifier_config.inspector_timeout,
    }


def _load_dev_generator_override() -> None:
    if "deterministic-test" in GENERATOR_OVERRIDES:
        return

    conftest_path = Path(__file__).resolve().parents[2] / "tests" / "conftest.py"
    if not conftest_path.is_file():
        return

    spec = importlib.util.spec_from_file_location(
        "_evokernel_tests_conftest",
        conftest_path,
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return

    installer = getattr(module, "install_deterministic_test_generator_override", None)
    if installer is None:
        return
    installer(GENERATOR_OVERRIDES)


def _write_run_report(*, artifact_dir: Path, report, runtime) -> None:
    reused_memory_ids = _collect_reused_memory_ids(
        report=report,
        loaded_memory_ids=set(runtime.loaded_memory_ids),
    )
    payload = {
        "task_id": report.task_id,
        "backend_id": report.backend_id,
        "retrieval_policy": runtime.config.retrieval.policy,
        "attempts": [
            {
                "attempt_id": attempt.attempt_id,
                "memory_id": attempt.memory_id,
                "stage": attempt.stage.value,
                "reward": attempt.reward,
                "verifier_outcome": attempt.verifier_outcome.model_dump(mode="json"),
                "selected_context_ids": list(attempt.selected_context_ids),
                "context_role_ids": {
                    key: list(value)
                    for key, value in attempt.context_role_ids.items()
                },
                "start_point_id": attempt.start_point_id,
            }
            for attempt in report.attempts
        ],
        "best_candidate": (
            report.best_candidate.model_dump(mode="json")
            if report.best_candidate is not None
            else None
        ),
        "memory": {
            "loaded_item_count": len(runtime.loaded_memory_ids),
            "reused_memory_ids": reused_memory_ids,
        },
    }
    (artifact_dir / "run_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_curriculum_report(
    *, work_root: Path, config: AppConfig, report
) -> None:
    payload = {
        "strategy": report.strategy,
        "total_tasks": report.total_count,
        "solved_tasks": report.solved_count,
        "task_ids": report.task_ids,
        "per_task": {
            task_id: {
                "solved": ep_report.best_candidate is not None,
                "attempts": len(ep_report.attempts),
            }
            for task_id, ep_report in report.task_reports
        },
    }
    artifact_dir = work_root / config.runtime.artifact_dir / "_curriculum"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "curriculum_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_reused_memory_ids(*, report, loaded_memory_ids: set[str]) -> list[str]:
    reused_memory_ids: list[str] = []
    seen: set[str] = set()
    for attempt in report.attempts:
        candidate_ids = [*attempt.selected_context_ids]
        if attempt.start_point_id is not None:
            candidate_ids.append(attempt.start_point_id)
        for memory_id in candidate_ids:
            if memory_id not in loaded_memory_ids or memory_id in seen:
                continue
            seen.add(memory_id)
            reused_memory_ids.append(memory_id)
    return reused_memory_ids


if __name__ == "__main__":
    raise SystemExit(main())
