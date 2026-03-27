from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from evokernel.backend.cpu_simd import CpuSimdBackend
from evokernel.config import AppConfig, load_runtime_config
from evokernel.generator.openai_compatible import OpenAICompatibleGenerator
from evokernel.memory.store import InMemoryStore
from evokernel.orchestrator.episode import run_episode
from evokernel.retrieval.q_store import QValueStore

GENERATOR_OVERRIDES: dict[str, Callable[[AppConfig], object] | object] = {}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_runtime_config(args.config)
        runtime, artifact_dir, memory_path = _build_runtime(args, config)
        report = run_episode(runtime, task_id=args.task)
        runtime.memory_store.save_jsonl(memory_path)
        _write_run_report(artifact_dir=artifact_dir, report=report, runtime=runtime)
        return 0 if report.best_candidate is not None else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evokernel")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--generator")
    parser.add_argument("--work-root")
    parser.add_argument("--reuse-memory", action="store_true")
    return parser


def _build_runtime(
    args: argparse.Namespace,
    config: AppConfig,
) -> tuple[SimpleNamespace, Path, Path]:
    work_root = (
        Path(args.work_root).resolve()
        if args.work_root is not None
        else (Path.cwd() / ".evokernel").resolve()
    )
    artifact_dir = work_root / config.runtime.artifact_dir / args.task
    artifact_dir.mkdir(parents=True, exist_ok=True)
    memory_path = work_root / "shared_memory.jsonl"

    memory_store = (
        InMemoryStore.load_jsonl(memory_path)
        if args.reuse_memory
        else InMemoryStore()
    )
    loaded_memory_items = list(getattr(memory_store, "_items", []))
    backend = CpuSimdBackend(work_root=artifact_dir)

    runtime = SimpleNamespace(
        backend=backend,
        backend_id=config.runtime.backend,
        backend_constraints=backend.prompt_constraints(),
        generator=_build_generator(args.generator, config),
        memory_store=memory_store,
        q_store=QValueStore(),
        config=config,
        loaded_memory_items=loaded_memory_items,
        loaded_memory_ids=[item.memory_id for item in loaded_memory_items],
    )
    return runtime, artifact_dir, memory_path


def _build_generator(
    generator_name: str | None,
    config: AppConfig,
):
    if generator_name == "deterministic-test":
        try:
            override = GENERATOR_OVERRIDES[generator_name]
        except KeyError as exc:
            raise ValueError(
                "generator override required for deterministic-test"
            ) from exc
        return override(config) if callable(override) else override
    return OpenAICompatibleGenerator.from_config(config.generator)


def _write_run_report(*, artifact_dir: Path, report, runtime) -> None:
    reused_memory_ids = _collect_reused_memory_ids(
        report=report,
        loaded_memory_ids=set(runtime.loaded_memory_ids),
    )
    payload = {
        "task_id": report.task_id,
        "backend_id": report.backend_id,
        "attempts": [
            {
                "attempt_id": attempt.attempt_id,
                "memory_id": attempt.memory_id,
                "stage": attempt.stage.value,
                "reward": attempt.reward,
                "verifier_outcome": attempt.verifier_outcome.model_dump(mode="json"),
                "selected_context_ids": list(attempt.selected_context_ids),
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
