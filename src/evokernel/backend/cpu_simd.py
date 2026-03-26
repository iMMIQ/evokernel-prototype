from __future__ import annotations

import ctypes
from pathlib import Path
from time import perf_counter
from typing import Any

from evokernel.backend.base import (
    CandidateArtifact,
    CompilationResult,
    StructuredBackendError,
)
from evokernel.backend.toolchain import CpuSimdToolchain


class CpuSimdBackend:
    def __init__(
        self,
        work_root: Path | None = None,
        toolchain: CpuSimdToolchain | None = None,
    ) -> None:
        self.work_root = work_root or Path.cwd() / ".evokernel" / "artifacts"
        self.toolchain = toolchain or CpuSimdToolchain()

    def prompt_constraints(self) -> list[str]:
        return [
            "Generate C or C++ kernel code for a CPU SIMD backend.",
            "Expose an entrypoint named evokernel_entry.",
            "Do not emit build scripts, shell commands, or prose.",
        ]

    def materialize_candidate(
        self, task: Any, candidate_code: str, attempt_id: str
    ) -> CandidateArtifact:
        work_dir = self.work_root / attempt_id
        work_dir.mkdir(parents=True, exist_ok=True)

        source_path = work_dir / "candidate.cpp"
        harness_path = work_dir / "harness.cpp"
        binary_path = work_dir / "candidate.so"

        source_path.write_text(candidate_code.rstrip() + "\n", encoding="utf-8")
        harness_path.write_text(
            self._build_harness(task=task),
            encoding="utf-8",
        )

        return CandidateArtifact(
            attempt_id=attempt_id,
            work_dir=work_dir,
            source_path=source_path,
            harness_path=harness_path,
            binary_path=binary_path,
            task=task,
        )

    def compile(self, artifact: CandidateArtifact) -> CompilationResult:
        return self.toolchain.compile(artifact)

    def load_callable(self, artifact: CandidateArtifact) -> Any:
        if not artifact.binary_path.exists():
            raise FileNotFoundError(
                f"Compiled artifact is missing: {artifact.binary_path}"
            )
        return ctypes.CDLL(str(artifact.binary_path))

    def run_reference_case(
        self, artifact: CandidateArtifact, case: dict[str, Any]
    ) -> Any:
        return artifact.task.reference_impl(**case)

    def measure_latency(
        self,
        artifact: CandidateArtifact,
        case: dict[str, Any],
        warmup_runs: int,
        timed_runs: int,
    ) -> float:
        callable_obj = self.load_callable(artifact)
        entrypoint = getattr(callable_obj, "evokernel_entry", None)
        if entrypoint is None:
            raise AttributeError("Compiled artifact does not export evokernel_entry")

        for _ in range(warmup_runs):
            entrypoint()

        started_at = perf_counter()
        for _ in range(timed_runs):
            entrypoint()
        elapsed = perf_counter() - started_at
        return elapsed * 1000.0 / max(timed_runs, 1)

    def extract_structured_error(
        self, stderr: str
    ) -> StructuredBackendError | None:
        normalized = stderr.strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        if "error:" in lowered:
            category = "compile_error"
        elif "undefined reference" in lowered:
            category = "link_error"
        else:
            category = "runtime_error"
        return StructuredBackendError(category=category, message=normalized)

    def _build_harness(self, task: Any) -> str:
        return (
            "// Auto-generated harness stub for EvoKernel CPU SIMD tasks.\n"
            f"// Task: {task.task_id}\n"
            "extern \"C\" void evokernel_entry();\n"
        )
