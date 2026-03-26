from __future__ import annotations

import ctypes
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from evokernel.backend.base import (
    CandidateArtifact,
    CompilationResult,
    ReferenceExecutionResult,
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
        compiler_info_path = work_dir / "toolchain.json"
        case_dir = work_dir / "cases"
        case_dir.mkdir(exist_ok=True)

        source_path.write_text(candidate_code.rstrip() + "\n", encoding="utf-8")
        harness_path.write_text(
            self._build_harness(task=task),
            encoding="utf-8",
        )
        compiler_info_path.write_text(
            json.dumps(
                {
                    "compiler": self.toolchain.compiler.executable,
                    "language": self.toolchain.compiler.language,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return CandidateArtifact(
            attempt_id=attempt_id,
            work_dir=work_dir,
            source_path=source_path,
            harness_path=harness_path,
            binary_path=binary_path,
            compiler_info_path=compiler_info_path,
            case_dir=case_dir,
            last_case_path=None,
            task=task,
        )

    def compile(self, artifact: CandidateArtifact) -> CompilationResult:
        result = self.toolchain.compile(artifact)
        compiler_info = json.loads(
            artifact.compiler_info_path.read_text(encoding="utf-8")
        )
        compiler_info["build_command"] = result.command
        artifact.compiler_info_path.write_text(
            json.dumps(compiler_info, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    def load_callable(self, artifact: CandidateArtifact) -> Any:
        if not artifact.binary_path.exists():
            raise FileNotFoundError(
                f"Compiled artifact is missing: {artifact.binary_path}"
            )
        return ctypes.CDLL(str(artifact.binary_path))

    def run_reference_case(
        self, artifact: CandidateArtifact, case: dict[str, Any]
    ) -> ReferenceExecutionResult:
        case_path = self._serialize_case(artifact=artifact, case=case)
        output = artifact.task.reference_impl(**case)
        return ReferenceExecutionResult(case_path=case_path, output=output)

    def measure_latency(
        self,
        artifact: CandidateArtifact,
        case: dict[str, Any],
        warmup_runs: int,
        timed_runs: int,
    ) -> float:
        case_path = self._serialize_case(artifact=artifact, case=case)
        callable_obj = self.load_callable(artifact)
        entrypoint = getattr(callable_obj, "evokernel_run_case", None)
        if entrypoint is None:
            raise AttributeError(
                "Compiled artifact does not export evokernel_run_case"
            )
        if hasattr(entrypoint, "argtypes"):
            entrypoint.argtypes = [ctypes.c_char_p]
        if hasattr(entrypoint, "restype"):
            entrypoint.restype = ctypes.c_int
        case_bytes = str(case_path).encode("utf-8")

        for _ in range(warmup_runs):
            entrypoint(case_bytes)

        started_at = perf_counter()
        for _ in range(timed_runs):
            entrypoint(case_bytes)
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
            "#include <cstdio>\n"
            "extern \"C\" void evokernel_entry();\n"
            "extern \"C\" int evokernel_run_case(const char* case_path) {\n"
            "    if (case_path == nullptr) {\n"
            "        return -1;\n"
            "    }\n"
            "    std::FILE* handle = std::fopen(case_path, \"rb\");\n"
            "    if (handle == nullptr) {\n"
            "        return -2;\n"
            "    }\n"
            "    int byte_count = 0;\n"
            "    while (std::fgetc(handle) != EOF) {\n"
            "        ++byte_count;\n"
            "    }\n"
            "    std::fclose(handle);\n"
            "    evokernel_entry();\n"
            "    return byte_count;\n"
            "}\n"
        )

    def _serialize_case(
        self, artifact: CandidateArtifact, case: dict[str, Any]
    ) -> Path:
        case_index = len(list(artifact.case_dir.glob("case-*.json")))
        case_path = artifact.case_dir / f"case-{case_index:04d}.json"
        payload = {
            "task_id": artifact.task.task_id,
            "attempt_id": artifact.attempt_id,
            "inputs": self._normalize_for_json(case),
        }
        case_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact.last_case_path = case_path
        return case_path

    def _normalize_for_json(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                key: self._normalize_for_json(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_for_json(item) for item in value]
        return value
