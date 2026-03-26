from __future__ import annotations

import ctypes
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.ctypeslib import ndpointer

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
        output = self._execute_candidate_case(artifact=artifact, case=case)
        return ReferenceExecutionResult(case_path=case_path, output=output)

    def measure_latency(
        self,
        artifact: CandidateArtifact,
        case: dict[str, Any],
        warmup_runs: int,
        timed_runs: int,
    ) -> float:
        self._serialize_case(artifact=artifact, case=case)
        runner = self._build_candidate_runner(
            callable_obj=self.load_callable(artifact),
            task_id=artifact.task.task_id,
            case=case,
        )

        for _ in range(warmup_runs):
            runner()

        started_at = perf_counter()
        for _ in range(timed_runs):
            runner()
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
            "extern \"C\" const char* evokernel_task_id() {\n"
            f"    return \"{task.task_id}\";\n"
            "}\n"
        )

    def _execute_candidate_case(
        self, artifact: CandidateArtifact, case: dict[str, Any]
    ) -> Any:
        runner = self._build_candidate_runner(
            callable_obj=self.load_callable(artifact),
            task_id=artifact.task.task_id,
            case=case,
        )
        return runner()

    def _build_candidate_runner(
        self, callable_obj: Any, task_id: str, case: dict[str, Any]
    ) -> Any:
        entrypoint = getattr(callable_obj, "evokernel_entry", None)
        if entrypoint is None:
            raise AttributeError("Compiled artifact does not export evokernel_entry")

        if task_id == "vector_add":
            return self._build_vector_add_runner(entrypoint, case)
        if task_id == "reduce_sum":
            return self._build_reduce_sum_runner(entrypoint, case)
        if task_id == "matmul_tiled":
            return self._build_matmul_tiled_runner(entrypoint, case)
        if task_id == "layernorm":
            return self._build_layernorm_runner(entrypoint, case)
        raise NotImplementedError(f"Unsupported CPU SIMD task: {task_id}")

    def _build_vector_add_runner(
        self, entrypoint: Any, case: dict[str, Any]
    ) -> Any:
        a = np.ascontiguousarray(case["a"], dtype=np.float32)
        b = np.ascontiguousarray(case["b"], dtype=np.float32)
        out = np.empty_like(a)
        if hasattr(entrypoint, "argtypes"):
            entrypoint.argtypes = [
                ndpointer(dtype=np.float32),
                ndpointer(dtype=np.float32),
                ndpointer(dtype=np.float32),
                ctypes.c_size_t,
            ]
            entrypoint.restype = None

        def run() -> np.ndarray:
            entrypoint(out, a, b, a.size)
            return np.array(out, copy=True)

        return run

    def _build_reduce_sum_runner(
        self, entrypoint: Any, case: dict[str, Any]
    ) -> Any:
        x = np.ascontiguousarray(case["x"], dtype=np.float32)
        out = np.zeros(1, dtype=np.float32)
        if hasattr(entrypoint, "argtypes"):
            entrypoint.argtypes = [
                ndpointer(dtype=np.float32),
                ndpointer(dtype=np.float32),
                ctypes.c_size_t,
            ]
            entrypoint.restype = None

        def run() -> np.float32:
            out[0] = 0.0
            entrypoint(out, x, x.size)
            return np.float32(out[0])

        return run

    def _build_matmul_tiled_runner(
        self, entrypoint: Any, case: dict[str, Any]
    ) -> Any:
        a = np.ascontiguousarray(case["a"], dtype=np.float32)
        b = np.ascontiguousarray(case["b"], dtype=np.float32)
        out = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)
        if hasattr(entrypoint, "argtypes"):
            entrypoint.argtypes = [
                ndpointer(dtype=np.float32),
                ndpointer(dtype=np.float32),
                ndpointer(dtype=np.float32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            entrypoint.restype = None

        def run() -> np.ndarray:
            entrypoint(out, a, b, a.shape[0], a.shape[1], b.shape[1])
            return np.array(out, copy=True)

        return run

    def _build_layernorm_runner(
        self, entrypoint: Any, case: dict[str, Any]
    ) -> Any:
        x = np.ascontiguousarray(case["x"], dtype=np.float32)
        gamma = np.ascontiguousarray(case["gamma"], dtype=np.float32)
        beta = np.ascontiguousarray(case["beta"], dtype=np.float32)
        out = np.empty_like(x)
        eps = float(case["eps"])
        if hasattr(entrypoint, "argtypes"):
            entrypoint.argtypes = [
                ndpointer(dtype=np.float32),
                ndpointer(dtype=np.float32),
                ndpointer(dtype=np.float32),
                ndpointer(dtype=np.float32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_float,
            ]
            entrypoint.restype = None

        def run() -> np.ndarray:
            entrypoint(out, x, gamma, beta, x.shape[0], x.shape[1], eps)
            return np.array(out, copy=True)

        return run

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
