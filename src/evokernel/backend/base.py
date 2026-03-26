from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class CandidateArtifact:
    attempt_id: str
    work_dir: Path
    source_path: Path
    harness_path: Path
    binary_path: Path
    task: Any


@dataclass(slots=True)
class CompilationResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    binary_path: Path


@dataclass(slots=True)
class StructuredBackendError:
    category: str
    message: str


class Backend(Protocol):
    def prompt_constraints(self) -> list[str]:
        ...

    def materialize_candidate(
        self, task: Any, candidate_code: str, attempt_id: str
    ) -> CandidateArtifact:
        ...

    def compile(self, artifact: CandidateArtifact) -> CompilationResult:
        ...

    def load_callable(self, artifact: CandidateArtifact) -> Any:
        ...

    def run_reference_case(
        self, artifact: CandidateArtifact, case: dict[str, Any]
    ) -> Any:
        ...

    def measure_latency(
        self,
        artifact: CandidateArtifact,
        case: dict[str, Any],
        warmup_runs: int,
        timed_runs: int,
    ) -> float:
        ...

    def extract_structured_error(
        self, stderr: str
    ) -> StructuredBackendError | None:
        ...
