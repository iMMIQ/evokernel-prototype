from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which
import subprocess

from evokernel.backend.base import CandidateArtifact, CompilationResult


@dataclass(slots=True)
class CompilerSpec:
    executable: str
    language: str = "c++"


class CpuSimdToolchain:
    def __init__(self, compiler: CompilerSpec | None = None) -> None:
        self._compiler = compiler or self._detect_compiler()

    @property
    def compiler(self) -> CompilerSpec:
        return self._compiler

    def build_command(self, artifact: CandidateArtifact) -> list[str]:
        return [
            self.compiler.executable,
            "-shared",
            "-fPIC",
            "-O3",
            "-std=c++17",
            str(artifact.source_path),
            "-o",
            str(artifact.binary_path),
        ]

    def compile(self, artifact: CandidateArtifact) -> CompilationResult:
        command = self.build_command(artifact)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        return CompilationResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            binary_path=artifact.binary_path,
        )

    def _detect_compiler(self) -> CompilerSpec:
        for executable in ("clang++", "clang", "g++", "gcc"):
            if which(executable):
                return CompilerSpec(executable=executable)
        raise RuntimeError(
            "No supported CPU SIMD compiler found; expected clang or gcc."
        )
