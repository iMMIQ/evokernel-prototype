from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class GenerationRequest:
    stage: str
    task_summary: str
    backend_constraints: list[str] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
    api_knowledge_context: list[str] = field(default_factory=list)
    profiler_summary: str | None = None
    feedback_summary: str | None = None


@dataclass(slots=True)
class GenerationResult:
    code: str
    raw_response: dict | None = None


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
