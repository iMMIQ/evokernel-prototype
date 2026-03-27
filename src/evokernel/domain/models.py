from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from evokernel.domain.enums import Stage


class EpisodeState(BaseModel):
    task_id: str
    stage: Stage
    remaining_budget: int
    start_points: list[str] = Field(default_factory=list)

    @classmethod
    def initial(cls, task_id: str, budget: int) -> "EpisodeState":
        return cls(
            task_id=task_id,
            stage=Stage.DRAFTING,
            remaining_budget=budget,
            start_points=[],
        )


class VerificationOutcome(BaseModel):
    anti_hack_passed: bool
    compile_passed: bool
    correctness_passed: bool
    latency_ms: float | None
    bottleneck_label: str | None = None
    profiler_summary: str | None = None
    latency_ratio_to_target: float | None = None
    error_category: str | None
    feedback_summary: str | None

    @property
    def is_feasible(self) -> bool:
        return (
            self.anti_hack_passed
            and self.compile_passed
            and self.correctness_passed
        )


MemoryKind = Literal[
    "backend_knowledge",
    "failure_summary",
    "success_summary",
    "generation_trace",
    "refinement_hint",
]


class MemoryItem(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    backend_id: str
    operator_family: str
    stage: Stage
    code: str
    summary: str
    context_summary: str | None = None
    memory_kind: MemoryKind = "generation_trace"
    reward: float
    is_feasible: bool
    became_start_point: bool
    verifier_outcome: VerificationOutcome
    parent_attempt_id: str | None = None
    parent_memory_id: str | None = None
    retrieval_text: str | None = None
    embedding: list[float] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def validate_feasibility(self) -> "MemoryItem":
        if self.is_feasible != self.verifier_outcome.is_feasible:
            raise ValueError(
                "is_feasible must match verifier_outcome.is_feasible"
            )
        if self.became_start_point and not self.is_feasible:
            raise ValueError(
                "became_start_point requires is_feasible=True"
            )
        return self
