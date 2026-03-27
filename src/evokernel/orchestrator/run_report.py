from __future__ import annotations

from dataclasses import dataclass, field

from evokernel.domain.enums import Stage
from evokernel.domain.models import MemoryItem, VerificationOutcome


@dataclass(slots=True)
class AttemptRecord:
    attempt_id: str
    memory_id: str
    stage: Stage
    reward: float
    verifier_outcome: VerificationOutcome
    selected_context_ids: list[str] = field(default_factory=list)
    start_point_id: str | None = None


@dataclass(slots=True)
class RunReport:
    task_id: str
    backend_id: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    best_candidate: MemoryItem | None = None
