from pydantic import BaseModel, Field

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
    error_category: str | None
    feedback_summary: str | None

    @property
    def is_feasible(self) -> bool:
        return (
            self.anti_hack_passed
            and self.compile_passed
            and self.correctness_passed
        )
