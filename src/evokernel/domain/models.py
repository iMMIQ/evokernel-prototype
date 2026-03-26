from pydantic import BaseModel, Field

from evokernel.domain.enums import Stage


class RetrievalConfig(BaseModel):
    final_context_count: int = 4
    over_retrieval_lambda: int = 3
    epsilon: float = 0.1
    alpha: float = 0.2


class GeneratorConfig(BaseModel):
    provider: str = "openai_compatible"
    model: str = "gpt-5.4"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


class RuntimeConfig(BaseModel):
    backend: str = "cpu_simd"
    artifact_dir: str = "artifacts"
    log_dir: str = "logs"


class BenchmarkConfig(BaseModel):
    tasks: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


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
