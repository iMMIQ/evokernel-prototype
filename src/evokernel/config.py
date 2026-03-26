from pathlib import Path
from typing import Any
import tomllib

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evokernel.domain.errors import ConfigLoadError


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalConfig(ConfigModel):
    final_context_count: int = 4
    over_retrieval_lambda: int = 3
    epsilon: float = 0.1
    alpha: float = 0.2


class GeneratorConfig(ConfigModel):
    provider: str = "openai_compatible"
    model: str = "gpt-5.4"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


class RuntimeConfig(ConfigModel):
    backend: str = "cpu_simd"
    artifact_dir: str = "artifacts"
    log_dir: str = "logs"


class BenchmarkConfig(ConfigModel):
    tasks: list[str] = Field(default_factory=list)


class AppConfig(ConfigModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


def load_runtime_config(path: str | Path) -> AppConfig:
    config_path = Path(path)

    try:
        with config_path.open("rb") as handle:
            raw_config: dict[str, Any] = tomllib.load(handle)
    except OSError as exc:
        raise ConfigLoadError(f"Failed to read config file: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigLoadError(f"Failed to parse config file: {config_path}") from exc

    try:
        return AppConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ConfigLoadError(f"Failed to validate config file: {config_path}") from exc
