from pathlib import Path
from typing import Any
import tomllib

from evokernel.domain.errors import ConfigLoadError
from evokernel.domain.models import AppConfig


def load_runtime_config(path: str | Path) -> AppConfig:
    config_path = Path(path)

    try:
        with config_path.open("rb") as handle:
            raw_config: dict[str, Any] = tomllib.load(handle)
    except OSError as exc:
        raise ConfigLoadError(f"Failed to read config file: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigLoadError(f"Failed to parse config file: {config_path}") from exc

    return AppConfig.model_validate(raw_config)
