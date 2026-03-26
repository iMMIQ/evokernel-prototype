from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class BenchmarkTolerances:
    atol: float
    rtol: float


@dataclass(slots=True)
class BenchmarkTask:
    task_id: str
    operator_family: str
    summary: str
    reference_impl: Callable[..., Any]
    randomized_inputs: list[dict[str, Any]] = field(default_factory=list)
    edge_case_inputs: list[dict[str, Any]] = field(default_factory=list)
    tolerances: BenchmarkTolerances = field(
        default_factory=lambda: BenchmarkTolerances(atol=1e-6, rtol=1e-6)
    )
    prompt_metadata: dict[str, Any] = field(default_factory=dict)
    baseline_data: dict[str, Any] = field(default_factory=dict)

