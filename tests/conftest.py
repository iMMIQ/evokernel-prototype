from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pytest

from evokernel.generator.base import GenerationRequest, GenerationResult


class _DeterministicTestGenerator:
    _VECTOR_ADD_CODE = """
#include <cstddef>

// fixture deterministic vector_add
extern "C" void evokernel_entry(
    float* out,
    const float* a,
    const float* b,
    std::size_t n
) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
""".strip()

    _REDUCE_SUM_CODE = """
#include <cstddef>

// fixture deterministic reduce_sum
extern "C" void evokernel_entry(
    float* out,
    const float* x,
    std::size_t n
) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    out[0] = sum;
}
""".strip()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        task_summary = request.task_summary.lower()
        if "add two float32 vectors" in task_summary:
            return GenerationResult(code=self._VECTOR_ADD_CODE)
        if "reduce a float32 vector to a scalar sum" in task_summary:
            return GenerationResult(code=self._REDUCE_SUM_CODE)
        raise ValueError(request.task_summary)


def install_deterministic_test_generator_override(
    overrides: MutableMapping[str, Any],
) -> None:
    overrides["deterministic-test"] = (
        lambda _config: _DeterministicTestGenerator()
    )


@pytest.fixture
def deterministic_test_generator_override(monkeypatch):
    import evokernel.cli as cli_module

    install_deterministic_test_generator_override(cli_module.GENERATOR_OVERRIDES)
    yield
    monkeypatch.delitem(
        cli_module.GENERATOR_OVERRIDES,
        "deterministic-test",
        raising=False,
    )
