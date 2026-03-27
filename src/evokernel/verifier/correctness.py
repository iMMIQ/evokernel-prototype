from __future__ import annotations

from typing import Any

import numpy as np


def compare_outputs(
    actual: Any,
    expected: Any,
    atol: float,
    rtol: float,
) -> tuple[bool, str | None]:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)

    if actual_array.shape != expected_array.shape:
        return (
            False,
            "shape mismatch: "
            f"expected {expected_array.shape}, got {actual_array.shape}",
        )

    if np.allclose(actual_array, expected_array, atol=atol, rtol=rtol, equal_nan=True):
        return True, None

    mismatch_mask = ~np.isclose(
        actual_array,
        expected_array,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
    first_index = tuple(int(index) for index in np.argwhere(mismatch_mask)[0])
    actual_value = actual_array[first_index].item()
    expected_value = expected_array[first_index].item()
    absolute_diff = np.abs(actual_array - expected_array)
    max_diff = float(np.max(absolute_diff))
    return (
        False,
        "value mismatch at "
        f"{first_index}: expected {expected_value}, got {actual_value} "
        f"(max abs diff {max_diff:.3e})",
    )
