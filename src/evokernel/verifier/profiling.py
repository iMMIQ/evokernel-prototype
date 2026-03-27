from __future__ import annotations


def aggregate_latency_measurements(values: list[float]) -> float:
    if not values:
        raise ValueError("aggregate_latency_measurements requires at least one value")

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
