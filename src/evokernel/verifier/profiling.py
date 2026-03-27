from __future__ import annotations


def aggregate_latency_measurements(values: list[float]) -> float:
    if not values:
        raise ValueError("aggregate_latency_measurements requires at least one value")

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def diagnose_performance(
    *,
    task,
    latency_ms: float,
) -> tuple[str | None, str | None, float | None]:
    target_latency = task.baseline_data.get("target_latency_ms")
    latency_ratio = None
    if isinstance(target_latency, (int, float)) and target_latency > 0:
        latency_ratio = latency_ms / float(target_latency)

    bottleneck = _infer_bottleneck(
        operator_family=task.operator_family,
        latency_ratio=latency_ratio,
    )
    summary = _build_profiler_summary(
        bottleneck=bottleneck,
        latency_ms=latency_ms,
        target_latency_ms=(
            float(target_latency)
            if isinstance(target_latency, (int, float))
            else None
        ),
        latency_ratio=latency_ratio,
    )
    return bottleneck, summary, latency_ratio


def _infer_bottleneck(
    *,
    operator_family: str,
    latency_ratio: float | None,
) -> str:
    if operator_family == "elementwise":
        if latency_ratio is not None and latency_ratio > 1.5:
            return "vectorization_gap"
        return "memory_bandwidth"
    if operator_family == "reduction":
        if latency_ratio is not None and latency_ratio > 1.5:
            return "reduction_folding"
        return "vectorization_gap"
    if operator_family == "matmul":
        if latency_ratio is not None and latency_ratio > 1.5:
            return "cache_locality"
        return "register_reuse"
    if operator_family == "normalization":
        return "normalization_stats"
    return "general_latency"


def _build_profiler_summary(
    *,
    bottleneck: str,
    latency_ms: float,
    target_latency_ms: float | None,
    latency_ratio: float | None,
) -> str:
    parts = [
        f"Profiler diagnosis: likely bottleneck={bottleneck}.",
        f"measured_latency_ms={latency_ms:.3f}.",
    ]
    if target_latency_ms is not None:
        parts.append(f"target_latency_ms={target_latency_ms:.3f}.")
    if latency_ratio is not None:
        parts.append(f"latency_ratio_to_target={latency_ratio:.2f}.")
    return " ".join(parts)
