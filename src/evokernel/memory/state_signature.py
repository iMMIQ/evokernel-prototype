from evokernel.domain.enums import Stage


def build_state_signature(
    backend_id: str,
    operator_family: str,
    stage: Stage,
    shape_bucket: str,
    error_category: str | None,
) -> str:
    normalized_error = error_category or "none"
    return "|".join(
        (
            backend_id,
            operator_family,
            stage.value,
            shape_bucket,
            normalized_error,
        )
    )
