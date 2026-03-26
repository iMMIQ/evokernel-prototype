from evokernel.generator.prompt_builder import build_generation_prompt


def test_build_generation_prompt_includes_stage_constraints_and_feedback():
    prompt = build_generation_prompt(
        stage="drafting",
        task_summary="vector add on float32 arrays",
        backend_constraints=["emit a C entrypoint", "use SIMD intrinsics"],
        retrieved_context=["failure: missing include", "api: use __m256"],
        feedback_summary="previous compile error: unknown intrinsic",
    )

    assert "vector add on float32 arrays" in prompt
    assert "previous compile error" in prompt
    assert "use SIMD intrinsics" in prompt
