from pathlib import Path

from evokernel.generator.prompt_builder import (
    build_generation_prompt,
    load_system_prompt,
)


def test_build_generation_prompt_includes_stage_constraints_and_feedback():
    prompt = build_generation_prompt(
        stage="drafting",
        task_summary="vector add on float32 arrays",
        backend_constraints=["emit a C entrypoint", "use SIMD intrinsics"],
        retrieved_context=["failure: missing include", "api: use __m256"],
        api_knowledge_context=["header: immintrin.h", "entrypoint: evokernel_entry"],
        feedback_summary="previous compile error: unknown intrinsic",
    )

    assert "vector add on float32 arrays" in prompt
    assert "previous compile error" in prompt
    assert "use SIMD intrinsics" in prompt
    assert "API Knowledge" in prompt
    assert "evokernel_entry" in prompt


def test_load_system_prompt_falls_back_when_prompt_file_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "evokernel.generator.prompt_builder.PROMPTS_DIR",
        Path("/tmp/evokernel-missing-prompts"),
    )

    prompt = load_system_prompt("drafting")

    assert "drafting" in prompt.lower()
    assert "return code only" in prompt.lower()
