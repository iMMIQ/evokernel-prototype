from pathlib import Path

from evokernel.generator.base import GenerationRequest


PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
FALLBACK_SYSTEM_PROMPTS = {
    "drafting": (
        "You are drafting a candidate EvoKernel implementation.\n\n"
        "Produce a complete C/C++ kernel candidate that follows the backend "
        "constraints exactly.\n"
        "Prefer simple, correct code over speculative optimizations when "
        "context is incomplete.\n\n"
        "Output rules:\n"
        "- Output raw source code only — no markdown, no code fences, "
        "no explanation.\n"
        "- Do NOT wrap code in ```c, ```cpp, or any other fence markers.\n"
        "- Do NOT include any text before or after the code."
    ),
    "refining": (
        "You are refining an existing EvoKernel implementation.\n\n"
        "Use retrieved context and verifier feedback to correct failures or "
        "improve the candidate while preserving the task goal.\n"
        "Keep changes targeted to the reported issues unless the context "
        "requires a broader fix.\n\n"
        "Output rules:\n"
        "- Output raw source code only — no markdown, no code fences, "
        "no explanation.\n"
        "- Do NOT wrap code in ```c, ```cpp, or any other fence markers.\n"
        "- Do NOT include any text before or after the code."
    ),
}


def load_system_prompt(stage: str) -> str:
    prompt_path = PROMPTS_DIR / f"{stage}_system.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    try:
        return FALLBACK_SYSTEM_PROMPTS[stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt stage: {stage}") from exc


def build_generation_prompt(
    stage: str,
    task_summary: str,
    backend_constraints: list[str],
    retrieved_context: list[str],
    api_knowledge_context: list[str] | None = None,
    profiler_summary: str | None = None,
    feedback_summary: str | None = None,
) -> str:
    sections = [
        f"Stage: {stage}",
        "Task Summary:\n" + task_summary.strip(),
        _format_list("Backend Constraints", backend_constraints),
        _format_list("API Knowledge", api_knowledge_context or []),
        _format_list("Retrieved Context", retrieved_context),
    ]
    if profiler_summary:
        sections.append("Profiler Diagnosis:\n" + profiler_summary.strip())
    if feedback_summary:
        sections.append("Verifier Feedback:\n" + feedback_summary.strip())
    return "\n\n".join(section for section in sections if section)


def build_prompts(request: GenerationRequest) -> tuple[str, str]:
    return (
        load_system_prompt(request.stage),
        build_generation_prompt(
            stage=request.stage,
            task_summary=request.task_summary,
            backend_constraints=request.backend_constraints,
            retrieved_context=request.retrieved_context,
            api_knowledge_context=request.api_knowledge_context,
            profiler_summary=request.profiler_summary,
            feedback_summary=request.feedback_summary,
        ),
    )


def _format_list(title: str, items: list[str]) -> str:
    if not items:
        return f"{title}:\n- none"
    bullet_lines = "\n".join(f"- {item}" for item in items)
    return f"{title}:\n{bullet_lines}"
