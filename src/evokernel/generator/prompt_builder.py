from pathlib import Path

from evokernel.generator.base import GenerationRequest


PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def load_system_prompt(stage: str) -> str:
    prompt_path = PROMPTS_DIR / f"{stage}_system.md"
    return prompt_path.read_text(encoding="utf-8").strip()


def build_generation_prompt(
    stage: str,
    task_summary: str,
    backend_constraints: list[str],
    retrieved_context: list[str],
    feedback_summary: str | None = None,
) -> str:
    sections = [
        f"Stage: {stage}",
        "Task Summary:\n" + task_summary.strip(),
        _format_list("Backend Constraints", backend_constraints),
        _format_list("Retrieved Context", retrieved_context),
    ]
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
            feedback_summary=request.feedback_summary,
        ),
    )


def _format_list(title: str, items: list[str]) -> str:
    if not items:
        return f"{title}:\n- none"
    bullet_lines = "\n".join(f"- {item}" for item in items)
    return f"{title}:\n{bullet_lines}"
