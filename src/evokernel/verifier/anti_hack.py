from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class AntiHackResult:
    passed: bool
    error_category: str | None
    feedback_summary: str | None


_DISALLOWED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bimport\s+numpy\s+as\s+np\b|\bfrom\s+numpy\s+import\b"),
        "numpy imports are disallowed in candidate code",
    ),
    (
        re.compile(r"\bnp\.(?:add|sum|matmul|dot|mean|var|sqrt|linalg)\b"),
        "numpy shortcut operations are disallowed in candidate code",
    ),
    (
        re.compile(r"\bimport\s+(?:torch|jax|cupy)\b|\bfrom\s+(?:torch|jax|cupy)\s+import\b"),
        "external tensor library imports are disallowed in candidate code",
    ),
)


def check_for_disallowed_patterns(candidate_code: str) -> AntiHackResult:
    for pattern, message in _DISALLOWED_PATTERNS:
        match = pattern.search(candidate_code)
        if match is None:
            continue
        return AntiHackResult(
            passed=False,
            error_category="anti_hack",
            feedback_summary=f"{message}: `{match.group(0)}`",
        )
    return AntiHackResult(
        passed=True,
        error_category=None,
        feedback_summary=None,
    )
