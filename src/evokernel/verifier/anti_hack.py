from __future__ import annotations

import json
from dataclasses import dataclass
import re
from typing import Any

import httpx


@dataclass(slots=True)
class AntiHackResult:
    passed: bool
    error_category: str | None
    feedback_summary: str | None


_DISALLOWED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bimport\s+numpy(?:\.\w+)*(?:\s+as\s+\w+)?\b|"
            r"\bfrom\s+numpy(?:\.\w+)*\s+import\b"
        ),
        "numpy imports are disallowed in candidate code",
    ),
    (
        re.compile(
            r"\b(?:np|numpy)\.(?:add|sum|matmul|dot|mean|var|sqrt|linalg)\b"
        ),
        "numpy shortcut operations are disallowed in candidate code",
    ),
    (
        re.compile(
            r"\bimport\s+(?:torch|jax|cupy)(?:\.\w+)*(?:\s+as\s+\w+)?\b|"
            r"\bfrom\s+(?:torch|jax|cupy)(?:\.\w+)*\s+import\b"
        ),
        "external tensor library imports are disallowed in candidate code",
    ),
    (
        re.compile(
            r"\b__import__\s*\(|"
            r"\bimportlib\s*\.\s*import_module\s*\("
        ),
        "dynamic imports are disallowed in candidate code",
    ),
    (
        re.compile(
            r"\b(?:subprocess|os\.system|popen|exec|eval)\s*\("
        ),
        "shell/exec calls are disallowed in candidate code",
    ),
)

_INSPECTOR_SYSTEM_PROMPT = (
    "You are a code-security inspector for a kernel synthesis benchmark. "
    "The candidate code must implement the computation using only low-level "
    "C/C++ and SIMD intrinsics — no high-level libraries, no hidden delegation "
    "to external tools, no obfuscated shortcuts.\n\n"
    "Analyze the code below and respond with ONLY a JSON object:\n"
    '{"cheating": true/false, "reason": "brief explanation"}\n\n'
    "Flag as cheating if the code:\n"
    "- Uses hidden or obfuscated imports (e.g., dlopen, macro-wrapped includes)\n"
    "- Delegates computation to an external process or library at runtime\n"
    "- Uses string concatenation or macros to evade static detection\n"
    "- Hardcodes expected outputs or uses lookup tables instead of computing\n\n"
    "Do NOT flag legitimate SIMD intrinsics, standard C math headers, or "
    "manual loop unrolling."
)


def check_for_disallowed_patterns(
    candidate_code: str,
    *,
    inspector: dict[str, Any] | None = None,
) -> AntiHackResult:
    for pattern, message in _DISALLOWED_PATTERNS:
        match = pattern.search(candidate_code)
        if match is None:
            continue
        return AntiHackResult(
            passed=False,
            error_category="anti_hack",
            feedback_summary=f"{message}: `{match.group(0)}`",
        )

    if inspector is not None:
        return inspect_with_model(
            candidate_code=candidate_code,
            base_url=inspector["base_url"],
            api_key=inspector["api_key"],
            model=inspector["model"],
            timeout=inspector.get("timeout", 15.0),
        )

    return AntiHackResult(
        passed=True,
        error_category=None,
        feedback_summary=None,
    )


def inspect_with_model(
    candidate_code: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 15.0,
) -> AntiHackResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _INSPECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": candidate_code},
        ],
        "temperature": 0.0,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return AntiHackResult(
            passed=True,
            error_category=None,
            feedback_summary=f"inspector call failed (assumed safe): {exc}",
        )

    content = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return AntiHackResult(
            passed=True,
            error_category=None,
            feedback_summary="inspector returned empty response (assumed safe)",
        )

    try:
        result = json.loads(content.strip())
        cheating = result.get("cheating", False)
        reason = result.get("reason", "no reason provided")
    except json.JSONDecodeError:
        cheating = "true" in content.lower()
        reason = content.strip()

    if cheating:
        return AntiHackResult(
            passed=False,
            error_category="anti_hack_inspector",
            feedback_summary=f"model inspector flagged: {reason}",
        )

    return AntiHackResult(
        passed=True,
        error_category=None,
        feedback_summary=None,
    )
