import os
import re
from dataclasses import dataclass

import httpx

from evokernel.config import GeneratorConfig
from evokernel.generator.base import GenerationRequest, GenerationResult
from evokernel.generator.prompt_builder import build_prompts


@dataclass(slots=True)
class OpenAICompatibleGenerator:
    model: str
    base_url: str
    api_key: str
    timeout: float = 120.0

    @classmethod
    def from_config(cls, config: GeneratorConfig) -> "OpenAICompatibleGenerator":
        api_key = config.api_key or os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key: set api_key in config or {config.api_key_env} env var"
            )
        base_url = config.base_url or "https://api.openai.com/v1"
        return cls(model=config.model, base_url=base_url, api_key=api_key)

    def build_payload(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        system_prompt, user_prompt = build_prompts(request)
        return self.generate_from_prompts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def generate_from_prompts(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        payload = self.build_payload(system_prompt=system_prompt, user_prompt=user_prompt)
        response_json = self._post_chat_completions(payload)
        return GenerationResult(
            code=self._extract_output_text(response_json),
            raw_response=response_json,
        )

    def _post_chat_completions(self, payload: dict) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def _extract_output_text(self, response_json: dict) -> str:
        choices = response_json.get("choices", [])
        if not choices:
            raise ValueError("No choices found in provider response")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            raise ValueError("No content found in provider response")
        code = content.strip()
        code = _strip_code_fences(code)
        return code


def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences if present."""
    match = re.match(r"^```[\w]*\n(.*?)```\s*$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Handle leading fence without closing
    match = re.match(r"^```[\w]*\n(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
