import os
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
    timeout: float = 30.0

    @classmethod
    def from_config(cls, config: GeneratorConfig) -> "OpenAICompatibleGenerator":
        api_key = os.environ[config.api_key_env]
        base_url = config.base_url or "https://api.openai.com/v1"
        return cls(model=config.model, base_url=base_url, api_key=api_key)

    def build_payload(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
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
        response_json = self._post_responses(payload)
        return GenerationResult(
            code=self._extract_output_text(response_json),
            raw_response=response_json,
        )

    def _post_responses(self, payload: dict) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def _extract_output_text(self, response_json: dict) -> str:
        texts: list[str] = []
        for output_item in response_json.get("output", []):
            for content_item in output_item.get("content", []):
                if content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if isinstance(text, str):
                        texts.append(text)
        return "\n".join(texts).strip()
