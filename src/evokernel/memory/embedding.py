from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from evokernel.config import EmbeddingConfig


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(slots=True)
class HashingTextEmbedder:
    dimensions: int = 256

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        if not tokens:
            tokens = ["<empty>"]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.astype(float).tolist()


@dataclass(slots=True)
class OpenAICompatibleTextEmbedder:
    model: str
    base_url: str
    api_key: str
    dimensions: int | None = None
    timeout: float = 30.0

    @classmethod
    def from_config(
        cls,
        config: EmbeddingConfig,
    ) -> "OpenAICompatibleTextEmbedder":
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key environment variable: {config.api_key_env}"
            )
        base_url = config.base_url or "https://api.openai.com/v1"
        dimensions = config.dimensions if config.dimensions > 0 else None
        return cls(
            model=config.model,
            base_url=base_url,
            api_key=api_key,
            dimensions=dimensions,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, object] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            response_json = response.json()

        embeddings = [
            record.get("embedding")
            for record in response_json.get("data", [])
            if isinstance(record.get("embedding"), list)
        ]
        if len(embeddings) != len(texts):
            raise ValueError("Embedding provider returned incomplete embeddings")
        return [
            [float(component) for component in embedding]
            for embedding in embeddings
        ]


def build_text_embedder(config: EmbeddingConfig) -> TextEmbedder:
    if config.provider == "hashing":
        return HashingTextEmbedder(dimensions=config.dimensions)
    if config.provider == "openai_compatible":
        return OpenAICompatibleTextEmbedder.from_config(config)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
