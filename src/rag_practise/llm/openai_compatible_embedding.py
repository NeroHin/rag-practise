from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from openai import OpenAI


class EmbeddingClient(Protocol):
    model_id: str

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...


class OpenAICompatibleEmbeddingClient:
    """Embedding client for OpenAI-compatible embedding APIs such as local OMLX."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        base_url: str,
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = client or OpenAI(api_key=api_key, base_url=base_url)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model_id, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)
