from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np
from pydantic import BaseModel, Field

from rag_practise.datasets import EvidenceDocument


class EmbeddingProvider(Protocol):
    model_id: str

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...


class HashEmbeddingClient:
    """Deterministic local embedding for tests and mock benchmark runs."""

    def __init__(self, model_id: str = "hash-embedding", dimensions: int = 256) -> None:
        self.model_id = model_id
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in _tokens(text):
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                rows[row_index, bucket] += 1.0
        return rows


class SearchHit(BaseModel):
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class FaissVectorIndex:
    def __init__(
        self,
        *,
        documents: list[EvidenceDocument],
        index: Any,
        metadata: dict[str, Any],
        normalize: bool = True,
    ) -> None:
        self.documents = documents
        self.index = index
        self.metadata = metadata
        self.normalize = normalize

    @classmethod
    def build(
        cls,
        documents: list[EvidenceDocument],
        embedding_client: EmbeddingProvider,
        *,
        normalize: bool = True,
        provider_metadata: dict[str, Any] | None = None,
    ) -> "FaissVectorIndex":
        vectors = embedding_client.embed_texts([document.text for document in documents])
        vectors = _as_float32(vectors)
        if normalize:
            faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        metadata = {
            "embedding_model_id": embedding_client.model_id,
            "dimension": int(vectors.shape[1]),
            "normalize": normalize,
            "vector_index": "faiss_flat_ip",
            **(provider_metadata or {}),
        }
        return cls(documents=documents, index=index, metadata=metadata, normalize=normalize)

    def search(
        self,
        query: str,
        embedding_client: EmbeddingProvider,
        *,
        top_k: int,
    ) -> list[SearchHit]:
        query_vector = _as_float32(embedding_client.embed_texts([query]))
        if self.normalize:
            faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, top_k)
        hits: list[SearchHit] = []
        for rank, (score, doc_index) in enumerate(zip(scores[0], indices[0]), start=1):
            if doc_index < 0:
                continue
            document = self.documents[int(doc_index)]
            hits.append(
                SearchHit(
                    doc_id=document.doc_id,
                    text=document.text,
                    score=float(score),
                    rank=rank,
                    metadata=document.metadata,
                )
            )
        return hits

    def write_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_float32(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    return array


def _tokens(text: str) -> list[str]:
    compact = " ".join(text.lower().split())
    latin_tokens = compact.split()
    char_tokens = [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]
    return [token for token in [*latin_tokens, *char_tokens] if token.strip()]

