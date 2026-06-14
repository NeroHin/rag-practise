from __future__ import annotations

from rag_practise.llm import LLMClient
from rag_practise.transforms.base import QueryTransform, TransformResult


class BaselineTransform(QueryTransform):
    name = "baseline"

    def transform(self, query: str, *, llm: LLMClient | None, model: str | None) -> TransformResult:
        return TransformResult(
            method=self.name,
            original_query=query,
            optimized_query=query,
            retrieval_queries=[query],
        )

