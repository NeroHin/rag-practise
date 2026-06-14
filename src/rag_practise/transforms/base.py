from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from rag_practise.llm import ChatCompletionRequest, ChatMessage, LLMClient


class TransformResult(BaseModel):
    method: str
    original_query: str
    optimized_query: str | None = None
    retrieval_queries: list[str]
    parse_ok: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def normalized_queries(self) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for query in self.retrieval_queries:
            value = " ".join(query.split())
            if value and value not in seen:
                seen.add(value)
                output.append(value)
        return output


class QueryTransform(ABC):
    name: str

    @abstractmethod
    def transform(self, query: str, *, llm: LLMClient | None, model: str | None) -> TransformResult:
        raise NotImplementedError


class PromptedTransform(QueryTransform):
    system_prompt = "You are a precise query transformation assistant for Chinese RAG retrieval."
    temperature = 0.0
    max_tokens = 512

    def complete(self, query: str, *, llm: LLMClient | None, model: str | None) -> str:
        if llm is None or model is None:
            return self.local_fallback(query)
        request = ChatCompletionRequest(
            model=model,
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=self.build_prompt(query)),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return llm.complete(request).content.strip()

    @abstractmethod
    def build_prompt(self, query: str) -> str:
        raise NotImplementedError

    def local_fallback(self, query: str) -> str:
        return query


def parse_json_object_or_array(content: str) -> tuple[Any | None, bool]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1)), True
        except json.JSONDecodeError:
            return None, False
    return None, False


def lines_from_text(content: str) -> list[str]:
    output: list[str] = []
    for line in content.splitlines():
        value = line.strip().strip("-*0123456789.、) ")
        if value:
            output.append(value)
    return output

