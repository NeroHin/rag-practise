from __future__ import annotations

from rag_practise.llm import LLMClient
from rag_practise.transforms.base import (
    PromptedTransform,
    TransformResult,
    lines_from_text,
    parse_json_object_or_array,
)


class MultiQueryTransform(PromptedTransform):
    name = "multi_query"
    system_prompt = "你是一個查詢拆解助手，將複雜問題拆解為 2-5 個語意獨立的子查詢。"

    def build_prompt(self, query: str) -> str:
        return (
            "TRANSFORM_METHOD: multi_query\n"
            "請根據以下查詢，輸出 2 到 5 個子查詢，格式為 JSON 陣列 (Array of strings)。"
            "每個子查詢需完整描述欲檢索的資訊要點，避免重複。\n\n"
            f"原始查詢：{query}"
        )

    def local_fallback(self, query: str) -> str:
        return f'["{query} 的核心事件與背景", "{query} 的主要數據與事實", "{query} 的影響與後續發展"]'

    def transform(self, query: str, *, llm: LLMClient | None, model: str | None) -> TransformResult:
        content = self.complete(query, llm=llm, model=model)
        parsed, parse_ok = parse_json_object_or_array(content)
        if isinstance(parsed, list):
            queries = [str(item) for item in parsed if str(item).strip()]
        else:
            queries = lines_from_text(content)
            parse_ok = False
        return TransformResult(
            method=self.name,
            original_query=query,
            optimized_query=query,
            retrieval_queries=[query, *queries],
            parse_ok=parse_ok,
            metadata={"queries": queries},
        )
