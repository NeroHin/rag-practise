from __future__ import annotations

from rag_practise.llm import LLMClient
from rag_practise.transforms.base import (
    PromptedTransform,
    TransformResult,
    lines_from_text,
    parse_json_object_or_array,
)


class ExpandTransform(PromptedTransform):
    name = "expand"
    system_prompt = "你是一個專業的查詢擴充助手，會為查詢提供同義詞、相關概念與上下位詞。"

    def build_prompt(self, query: str) -> str:
        return (
            "TRANSFORM_METHOD: expand\n"
            "請針對以下查詢提供 5 個擴充詞彙或短語。"
            "輸出格式必須為 JSON 陣列，元素類型需為字串，且不可包含註解或其他文字。\n\n"
            f"原始查詢：{query}"
        )

    def local_fallback(self, query: str) -> str:
        return f'["{query}", "{query} 背景", "{query} 相關概念", "{query} 影響", "{query} 數據"]'

    def transform(self, query: str, *, llm: LLMClient | None, model: str | None) -> TransformResult:
        content = self.complete(query, llm=llm, model=model)
        parsed, parse_ok = parse_json_object_or_array(content)
        if isinstance(parsed, list):
            expansions = [str(item) for item in parsed if str(item).strip()]
        else:
            expansions = lines_from_text(content)
            parse_ok = False
        return TransformResult(
            method=self.name,
            original_query=query,
            optimized_query=query,
            retrieval_queries=[query, *expansions],
            parse_ok=parse_ok,
            metadata={"expansions": expansions},
        )
