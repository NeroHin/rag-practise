from __future__ import annotations

from rag_practise.llm import LLMClient
from rag_practise.transforms.base import PromptedTransform, TransformResult


class RewriteTransform(PromptedTransform):
    name = "rewrite"
    system_prompt = (
        "你是一個專業的查詢優化助手。請將以下原始查詢重寫得更詳細，"
        "回應必須使用正體中文，且僅輸出重寫後的查詢內容。"
    )

    def build_prompt(self, query: str) -> str:
        return (
            "TRANSFORM_METHOD: rewrite\n"
            "以下是原始查詢，請根據查詢意圖提供更完整、具體的改寫。\n\n"
            f"原始查詢：{query}\n\n"
            "輸出格式要求：\n"
            "- 不得包含任何說明、標題或多餘文字。\n"
            "- 需維持句子結構且語意通順。\n"
            "- 需保留原始查詢中的時間、地點、人物、組織、事件與限制條件。"
        )

    def local_fallback(self, query: str) -> str:
        return f"{query}，請補充相關背景、主要事實、數據與影響。"

    def transform(self, query: str, *, llm: LLMClient | None, model: str | None) -> TransformResult:
        rewritten = self.complete(query, llm=llm, model=model)
        return TransformResult(
            method=self.name,
            original_query=query,
            optimized_query=rewritten,
            retrieval_queries=[rewritten],
        )
