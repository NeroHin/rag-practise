from __future__ import annotations

from rag_practise.llm import LLMClient
from rag_practise.transforms.base import (
    PromptedTransform,
    TransformResult,
    lines_from_text,
    parse_json_object_or_array,
)


class HyDETransform(PromptedTransform):
    name = "hyde"
    max_tokens = 768
    system_prompt = (
        "角色設定：你是「HyDE 假設性內容生成器」。你的任務是根據使用者查詢，"
        "產生 3-5 段可供檢索的「假設性但合理」內容片段，用於向量檢索與語義搜尋。\n"
        "要求：\n"
        "- 以中立、百科式語氣撰寫，避免不可驗證的精確數字或來源斷言。\n"
        "- 每段 80-160 字為原則，聚焦關鍵實體、專有名詞、同義詞、縮寫與別名。\n"
        "- 片段之間需語義去重，涵蓋不同面向。\n"
        "- 優先保留查詢中的關鍵實體與限制條件。\n"
        "- 僅產生可檢索內容，不輸出最終答案或觀點結論。\n"
        "- 所有輸出須使用正體中文。"
    )

    def build_prompt(self, query: str) -> str:
        return (
            "TRANSFORM_METHOD: hyde\n"
            "請依下列規則回覆：\n"
            "1. 只輸出 JSON 物件，不得包含註解或多餘文字。\n"
            "2. JSON 格式必須為 {\"optimized_query\": str, \"passages\": [str, ...]}。\n"
            "3. passages 請提供 3-5 段，每段 80-160 字；避免流水帳，聚焦可檢索的關鍵訊息與專有名詞。\n"
            "4. 涵蓋多元面向，並加入合理的同義詞、縮寫、別名。\n"
            "5. 若原查詢過於抽象，請先假設合理情境後再生成片段，但避免杜撰來源或精確數字。\n"
            "6. 片段內容彼此去重。\n\n"
            f"原始查詢：{query}"
        )

    def local_fallback(self, query: str) -> str:
        return (
            '{"optimized_query": "'
            + query
            + '", "passages": ["'
            + query
            + ' 的相關文件通常會包含事件背景、主要數據、原因與影響。"]}'
        )

    def transform(self, query: str, *, llm: LLMClient | None, model: str | None) -> TransformResult:
        content = self.complete(query, llm=llm, model=model)
        parsed, parse_ok = parse_json_object_or_array(content)
        optimized = query
        passages: list[str] = []
        if isinstance(parsed, dict):
            optimized = str(parsed.get("optimized_query") or query)
            passages = [str(item) for item in parsed.get("passages", []) if str(item).strip()]
        elif isinstance(parsed, list):
            passages = [str(item) for item in parsed if str(item).strip()]
        else:
            passages = lines_from_text(content)
            parse_ok = False
        return TransformResult(
            method=self.name,
            original_query=query,
            optimized_query=optimized,
            retrieval_queries=[optimized, *passages],
            parse_ok=parse_ok,
            metadata={"passages": passages},
        )
