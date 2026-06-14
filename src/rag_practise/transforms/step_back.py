from __future__ import annotations

import re

from rag_practise.llm import LLMClient
from rag_practise.transforms.base import PromptedTransform, TransformResult, parse_json_object_or_array


class StepBackTransform(PromptedTransform):
    name = "step_back"
    max_tokens = 768
    system_prompt = (
        "你是一個專精於 Step-Back Prompting 的查詢優化助手。你的任務不是回答原始問題，"
        "而是幫助重寫問題，讓後續模型更容易基於高層概念回答它。\n\n"
        "流程分兩步：\n\n"
        "### Step 1: Abstraction（抽象化）\n"
        "- 生成一個更高層的 Step-Back Question，以概括原問題的核心概念或原理。\n"
        "- 根據該問題，提供一段 Step-Back Answer（高層背景或理論）。\n\n"
        "### Step 2: Rewrite（基於抽象的重寫）\n"
        "- 根據 Step-Back Answer，將原始問題改寫成一個更具語義結構、範圍明確、"
        "且能引導模型聚焦核心議題的新查詢（Optimized Query）。\n\n"
        "最終輸出格式：\n"
        "### Step 1: Abstraction\n"
        "Stepback Question: ...\n"
        "Stepback Answer: ...\n\n"
        "### Step 2: Rewrite\n"
        "Optimized Query: ...（這是新的查詢，不是答案）"
    )

    def build_prompt(self, query: str) -> str:
        return (
            "TRANSFORM_METHOD: step_back\n"
            "請使用 Step-Back Prompting 方法來重寫以下問題，讓它更具抽象性、結構清晰、"
            "並聚焦於關鍵議題。注意要保持和原問題的同樣動機和意圖。\n\n"
            f"原問題：{query}"
        )

    def local_fallback(self, query: str) -> str:
        return (
            '{"stepback_question": "這類事件通常涉及哪些背景因素？", '
            '"stepback_answer": "相關背景可能包含政策、市場、技術、財務數據與利害關係人影響。", '
            f'"optimized_query": "{query} 背景 原因 影響"' 
            "}"
        )

    def transform(self, query: str, *, llm: LLMClient | None, model: str | None) -> TransformResult:
        content = self.complete(query, llm=llm, model=model)
        parsed, parse_ok = parse_json_object_or_array(content)
        optimized = query
        stepback_question = ""
        stepback_answer = ""
        if isinstance(parsed, dict):
            optimized = str(parsed.get("optimized_query") or query)
            stepback_question = str(parsed.get("stepback_question") or "")
            stepback_answer = str(parsed.get("stepback_answer") or "")
        else:
            markdown_result = _parse_step_back_markdown(content)
            if markdown_result is None:
                parse_ok = False
                stepback_question = "這個問題背後的一般性背景是什麼？"
                stepback_answer = content.strip()
            else:
                parse_ok = True
                stepback_question, stepback_answer, optimized = markdown_result
        return TransformResult(
            method=self.name,
            original_query=query,
            optimized_query=optimized,
            retrieval_queries=[optimized, stepback_question, stepback_answer],
            parse_ok=parse_ok,
            metadata={
                "stepback_question": stepback_question,
                "stepback_answer": stepback_answer,
            },
        )


def _parse_step_back_markdown(content: str) -> tuple[str, str, str] | None:
    question = _match_label(content, "Stepback Question")
    answer = _match_label(content, "Stepback Answer")
    optimized = _match_label(content, "Optimized Query")
    if question and answer and optimized:
        return question, answer, optimized
    return None


def _match_label(content: str, label: str) -> str | None:
    pattern = rf"{re.escape(label)}\s*[:：]\s*(.+?)(?=\n[A-Za-z ]+\s*[:：]|\n###|\Z)"
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        return None
    value = " ".join(match.group(1).strip().split())
    return value or None
