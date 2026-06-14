from __future__ import annotations

from typing import Any

from rag_practise.llm import LLMClient
from rag_practise.transforms.base import (
    PromptedTransform,
    TransformResult,
    lines_from_text,
    parse_json_object_or_array,
)


class ChildQueryTransform(PromptedTransform):
    name = "child_query"
    system_prompt = "你是一個子查詢生成器，會根據主查詢建立層級化的子查詢樹。"

    def build_prompt(self, query: str) -> str:
        return (
            "TRANSFORM_METHOD: child_query\n"
            "請針對以下主查詢產生 3-6 個第一層子查詢，必要時再為每個子查詢提供 1-3 個次子查詢。"
            "輸出格式必須為 JSON 物件："
            "{\"children\":[{\"query\":str,\"subqueries\":[str,...]}]}。"
            "所有字串需為正體中文。若沒有次子查詢，subqueries 請使用空陣列。\n\n"
            f"主查詢：{query}"
        )

    def local_fallback(self, query: str) -> str:
        return (
            '{"children": ['
            f'{{"query": "{query} 主要事實", "subqueries": []}}, '
            f'{{"query": "{query} 背景", "subqueries": []}}, '
            f'{{"query": "{query} 影響", "subqueries": []}}'
            "]}"
        )

    def transform(self, query: str, *, llm: LLMClient | None, model: str | None) -> TransformResult:
        content = self.complete(query, llm=llm, model=model)
        parsed, parse_ok = parse_json_object_or_array(content)
        children: list[str] = []
        if isinstance(parsed, dict):
            for item in parsed.get("children", []):
                children.extend(_child_to_queries(item))
        elif isinstance(parsed, list):
            children = [str(item) for item in parsed if str(item).strip()]
        else:
            children = lines_from_text(content)
            parse_ok = False
        return TransformResult(
            method=self.name,
            original_query=query,
            optimized_query=query,
            retrieval_queries=[query, *children],
            parse_ok=parse_ok,
            metadata={"children": children},
        )


def _child_to_queries(item: Any) -> list[str]:
    if isinstance(item, str):
        value = item.strip()
        return [value] if value else []
    if isinstance(item, dict):
        output: list[str] = []
        value = item.get("query") or item.get("question")
        if value:
            output.append(str(value).strip())
        subqueries = item.get("subqueries") or item.get("children") or []
        if isinstance(subqueries, list):
            output.extend(str(subquery).strip() for subquery in subqueries if str(subquery).strip())
        return [query for query in output if query]
    return []
