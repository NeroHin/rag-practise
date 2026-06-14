from __future__ import annotations

from rag_practise.llm import ChatCompletionRequest, ChatCompletionResult, LLMClient, TokenUsage
from rag_practise.transforms import build_transform


class MethodFakeLLM(LLMClient):
    provider = "fake"

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        prompt = request.messages[-1].content
        if "TRANSFORM_METHOD: rewrite" in prompt:
            content = "台積電 2023 第二季 財報 營收 毛利率"
        elif "TRANSFORM_METHOD: expand" in prompt:
            content = '["台積電 第二季營收", "台積電 毛利率", "台積電 財測"]'
        elif "TRANSFORM_METHOD: multi_query" in prompt:
            content = '["台積電 Q2 營收", "TSMC 2023 第二季 財報", "台積電 法說會"]'
        elif "TRANSFORM_METHOD: child_query" in prompt:
            content = (
                '{"children":[{"query":"台積電 Q2 營收","subqueries":["台積電 Q2 月營收"]},'
                '{"query":"台積電 Q2 毛利率","subqueries":[]}]}'
            )
        elif "TRANSFORM_METHOD: hyde" in prompt:
            content = (
                '{"optimized_query":"台積電 Q2 財報",'
                '"passages":["台積電第二季財報說明營收、毛利率與展望。"]}'
            )
        elif "TRANSFORM_METHOD: step_back" in prompt:
            content = (
                "### Step 1: Abstraction\n"
                "Stepback Question: 半導體公司財報通常看哪些指標？\n"
                "Stepback Answer: 通常看營收、毛利率、資本支出與需求展望。\n\n"
                "### Step 2: Rewrite\n"
                "Optimized Query: 台積電 Q2 財報 指標"
            )
        else:
            content = "unknown"
        return ChatCompletionResult(
            content=content,
            model=request.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def test_all_query_transforms_produce_retrieval_queries() -> None:
    llm = MethodFakeLLM()
    methods = ["baseline", "rewrite", "expand", "multi_query", "child_query", "hyde", "step_back"]

    results = [
        build_transform(method).transform("台積電第二季財報重點是什麼？", llm=llm, model="fake-model")
        for method in methods
    ]

    assert [result.method for result in results] == methods
    assert all(result.normalized_queries() for result in results)
    assert all(result.parse_ok for result in results)
    assert results[3].retrieval_queries[1:] == [
        "台積電 Q2 營收",
        "TSMC 2023 第二季 財報",
        "台積電 法說會",
    ]
    assert results[4].retrieval_queries[1:] == [
        "台積電 Q2 營收",
        "台積電 Q2 月營收",
        "台積電 Q2 毛利率",
    ]
    assert "台積電第二季財報說明" in results[5].retrieval_queries[1]
    assert results[6].metadata["stepback_question"] == "半導體公司財報通常看哪些指標？"


def test_prompts_reference_notion_article_patterns() -> None:
    rewrite = build_transform("rewrite")
    expand = build_transform("expand")
    multi_query = build_transform("multi_query")
    child_query = build_transform("child_query")
    hyde = build_transform("hyde")
    step_back = build_transform("step_back")

    assert "專業的查詢優化助手" in rewrite.system_prompt
    assert "同義詞、相關概念與上下位詞" in expand.system_prompt
    assert "2-5 個語意獨立的子查詢" in multi_query.system_prompt
    assert "層級化的子查詢樹" in child_query.system_prompt
    assert "HyDE 假設性內容生成器" in hyde.system_prompt
    assert "Step-Back Prompting" in step_back.system_prompt
