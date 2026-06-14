from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from rag_practise.llm import ChatCompletionRequest, ChatMessage, LLMClient

ScoreLabel = Literal[1, 2, 3, 4, 5]


QUERY_JUDGE_SYSTEM_PROMPT = """你是一個嚴格、穩定的 RAG Query Transformation 評估員。
你的工作是比較原始查詢與優化後查詢，判斷 query transformation 是否保留原始意圖，並是否讓查詢更清晰、更適合檢索。

評分規則：
- 所有分數只能是 1, 2, 3, 4, 5 的整數。
- 1 = 很差，2 = 偏差，3 = 普通或無明顯改善，4 = 良好，5 = 優秀。
- intent_preservation_score 評估是否保留原始查詢的核心任務、實體、條件、時間、地點與限制。
- clarity_enhancement_score 評估優化後查詢是否更明確、無歧義、具體，且更適合檢索系統找到相關文件。
- 不要因為查詢變長就自動給高分；如果新增資訊改變原意，intent 必須降分。
- 不要懲罰合理的同義詞、拆解、多查詢或 HyDE passages；只要仍服務於原始問題即可。
- 回覆必須是單一 JSON object，不得輸出 Markdown、註解或其他文字。"""


RETRIEVAL_JUDGE_SYSTEM_PROMPT = """你是一個嚴格、穩定的 RAG Retrieval / Answer Support 評估員。
你會看到原始問題、參考答案、baseline 檢索結果，以及 candidate 方法的檢索結果。
你的工作不是判斷文字是否好看，而是判斷 candidate 檢索結果是否比 baseline 更可能支持正確回答。

評分規則：
- 所有分數只能是 1, 2, 3, 4, 5 的整數。
- answer_preference_score 評估 candidate retrieved contexts 相對 baseline 是否更能回答原始問題。
- faithfulness_score 評估 candidate retrieved contexts 是否足以支持 reference_answer；若 contexts 缺少答案依據，必須低分。
- 如果 candidate 和 baseline 幾乎等價，winner 使用 "tie"。
- 如果 candidate 因查詢改寫偏離問題而找到不相關內容，answer_preference_score 和 faithfulness_score 都應降低。
- 只根據提供的 contexts 評估，不使用外部知識。
- 回覆必須是單一 JSON object，不得輸出 Markdown、註解或其他文字。"""


class QueryJudgeResult(BaseModel):
    intent_preservation_score: ScoreLabel
    intent_preservation_reason: str
    clarity_enhancement_score: ScoreLabel
    clarity_enhancement_reason: str


class RetrievalJudgeResult(BaseModel):
    answer_preference_score: ScoreLabel
    answer_preference_winner: Literal["baseline", "candidate", "tie"]
    answer_preference_reason: str
    faithfulness_score: ScoreLabel
    faithfulness_reason: str
    supported_evidence_doc_ids: list[str] = Field(default_factory=list)
    missing_or_unsupported_aspects: list[str] = Field(default_factory=list)

    @field_validator("supported_evidence_doc_ids", "missing_or_unsupported_aspects", mode="before")
    @classmethod
    def coerce_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]


class JudgeOutputParseError(ValueError):
    def __init__(self, message: str, *, raw_content: str) -> None:
        preview = raw_content.strip().replace("\n", " ")[:500]
        super().__init__(f"{message}; raw_content_preview={preview!r}")
        self.raw_content = raw_content


class LLMJudge:
    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        max_context_chars: int = 1200,
        use_structured_output: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.max_context_chars = max_context_chars
        self.use_structured_output = use_structured_output

    def judge_query(self, *, original_query: str, optimized_queries: list[str]) -> QueryJudgeResult:
        result = self.client.complete(
            ChatCompletionRequest(
                model=self.model,
                messages=[
                    ChatMessage(role="system", content=QUERY_JUDGE_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=build_query_judge_prompt(
                            original_query=original_query,
                            optimized_queries=optimized_queries,
                        ),
                    ),
                ],
                temperature=0,
                max_tokens=1200,
                response_format=(
                    query_judge_response_format() if self.use_structured_output else None
                ),
            )
        )
        return parse_query_judge_result(result.content)

    def judge_retrieval(
        self,
        *,
        original_query: str,
        reference_answer: str,
        baseline_queries: list[str],
        candidate_queries: list[str],
        baseline_contexts: list[dict[str, str]],
        candidate_contexts: list[dict[str, str]],
    ) -> RetrievalJudgeResult:
        result = self.client.complete(
            ChatCompletionRequest(
                model=self.model,
                messages=[
                    ChatMessage(role="system", content=RETRIEVAL_JUDGE_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=build_retrieval_judge_prompt(
                            original_query=original_query,
                            reference_answer=reference_answer,
                            baseline_queries=baseline_queries,
                            candidate_queries=candidate_queries,
                            baseline_contexts=baseline_contexts,
                            candidate_contexts=candidate_contexts,
                            max_context_chars=self.max_context_chars,
                        ),
                    ),
                ],
                temperature=0,
                max_tokens=1800,
                response_format=(
                    retrieval_judge_response_format() if self.use_structured_output else None
                ),
            )
        )
        return parse_retrieval_judge_result(result.content)


def build_query_judge_prompt(*, original_query: str, optimized_queries: list[str]) -> str:
    payload = {
        "task": "score_query_transformation",
        "original_query": original_query,
        "optimized_queries": optimized_queries,
        "output_schema": {
            "intent_preservation_score": "integer 1-5",
            "intent_preservation_reason": "string, concise Traditional Chinese",
            "clarity_enhancement_score": "integer 1-5",
            "clarity_enhancement_reason": "string, concise Traditional Chinese",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_retrieval_judge_prompt(
    *,
    original_query: str,
    reference_answer: str,
    baseline_queries: list[str],
    candidate_queries: list[str],
    baseline_contexts: list[dict[str, str]],
    candidate_contexts: list[dict[str, str]],
    max_context_chars: int = 1200,
) -> str:
    payload = {
        "task": "score_retrieval_support_against_baseline",
        "original_query": original_query,
        "reference_answer": reference_answer,
        "baseline": {
            "queries": baseline_queries,
            "retrieved_contexts": _trim_contexts(baseline_contexts, max_context_chars),
        },
        "candidate": {
            "queries": candidate_queries,
            "retrieved_contexts": _trim_contexts(candidate_contexts, max_context_chars),
        },
        "output_schema": {
            "answer_preference_score": "integer 1-5",
            "answer_preference_winner": "baseline | candidate | tie",
            "answer_preference_reason": "string, concise Traditional Chinese",
            "faithfulness_score": "integer 1-5",
            "faithfulness_reason": "string, concise Traditional Chinese",
            "supported_evidence_doc_ids": "array of strings",
            "missing_or_unsupported_aspects": "array of strings",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_query_judge_result(content: str) -> QueryJudgeResult:
    return QueryJudgeResult.model_validate(_parse_json(content))


def parse_retrieval_judge_result(content: str) -> RetrievalJudgeResult:
    return RetrievalJudgeResult.model_validate(_parse_json(content))


def query_judge_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "query_judge_result",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent_preservation_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "intent_preservation_reason": {"type": "string"},
                    "clarity_enhancement_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "clarity_enhancement_reason": {"type": "string"},
                },
                "required": [
                    "intent_preservation_score",
                    "intent_preservation_reason",
                    "clarity_enhancement_score",
                    "clarity_enhancement_reason",
                ],
            },
        },
    }


def retrieval_judge_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "retrieval_judge_result",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "answer_preference_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "answer_preference_winner": {
                        "type": "string",
                        "enum": ["baseline", "candidate", "tie"],
                    },
                    "answer_preference_reason": {"type": "string"},
                    "faithfulness_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    "faithfulness_reason": {"type": "string"},
                    "supported_evidence_doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "missing_or_unsupported_aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "answer_preference_score",
                    "answer_preference_winner",
                    "answer_preference_reason",
                    "faithfulness_score",
                    "faithfulness_reason",
                    "supported_evidence_doc_ids",
                    "missing_or_unsupported_aspects",
                ],
            },
        },
    }


def score_to_unit(score: int) -> float:
    return round(max(1, min(5, score)) / 5, 4)


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        raise JudgeOutputParseError("Judge returned empty content", raw_content=content)
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not match:
            raise JudgeOutputParseError("Judge output did not contain a JSON object", raw_content=content)
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise JudgeOutputParseError(
                f"Judge JSON object was invalid: {exc}", raw_content=content
            ) from exc
    if not isinstance(value, dict):
        raise ValidationError.from_exception_data("Judge output must be a JSON object", [])
    return value


def _trim_contexts(contexts: list[dict[str, str]], max_context_chars: int) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for context in contexts:
        text = context.get("text", "")
        output.append(
            {
                "doc_id": context.get("doc_id", ""),
                "text": text[:max_context_chars],
            }
        )
    return output
