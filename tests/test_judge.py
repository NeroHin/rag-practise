from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag_practise.datasets import CrudRagCase, EvidenceDocument
from rag_practise.evaluation.judge import (
    LLMJudge,
    build_query_judge_prompt,
    build_retrieval_judge_prompt,
    parse_query_judge_result,
    parse_retrieval_judge_result,
    score_to_unit,
)
from rag_practise.experiments.judge_runner import (
    judge_benchmark_records,
    judge_benchmark_records_async,
)
from rag_practise.experiments.runner import BenchmarkRecord
from rag_practise.llm import ChatCompletionRequest, ChatCompletionResult, LLMClient, TokenUsage


class FakeJudgeLLM(LLMClient):
    provider = "fake_judge"

    def __init__(self) -> None:
        self.requests: list[ChatCompletionRequest] = []

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.requests.append(request)
        prompt = request.messages[-1].content
        if "score_query_transformation" in prompt:
            content = json.dumps(
                {
                    "intent_preservation_score": 5,
                    "intent_preservation_reason": "保留原始問題意圖。",
                    "clarity_enhancement_score": 4,
                    "clarity_enhancement_reason": "查詢更具體。",
                },
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {
                    "answer_preference_score": 4,
                    "answer_preference_winner": "candidate",
                    "answer_preference_reason": "candidate context 較能支持回答。",
                    "faithfulness_score": 5,
                    "faithfulness_reason": "答案可由 context 支撐。",
                    "supported_evidence_doc_ids": ["gold_1"],
                    "missing_or_unsupported_aspects": [],
                },
                ensure_ascii=False,
            )
        return ChatCompletionResult(
            content=content,
            model=request.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )


class FailingJudgeLLM(LLMClient):
    provider = "failing_judge"

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        return ChatCompletionResult(
            content="",
            model=request.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )


def test_judge_prompts_are_json_and_parseable() -> None:
    query_prompt = build_query_judge_prompt(
        original_query="原始問題",
        optimized_queries=["優化問題"],
    )
    retrieval_prompt = build_retrieval_judge_prompt(
        original_query="原始問題",
        reference_answer="答案",
        baseline_queries=["原始問題"],
        candidate_queries=["優化問題"],
        baseline_contexts=[{"doc_id": "b1", "text": "baseline"}],
        candidate_contexts=[{"doc_id": "c1", "text": "candidate"}],
    )

    assert json.loads(query_prompt)["output_schema"]["intent_preservation_score"]
    assert json.loads(retrieval_prompt)["output_schema"]["faithfulness_score"]
    assert parse_query_judge_result(
        '{"intent_preservation_score":5,"intent_preservation_reason":"ok",'
        '"clarity_enhancement_score":4,"clarity_enhancement_reason":"ok"}'
    ).clarity_enhancement_score == 4
    assert parse_retrieval_judge_result(
        '{"answer_preference_score":4,"answer_preference_winner":"candidate",'
        '"answer_preference_reason":"ok","faithfulness_score":5,'
        '"faithfulness_reason":"ok","supported_evidence_doc_ids":["d1"],'
        '"missing_or_unsupported_aspects":[]}'
    ).faithfulness_score == 5
    assert score_to_unit(4) == 0.8


def test_llm_judge_requests_structured_output() -> None:
    client = FakeJudgeLLM()
    judge = LLMJudge(client=client, model="fake-judge")

    judge.judge_query(original_query="原始問題", optimized_queries=["優化問題"])
    judge.judge_retrieval(
        original_query="原始問題",
        reference_answer="答案",
        baseline_queries=["原始問題"],
        candidate_queries=["優化問題"],
        baseline_contexts=[{"doc_id": "b1", "text": "baseline"}],
        candidate_contexts=[{"doc_id": "c1", "text": "candidate"}],
    )

    assert client.requests[0].response_format["json_schema"]["name"] == "query_judge_result"
    assert client.requests[0].max_tokens == 1200
    assert client.requests[1].response_format["json_schema"]["name"] == "retrieval_judge_result"
    assert client.requests[1].max_tokens == 1800


def test_judge_benchmark_records_updates_four_metrics(tmp_path: Path) -> None:
    case = CrudRagCase(
        case_id="case_1",
        question="台積電第二季營收是多少？",
        answer="4800 億元",
        source_task="fixture",
        gold_docs=[
            EvidenceDocument(
                doc_id="gold_1",
                text="台積電第二季營收為 4800 億元。",
            )
        ],
    )
    records = [
        BenchmarkRecord(
            run_id="run",
            case_id="case_1",
            provider="openai",
            model_id="openai_fast",
            model="test-model",
            method="baseline",
            original_query=case.question,
            optimized_query=case.question,
            retrieval_queries=[case.question],
            query_count=1,
            retrieved_doc_ids=["gold_1"],
            gold_doc_ids=["gold_1"],
            intent_preservation=0.1,
            clarity_enhancement=0.1,
            recall_at_5=1,
            gold_doc_hit_count=1,
            answer_preference=0.1,
            faithfulness=0.1,
            total_latency_ms=1,
            estimated_cost_usd=0,
            billed_cost_usd=0,
            parse_ok=False,
            transform_events=0,
            metadata={"judge_metrics_status": "judge_failed", "llm_judge_error": "old failure"},
        )
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    distractors_path = tmp_path / "distractors.jsonl"
    distractors_path.write_text("", encoding="utf-8")

    judged = judge_benchmark_records(
        records_path=records_path,
        cases_path=cases_path,
        distractors_path=distractors_path,
        output_dir=tmp_path / "judged",
        judge=LLMJudge(client=FakeJudgeLLM(), model="fake-judge"),
    )

    assert judged[0].intent_preservation == 1.0
    assert judged[0].clarity_enhancement == 0.8
    assert judged[0].answer_preference == 0.8
    assert judged[0].faithfulness == 1.0
    assert judged[0].parse_ok is True
    assert judged[0].metadata["judge_metrics_status"] == "judged"
    assert "llm_judge_error" not in judged[0].metadata
    assert "llm_judge" in judged[0].metadata
    assert (tmp_path / "judged" / "judge_manifest.json").exists()


def test_judge_failure_does_not_overwrite_parse_ok(tmp_path: Path) -> None:
    case = CrudRagCase(
        case_id="case_1",
        question="台積電第二季營收是多少？",
        answer="4800 億元",
        source_task="fixture",
        gold_docs=[
            EvidenceDocument(
                doc_id="gold_1",
                text="台積電第二季營收為 4800 億元。",
            )
        ],
    )
    record = BenchmarkRecord(
        run_id="run",
        case_id="case_1",
        provider="openai",
        model_id="openai_fast",
        model="test-model",
        method="baseline",
        original_query=case.question,
        optimized_query=case.question,
        retrieval_queries=[case.question],
        query_count=1,
        retrieved_doc_ids=["gold_1"],
        gold_doc_ids=["gold_1"],
        intent_preservation=0.0,
        clarity_enhancement=0.0,
        recall_at_5=1,
        gold_doc_hit_count=1,
        answer_preference=0.0,
        faithfulness=0.0,
        total_latency_ms=1,
        estimated_cost_usd=0,
        billed_cost_usd=0,
        parse_ok=True,
        transform_events=0,
        metadata={},
    )
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    distractors_path = tmp_path / "distractors.jsonl"
    distractors_path.write_text("", encoding="utf-8")

    judged = judge_benchmark_records(
        records_path=records_path,
        cases_path=cases_path,
        distractors_path=distractors_path,
        output_dir=tmp_path / "failed_judge",
        judge=LLMJudge(client=FailingJudgeLLM(), model="fake-judge"),
    )

    assert judged[0].parse_ok is True
    assert judged[0].metadata["judge_metrics_status"] == "judge_failed"
    assert "raw_content_preview=''" in judged[0].metadata["llm_judge_error"]


def test_judge_benchmark_records_async_updates_four_metrics(tmp_path: Path) -> None:
    case = CrudRagCase(
        case_id="case_1",
        question="台積電第二季營收是多少？",
        answer="4800 億元",
        source_task="fixture",
        gold_docs=[
            EvidenceDocument(
                doc_id="gold_1",
                text="台積電第二季營收為 4800 億元。",
            )
        ],
    )
    records = [
        BenchmarkRecord(
            run_id="run",
            case_id="case_1",
            provider="openai",
            model_id="openai_fast",
            model="test-model",
            method=method,
            original_query=case.question,
            optimized_query=case.question,
            retrieval_queries=[case.question],
            query_count=1,
            retrieved_doc_ids=["gold_1"],
            gold_doc_ids=["gold_1"],
            intent_preservation=0.0,
            clarity_enhancement=0.0,
            recall_at_5=1,
            gold_doc_hit_count=1,
            answer_preference=0.0,
            faithfulness=0.0,
            total_latency_ms=1,
            estimated_cost_usd=0,
            billed_cost_usd=0,
            parse_ok=True,
            transform_events=0,
            metadata={"judge_metrics_status": "not_judged"},
        )
        for method in ["baseline", "rewrite"]
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    distractors_path = tmp_path / "distractors.jsonl"
    distractors_path.write_text("", encoding="utf-8")

    judged = asyncio.run(
        judge_benchmark_records_async(
            records_path=records_path,
            cases_path=cases_path,
            distractors_path=distractors_path,
            output_dir=tmp_path / "judged_async",
            judge_factory=lambda: LLMJudge(client=FakeJudgeLLM(), model="fake-judge"),
            max_concurrency=2,
        )
    )

    assert [record.method for record in judged] == ["baseline", "rewrite"]
    assert all(record.intent_preservation == 1.0 for record in judged)
    assert all(record.clarity_enhancement == 0.8 for record in judged)
    assert all(record.answer_preference == 0.8 for record in judged)
    assert all(record.faithfulness == 1.0 for record in judged)
    assert all(record.metadata["judge_metrics_status"] == "judged" for record in judged)
    assert all("llm_judge" in record.metadata for record in judged)
    assert (tmp_path / "judged_async" / "records.jsonl").exists()
    assert (tmp_path / "judged_async" / "judge_records.checkpoint.jsonl").exists()
