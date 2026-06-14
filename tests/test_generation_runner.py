from __future__ import annotations

import asyncio
from pathlib import Path

from rag_practise.datasets import CrudRagCase, EvidenceDocument
from rag_practise.experiments.generation_runner import generate_answers_for_records_async
from rag_practise.experiments.model_config import ModelConfig
from rag_practise.experiments.runner import BenchmarkRecord
from rag_practise.llm import ChatCompletionRequest, ChatCompletionResult, LLMClient, TokenUsage


class FakeGenerationLLM(LLMClient):
    provider = "fake_generate"

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        return ChatCompletionResult(
            content="台積電第二季營收為 4800 億元。",
            model=request.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )


def test_generate_answers_for_records_async_writes_answer_fields(tmp_path: Path) -> None:
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
        model="test-transform",
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
        total_latency_ms=5,
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
    model_config = ModelConfig(
        id="generate_model",
        provider="openai",
        model="fake-generate",
        roles=["generate"],
        input_usd_per_1k_tokens=0.001,
        output_usd_per_1k_tokens=0.002,
    )

    generated = asyncio.run(
        generate_answers_for_records_async(
            records_path=records_path,
            cases_path=cases_path,
            distractors_path=distractors_path,
            output_dir=tmp_path / "generated",
            client_factory=FakeGenerationLLM,
            model_config=model_config,
            max_concurrency=2,
        )
    )

    assert generated[0].generated_answer == "台積電第二季營收為 4800 億元。"
    assert generated[0].generation_model_id == "generate_model"
    assert generated[0].generation_provider == "openai"
    assert generated[0].generation_estimated_cost_usd == 0.00014
    assert generated[0].total_latency_ms >= 5
    assert generated[0].metadata["answer_generation_status"] == "generated"
    assert (tmp_path / "generated" / "records.jsonl").exists()
