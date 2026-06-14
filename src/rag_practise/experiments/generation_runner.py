from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from rag_practise.datasets import (
    EvidenceDocument,
    build_index_documents,
    load_compact_cases,
    load_compact_distractors,
)
from rag_practise.experiments.events import EventLogger, EventLoggingLLMClient
from rag_practise.experiments.model_config import ModelConfig
from rag_practise.experiments.runner import BenchmarkRecord, load_benchmark_records, write_report_artifacts
from rag_practise.generation import generate_answer
from rag_practise.llm import LLMClient
from rag_practise.retrieval.index import SearchHit

GenerationClientFactory = Callable[[], LLMClient]


def generate_answers_for_records(
    *,
    records_path: Path,
    cases_path: Path,
    distractors_path: Path,
    output_dir: Path,
    llm: LLMClient,
    model_config: ModelConfig,
    limit: int | None = None,
) -> list[BenchmarkRecord]:
    records = load_benchmark_records(records_path)
    if limit is not None:
        records = records[:limit]
    documents = _document_map(cases_path=cases_path, distractors_path=distractors_path)
    generated = [
        _generate_answer_for_record(
            record=record,
            documents=documents,
            llm=llm,
            model_config=model_config,
        )
        for record in records
    ]
    write_report_artifacts(output_dir, generated)
    return generated


async def generate_answers_for_records_async(
    *,
    records_path: Path,
    cases_path: Path,
    distractors_path: Path,
    output_dir: Path,
    client_factory: GenerationClientFactory,
    model_config: ModelConfig,
    limit: int | None = None,
    max_concurrency: int = 4,
) -> list[BenchmarkRecord]:
    records = load_benchmark_records(records_path)
    if limit is not None:
        records = records[:limit]
    documents = _document_map(cases_path=cases_path, distractors_path=distractors_path)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_one(record: BenchmarkRecord) -> BenchmarkRecord:
        async with semaphore:
            return await asyncio.to_thread(
                _generate_answer_for_record,
                record=record,
                documents=documents,
                llm=client_factory(),
                model_config=model_config,
            )

    generated = list(await asyncio.gather(*(generate_one(record) for record in records)))
    write_report_artifacts(output_dir, generated)
    return generated


def _generate_answer_for_record(
    *,
    record: BenchmarkRecord,
    documents: dict[str, EvidenceDocument],
    llm: LLMClient,
    model_config: ModelConfig,
) -> BenchmarkRecord:
    logger = EventLogger()
    client = EventLoggingLLMClient(
        llm,
        logger=logger,
        pricing=model_config.pricing_policy(),
    )
    metadata = dict(record.metadata)
    try:
        answer = generate_answer(
            llm=client,
            model=model_config.model,
            question=record.original_query,
            contexts=_contexts_for_record(record, documents),
        )
        events = logger.events
        generation_latency_ms = sum(event.latency_ms for event in events)
        generation_estimated_cost_usd = sum(event.estimated_cost_usd for event in events)
        generation_billed_cost_usd = sum(event.billed_cost_usd for event in events)
        metadata["answer_generation_status"] = "generated"
        return record.model_copy(
            update={
                "generated_answer": answer,
                "generation_model_id": model_config.id,
                "generation_provider": model_config.provider,
                "generation_latency_ms": round(generation_latency_ms, 4),
                "generation_estimated_cost_usd": round(generation_estimated_cost_usd, 8),
                "generation_billed_cost_usd": round(generation_billed_cost_usd, 8),
                "total_latency_ms": round(record.total_latency_ms + generation_latency_ms, 4),
                "estimated_cost_usd": round(
                    record.estimated_cost_usd + generation_estimated_cost_usd, 8
                ),
                "billed_cost_usd": round(record.billed_cost_usd + generation_billed_cost_usd, 8),
                "metadata": metadata,
            }
        )
    except Exception as exc:
        metadata["answer_generation_status"] = "generation_failed"
        metadata["answer_generation_error"] = str(exc)
        return record.model_copy(update={"parse_ok": False, "metadata": metadata})


def _document_map(*, cases_path: Path, distractors_path: Path) -> dict[str, EvidenceDocument]:
    cases = load_compact_cases(cases_path)
    distractors = load_compact_distractors(distractors_path)
    return {document.doc_id: document for document in build_index_documents(cases, distractors)}


def _contexts_for_record(
    record: BenchmarkRecord, documents: dict[str, EvidenceDocument]
) -> list[SearchHit]:
    contexts: list[SearchHit] = []
    for rank, doc_id in enumerate(record.retrieved_doc_ids, start=1):
        document = documents.get(doc_id)
        if document is not None:
            contexts.append(
                SearchHit(
                    doc_id=document.doc_id,
                    text=document.text,
                    score=0.0,
                    rank=rank,
                    metadata=document.metadata,
                )
            )
    return contexts
