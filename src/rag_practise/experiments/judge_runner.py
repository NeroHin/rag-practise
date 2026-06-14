from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable

from rag_practise.datasets import (
    CrudRagCase,
    EvidenceDocument,
    build_index_documents,
    load_compact_cases,
    load_compact_distractors,
)
from rag_practise.evaluation.judge import LLMJudge, score_to_unit
from rag_practise.experiments.runner import BenchmarkRecord, load_benchmark_records, write_report_artifacts

JudgeFactory = Callable[[], LLMJudge]


def judge_benchmark_records(
    *,
    records_path: Path,
    cases_path: Path,
    distractors_path: Path,
    output_dir: Path,
    judge: LLMJudge,
    limit: int | None = None,
    max_attempts: int = 1,
) -> list[BenchmarkRecord]:
    records = load_benchmark_records(records_path)
    if limit is not None:
        records = records[:limit]
    cases = {case.case_id: case for case in load_compact_cases(cases_path)}
    documents = _document_map(
        cases=list(cases.values()),
        distractors=load_compact_distractors(distractors_path),
    )
    baseline_by_key = {
        (record.provider, record.model_id, record.case_id): record
        for record in load_benchmark_records(records_path)
        if record.method == "baseline"
    }

    judged_records = [
        _judge_record(
            record=record,
            case=cases[record.case_id],
            baseline=baseline_by_key.get((record.provider, record.model_id, record.case_id), record),
            documents=documents,
            judge=judge,
            max_attempts=max_attempts,
        )
        for record in records
    ]

    write_report_artifacts(output_dir, judged_records)
    _write_judge_manifest(output_dir)
    return judged_records


async def judge_benchmark_records_async(
    *,
    records_path: Path,
    cases_path: Path,
    distractors_path: Path,
    output_dir: Path,
    judge_factory: JudgeFactory,
    limit: int | None = None,
    max_concurrency: int = 4,
    resume: bool = True,
    retry_failed: bool = False,
    max_attempts: int = 1,
) -> list[BenchmarkRecord]:
    records = load_benchmark_records(records_path)
    if limit is not None:
        records = records[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "judge_records.checkpoint.jsonl"
    judged_by_key = (
        {
            _record_key(record): record
            for record in _load_checkpoint_records(checkpoint_path)
            if not retry_failed or record.metadata.get("judge_metrics_status") == "judged"
        }
        if resume
        else {}
    )
    records_to_judge = [record for record in records if _record_key(record) not in judged_by_key]
    cases = {case.case_id: case for case in load_compact_cases(cases_path)}
    documents = _document_map(
        cases=list(cases.values()),
        distractors=load_compact_distractors(distractors_path),
    )
    baseline_by_key = {
        (record.provider, record.model_id, record.case_id): record
        for record in load_benchmark_records(records_path)
        if record.method == "baseline"
    }
    semaphore = asyncio.Semaphore(max_concurrency)

    async def judge_one(record: BenchmarkRecord) -> BenchmarkRecord:
        async with semaphore:
            return await asyncio.to_thread(
                _judge_record,
                record=record,
                case=cases[record.case_id],
                baseline=baseline_by_key.get(
                    (record.provider, record.model_id, record.case_id), record
                ),
                documents=documents,
                judge=judge_factory(),
                max_attempts=max_attempts,
            )

    judged_records = list(judged_by_key.values())
    tasks = [asyncio.create_task(judge_one(record)) for record in records_to_judge]
    for task in asyncio.as_completed(tasks):
        judged_record = await task
        judged_records.append(judged_record)
        _append_checkpoint_record(checkpoint_path, judged_record)
    write_report_artifacts(output_dir, judged_records)
    _write_judge_manifest(output_dir)
    return judged_records


def _judge_record(
    *,
    record: BenchmarkRecord,
    case: CrudRagCase,
    baseline: BenchmarkRecord,
    documents: dict[str, EvidenceDocument],
    judge: LLMJudge,
    max_attempts: int = 1,
) -> BenchmarkRecord:
    last_exception: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _judge_record_once(
                record=record,
                case=case,
                baseline=baseline,
                documents=documents,
                judge=judge,
            )
        except Exception as exc:
            last_exception = exc
            if attempt < max_attempts:
                time.sleep(min(2.0, 0.25 * attempt))
    metadata = dict(record.metadata)
    metadata["judge_metrics_status"] = "judge_failed"
    metadata["llm_judge_error"] = str(last_exception)
    metadata["llm_judge_attempts"] = max_attempts
    return record.model_copy(update={"metadata": metadata})


def _judge_record_once(
    *,
    record: BenchmarkRecord,
    case: CrudRagCase,
    baseline: BenchmarkRecord,
    documents: dict[str, EvidenceDocument],
    judge: LLMJudge,
) -> BenchmarkRecord:
    query_result = judge.judge_query(
        original_query=record.original_query,
        optimized_queries=record.retrieval_queries,
    )
    retrieval_result = judge.judge_retrieval(
        original_query=record.original_query,
        reference_answer=case.answer,
        baseline_queries=baseline.retrieval_queries,
        candidate_queries=record.retrieval_queries,
        baseline_contexts=_contexts_for_record(baseline, documents),
        candidate_contexts=_contexts_for_record(record, documents),
    )
    metadata = dict(record.metadata)
    metadata["judge_metrics_status"] = "judged"
    metadata.pop("llm_judge_error", None)
    metadata["llm_judge"] = {
        "query": query_result.model_dump(mode="json"),
        "retrieval": retrieval_result.model_dump(mode="json"),
    }
    return record.model_copy(
        update={
            "intent_preservation": score_to_unit(query_result.intent_preservation_score),
            "clarity_enhancement": score_to_unit(query_result.clarity_enhancement_score),
            "answer_preference": score_to_unit(retrieval_result.answer_preference_score),
            "faithfulness": score_to_unit(retrieval_result.faithfulness_score),
            "parse_ok": _parse_ok_after_judge_success(record),
            "metadata": metadata,
        }
    )


def _parse_ok_after_judge_success(record: BenchmarkRecord) -> bool:
    if record.parse_ok:
        return True
    metadata = record.metadata
    if metadata.get("answer_generation_status") == "generation_failed":
        return False
    if any(
        key in metadata for key in ("error", "provider_preflight_error", "answer_generation_error")
    ):
        return False
    if metadata.get("judge_metrics_status") == "judge_failed" or "llm_judge_error" in metadata:
        return True
    return record.parse_ok


def _document_map(
    *, cases: list[CrudRagCase], distractors: list[EvidenceDocument]
) -> dict[str, EvidenceDocument]:
    return {document.doc_id: document for document in build_index_documents(cases, distractors)}


def _contexts_for_record(
    record: BenchmarkRecord, documents: dict[str, EvidenceDocument]
) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    for doc_id in record.retrieved_doc_ids:
        document = documents.get(doc_id)
        if document is not None:
            contexts.append({"doc_id": document.doc_id, "text": document.text})
    return contexts


def _write_judge_manifest(output_dir: Path) -> None:
    manifest = {
        "judge_metrics": [
            "intent_preservation",
            "clarity_enhancement",
            "answer_preference",
            "faithfulness",
        ],
        "score_scale": "Stored metric values are normalized from judge integer scores 1-5 into 0.2-1.0.",
    }
    (output_dir / "judge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _record_key(record: BenchmarkRecord) -> tuple[str, str, str, str]:
    return (record.provider, record.model_id, record.case_id, record.method)


def _load_checkpoint_records(path: Path) -> list[BenchmarkRecord]:
    if not path.exists():
        return []
    records_by_key: dict[tuple[str, str, str, str], BenchmarkRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            record = BenchmarkRecord.model_validate_json(value)
            records_by_key[_record_key(record)] = record
    return list(records_by_key.values())


def _append_checkpoint_record(path: Path, record: BenchmarkRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
