from __future__ import annotations

import asyncio
import csv
import json
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from rag_practise.datasets import (
    CrudRagCase,
    EvidenceDocument,
    build_index_documents,
    load_compact_cases,
    load_compact_distractors,
)
from rag_practise.evaluation import gold_doc_hit_count, recall_at_k
from rag_practise.experiments.events import EventLogger
from rag_practise.experiments.events import EventLoggingLLMClient
from rag_practise.experiments.model_config import ModelConfig, build_llm_client, load_model_configs
from rag_practise.llm import ChatCompletionRequest, ChatMessage, LLMClient
from rag_practise.retrieval import FaissVectorIndex, HashEmbeddingClient, retrieve_with_rrf
from rag_practise.retrieval.index import EmbeddingProvider
from rag_practise.transforms import build_transform
from rag_practise.transforms.base import TransformResult

LLMClientFactory = Callable[[], LLMClient]


DEFAULT_TRANSFORMS = [
    "baseline",
    "rewrite",
    "expand",
    "multi_query",
    "child_query",
    "hyde",
    "step_back",
]


class ExperimentConfig(BaseModel):
    cases_path: Path
    distractors_path: Path
    output_dir: Path = Path("reports/mock-crud-rag-compact")
    transforms: list[str] = Field(default_factory=lambda: list(DEFAULT_TRANSFORMS))
    top_k: int = 5
    transform_model: str | None = None
    transform_model_id: str = "local_fallback"
    transform_provider: str = "local"
    transform_timeout_seconds: float = 45.0
    run_id: str = "mock-crud-rag-compact"


class BenchmarkRecord(BaseModel):
    run_id: str
    case_id: str
    provider: str
    model_id: str
    model: str | None
    method: str
    original_query: str
    optimized_query: str | None
    retrieval_queries: list[str]
    query_count: int
    retrieved_doc_ids: list[str]
    gold_doc_ids: list[str]
    intent_preservation: float
    clarity_enhancement: float
    recall_at_5: float
    gold_doc_hit_count: int
    answer_preference: float
    faithfulness: float
    generated_answer: str | None = None
    generation_model_id: str | None = None
    generation_provider: str | None = None
    generation_latency_ms: float = 0.0
    generation_estimated_cost_usd: float = 0.0
    generation_billed_cost_usd: float = 0.0
    total_latency_ms: float
    estimated_cost_usd: float
    billed_cost_usd: float
    parse_ok: bool
    transform_events: int
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    dataset = raw.get("dataset", {})
    benchmark = raw.get("benchmark", {})
    return ExperimentConfig(
        cases_path=Path(dataset.get("cases_output_path", "dataset/crud_rag_20_cases.jsonl")),
        distractors_path=Path(
            dataset.get("distractors_output_path", "dataset/crud_rag_100_distractors.jsonl")
        ),
        output_dir=Path(benchmark.get("output_dir", "reports/mock-crud-rag-compact")),
        transforms=list(raw.get("transforms", [])) or list(DEFAULT_TRANSFORMS),
        top_k=int(raw.get("retrieval", {}).get("top_k", 5)),
        transform_model=raw.get("transform_model"),
        run_id=str(benchmark.get("run_id", "mock-crud-rag-compact")),
    )


def run_benchmark(
    *,
    config: ExperimentConfig,
    llm: LLMClient | None = None,
    embedding_client: EmbeddingProvider | None = None,
    logger: EventLogger | None = None,
) -> list[BenchmarkRecord]:
    cases = load_compact_cases(config.cases_path)
    distractors = load_compact_distractors(config.distractors_path)
    return run_benchmark_from_data(
        config=config,
        cases=cases,
        distractors=distractors,
        llm=llm,
        embedding_client=embedding_client,
        logger=logger,
    )


def run_benchmark_from_data(
    *,
    config: ExperimentConfig,
    cases: list[CrudRagCase],
    distractors: list[EvidenceDocument],
    llm: LLMClient | None = None,
    embedding_client: EmbeddingProvider | None = None,
    logger: EventLogger | None = None,
) -> list[BenchmarkRecord]:
    logger = logger or EventLogger()
    embedding_client = embedding_client or HashEmbeddingClient()
    documents = build_index_documents(cases, distractors)
    vector_index = FaissVectorIndex.build(
        documents,
        embedding_client,
        provider_metadata=_embedding_provider_metadata(embedding_client),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    vector_index.write_metadata(config.output_dir / "index_metadata.json")

    records: list[BenchmarkRecord] = []
    for case in cases:
        for method in config.transforms:
            records.append(
                _run_case_method_record(
                    config=config,
                    case=case,
                    method=method,
                    vector_index=vector_index,
                    embedding_client=embedding_client,
                    llm=llm,
                    logger=logger,
                    use_hard_timeout=True,
                )
            )
    write_report_artifacts(config.output_dir, records)
    return records


async def run_benchmark_from_data_async(
    *,
    config: ExperimentConfig,
    cases: list[CrudRagCase],
    distractors: list[EvidenceDocument],
    llm_factory: LLMClientFactory | None = None,
    embedding_client: EmbeddingProvider | None = None,
    max_concurrency: int = 4,
) -> list[BenchmarkRecord]:
    embedding_client = embedding_client or HashEmbeddingClient()
    documents = build_index_documents(cases, distractors)
    vector_index = FaissVectorIndex.build(
        documents,
        embedding_client,
        provider_metadata=_embedding_provider_metadata(embedding_client),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    vector_index.write_metadata(config.output_dir / "index_metadata.json")

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _run_case_method_record_async(
            config=config,
            case=case,
            method=method,
            vector_index=vector_index,
            embedding_client=embedding_client,
            llm_factory=llm_factory,
            semaphore=semaphore,
        )
        for case in cases
        for method in config.transforms
    ]
    records = list(await asyncio.gather(*tasks))
    write_report_artifacts(config.output_dir, records)
    return records


def _run_case_method_record(
    *,
    config: ExperimentConfig,
    case: CrudRagCase,
    method: str,
    vector_index: FaissVectorIndex,
    embedding_client: EmbeddingProvider,
    llm: LLMClient | None,
    logger: EventLogger,
    use_hard_timeout: bool,
) -> BenchmarkRecord:
    transform = build_transform(method)
    active_logger = getattr(llm, "logger", logger)
    event_offset = len(active_logger.events)
    started = time.perf_counter()
    try:
        if use_hard_timeout:
            with _hard_timeout(config.transform_timeout_seconds):
                transform_result = transform.transform(
                    case.question,
                    llm=llm,
                    model=config.transform_model,
                )
        else:
            transform_result = transform.transform(
                case.question,
                llm=llm,
                model=config.transform_model,
            )
    except Exception as exc:
        transform_result = TransformResult(
            method=method,
            original_query=case.question,
            optimized_query=case.question,
            retrieval_queries=[case.question],
            parse_ok=False,
            metadata={"error": str(exc)},
        )
    latency_ms = (time.perf_counter() - started) * 1000
    events = active_logger.since(event_offset)
    return _record_from_transform_result(
        config=config,
        case=case,
        method=method,
        transform_result=transform_result,
        vector_index=vector_index,
        embedding_client=embedding_client,
        latency_ms=latency_ms,
        events=events,
    )


async def _run_case_method_record_async(
    *,
    config: ExperimentConfig,
    case: CrudRagCase,
    method: str,
    vector_index: FaissVectorIndex,
    embedding_client: EmbeddingProvider,
    llm_factory: LLMClientFactory | None,
    semaphore: asyncio.Semaphore,
) -> BenchmarkRecord:
    async with semaphore:
        logger = EventLogger()
        llm = llm_factory() if method != "baseline" and llm_factory is not None else None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _run_case_method_record,
                    config=config,
                    case=case,
                    method=method,
                    vector_index=vector_index,
                    embedding_client=embedding_client,
                    llm=llm,
                    logger=logger,
                    use_hard_timeout=False,
                ),
                timeout=config.transform_timeout_seconds + 5,
            )
        except Exception as exc:
            fallback = TransformResult(
                method=method,
                original_query=case.question,
                optimized_query=case.question,
                retrieval_queries=[case.question],
                parse_ok=False,
                metadata={"error": str(exc)},
            )
            return _record_from_transform_result(
                config=config,
                case=case,
                method=method,
                transform_result=fallback,
                vector_index=vector_index,
                embedding_client=embedding_client,
                latency_ms=(config.transform_timeout_seconds + 5) * 1000,
                events=[],
            )


def _record_from_transform_result(
    *,
    config: ExperimentConfig,
    case: CrudRagCase,
    method: str,
    transform_result: TransformResult,
    vector_index: FaissVectorIndex,
    embedding_client: EmbeddingProvider,
    latency_ms: float,
    events: list[Any],
) -> BenchmarkRecord:
    gold_doc_ids = {document.doc_id for document in case.gold_docs}
    retrieval_queries = transform_result.normalized_queries()
    hits = retrieve_with_rrf(
        queries=retrieval_queries,
        vector_index=vector_index,
        embedding_client=embedding_client,
        top_k=config.top_k,
    )
    hit_count = gold_doc_hit_count(hits, gold_doc_ids)
    recall = recall_at_k(hits, gold_doc_ids, k=config.top_k)
    return BenchmarkRecord(
        run_id=config.run_id,
        case_id=case.case_id,
        provider=config.transform_provider,
        model_id=config.transform_model_id,
        model=config.transform_model,
        method=method,
        original_query=case.question,
        optimized_query=transform_result.optimized_query,
        retrieval_queries=retrieval_queries,
        query_count=len(retrieval_queries),
        retrieved_doc_ids=[hit.doc_id for hit in hits],
        gold_doc_ids=sorted(gold_doc_ids),
        intent_preservation=0.0,
        clarity_enhancement=0.0,
        recall_at_5=recall,
        gold_doc_hit_count=hit_count,
        answer_preference=0.0,
        faithfulness=0.0,
        total_latency_ms=round(latency_ms, 4),
        estimated_cost_usd=round(sum(event.estimated_cost_usd for event in events), 8),
        billed_cost_usd=round(sum(event.billed_cost_usd for event in events), 8),
        parse_ok=transform_result.parse_ok,
        transform_events=len(events),
        metadata=_metadata_with_metric_source(transform_result.metadata),
    )


def _metadata_with_metric_source(metadata: dict[str, Any]) -> dict[str, Any]:
    output = dict(metadata)
    output.setdefault("judge_metrics_status", "not_judged")
    output.setdefault(
        "judge_metrics_note",
        "intent_preservation, clarity_enhancement, answer_preference, and faithfulness "
        "require experiment judge-records.",
    )
    return output


def run_model_matrix(
    *,
    config: ExperimentConfig,
    models_config_path: Path,
    embedding_client: EmbeddingProvider | None = None,
    output_dir: Path | None = None,
    resume: bool = True,
    rerun_model_ids: set[str] | None = None,
) -> list[BenchmarkRecord]:
    models = load_model_configs(models_config_path, role="transform")
    matrix_output_dir = output_dir or config.output_dir
    all_records: list[BenchmarkRecord] = []
    expected_records = _count_jsonl(config.cases_path) * len(config.transforms)
    rerun_model_ids = rerun_model_ids or set()
    for model_config in models:
        model_output_dir = matrix_output_dir / model_config.id
        existing_records_path = model_output_dir / "records.jsonl"
        if resume and model_config.id not in rerun_model_ids and existing_records_path.exists():
            existing_records = load_benchmark_records(existing_records_path)
            if len(existing_records) == expected_records:
                all_records.extend(existing_records)
                continue
        model_run_config = config.model_copy(
            update={
                "output_dir": model_output_dir,
                "run_id": f"{config.run_id}-{model_config.id}",
                "transform_model": model_config.model,
                "transform_model_id": model_config.id,
                "transform_provider": model_config.provider,
                "transform_timeout_seconds": model_config.timeout_seconds,
            }
        )
        logger = EventLogger()
        raw_client = build_llm_client(model_config)
        preflight_error = _preflight_transform_model(raw_client, model_config)
        if preflight_error is not None:
            unavailable_records = run_benchmark(
                config=model_run_config,
                llm=None,
                embedding_client=embedding_client,
                logger=logger,
            )
            unavailable_records = _mark_model_unavailable(unavailable_records, preflight_error)
            write_report_artifacts(model_output_dir, unavailable_records)
            all_records.extend(unavailable_records)
            continue
        client = EventLoggingLLMClient(
            raw_client,
            logger=logger,
            pricing=model_config.pricing_policy(),
        )
        all_records.extend(
            run_benchmark(
                config=model_run_config,
                llm=client,
                embedding_client=embedding_client,
                logger=logger,
            )
        )
    write_report_artifacts(matrix_output_dir, all_records)
    write_matrix_manifest(matrix_output_dir, models)
    return all_records


async def run_model_matrix_async(
    *,
    config: ExperimentConfig,
    models_config_path: Path,
    embedding_client: EmbeddingProvider | None = None,
    output_dir: Path | None = None,
    resume: bool = True,
    rerun_model_ids: set[str] | None = None,
    max_concurrency: int = 4,
) -> list[BenchmarkRecord]:
    models = load_model_configs(models_config_path, role="transform")
    matrix_output_dir = output_dir or config.output_dir
    all_records: list[BenchmarkRecord] = []
    expected_records = _count_jsonl(config.cases_path) * len(config.transforms)
    rerun_model_ids = rerun_model_ids or set()
    cases = load_compact_cases(config.cases_path)
    distractors = load_compact_distractors(config.distractors_path)

    for model_config in models:
        model_output_dir = matrix_output_dir / model_config.id
        existing_records_path = model_output_dir / "records.jsonl"
        if resume and model_config.id not in rerun_model_ids and existing_records_path.exists():
            existing_records = load_benchmark_records(existing_records_path)
            if len(existing_records) == expected_records:
                all_records.extend(existing_records)
                continue

        model_run_config = config.model_copy(
            update={
                "output_dir": model_output_dir,
                "run_id": f"{config.run_id}-{model_config.id}",
                "transform_model": model_config.model,
                "transform_model_id": model_config.id,
                "transform_provider": model_config.provider,
                "transform_timeout_seconds": model_config.timeout_seconds,
            }
        )
        preflight_error = _preflight_transform_model(build_llm_client(model_config), model_config)
        if preflight_error is not None:
            unavailable_records = await run_benchmark_from_data_async(
                config=model_run_config,
                cases=cases,
                distractors=distractors,
                llm_factory=None,
                embedding_client=embedding_client,
                max_concurrency=max_concurrency,
            )
            unavailable_records = _mark_model_unavailable(unavailable_records, preflight_error)
            write_report_artifacts(model_output_dir, unavailable_records)
            all_records.extend(unavailable_records)
            continue

        def llm_factory() -> LLMClient:
            return EventLoggingLLMClient(
                build_llm_client(model_config),
                logger=EventLogger(),
                pricing=model_config.pricing_policy(),
            )

        all_records.extend(
            await run_benchmark_from_data_async(
                config=model_run_config,
                cases=cases,
                distractors=distractors,
                llm_factory=llm_factory,
                embedding_client=embedding_client,
                max_concurrency=max_concurrency,
            )
        )

    write_report_artifacts(matrix_output_dir, all_records)
    write_matrix_manifest(matrix_output_dir, models)
    return all_records


def write_report_artifacts(output_dir: Path, records: list[BenchmarkRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    rows = [record.model_dump(mode="json") for record in records]
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(records)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "report.md").write_text(render_markdown_report(summary), encoding="utf-8")


def load_benchmark_records(path: Path) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                records.append(BenchmarkRecord.model_validate_json(value))
    return records


def write_matrix_manifest(output_dir: Path, models: list[ModelConfig]) -> None:
    manifest = {
        "models": [
            {
                "id": model.id,
                "provider": model.provider,
                "model": model.model,
                "benchmark_tier": model.benchmark_tier,
                "pricing_mode": model.pricing_mode,
            }
            for model in models
        ]
    }
    (output_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _preflight_transform_model(client: LLMClient, model_config: ModelConfig) -> str | None:
    try:
        with _hard_timeout(model_config.timeout_seconds):
            client.complete(
                ChatCompletionRequest(
                    model=model_config.model,
                    messages=[
                        ChatMessage(
                            role="system",
                            content="Return compact JSON only.",
                        ),
                        ChatMessage(
                            role="user",
                            content='Return exactly: {"ok": true}',
                        ),
                    ],
                    temperature=0,
                    max_tokens=32,
                )
            )
    except Exception as exc:
        return str(exc)
    return None


@contextmanager
def _hard_timeout(seconds: float):
    def handle_timeout(signum, frame):
        raise TimeoutError(f"preflight timed out after {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _mark_model_unavailable(
    records: list[BenchmarkRecord], preflight_error: str
) -> list[BenchmarkRecord]:
    output: list[BenchmarkRecord] = []
    for record in records:
        if record.method == "baseline":
            output.append(record)
            continue
        metadata = dict(record.metadata)
        metadata["provider_preflight_error"] = preflight_error
        output.append(record.model_copy(update={"parse_ok": False, "metadata": metadata}))
    return output


def _embedding_provider_metadata(embedding_client: EmbeddingProvider) -> dict[str, Any]:
    if isinstance(embedding_client, HashEmbeddingClient):
        return {"provider": "local_hash"}
    return {
        "provider": "openai_compatible",
        "base_url_env": "OMLX_HOST_URL",
        "api_key_env": "OMLX_API_KEY",
    }


def build_summary(records: list[BenchmarkRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([record.model_dump(mode="json") for record in records])
    group_keys = ["method"]
    if frame["provider"].nunique() > 1 or frame["model_id"].nunique() > 1:
        group_keys = ["provider", "model_id", "model", "method"]
    summary = (
        frame.groupby(group_keys, as_index=False)
        .agg(
            cases=("case_id", "count"),
            intent_preservation=("intent_preservation", "mean"),
            clarity_enhancement=("clarity_enhancement", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            gold_doc_hit_count=("gold_doc_hit_count", "mean"),
            answer_preference=("answer_preference", "mean"),
            faithfulness=("faithfulness", "mean"),
            generation_latency_ms=("generation_latency_ms", "mean"),
            generation_estimated_cost_usd=("generation_estimated_cost_usd", "sum"),
            generation_billed_cost_usd=("generation_billed_cost_usd", "sum"),
            total_latency_ms=("total_latency_ms", "mean"),
            estimated_cost_usd=("estimated_cost_usd", "sum"),
            billed_cost_usd=("billed_cost_usd", "sum"),
            parse_ok=("parse_ok", "mean"),
            query_count=("query_count", "mean"),
        )
        .sort_values(["recall_at_5", "gold_doc_hit_count", "answer_preference"], ascending=False)
    )
    numeric_columns = summary.select_dtypes(include="number").columns
    summary[numeric_columns] = summary[numeric_columns].round(4)
    return summary


def render_markdown_report(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# Query Transformation Benchmark\n\nNo records.\n"
    return (
        "# Query Transformation Benchmark\n\n"
        "Benchmark compare for baseline, rewrite, expand, multi_query, "
        "child_query, hyde, and step_back.\n\n"
        + _markdown_table(summary)
        + "\n"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = frame.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
