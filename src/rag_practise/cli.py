from __future__ import annotations

import asyncio
import os
from pathlib import Path

from rich.console import Console
import typer

from rag_practise.datasets import (
    build_index_documents,
    download_crud_split,
    load_compact_cases,
    load_compact_distractors,
    prepare_crud_rag_compact,
)
from rag_practise.evaluation import LLMJudge
from rag_practise.experiments import (
    generate_answers_for_records,
    generate_answers_for_records_async,
    judge_benchmark_records,
    judge_benchmark_records_async,
    load_experiment_config,
    run_benchmark,
    run_model_matrix,
    run_model_matrix_async,
)
from rag_practise.experiments.model_config import build_llm_client, load_model_configs
from rag_practise.llm import OpenAICompatibleEmbeddingClient
from rag_practise.retrieval import FaissVectorIndex, HashEmbeddingClient

app = typer.Typer(help="RAG query transformation experiment CLI.")
datasets_app = typer.Typer(help="Dataset download and preparation commands.")
index_app = typer.Typer(help="Vector index commands.")
experiment_app = typer.Typer(help="Experiment runner commands.")
console = Console()


@app.callback()
def main() -> None:
    """Run dataset, indexing, experiment, and report commands."""


@datasets_app.command("download")
def datasets_download(
    name: str = typer.Argument(..., help="Dataset name. Currently only crud-rag is supported."),
    output: Path = typer.Option(Path("dataset/crud_split_merged.json"), help="Output JSON path."),
) -> None:
    if name != "crud-rag":
        raise typer.BadParameter("Only crud-rag is supported.")
    path = download_crud_split(output)
    console.print(f"Downloaded CRUD-RAG split to {path}")


@datasets_app.command("prepare")
def datasets_prepare(
    name: str = typer.Argument(..., help="Dataset name. Currently only crud-rag is supported."),
    eval_source_path: Path = typer.Option(Path("dataset/crud_split_merged.json")),
    corpus_source_path: Path = typer.Option(Path("dataset/documents_dup_part_10_part_1")),
    cases_output_path: Path = typer.Option(Path("dataset/crud_rag_20_cases.jsonl")),
    distractors_output_path: Path = typer.Option(Path("dataset/crud_rag_100_distractors.jsonl")),
    sample_size: int = typer.Option(20, min=1),
    distractor_sample_size: int = typer.Option(100, min=1),
    seed: int = typer.Option(42),
) -> None:
    if name != "crud-rag":
        raise typer.BadParameter("Only crud-rag is supported.")
    cases, distractors = prepare_crud_rag_compact(
        eval_source_path=eval_source_path,
        corpus_source_path=corpus_source_path,
        cases_output_path=cases_output_path,
        distractors_output_path=distractors_output_path,
        sample_size=sample_size,
        distractor_sample_size=distractor_sample_size,
        seed=seed,
    )
    console.print(
        f"Prepared {len(cases)} cases and {len(distractors)} distractors: "
        f"{cases_output_path}, {distractors_output_path}"
    )


@index_app.command("build")
def index_build(
    config: Path = typer.Option(Path("configs/experiment.crud-rag.yaml"), help="Experiment config."),
    embedding_provider: str = typer.Option("hash", help="hash or omlx."),
    embedding_model: str = typer.Option("local-embedding-model", help="Embedding model id."),
) -> None:
    experiment_config = load_experiment_config(config)
    cases = load_compact_cases(experiment_config.cases_path)
    distractors = load_compact_distractors(experiment_config.distractors_path)
    documents = build_index_documents(cases, distractors)
    embedding_client = _build_embedding_client(embedding_provider, embedding_model)
    provider_metadata = (
        {
            "provider": "openai_compatible",
            "base_url_env": "OMLX_HOST_URL",
            "api_key_env": "OMLX_API_KEY",
        }
        if embedding_provider == "omlx"
        else {"provider": "local_hash"}
    )
    vector_index = FaissVectorIndex.build(
        documents,
        embedding_client,
        provider_metadata=provider_metadata,
    )
    experiment_config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = experiment_config.output_dir / "index_metadata.json"
    vector_index.write_metadata(metadata_path)
    console.print(f"Built FAISS index metadata for {len(documents)} docs: {metadata_path}")


@experiment_app.command("run")
def experiment_run(
    config: Path = typer.Option(Path("configs/experiment.crud-rag.yaml"), help="Experiment config."),
    embedding_provider: str = typer.Option("hash", help="hash or omlx."),
    embedding_model: str = typer.Option("local-embedding-model", help="Embedding model id."),
) -> None:
    experiment_config = load_experiment_config(config)
    embedding_client = _build_embedding_client(embedding_provider, embedding_model)
    records = run_benchmark(config=experiment_config, embedding_client=embedding_client)
    console.print(
        f"Ran {len(records)} benchmark records. Artifacts: {experiment_config.output_dir}"
    )


@experiment_app.command("run-matrix")
def experiment_run_matrix(
    config: Path = typer.Option(Path("configs/experiment.crud-rag.yaml"), help="Experiment config."),
    models_config: Path = typer.Option(Path("configs/models.yaml"), help="Models config."),
    output_dir: Path = typer.Option(Path("reports/model-matrix-crud-rag-compact")),
    embedding_provider: str = typer.Option("hash", help="hash or omlx."),
    embedding_model: str = typer.Option("Qwen3-Embedding-0.6B-4bit-DWQ", help="Embedding model id."),
    resume: bool = typer.Option(True, help="Reuse completed per-model records when present."),
    rerun_model_id: list[str] = typer.Option(
        [],
        help="Model id to rerun even when completed records exist. Repeat for multiple models.",
    ),
) -> None:
    experiment_config = load_experiment_config(config).model_copy(
        update={"output_dir": output_dir, "run_id": "model-matrix-crud-rag-compact"}
    )
    embedding_client = _build_embedding_client(embedding_provider, embedding_model)
    records = run_model_matrix(
        config=experiment_config,
        models_config_path=models_config,
        embedding_client=embedding_client,
        output_dir=output_dir,
        resume=resume,
        rerun_model_ids=set(rerun_model_id),
    )
    console.print(f"Ran {len(records)} matrix benchmark records. Artifacts: {output_dir}")


@experiment_app.command("run-matrix-async")
def experiment_run_matrix_async(
    config: Path = typer.Option(Path("configs/experiment.crud-rag.yaml"), help="Experiment config."),
    models_config: Path = typer.Option(Path("configs/models.yaml"), help="Models config."),
    output_dir: Path = typer.Option(Path("reports/model-matrix-crud-rag-compact-async")),
    embedding_provider: str = typer.Option("hash", help="hash or omlx."),
    embedding_model: str = typer.Option("Qwen3-Embedding-0.6B-4bit-DWQ", help="Embedding model id."),
    resume: bool = typer.Option(True, help="Reuse completed per-model records when present."),
    rerun_model_id: list[str] = typer.Option(
        [],
        help="Model id to rerun even when completed records exist. Repeat for multiple models.",
    ),
    max_concurrency: int = typer.Option(4, min=1, help="Maximum concurrent transform calls."),
    generate: bool = typer.Option(False, "--generate", help="Generate answers after matrix records."),
    generation_model_id: str = typer.Option("openai_fast", help="Generate model config id."),
    generation_max_concurrency: int = typer.Option(
        4, min=1, help="Maximum concurrent answer generations."
    ),
    generation_output_dir: Path | None = typer.Option(
        None,
        help="Generated artifacts directory. Defaults to output-dir and overwrites aggregate report.",
    ),
    judge: bool = typer.Option(False, "--judge", help="Run LLM-as-a-Judge after matrix records."),
    judge_model_id: str = typer.Option("openai_judge_gpt5_mini", help="Judge model config id."),
    judge_max_concurrency: int = typer.Option(4, min=1, help="Maximum concurrent judge records."),
    judge_resume: bool = typer.Option(True, help="Resume judge records from checkpoint."),
    judge_retry_failed: bool = typer.Option(False, help="Retry failed judge checkpoint records."),
    judge_max_attempts: int = typer.Option(1, min=1, help="Judge attempts per record."),
    judge_output_dir: Path | None = typer.Option(
        None,
        help="Judged artifacts directory. Defaults to output-dir and overwrites aggregate report.",
    ),
) -> None:
    experiment_config = load_experiment_config(config).model_copy(
        update={"output_dir": output_dir, "run_id": "model-matrix-crud-rag-compact-async"}
    )
    embedding_client = _build_embedding_client(embedding_provider, embedding_model)
    records = asyncio.run(
        run_model_matrix_async(
            config=experiment_config,
            models_config_path=models_config,
            embedding_client=embedding_client,
            output_dir=output_dir,
            resume=resume,
            rerun_model_ids=set(rerun_model_id),
            max_concurrency=max_concurrency,
        )
    )
    console.print(
        f"Ran {len(records)} async matrix benchmark records. Artifacts: {output_dir}"
    )
    records_path = output_dir / "records.jsonl"
    if generate:
        generation_config = _load_model_config_by_id(
            models_config, generation_model_id, role="generate"
        )
        generated_output_dir = generation_output_dir or output_dir
        generated_records = asyncio.run(
            generate_answers_for_records_async(
                records_path=records_path,
                cases_path=experiment_config.cases_path,
                distractors_path=experiment_config.distractors_path,
                output_dir=generated_output_dir,
                client_factory=lambda: build_llm_client(generation_config),
                model_config=generation_config,
                max_concurrency=generation_max_concurrency,
            )
        )
        records_path = generated_output_dir / "records.jsonl"
        console.print(
            f"Generated answers for {len(generated_records)} async matrix records. "
            f"Artifacts: {generated_output_dir}"
        )
    if judge:
        judge_config = _load_model_config_by_id(models_config, judge_model_id)
        judged_output_dir = judge_output_dir or output_dir
        judged_records = asyncio.run(
            judge_benchmark_records_async(
                records_path=records_path,
                cases_path=experiment_config.cases_path,
                distractors_path=experiment_config.distractors_path,
                output_dir=judged_output_dir,
                judge_factory=lambda: LLMJudge(
                    client=build_llm_client(judge_config), model=judge_config.model
                ),
                max_concurrency=judge_max_concurrency,
                resume=judge_resume,
                retry_failed=judge_retry_failed,
                max_attempts=judge_max_attempts,
            )
        )
        console.print(
            f"Judged {len(judged_records)} async matrix records. Artifacts: {judged_output_dir}"
        )


@experiment_app.command("generate-answers")
def experiment_generate_answers(
    records_path: Path = typer.Option(
        Path("reports/model-matrix-crud-rag-compact-async/records.jsonl"),
        help="Benchmark records JSONL with retrieved_doc_ids.",
    ),
    config: Path = typer.Option(Path("configs/experiment.crud-rag.yaml"), help="Experiment config."),
    models_config: Path = typer.Option(Path("configs/models.yaml"), help="Models config."),
    generation_model_id: str = typer.Option("openai_fast", help="Generate model config id."),
    output_dir: Path = typer.Option(Path("reports/model-matrix-crud-rag-compact-async-generated")),
    limit: int | None = typer.Option(None, min=1, help="Optional max records for smoke runs."),
    max_concurrency: int = typer.Option(1, min=1, help="Maximum concurrent answer generations."),
) -> None:
    experiment_config = load_experiment_config(config)
    generation_config = _load_model_config_by_id(models_config, generation_model_id, role="generate")
    if max_concurrency == 1:
        records = generate_answers_for_records(
            records_path=records_path,
            cases_path=experiment_config.cases_path,
            distractors_path=experiment_config.distractors_path,
            output_dir=output_dir,
            llm=build_llm_client(generation_config),
            model_config=generation_config,
            limit=limit,
        )
    else:
        records = asyncio.run(
            generate_answers_for_records_async(
                records_path=records_path,
                cases_path=experiment_config.cases_path,
                distractors_path=experiment_config.distractors_path,
                output_dir=output_dir,
                client_factory=lambda: build_llm_client(generation_config),
                model_config=generation_config,
                limit=limit,
                max_concurrency=max_concurrency,
            )
        )
    console.print(f"Generated answers for {len(records)} records. Artifacts: {output_dir}")


@experiment_app.command("judge-records")
def experiment_judge_records(
    records_path: Path = typer.Option(
        Path("reports/model-matrix-crud-rag-compact-async/records.jsonl"),
        help="Benchmark records JSONL to judge.",
    ),
    config: Path = typer.Option(Path("configs/experiment.crud-rag.yaml"), help="Experiment config."),
    models_config: Path = typer.Option(Path("configs/models.yaml"), help="Models config."),
    judge_model_id: str = typer.Option("openai_judge_gpt5_mini", help="Judge model config id."),
    output_dir: Path = typer.Option(Path("reports/model-matrix-crud-rag-compact-async-judged")),
    limit: int | None = typer.Option(None, min=1, help="Optional max records for smoke runs."),
    max_concurrency: int = typer.Option(1, min=1, help="Maximum concurrent judge records."),
    resume: bool = typer.Option(True, help="Resume async judge records from checkpoint."),
    retry_failed: bool = typer.Option(False, help="Retry failed judge checkpoint records."),
    max_attempts: int = typer.Option(1, min=1, help="Judge attempts per record."),
) -> None:
    experiment_config = load_experiment_config(config)
    judge_config = _load_model_config_by_id(models_config, judge_model_id)
    if max_concurrency == 1:
        judge = LLMJudge(client=build_llm_client(judge_config), model=judge_config.model)
        records = judge_benchmark_records(
            records_path=records_path,
            cases_path=experiment_config.cases_path,
            distractors_path=experiment_config.distractors_path,
            output_dir=output_dir,
            judge=judge,
            limit=limit,
            max_attempts=max_attempts,
        )
    else:
        records = asyncio.run(
            judge_benchmark_records_async(
                records_path=records_path,
                cases_path=experiment_config.cases_path,
                distractors_path=experiment_config.distractors_path,
                output_dir=output_dir,
                judge_factory=lambda: LLMJudge(
                    client=build_llm_client(judge_config), model=judge_config.model
                ),
                limit=limit,
                max_concurrency=max_concurrency,
                resume=resume,
                retry_failed=retry_failed,
                max_attempts=max_attempts,
            )
        )
    console.print(f"Judged {len(records)} records. Artifacts: {output_dir}")


def _build_embedding_client(provider: str, model_id: str):
    if provider == "hash":
        return HashEmbeddingClient(model_id="hash-embedding")
    if provider == "omlx":
        api_key = os.environ.get("OMLX_API_KEY")
        base_url = os.environ.get("OMLX_HOST_URL")
        if not api_key or not base_url:
            raise typer.BadParameter("OMLX_API_KEY and OMLX_HOST_URL are required for omlx.")
        return OpenAICompatibleEmbeddingClient(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
        )
    raise typer.BadParameter("embedding_provider must be hash or omlx.")


def _load_model_config_by_id(models_config: Path, model_id: str, *, role: str = "judge"):
    for config in load_model_configs(models_config, role=role):
        if config.id == model_id:
            return config
    raise typer.BadParameter(f"Unknown {role} model id: {model_id}")


app.add_typer(datasets_app, name="datasets")
app.add_typer(index_app, name="index")
app.add_typer(experiment_app, name="experiment")


if __name__ == "__main__":
    app()
