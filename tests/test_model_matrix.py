from __future__ import annotations

import asyncio
from pathlib import Path

from rag_practise.datasets import CrudRagCase, EvidenceDocument
from rag_practise.experiments import ExperimentConfig
from rag_practise.experiments.model_config import load_model_configs
from rag_practise.experiments.runner import run_model_matrix, run_model_matrix_async
from rag_practise.llm import ChatCompletionRequest, ChatCompletionResult, TokenUsage
from rag_practise.retrieval import HashEmbeddingClient


class StaticClientFactory:
    provider = "fake"

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        return ChatCompletionResult(
            content='["台積電 第二季 營收", "台積電 毛利率"]',
            model=request.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )


def test_load_model_configs_filters_transform_role(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
models:
  - id: transform_model
    provider: openai
    model: test-transform
    roles: [transform]
  - id: generate_model
    provider: openai
    model: test-generate
    roles: [generate]
""",
        encoding="utf-8",
    )

    configs = load_model_configs(path)

    assert [config.id for config in configs] == ["transform_model"]


def test_run_model_matrix_writes_provider_model_summary(
    tmp_path: Path, monkeypatch
) -> None:
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        """
models:
  - id: fake_a
    provider: openai
    model: fake-a
    roles: [transform]
  - id: fake_b
    provider: openrouter
    model: fake-b
    roles: [transform]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rag_practise.experiments.runner.build_llm_client",
        lambda config: StaticClientFactory(),
    )
    case = CrudRagCase(
        case_id="case_1",
        question="台積電第二季營收是多少？",
        answer="4800 億元",
        source_task="fixture",
        gold_docs=[
            EvidenceDocument(
                doc_id="gold_tsmc",
                text="台積電第二季營收為 4800 億元，毛利率為 54%。",
            )
        ],
    )
    cases_path = tmp_path / "cases.jsonl"
    distractors_path = tmp_path / "distractors.jsonl"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    distractors_path.write_text(
        EvidenceDocument(doc_id="d1", text="Steam 遊戲折扣。").model_dump_json() + "\n",
        encoding="utf-8",
    )
    config = ExperimentConfig(
        cases_path=cases_path,
        distractors_path=distractors_path,
        output_dir=tmp_path / "matrix",
        transforms=["baseline", "multi_query"],
        run_id="matrix-test",
    )

    records = run_model_matrix(
        config=config,
        models_config_path=models_path,
        embedding_client=HashEmbeddingClient(dimensions=128),
        output_dir=tmp_path / "matrix",
    )

    assert len(records) == 4
    assert {record.model_id for record in records} == {"fake_a", "fake_b"}
    summary = (tmp_path / "matrix" / "summary.csv").read_text(encoding="utf-8")
    assert "provider,model_id,model,method" in summary
    assert (tmp_path / "matrix" / "matrix_manifest.json").exists()


def test_run_model_matrix_async_writes_provider_model_summary(
    tmp_path: Path, monkeypatch
) -> None:
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        """
models:
  - id: fake_a
    provider: openai
    model: fake-a
    roles: [transform]
  - id: fake_b
    provider: openrouter
    model: fake-b
    roles: [transform]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rag_practise.experiments.runner.build_llm_client",
        lambda config: StaticClientFactory(),
    )
    case = CrudRagCase(
        case_id="case_1",
        question="台積電第二季營收是多少？",
        answer="4800 億元",
        source_task="fixture",
        gold_docs=[
            EvidenceDocument(
                doc_id="gold_tsmc",
                text="台積電第二季營收為 4800 億元，毛利率為 54%。",
            )
        ],
    )
    cases_path = tmp_path / "cases.jsonl"
    distractors_path = tmp_path / "distractors.jsonl"
    cases_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    distractors_path.write_text(
        EvidenceDocument(doc_id="d1", text="Steam 遊戲折扣。").model_dump_json() + "\n",
        encoding="utf-8",
    )
    config = ExperimentConfig(
        cases_path=cases_path,
        distractors_path=distractors_path,
        output_dir=tmp_path / "matrix_async",
        transforms=["baseline", "multi_query"],
        run_id="matrix-async-test",
    )

    records = asyncio.run(
        run_model_matrix_async(
            config=config,
            models_config_path=models_path,
            embedding_client=HashEmbeddingClient(dimensions=128),
            output_dir=tmp_path / "matrix_async",
            max_concurrency=2,
        )
    )

    assert len(records) == 4
    assert {record.model_id for record in records} == {"fake_a", "fake_b"}
    summary = (tmp_path / "matrix_async" / "summary.csv").read_text(encoding="utf-8")
    assert "provider,model_id,model,method" in summary
    assert (tmp_path / "matrix_async" / "matrix_manifest.json").exists()
