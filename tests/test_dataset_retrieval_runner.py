from __future__ import annotations

import json
from pathlib import Path

from rag_practise.datasets import (
    CrudRagCase,
    EvidenceDocument,
    build_index_documents,
    load_compact_cases,
    prepare_crud_rag_compact,
)
from rag_practise.evaluation import recall_at_k
from rag_practise.experiments import ExperimentConfig, run_benchmark_from_data
from rag_practise.retrieval import FaissVectorIndex, HashEmbeddingClient, retrieve_with_rrf


def test_crud_rag_compact_loader_writes_schema(tmp_path: Path) -> None:
    source = tmp_path / "split.json"
    corpus = tmp_path / "corpus.txt"
    source.write_text(
        json.dumps(
            {
                "questanswer_1doc": [
                    {
                        "question": "台積電第二季營收是多少？",
                        "answer": "台積電第二季營收為 4800 億元。",
                        "news1": "台積電第二季營收為 4800 億元，毛利率為 54%。",
                    },
                    {
                        "question": "京東第二季營收是多少？",
                        "answer": "京東第二季營收為 2879 億元。",
                        "news1": "京東集團第二季營收為 2879 億元，服務收入成長。",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    corpus.write_text("無關文件一\n無關文件二\n", encoding="utf-8")

    cases, distractors = prepare_crud_rag_compact(
        eval_source_path=source,
        corpus_source_path=corpus,
        cases_output_path=tmp_path / "cases.jsonl",
        distractors_output_path=tmp_path / "distractors.jsonl",
        sample_size=1,
        distractor_sample_size=2,
        seed=7,
    )

    loaded = load_compact_cases(tmp_path / "cases.jsonl")
    assert len(cases) == 1
    assert len(loaded) == 1
    assert loaded[0].gold_docs[0].doc_id.startswith("gold_")
    assert len(distractors) == 2


def test_hash_faiss_retriever_hits_gold_doc() -> None:
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
    distractors = [
        EvidenceDocument(doc_id="d1", text="Steam 遊戲最低價格門檻更新。"),
        EvidenceDocument(doc_id="d2", text="Valve 發布遊戲折扣政策。"),
    ]
    embedding = HashEmbeddingClient(dimensions=128)
    index = FaissVectorIndex.build(build_index_documents([case], distractors), embedding)

    hits = retrieve_with_rrf(
        queries=["台積電 第二季 營收"],
        vector_index=index,
        embedding_client=embedding,
        top_k=2,
    )

    assert hits[0].doc_id == "gold_tsmc"
    assert recall_at_k(hits, {"gold_tsmc"}, k=2) == 1.0


def test_benchmark_compare_writes_records_summary_and_report(tmp_path: Path) -> None:
    cases = [
        CrudRagCase(
            case_id="case_tsmc",
            question="台積電第二季營收是多少？",
            answer="4800 億元",
            source_task="fixture",
            gold_docs=[
                EvidenceDocument(
                    doc_id="gold_tsmc",
                    text="台積電第二季營收為 4800 億元，毛利率為 54%。",
                )
            ],
        ),
        CrudRagCase(
            case_id="case_jd",
            question="京東第二季營收是多少？",
            answer="2879 億元",
            source_task="fixture",
            gold_docs=[
                EvidenceDocument(
                    doc_id="gold_jd",
                    text="京東集團第二季營收為 2879 億元，服務收入成長。",
                )
            ],
        ),
    ]
    distractors = [
        EvidenceDocument(doc_id="d1", text="Steam 最低價格門檻更新。"),
        EvidenceDocument(doc_id="d2", text="英特爾 XeSS 版本更新。"),
    ]
    config = ExperimentConfig(
        cases_path=tmp_path / "unused_cases.jsonl",
        distractors_path=tmp_path / "unused_distractors.jsonl",
        output_dir=tmp_path / "report",
        run_id="test-run",
    )

    records = run_benchmark_from_data(
        config=config,
        cases=cases,
        distractors=distractors,
        embedding_client=HashEmbeddingClient(dimensions=128),
    )

    methods = {record.method for record in records}
    assert methods == {
        "baseline",
        "rewrite",
        "expand",
        "multi_query",
        "child_query",
        "hyde",
        "step_back",
    }
    assert len(records) == 14
    assert all(record.intent_preservation == 0.0 for record in records)
    assert all(record.clarity_enhancement == 0.0 for record in records)
    assert all(record.answer_preference == 0.0 for record in records)
    assert all(record.faithfulness == 0.0 for record in records)
    assert all(record.metadata["judge_metrics_status"] == "not_judged" for record in records)
    assert (tmp_path / "report" / "records.jsonl").exists()
    assert (tmp_path / "report" / "summary.csv").exists()
    assert (tmp_path / "report" / "report.md").read_text(encoding="utf-8").startswith(
        "# Query Transformation Benchmark"
    )
