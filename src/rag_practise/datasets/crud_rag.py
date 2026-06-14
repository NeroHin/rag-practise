from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


CRUD_SPLIT_URL = (
    "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/data/"
    "crud_split/split_merged.json"
)


class EvidenceDocument(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrudRagCase(BaseModel):
    case_id: str
    question: str
    answer: str
    gold_docs: list[EvidenceDocument]
    source_task: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def download_crud_split(output_path: Path, *, url: str = CRUD_SPLIT_URL) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def prepare_crud_rag_compact(
    *,
    eval_source_path: Path,
    corpus_source_path: Path,
    cases_output_path: Path,
    distractors_output_path: Path,
    sample_size: int = 20,
    distractor_sample_size: int = 100,
    seed: int = 42,
) -> tuple[list[CrudRagCase], list[EvidenceDocument]]:
    cases = load_crud_cases(eval_source_path)
    rng = random.Random(seed)
    sampled_cases = rng.sample(cases, k=min(sample_size, len(cases)))
    distractors = sample_corpus_distractors(
        corpus_source_path, sample_size=distractor_sample_size, seed=seed
    )
    write_jsonl(cases_output_path, [case.model_dump(mode="json") for case in sampled_cases])
    write_jsonl(
        distractors_output_path,
        [document.model_dump(mode="json") for document in distractors],
    )
    return sampled_cases, distractors


def load_compact_cases(path: Path) -> list[CrudRagCase]:
    return [CrudRagCase.model_validate(row) for row in read_jsonl(path)]


def load_compact_distractors(path: Path) -> list[EvidenceDocument]:
    return [EvidenceDocument.model_validate(row) for row in read_jsonl(path)]


def load_crud_cases(path: Path) -> list[CrudRagCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    output: list[CrudRagCase] = []
    if isinstance(raw, dict):
        for task_name, items in raw.items():
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                case = _case_from_item(item, task_name=task_name, index=index)
                if case is not None:
                    output.append(case)
    elif isinstance(raw, list):
        for index, item in enumerate(raw):
            case = _case_from_item(item, task_name="unknown", index=index)
            if case is not None:
                output.append(case)
    return output


def sample_corpus_distractors(path: Path, *, sample_size: int, seed: int) -> list[EvidenceDocument]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    indexed = list(enumerate(lines))
    sample = rng.sample(indexed, k=min(sample_size, len(indexed)))
    return [
        EvidenceDocument(
            doc_id=f"distractor_{line_no}",
            text=text,
            metadata={"source_path": str(path), "line_no": line_no},
        )
        for line_no, text in sample
    ]


def build_index_documents(
    cases: list[CrudRagCase], distractors: list[EvidenceDocument]
) -> list[EvidenceDocument]:
    by_id: dict[str, EvidenceDocument] = {}
    for case in cases:
        for document in case.gold_docs:
            by_id[document.doc_id] = document
    for document in distractors:
        by_id[document.doc_id] = document
    return list(by_id.values())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                rows.append(json.loads(value))
    return rows


def _case_from_item(item: Any, *, task_name: str, index: int) -> CrudRagCase | None:
    if not isinstance(item, dict):
        return None
    question = _first_text(item, "question", "questions", "query", "Question", "q")
    answer = _first_text(item, "answer", "answers", "Answer", "a", "gold_answer")
    docs = _extract_gold_docs(item, task_name=task_name, index=index)
    if not question or not answer or not docs:
        return None
    case_id = str(item.get("id") or item.get("case_id") or f"{task_name}_{index}")
    return CrudRagCase(
        case_id=case_id,
        question=question,
        answer=answer,
        gold_docs=docs,
        source_task=task_name,
        metadata={"raw_keys": sorted(item.keys())},
    )


def _extract_gold_docs(item: dict[str, Any], *, task_name: str, index: int) -> list[EvidenceDocument]:
    docs: list[EvidenceDocument] = []
    for key, value in item.items():
        key_lower = key.lower()
        if not (
            key_lower.startswith("news")
            or key_lower.startswith("doc")
            or key_lower in {"context", "contexts", "documents"}
        ):
            continue
        if isinstance(value, str) and value.strip():
            docs.append(_document_from_text(value, task_name=task_name, index=index, key=key))
        elif isinstance(value, list):
            for nested_index, nested in enumerate(value):
                text = nested.get("text") if isinstance(nested, dict) else nested
                if isinstance(text, str) and text.strip():
                    docs.append(
                        _document_from_text(
                            text,
                            task_name=task_name,
                            index=index,
                            key=f"{key}_{nested_index}",
                        )
                    )
    return docs


def _document_from_text(text: str, *, task_name: str, index: int, key: str) -> EvidenceDocument:
    normalized = " ".join(text.split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return EvidenceDocument(
        doc_id=f"gold_{digest}",
        text=normalized,
        metadata={"source_task": task_name, "source_index": index, "source_key": key},
    )


def _first_text(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
