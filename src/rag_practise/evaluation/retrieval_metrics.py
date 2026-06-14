from __future__ import annotations

from rag_practise.retrieval import SearchHit


def gold_doc_hit_count(hits: list[SearchHit], gold_doc_ids: set[str]) -> int:
    return sum(1 for hit in hits if hit.doc_id in gold_doc_ids)


def recall_at_k(hits: list[SearchHit], gold_doc_ids: set[str], *, k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    top_ids = {hit.doc_id for hit in hits[:k]}
    return round(len(top_ids & gold_doc_ids) / len(gold_doc_ids), 4)

