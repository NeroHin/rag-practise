from __future__ import annotations

from rag_practise.retrieval.index import EmbeddingProvider, FaissVectorIndex, SearchHit


def retrieve_with_rrf(
    *,
    queries: list[str],
    vector_index: FaissVectorIndex,
    embedding_client: EmbeddingProvider,
    top_k: int = 5,
    per_query_k: int | None = None,
    rrf_k: int = 60,
) -> list[SearchHit]:
    per_query_k = per_query_k or top_k
    scores: dict[str, float] = {}
    hits_by_id: dict[str, SearchHit] = {}
    for query in queries:
        for hit in vector_index.search(query, embedding_client, top_k=per_query_k):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + hit.rank)
            hits_by_id[hit.doc_id] = hit

    ordered_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]
    merged: list[SearchHit] = []
    for rank, doc_id in enumerate(ordered_ids, start=1):
        hit = hits_by_id[doc_id]
        merged.append(hit.model_copy(update={"rank": rank, "score": scores[doc_id]}))
    return merged

