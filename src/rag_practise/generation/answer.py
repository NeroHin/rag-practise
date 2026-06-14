from __future__ import annotations

from rag_practise.llm import ChatCompletionRequest, ChatMessage, LLMClient
from rag_practise.retrieval import SearchHit


def generate_answer(
    *,
    llm: LLMClient,
    model: str,
    question: str,
    contexts: list[SearchHit],
    max_context_chars: int = 3000,
) -> str:
    context_text = "\n\n".join(
        f"[{index}] {hit.text}" for index, hit in enumerate(contexts, start=1)
    )[:max_context_chars]
    request = ChatCompletionRequest(
        model=model,
        messages=[
            ChatMessage(
                role="system",
                content="Answer using only the provided context. If unsupported, say it is unknown.",
            ),
            ChatMessage(role="user", content=f"Question: {question}\n\nContext:\n{context_text}"),
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return llm.complete(request).content.strip()

