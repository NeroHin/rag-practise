from __future__ import annotations

from typing import Any

from google import genai

from rag_practise.llm.base import ChatCompletionRequest, ChatCompletionResult


class GeminiChatClient:
    """Minimal Gemini chat adapter with an injectable SDK-like client."""

    def __init__(self, *, api_key: str, provider: str = "gemini", client: Any | None = None) -> None:
        self.provider = provider
        self._client = client or genai.Client(api_key=api_key)

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        response = self._client.models.generate_content(
            model=request.model,
            contents=_to_gemini_contents(request.messages),
        )
        return ChatCompletionResult(
            content=getattr(response, "text", "") or "",
            model=request.model,
            provider=self.provider,
            raw=response,
        )


def _to_gemini_contents(messages: list[Any]) -> str:
    return "\n\n".join(f"{message.role}: {message.content}" for message in messages)
