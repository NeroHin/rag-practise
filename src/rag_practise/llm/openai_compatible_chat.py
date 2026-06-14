from __future__ import annotations

from typing import Any

from openai import OpenAI

from rag_practise.llm.base import ChatCompletionRequest, ChatCompletionResult, TokenUsage


class OpenAICompatibleChatClient:
    """Chat client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 0,
        client: Any | None = None,
    ) -> None:
        self.provider = provider
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
        }
        if request.temperature is not None and not request.model.startswith("gpt-5"):
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            if request.model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = request.max_tokens
            else:
                kwargs["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return ChatCompletionResult(
            content=content,
            model=getattr(response, "model", request.model),
            provider=self.provider,
            usage=_extract_usage(response),
            raw=response,
        )


def _extract_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
