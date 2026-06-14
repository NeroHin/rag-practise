from __future__ import annotations

import json
import urllib.error
import urllib.request

from rag_practise.llm.base import ChatCompletionRequest, ChatCompletionResult, TokenUsage


class HttpOpenAICompatibleChatClient:
    """Small HTTP client for OpenAI-compatible providers where the SDK is problematic."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        payload = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format

        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider} HTTP {exc.code}: {body}") from exc

        content = raw["choices"][0]["message"].get("content") or ""
        usage = raw.get("usage") or {}
        return ChatCompletionResult(
            content=content,
            model=raw.get("model") or request.model,
            provider=self.provider,
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            raw=raw,
        )
