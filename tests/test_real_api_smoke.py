from __future__ import annotations

import os

import pytest

from rag_practise.llm import ChatCompletionRequest, ChatMessage, GeminiChatClient
from rag_practise.llm.openai_compatible_chat import OpenAICompatibleChatClient


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_API_TESTS") != "1",
    reason="set RUN_REAL_API_TESTS=1 to run live provider smoke tests",
)


SMOKE_MESSAGES = [
    ChatMessage(
        role="system",
        content="You are a smoke-test endpoint. Return compact JSON only.",
    ),
    ChatMessage(
        role="user",
        content='Return exactly this JSON shape with your provider name: {"ok": true, "provider": "..."}',
    ),
]


@pytest.mark.parametrize(
    ("provider", "model", "api_key_env", "base_url"),
    [
        ("openai", "gpt-4.1-nano-2025-04-14", "OPENAI_API_KEY", None),
        (
            "openrouter",
            "qwen/qwen-2.5-7b-instruct",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
        ),
        (
            "nvidia_nim",
            "meta/llama-4-maverick-17b-128e-instruct",
            "NVIDIA_API_KEY",
            "https://integrate.api.nvidia.com/v1",
        ),
    ],
)
def test_openai_compatible_real_api_smoke(
    provider: str, model: str, api_key_env: str, base_url: str | None
) -> None:
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} is not set")
    client = OpenAICompatibleChatClient(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )

    result = client.complete(
        ChatCompletionRequest(
            model=model,
            messages=SMOKE_MESSAGES,
            temperature=0,
            max_tokens=64,
        )
    )

    assert result.provider == provider
    assert result.content.strip()


def test_gemini_real_api_smoke() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not set")
    client = GeminiChatClient(api_key=api_key)

    result = client.complete(
        ChatCompletionRequest(
            model="gemini-3.1-flash-lite",
            messages=SMOKE_MESSAGES,
            temperature=0,
            max_tokens=64,
        )
    )

    assert result.provider == "gemini"
    assert result.content.strip()
