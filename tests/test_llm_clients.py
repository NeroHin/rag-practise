from __future__ import annotations

from types import SimpleNamespace

from rag_practise.llm import (
    ChatCompletionRequest,
    ChatMessage,
    GeminiChatClient,
    HttpOpenAICompatibleChatClient,
    OpenAICompatibleChatClient,
)


class FakeOpenAICompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="mock openai-compatible reply"))
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5, total_tokens=12),
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions())


class FakeGeminiModels:
    def __init__(self) -> None:
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="mock gemini reply")


class FakeGeminiClient:
    def __init__(self) -> None:
        self.models = FakeGeminiModels()


def test_openai_compatible_chat_client_uses_injected_client() -> None:
    fake_client = FakeOpenAIClient()
    client = OpenAICompatibleChatClient(
        provider="openrouter",
        api_key="test-key",
        base_url="https://example.test/v1",
        client=fake_client,
    )
    request = ChatCompletionRequest(
        model="openrouter/test-model",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.1,
        max_tokens=32,
    )

    result = client.complete(request)

    assert result.content == "mock openai-compatible reply"
    assert result.provider == "openrouter"
    assert result.model == "openrouter/test-model"
    assert result.usage.prompt_tokens == 7
    assert fake_client.chat.completions.calls == [
        {
            "model": "openrouter/test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.1,
            "max_tokens": 32,
        }
    ]


def test_openai_compatible_chat_client_uses_max_completion_tokens_for_gpt5() -> None:
    fake_client = FakeOpenAIClient()
    client = OpenAICompatibleChatClient(
        provider="openai",
        api_key="test-key",
        client=fake_client,
    )

    client.complete(
        ChatCompletionRequest(
            model="gpt-5-mini-2025-08-07",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0,
            max_tokens=32,
        )
    )

    assert fake_client.chat.completions.calls[0]["max_completion_tokens"] == 32
    assert "max_tokens" not in fake_client.chat.completions.calls[0]
    assert "temperature" not in fake_client.chat.completions.calls[0]


def test_openai_compatible_chat_client_passes_response_format() -> None:
    fake_client = FakeOpenAIClient()
    client = OpenAICompatibleChatClient(
        provider="openai",
        api_key="test-key",
        client=fake_client,
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "test_schema", "schema": {"type": "object"}},
    }

    client.complete(
        ChatCompletionRequest(
            model="gpt-4.1-nano-2025-04-14",
            messages=[ChatMessage(role="user", content="hello")],
            response_format=response_format,
        )
    )

    assert fake_client.chat.completions.calls[0]["response_format"] == response_format


def test_gemini_chat_client_uses_injected_client() -> None:
    fake_client = FakeGeminiClient()
    client = GeminiChatClient(api_key="test-key", client=fake_client)
    request = ChatCompletionRequest(
        model="gemini-test-model",
        messages=[
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="hello"),
        ],
    )

    result = client.complete(request)

    assert result.content == "mock gemini reply"
    assert result.provider == "gemini"
    assert fake_client.models.calls == [
        {
            "model": "gemini-test-model",
            "contents": "system: You are concise.\n\nuser: hello",
        }
    ]


def test_http_openai_compatible_chat_client_parses_response(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"model":"qwen/test","choices":[{"message":{"content":"ok"}}],'
                b'"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}'
            )

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = HttpOpenAICompatibleChatClient(
        provider="openrouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        timeout=12,
    )

    result = client.complete(
        ChatCompletionRequest(
            model="qwen/test",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0,
            max_tokens=8,
        )
    )

    assert result.content == "ok"
    assert result.provider == "openrouter"
    assert result.usage.total_tokens == 5
    assert calls[0][1] == 12
