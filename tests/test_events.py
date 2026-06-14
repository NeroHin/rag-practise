from __future__ import annotations

from rag_practise.experiments.events import EventLogger, EventLoggingLLMClient, PricingPolicy
from rag_practise.llm import ChatCompletionRequest, ChatCompletionResult, ChatMessage, LLMClient, TokenUsage


class CostedFakeLLM(LLMClient):
    provider = "fake"

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        return ChatCompletionResult(
            content="ok",
            model=request.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )


def test_event_logging_llm_client_records_latency_tokens_and_cost() -> None:
    logger = EventLogger()
    client = EventLoggingLLMClient(
        CostedFakeLLM(),
        logger=logger,
        pricing=PricingPolicy(input_usd_per_1k_tokens=0.001, output_usd_per_1k_tokens=0.002),
    )

    result = client.complete(
        ChatCompletionRequest(
            model="fake-model",
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert result.content == "ok"
    assert len(logger.events) == 1
    event = logger.events[0]
    assert event.status == "ok"
    assert event.prompt_tokens == 100
    assert event.completion_tokens == 50
    assert event.estimated_cost_usd == 0.0002


def test_event_logging_uses_raw_usage_cost_when_pricing_is_zero() -> None:
    class RawCostFakeLLM(LLMClient):
        provider = "openrouter"

        def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
            return ChatCompletionResult(
                content="ok",
                model=request.model,
                provider=self.provider,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                raw={"usage": {"cost": 0.000003}},
            )

    logger = EventLogger()
    client = EventLoggingLLMClient(RawCostFakeLLM(), logger=logger)

    client.complete(
        ChatCompletionRequest(
            model="qwen/test",
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert logger.events[0].estimated_cost_usd == 0.000003
    assert logger.events[0].billed_cost_usd == 0.000003
