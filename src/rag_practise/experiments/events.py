from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from rag_practise.llm import ChatCompletionRequest, ChatCompletionResult, LLMClient


class PricingPolicy(BaseModel):
    input_usd_per_1k_tokens: float = 0.0
    output_usd_per_1k_tokens: float = 0.0
    billed_input_usd_per_1k_tokens: float = 0.0
    billed_output_usd_per_1k_tokens: float = 0.0


class LLMCallEvent(BaseModel):
    provider: str
    model: str
    status: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float = 0.0
    billed_cost_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventLogger:
    def __init__(self) -> None:
        self.events: list[LLMCallEvent] = []

    def append(self, event: LLMCallEvent) -> None:
        self.events.append(event)

    def since(self, offset: int) -> list[LLMCallEvent]:
        return self.events[offset:]


class EventLoggingLLMClient:
    def __init__(
        self,
        client: LLMClient,
        *,
        logger: EventLogger,
        pricing: PricingPolicy | None = None,
    ) -> None:
        self.client = client
        self.provider = client.provider
        self.logger = logger
        self.pricing = pricing or PricingPolicy()

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        started = time.perf_counter()
        try:
            result = self.client.complete(request)
            latency_ms = (time.perf_counter() - started) * 1000
            event = _event_from_result(
                provider=self.provider,
                result=result,
                latency_ms=latency_ms,
                pricing=self.pricing,
                status="ok",
            )
            self.logger.append(event)
            return result
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            self.logger.append(
                LLMCallEvent(
                    provider=self.provider,
                    model=request.model,
                    status="error",
                    latency_ms=latency_ms,
                    error=str(exc),
                )
            )
            raise


def _event_from_result(
    *,
    provider: str,
    result: ChatCompletionResult,
    latency_ms: float,
    pricing: PricingPolicy,
    status: str,
) -> LLMCallEvent:
    prompt_tokens = result.usage.prompt_tokens or 0
    completion_tokens = result.usage.completion_tokens or 0
    estimated = (
        prompt_tokens * pricing.input_usd_per_1k_tokens
        + completion_tokens * pricing.output_usd_per_1k_tokens
    ) / 1000
    raw_cost = _raw_usage_cost(result.raw)
    if estimated == 0 and raw_cost is not None:
        estimated = raw_cost
    billed = (
        prompt_tokens * pricing.billed_input_usd_per_1k_tokens
        + completion_tokens * pricing.billed_output_usd_per_1k_tokens
    ) / 1000
    if billed == 0 and raw_cost is not None:
        billed = raw_cost
    return LLMCallEvent(
        provider=provider,
        model=result.model,
        status=status,
        latency_ms=round(latency_ms, 4),
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        total_tokens=result.usage.total_tokens,
        estimated_cost_usd=round(estimated, 8),
        billed_cost_usd=round(billed, 8),
    )


def _raw_usage_cost(raw: object | None) -> float | None:
    if not isinstance(raw, dict):
        return None
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if isinstance(cost, int | float):
        return float(cost)
    return None
