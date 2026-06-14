from __future__ import annotations

import os
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from rag_practise.experiments.events import PricingPolicy
from rag_practise.llm import (
    GeminiChatClient,
    HttpOpenAICompatibleChatClient,
    LLMClient,
    OpenAICompatibleChatClient,
)


class ModelConfig(BaseModel):
    id: str
    provider: str
    model: str
    roles: list[str] = Field(default_factory=list)
    benchmark_tier: str = "smoke"
    pricing_mode: str = "manual"
    api_key_env: str | None = None
    base_url: str | None = None
    input_usd_per_1k_tokens: float = 0.0
    output_usd_per_1k_tokens: float = 0.0
    billed_input_usd_per_1k_tokens: float = 0.0
    billed_output_usd_per_1k_tokens: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 0

    def pricing_policy(self) -> PricingPolicy:
        return PricingPolicy(
            input_usd_per_1k_tokens=self.input_usd_per_1k_tokens,
            output_usd_per_1k_tokens=self.output_usd_per_1k_tokens,
            billed_input_usd_per_1k_tokens=self.billed_input_usd_per_1k_tokens,
            billed_output_usd_per_1k_tokens=self.billed_output_usd_per_1k_tokens,
        )


def load_model_configs(path: Path, *, role: str = "transform") -> list[ModelConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    configs = [ModelConfig.model_validate(item) for item in raw.get("models", [])]
    return [config for config in configs if role in config.roles]


def build_llm_client(config: ModelConfig) -> LLMClient:
    api_key_env = config.api_key_env or _default_api_key_env(config.provider)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for provider {config.provider}")

    if config.provider == "gemini":
        return GeminiChatClient(api_key=api_key, provider=config.provider)
    if config.provider in {"openrouter", "nvidia_nim"}:
        return HttpOpenAICompatibleChatClient(
            provider=config.provider,
            api_key=api_key,
            base_url=config.base_url or _default_base_url(config.provider) or "",
            timeout=config.timeout_seconds,
        )

    return OpenAICompatibleChatClient(
        provider=config.provider,
        api_key=api_key,
        base_url=config.base_url or _default_base_url(config.provider),
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def _default_api_key_env(provider: str) -> str:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "nvidia_nim": "NVIDIA_API_KEY",
    }
    try:
        return mapping[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def _default_base_url(provider: str) -> str | None:
    mapping: dict[str, str | None] = {
        "openai": None,
        "gemini": None,
        "openrouter": "https://openrouter.ai/api/v1",
        "nvidia_nim": "https://integrate.api.nvidia.com/v1",
    }
    return mapping.get(provider)
