"""Which client reads, which gives the second opinion, and what happens when one is down."""

import asyncio
import ssl
from dataclasses import dataclass, field

import httpx

from agent.config import Settings
from agent.llm.base import LLMClient, LLMError, LLMRequest, LLMResponse
from agent.llm.gemini import GeminiClient
from agent.llm.openai_compat import OpenAICompatClient


@dataclass
class ModelRouter:
    readers: list[LLMClient]
    second_text: LLMClient | None = None
    second_vision: LLMClient | None = None
    max_retries: int = 1
    http: httpx.AsyncClient | None = field(default=None, repr=False)

    async def read(self, request: LLMRequest) -> LLMResponse:
        """Primary first, then any fallback. If all fail, the primary's error is the one reported."""
        last: LLMError | None = None
        for client in self.readers:
            try:
                return await self.call(client, request)
            except LLMError as e:
                last = last or e
        raise last or LLMError("unsupported", "No configured model can read this document")

    async def call(self, client: LLMClient, request: LLMRequest) -> LLMResponse:
        for attempt in range(self.max_retries + 1):
            try:
                return await client.complete(request)
            except LLMError as e:
                if not e.retryable or attempt == self.max_retries:
                    raise
                await asyncio.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")

    def second_for(self, has_images: bool, reader: str | None) -> LLMClient | None:
        """A second opinion from the model that did the reading is not a second opinion."""
        client = self.second_vision if has_images else self.second_text
        if client is None or f"{client.provider}:{client.model}" == reader:
            return None
        return client

    async def aclose(self) -> None:
        if self.http is not None:
            await self.http.aclose()


def system_ssl_context() -> ssl.SSLContext:
    import truststore

    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def build_router(settings: Settings) -> ModelRouter:
    if not settings.live:
        from agent.llm.offline import OfflineAnchors, OfflineRules

        return ModelRouter(readers=[OfflineRules()], second_text=OfflineAnchors(), max_retries=0)

    http = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.llm_timeout_s, connect=10), verify=system_ssl_context()
    )
    assert settings.gemini_api_key
    readers: list[LLMClient] = [
        GeminiClient(
            http, settings.gemini_api_key, settings.primary_model,
            settings.primary_thinking_level, settings.primary_temperature,
        )
    ]  # fmt: skip
    second_text: LLMClient | None = None
    second_vision: LLMClient | None = None
    if settings.groq_api_key:

        def groq(model: str, vision: bool, effort: str) -> OpenAICompatClient:
            assert settings.groq_api_key
            return OpenAICompatClient(
                http, settings.groq_api_key, settings.groq_base_url, model,
                vision=vision, reasoning_effort=effort,
            )  # fmt: skip

        if settings.fallback_model:
            readers.append(
                groq(settings.fallback_model, True, settings.second_vision_reasoning_effort)
            )
        if settings.second_model:
            second_text = groq(settings.second_model, False, settings.second_reasoning_effort)
        if settings.second_vision_model:
            second_vision = groq(
                settings.second_vision_model, True, settings.second_vision_reasoning_effort
            )
    return ModelRouter(readers, second_text, second_vision, settings.llm_max_retries, http)
