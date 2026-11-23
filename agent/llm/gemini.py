"""Gemini over REST. `generateContent` rather than the newer Interactions API because it is
stateless: nothing about the document is stored server-side between calls."""

import base64
import time
from typing import Any

import httpx

from agent.llm.base import LLMError, LLMRequest, LLMResponse, error_kind


class GeminiClient:
    provider = "gemini"
    vision = True
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        http: httpx.AsyncClient,
        api_key: str,
        model: str,
        thinking_level: str = "",
        temperature: float | None = None,
    ):
        self._http = http
        self._key = api_key
        self.model = model
        self._thinking = thinking_level
        self._temperature = temperature

    async def complete(self, request: LLMRequest) -> LLMResponse:
        parts: list[dict[str, Any]] = [{"text": request.user}]
        for image in request.images:
            data = base64.b64encode(image).decode()
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": data}})
        config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": request.schema,
            "maxOutputTokens": request.max_output_tokens,
        }
        if self._thinking:
            config["thinkingConfig"] = {"thinkingLevel": self._thinking}
        if self._temperature is not None:
            config["temperature"] = self._temperature
        body = {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": config,
        }
        started = time.perf_counter()
        try:
            r = await self._http.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self._key},
                json=body,
            )
        except httpx.TimeoutException as e:
            raise LLMError("timeout", "Gemini timed out", self.provider) from e
        except httpx.HTTPError as e:
            raise LLMError(
                "unavailable", f"Gemini unreachable: {type(e).__name__}", self.provider
            ) from e
        latency = int((time.perf_counter() - started) * 1000)
        if r.status_code != 200:
            kind = error_kind(r.status_code)
            if r.status_code == 400 and "API_KEY_INVALID" in r.text:
                kind = "auth"
            raise LLMError(kind, f"Gemini returned {r.status_code}: {_message(r)}", self.provider)
        payload = r.json()
        usage = payload.get("usageMetadata", {})
        candidates = payload.get("candidates") or [{}]
        text = "".join(
            p.get("text", "")
            for p in candidates[0].get("content", {}).get("parts", [])
            if not p.get("thought")
        )
        # An empty or truncated answer is returned as-is: it fails schema validation downstream,
        # which routes it to repair like any other malformed output.
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            tokens_in=int(usage.get("promptTokenCount", 0)),
            # Thinking tokens are billed as output.
            tokens_out=int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0)),
            latency_ms=latency,
        )  # fmt: skip


def _message(r: httpx.Response) -> str:
    try:
        return str(r.json()["error"]["message"])[:200]
    except Exception:
        return "unexpected response"
