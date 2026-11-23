"""Any OpenAI-compatible chat endpoint. Used for Groq."""

import base64
import time
from typing import Any

import httpx

from agent.llm.base import LLMError, LLMRequest, LLMResponse, error_kind


class OpenAICompatClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        api_key: str,
        base_url: str,
        model: str,
        provider: str = "groq",
        vision: bool = False,
        reasoning_effort: str = "",
    ):
        self._http = http
        self._key = api_key
        self._base_url = base_url.rstrip("/")
        self._effort = reasoning_effort
        self.model = model
        self.provider = provider
        self.vision = vision

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.images and not self.vision:
            raise LLMError("unsupported", f"{self.model} cannot read images", self.provider)
        content: list[dict[str, Any]] = [{"type": "text", "text": request.user}]
        for image in request.images:
            url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": content if request.images else request.user},
            ],
            "temperature": 0,
            "max_completion_tokens": request.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": True,
                    "schema": request.schema,
                },
            },
        }
        if self._effort:
            body["reasoning_effort"] = self._effort
        started = time.perf_counter()
        try:
            r = await self._http.post(
                f"{self._base_url}/chat/completions",
                headers={"authorization": f"Bearer {self._key}"},
                json=body,
            )
        except httpx.TimeoutException as e:
            raise LLMError("timeout", f"{self.provider} timed out", self.provider) from e
        except httpx.HTTPError as e:
            raise LLMError(
                "unavailable", f"{self.provider} unreachable: {type(e).__name__}", self.provider
            ) from e
        latency = int((time.perf_counter() - started) * 1000)
        if r.status_code != 200:
            # Groq answers 413 when one request is larger than the free per-minute token budget.
            kind = "quota" if r.status_code == 413 else error_kind(r.status_code)
            raise LLMError(
                kind, f"{self.provider} returned {r.status_code}: {_message(r)}", self.provider
            )
        payload = r.json()
        usage = payload.get("usage") or {}
        choices = payload.get("choices") or [{}]
        return LLMResponse(
            text=(choices[0].get("message") or {}).get("content") or "",
            provider=self.provider,
            model=self.model,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            latency_ms=latency,
        )


def _message(r: httpx.Response) -> str:
    try:
        return str(r.json()["error"]["message"])[:200]
    except Exception:
        return "unexpected response"
