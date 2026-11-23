import json

import httpx
import pytest

from agent.llm.base import LLMError, LLMRequest
from agent.llm.gemini import GeminiClient
from agent.llm.openai_compat import OpenAICompatClient
from agent.llm.pricing import PRICES, cost_usd, parse_overrides
from agent.llm.router import ModelRouter, build_router
from agent.llm.schema import SchemaError, parse_wire, wire_schema, wire_to_fields
from tests.conftest import FakeLLM, good_wire, make_settings


def request(images=()):
    return LLMRequest("extract", "sys", "user", wire_schema(), images=list(images))


def http(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def gemini_ok(text):
    return {
        "candidates": [
            {"content": {"parts": [{"text": "hidden", "thought": True}, {"text": text}]}}
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 5,
        },
    }


async def test_gemini_request_and_response_shape():
    seen = {}

    def handler(r):
        seen.update(url=str(r.url), key=r.headers["x-goog-api-key"], body=json.loads(r.content))
        return httpx.Response(200, json=gemini_ok('{"a":1}'))

    out = await GeminiClient(http(handler), "K", "m1", "minimal").complete(request([b"jpg"]))
    assert out.text == '{"a":1}' and out.tokens_out == 25, "thinking tokens are billed as output"
    assert seen["url"].endswith("/models/m1:generateContent") and "K" not in seen["url"]
    config = seen["body"]["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "temperature" not in config, "deprecated on current Gemini models"
    assert seen["body"]["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"


async def test_groq_request_uses_strict_schema_and_images():
    seen = {}

    def handler(r):
        seen.update(json.loads(r.content))
        usage = {"prompt_tokens": 7, "completion_tokens": 3}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "{}"}}], "usage": usage}
        )

    client = OpenAICompatClient(
        http(handler), "K", "https://x/v1", "m", vision=True, reasoning_effort="low"
    )
    out = await client.complete(request([b"jpg"]))
    assert (out.tokens_in, out.tokens_out) == (7, 3)
    assert seen["response_format"]["json_schema"]["strict"] is True
    assert seen["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


@pytest.mark.parametrize(
    ("status", "kind"), [(429, "quota"), (503, "unavailable"), (401, "auth"), (400, "bad_request")]
)
async def test_errors_are_classified_without_leaking_the_key(status, kind):
    h = http(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    for client in (
        GeminiClient(h, "SECRETKEY", "m"),
        OpenAICompatClient(h, "SECRETKEY", "https://x", "m"),
    ):
        with pytest.raises(LLMError) as e:
            await client.complete(request())
        assert e.value.kind == kind and "SECRETKEY" not in str(e.value)


async def test_groq_request_over_the_token_budget_counts_as_quota():
    client = OpenAICompatClient(http(lambda r: httpx.Response(413, json={})), "K", "https://x", "m")
    with pytest.raises(LLMError) as e:
        await client.complete(request())
    assert e.value.kind == "quota"


async def test_text_only_client_refuses_images_before_calling():
    def handler(r):
        raise AssertionError("must not be called")

    with pytest.raises(LLMError, match="cannot read images"):
        await OpenAICompatClient(http(handler), "K", "https://x", "m").complete(request([b"x"]))


async def test_router_falls_back_and_retries(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("agent.llm.router.asyncio.sleep", no_sleep)
    primary = FakeLLM(LLMError("quota", "429"))
    router = ModelRouter([primary, FakeLLM(good_wire(), model="other")])
    assert (await router.read(request())).model == "other"
    assert len(primary.requests) == 1, "quota errors are not retried"
    flaky = FakeLLM(LLMError("timeout", "slow"), good_wire())
    assert (await ModelRouter([flaky]).read(request())).text
    assert len(flaky.requests) == 2


async def test_router_reports_the_primary_error():
    router = ModelRouter([FakeLLM(LLMError("quota", "a")), FakeLLM(LLMError("bad_request", "b"))])
    with pytest.raises(LLMError) as e:
        await router.read(request())
    assert e.value.kind == "quota"


async def test_live_router_is_built_from_settings():
    r = build_router(make_settings(offline=False, gemini_api_key="g", groq_api_key="q"))
    assert [c.model for c in r.readers] == ["gemini-3.5-flash-lite", "qwen/qwen3.8-27b"]
    assert r.second_for(False, "gemini:x").model == "openai/gpt-oss-120b"
    assert r.second_for(True, "gemini:x").model == "qwen/qwen3.8-27b"
    # If the fallback did the reading, the same model cannot also be the second opinion.
    assert r.second_for(True, "groq:qwen/qwen3.8-27b") is None
    await r.aclose()


def test_default_models_all_have_a_price():
    s = make_settings()
    for model in (s.primary_model, s.second_model, s.second_vision_model, s.fallback_model):
        assert model in PRICES


def test_pricing():
    assert cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000, {}) == pytest.approx(0.75)
    assert parse_overrides("a/b=1/2, c=0.5/0.25") == {"a/b": (1.0, 2.0), "c": (0.5, 0.25)}
    with pytest.raises(KeyError):
        cost_usd("mystery", 1, 1, {})


def test_wire_schema_is_strict_mode_compatible():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert sorted(node["required"]) == sorted(node["properties"])
            for v in node.values():
                walk(v)

    walk(wire_schema(transcript=True))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "nope",
        "[]",
        "{}",
        json.dumps({**good_wire(), "line_items": "x"}),
        json.dumps({**good_wire(), "extra": 1}),
    ],
)
def test_parse_wire_rejects(text):
    with pytest.raises(SchemaError):
        parse_wire(text)


def test_parse_wire_accepts_numbers_for_amounts():
    fields = wire_to_fields(parse_wire(json.dumps(good_wire(total=354.5))))
    assert str(fields.amounts.total) == "354.5"
