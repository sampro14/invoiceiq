import json
import time

import pytest
from fastapi.testclient import TestClient

from agent.deps import Deps
from agent.llm.router import ModelRouter
from app.limits import RateLimiter
from app.main import create_app
from app.samples import all_samples
from tests.conftest import SAMPLES, FakeLLM, good_wire, make_settings, page_text


def client_for(tmp_path, deps=None, **settings):
    s = make_settings(checkpoint_db=str(tmp_path / "c.sqlite"), **settings)
    if deps:
        deps.settings = s
    return TestClient(create_app(s, deps))


@pytest.fixture
def client(tmp_path):
    with client_for(tmp_path, runs_per_ip_per_hour=100) as c:
        yield c


def wait(client, run_id):
    for _ in range(200):
        r = client.get(f"/v1/runs/{run_id}").json()
        if r["status"] != "running":
            return r
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def run_sample(client, sample_id):
    r = client.post("/v1/extract", data={"sample_id": sample_id})
    assert r.status_code == 202, r.text
    return wait(client, r.json()["run_id"])


def test_health_and_index(client):
    assert client.get("/healthz").text == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert "unsafe" not in page.headers["content-security-policy"]


def test_clean_sample_is_approved(client):
    r = run_sample(client, "clean")
    assert (r["status"], r["decision"], r["mode"]) == ("completed", "approved", "offline")
    assert r["fields"]["amounts"]["total"] == "95507.79"
    assert r["metrics"]["cost_usd"] == 0 and r["metrics"]["latency_ms"] >= 0


def test_wrong_total_waits_for_review_and_can_be_corrected(client):
    r = run_sample(client, "wrong-total")
    assert (r["status"], r["decision"]) == ("awaiting_review", "review")
    printed = r["fields"]["amounts"]["total"]
    url = f"/v1/runs/{r['run_id']}/review"
    fixed = client.post(url, json={"corrections": {"amounts.total": "82521.06"}})
    assert fixed.status_code == 200, fixed.text
    body = fixed.json()
    assert (body["status"], body["decision"], body["overrides"]) == ("completed", "approved", [])
    assert body["corrections"] == [
        {"path": "amounts.total", "before": printed, "after": "82521.06"}
    ]
    assert client.post(url, json={}).status_code == 409, "a run can be reviewed once"


def test_injection_sample_is_flagged_and_unchanged(client):
    r = run_sample(client, "injection")
    assert r["decision"] == "review"
    assert float(r["fields"]["amounts"]["total"]) > 1000


def test_photo_without_a_vision_model_fails_clearly(client):
    r = run_sample(client, "photo")
    assert r["status"] == "failed" and "GEMINI_API_KEY" in r["error"]


def test_event_stream_replays_every_node_and_ends_with_the_result(client):
    run_id = client.post("/v1/extract", data={"sample_id": "clean"}).json()["run_id"]
    with client.stream("GET", f"/v1/runs/{run_id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    ended = [e["node"] for e in events if e["type"] == "node_end"]
    assert ended == ["ingest", "extract_text", "cross_check", "validate"]
    assert events[-1]["type"] == "result" and events[-1]["result"]["decision"] == "approved"
    assert any(e["type"] == "partial" and "fields" in e for e in events)


def test_upload_works_and_is_forgotten(tmp_path):
    w = good_wire()
    deps = Deps(make_settings(), ModelRouter([FakeLLM(w)], FakeLLM(w, model="offline-anchors")))
    with client_for(tmp_path, deps) as c:
        r = c.post(
            "/v1/extract", files={"file": ("x.exe", page_text(w), "application/octet-stream")}
        )
        assert r.status_code == 202, "the signature decides, not the name"
        assert wait(c, r.json()["run_id"])["decision"] == "approved"
        assert len(deps.store) == 0
    raw = (tmp_path / "c.sqlite").read_bytes()
    assert b"%PDF" not in raw and b"Deccan" in raw


def test_large_upload_is_never_spooled_to_disk(tmp_path, monkeypatch):
    import tempfile

    def forbidden(*a, **k):
        raise AssertionError("upload was rolled over to disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", forbidden)
    w = good_wire()
    deps = Deps(make_settings(), ModelRouter([FakeLLM(w)], None))
    with client_for(tmp_path, deps) as c:
        data = page_text(w) + b"\n%" + b"x" * (3 * 1024 * 1024)
        assert c.post("/v1/extract", files={"file": ("a.pdf", data)}).status_code == 202


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({}, 422),
        ({"data": {"sample_id": "nope"}}, 404),
        ({"files": {"file": ("a.pdf", b"<html>")}}, 415),
        ({"files": {"file": ("a.pdf", b"%PDF-" + b"x" * (5 * 1024 * 1024 + 10))}}, 413),
        ({"files": {"file": ("a.pdf", b"%PDF-" + b"x" * (6 * 1024 * 1024))}}, 413),
        ({"files": {"file": ("a.pdf", b"%PDF-")}, "data": {"sample_id": "clean"}}, 422),
    ],
)
def test_bad_requests(client, kwargs, status):
    assert client.post("/v1/extract", **kwargs).status_code == status


def test_chunked_body_without_a_length_is_still_limited(client):
    def chunks():
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n\r\n'
        for _ in range(8):
            yield b"x" * 1024 * 1024
        yield b"\r\n--b--\r\n"

    r = client.post(
        "/v1/extract", content=chunks(), headers={"content-type": "multipart/form-data; boundary=b"}
    )
    assert r.status_code == 413


def test_bad_corrections_are_refused_and_leave_the_run_waiting(client):
    r = run_sample(client, "wrong-total")
    url = f"/v1/runs/{r['run_id']}/review"
    for corrections in (
        {"__class__": "x"}, {"amounts": "1"}, {"amounts.nope": "1"}, {"line_items.99.quantity": "1"},
        {"amounts.total": "abc"}, {"vendor.name": "x" * 501}, {"amounts.total": {"a": 1}},
        {"line_items": "x"}, {"line_items": [{"evil": 1}]},
        {"line_items": [], "line_items.0.quantity": "1"},
    ):  # fmt: skip
        assert client.post(url, json={"corrections": corrections}).status_code == 422, corrections
    assert client.get(f"/v1/runs/{r['run_id']}").json()["status"] == "awaiting_review"


def test_unknown_runs(client):
    for path in ("/v1/runs/zzz", "/v1/runs/" + "0" * 32, "/v1/runs/" + "0" * 32 + "/events"):
        assert client.get(path).status_code == 404
    assert client.post("/v1/runs/" + "0" * 32 + "/review", json={}).status_code == 404


def test_paused_run_survives_a_restart(tmp_path):
    with client_for(tmp_path) as c:
        run_id = run_sample(c, "wrong-total")["run_id"]
    with client_for(tmp_path) as c:
        assert c.get(f"/v1/runs/{run_id}").json()["status"] == "awaiting_review"
        r = c.post(f"/v1/runs/{run_id}/review", json={"corrections": {"amounts.total": "82521.06"}})
        assert r.json()["decision"] == "approved"


def test_old_runs_are_purged(tmp_path):
    with client_for(tmp_path) as c:
        run_id = run_sample(c, "clean")["run_id"]
    with client_for(tmp_path, run_ttl_s=-1) as c:
        time.sleep(0.3)  # the janitor runs at startup
        assert c.get(f"/v1/runs/{run_id}").status_code == 404


def test_quota_answers_429_and_replays_cached_samples(tmp_path, monkeypatch):
    with client_for(tmp_path, runs_per_ip_per_hour=1) as c:
        first = run_sample(c, "clean")
        denied = c.post("/v1/extract", data={"sample_id": "injection"})
        assert denied.status_code == 429 and int(denied.headers["retry-after"]) > 0
        recorded = {"events": [{"type": "node_start", "node": "ingest"}], "result": first}
        monkeypatch.setattr(type(all_samples()["clean"]), "cached_run", lambda self: recorded)
        r = c.post("/v1/extract", data={"sample_id": "clean"})
        assert r.json()["cached"] is True
        replay = wait(c, r.json()["run_id"])
        assert replay["cached"] and replay["run_id"] != first["run_id"]


def test_abuse_switch_keeps_samples_only(tmp_path):
    with client_for(tmp_path, uploads_enabled=False) as c:
        r = c.post("/v1/extract", files={"file": ("a.pdf", (SAMPLES / "clean.pdf").read_bytes())})
        assert r.status_code == 403
        assert run_sample(c, "clean")["decision"] == "approved"


def test_client_ip_header_is_ignored_unless_trusted(tmp_path):
    def two(c):
        return [
            c.post(
                "/v1/extract",
                data={"sample_id": "clean"},
                headers={"cf-connecting-ip": f"9.9.9.{i}"},
            ).status_code
            for i in range(2)
        ]

    with client_for(tmp_path, runs_per_ip_per_hour=1) as c:
        assert two(c) == [202, 429]
    with client_for(tmp_path, runs_per_ip_per_hour=1, trusted_ip_header="cf-connecting-ip") as c:
        assert two(c) == [202, 202]


def test_rate_limiter_windows():
    now = [1_800_000_000.0]
    limiter = RateLimiter(2, 3, now=lambda: now[0])
    assert limiter.check("a") is None and limiter.check("a") is None
    assert limiter.check("a")[1] <= 3600
    assert limiter.check("b") is None
    message, wait_s = limiter.check("c")
    assert "quota" in message and 0 < wait_s <= 25 * 3600
    now[0] += wait_s + 1
    assert limiter.check("c") is None, "the daily cap resets at midnight Pacific"


def test_samples_config_and_docs(client):
    samples = client.get("/v1/samples").json()
    assert [s["id"] for s in samples] == ["clean", "photo", "wrong-total", "injection"]
    assert client.get(samples[0]["preview_url"]).headers["content-type"] == "image/png"
    assert client.get("/v1/config").json()["mode"] == "offline"
    assert "/v1/extract" in client.get("/openapi.json").json()["paths"]
