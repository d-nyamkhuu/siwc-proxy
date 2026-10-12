"""HTTP contracts of the proxy against a mocked OpenAI."""

import gzip
import json
import socket
import threading

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

from siwc_proxy.config import Settings
from siwc_proxy.server import create_app
from support import (BASE, BODY, HEADERS, KEY, TEXT_ITEM, Chunks, live_server, parse_chunks, running, sse,
                     terminal, text_events, tool_events, write_credentials)


def streamed(events, **headers):
    return lambda request: httpx.Response(200, headers={"content-type": "text/event-stream", **headers},
                                          stream=Chunks(sse(events)))


# Chat Completions

def test_streaming_unicode_usage_and_metadata(directory, fake):
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/chat/completions",
                               json={**BODY, "stream": True, "stream_options": {"include_usage": True}})
        chunks = parse_chunks(response)
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == "Hi 世界"
        assert len({(c["id"], c["created"], c["model"]) for c in chunks}) == 1
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 14
        assert all(c["usage"] is None for c in chunks[:-1])
        assert response.text.endswith("data: [DONE]\n\n")
        assert "private" not in response.text


def test_chat_upstream_request(directory, fake):
    with running(directory, fake) as (client, fake, app):
        response = client.post("/v1/chat/completions", json=BODY, headers={
            "OpenAI-Organization": "org", "Cookie": "local=secret", "Idempotency-Key": "once"})
        assert response.json()["choices"][0]["message"]["content"] == "Hi 世界"
        call = fake.calls[-1]
        assert str(call.url) == "https://api.openai.com/v1/responses"
        assert call.headers["authorization"] == "Bearer oauth-secret"
        assert call.headers["accept-encoding"] == "identity"
        assert call.headers["idempotency-key"] == "once"
        assert not any(h in call.headers for h in ("openai-organization", "cookie"))
        assert json.loads(call.content)["store"] is False and KEY not in call.content.decode()
        assert response.headers["x-request-id"] == "req-test"
        assert response.headers["cache-control"] == "no-store"
        assert not app.state.active


def test_lenient_chat_reports_ignored_params(directory, fake):
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/chat/completions", json={**BODY, "user": "u", "seed": 1})
        assert response.status_code == 200
        assert response.headers["x-siwc-ignored-params"] == "user,seed"
    with running(directory, fake, strict=True) as (client, fake, _):
        calls = len(fake.calls)
        response = client.post("/v1/chat/completions", json={**BODY, "user": "u"})
        assert response.status_code == 400 and response.json()["error"]["param"] == "user"
        assert len(fake.calls) == calls


@pytest.mark.parametrize("streaming", [False, True])
def test_interleaved_tools_have_dense_indexes_and_no_duplicates(directory, fake, streaming):
    stream = Chunks(sse(tool_events()), size=1)
    fake.api = lambda r: httpx.Response(200, stream=stream)
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
        if streaming:
            chunks = parse_chunks(response)
            tool_calls, arguments = {}, {}
            for chunk in chunks:
                for tool in chunk["choices"][0]["delta"].get("tool_calls", []):
                    if "id" in tool:
                        assert tool["index"] not in tool_calls
                        tool_calls[tool["index"]] = tool["id"]
                    arguments[tool["index"]] = arguments.get(tool["index"], "") + tool["function"].get("arguments", "")
            assert tool_calls == {0: "call-a", 1: "call-b"}
            assert arguments == {0: '{"x":1}', 1: "{}"}
            assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        else:
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["content"] is None
            assert [t["function"]["arguments"] for t in choice["message"]["tool_calls"]] == ['{"x":1}', '{}']
        assert stream.closed


@pytest.mark.parametrize("reason,finish", [("max_output_tokens", "length"), ("content_filter", "content_filter")])
@pytest.mark.parametrize("streaming", [False, True])
def test_incomplete_and_refusal(directory, fake, reason, finish, streaming):
    item = {"type": "message", "content": [{"type": "refusal", "refusal": "Cannot help"}]}
    fake.api = streamed([{"type": "response.refusal.delta", "output_index": 0, "content_index": 0, "delta": "Cannot "},
                         {"type": "response.incomplete", "response": terminal(
                             [item], status="incomplete", incomplete_details={"reason": reason})}])
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
        if streaming:
            chunks = parse_chunks(response)
            assert "".join(c["choices"][0]["delta"].get("refusal", "") for c in chunks) == "Cannot help"
            assert chunks[-1]["choices"][0]["finish_reason"] == finish
        else:
            assert response.json()["choices"][0]["message"]["refusal"] == "Cannot help"
            assert response.json()["choices"][0]["finish_reason"] == finish


@pytest.mark.parametrize("bad", [b"data: not-json\n\n", b'data: {"type":"response.completed"}\n\n',
    b"data: [DONE]\n\n", b'data: {"type":"response.output_text.delta"}\n\n', b'data: {"type":"future"}',
    b'data: {"type":"future","data":"\xff"}\n\n', b"", sse(text_events()[:-1]),
    sse([{"type": "error", "message": "Failed", "code": "provider_error"}]),
    sse([{"type": "response.incomplete", "response": terminal(status="incomplete",
                                                               incomplete_details={"reason": "unknown"})}]),
    sse(text_events()[:4] + [{"type": "response.completed", "response": terminal([])}]),
    sse(text_events()[:4] + [{"type": "response.completed", "response": terminal([{
        **TEXT_ITEM, "content": [{"type": "output_text", "text": "Conflicting output"}]}])}]),
])
@pytest.mark.parametrize("streaming", [False, True])
def test_stream_failures_never_report_success(directory, fake, bad, streaming):
    stream = Chunks(bad)
    fake.api = lambda r: httpx.Response(200, headers={"x-request-id": "req-failed"}, stream=stream)
    with running(directory, fake) as (client, fake, app):
        response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
        assert response.headers["x-request-id"] == "req-failed"
        if streaming:
            assert "error" in parse_chunks(response)[-1]
            assert "data: [DONE]" not in response.text
            assert not any(c["choices"][0].get("finish_reason") for c in parse_chunks(response) if c.get("choices"))
        else:
            assert response.status_code == 502 and "error" in response.json()
        assert len(fake.calls) == 1 and stream.closed and not app.state.active


@pytest.mark.parametrize("code,status", [("subscription_sharing_usage_limit_exceeded", 429),
                                         ("subscription_sharing_usage_unavailable", 503),
                                         ("chatpass_v2_blocked", 403), ("something_else", 502)])
@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
def test_failed_responses_keep_their_meaning(directory, fake, code, status, endpoint):
    fake.api = streamed([{"type": "response.failed", "response": terminal(
        status="failed", error={"message": "Plan limit", "code": code})}])
    body = BODY if endpoint.startswith("chat") else {"model": "test-model", "input": "Hi"}
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/" + endpoint, json=body)
        assert response.status_code == status
        error = response.json()["error"]
        assert error["code"] == code
        if status == 429:  # Retrying does not lift a plan limit; point to where usage is managed.
            assert error["message"] == "Plan limit Manage usage: https://chatgpt.com/settings/usage"
            assert response.headers["x-should-retry"] == "false"
        else:
            assert error["message"] == "Plan limit" and "x-should-retry" not in response.headers


def test_admission_usage_limit_is_not_retried_by_sdks(directory, fake):
    fake.api = lambda r: httpx.Response(429, json={"error": {
        "message": "Limit", "type": "rate_limit_error", "code": "subscription_sharing_usage_limit_exceeded"}})
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/responses", json={"model": "m", "input": "Hi"})
        assert response.status_code == 429 and response.headers["x-should-retry"] == "false"


def test_responses_drop_rejected_params_and_system_items(directory, fake):
    system = {"type": "message", "role": "system", "content": "Be terse."}
    body = {"model": "m", "input": [system, {"role": "system", "content": "x"}, {"role": "user", "content": "Hi"}],
            "temperature": 0.2, "top_p": 1, "metadata": {"a": "b"}, "user": "u", "truncation": "auto",
            "max_output_tokens": 50, "prompt_cache_key": "k"}
    with running(directory, fake) as (client, fake, _):
        response = client.post("/v1/responses", json=body)
        assert response.headers["x-siwc-ignored-params"] == "temperature,top_p,metadata,user,truncation"
        sent = json.loads(fake.calls[-1].content)
        assert sent["input"][0] == {**system, "role": "developer"}
        assert sent["input"][1] == {"role": "system", "content": "x"}  # accepted as is
        assert "max_output_tokens" in sent and "temperature" not in sent
        assert "prompt_cache_key" not in sent  # Replaced upstream; the session_id header carries it instead.
    with running(directory, fake, strict=True) as (client, *_):
        response = client.post("/v1/responses", json=body)
        assert response.status_code == 400 and response.json()["error"]["param"] == "temperature"


def test_turns_of_a_conversation_share_a_cache_session(directory, fake):
    fake.api = streamed(text_events())
    system = {"role": "system", "content": "Long shared prompt"}
    turn1 = [system, {"role": "user", "content": "Hi"}]
    turn2 = turn1 + [{"role": "assistant", "content": "Hello"}, {"role": "user", "content": "More"}]
    edited = [system, {"role": "user", "content": "Edited"}] + turn2[2:]
    other = [{"role": "system", "content": "Different prompt"}, {"role": "user", "content": "Hi"}]
    with running(directory, fake) as (client, fake, _):
        def session(path, body, **headers):
            client.post(path, json=body, headers=headers)
            return fake.calls[-1].headers["session_id"]

        chat = [session("/v1/chat/completions", {"model": "m", "messages": m}) for m in (turn1, turn2, edited, other)]
        assert chat[0] == chat[1] == chat[2] != chat[3]  # Edits after the first item keep the session.
        assert len(chat[0]) == 32 and "Long" not in chat[0]
        responses = [session("/v1/responses", {"model": "m", "input": m}) for m in (turn1, turn2)]
        assert responses[0] == responses[1] == chat[0]
        assert session("/v1/responses", {"model": "m", "input": turn1, "instructions": "x"}) != chat[0]
        assert session("/v1/responses", {"model": "other", "input": turn1}) != chat[0]
        keyed = [session("/v1/chat/completions", {"model": "m", "messages": m, "prompt_cache_key": "conv-1"})
                 for m in (turn1, other)]
        assert keyed[0] == keyed[1] != chat[0]
        assert "prompt_cache_key" not in json.loads(fake.calls[-1].content)
        assert session("/v1/responses", {"model": "m", "input": turn1}, session_id="mine") == "mine"
        client.get("/v1/models")
        assert "session_id" not in fake.calls[-1].headers


def test_multiline_sse_comments_crlf_and_bom(directory, fake):
    event = json.dumps({"type": "response.completed", "response": terminal()}, ensure_ascii=False)
    split = event.index('"response":')
    raw = ("﻿: ping\r\nevent: response.completed\r\ndata: " + event[:split] + "\r\ndata: " + event[split:]
           + "\r\n\r\n").encode()
    fake.api = lambda r: httpx.Response(200, stream=Chunks(raw, size=1))
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hi 世界"


# Responses

def test_responses_normalized_and_aggregated(directory, fake):
    with running(directory, fake) as (client, fake, _):
        response = client.post("/v1/responses?x=a%20b&x=c", json={"model": "test-model", "input": "Hi", "store": True},
                               headers={"OpenAI-Beta": "test=v1"})
        assert response.status_code == 200
        assert response.json()["output"][0]["content"][0]["text"] == "Hi 世界"
        assert response.headers["content-type"] == "application/json"
        assert response.headers["x-siwc-ignored-params"] == "store"
        sent = json.loads(fake.calls[-1].content)
        assert sent == {"model": "test-model", "input": [{"role": "user", "content": "Hi"}],
                        "store": False, "stream": True}
        assert fake.calls[-1].url.query == b"x=a%20b&x=c"
        assert fake.calls[-1].headers["openai-beta"] == "test=v1"


def test_responses_fill_output_from_items(directory, fake):
    item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "From item"}]}
    fake.api = streamed([{"type": "response.output_item.done", "output_index": 0, "item": item},
                         {"type": "response.completed", "response": terminal([])}])
    with running(directory, fake) as (client, *_):
        assert client.post("/v1/responses", json={"model": "m", "input": "Hi"}).json()["output"] == [item]



def empty_terminal_events():
    """What the ChatGPT backend sends: items only in output_item.done, and a terminal response without output."""
    item = {**TEXT_ITEM, "id": "msg-1", "content": [{**TEXT_ITEM["content"][0], "annotations": [
        {"type": "url_citation", "url": "https://example.com/a", "title": "A", "start_index": 3, "end_index": 5}]}]}
    return [{"type": "response.created", "response": terminal([], status="in_progress")},
            {"type": "response.output_item.done", "output_index": 1, "item": {"type": "reasoning", "id": "r"}},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": terminal([])}], item


def test_relayed_stream_terminal_gets_output(directory, fake):
    events, item = empty_terminal_events()
    raw = sse(events).replace(b"\n", b"\r\n")
    fake.api = lambda r: httpx.Response(200, stream=Chunks(gzip.compress(raw), size=5), headers={
        "content-encoding": "gzip", "content-length": "999", "content-type": "text/event-stream"})
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/responses", json={"model": "m", "input": "Hi", "stream": True})
        assert "content-length" not in response.headers
        blocks = response.content.split(b"\r\n\r\n")
        assert b"\r\n\r\n".join(blocks[:3]) + b"\r\n\r\n" == sse(events[:3]).replace(b"\n", b"\r\n")
        last = blocks[3].decode().splitlines()
        assert last[0] == "event: response.completed"
        assert json.loads(last[1][6:])["response"]["output"] == [item, {"type": "reasoning", "id": "r"}]


@pytest.mark.parametrize("streaming", [False, True])
def test_chat_from_items_with_citations(directory, fake, streaming):
    fake.api = streamed(empty_terminal_events()[0])
    citation = {"type": "url_citation", "url_citation": {"url": "https://example.com/a", "title": "A",
                                                          "start_index": 3, "end_index": 5}}
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
        assert response.status_code == 200
        if streaming:
            deltas = [c["choices"][0]["delta"] for c in parse_chunks(response) if c["choices"]]
            assert "".join(d.get("content", "") for d in deltas) == "Hi 世界"
            assert [d["annotations"] for d in deltas if "annotations" in d] == [[citation]]
        else:
            message = response.json()["choices"][0]["message"]
            assert message["content"] == "Hi 世界" and message["annotations"] == [citation]


def test_upstream_request_id_and_edge_headers(directory, fake):
    fake.api = streamed(text_events(), **{"x-oai-request-id": "req-oai", "date": "Thu, 01 Jan 1970 00:00:00 GMT",
                                          "server": "cloudflare", "alt-svc": "h3"})
    with running(directory, fake) as (client, *_):
        for path, body, stream in [("/v1/responses", {"model": "m", "input": "Hi"}, False),
                                   ("/v1/responses", {"model": "m", "input": "Hi"}, True),
                                   ("/v1/chat/completions", BODY, False)]:
            response = client.post(path, json={**body, "stream": stream})
            assert response.headers["x-request-id"] == "req-oai"
            assert "1970" not in response.headers.get("date", "")
            assert "alt-svc" not in response.headers and response.headers.get("server") != "cloudflare"

def test_responses_stream_is_relayed_unchanged(directory, fake):
    raw = sse(text_events()) + b'event: new.event\ndata: {"unknown":true}\n\n'
    compressed = gzip.compress(raw)
    stream = Chunks(compressed)
    fake.api = lambda r: httpx.Response(200, stream=stream, headers={
        "content-encoding": "gzip", "content-type": "text/event-stream", "connection": "keep-alive, x-hop",
        "x-hop": "remove", "set-cookie": "a=1"})
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/responses", json={"model": "m", "input": "Hi", "stream": True})
        assert response.content == raw
        assert "content-encoding" not in response.headers  # relayed decoded so terminal events can be patched
        assert not any(h in response.headers for h in ("x-hop", "connection", "set-cookie"))
        assert stream.closed


@pytest.mark.parametrize("body,param", [({"model": "m", "input": "x", "previous_response_id": "r"}, "previous_response_id"),
                                        ({"model": "m", "input": "x", "background": True}, "background"),
                                        ({"model": "m", "input": "x", "stream": "yes"}, "stream")])
def test_responses_unsupported_features(directory, fake, body, param):
    with running(directory, fake) as (client, fake, _):
        response = client.post("/v1/responses", json=body)
        assert response.status_code == 400 and response.json()["error"]["param"] == param
        response = client.post("/v1/responses", json={"model": "m", "input": "x", "store": True})
        assert response.status_code == 200
    with running(directory, fake, strict=True) as (client, *_):
        assert client.post("/v1/responses", json={"model": "m", "input": "x", "store": True}).status_code == 400


# Errors and boundaries

@pytest.mark.parametrize("status", [400, 403, 429, 500])
@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
def test_upstream_errors_preserved(directory, fake, status, endpoint):
    body = b'{"error":{"message":"Denied","code":"permission_denied"}}'
    fake.api = lambda r: httpx.Response(status, content=body, headers={
        "x-request-id": "req-denied", "retry-after": "2", "x-ratelimit-remaining-tokens": "100"})
    with running(directory, fake) as (client, fake, app):
        response = client.post("/v1/" + endpoint, json=BODY if endpoint.startswith("chat") else {"model": "m", "input": "x"})
        assert response.status_code == status and response.content == body
        assert response.headers["retry-after"] == "2" and response.headers["x-request-id"] == "req-denied"
        assert response.headers["x-ratelimit-remaining-tokens"] == "100"
        assert len(fake.calls) == 1 and not app.state.active


@pytest.mark.parametrize("body,message,code", [
    (b'{"detail":"Plan usage limit reached"}', "Plan usage limit reached", None),
    (b'{"detail":{"message":"Limit","code":"subscription_sharing_usage_limit_exceeded"}}', "Limit",
     "subscription_sharing_usage_limit_exceeded"),
    (b"<html>gateway</html>", "OpenAI returned HTTP 429.", None),
])
def test_detail_bodies_become_openai_errors(directory, fake, body, message, code):
    fake.api = lambda r: httpx.Response(429, content=body)
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == 429
        assert response.json() == {"error": {"message": message, "type": "rate_limit_error", "param": None, "code": code}}


def test_retryable_admission_503_is_retried(directory, fake, monkeypatch):
    monkeypatch.setattr("siwc_proxy.server.backoff", lambda attempt: 0)
    answers = iter([httpx.Response(503, json={"detail": {"code": "subscription_sharing_usage_unavailable"}},
                                   headers={"retry-after": "0"})])
    fake.api = lambda r: next(answers, None) or streamed(text_events())(r)
    with running(directory, fake) as (client, fake, _):
        assert client.post("/v1/chat/completions", json=BODY).status_code == 200
        assert len(fake.calls) == 2


def test_other_503_is_not_retried(directory, fake):
    fake.api = lambda r: httpx.Response(503, json={"error": {"message": "Down", "code": "overloaded"}})
    with running(directory, fake) as (client, fake, _):
        assert client.post("/v1/chat/completions", json=BODY).status_code == 503
        assert len(fake.calls) == 1


@pytest.mark.parametrize("error,status,attempts", [(httpx.ConnectError("secret network detail"), 502, 3),
                                                   (httpx.ReadTimeout("secret timeout detail"), 504, 1)])
def test_network_errors_are_sanitized(directory, fake, monkeypatch, error, status, attempts):
    monkeypatch.setattr("siwc_proxy.server.backoff", lambda attempt: 0)

    def fail(request):
        raise error
    fake.api = fail
    with running(directory, fake) as (client, fake, _):
        response = client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == status and "secret" not in response.text
        assert len(fake.calls) == attempts  # only connection failures are retried


@pytest.mark.parametrize("streaming", [False, True])
def test_network_error_after_stream_start(directory, fake, streaming):
    class Broken(Chunks):
        async def __aiter__(self):
            yield sse(text_events()[:4])
            raise httpx.ReadTimeout("private provider detail")

    broken = Broken(b"")
    fake.api = lambda r: httpx.Response(200, stream=broken)
    with running(directory, fake) as (client, *_):
        response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
        if streaming:
            assert parse_chunks(response)[-1]["error"]["code"] == "upstream_timeout"
            assert "[DONE]" not in response.text
        else:
            assert response.status_code == 504
        assert "private" not in response.text and broken.closed


@pytest.mark.parametrize("path", ["/v1/models", "/v1/responses", "/v1/chat/completions", "/auth/status", "/v1/unknown"])
def test_local_key_and_host_checked_before_openai(directory, fake, path, caplog):
    with running(directory, fake) as (client, fake, _):
        method = client.get if path.endswith(("models", "status")) else client.post
        for authorization in ("", "Bearer wrong", "Basic " + KEY):
            response = method(path, headers={"Authorization": authorization})
            assert response.status_code == 401 and response.json()["error"]["code"] == "invalid_api_key"
            assert response.headers["www-authenticate"] == "Bearer"
        assert method(path, headers={"Host": "evil.example"}).status_code == 400
        assert method(path, headers={"Host": "127.0.0.1:9999"}).status_code == 400
        assert not fake.calls
        assert not any(secret in caplog.text for secret in (KEY, "oauth-secret", "refresh-secret"))


def test_invalid_bodies_do_not_reach_openai(directory, fake):
    with running(directory, fake) as (client, fake, _):
        for path in ("/v1/chat/completions", "/v1/responses"):
            for content in (b"invalid", b"[]", b"null"):
                response = client.post(path, content=content)
                assert response.status_code == 400 and response.json()["error"]["type"] == "invalid_request_error"
        assert client.post("/v1/chat/completions", content=b"{}").status_code == 400
        assert not fake.calls


@pytest.mark.parametrize("host,allowed,ok", [("localhost:8787", [], True), ("[::1]:8787", [], True),
                                             ("host.docker.internal:8787", [], False),
                                             ("host.docker.internal:8787", ["host.docker.internal"], True),
                                             ("anything:1", ["*"], True)])
def test_allowed_hosts(directory, fake, host, allowed, ok):
    with running(directory, fake, allowed_hosts=allowed) as (client, *_):
        assert (client.get("/healthz", headers={"Host": host}).status_code == 200) is ok


def test_unknown_routes_are_openai_shaped(directory, fake):
    with running(directory, fake) as (client, *_):
        response = client.get("/v1/embeddings")
        assert response.status_code == 404 and response.json()["error"]["type"] == "not_found_error"
        response = client.get("/v1/chat/completions")
        assert response.status_code == 405 and "error" in response.json()


# Models

def test_models_catalog_cache_and_retrieve(directory, fake):
    fake.api = lambda r: httpx.Response(200, json={"models": [
        {"slug": "test-model", "display_name": "Test", "visibility": "list"},
        {"slug": "hidden", "visibility": "hidden"}]})
    with running(directory, fake) as (client, fake, _):
        listing = client.get("/v1/models").json()
        assert [m["id"] for m in listing["data"]] == ["test-model"]
        assert listing["data"][0]["object"] == "model" and listing["data"][0]["owned_by"] == "openai"
        assert client.get("/v1/models/test-model").json()["display_name"] == "Test"
        missing = client.get("/v1/models/hidden")
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "model_not_found"
        assert len(fake.calls) == 1



def test_models_are_slimmed(directory, fake):
    fake.api = lambda r: httpx.Response(200, json={"models": [{
        "slug": "big", "visibility": "list", "display_name": "Big", "context_window": 1000,
        "supported_reasoning_levels": [{"effort": "low", "description": "fast"}, {"effort": "high"}],
        "base_instructions": "x" * 50_000, "shell_type": "default"}]})
    with running(directory, fake) as (client, *_):
        assert client.get("/v1/models").json()["data"] == [{
            "id": "big", "object": "model", "created": 0, "owned_by": "openai", "display_name": "Big",
            "context_window": 1000, "supported_reasoning_levels": ["low", "high"]}]

def test_invalid_catalog(directory, fake):
    fake.api = lambda r: httpx.Response(200, json={"models": "nope"})
    with running(directory, fake) as (client, *_):
        assert client.get("/v1/models").status_code == 502


# Credentials seen by the server

def test_signed_out_proxy_starts_and_points_to_login(tmp_path, fake):
    with running(tmp_path / "empty", fake) as (client, fake, _):
        response = client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == 401
        error = response.json()["error"]
        assert error["code"] == "chatgpt_login_required" and BASE + "/auth/start/" in error["message"]
        assert client.get("/auth/status").json()["state"] == "signed_out"
        assert not fake.calls


def test_generated_api_key_is_saved_privately(tmp_path):
    settings = Settings(data_dir=tmp_path / "keyed")
    key = settings.resolve_api_key()
    assert key.startswith("sk-siwc-") and settings.resolve_api_key() == key
    assert (tmp_path / "keyed" / "proxy_key").stat().st_mode & 0o777 == 0o600


def test_changed_credentials_and_scope_are_read_each_request(directory, fake):
    with running(directory, fake) as (client, fake, _):
        write_credentials(directory, access_token="replaced-token")
        assert client.post("/v1/chat/completions", json=BODY).status_code == 200
        assert fake.calls[-1].headers["authorization"] == "Bearer replaced-token"
        write_credentials(directory, scope="openid offline_access")
        response = client.post("/v1/responses", json={"model": "m", "input": "x"})
        assert response.status_code == 403 and "consent" in response.json()["error"]["message"]
        assert len(fake.calls) == 1


def test_corrupt_credentials(directory, fake):
    with running(directory, fake) as (client, *_):
        (directory / "credentials.json").write_text("{not json")
        response = client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == 500 and response.json()["error"]["code"] == "credential_file_error"


# Real HTTP streaming

@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
@pytest.mark.parametrize("disconnect", [False, True])
def test_real_http_streaming_and_disconnect_cleanup(directory, fake, endpoint, disconnect):
    gate, closed = threading.Event(), threading.Event()

    class Delayed(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse(text_events()[:4])
            while not gate.is_set():
                await anyio.sleep(0.01)
            yield sse(text_events()[4:])

        async def aclose(self):
            closed.set()

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    fake.api = lambda r: httpx.Response(200, stream=Delayed(), headers={"content-type": "text/event-stream"})
    app = create_app(Settings(data_dir=directory, api_key=KEY, port=port), transport=fake.transport)
    body = {**BODY, "stream": True} if endpoint.startswith("chat") else {"model": "m", "input": "Hi", "stream": True}
    try:
        with live_server(app, sock):
            with httpx.Client(timeout=2, trust_env=False) as client:
                with client.stream("POST", f"http://127.0.0.1:{port}/v1/{endpoint}", headers=HEADERS,
                                   json=body) as response:
                    assert response.status_code == 200
                    lines = response.iter_lines()
                    for line in lines:
                        if '"Hi "' in line:
                            break
                    else:
                        pytest.fail("No incremental text arrived")
                    assert not gate.is_set() and not closed.is_set()
                    if not disconnect:
                        gate.set()
                        rest = "\n".join(lines)
                        assert "[DONE]" in rest if endpoint.startswith("chat") else "response.completed" in rest
            assert closed.wait(2), "Disconnect did not close the upstream stream"
    finally:
        gate.set()


def test_healthz_needs_no_key(directory):
    app = create_app(Settings(data_dir=directory, api_key=KEY))
    with TestClient(app, base_url=BASE) as client:
        assert client.get("/healthz").json() == {"status": "ok", "service": "siwc-proxy"}
