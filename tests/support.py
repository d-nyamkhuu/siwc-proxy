"""Shared fakes: saved credentials, an in-memory OpenAI (issuer + API), and SSE helpers."""

import json
import os
import threading
import time
from contextlib import contextmanager
from urllib.parse import parse_qs

import httpx
import uvicorn
from fastapi.testclient import TestClient
from joserfc import jwt
from joserfc.jwk import KeySet, RSAKey

from siwc_proxy.config import Settings
from siwc_proxy.constants import DIRECT_SCOPE, ISSUER, SCOPES
from siwc_proxy.server import create_app

KEY = "local-test-key"
HEADERS = {"Authorization": "Bearer " + KEY}
BASE = "http://127.0.0.1:8787"
BODY = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
TEXT_ITEM = {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Hi 世界", "annotations": []}]}
SIGNING_KEY = RSAKey.generate_key(2048, parameters={"kid": "test-key"}, private=True)


def terminal(output=None, status="completed", **extra):
    return {"id": "resp-test", "object": "response", "model": "test-model", "created_at": 123,
            "status": status, "output": [TEXT_ITEM] if output is None else output,
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                      "input_tokens_details": {"cached_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 1}},
            **extra}


def text_events():
    return [
        {"type": "response.created", "response": terminal([], status="in_progress")},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "reasoning", "id": "private"}},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hi "},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "世界"},
        {"type": "response.output_text.done", "output_index": 0, "content_index": 0, "text": "Hi 世界"},
        {"type": "response.completed", "response": terminal()},
    ]


def tool_events():
    a = {"type": "function_call", "call_id": "call-a", "name": "a", "arguments": ""}
    b = {"type": "function_call", "call_id": "call-b", "name": "b", "arguments": ""}
    return [
        {"type": "response.output_item.added", "output_index": 1, "item": a},
        {"type": "response.output_item.added", "output_index": 3, "item": b},
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '{"x":'},
        {"type": "response.function_call_arguments.delta", "output_index": 3, "delta": '{}'},
        {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '1}'},
        {"type": "response.function_call_arguments.done", "output_index": 1, "arguments": '{"x":1}'},
        {"type": "response.output_item.done", "output_index": 3, "item": {**b, "arguments": "{}"}},
        {"type": "response.completed", "response": terminal([
            {"type": "reasoning"}, {**a, "arguments": '{"x":1}'}, {"type": "reasoning"}, {**b, "arguments": '{}'}])},
    ]


def sse(events):
    return "".join("event: " + e["type"] + "\ndata: " + json.dumps(e, ensure_ascii=False) + "\n\n"
                   for e in events).encode()


def parse_chunks(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


class Chunks(httpx.AsyncByteStream):
    """A streamed body (unlike content=, which httpx pre-reads) that records being closed."""

    def __init__(self, data, size=7):
        self.data, self.size, self.closed = data, size, False

    async def __aiter__(self):
        for i in range(0, len(self.data), self.size):
            yield self.data[i:i + self.size]

    async def aclose(self):
        self.closed = True


def id_token(client_id, sub="test-subject", nonce=None, **claims):
    now = int(time.time())
    payload = {"iss": ISSUER, "aud": client_id, "sub": sub, "iat": now, "exp": now + 600,
               "email": "user@example.com", "name": "Test User", **claims}
    if nonce:
        payload["nonce"] = nonce
    return jwt.encode({"alg": "RS256", "kid": "test-key"}, payload, SIGNING_KEY)


def write_credentials(directory, **account):
    record = {"version": 2, "ext_agent_host_id": "urn:uuid:proxy-test", "account": {
        "client_id": "oaiapp_registered", "subject": "test-subject", "email": "user@example.com",
        "scope": SCOPES, "access_token": "oauth-secret", "refresh_token": "refresh-secret",
        "id_token": id_token("oaiapp_registered"), "expires_at": int(time.time()) + 3600, **account}}
    record["account"] = {k: v for k, v in record["account"].items() if v is not None}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "credentials.json").write_text(json.dumps(record))


def saved(directory):
    return json.loads((directory / "credentials.json").read_text())


def assert_private(path):
    """Owner-only file mode on POSIX; on Windows files rely on the profile folder's access rules."""
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


class FakeOpenAI:
    """auth.openai.com (discovery, JWKS, token, revocation) and api.openai.com in one MockTransport."""

    def __init__(self):
        self.calls, self.token_calls, self.revocations = [], [], []
        self.api = lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream", "x-request-id": "req-test"},
            stream=Chunks(sse(text_events())))
        self.token = None  # optional override: callable(form) -> httpx.Response
        self.revoke = lambda form: httpx.Response(200)
        self.codes = {}  # code -> (client_id, nonce, scope, sub)
        self.rotation = 0
        self.token_delay = 0

    @property
    def transport(self):
        return httpx.MockTransport(self.handle)

    async def handle(self, request):
        self.calls.append(request)
        url = str(request.url.copy_with(query=None))
        if url == ISSUER + "/.well-known/openid-configuration":
            return httpx.Response(200, json={
                "issuer": ISSUER, "authorization_endpoint": ISSUER + "/oauth/authorize",
                "token_endpoint": ISSUER + "/oauth/token", "jwks_uri": ISSUER + "/jwks",
                "revocation_endpoint": ISSUER + "/oauth/revoke",
                "id_token_signing_alg_values_supported": ["RS256"]})
        if url == ISSUER + "/jwks":
            return httpx.Response(200, json=KeySet([SIGNING_KEY]).as_dict(private=False))
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if url == ISSUER + "/oauth/token":
            self.token_calls.append(form)
            if self.token_delay:
                import anyio
                await anyio.sleep(self.token_delay)
            if self.token:
                return self.token(form)
            return self.default_token(form)
        if url == ISSUER + "/oauth/revoke":
            self.revocations.append(form)
            return self.revoke(form)
        result = self.api(request)
        if hasattr(result, "__await__"):
            result = await result
        return result

    def default_token(self, form):
        if form["grant_type"] == "authorization_code":
            client_id, nonce, scope, sub = self.codes[form["code"]]
            assert form["client_id"] == client_id
            return httpx.Response(200, json={
                "access_token": "new-access", "refresh_token": "new-refresh", "token_type": "Bearer",
                "expires_in": 3600, "scope": scope, "id_token": id_token(client_id, sub, nonce)})
        self.rotation += 1
        return httpx.Response(200, json={
            "access_token": f"refreshed-{self.rotation}", "refresh_token": f"rotated-{self.rotation}",
            "token_type": "Bearer", "expires_in": 3600})


@contextmanager
def running(directory, fake=None, **settings):
    fake = fake or FakeOpenAI()
    app = create_app(Settings(data_dir=directory, api_key=KEY, **settings), transport=fake.transport)
    with TestClient(app, base_url=BASE, headers=HEADERS) as client:
        yield client, fake, app


@contextmanager
def live_server(app, sock):
    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise AssertionError("Test proxy failed to start")
        time.sleep(0.01)
    try:
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


__all__ = ["BASE", "BODY", "Chunks", "DIRECT_SCOPE", "FakeOpenAI", "HEADERS", "KEY", "TEXT_ITEM", "id_token",
           "live_server", "parse_chunks", "running", "saved", "sse", "terminal", "text_events", "tool_events",
           "write_credentials"]
