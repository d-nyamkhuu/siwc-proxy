"""Refresh rules: single-flight across tasks and processes, what is kept on failure, upstream 401 retry."""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import anyio
import httpx2
import pytest

from siwc_proxy.auth import ChatGPTAuth
from siwc_proxy.config import Settings
from siwc_proxy.constants import RESOURCE
from siwc_proxy.errors import OpenAIError
from siwc_proxy.server import create_app
from siwc_proxy.store import Store
from support import BASE, BODY, HEADERS, KEY, assert_private, id_token, running, saved, write_credentials


def auth_for(directory, fake):
    return ChatGPTAuth(Settings(data_dir=directory, api_key=KEY), Store(directory), transport=fake.transport)


@pytest.fixture
def expired(tmp_path):
    folder = tmp_path / "expired"
    write_credentials(folder, expires_at=int(time.time()) - 10)
    return folder


def test_refresh_request_and_rotation_are_saved(expired, fake):
    token = anyio.run(lambda: auth_for(expired, fake).access_token())
    assert token == "refreshed-1"
    [form] = fake.token_calls
    assert form == {"grant_type": "refresh_token", "refresh_token": "refresh-secret", "resource": RESOURCE,
                    "client_id": "oaiapp_registered"}
    account = saved(expired)["account"]
    assert account["refresh_token"] == "rotated-1" and account["access_token"] == "refreshed-1"
    assert account["expires_at"] > time.time() + 3000
    assert "chatgpt.tokens.use.direct" in account["scope"]  # omitted scope keeps the previous grant
    assert_private(expired / "credentials.json")


def test_valid_token_is_not_refreshed(directory, fake):
    assert anyio.run(lambda: auth_for(directory, fake).access_token()) == "oauth-secret"
    assert not fake.calls


def test_concurrent_requests_refresh_once(expired, fake):
    fake.token_delay = 0.1
    fake.api = lambda r: httpx2.Response(200, json={"object": "list", "data": []})

    async def exercise():
        app = create_app(Settings(data_dir=expired, api_key=KEY), transport=fake.transport)
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app), base_url=BASE, headers=HEADERS) as client:
                results = await asyncio_gather([client.get("/v1/models?n=" + str(i)) for i in range(8)])
        assert all(r.status_code == 200 for r in results)
        assert {c.headers["authorization"] for c in fake.calls if c.url.host == "api.openai.com"} == {
            "Bearer refreshed-1"}

    anyio.run(exercise)
    assert len(fake.token_calls) == 1


async def asyncio_gather(coroutines):
    results = [None] * len(coroutines)

    async def run(i, coroutine):
        results[i] = await coroutine
    async with anyio.create_task_group() as tasks:
        for i, coroutine in enumerate(coroutines):
            tasks.start_soon(run, i, coroutine)
    return results


def test_separate_stores_share_the_file_lock(expired, fake):
    """Two Store objects behave like two processes: the second reuses the first one's rotation."""
    fake.token_delay = 0.1

    async def exercise():
        first, second = auth_for(expired, fake), auth_for(expired, fake)
        return await asyncio_gather([first.access_token(), second.access_token()])

    assert anyio.run(exercise) == ["refreshed-1", "refreshed-1"]
    assert len(fake.token_calls) == 1


def test_processes_refresh_once(expired, tmp_path):
    log = tmp_path / "refreshes.log"
    script = textwrap.dedent(f"""
        import sys, time, anyio, httpx2
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from support import FakeOpenAI
        from siwc_proxy.auth import ChatGPTAuth
        from siwc_proxy.config import Settings
        from siwc_proxy.store import Store

        fake = FakeOpenAI()
        def token(form):
            with open({str(log)!r}, "a") as f:
                f.write(form["refresh_token"] + "\\n")
            time.sleep(0.3)
            stamp = str(time.monotonic_ns())
            return httpx2.Response(200, json={{"access_token": "a-" + stamp, "refresh_token": "r-" + stamp,
                                              "expires_in": 3600}})
        fake.token = token
        directory = {str(expired)!r}
        auth = ChatGPTAuth(Settings(data_dir=directory, api_key="k"), Store(directory), transport=fake.transport)
        print(anyio.run(auth.access_token))
    """)
    processes = [subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
                 for _ in range(3)]
    tokens = {p.communicate(timeout=60)[0].strip() for p in processes}
    assert all(p.returncode == 0 for p in processes)
    assert log.read_text().splitlines() == ["refresh-secret"]
    assert len(tokens) == 1 and tokens == {saved(expired)["account"]["access_token"]}


@pytest.mark.parametrize("answer", [
    lambda form: httpx2.Response(500, text="<html>oops</html>"),
    lambda form: httpx2.Response(503, json={"error": "temporarily_unavailable"}),
    lambda form: httpx2.Response(400, json={"error": "invalid_request"}),
    lambda form: httpx2.Response(200, json={"token_type": "Bearer"}),
])
def test_transient_failures_keep_credentials(expired, fake, answer):
    before = (expired / "credentials.json").read_bytes()
    fake.token = answer
    with pytest.raises(OpenAIError) as caught:
        anyio.run(lambda: auth_for(expired, fake).access_token())
    assert caught.value.status == 502 and caught.value.code == "credential_refresh_failed"
    assert (expired / "credentials.json").read_bytes() == before


def test_network_failure_keeps_credentials(expired, fake):
    before = (expired / "credentials.json").read_bytes()

    def fail(form):
        raise httpx2.ReadTimeout("secret")
    fake.token = fail
    with pytest.raises(OpenAIError) as caught:
        anyio.run(lambda: auth_for(expired, fake).access_token())
    assert caught.value.status == 504
    assert (expired / "credentials.json").read_bytes() == before


def test_invalid_client_keeps_credentials(expired, fake):
    before = (expired / "credentials.json").read_bytes()
    fake.token = lambda form: httpx2.Response(401, json={"error": "invalid_client"})
    with pytest.raises(OpenAIError) as caught:
        anyio.run(lambda: auth_for(expired, fake).access_token())
    assert caught.value.status == 502 and caught.value.code == "chatgpt_client_rejected"
    assert (expired / "credentials.json").read_bytes() == before


@pytest.mark.parametrize("code", ["invalid_grant", "refresh_token_reused", "refresh_token_expired",
                                  "refresh_token_invalidated", "invalid_refresh_token", "token_expired"])
def test_terminal_refresh_errors_clear_tokens_but_keep_registration(expired, fake, code):
    fake.token = lambda form: httpx2.Response(400, json={"error": code})
    with pytest.raises(OpenAIError) as caught:
        anyio.run(lambda: auth_for(expired, fake).access_token())
    assert caught.value.status == 401 and caught.value.code == "chatgpt_login_required"
    record = saved(expired)
    account = record["account"]
    assert record["ext_agent_host_id"] == "urn:uuid:proxy-test"
    assert account["client_id"] == "oaiapp_registered" and account["subject"] == "test-subject"
    assert "id_token" in account  # kept as id_token_hint
    assert not any(k in account for k in ("access_token", "refresh_token", "expires_at"))


def test_unverifiable_id_token_after_rotation_keeps_new_tokens(expired, fake):
    fake.token = lambda form: httpx2.Response(200, json={
        "access_token": "new-a", "refresh_token": "new-r", "expires_in": 3600, "id_token": "not.a.jwt"})
    assert anyio.run(lambda: auth_for(expired, fake).access_token()) == "new-a"
    assert saved(expired)["account"]["refresh_token"] == "new-r"


def test_identity_change_on_refresh_clears_tokens(expired, fake):
    fake.token = lambda form: httpx2.Response(200, json={
        "access_token": "new-a", "refresh_token": "new-r", "expires_in": 3600,
        "id_token": id_token("oaiapp_registered", sub="someone-else")})
    with pytest.raises(OpenAIError) as caught:
        anyio.run(lambda: auth_for(expired, fake).access_token())
    assert caught.value.status == 401
    assert "access_token" not in saved(expired)["account"]


def test_earliest_refresh_at_is_respected(tmp_path, fake):
    folder = tmp_path / "early"
    now = int(time.time())
    # Inside the leeway window, but the server asked us not to refresh yet and the token still works.
    write_credentials(folder, expires_at=now + 30, earliest_refresh_at=now + 20)
    assert anyio.run(lambda: auth_for(folder, fake).access_token()) == "oauth-secret"
    assert not fake.token_calls
    write_credentials(folder, expires_at=now + 30, earliest_refresh_at=now - 1)
    assert anyio.run(lambda: auth_for(folder, fake).access_token()) == "refreshed-1"


def test_upstream_401_refreshes_and_retries_once(directory, fake):
    statuses = iter([401, 200])
    fake.api = lambda r: httpx2.Response(next(statuses), json={"object": "list", "data": []})
    with running(directory, fake) as (client, fake, _):
        assert client.get("/v1/models").status_code == 200
        api = [c for c in fake.calls if c.url.host == "api.openai.com"]
        assert [c.headers["authorization"] for c in api] == ["Bearer oauth-secret", "Bearer refreshed-1"]
        assert len(fake.token_calls) == 1


def test_second_upstream_401_is_relayed(directory, fake):
    fake.api = lambda r: httpx2.Response(401, json={"error": {"message": "Bad token", "code": "invalid_token"}})
    with running(directory, fake) as (client, fake, _):
        response = client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == 401 and response.json()["error"]["code"] == "invalid_token"
        assert len([c for c in fake.calls if c.url.host == "api.openai.com"]) == 2
        assert len(fake.token_calls) == 1


def test_refresh_skipped_when_another_caller_already_replaced_the_token(directory, fake):
    async def exercise():
        auth = auth_for(directory, fake)
        write_credentials(directory, access_token="someone-else-refreshed")
        return await auth.access_token(force=True, rejected="oauth-secret")

    assert anyio.run(exercise) == "someone-else-refreshed"
    assert not fake.token_calls
