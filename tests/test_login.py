"""Browser sign-in through Authlib: SIWC authorize parameters, issued client_id, ID-token checks, logout."""

import time
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx
import pytest

from siwc_proxy.auth import ChatGPTAuth
from siwc_proxy.config import Settings
from siwc_proxy.constants import DIRECT_SCOPE, RESOURCE, SCOPES
from siwc_proxy.store import Store
from support import BASE, BODY, KEY, id_token, running, saved, write_credentials


def start(client, consent=False):
    """Follow the launch link to OpenAI's authorize URL and return its query parameters."""
    url = client.post("/auth/login", params={"consent": consent}).json()["url"]
    assert url.startswith(BASE + "/auth/start/")
    response = client.get(url.removeprefix(BASE), follow_redirects=False)
    assert response.status_code == 302
    location = urlsplit(response.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == "https://auth.openai.com/oauth/authorize"
    return {k: v[0] for k, v in parse_qs(location.query).items()}


def approve(fake, params, issued=None, scope=SCOPES, sub="test-subject", code="code-1"):
    fake.codes[code] = (issued or params["client_id"], params["nonce"], scope, sub)
    query = {"code": code, "state": params["state"]}
    if issued:
        query["client_id"] = issued
    return query


def test_first_sign_in_registers_and_saves(tmp_path, fake):
    folder = tmp_path / "new"
    with running(folder, fake) as (client, fake, app):
        assert client.post("/v1/chat/completions", json=BODY).status_code == 401
        params = start(client)
        assert params["client_id"] == "dynamic_agent_client"
        assert params["agent_name_hint"] == "SIWC OpenAI proxy"
        assert params["resource"] == RESOURCE
        assert params["redirect_uri"] == BASE + "/auth/callback"
        assert params["code_challenge_method"] == "S256" and params["code_challenge"]
        assert params["scope"] == SCOPES and params["response_type"] == "code"
        assert params["ext_agent_host_id"].startswith("urn:uuid:")
        assert not any(k in params for k in ("id_token_hint", "login_hint", "prompt"))

        page = client.get("/auth/callback", params=approve(fake, params, issued="oaiapp_issued"))
        assert page.status_code == 200 and "Signed in as user@example.com" in page.text
        assert page.headers["referrer-policy"] == "no-referrer"
        exchange = fake.token_calls[-1]
        assert exchange["client_id"] == "oaiapp_issued" and exchange["resource"] == RESOURCE
        assert exchange["code_verifier"] and "client_secret" not in exchange

        record = saved(folder)
        assert record["ext_agent_host_id"] == params["ext_agent_host_id"]
        account = record["account"]
        assert account["client_id"] == "oaiapp_issued" and account["subject"] == "test-subject"
        assert account["access_token"] == "new-access" and DIRECT_SCOPE in account["scope"]
        assert (folder / "credentials.json").stat().st_mode & 0o777 == 0o600
        assert app.state.auth.signed_in.is_set()
        assert client.get("/auth/status").json()["state"] == "ready"
        assert client.post("/v1/chat/completions", json=BODY).status_code == 200


def test_reauth_reuses_registration_and_sends_hints(directory, fake):
    with running(directory, fake) as (client, fake, _):
        params = start(client, consent=True)
        assert params["client_id"] == "oaiapp_registered"
        assert params["ext_agent_host_id"] == "urn:uuid:proxy-test"
        assert params["login_hint"] == "user@example.com"
        assert params["id_token_hint"] == saved(directory)["account"]["id_token"]
        assert params["prompt"] == "consent"
        assert "agent_name_hint" not in params
        assert client.get("/auth/callback", params=approve(fake, params)).status_code == 200
        assert saved(directory)["account"]["access_token"] == "new-access"


def test_different_issued_client_is_rejected(directory, fake):
    with running(directory, fake) as (client, fake, _):
        params = start(client)
        page = client.get("/auth/callback", params=approve(fake, params, issued="oaiapp_other"))
        assert page.status_code == 400 and "different registration" in page.text
        assert saved(directory)["account"]["access_token"] == "oauth-secret"


def test_different_account_is_rejected(directory, fake):
    with running(directory, fake) as (client, fake, _):
        params = start(client)
        page = client.get("/auth/callback", params=approve(fake, params, sub="someone-else"))
        assert page.status_code == 400 and "logout --forget" in page.text
        assert saved(directory)["account"]["access_token"] == "oauth-secret"


def test_missing_direct_scope_needs_consent(tmp_path, fake):
    with running(tmp_path / "new", fake) as (client, fake, _):
        params = start(client)
        page = client.get("/auth/callback", params=approve(fake, params, issued="oaiapp_x", scope="openid email"))
        assert "login --consent" in page.text
        response = client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == 403 and response.json()["error"]["code"] == "chatgpt_consent_required"
        assert "consent=true" in client.get("/auth/status").json()["login_url"]


@pytest.mark.parametrize("tamper", ["state", "nonce", "audience", "issuer", "expired", "signature"])
def test_callback_validation(tmp_path, fake, tamper):
    folder = tmp_path / "new"
    with running(folder, fake) as (client, fake, _):
        params = start(client)
        query = approve(fake, params, issued="oaiapp_issued")
        claims = {}
        if tamper == "state":
            query["state"] = "forged"
        elif tamper == "nonce":
            fake.codes["code-1"] = ("oaiapp_issued", "other-nonce", SCOPES, "test-subject")
        elif tamper == "audience":
            claims = {"aud": "oaiapp_other"}
        elif tamper == "issuer":
            claims = {"iss": "https://evil.example"}
        elif tamper == "expired":
            claims = {"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200}
        if claims or tamper == "signature":
            def token(form):
                value = id_token("oaiapp_issued", nonce=params["nonce"], **claims)
                if tamper == "signature":
                    value = value[:-4] + ("AAAA" if not value.endswith("AAAA") else "BBBB")
                return httpx.Response(200, json={"access_token": "a", "refresh_token": "r", "expires_in": 60,
                                                 "id_token": value})
            fake.token = token
        page = client.get("/auth/callback", params=query)
        assert page.status_code == 400
        assert not (folder / "credentials.json").exists() or "account" not in saved(folder)


def test_state_is_single_use_and_bound_to_the_browser(tmp_path, fake):
    with running(tmp_path / "new", fake) as (client, fake, app):
        params = start(client)
        query = approve(fake, params, issued="oaiapp_issued")
        client.cookies.clear()  # a different browser
        assert client.get("/auth/callback", params=query).status_code == 400
    with running(tmp_path / "again", fake) as (client, fake, _):
        params = start(client)
        query = approve(fake, params, issued="oaiapp_issued")
        assert client.get("/auth/callback", params=query).status_code == 200
        assert client.get("/auth/callback", params=query).status_code == 400


def test_cancelled_sign_in(tmp_path, fake):
    with running(tmp_path / "new", fake) as (client, fake, _):
        params = start(client)
        page = client.get("/auth/callback", params={"error": "access_denied", "state": params["state"]})
        assert page.status_code == 400 and "cancelled" in page.text


def test_callback_requires_the_registered_host(directory, fake):
    with running(directory, fake) as (client, *_):
        page = client.get("/auth/callback", params={"code": "c", "state": "s"}, headers={"Host": "localhost:8787"})
        assert page.status_code == 400


def test_start_moves_to_the_registered_host(directory, fake):
    with running(directory, fake) as (client, *_):
        url = client.post("/auth/login").json()["url"].removeprefix(BASE)
        response = client.get(url, headers={"Host": "localhost:8787"}, follow_redirects=False)
        assert response.status_code == 303 and response.headers["location"] == BASE + url
        assert client.get("/auth/start/forged", follow_redirects=False).status_code == 404


def test_logout_revokes_and_keeps_registration(directory, fake):
    with running(directory, fake) as (client, fake, _):
        assert client.post("/auth/logout").json() == {"state": "reauth_required", "revoked": True}
        assert fake.revocations == [{"token": "refresh-secret", "token_type_hint": "refresh_token",
                                     "client_id": "oaiapp_registered"}]
        account = saved(directory)["account"]
        assert account["client_id"] == "oaiapp_registered" and "refresh_token" not in account
        assert client.post("/v1/chat/completions", json=BODY).status_code == 401
        assert start(client)["client_id"] == "oaiapp_registered"


def test_logout_retries_server_errors_then_gives_up_locally(directory, fake, monkeypatch):
    monkeypatch.setattr("siwc_proxy.auth.backoff", lambda attempt: 0)
    fake.revoke = lambda form: httpx.Response(503)
    auth = ChatGPTAuth(Settings(data_dir=directory, api_key=KEY), Store(directory), transport=fake.transport)
    assert anyio.run(auth.logout) is False
    assert len(fake.revocations) == 4
    assert "access_token" not in saved(directory)["account"]


def test_logout_forget_removes_registration(directory, fake):
    auth = ChatGPTAuth(Settings(data_dir=directory, api_key=KEY), Store(directory), transport=fake.transport)
    assert anyio.run(lambda: auth.logout(forget=True)) is True
    assert not (directory / "credentials.json").exists()


def test_status_never_contains_tokens(directory, fake):
    write_credentials(directory)
    with running(directory, fake) as (client, *_):
        text = client.get("/auth/status").text
        assert "oauth-secret" not in text and "refresh-secret" not in text and "eyJ" not in text
