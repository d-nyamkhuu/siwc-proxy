"""The siwc-proxy command line."""

import json
import threading
from urllib.parse import parse_qs, urlsplit

import httpx
from click.testing import CliRunner

from siwc_proxy import cli, server
from support import FakeOpenAI, saved


def invoke(directory, *args, **kwargs):
    return CliRunner().invoke(cli.main, ["--data-dir", str(directory), *args], catch_exceptions=False, **kwargs)


def test_status_hides_tokens_and_shows_key_on_request(directory):
    result = invoke(directory, "status", "--json")
    status = json.loads(result.stdout)
    assert status["state"] == "ready" and status["email"] == "user@example.com"
    assert "api_key" not in status and "oauth-secret" not in result.output
    status = json.loads(invoke(directory, "status", "--json", "--show-key").stdout)
    assert status["api_key"].startswith("sk-siwc-")


def test_token_prints_only_the_token(directory):
    result = invoke(directory, "token")
    assert result.exit_code == 0 and result.stdout == "oauth-secret\n"


def test_token_when_signed_out(tmp_path):
    result = CliRunner().invoke(cli.main, ["--data-dir", str(tmp_path), "token"])
    assert result.exit_code == 1 and "siwc-proxy login" in result.output and not result.stdout.strip()


def test_logout_without_network_still_clears_locally(directory, monkeypatch):
    monkeypatch.setattr("siwc_proxy.auth.backoff", lambda attempt: 0)
    result = invoke(directory, "logout")
    assert result.exit_code == 0
    assert "access_token" not in saved(directory)["account"]


def test_standalone_login_serves_its_own_callback(tmp_path, monkeypatch):
    """`login` with no proxy running hosts the callback on a free port and exits when sign-in completes."""
    fake = FakeOpenAI()
    original = server.create_app
    monkeypatch.setattr(server, "create_app", lambda settings, **kw: original(settings, transport=fake.transport, **kw))
    monkeypatch.setattr(cli, "running_proxy", lambda settings, key: None)

    def browser(url, open_browser):
        def act():
            with httpx.Client(trust_env=False, timeout=5) as client:
                response = client.get(url, follow_redirects=False)
                params = {k: v[0] for k, v in parse_qs(urlsplit(response.headers["location"]).query).items()}
                fake.codes["c"] = ("oaiapp_cli", params["nonce"], "openid " + "chatgpt.tokens.use.direct", "sub-1")
                origin = urlsplit(url)
                client.get(f"{origin.scheme}://{origin.netloc}/auth/callback",
                           params={"code": "c", "state": params["state"], "client_id": "oaiapp_cli"})
        threading.Thread(target=act, daemon=True).start()

    monkeypatch.setattr(cli, "open_link", browser)
    folder = tmp_path / "cli"
    result = invoke(folder, "login", "--no-browser", "--timeout", "20")
    assert result.exit_code == 0, result.output
    assert "Signed in as user@example.com" in result.output
    assert saved(folder)["account"]["client_id"] == "oaiapp_cli"
