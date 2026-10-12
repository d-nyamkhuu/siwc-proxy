"""The siwc-proxy command line."""

import json
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit

import httpx
from click.testing import CliRunner
from fastapi.responses import JSONResponse

from siwc_proxy import cli, server
from siwc_proxy.config import Settings
from support import KEY, FakeOpenAI, live_server, saved, write_credentials


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


def test_running_proxy_is_trusted_only_after_proving_the_key(directory, fake):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    settings = Settings(data_dir=directory, port=port)
    with live_server(server.create_app(Settings(data_dir=directory, api_key=KEY, port=port), transport=fake.transport),
                     sock):
        client = cli.running_proxy(settings, KEY)
        assert client.get("/auth/status").json()["state"] == "ready"
        client.close()
        assert cli.running_proxy(settings, "sk-siwc-other") is None


def test_key_is_not_sent_to_an_impostor_on_the_port(tmp_path):
    seen = []

    async def impostor(scope, receive, send):
        if scope["type"] == "lifespan":
            while (await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        seen.append(dict(scope["headers"]).get(b"authorization"))
        await JSONResponse({"status": "ok", "service": "siwc-proxy", "proof": "0" * 64})(scope, receive, send)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    settings = Settings(data_dir=tmp_path, port=sock.getsockname()[1])
    with live_server(impostor, sock):
        assert cli.running_proxy(settings, KEY) is None
    assert seen == [None]


def test_login_refuses_an_unexpected_link_from_the_proxy(tmp_path, monkeypatch):
    def proxy(request):
        if request.url.path == "/auth/status":
            return httpx.Response(200, json={"state": "signed_out"})
        return httpx.Response(200, json={"url": "file:///etc/passwd"})

    opened = []
    monkeypatch.setattr(cli, "running_proxy", lambda settings, key: httpx.Client(
        base_url=settings.origin, transport=httpx.MockTransport(proxy)))
    monkeypatch.setattr(cli, "open_link", lambda url, browser: opened.append(url))
    result = CliRunner().invoke(cli.main, ["--data-dir", str(tmp_path), "login", "--no-browser"])
    assert result.exit_code == 1 and "unexpected sign-in link" in result.output and not opened


def test_login_through_a_running_proxy(directory, fake, monkeypatch):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = server.create_app(Settings(data_dir=directory, api_key=KEY, port=port), transport=fake.transport)
    opened = []

    def browser(url, open_browser):
        opened.append(url)
        write_credentials(directory, expires_at=int(time.time()) + 7200)  # as if sign-in completed

    monkeypatch.setattr(cli, "open_link", browser)
    with live_server(app, sock):
        result = CliRunner().invoke(cli.main, ["--data-dir", str(directory), "login", "--no-browser"],
                                    env={"SIWC_PROXY_PORT": str(port), "SIWC_PROXY_API_KEY": KEY})
    assert result.exit_code == 0, result.output
    assert opened[0].startswith(f"http://127.0.0.1:{port}/auth/start/") and "Signed in as" in result.output
