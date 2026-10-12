"""The siwc-proxy command line."""

import io
import json
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit

import click
import httpx
import pytest
from click.testing import CliRunner
from fastapi.responses import JSONResponse

from siwc_proxy import cli, server
from siwc_proxy.auth import ChatGPTAuth
from siwc_proxy.config import Settings
from support import KEY, FakeOpenAI, live_server, saved, write_credentials


def invoke(directory, *args, **kwargs):
    return CliRunner().invoke(cli.main, ["--data-dir", str(directory), *args], catch_exceptions=False, **kwargs)


@pytest.mark.parametrize("args", [["--data-dir", "{flag}", "status"], ["status", "--data-dir", "{flag}"],
                                  ["--data-dir", "{other}", "status", "--data-dir", "{flag}"], ["status"]],
                         ids=["before", "after", "after-wins", "env"])
def test_data_dir_before_or_after_the_command(tmp_path, args):
    folders = {"flag": str(tmp_path / "flag"), "other": str(tmp_path / "other")}
    result = CliRunner().invoke(cli.main, [arg.format(**folders) for arg in args] + ["--json"],
                                env={"SIWC_PROXY_DATA_DIR": str(tmp_path / "env")}, catch_exceptions=False)
    expected = tmp_path / ("env" if args == ["status"] else "flag")
    assert json.loads(result.stdout)["data_dir"] == str(expected.resolve())


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


@pytest.mark.parametrize("by_flag", [False, True], ids=["env", "flag"])
def test_login_through_a_running_proxy(directory, fake, monkeypatch, by_flag):
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
        args, env = (["--port", str(port)], {}) if by_flag else ([], {"SIWC_PROXY_PORT": str(port)})
        result = CliRunner().invoke(cli.main, ["--data-dir", str(directory), "login", "--no-browser", *args],
                                    env={**env, "SIWC_PROXY_API_KEY": KEY})
    assert result.exit_code == 0, result.output
    assert opened[0].startswith(f"http://127.0.0.1:{port}/auth/start/") and "Signed in as" in result.output


class Paste(io.RawIOBase):
    """Terminal input typed only once the prompt is reached, so it can depend on the printed link."""

    def __init__(self, make):
        self.make, self.data = make, None

    def readable(self):
        return True

    def readinto(self, buffer):
        if not len(buffer):
            return 0
        if self.data is None:
            self.data = self.make().encode()
        count = min(len(buffer), len(self.data))
        buffer[:count], self.data = self.data[:count], self.data[count:]
        return count


def remote_login(folder, monkeypatch, answer, *args):
    """Run `login --remote`; `answer(params)` gets the authorize URL's query and returns what is pasted."""
    fake = FakeOpenAI()
    links = []

    class Auth(ChatGPTAuth):
        def __init__(self, settings, store):
            super().__init__(settings, store, transport=fake.transport)

        async def pasted_sign_in_url(self, session, consent=False):
            links.append(await super().pasted_sign_in_url(session, consent))
            return links[-1]

    monkeypatch.setattr(cli, "ChatGPTAuth", Auth)
    monkeypatch.setattr(cli, "running_proxy", lambda *a: pytest.fail("--remote must not use a running proxy"))

    def typed():
        params = {k: v[0] for k, v in parse_qs(urlsplit(links[0]).query).items()}
        return answer(fake, params)

    result = CliRunner().invoke(cli.main, ["--data-dir", str(folder), "login", "--remote", *args],
                                input=Paste(typed))
    return result, links, fake


def test_remote_login_with_a_pasted_address(tmp_path, monkeypatch):
    def answer(fake, params):
        fake.codes["c"] = ("oaiapp_remote", params["nonce"], "openid chatgpt.tokens.use.direct", "sub-1")
        return ("not the address\n"
                f"  {params['redirect_uri']}?code=c&state={params['state']}&client_id=oaiapp_remote  \n")

    folder = tmp_path / "remote"
    result, links, fake = remote_login(folder, monkeypatch, answer, "--port", "40123")
    assert result.exit_code == 0, result.output
    assert links[0].startswith("https://auth.openai.com/oauth/authorize?") and links[0] in result.output
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A40123%2Fauth%2Fcallback" in links[0]
    assert "contains code= and state=" in result.output  # the first paste was rejected and asked again
    assert "Signed in as user@example.com" in result.output
    assert fake.token_calls[-1]["code_verifier"]
    assert saved(folder)["account"]["client_id"] == "oaiapp_remote"


def test_remote_login_reports_a_cancelled_sign_in(tmp_path, monkeypatch):
    result, *_ = remote_login(tmp_path, monkeypatch, lambda fake, params: (
        f"error=access_denied&state={params['state']}\n"))
    assert result.exit_code == 1 and "Sign-in was cancelled." in result.output
    assert not (tmp_path / "credentials.json").exists() or "account" not in saved(tmp_path)


def test_remote_login_rejects_a_forged_state(tmp_path, monkeypatch):
    result, *_ = remote_login(tmp_path, monkeypatch, lambda fake, params: "?code=c&state=forged\n")
    assert result.exit_code == 1 and "invalid, expired" in result.output


@pytest.mark.parametrize("text", ["http://127.0.0.1:5/auth/callback?code=c&state=s", "?code=c&state=s",
                                  "code=c&state=s", "127.0.0.1:5/auth/callback?code=c&state=s"])
def test_pasted_query_accepts_an_address_or_its_query(text):
    assert cli.pasted_query(text) == "code=c&state=s"


@pytest.mark.parametrize("text", ["", "http://127.0.0.1:5/auth/callback", "code=c", "state=s"])
def test_pasted_query_rejects_other_text(text):
    with pytest.raises(click.BadParameter):
        cli.pasted_query(text)
