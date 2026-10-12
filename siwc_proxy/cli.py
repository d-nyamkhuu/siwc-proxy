"""siwc-proxy: serve | login | logout | status | token. Messages go to stderr; `token` prints to stdout."""

import json
import logging
import secrets
import socket
import sys
import time
import webbrowser
from contextlib import closing
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import anyio
import click
import httpx
import uvicorn

from .auth import ChatGPTAuth, SignInError
from .config import Settings
from .constants import USAGE_URL
from .errors import OpenAIError
from .store import CredentialFileError, Store


def remember_data_dir(ctx, param, value):
    if value is not None:
        ctx.ensure_object(dict)["data_dir"] = value


# On the group and on every command, so it works before or after the command name. The environment variable is
# left to Settings, so a flag at either level wins over it.
data_dir_option = click.option(
    "--data-dir", type=click.Path(file_okay=False, path_type=Path), expose_value=False, callback=remember_data_dir,
    help="Credential directory (default: SIWC_PROXY_DATA_DIR, else the per-user data directory).")


def make_settings(ctx, **overrides):
    values = {k: v for k, v in overrides.items() if v not in (None, ())}
    if ctx.obj.get("data_dir"):
        values["data_dir"] = ctx.obj["data_dir"]
    try:
        settings = Settings(**values)
    except ValueError as error:
        raise click.UsageError(str(error)) from None
    return settings


def configure_logging(level):
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every outbound request at INFO; keep that for --log-level debug.
    logging.getLogger("httpx").setLevel(logging.DEBUG if level == "debug" else logging.WARNING)
    # Their debug output includes request headers and token responses.
    for name in ("authlib", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def running_proxy(settings, api_key):
    """An httpx client for a proxy already serving this data directory, or None.

    The key is sent only after the server proves it knows it, so another program on the port never sees it.
    """
    from .server import healthz_proof

    challenge = secrets.token_hex(16)
    client = httpx.Client(base_url=settings.origin, trust_env=False, timeout=10)
    try:
        response = client.get("/healthz", params={"challenge": challenge})
        proof = response.json().get("proof") if response.status_code == 200 else None
        if isinstance(proof, str) and secrets.compare_digest(proof.encode(), healthz_proof(api_key, challenge).encode()):
            client.headers["Authorization"] = "Bearer " + api_key
            response = client.get("/auth/status")
            if response.status_code == 200:
                return client
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    client.close()
    return None


def fail(error):
    if isinstance(error, OpenAIError):
        raise click.ClickException(error.payload["error"]["message"])
    raise click.ClickException(str(error))


def open_link(url, browser):
    click.echo(f"Sign in with ChatGPT: {click.style(url, fg='cyan', bold=True)}", err=True)
    if browser:
        webbrowser.open(url)


@click.group()
@data_dir_option
@click.version_option(package_name="siwc-proxy")
@click.pass_context
def main(ctx):
    """Local OpenAI-compatible API backed by Sign in with ChatGPT."""
    ctx.ensure_object(dict)  # shared with the command, whose own --data-dir lands here too


@main.command()
@data_dir_option
@click.option("--host", help="Interface to bind (default 127.0.0.1). The proxy speaks plain HTTP; "
              "only bind other interfaces on a trusted network.")
@click.option("--port", type=click.IntRange(1, 65535), help="Port (default 8787).")
@click.option("--allowed-host", "allowed_hosts", multiple=True,
              help="Extra Host header to accept, e.g. host.docker.internal. Repeatable; '*' disables the check.")
@click.option("--strict/--lenient", default=None, help="Reject Chat parameters that cannot be honored.")
@click.option("--log-level", type=click.Choice(["debug", "info", "warning", "error"]))
@click.option("--read-timeout", type=click.FloatRange(min=0, min_open=True), help="Upstream read timeout, seconds.")
@click.pass_context
def serve(ctx, **options):
    """Run the proxy."""
    from .server import create_app

    settings = make_settings(ctx, **options)
    configure_logging(settings.log_level)
    if not settings.loopback:
        click.secho(f"Warning: listening on {settings.host} over plain HTTP; the key and prompts are unencrypted "
                    "on the network, and anyone who can reach it and has the key can use your ChatGPT plan.",
                    fg="yellow", err=True)
    try:
        app = create_app(settings)
    except (OSError, ValueError, CredentialFileError) as error:
        fail(error)
    click.echo(f"OpenAI base URL: http://{settings.host}:{settings.port}/v1  (key: `siwc-proxy status --show-key`)",
               err=True)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level,
                access_log=False, proxy_headers=False, server_header=False)


@main.command()
@data_dir_option
@click.option("--consent", is_flag=True, help="Ask again for permission to use the ChatGPT plan.")
@click.option("--port", type=click.IntRange(1, 65535), default=None,
              help="Port of the running proxy (default 8787). With no proxy running, the sign-in callback "
              "listens there (default: any free port).")
@click.option("--timeout", type=click.FloatRange(min=1, max=3600), default=300, show_default=True)
@click.option("--browser/--no-browser", default=True, help="Open the sign-in link in a browser.")
@click.option("--remote", is_flag=True, help="Sign in with a browser on another computer: open the link there, "
              "then paste back the address the browser ends up on.")
@click.pass_context
def login(ctx, consent, port, timeout, browser, remote):
    """Sign in through the browser (via the running proxy if there is one)."""
    settings = make_settings(ctx, port=port)
    configure_logging("warning")
    if remote:
        return remote_login(settings, consent, port)
    api_key = settings.resolve_api_key()
    client = running_proxy(settings, api_key)
    if client is not None:
        with closing(client):  # already used, so `with client` would raise
            def snapshot():
                return {k: v for k, v in client.get("/auth/status").json().items() if k != "login_url"}
            before = snapshot()
            url = client.post("/auth/login", params={"consent": str(consent).lower()}).json().get("url")
            if not isinstance(url, str) or not url.startswith(settings.origin + "/auth/start/"):
                raise click.ClickException("The running proxy returned an unexpected sign-in link.")
            open_link(url, browser)
            end = time.monotonic() + timeout
            # Any completed sign-in changes the saved token expiry, even for the same account.
            while (status := snapshot()) == before:
                if time.monotonic() > end:
                    raise click.ClickException("Sign-in timed out.")
                time.sleep(1)
        return report(status)
    from .server import create_app

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port if port is not None else 0))
    except OSError as error:
        raise click.ClickException(f"Cannot listen on 127.0.0.1:{port}: {error.strerror}") from None
    settings.port = sock.getsockname()[1]
    app = create_app(settings, api_key=api_key)
    auth = app.state.auth
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on"))
    outcome = {}

    async def run():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(server.serve, [sock])
            while not server.started:
                if server.should_exit:
                    return
                await anyio.sleep(0.05)
            open_link(auth.login_url(consent), browser)
            with anyio.move_on_after(timeout) as scope:
                await auth.signed_in.wait()
            outcome["timed_out"] = scope.cancelled_caught
            await anyio.sleep(0.5)  # let the callback page reach the browser
            server.should_exit = True

    try:
        anyio.run(run)
    except KeyboardInterrupt:
        raise click.ClickException("Sign-in cancelled.") from None
    finally:
        sock.close()
    if outcome.get("timed_out", True):
        raise click.ClickException("Sign-in did not complete.")
    report(ChatGPTAuth(settings, Store(settings.data_dir)).status())


def pasted_query(text):
    """The query of a pasted callback address (or of just its `?code=...&state=...` part)."""
    text = text.strip()
    query = urlsplit(text).query if "://" in text else text.partition("?")[2] or text
    params = parse_qs(query)
    if "state" not in params or not ("code" in params or "error" in params):
        raise click.BadParameter("Paste the whole address from the browser's address bar; "
                                 "it starts with http://127.0.0.1 and contains code= and state=.")
    return query


def remote_login(settings, consent, port):
    """Sign in with the browser elsewhere. OpenAI sends it to a 127.0.0.1 callback that cannot load there,
    and the address it was sent to (code and state) is pasted here; PKCE keeps the code useless anywhere else."""
    # The registered redirect URI accepts any loopback port; a random one is unlikely to reach a real service.
    settings.port = port or 49152 + secrets.randbelow(16384)
    auth = ChatGPTAuth(settings, Store(settings.data_dir))
    session = {}
    try:
        url = anyio.run(lambda: auth.pasted_sign_in_url(session, consent))
    except CredentialFileError as error:
        fail(error)
    except (httpx.HTTPError, ValueError) as error:
        raise click.ClickException(f"Could not reach OpenAI to start sign-in ({type(error).__name__}).") from None
    click.echo(f"Open this link in a browser on any computer and sign in:\n\n{click.style(url, fg='cyan', bold=True)}\n\n"
               f"The browser then goes to {settings.redirect_uri}?code=... and fails to load it. That is expected:\n"
               "copy the full address from its address bar and paste it below within 15 minutes.", err=True)
    query = click.prompt("Address", err=True, value_proc=pasted_query)
    try:
        anyio.run(lambda: auth.complete_sign_in(auth.pasted_request(session, query)))
    except SignInError as error:
        raise click.ClickException(str(error)) from None
    except CredentialFileError as error:
        fail(error)
    report(auth.status())


def report(status):
    who = status.get("email") or status.get("subject") or "ChatGPT account"
    if status["state"] == "ready":
        click.echo(f"Signed in as {who}. Requests through the proxy use your ChatGPT plan; "
                   f"manage usage at {USAGE_URL}", err=True)
    elif status["state"] == "consent_required":
        raise click.ClickException(f"Signed in as {who}, but ChatGPT plan usage was not allowed. "
                                   "Run `siwc-proxy login --consent`.")
    else:
        raise click.ClickException("Sign-in did not complete.")


@main.command()
@data_dir_option
@click.option("--forget", is_flag=True, help="Also delete the registration (needed to switch accounts).")
@click.pass_context
def logout(ctx, forget):
    """Revoke and remove saved tokens."""
    settings = make_settings(ctx)
    configure_logging("warning")
    auth = ChatGPTAuth(settings, Store(settings.data_dir))
    try:
        revoked = anyio.run(lambda: auth.logout(forget=forget))
    except CredentialFileError as error:
        fail(error)
    if revoked is None:
        click.echo("Not signed in." if not forget else "Registration removed.", err=True)
    else:
        click.echo("Signed out." if revoked else "Signed out locally, but OpenAI did not confirm revocation. "
                   "You can disconnect the app in ChatGPT settings.", err=True)


@main.command()
@data_dir_option
@click.option("--show-key", is_flag=True, help="Also print the local proxy API key.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output on stdout.")
@click.pass_context
def status(ctx, show_key, as_json):
    """Show the sign-in state (never prints tokens)."""
    settings = make_settings(ctx)
    try:
        result = ChatGPTAuth(settings, Store(settings.data_dir)).status()
    except CredentialFileError as error:
        fail(error)
    result["data_dir"] = str(settings.data_dir)
    result["base_url"] = f"http://{settings.host}:{settings.port}/v1"
    if show_key:
        result["api_key"] = settings.resolve_api_key()
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    for key, value in result.items():
        if value not in (None, [], ""):
            click.echo(f"{key}: {' '.join(value) if isinstance(value, list) else value}")


@main.command()
@data_dir_option
@click.pass_context
def token(ctx):
    """Print a current ChatGPT access token (refreshing if needed) to stdout."""
    settings = make_settings(ctx)
    configure_logging("warning")
    auth = ChatGPTAuth(settings, Store(settings.data_dir))
    try:
        value = anyio.run(lambda: auth.access_token())
    except (OpenAIError, CredentialFileError) as error:
        fail(error)
    sys.stdout.write(value + "\n")
