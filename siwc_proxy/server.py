"""FastAPI app: OpenAI-compatible /v1 routes, sign-in routes, and the local access boundary."""

import hashlib
import hmac
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from html import escape

import anyio
import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from . import responses as responses_api
from .auth import ChatGPTAuth, SignInError, backoff
from .chat import ChatStream, translate_request
from .config import Settings
from .constants import RESOURCE, RETRYABLE_CODES, USAGE_URL
from .errors import OpenAIError, error_code, no_retry, normalize_error_body, protocol_error
from .sse import encode_sse, response_events, with_terminal_output
from .store import CredentialFileError, Store

log = logging.getLogger("siwc_proxy")

REQUEST_HEADERS = {"openai-beta", "idempotency-key", "x-client-request-id"}
# Upstream response headers clients use: request ids, retry hints, rate limits. Everything else describes
# OpenAI's edge or the ChatGPT backend rather than this server, and is dropped.
RESPONSE_HEADERS = {"x-request-id", "retry-after", "retry-after-ms", "x-should-retry"}
RESPONSE_PREFIXES = ("openai-", "x-ratelimit-", "x-codex-")
# Catalog fields worth exposing; the rest (Codex prompt templates and client hints) is ~80 KB per model.
MODEL_FIELDS = ("display_name", "description", "context_window", "max_context_window", "input_modalities",
                "default_reasoning_level", "supported_reasoning_levels", "supports_parallel_tool_calls",
                "support_verbosity", "supports_search_tool")
MAX_RETRIES = 2


def is_public(path):
    return path in ("/healthz", "/auth/callback") or path.startswith("/auth/start/")


def healthz_proof(api_key, challenge):
    """Lets the CLI check that a server knows the key before sending it."""
    return hmac.new(api_key.encode(), ("siwc-proxy healthz " + challenge).encode(), hashlib.sha256).hexdigest()


def printable(text, limit=200):
    """Log-safe text: control characters cannot forge or split log lines."""
    return "".join(c if c.isprintable() else "?" for c in text[:limit])


class Boundary:
    """Host and bearer-key checks before any body is read, no-store responses, and access logs."""

    def __init__(self, app, settings, api_key):
        self.app, self.settings, self.key = app, settings, api_key.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if not self.settings.host_allowed(headers.get("host")):
            error = OpenAIError("Host not allowed; start the proxy with --allowed-host to permit it.", code="invalid_host")
            await error.response()(scope, receive, send)
            return
        if not is_public(scope["path"]):
            scheme, _, supplied = headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not secrets.compare_digest(supplied.encode(), self.key):
                error = OpenAIError("Invalid local proxy API key.", status=401, code="invalid_api_key",
                                    error_type="authentication_error")
                await error.response()(scope, receive, send)
                return

        started, seen = time.perf_counter(), {}

        async def no_store(message):
            if message["type"] == "http.response.start":
                message["headers"] = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                message["headers"].append((b"cache-control", b"no-store"))
                seen["status"] = message["status"]
                seen["request_id"] = Headers(raw=message["headers"]).get("x-request-id")
            await send(message)

        try:
            await self.app(scope, receive, no_store)
        finally:
            model = scope.get("state", {}).get("model")
            path = "/auth/start/…" if scope["path"].startswith("/auth/start/") else printable(scope["path"])
            log.info("%s %s %s%s%s %.0fms", scope["method"], path, seen.get("status", "-"),
                     f" model={printable(model)}" if isinstance(model, str) else "",
                     f" request_id={printable(seen['request_id'])}" if seen.get("request_id") else "",
                     (time.perf_counter() - started) * 1000)


class Relayed(Exception):
    """Carries a ready response (an upstream error) out of a helper."""

    def __init__(self, response):
        self.response = response


class UpstreamStreamingResponse(StreamingResponse):
    """Close the upstream stream even when an ASGI send fails between iterations."""

    def __init__(self, *args, upstream, active, **kwargs):
        super().__init__(*args, **kwargs)
        self.upstream, self.active = upstream, active

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    if hasattr(self.body_iterator, "aclose"):
                        await self.body_iterator.aclose()
                finally:
                    await self.upstream.aclose()
                    self.active.discard(self.upstream)


def response_headers(response, *, translated=False):
    # Bodies are always decoded (relayed streams may be patched), so encoding and framing headers never apply.
    headers = {k: v for k, v in response.headers.items()
               if k.lower() in RESPONSE_HEADERS or k.lower().startswith(RESPONSE_PREFIXES)
               or (k.lower() == "content-type" and not translated)}
    # The ChatGPT backend names its request id x-oai-request-id; SDKs read x-request-id.
    if "x-request-id" not in response.headers and "x-oai-request-id" in response.headers:
        headers["x-request-id"] = response.headers["x-oai-request-id"]
    return headers


def model_entry(model):
    entry = {"id": model["slug"], "object": "model", "created": model.get("created", 0),
             "owned_by": model.get("owned_by", "openai")}
    entry.update((key, model[key]) for key in MODEL_FIELDS if model.get(key) is not None)
    levels = entry.get("supported_reasoning_levels")
    if isinstance(levels, list):
        entry["supported_reasoning_levels"] = [level.get("effort") if isinstance(level, dict) else level
                                               for level in levels]
    return entry


def network_error(error):
    if isinstance(error, httpx2.TimeoutException):
        return OpenAIError("OpenAI request timed out.", status=504, code="upstream_timeout")
    return OpenAIError("Could not reach OpenAI.", status=502, code="upstream_connection_error")


def retry_delay(response, attempt):
    try:
        return min(10.0, max(0.0, float(response.headers.get("retry-after", ""))))
    except ValueError:
        return backoff(attempt)


def ignored_header(ignored):
    return {"x-siwc-ignored-params": ",".join(ignored)} if ignored else {}


def parse_json(content):
    try:
        return json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        raise OpenAIError("Request body must be valid JSON.") from None


def page(message, status=200):
    html = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
            "<title>SIWC proxy</title><body style='font:16px system-ui;margin:3rem auto;max-width:36rem;padding:0 1rem'>"
            f"<p>{escape(message)}</p>")
    return HTMLResponse(html, status_code=status, headers={"Referrer-Policy": "no-referrer"})


def create_app(settings=None, *, transport=None, api_key=None):
    """`transport` replaces the network for both OAuth and inference (tests)."""
    settings = settings or Settings()
    api_key = api_key or settings.resolve_api_key()
    store = Store(settings.data_dir)
    auth = ChatGPTAuth(settings, store, transport=transport)
    session_secret = secrets.token_bytes(32)  # Per process: cache sessions only need to last a conversation.

    @asynccontextmanager
    async def lifespan(app):
        app.state.active = set()
        auth.serving, auth.signed_in = True, anyio.Event()
        timeout = httpx2.Timeout(30, read=settings.read_timeout)
        # The transport retries connection failures, before any request is sent.
        upstream_transport = transport or httpx2.AsyncHTTPTransport(retries=MAX_RETRIES)
        async with httpx2.AsyncClient(transport=upstream_transport, trust_env=False, follow_redirects=False,
                                     timeout=timeout) as client:
            app.state.upstream = client
            try:
                status = auth.status()
                if status["state"] != "ready":
                    log.warning("ChatGPT %s. Sign in: %s", status["state"].replace("_", " "), status["login_url"])
            except CredentialFileError as error:
                log.error("%s", error)
            try:
                yield
            finally:
                auth.serving = False
                with anyio.CancelScope(shield=True):
                    for response in list(app.state.active):
                        await response.aclose()
                    app.state.active.clear()

    app = FastAPI(title="SIWC OpenAI-compatible proxy", lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.state.settings, app.state.auth, app.state.store = settings, auth, store
    app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32), session_cookie="siwc_proxy_session",
                       max_age=900, same_site="lax", https_only=False)
    app.add_middleware(Boundary, settings=settings, api_key=api_key)

    @app.exception_handler(OpenAIError)
    async def openai_error(request, error):
        return error.response()

    @app.exception_handler(Relayed)
    async def relayed(request, error):
        return error.response

    @app.exception_handler(httpx2.HTTPError)
    async def http_error(request, error):
        return network_error(error).response()

    @app.exception_handler(CredentialFileError)
    async def credential_file_error(request, error):
        return OpenAIError(str(error), status=500, code="credential_file_error").response()

    @app.exception_handler(StarletteHTTPException)
    async def not_found(request, error):
        if error.status_code == 404:
            message = f"Unknown request URL: {request.method} {request.url.path}."
            return OpenAIError(message, status=404, code="unknown_url").response()
        return OpenAIError(str(error.detail), status=error.status_code).response(error.headers)

    async def close(response):
        with anyio.CancelScope(shield=True):
            await response.aclose()
            app.state.active.discard(response)

    async def send(request, method, path, body=None, *, query=None):
        """Open an upstream stream; refresh once on 401 and retry admission-time 503s.

        Nothing is retried once the response body is being relayed.
        """
        client = app.state.upstream
        headers = {k: v for k, v in request.headers.items() if k.lower() in REQUEST_HEADERS}
        headers["Accept-Encoding"] = "identity"
        content = None
        if body is not None:
            session = responses_api.cache_session(body, session_secret)
            headers.update({"Content-Type": "application/json", "Accept": "text/event-stream",
                            "session_id": request.headers.get("session_id") or session})
            content = json.dumps(body).encode()
        token, refreshed, attempt = await auth.access_token(), False, 0
        while True:
            upstream = client.build_request(method, RESOURCE + path, headers={**headers, "Authorization": "Bearer " + token},
                                            content=content)
            # Preserve repeated query parameters and their original percent-encoding.
            upstream.url = upstream.url.copy_with(query=query or None)
            response = await client.send(upstream, stream=True)
            if response.status_code == 401 and not refreshed:
                # Admission rejected the token, so no inference ran: refresh and resend once.
                await response.aclose()
                refreshed = True
                token = await auth.access_token(force=True, rejected=token)
                continue
            if response.status_code == 503 and attempt < MAX_RETRIES:
                await response.aread()
                await response.aclose()
                if error_code(response.content) in RETRYABLE_CODES:
                    attempt += 1
                    log.info("Retrying after upstream %s", error_code(response.content))
                    await anyio.sleep(retry_delay(response, attempt))
                    continue
                return response
            if body is not None and response.is_success and "content-type" not in response.headers:
                # The ChatGPT backend can stream Responses events without a Content-Type; SSE decoders and
                # clients of the relayed stream need one.
                response.headers["Content-Type"] = "text/event-stream"
            app.state.active.add(response)
            return response

    async def upstream_error(response):
        try:
            content = await response.aread()
        finally:
            await close(response)
        return Response(normalize_error_body(response.status_code, content), status_code=response.status_code,
                        headers={**response_headers(response, translated=True), **no_retry(error_code(content))},
                        media_type="application/json")

    def relay(response, extra=None):
        async def raw():
            try:
                async for chunk in with_terminal_output(response.aiter_bytes()):
                    yield chunk
            finally:
                await close(response)
        return UpstreamStreamingResponse(raw(), status_code=response.status_code,
                                         headers={**response_headers(response), **(extra or {})},
                                         upstream=response, active=app.state.active)

    # Health and sign-in

    @app.get("/healthz")
    async def healthz(challenge: str | None = None):
        result = {"status": "ok", "service": "siwc-proxy"}
        if challenge:
            result["proof"] = healthz_proof(api_key, challenge[:128])
        return result

    @app.get("/auth/status")
    async def auth_status():
        return auth.status()

    @app.post("/auth/login")
    async def login(consent: bool = False):
        return {"url": auth.login_url(consent=consent)}

    @app.post("/auth/logout")
    async def logout(forget: bool = False):
        revoked = await auth.logout(forget=forget)
        return {"state": "signed_out" if forget else auth.status()["state"], "revoked": revoked}

    @app.get("/auth/start/{launch}")
    async def start(launch: str, request: Request, consent: bool = False):
        if not auth.check_launch(launch):
            return page("This sign-in link expired. Run `siwc-proxy login` for a new one.", 404)
        if request.headers.get("host") != settings.origin.removeprefix("http://"):
            # The state cookie must be set on the host OpenAI redirects back to.
            query = "?" + request.url.query if request.url.query else ""
            return RedirectResponse(settings.origin + request.url.path + query, status_code=303)
        return await auth.authorize_redirect(request, consent=consent)

    @app.get("/auth/callback")
    async def callback(request: Request):
        if request.headers.get("host") != settings.origin.removeprefix("http://"):
            return page("Open the sign-in link on 127.0.0.1.", 400)
        try:
            record = await auth.complete_sign_in(request)
        except SignInError as error:
            return page(str(error), 400)
        who = record.account.email or "your ChatGPT account"
        if record.state == "consent_required":
            return page(f"Signed in as {who}, but ChatGPT plan usage was not allowed. "
                        "Run `siwc-proxy login --consent` to allow it.")
        return page(f"Signed in as {who}. Requests through this proxy now use your ChatGPT plan; "
                    f"manage usage in ChatGPT settings ({USAGE_URL}). You can close this tab.")

    # OpenAI API

    async def catalog(request):
        response = await send(request, "GET", "/models", query=request.scope["query_string"])
        if not response.is_success:
            raise Relayed(await upstream_error(response))
        try:
            await response.aread()
            payload = response.json()
            if not isinstance(payload, dict):
                raise protocol_error("OpenAI returned an invalid model catalog.")
            if "models" in payload:
                entries = payload["models"]
                if not isinstance(entries, list):
                    raise protocol_error("OpenAI returned an invalid model catalog.")
                payload = {"object": "list", "data": [
                    model_entry(model) for model in entries if model.get("visibility") == "list"]}
            elif payload.get("object") != "list" or not isinstance(payload.get("data"), list):
                raise protocol_error("OpenAI returned an invalid model catalog.")
        except (ValueError, KeyError, TypeError, AttributeError):
            raise protocol_error("OpenAI returned an invalid model catalog.") from None
        finally:
            await close(response)
        return payload, response_headers(response, translated=True)

    @app.get("/v1/models")
    async def models(request: Request):
        payload, headers = await catalog(request)
        return JSONResponse(payload, headers=headers)

    @app.get("/v1/models/{model_id:path}")
    async def model(model_id: str, request: Request):
        payload, headers = await catalog(request)
        for entry in payload["data"]:
            if entry.get("id") == model_id:
                return JSONResponse(entry, headers=headers)
        raise OpenAIError(f"The model '{model_id}' does not exist or is not available.", "model",
                          status=404, code="model_not_found")

    @app.post("/v1/responses")
    async def responses(request: Request):
        body, streaming, ignored = responses_api.normalize(parse_json(await request.body()), strict=settings.strict)
        request.state.model = body.get("model")
        response = await send(request, "POST", "/responses", body)
        if not response.is_success:
            return await upstream_error(response)
        if streaming:
            return relay(response, ignored_header(ignored))
        headers = {**response_headers(response, translated=True), **ignored_header(ignored)}
        try:
            result = await responses_api.aggregate(response_events(response))
        except OpenAIError as error:
            return error.response(headers)
        except httpx2.HTTPError as error:
            return network_error(error).response(headers)
        finally:
            await close(response)
        return JSONResponse(result, headers=headers)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = parse_json(await request.body())
        translated, ignored = translate_request(body, strict=settings.strict)
        request.state.model = translated["model"]
        options = body.get("stream_options")
        include_usage = bool(body.get("stream") and isinstance(options, dict) and options.get("include_usage"))
        response = await send(request, "POST", "/responses", translated)
        if not response.is_success:
            return await upstream_error(response)
        headers = {**response_headers(response, translated=True), **ignored_header(ignored)}
        state = ChatStream(translated["model"], include_usage=include_usage)

        async def converted():
            try:
                async for event in response_events(response):
                    try:
                        chunks = state.consume(event)
                    except (KeyError, TypeError, ValueError, AttributeError):
                        raise protocol_error() from None
                    for chunk in chunks:
                        yield chunk
                    if state.terminal is not None:
                        return
                raise protocol_error()
            finally:
                await close(response)

        if body.get("stream"):
            async def sse():
                try:
                    async for chunk in converted():
                        yield encode_sse(chunk)
                    yield b"data: [DONE]\n\n"
                except OpenAIError as error:
                    yield encode_sse(error.payload)
                except httpx2.HTTPError as error:
                    yield encode_sse(network_error(error).payload)
            return UpstreamStreamingResponse(sse(), headers=headers, media_type="text/event-stream",
                                             upstream=response, active=app.state.active)
        try:
            async for _ in converted():
                pass
            return JSONResponse(state.completion(), headers=headers)
        except OpenAIError as error:
            return error.response(headers)
        except httpx2.HTTPError as error:
            return network_error(error).response(headers)

    return app
