"""Sign in with ChatGPT on Authlib: authorization code + PKCE, ID-token checks, refresh, revocation.

Authlib does the OAuth/OIDC work (state, PKCE, nonce, discovery, JWKS, token requests). This module
adds the SIWC rules: the dynamic-client registration swap, `resource` on token requests, identity and
scope checks, cross-process refresh serialization, and when to keep or clear saved tokens.
"""

import logging
import random
import secrets
import time

import anyio
import httpx
from authlib.integrations.base_client import OAuthError
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.integrations.starlette_client import OAuth
from authlib.oauth2.rfc6749 import OAuth2Token
from joserfc.errors import JoseError

from .constants import (AGENT_NAME, DYNAMIC_CLIENT, ISSUER, LEEWAY, METADATA_URL, RESOURCE, SCOPES,
                        TERMINAL_REFRESH_ERRORS)
from .errors import OpenAIError
from .store import Account, Record

log = logging.getLogger("siwc_proxy.auth")

LAUNCH_TTL = 900
BACKOFF_BASE = 0.5


def backoff(attempt):
    """Jittered exponential delay for attempt 1, 2, ..."""
    return min(8.0, BACKOFF_BASE * 2 ** (attempt - 1)) * (0.5 + random.random() / 2)


def login_required(message):
    return OpenAIError(message, status=401, code="chatgpt_login_required", error_type="authentication_error")


def consent_required(message):
    return OpenAIError(message, status=403, code="chatgpt_consent_required", error_type="permission_error")


def refresh_failed(timeout=False):
    return OpenAIError("OpenAI could not refresh the ChatGPT session; saved credentials were kept. Try again.",
                       status=504 if timeout else 502, code="credential_refresh_failed")


class SignInError(Exception):
    """A browser sign-in that did not complete; the message is shown on the callback page."""


class MemoryCache:
    """Server-side Authlib state (PKCE verifier, nonce); the session cookie only carries a marker."""

    def __init__(self):
        self._items = {}

    async def get(self, key):
        value, expires = self._items.get(key, (None, 0))
        if expires < time.monotonic():
            self._items.pop(key, None)
            return None
        return value

    async def set(self, key, value, expires=None):
        now = time.monotonic()
        self._items = {k: v for k, v in self._items.items() if v[1] >= now}
        self._items[key] = (value, now + (expires or LAUNCH_TTL))

    async def delete(self, key):
        self._items.pop(key, None)


class ChatGPTAuth:
    def __init__(self, settings, store, *, transport=None, timeout=30):
        self.settings, self.store = settings, store
        self.http = {"trust_env": False, "timeout": timeout}
        if transport is not None:
            self.http["transport"] = transport
        self.cache = MemoryCache()
        self.metadata = {}
        self.launches = {}
        self.serving = False
        self.signed_in = None  # anyio.Event set by a completed browser sign-in while serving

    # Authlib clients

    def remote(self, client_id):
        """Authlib app for one client_id; all share the state cache and the loaded discovery/JWKS."""
        remote = OAuth(cache=self.cache).register(
            "chatgpt",
            client_id=client_id,
            server_metadata_url=METADATA_URL,
            access_token_params={"resource": RESOURCE},
            client_kwargs={"scope": SCOPES, "code_challenge_method": "S256",
                           "token_endpoint_auth_method": "none", **self.http},
        )
        remote.server_metadata = self.metadata
        remote.framework.expires_in = LAUNCH_TTL
        return remote

    async def server_metadata(self):
        return await self.remote(DYNAMIC_CLIENT).load_server_metadata()

    @staticmethod
    def claims_options(client_id):
        return {"iss": {"essential": True, "value": ISSUER},
                "aud": {"essential": True, "value": client_id},
                "sub": {"essential": True}}

    def client(self, account, **kwargs):
        return AsyncOAuth2Client(client_id=account.client_id, token_endpoint_auth_method="none",
                                 revocation_endpoint_auth_method="none", leeway=LEEWAY, **self.http, **kwargs)

    # Browser sign-in

    def login_url(self, consent=False):
        if not self.serving:
            return None
        now = time.monotonic()
        self.launches = {k: v for k, v in self.launches.items() if v > now + 60}
        launch = next(iter(self.launches), None)
        if launch is None:
            launch = secrets.token_urlsafe(24)
            self.launches[launch] = now + LAUNCH_TTL
        return f"{self.settings.origin}/auth/start/{launch}" + ("?consent=true" if consent else "")

    def check_launch(self, launch):
        now = time.monotonic()
        return any(secrets.compare_digest(launch.encode(), k.encode()) and v > now for k, v in self.launches.items())

    def login_hint(self, consent=False):
        url = self.login_url(consent)
        command = "siwc-proxy login" + (" --consent" if consent else "")
        return f"open {url} or run `{command}`" if url else f"run `{command}`"

    async def authorize_redirect(self, request, *, consent=False):
        async with self.store.transaction():
            record = self.store.read()
            if record is None:
                record = Record()
                self.store.write(record)  # ext_agent_host_id must stay stable across sign-ins
        account = record.account
        client_id = account.client_id if account else DYNAMIC_CLIENT
        params = {"resource": RESOURCE, "ext_agent_host_id": record.ext_agent_host_id}
        if client_id == DYNAMIC_CLIENT:
            params["agent_name_hint"] = AGENT_NAME
        if account and account.email:
            params["login_hint"] = account.email
        if account and account.id_token:
            params["id_token_hint"] = account.id_token
        if consent:
            params["prompt"] = "consent"
        return await self.remote(client_id).authorize_redirect(request, self.settings.redirect_uri, **params)

    async def complete_sign_in(self, request):
        """Finish the callback and save the account; returns the saved record."""
        query = request.query_params
        record = self.store.read() or Record()
        account = record.account
        expected = account.client_id if account else DYNAMIC_CLIENT
        # A first sign-in returns the client_id issued to this installation; later ones must keep it.
        issued = query.get("client_id") or expected
        if account and issued != account.client_id:
            raise SignInError("OpenAI returned a different registration than the saved one. Sign-in was not saved.")
        remote = self.remote(issued)
        state = query.get("state")
        # Checked first so a callback from another browser does not consume the real one's state.
        if not state or await remote.framework.get_state_data(request.session, state) is None:
            raise SignInError("This sign-in is invalid, expired, or was started in another browser. Start again.")
        try:
            token = await remote.authorize_access_token(
                request, claims_options=self.claims_options(issued), leeway=LEEWAY)
        except OAuthError as error:
            log.warning("Sign-in failed: %s", error.error)
            if error.error == "access_denied":
                raise SignInError("Sign-in was cancelled.") from None
            raise SignInError("OpenAI did not complete the sign-in. Start again.") from None
        except (JoseError, ValueError, KeyError, TypeError) as error:
            log.warning("Sign-in failed: invalid ID token (%s)", type(error).__name__)
            raise SignInError("OpenAI returned an identity token that could not be verified.") from None
        except httpx.HTTPError as error:
            log.warning("Sign-in failed: %s", type(error).__name__)
            raise SignInError("Could not reach OpenAI to finish sign-in. Start again.") from None
        claims = token.pop("userinfo", None)
        if not claims or not token.get("access_token"):
            raise SignInError("OpenAI did not return an identity token. Start again.")
        if account and account.subject and claims["sub"] != account.subject:
            raise SignInError("That is a different ChatGPT account than the saved one. "
                              "Run `siwc-proxy logout --forget` to switch accounts.")
        async with self.store.transaction():
            current = self.store.read() or record
            saved = current.account if current.account and current.account.client_id == issued else None
            saved = saved or Account(client_id=issued)
            saved.subject, saved.issuer = claims["sub"], ISSUER
            saved.email, saved.name = claims.get("email"), claims.get("name")
            # A code exchange that omits scope granted what was requested (RFC 6749 §5.1).
            saved.apply_tokens(token, default_scope=SCOPES)
            current.account = saved
            self.store.write(current)
        self.launches.clear()
        log.info("Signed in to ChatGPT (%s).", current.state)
        if self.signed_in is not None:
            self.signed_in.set()
        return current

    # Tokens

    def require_ready(self, record):
        state = record.state if record else "signed_out"
        if state == "ready":
            return record.account
        if state == "consent_required":
            raise consent_required("ChatGPT plan usage is not allowed for this sign-in; " + self.login_hint(True) + ".")
        raise login_required("Sign in with ChatGPT: " + self.login_hint() + ".")

    @staticmethod
    def refresh_due(account):
        if not OAuth2Token(account.token()).is_expired(leeway=LEEWAY):
            return False  # Unknown expiry counts as valid; an upstream 401 forces a refresh.
        now = time.time()
        # Honor earliest_refresh_at while the current token still works.
        return not (account.earliest_refresh_at and now < account.earliest_refresh_at
                    and account.expires_at and now < account.expires_at)

    async def access_token(self, *, force=False, rejected=None):
        """A usable access token, refreshing at most once across threads and processes.

        `force` refreshes even before expiry (after an upstream 401); `rejected` is the token that was
        refused, so a token another caller already replaced is reused instead of refreshed again.
        """
        account = self.require_ready(self.store.read())
        if not force and not self.refresh_due(account):
            return account.access_token
        async with self.store.transaction():
            record = self.store.read()
            account = self.require_ready(record)
            if force and rejected is not None and account.access_token != rejected:
                return account.access_token
            if not force and not self.refresh_due(account):
                return account.access_token
            await self._refresh(record)
            return self.require_ready(record).access_token

    async def _refresh(self, record):
        account = record.account
        try:
            metadata = await self.server_metadata()
        except (httpx.HTTPError, ValueError) as error:
            log.warning("Refresh failed: discovery %s", type(error).__name__)
            raise refresh_failed(isinstance(error, httpx.TimeoutException)) from None
        previous_scope, issued = account.scope, {}

        async def persist(token, refresh_token=None, access_token=None):
            # Saved before any other check: the old refresh token is already spent.
            if not token.get("access_token"):
                raise ValueError("token response without access_token")
            account.apply_tokens(token, default_scope=previous_scope)
            self.store.write(record)
            issued.update(token)

        try:
            async with self.client(account, token=account.token(), update_token=persist) as client:
                await client.refresh_token(metadata["token_endpoint"], refresh_token=account.refresh_token,
                                           resource=RESOURCE)
        except OAuthError as error:
            if error.error in TERMINAL_REFRESH_ERRORS:
                log.warning("Refresh rejected (%s); sign-in required.", error.error)
                account.clear_tokens()
                self.store.write(record)
                raise login_required(f"The ChatGPT session ended ({error.error}). Sign in again: "
                                     + self.login_hint() + ".") from None
            if error.error == "invalid_client":
                log.warning("Refresh rejected (invalid_client); saved credentials were kept.")
                raise OpenAIError("OpenAI rejected this proxy's client registration (invalid_client); saved "
                                  "credentials were kept. If it persists, run `siwc-proxy logout --forget` and sign "
                                  "in again.", status=502, code="chatgpt_client_rejected") from None
            log.warning("Refresh failed: %s", error.error)
            raise refresh_failed() from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            log.warning("Refresh failed: %s", type(error).__name__)
            raise refresh_failed(isinstance(error, httpx.TimeoutException)) from None
        log.info("Refreshed the ChatGPT access token.")
        if issued.get("id_token"):
            await self._check_identity(record, issued)

    async def _check_identity(self, record, token):
        account = record.account
        try:
            claims = await self.remote(account.client_id).parse_id_token(
                token, nonce=None, claims_options=self.claims_options(account.client_id), leeway=LEEWAY)
        except (JoseError, ValueError, KeyError, TypeError, httpx.HTTPError) as error:
            log.warning("Could not verify the refreshed ID token (%s); keeping tokens.", type(error).__name__)
            return
        if account.subject and claims["sub"] != account.subject:
            log.warning("Refreshed identity changed; clearing tokens.")
            account.clear_tokens()
            self.store.write(record)
            raise login_required("The ChatGPT identity changed. Sign in again: " + self.login_hint() + ".")

    # Sign-out and status

    async def logout(self, *, forget=False):
        """Revoke and clear tokens, keeping the registration unless `forget`.

        Returns True when revocation was confirmed, False when it failed, None when nothing was signed in.
        """
        async with self.store.transaction():
            record = self.store.read()
            account = record.account if record else None
            revoked = None
            if account and (account.refresh_token or account.access_token):
                revoked = await self._revoke(account)
            if forget:
                self.store.delete()
            elif account:
                account.clear_tokens()
                self.store.write(record)
        return revoked

    async def _revoke(self, account):
        token, hint = ((account.refresh_token, "refresh_token") if account.refresh_token
                       else (account.access_token, "access_token"))
        attempts = 4
        for attempt in range(1, attempts + 1):
            try:
                url = (await self.server_metadata()).get("revocation_endpoint")
                if not url:
                    log.warning("OpenAI does not advertise a revocation endpoint; tokens were removed locally.")
                    return False
                async with self.client(account) as client:
                    response = await client.revoke_token(url, token=token, token_type_hint=hint)
                if response.is_success:
                    return True
                if response.status_code < 500:
                    log.warning("OpenAI rejected token revocation (HTTP %s).", response.status_code)
                    return False
            except httpx.TransportError as error:
                log.warning("Revocation attempt %s failed: %s", attempt, type(error).__name__)
            if attempt < attempts:
                await anyio.sleep(backoff(attempt))
        log.warning("Token revocation did not complete; tokens were removed locally.")
        return False

    def status(self):
        record = self.store.read()
        account = record.account if record else None
        result = {"state": record.state if record else "signed_out", "registered": account is not None}
        if account:
            result.update(email=account.email, name=account.name, subject=account.subject,
                          client_id=account.client_id, scopes=sorted(account.scopes),
                          expires_at=account.expires_at, earliest_refresh_at=account.earliest_refresh_at)
        if result["state"] != "ready" and self.serving:
            result["login_url"] = self.login_url(consent=result["state"] == "consent_required")
        return result
