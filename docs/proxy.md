# siwc-proxy reference

The proxy holds one ChatGPT account's Sign in with ChatGPT (SIWC) credential.
It serves an OpenAI-compatible API on **`http://127.0.0.1:8787/v1`** (note the `/v1`).
Clients authenticate to the proxy with a local bearer key. Upstream, the proxy
replaces that key with the account's OAuth access token. The two credentials are independent.

## Configuration

Every setting can come from a `serve` flag or a `SIWC_PROXY_*` environment variable:

| Flag | Environment | Default | Meaning |
| --- | --- | --- | --- |
| `--data-dir` | `SIWC_PROXY_DATA_DIR` | per-user data dir (`platformdirs`, app `siwc-proxy`) | Credentials, host identity, generated key |
| — | `SIWC_PROXY_API_KEY` | generated into `<data-dir>/proxy_key` | Local bearer key clients must send |
| `--host` | `SIWC_PROXY_HOST` | `127.0.0.1` | Interface to bind. Non-loopback prints a warning. |
| `--port` | `SIWC_PROXY_PORT` | `8787` | Port for the API **and** the OAuth callback |
| `--allowed-host` (repeatable) | `SIWC_PROXY_ALLOWED_HOSTS` (comma-separated) | none | Extra `Host` headers to accept (see below) |
| `--strict` / `--lenient` | `SIWC_PROXY_STRICT` | lenient | Reject Chat options that can't be honored instead of dropping them |
| `--log-level` | `SIWC_PROXY_LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |
| `--read-timeout` | `SIWC_PROXY_READ_TIMEOUT` | `300` | Seconds to wait between upstream bytes |

On first run with the default data directory, credentials from the older
`openai-signin-cli` directory are imported once. This keeps the existing client registration.

### Docker, `localhost`, and LAN access

To block DNS rebinding, the proxy accepts only these `Host` headers: `127.0.0.1`,
`localhost`, and `[::1]`, each on its own port. Add more as needed:

```bash
uv run siwc-proxy serve --allowed-host host.docker.internal
```

```bash
uv run siwc-proxy serve --host 0.0.0.0 --allowed-host my-laptop.lan
```

A bare name matches any port; `name:port` matches exactly; `*` disables the check.
Anyone who can reach a non-loopback listener and has the key can spend your plan usage.

## Signing in

The proxy completes the OAuth redirect itself at `http://127.0.0.1:<port>/auth/callback`.
SIWC accepts any port on that loopback redirect URI.

- **First sign-in** registers a client dynamically (`dynamic_agent_client`) and
  saves the issued `client_id` and a stable `ext_agent_host_id`.
- **Later sign-ins** reuse that `client_id`. They send `id_token_hint` and `login_hint`,
  so you get no new registration screen. `--consent` adds `prompt=consent`;
  use it when plan usage was declined.
- A sign-in that returns a different account or registration is rejected.
  To switch accounts, run `siwc-proxy logout --forget` first.

Ways to start a sign-in:

1. `siwc-proxy serve` prints a link when not signed in.
   API calls also answer `401 chatgpt_login_required` with the link in the message.
2. `siwc-proxy login` with the proxy running asks the proxy for a link, opens it,
   and waits. No restart is needed.
3. `siwc-proxy login` with no proxy running serves the callback on a free
   loopback port (`--port` to pick one) and exits when done.

Links are launch URLs, valid for 15 minutes or until a sign-in completes, whichever
comes first. They redirect to OpenAI with a fresh, single-use OAuth state.
Open them in a browser on the same machine, because the callback must reach `127.0.0.1`.

### Token lifecycle

- Access tokens are refreshed 60 s before expiry, respecting `earliest_refresh_at`.
  Each refresh sends `resource=https://api.openai.com/v1`.
- Refresh is single-flight within the process (an async lock) and across
  processes (`credentials.lock` file lock). After taking the lock, the proxy
  re-reads the file. If another process already rotated the tokens, it adopts
  them instead of spending its own refresh token. You can run `siwc-proxy token`,
  `login`, or a second proxy against the same directory at any time.
- The rotated token set is written atomically (0600) **before** the new ID token
  is validated. The new refresh token is never lost.
- Only terminal OAuth errors clear tokens: `invalid_grant`, `invalid_refresh_token`,
  `token_expired`, `refresh_token_expired`, `refresh_token_invalidated`, and
  `refresh_token_reused`. So does a changed `sub`. `invalid_client` means a
  client-configuration problem, so tokens are kept and requests get a 502
  `chatgpt_client_rejected`.
  Even then, the registration, host id, and ID token (used as `id_token_hint`) are kept.
  Network errors, 5xx, and malformed responses keep everything; the request gets a 502 or 504.
- If OpenAI rejects a token with 401 at admission, the proxy refreshes once and
  retries once. A second 401 is relayed.

`logout` revokes the refresh token, with retries, and clears the tokens.
`--forget` deletes the registration too.

## Routes

| Route | Auth | Notes |
| --- | --- | --- |
| `GET /v1/models`, `GET /v1/models/{id}` | key | Account-visible models in OpenAI list shape, plus `display_name`, `context_window`, `input_modalities`, `supported_reasoning_levels` (effort names) and similar fields. Codex-only fields are dropped. Cached 5 minutes. |
| `POST /v1/responses` | key | Responses API; see below |
| `POST /v1/chat/completions` | key | Chat Completions, translated through Responses |
| `GET /auth/status` | key | `state` (`ready`, `signed_out`, `reauth_required`, `consent_required`), email, scopes, expiry, `login_url` |
| `POST /auth/login?consent=` | key | `{"url": ...}`, a launch link |
| `POST /auth/logout?forget=` | key | `{"state", "revoked"}` |
| `GET /auth/start/{launch}`, `GET /auth/callback` | link / OAuth state | Browser only |
| `GET /healthz` | none | Liveness |
| anything else | key | OpenAI-shaped `404 unknown_url` |

The proxy forwards the request headers `OpenAI-Beta`, `Idempotency-Key`, and
`X-Client-Request-Id`, plus the query string. It relays `retry-after`, rate-limit
headers, and the request id back. The ChatGPT backend calls the request id
`x-oai-request-id`, so the proxy also sends it as `x-request-id`, which SDKs
expose as `_request_id`. Organization and project headers are not forwarded.
`date`, `server`, and other edge headers from OpenAI's CDN are not relayed.
Responses carry `Cache-Control: no-store`.

### `/v1/responses`

SIWC inference must be `store=false` and `stream=true`. The proxy handles that,
so ordinary calls work:

- `store` is forced to `false`. A non-false value is dropped in lenient mode
  and rejected with `--strict`.
- `stream` is always `true` upstream. If the caller didn't ask to stream, the
  events are aggregated into one `Response` JSON object.
- A string `input` is wrapped as a user message.
- Typed `{"type": "message", "role": "system"}` items, which ChatGPT sign-in
  rejects, are sent as `developer` messages. Untyped `{"role": "system"}` items
  are accepted upstream as they are.
- `temperature`, `top_p`, `metadata`, `user`, `safety_identifier`, `truncation`,
  `prompt_cache_retention`, and `max_tool_calls` are rejected by ChatGPT sign-in.
  In lenient mode they are dropped and listed in `x-siwc-ignored-params`; with
  `--strict` they get a 400.
- `previous_response_id`, `conversation`, and `background` are rejected with a
  clear 400. They need stored responses; send the full conversation instead.
- Everything else, including unknown fields and events, passes through unchanged,
  with one exception. The ChatGPT backend sends each output item in
  `response.output_item.done` but leaves `output` empty in the terminal
  `response.completed`/`response.incomplete` event. The proxy fills that
  `output` from the items, so `client.responses.stream(...).get_final_response()`
  and aggregated responses contain the answer.

### `/v1/chat/completions`

Requests are translated to Responses. The answer is translated back as Chat
JSON, or as Chat SSE chunks ending in `data: [DONE]`.

| Chat input | Behavior |
| --- | --- |
| `system`, `developer`, `user`, `assistant`, `tool` messages | Order kept; tool results matched to call IDs |
| `text` / `image_url` parts | Responses parts; HTTPS URLs or inline base64 images |
| `tools`, `tool_choice`, `parallel_tool_calls` | Function tools; tool calls returned for your code to run |
| `response_format` | Text, JSON object, or JSON schema via `text.format` |
| `max_tokens` / `max_completion_tokens` | `max_output_tokens` |
| `reasoning_effort`, `verbosity` | Responses `reasoning` / `text` settings |
| `web_search_options` | Hosted `web_search` tool. Citations come back as `message.annotations` (`url_citation`); when streaming, they arrive in one `delta.annotations` chunk before the finish chunk |
| `service_tier` | Forwarded (OpenAI serves `priority` as `default` and rejects `flex`) |
| `temperature`, `top_p`, `metadata` | Dropped in lenient mode (ChatGPT sign-in rejects them) |
| `stream_options.include_usage` | Final usage chunk |
| `prompt_cache_key` | Names the cache session (see [Prompt caching](#prompt-caching)) |

**Lenient mode (default).** Options with no Responses equivalent that don't
change the shape of the answer are dropped instead of rejected. They are
reported in the `x-siwc-ignored-params` response header, for example
`x-siwc-ignored-params: user,seed,stop`. The dropped options are:

- `temperature`, `top_p`, `metadata`;
- `user`, `safety_identifier`, `seed`, `stop`;
- non-zero penalties, `logit_bias`, `store`;
- message `name`;
- `stream_options` without `stream`.

**`--strict`** rejects them with a 400 whose `param` and message name the option.

These are always rejected: `n > 1`, `logprobs`, audio, and legacy
`functions`/`function_call`. For `functions`, use `tools`.

### Prompt caching

Neither API keeps conversation state here: every Chat or Responses call is a new
upstream request that carries the whole history. To keep long chats fast, turns
need to reuse the prompt cache. ChatGPT sign-in keys that cache on a `session_id`
request header and replaces any `prompt_cache_key` with a random value per
request, so without a session no turn ever hits the cache.

The proxy sends a `session_id` on every inference request:

- The caller's own `session_id` header, if there is one, is forwarded as is.
- Otherwise, a `prompt_cache_key` from the body (Chat or Responses) names the
  session; it is removed from the body, since upstream ignores it there.
- Otherwise, the session is derived from the model, `instructions` and the
  first input item (usually the system prompt). These stay the same as a
  conversation grows, so every turn lands in one session, and editing a later
  message keeps the cached prefix up to the edit. Conversations that start the
  same way share a session, which also shares the cache for that opening.

Session values are HMACs under a key made at startup, so conversation text is
never sent as an ID; a proxy restart starts new sessions. Usage reports hits in
`prompt_tokens_details.cached_tokens` (Chat) or
`input_tokens_details.cached_tokens` (Responses). Caching is best effort: in
live tests, cached tokens grew with the history to about 80% of the prompt,
but an occasional turn missed.

## Errors

All errors use OpenAI's `{"error": {"message", "type", "param", "code"}}` shape,
so SDK exception classes work:

| Situation | Status | `code` |
| --- | --- | --- |
| Missing or wrong local key | 401 | `invalid_api_key` |
| Not signed in / refresh rejected | 401 | `chatgpt_login_required` (message includes the sign-in link) |
| Plan usage not allowed | 403 | `chatgpt_consent_required` |
| Plan usage limit reached | 429 | `subscription_sharing_usage_limit_exceeded`. Sent with `x-should-retry: false`, so OpenAI SDKs don't retry automatically. Usage is managed at <https://chatgpt.com/settings/usage>. |
| Usage/user temporarily unavailable | 503 | `subscription_sharing_usage_unavailable`, `subscription_sharing_user_unavailable` |
| Not eligible / route not supported | 403 | `subscription_sharing_user_not_eligible`, `subscription_sharing_route_not_supported` |
| Unsupported capability | 400 | `subscription_sharing_unsupported_capability` |
| Disallowed `Host` header | 400 | `invalid_host` |
| OpenAI unreachable / timed out | 502 / 504 | `upstream_connection_error` / `upstream_timeout` |
| Refresh failed temporarily | 502 | `credential_refresh_failed` |
| Client registration rejected at refresh | 502 | `chatgpt_client_rejected` (run `logout --forget`, then sign in, if it persists) |

These codes get the same status whether they arrive as an HTTP error or as a
`response.failed` event. Admission bodies like `{"detail": ...}` are rewrapped
in the OpenAI shape.

**Retries.** The proxy retries only before any inference starts:

- connection failures: up to 2 times;
- 503s with the two `*_unavailable` codes: honoring `Retry-After`, capped at 10 s;
- the single 401 refresh.

Once streaming has started, nothing is retried. A stream that fails midway ends
with an error event and no `[DONE]`. Disconnecting closes the upstream request,
but work already done may still count against your plan.

## Security model

- The proxy is designed for one trusted local user. It binds to loopback by
  default and requires the bearer key on every API and status route, using a
  constant-time comparison.
- `credentials.json` and `proxy_key` are written atomically with mode 0600 in a
  0700 directory. This is plaintext protected by file permissions, not an
  encrypted vault. Don't commit or copy the data directory. Copying it also
  duplicates the host identity.
- Tokens and request bodies are never logged, and the access log is disabled.
  Log lines carry method, path, status, model, latency, and upstream request id.
  `status` never prints tokens; `token` prints only to stdout.
- The callback only accepts `127.0.0.1:<port>`. The OAuth state is bound to the
  browser through a signed session cookie and can be used once.

## Live smoke test

Automated tests mock OpenAI. To verify with a real account:

1. Start `uv run siwc-proxy serve` while signed out. Open the printed link, then
   confirm that `uv run siwc-proxy status` reports `ready`.
2. With the `openai` SDK at the base URL, run `models.list()`, chat (plain and
   `stream=True`), and `responses.create(input="hi")`. Then do a tool-call
   round trip.
3. Edit `expires_at` in `credentials.json` to a past time. Then send two
   requests at once and run `uv run siwc-proxy token` in another terminal.
   Expect exactly one refresh in the log and intact credentials.
4. Run `uv run siwc-proxy logout`. The next request should return 401 with a
   link. Signing in again should reuse the same `client_id`, with no
   registration screen.
