# siwc-proxy

A local **OpenAI-compatible API** backed by
[Sign in with ChatGPT token sharing](https://developers.openai.com/siwc/token-sharing-open-source/sign-in).
Sign in once; the proxy keeps the ChatGPT plan credential fresh and serves
`/v1/chat/completions`, `/v1/responses`, and `/v1/models` to any OpenAI SDK,
framework, or tool that lets you set a base URL.

- **Sign-in and re-auth handled for you.** The proxy hosts its own OAuth callback,
  registers its client on first use, and reuses that registration afterwards.
  [Authlib](https://docs.authlib.org/) does PKCE, state, nonce, and ID-token checks.
- **Safe token refresh.** Rotating refresh tokens are refreshed once, even across
  processes (file lock). Refreshed tokens are saved before anything else can fail.
  A temporary outage never deletes credentials. An upstream 401 triggers one refresh and retry.
- **Works with ordinary client code.** `responses.create(input="hi")` without
  `stream`/`store` works: the proxy streams upstream and aggregates the result.
  Chat Completions is translated to Responses. Options that can't be honored,
  such as `user` and `seed`, are dropped and listed in a response header.
- **Errors in OpenAI's shape and status.** For example, a plan usage limit becomes
  a 429 `RateLimitError`, so normal client backoff works.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). uv installs a suitable Python (3.12+) if needed.

```bash
uv sync
```

```bash
uv run siwc-proxy serve
```

If you are not signed in yet, `serve` prints a sign-in link. Open it in a browser
**on the same computer**, or run this in a second terminal:

```bash
uv run siwc-proxy login
```

Then point any OpenAI client at the proxy:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="<local key>")
model = client.models.list().data[0].id
print(client.responses.create(model=model, input="Say hello.").output_text)
```

The local key is generated on first run and shown by:

```bash
uv run siwc-proxy status --show-key
```

Or choose your own key with `SIWC_PROXY_API_KEY`. The key only protects the
local proxy; it is never sent to OpenAI.

To install the command globally instead of using `uv run`:

```bash
uv tool install .
```

## Commands

| Command | What it does |
| --- | --- |
| `siwc-proxy serve` | Run the proxy (default `127.0.0.1:8787`). Starts even when signed out. |
| `siwc-proxy login [--consent]` | Sign in or re-authorize. Uses the running proxy if there is one; otherwise hosts the callback on a temporary port. |
| `siwc-proxy logout [--forget]` | Revoke and clear tokens. `--forget` also drops the registration (needed to switch accounts). |
| `siwc-proxy status [--json] [--show-key]` | Account, state, scopes, expiry. Never prints tokens. |
| `siwc-proxy token` | Print a fresh access token to stdout, for scripts. |

Every command accepts `--data-dir` (or `SIWC_PROXY_DATA_DIR`) to choose a credential directory.
Use one directory per account.

See **[docs/proxy.md](docs/proxy.md)** for configuration, routes, the Chat
translation rules, error mapping, Docker/LAN access, and the security model.

## Development

```bash
uv run pytest -q
```

The tests run the real `openai` SDK against the app. OpenAI's OAuth and API are
mocked: real RSA-signed ID tokens, rotating refresh tokens, and multi-process refresh races.
No live account is used.

## References

- [Registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Profiles and sessions](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
- [Token reference](https://developers.openai.com/siwc/token-sharing-open-source/token-reference)
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
