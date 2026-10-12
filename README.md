# siwc-proxy

A local **OpenAI-compatible API** backed by
[Sign in with ChatGPT token sharing](https://developers.openai.com/siwc/token-sharing-open-source).
Sign in once; the proxy keeps the ChatGPT plan credential fresh and serves
`/v1/chat/completions`, `/v1/responses`, and `/v1/models` to any OpenAI SDK,
framework, or tool that lets you set a base URL.

- **Sign-in and re-auth handled for you.** The proxy hosts its own OAuth callback,
  registers its client on first use, and reuses that registration afterwards.
  [Authlib](https://docs.authlib.org/) does PKCE, state, nonce, and ID-token checks.
- **Safe token refresh.** Rotating refresh tokens are refreshed once, even across
  processes (file lock). Refreshed tokens are saved before they are checked, so
  a failed check never loses the new refresh token.
  A temporary outage never deletes credentials. An upstream 401 triggers one refresh and retry.
- **Works with ordinary client code.** `responses.create(input="hi")` without
  `stream`/`store` works: the proxy streams upstream and aggregates the result.
  Chat Completions is translated to Responses. Options that can't be honored,
  such as `user` and `seed`, are dropped and listed in a response header.
- **Errors in OpenAI's shape and status.** For example, a plan usage limit becomes
  a 429 `RateLimitError`, so normal client backoff works.

> [!IMPORTANT]
> **Personal, local use only.** This is an independent open-source project, not
> affiliated with or endorsed by OpenAI. It lets *you* use *your own* ChatGPT plan
> from tools on your own computer. Don't host it for others, expose it publicly,
> or share access to your plan. OpenAI's token-sharing docs cover open-source and
> locally hosted apps; paid or remotely hosted products need OpenAI's approval
> through its interest form (see the [token-sharing docs](https://developers.openai.com/siwc/token-sharing-open-source)).
> Current limits are listed under [preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).
> Sign in with ChatGPT token sharing is a preview, and the ChatGPT backend this
> proxy talks to can change or break without notice. Usage counts against your plan.

## Quick start

Install the `siwc-proxy` command with [uv](https://docs.astral.sh/uv/).
uv gets a suitable Python (3.12+) for it if you don't have one:

```bash
uv tool install siwc-proxy
```

Start the proxy:

```bash
siwc-proxy serve
```

If you are not signed in yet, `serve` prints a sign-in link. Open it in a browser
**on the same computer**, or run this in a second terminal:

```bash
siwc-proxy login
```

On a remote server or any machine without a browser, run `siwc-proxy login --remote`
there instead. Open the printed link on your own computer and sign in. Then copy the
`http://127.0.0.1:...` address the browser ends up on (the page fails to load;
that is expected) and paste it back into the terminal.

Then point any OpenAI client at the proxy:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="<local key>")
model = client.models.list().data[0].id
print(client.responses.create(model=model, input="Say hello.").output_text)
```

The local key is generated on first run and shown by:

```bash
siwc-proxy status --show-key
```

Or choose your own key with `SIWC_PROXY_API_KEY`. The key only protects the
local proxy; it is never sent to OpenAI.

### More ways to run it

- **Without installing:** `uvx siwc-proxy serve`. Put `uvx` in front of every
  `siwc-proxy` command in this README.
- **Upgrade:** `uv tool upgrade siwc-proxy`.
- **Not on your `PATH` after `uv tool install`?** Run `uv tool update-shell` and
  open a new terminal.
- **pip or pipx:** `pipx install siwc-proxy`, or `pip install siwc-proxy` in a virtual
  environment. `python -m siwc_proxy` works like the `siwc-proxy` command.
- **One-file scripts with uv:** declare `openai` inline and run the file with `uv run hello.py`:

  ```python
  # /// script
  # dependencies = ["openai"]
  # ///
  import os
  from openai import OpenAI

  client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key=os.environ["SIWC_PROXY_API_KEY"])
  model = client.models.list().data[0].id
  print(client.responses.create(model=model, input="Say hello.").output_text)
  ```

## Commands

| Command | What it does |
| --- | --- |
| `siwc-proxy serve` | Run the proxy (default `127.0.0.1:8787`). Starts even when signed out. |
| `siwc-proxy login [--consent] [--remote]` | Sign in or re-authorize. Uses the running proxy if there is one; otherwise hosts the callback on a temporary port. `--remote` signs in with a browser on another computer by pasting back the address it lands on. `--consent` shows ChatGPT's permission screen again, in case you didn't allow plan usage before. |
| `siwc-proxy logout [--forget]` | Revoke and clear tokens. `--forget` also drops the registration (needed to switch accounts). |
| `siwc-proxy status [--json] [--show-key]` | Account, state, scopes, expiry. Never prints tokens. |
| `siwc-proxy token` | Print a fresh access token to stdout, for scripts. |

Every command accepts `--data-dir`, before or after the command name (or `SIWC_PROXY_DATA_DIR`),
to choose a credential directory.
Use one directory per account.

`serve --host` can bind a non-loopback interface for Docker or LAN use, but the
proxy speaks plain HTTP: the key and your prompts then cross the network
unencrypted. See the warning in [docs/proxy.md](https://github.com/d-nyamkhuu/siwc-proxy/blob/main/docs/proxy.md#docker-localhost-and-lan-access).

See **[docs/proxy.md](https://github.com/d-nyamkhuu/siwc-proxy/blob/main/docs/proxy.md)** for configuration, routes, the Chat
translation rules, error mapping, and Docker/LAN access.

## Development

```bash
git clone https://github.com/d-nyamkhuu/siwc-proxy.git
cd siwc-proxy
uv sync
uv run pytest -q
uv run siwc-proxy serve   # run the checkout
```

The tests run the real `openai` SDK against the app. OpenAI's OAuth and API are
mocked: real RSA-signed ID tokens, rotating refresh tokens, and multi-process refresh races.
No live account is used.

### Releasing

The version lives only in `pyproject.toml`. Bump it, then push a matching tag:

```bash
uv version --bump minor   # or patch / major
git commit -am "Release $(uv version --short)"
git tag "v$(uv version --short)"
git push origin main --tags
```

The [release workflow](https://github.com/d-nyamkhuu/siwc-proxy/blob/main/.github/workflows/release.yml) runs the tests, builds the
wheel and sdist, smoke-tests both, and publishes to PyPI with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored.
Publishing waits for approval of the `pypi` deployment on the workflow run. After it publishes,
the workflow creates the GitHub release with generated notes and the built files attached.
Before the first release, add a pending publisher on PyPI for project `siwc-proxy`,
repository `d-nyamkhuu/siwc-proxy`, workflow `release.yml`, environment `pypi`.

## References

- [Token sharing for open-source and locally hosted apps](https://developers.openai.com/siwc/token-sharing-open-source)
- [Registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Profiles and sessions](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
- [Token reference](https://developers.openai.com/siwc/token-sharing-open-source/token-reference)
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
