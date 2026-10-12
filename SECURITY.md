# Security policy

siwc-proxy holds OAuth tokens for your ChatGPT account and a local API key, so security reports are welcome.

## Reporting a vulnerability

Report privately through
[GitHub's private vulnerability reporting](https://github.com/d-nyamkhuu/siwc-proxy/security/advisories/new).
Please don't open a public issue for a vulnerability.

Include the version (`siwc-proxy --version`), your OS, and steps to reproduce. Never include real
access tokens, refresh tokens, `credentials.json`, or your `proxy_key`: revoke them first
(`siwc-proxy logout`) if they may have leaked.

You should get a reply within a week. Fixes go into the latest release only.

## Scope

In scope: anything that lets another program, website, or user on the network read your tokens or
local key, or use your ChatGPT plan through the proxy without the key. Examples are a bypass of the
`Host` check or the key check, a token written to logs or to a file other users can read, or a flaw in
the OAuth sign-in flow.

Out of scope: running `serve --host` on a network you don't trust, or `--allowed-host '*'`, since the
docs warn against both. Also out of scope are bugs in OpenAI's own services; report those to
[OpenAI](https://openai.com/security/disclosure/).
