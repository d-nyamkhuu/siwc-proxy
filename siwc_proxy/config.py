"""Settings from SIWC_PROXY_* environment variables and CLI flags."""

import ipaddress
import secrets
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from platformdirs import user_data_dir
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .store import create_private


def default_data_dir():
    return Path(user_data_dir("siwc-proxy", appauthor=False))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIWC_PROXY_", validate_assignment=True)

    host: str = "127.0.0.1"
    port: int = Field(8787, ge=0, le=65535)
    data_dir: Path = Field(default_factory=default_data_dir)
    api_key: SecretStr | None = None
    # Extra Host header values accepted besides loopback; "name" matches any port, "*" disables the check.
    allowed_hosts: Annotated[list[str], NoDecode] = []
    strict: bool = False
    log_level: str = "info"
    read_timeout: float = Field(300, gt=0)

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def split_hosts(cls, value):
        if isinstance(value, str):
            return [host.strip() for host in value.split(",") if host.strip()]
        return value

    @field_validator("data_dir")
    @classmethod
    def resolve(cls, value):
        return Path(value).expanduser().resolve()

    @property
    def origin(self):
        """Browser-facing origin; the registered redirect URI only allows 127.0.0.1 with a variable port."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def redirect_uri(self):
        return self.origin + "/auth/callback"

    def host_allowed(self, header):
        """DNS-rebinding guard: loopback names on our port, plus configured hosts."""
        header = (header or "").lower()
        try:
            parts = urlsplit("//" + header)
            name, port = parts.hostname, parts.port
        except ValueError:
            return False
        allowed = {host.lower() for host in self.allowed_hosts}
        if "*" in allowed or header in allowed or name in allowed:
            return True
        return port == self.port and name in ("127.0.0.1", "localhost", "::1")

    @property
    def loopback(self):
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return self.host == "localhost"

    def resolve_api_key(self):
        """The configured key, or one generated once and kept in <data_dir>/proxy_key."""
        if self.api_key is not None:
            key = self.api_key.get_secret_value()
        else:
            path = self.data_dir / "proxy_key"
            if not path.exists():
                # Concurrent first runs (serve and login) must agree on one key: the first creator wins.
                create_private(path, "sk-siwc-" + secrets.token_urlsafe(32) + "\n")
            key = path.read_text(encoding="utf-8").strip()
        if not key or any(c.isspace() for c in key):
            raise ValueError("The local API key must be nonempty and contain no whitespace.")
        return key
