"""Credential file: one registration, atomic 0600 writes, cross-process locking."""

import json
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .constants import DIRECT_SCOPE, ISSUER

TOKEN_FIELDS = ("access_token", "refresh_token", "token_type", "expires_at", "earliest_refresh_at")


class CredentialFileError(Exception):
    pass


class Account(BaseModel):
    model_config = ConfigDict(extra="allow")

    client_id: str
    subject: str | None = None
    issuer: str = ISSUER
    email: str | None = None
    name: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    # Kept after sign-out as id_token_hint for the next authorization.
    id_token: str | None = None
    token_type: str | None = None
    scope: str = ""
    expires_at: int | None = None
    earliest_refresh_at: int | None = None

    @model_validator(mode="before")
    @classmethod
    def legacy(cls, data):
        # signin_cli.py stored a "scopes" list and float timestamps.
        if isinstance(data, dict):
            data = dict(data)
            if "scope" not in data and isinstance(data.get("scopes"), list):
                data["scope"] = " ".join(data.pop("scopes"))
            for key in ("expires_at", "earliest_refresh_at"):
                if isinstance(data.get(key), float):
                    data[key] = int(data[key])
            data.pop("expires_in", None)
        return data

    @property
    def scopes(self):
        return set(self.scope.split())

    @property
    def state(self):
        if not self.access_token or not self.refresh_token:
            return "reauth_required" if self.subject else "signed_out"
        if DIRECT_SCOPE not in self.scopes:
            return "consent_required"
        return "ready"

    def token(self):
        """Authlib OAuth2Token fields."""
        return {k: v for k, v in {"access_token": self.access_token, "refresh_token": self.refresh_token,
                                  "token_type": self.token_type or "Bearer", "expires_at": self.expires_at,
                                  "scope": self.scope}.items() if v is not None}

    def apply_tokens(self, token, *, default_scope):
        """Copy a token endpoint response; the scope is unchanged when omitted (RFC 6749 §5.1)."""
        for key in ("access_token", "refresh_token", "token_type", "id_token"):
            if token.get(key):
                setattr(self, key, token[key])
        self.scope = token.get("scope") or default_scope
        self.expires_at = int(token["expires_at"]) if token.get("expires_at") is not None else None
        earliest = token.get("earliest_refresh_at")
        self.earliest_refresh_at = int(earliest) if isinstance(earliest, (int, float)) else None

    def clear_tokens(self):
        for key in TOKEN_FIELDS:
            setattr(self, key, None)
        self.scope = ""


class Record(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: int = 2
    ext_agent_host_id: str = Field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    account: Account | None = None

    @model_validator(mode="before")
    @classmethod
    def empty_account(cls, data):
        if isinstance(data, dict) and data.get("account") == {}:
            data = {**data, "account": None}
        return data

    @property
    def state(self):
        return self.account.state if self.account else "signed_out"


def write_private(path, text):
    """Atomically replace path with owner-only permissions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            if os.name == "posix":
                os.fchmod(file.fileno(), 0o600)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


class Store:
    """Reads are lock-free; changes go through `transaction()`, which serializes across processes."""

    def __init__(self, directory, *, lock_timeout=60):
        self.directory = Path(directory)
        self.path = self.directory / "credentials.json"
        self._file_lock = FileLock(self.directory / "credentials.lock", timeout=lock_timeout, mode=0o600,
                                   thread_local=False)
        self._lock = anyio.Lock()
        self._cached = None

    def read(self):
        """Current record, or None before the first sign-in; reparsed only when the file changes."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._cached = None
            return None
        key = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if self._cached and self._cached[0] == key:
            return self._cached[1].model_copy(deep=True)
        try:
            record = Record.model_validate_json(self.path.read_bytes())
        except (OSError, ValueError, ValidationError):
            raise CredentialFileError(f"Cannot read {self.path}; fix or delete it and sign in again.") from None
        self._cached = (key, record)
        return record.model_copy(deep=True)

    def write(self, record):
        write_private(self.path, json.dumps(record.model_dump(mode="json", exclude_none=True), indent=2) + "\n")
        self._cached = None

    def delete(self):
        self.path.unlink(missing_ok=True)
        self._cached = None

    @asynccontextmanager
    async def transaction(self):
        """Exclusive section for read-modify-write, shared with other processes using this directory."""
        async with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                await anyio.to_thread.run_sync(self._file_lock.acquire)
            except Timeout:
                raise CredentialFileError("Timed out waiting for another process to update credentials.") from None
            try:
                yield
            finally:
                self._file_lock.release()


def now():
    return int(time.time())
