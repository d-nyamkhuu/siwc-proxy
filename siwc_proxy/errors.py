"""OpenAI-shaped errors and normalization of upstream error bodies."""

import json

from fastapi.responses import JSONResponse

from .constants import CODE_STATUS, USAGE_LIMIT_CODE, USAGE_URL

ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 429: "rate_limit_error"}


class OpenAIError(Exception):
    def __init__(self, message, param=None, *, status=400, code=None, error_type=None):
        super().__init__(message)
        self.status = status
        self.payload = {"error": {
            "message": message,
            "type": error_type or ERROR_TYPES.get(status, "api_error"),
            "param": param,
            "code": code,
        }}

    @property
    def code(self):
        return self.payload["error"]["code"]

    def response(self, headers=None):
        headers = {**(headers or {}), **no_retry(self.code)}
        if self.status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(self.payload, status_code=self.status, headers=headers)


def no_retry(code):
    """A plan usage limit is not lifted by retrying; `x-should-retry` stops OpenAI SDKs' automatic retries."""
    return {"x-should-retry": "false"} if code == USAGE_LIMIT_CODE else {}


def protocol_error(message="OpenAI returned an invalid or interrupted Responses stream."):
    return OpenAIError(message, status=502, code="invalid_upstream_response")


def status_for_code(code, default=502):
    if isinstance(code, str) and code.startswith("chatpass_v2_"):
        return 403
    return CODE_STATUS.get(code, default)


def failure(error, default_message="OpenAI did not complete the response."):
    """Error for a `response.failed` or `error` stream event, keeping its code's meaning."""
    error = error if isinstance(error, dict) else {}
    code = error.get("code") or "upstream_response_failed"
    message = error.get("message") or default_message
    if code == USAGE_LIMIT_CODE and USAGE_URL not in message:
        message += f" Manage usage: {USAGE_URL}"
    return OpenAIError(message, status=status_for_code(code), code=code)


def error_code(content):
    """Best-effort `code` from an upstream error body ({"error": {...}} or {"detail": ...})."""
    try:
        body = json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        return None
    if not isinstance(body, dict):
        return None
    detail = body.get("error") if isinstance(body.get("error"), dict) else body.get("detail")
    return detail.get("code") if isinstance(detail, dict) else None


def normalize_error_body(status, content):
    """Return bytes to relay: OpenAI-shaped bodies unchanged, anything else rewrapped."""
    try:
        body = json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        body = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return content
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        message, code = detail.get("message") or detail.get("detail"), detail.get("code")
    elif isinstance(detail, str):
        message, code = detail, None
    else:
        message, code = None, None
    payload = {"error": {"message": message if isinstance(message, str) else f"OpenAI returned HTTP {status}.",
                         "type": ERROR_TYPES.get(status, "api_error"), "param": None, "code": code}}
    return json.dumps(payload).encode()
