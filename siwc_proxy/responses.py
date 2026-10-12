"""Adapt ordinary Responses API calls to ChatGPT sign-in's stateless, streaming-only transport."""

import hashlib
import hmac
import json

from .errors import OpenAIError, failure, protocol_error
from .sse import fill_output, remember_item

STORAGE_PARAMS = {
    "previous_response_id": "previous_response_id needs stored responses, which ChatGPT sign-in does not "
                            "support; send the whole conversation in input instead.",
    "conversation": "Conversations need stored responses, which ChatGPT sign-in does not support.",
    "background": "Background responses are not supported with ChatGPT sign-in.",
}


# Rejected by ChatGPT sign-in and safe to drop: none changes the shape of the answer.
DROPPABLE = ("temperature", "top_p", "metadata", "user", "safety_identifier", "truncation",
             "prompt_cache_retention", "max_tool_calls")


def system_to_developer(item):
    """ChatGPT sign-in rejects typed system message items; developer messages carry the same guidance."""
    if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "system":
        return {**item, "role": "developer"}
    return item


def cache_session(body, secret):
    """Remove `prompt_cache_key` from an upstream body and return the `session_id` header to send instead.

    ChatGPT sign-in keys its prompt cache on the `session_id` header and replaces `prompt_cache_key` with a
    random value, so without a session every turn of a conversation misses the cache. A caller's own key is
    used when given; otherwise the session comes from the model, instructions and first input item, which stay
    the same as a conversation grows. Values are keyed hashes so conversation text never leaves as an ID.
    """
    key = body.pop("prompt_cache_key", None)
    if not isinstance(key, str) or not key:
        first = body["input"][0] if isinstance(body.get("input"), list) and body["input"] else None
        key = json.dumps([body.get("model"), body.get("instructions"), first], sort_keys=True)
    return hmac.new(secret, key.encode(), hashlib.sha256).hexdigest()[:32]


def normalize(body, *, strict=False):
    """Return (upstream body, whether the caller asked to stream, ignored parameter names)."""
    if not isinstance(body, dict):
        raise OpenAIError("Request body must be a JSON object.")
    body, ignored = dict(body), []
    stream = body.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise OpenAIError("Invalid parameter type.", "stream")
    if body.get("store") not in (None, False):
        if strict:
            raise OpenAIError("ChatGPT sign-in requires store=false.", "store")
        ignored.append("store")
    for key in DROPPABLE:
        if body.get(key) is not None:
            if strict:
                raise OpenAIError(f"Unsupported parameter: '{key}' (rejected by ChatGPT sign-in).", key)
            ignored.append(key)
            del body[key]
    for key, message in STORAGE_PARAMS.items():
        if body.get(key) not in (None, False):
            raise OpenAIError(message, key)
    if isinstance(body.get("input"), str):
        body["input"] = [{"role": "user", "content": body["input"]}]
    elif isinstance(body.get("input"), list):
        body["input"] = [system_to_developer(item) for item in body["input"]]
    body.update(store=False, stream=True)
    return body, stream is True, ignored


async def aggregate(events):
    """Final Response object from a Responses event stream, or an error with the failure's status."""
    items = {}
    async for event in events:
        kind = event["type"]
        if kind == "response.output_item.done":
            remember_item(items, event)
        elif kind in ("response.completed", "response.incomplete"):
            response = event.get("response")
            if not isinstance(response, dict):
                raise protocol_error()
            return fill_output(response, items)
        elif kind == "response.failed":
            response = event.get("response")
            raise failure(response.get("error") if isinstance(response, dict) else None)
        elif kind == "error":
            raise failure(event.get("error") if isinstance(event.get("error"), dict) else event,
                          "OpenAI returned a stream error.")
    raise protocol_error()
