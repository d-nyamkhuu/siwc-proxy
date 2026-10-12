"""Server-Sent Events decoding and relaying for Responses streams."""

import json
import re

from httpx2 import EventSource, SSEError

from .errors import protocol_error


async def response_events(response):
    """Responses events from an upstream SSE body; events without data (keep-alives) are skipped."""
    try:
        # No event size cap: terminal events repeat the whole answer, which may exceed httpx2's 1 MiB default.
        async for sse in EventSource(response, max_event_size=None):
            if not sse.data:
                continue
            try:
                event = sse.json()
            except (ValueError, RecursionError):
                raise protocol_error() from None
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise protocol_error()
            yield event
    except SSEError:
        raise protocol_error() from None


def encode_sse(value):
    return ("data: " + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode("utf-8")


TERMINAL_EVENTS = ("response.completed", "response.incomplete")


def fill_output(response, items):
    """The ChatGPT backend streams items in `output_item.done` events and sends terminal responses with
    an empty `output`; restore it so the terminal event describes the whole answer."""
    if isinstance(response, dict) and not response.get("output") and items:
        response["output"] = [items[i] for i in sorted(items)]
    return response


def token_usage(response):
    """(input, cached input, output) tokens a terminal response reports, or None."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    details = usage.get("input_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    counts = usage.get("input_tokens"), cached, usage.get("output_tokens")
    return counts if all(type(n) is int and n >= 0 for n in counts) else None


def remember_item(items, event):
    if isinstance(event.get("output_index"), int) and isinstance(event.get("item"), dict):
        items[event["output_index"]] = event["item"]


async def with_terminal_output(chunks, on_terminal=None):
    """Relay Responses SSE bytes unchanged except for terminal events that need `fill_output`.

    `on_terminal` is called with the terminal response.
    """
    buffer, items = b"", {}
    async for chunk in chunks:
        buffer += chunk
        while match := re.search(rb"\r\n\r\n|\n\n|\r\r", buffer):
            block, buffer = buffer[:match.end()], buffer[match.end():]
            yield patch_block(block, items, on_terminal)
    if buffer:
        yield buffer


def patch_block(block, items, on_terminal=None):
    if not any(name.encode() in block for name in ("response.output_item.done", *TERMINAL_EVENTS)):
        return block
    try:
        lines = re.split(r"\r\n|\r|\n", block.decode("utf-8"))
        event = json.loads("\n".join(line[5:].removeprefix(" ") for line in lines if line.startswith("data:")))
    except (UnicodeError, ValueError, RecursionError):
        return block
    if not isinstance(event, dict):
        return block
    if event.get("type") == "response.output_item.done":
        remember_item(items, event)
    elif event.get("type") in TERMINAL_EVENTS and isinstance(event.get("response"), dict):
        if on_terminal:
            on_terminal(event["response"])
        if event["response"].get("output") or not items:
            return block
        fill_output(event["response"], items)
        names = "".join(line + "\n" for line in lines if line.startswith("event:"))
        return (names + "data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode()
    return block
