"""Incremental Server-Sent Events decoding for Responses streams."""

import codecs
import json
import re

from .errors import protocol_error


async def response_events(response):
    """Incrementally decode SSE (UTF-8, CR/LF, comments, and multiline data)."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer, data = "", []
    first = True

    def line(value):
        if value == "":
            if not data:
                return None
            raw = "\n".join(data)
            data.clear()
            try:
                event = json.loads(raw)
            except (ValueError, RecursionError):
                raise protocol_error() from None
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise protocol_error()
            return event
        field, sep, value = value.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
        return None

    try:
        async for chunk in response.aiter_bytes():
            text = decoder.decode(chunk)
            if first and text:
                text = text.removeprefix("\ufeff")
                first = False
            buffer += text
            while match := re.search(r"\r\n|\r|\n", buffer):
                if match.group() == "\r" and match.end() == len(buffer):
                    break  # A CRLF pair may straddle network packets.
                event = line(buffer[:match.start()])
                buffer = buffer[match.end():]
                if event is not None:
                    yield event
        buffer += decoder.decode(b"", final=True)
        if buffer.endswith("\r"):
            event = line(buffer[:-1])
            if event is not None:
                yield event
            buffer = ""
        if buffer or data:
            raise protocol_error()
    except UnicodeError:
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


def remember_item(items, event):
    if isinstance(event.get("output_index"), int) and isinstance(event.get("item"), dict):
        items[event["output_index"]] = event["item"]


async def with_terminal_output(chunks):
    """Relay Responses SSE bytes unchanged except for terminal events that need `fill_output`."""
    buffer, items = b"", {}
    async for chunk in chunks:
        buffer += chunk
        while match := re.search(rb"\r\n\r\n|\n\n|\r\r", buffer):
            block, buffer = buffer[:match.end()], buffer[match.end():]
            yield patch_block(block, items)
    if buffer:
        yield buffer


def patch_block(block, items):
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
    elif event.get("type") in TERMINAL_EVENTS and isinstance(event.get("response"), dict) \
            and not event["response"].get("output") and items:
        fill_output(event["response"], items)
        names = "".join(line + "\n" for line in lines if line.startswith("event:"))
        return (names + "data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode()
    return block
