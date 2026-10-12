"""Stateless Chat Completions to Responses translation; no credentials or HTTP I/O."""

import re
import time
import uuid

from .errors import OpenAIError, failure, protocol_error
from .sse import fill_output, remember_item

ANY = object()  # A neutral value nothing equals: the option is dropped whenever it is set.

# Chat options dropped (and reported) in lenient mode when set to anything but their neutral value.
# ChatGPT sign-in rejects sampling controls and metadata, so those go too.
DROPPABLE = {"user": None, "safety_identifier": None, "seed": None, "stop": None,
             "presence_penalty": 0, "frequency_penalty": 0, "logit_bias": None, "store": False,
             "temperature": ANY, "top_p": ANY, "metadata": ANY}


def require(value, kind, param):
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise OpenAIError("Invalid parameter type.", param)
    return value


def string(value, param, *, empty=True):
    require(value, str, param)
    if not empty and not value:
        raise OpenAIError("This parameter must not be empty.", param)
    return value


def only_keys(value, allowed, param=""):
    require(value, dict, param or None)
    for key in value:
        if key not in allowed and value[key] is not None:
            name = f"{param}.{key}" if param else key
            raise OpenAIError(f"Unsupported parameter: '{name}'.", name)


def message_content(content, role, param):
    if isinstance(content, str):
        return content
    require(content, list, param)
    parts = []
    for i, part in enumerate(content):
        path = f"{param}[{i}]"
        require(part, dict, path)
        if part.get("type") == "text":
            only_keys(part, {"type", "text"}, path)
            text = string(part.get("text"), path + ".text")
            parts.append({"type": "output_text" if role == "assistant" else "input_text", "text": text})
            if role == "assistant":
                parts[-1]["annotations"] = []
        elif part.get("type") == "refusal" and role == "assistant":
            only_keys(part, {"type", "refusal"}, path)
            parts.append({"type": "refusal", "refusal": string(part.get("refusal"), path + ".refusal")})
        elif part.get("type") == "image_url" and role == "user":
            only_keys(part, {"type", "image_url"}, path)
            image = part.get("image_url")
            only_keys(image, {"url", "detail"}, path + ".image_url")
            url = string(image.get("url"), path + ".image_url.url", empty=False)
            if not (url.startswith("https://") or re.match(r"^data:image/(png|jpeg|webp|gif);base64,", url)):
                raise OpenAIError("Use an HTTPS image URL or an inline base64 image.", path + ".image_url.url")
            detail = image.get("detail", "auto")
            if detail not in ("auto", "low", "high"):
                raise OpenAIError("Unsupported image detail.", path + ".image_url.detail")
            parts.append({"type": "input_image", "image_url": url, "detail": detail})
        else:
            raise OpenAIError("Unsupported message content part.", path + ".type")
    return parts


def translate_request(body, *, strict=False):
    """Return (Responses request, ignored parameter names), retaining Chat defaults where APIs differ.

    Unless strict, options with no Responses equivalent that do not change the shape of the
    answer are dropped and reported instead of rejected.
    """
    require(body, dict, None)
    ignored = []
    if not strict:
        body = dict(body)
        for key, default in DROPPABLE.items():
            if body.get(key) is not None and body[key] != default:
                ignored.append(key)
                body.pop(key)
        if body.get("stream_options") is not None and not body.get("stream"):
            ignored.append("stream_options")
            body.pop("stream_options")
    if body.get("functions") is not None or body.get("function_call") is not None:
        raise OpenAIError("Legacy functions are not supported; use tools and tool_choice.",
                          "functions" if body.get("functions") is not None else "function_call")
    shared = {"parallel_tool_calls", "service_tier", "prompt_cache_key"}
    special = {"model", "messages", "stream", "stream_options", "tools", "tool_choice",
               "response_format", "max_tokens", "max_completion_tokens", "reasoning_effort",
               "verbosity", "store", "n", "stop", "presence_penalty", "frequency_penalty",
               "seed", "logprobs", "top_logprobs", "web_search_options"}
    only_keys(body, shared | special)
    model = string(body.get("model"), "model", empty=False)
    messages = require(body.get("messages"), list, "messages")
    if not messages:
        raise OpenAIError("At least one message is required.", "messages")
    if body.get("stream") is not None:
        require(body["stream"], bool, "stream")
    neutral = {"store": False, "n": 1, "presence_penalty": 0, "frequency_penalty": 0,
               "logprobs": False, "top_logprobs": 0, "stop": None, "seed": None}
    for key, default in neutral.items():
        value = body.get(key)
        if value is not None and (value != default or (key in {"store", "logprobs"} and not isinstance(value, bool))
                                  or (key == "n" and type(value) is not int)):
            raise OpenAIError(f"Unsupported value for '{key}': ChatGPT sign-in has no equivalent.", key)
    options = body.get("stream_options")
    if options is not None:
        only_keys(options, {"include_usage"}, "stream_options")
        if not body.get("stream"):
            raise OpenAIError("stream_options requires stream=true.", "stream_options")
        if options.get("include_usage") is not None:
            require(options["include_usage"], bool, "stream_options.include_usage")

    result = {"model": model, "input": [], "store": False, "stream": True}
    pending_calls, seen_calls = set(), set()
    for i, message in enumerate(messages):
        path = f"messages[{i}]"
        if not strict and isinstance(message, dict) and message.get("name") is not None:
            message = {k: v for k, v in message.items() if k != "name"}
            if "messages[].name" not in ignored:
                ignored.append("messages[].name")
        only_keys(message, {"role", "content", "tool_calls", "tool_call_id", "refusal"}, path)
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise OpenAIError("Unsupported message role.", path + ".role")
        if role != "assistant" and (message.get("tool_calls") is not None or message.get("refusal") is not None):
            raise OpenAIError("Only assistant messages can contain tool calls or refusals.", path)
        if role != "tool" and message.get("tool_call_id") is not None:
            raise OpenAIError("Only tool messages can contain tool_call_id.", path + ".tool_call_id")
        if role == "tool":
            call_id = string(message.get("tool_call_id"), path + ".tool_call_id", empty=False)
            if call_id not in pending_calls:
                raise OpenAIError("Tool result must match an outstanding assistant tool call.", path + ".tool_call_id")
            output = message_content(message.get("content"), role, path + ".content")
            if isinstance(output, list):
                output = "".join(part["text"] for part in output)
            result["input"].append({"type": "function_call_output", "call_id": call_id, "output": output})
            pending_calls.remove(call_id)
            continue
        content = message.get("content")
        refusal = message.get("refusal")
        calls = message.get("tool_calls")
        if content is None and not (role == "assistant" and (calls or refusal)):
            raise OpenAIError("Message content is required.", path + ".content")
        if content is not None or refusal is not None:
            converted = message_content(content, role, path + ".content") if content is not None else []
            if role == "assistant" and (isinstance(converted, list) or refusal is not None):
                if isinstance(converted, str):
                    converted = [{"type": "output_text", "text": converted, "annotations": []}]
                if refusal is not None:
                    converted.append({"type": "refusal", "refusal": string(refusal, path + ".refusal")})
                item = {"type": "message", "role": role, "status": "completed", "content": converted}
            else:
                item = {"role": role, "content": converted}
            result["input"].append(item)
        if calls is not None:
            require(calls, list, path + ".tool_calls")
            for j, call in enumerate(calls):
                cp = f"{path}.tool_calls[{j}]"
                only_keys(call, {"id", "type", "function"}, cp)
                if call.get("type") != "function":
                    raise OpenAIError("Only function tools are supported.", cp + ".type")
                call_id = string(call.get("id"), cp + ".id", empty=False)
                if call_id in seen_calls:
                    raise OpenAIError("Tool-call IDs must be unique.", cp + ".id")
                function = call.get("function")
                only_keys(function, {"name", "arguments"}, cp + ".function")
                result["input"].append({"type": "function_call", "call_id": call_id,
                    "name": string(function.get("name"), cp + ".function.name", empty=False),
                    "arguments": string(function.get("arguments"), cp + ".function.arguments")})
                pending_calls.add(call_id)
                seen_calls.add(call_id)
    if pending_calls:
        raise OpenAIError("Provide a tool result for every assistant tool call.", "messages")

    if body.get("tools") is not None:
        result["tools"] = []
        for i, tool in enumerate(require(body["tools"], list, "tools")):
            path = f"tools[{i}]"
            only_keys(tool, {"type", "function"}, path)
            if tool.get("type") != "function":
                raise OpenAIError("Only function tools are supported.", path + ".type")
            function = tool.get("function")
            only_keys(function, {"name", "description", "parameters", "strict"}, path + ".function")
            string(function.get("name"), path + ".function.name", empty=False)
            strict = function.get("strict")
            if strict is not None:
                require(strict, bool, path + ".function.strict")
            result["tools"].append({"type": "function", **{k: v for k, v in function.items() if v is not None},
                                    "strict": strict if strict is not None else False})
    search = body.get("web_search_options")
    if search is not None:
        only_keys(search, {"search_context_size", "user_location"}, "web_search_options")
        tool = {"type": "web_search"}
        if search.get("search_context_size") is not None:
            tool["search_context_size"] = search["search_context_size"]
        location = search.get("user_location")
        if location is not None:
            only_keys(location, {"type", "approximate"}, "web_search_options.user_location")
            tool["user_location"] = {"type": "approximate", **(location.get("approximate") or {})}
        result.setdefault("tools", []).append(tool)
    choice = body.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str) and choice in ("auto", "none", "required"):
            result["tool_choice"] = choice
        else:
            only_keys(choice, {"type", "function"}, "tool_choice")
            if choice.get("type") != "function":
                raise OpenAIError("Unsupported tool choice.", "tool_choice.type")
            only_keys(choice.get("function"), {"name"}, "tool_choice.function")
            result["tool_choice"] = {"type": "function", "name": string(choice["function"].get("name"), "tool_choice.function.name", empty=False)}
    fmt = body.get("response_format")
    if fmt is not None:
        only_keys(fmt, {"type", "json_schema"}, "response_format")
        if fmt.get("type") == "json_schema":
            schema = fmt.get("json_schema")
            only_keys(schema, {"name", "description", "schema", "strict"}, "response_format.json_schema")
            string(schema.get("name"), "response_format.json_schema.name", empty=False)
            require(schema.get("schema"), dict, "response_format.json_schema.schema")
            result["text"] = {"format": {"type": "json_schema", **{k: v for k, v in schema.items() if v is not None}}}
        elif fmt.get("type") in ("text", "json_object") and fmt.get("json_schema") is None:
            result["text"] = {"format": {"type": fmt["type"]}}
        else:
            raise OpenAIError("Unsupported response format.", "response_format")
    limits = [key for key in ("max_tokens", "max_completion_tokens") if body.get(key) is not None]
    if len(limits) > 1:
        raise OpenAIError("Supply only one token-limit parameter.", "max_completion_tokens")
    if limits:
        value = require(body[limits[0]], int, limits[0])
        if value <= 0:
            raise OpenAIError("Token limit must be positive.", limits[0])
        result["max_output_tokens"] = value
    if body.get("reasoning_effort") is not None:
        result["reasoning"] = {"effort": body["reasoning_effort"]}
    if body.get("verbosity") is not None:
        result.setdefault("text", {})["verbosity"] = body["verbosity"]
    result.update({key: body[key] for key in shared if body.get(key) is not None})
    return result, ignored



def finish_reason(response):
    status = response.get("status")
    if status == "completed":
        return "tool_calls" if any(item.get("type") == "function_call" for item in response.get("output", [])) else "stop"
    if status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            return "length"
        if reason == "content_filter":
            return "content_filter"
    raise failure(response.get("error"))


def usage(response):
    original = response.get("usage")
    if original is None:
        return None
    if not isinstance(original, dict) or any(type(original.get(k)) is not int for k in ("input_tokens", "output_tokens", "total_tokens")):
        raise protocol_error()
    result = {"prompt_tokens": original["input_tokens"], "completion_tokens": original["output_tokens"], "total_tokens": original["total_tokens"]}
    for source, target, key in (("input_tokens_details", "prompt_tokens_details", "cached_tokens"),
                                ("output_tokens_details", "completion_tokens_details", "reasoning_tokens")):
        if isinstance(original.get(source), dict) and key in original[source]:
            result[target] = {key: original[source][key]}
    return result


def citations(output):
    """Responses `url_citation` annotations in Chat shape, offset into the joined message text."""
    result, offset = [], 0
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") != "output_text":
                continue
            for note in part.get("annotations") or []:
                if isinstance(note, dict) and note.get("type") == "url_citation" and isinstance(note.get("url"), str):
                    cite = {"url": note["url"], "title": note.get("title") or ""}
                    for field in ("start_index", "end_index"):
                        if isinstance(note.get(field), int):
                            cite[field] = note[field] + offset
                    result.append({"type": "url_citation", "url_citation": cite})
            offset += len(part.get("text") or "")
    return result


class ChatStream:
    """One request's event conversion state; tool indexes exclude reasoning items."""

    def __init__(self, model, include_usage=False):
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.model = model
        self.include_usage = include_usage
        self.started = False
        self.terminal = None
        self.tools = {}
        self.parts = {}
        self.items = {}

    def chunk(self, delta, finish=None, *, token_usage=None, choices=True):
        result = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                  "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}] if choices else []}
        if self.include_usage:
            result["usage"] = token_usage
        return result

    def start(self):
        if self.started:
            return []
        self.started = True
        return [self.chunk({"role": "assistant", "content": ""})]

    def part(self, output_index, content_index, kind, value, *, full=False):
        if kind not in ("output_text", "refusal") or not isinstance(value, str):
            raise protocol_error()
        key = (output_index, content_index, kind)
        old = self.parts.get(key, "")
        if full:
            if not value.startswith(old):
                raise protocol_error("OpenAI returned inconsistent stream content.")
            value = value[len(old):]
        self.parts[key] = old + value
        return [self.chunk({"content" if kind == "output_text" else "refusal": value})] if value else []

    def tool(self, output_index, item):
        if not isinstance(item.get("call_id"), str) or not isinstance(item.get("name"), str):
            raise protocol_error()
        if output_index not in self.tools:
            self.tools[output_index] = {"index": len(self.tools), "id": item["call_id"], "name": item["name"], "arguments": ""}
            tool = self.tools[output_index]
            chunks = [self.chunk({"tool_calls": [{"index": tool["index"], "id": tool["id"], "type": "function",
                                                   "function": {"name": tool["name"], "arguments": ""}}]})]
        else:
            tool, chunks = self.tools[output_index], []
            if tool["id"] != item["call_id"] or tool["name"] != item["name"]:
                raise protocol_error()
        chunks.extend(self.arguments(output_index, item.get("arguments", ""), full=True))
        return chunks

    def arguments(self, output_index, value, *, full=False):
        if output_index not in self.tools or not isinstance(value, str):
            raise protocol_error()
        tool = self.tools[output_index]
        if full:
            if not value.startswith(tool["arguments"]):
                raise protocol_error("OpenAI returned inconsistent function arguments.")
            value = value[len(tool["arguments"]):]
        tool["arguments"] += value
        return [self.chunk({"tool_calls": [{"index": tool["index"], "function": {"arguments": value}}]})] if value else []

    def item(self, index, item):
        if not isinstance(item, dict):
            raise protocol_error()
        if item.get("type") == "function_call":
            return self.tool(index, item)
        chunks = []
        if item.get("type") == "message":
            for ci, part in enumerate(item.get("content", [])):
                kind = part.get("type")
                if kind in ("output_text", "refusal"):
                    chunks.extend(self.part(index, ci, kind, part.get("text" if kind == "output_text" else "refusal"), full=True))
        return chunks

    def consume(self, event):
        if self.terminal is not None:
            raise protocol_error()
        kind = event["type"]
        if kind == "error":
            raise failure(event, "OpenAI returned a stream error.")
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            response = fill_output(event.get("response"), self.items)
            if not isinstance(response, dict):
                raise protocol_error()
            expected = kind.removeprefix("response.")
            if response.get("status") != expected:
                raise protocol_error()
            if expected == "failed":
                finish_reason(response)
            if not isinstance(response.get("output"), list):
                raise protocol_error()
            finish = finish_reason(response)
            token_usage = usage(response)
            final_tools, final_parts = set(), set()
            for oi, item in enumerate(response["output"]):
                if not isinstance(item, dict):
                    raise protocol_error()
                if item.get("type") == "function_call":
                    final_tools.add(oi)
                elif item.get("type") == "message":
                    for ci, part in enumerate(item.get("content", [])):
                        final_parts.add((oi, ci, part.get("type")))
            if not set(self.tools).issubset(final_tools) or not set(self.parts).issubset(final_parts):
                raise protocol_error("OpenAI's terminal response omitted previously streamed output.")
            chunks = self.start()
            for oi, item in enumerate(response["output"]):
                chunks.extend(self.item(oi, item))
            self.terminal = response
            if notes := citations(response["output"]):
                chunks.append(self.chunk({"annotations": notes}))
            chunks.append(self.chunk({}, finish))
            if self.include_usage:
                chunks.append(self.chunk({}, token_usage=token_usage, choices=False))
            return chunks
        if kind == "response.created":
            response = event.get("response")
            if not isinstance(response, dict) or self.started:
                raise protocol_error()
            self.model = response.get("model") or self.model
            self.created = response.get("created_at", self.created)
            return self.start()
        if kind in ("response.output_item.added", "response.output_item.done"):
            if kind == "response.output_item.done":
                remember_item(self.items, event)
            return self.start() + self.item(event["output_index"], event["item"])
        if kind in ("response.output_text.delta", "response.refusal.delta"):
            part_kind = "output_text" if kind == "response.output_text.delta" else "refusal"
            return self.start() + self.part(event["output_index"], event["content_index"], part_kind, event["delta"])
        if kind in ("response.output_text.done", "response.refusal.done"):
            part_kind = "output_text" if kind == "response.output_text.done" else "refusal"
            return self.start() + self.part(event["output_index"], event["content_index"], part_kind,
                                            event["text" if part_kind == "output_text" else "refusal"], full=True)
        if kind in ("response.function_call_arguments.delta", "response.function_call_arguments.done"):
            full = kind.endswith(".done")
            return self.start() + self.arguments(event["output_index"], event["arguments" if full else "delta"], full=full)
        return []  # Progress and private reasoning events have no Chat equivalent.

    def completion(self):
        if self.terminal is None:
            raise protocol_error()
        text, refusal, calls = [], [], []
        for item in self.terminal["output"]:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text.append(part["text"])
                    elif part.get("type") == "refusal":
                        refusal.append(part["refusal"])
            elif item.get("type") == "function_call":
                calls.append({"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}})
        message = {"role": "assistant", "content": "".join(text) if text else None, "refusal": "".join(refusal) if refusal else None}
        if calls:
            message["tool_calls"] = calls
        if notes := citations(self.terminal["output"]):
            message["annotations"] = notes
        return {"id": self.id, "object": "chat.completion", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(self.terminal), "logprobs": None}],
                "usage": usage(self.terminal)}
