"""Chat Completions → Responses translation, in lenient (default) and strict modes."""

import pytest

from siwc_proxy.chat import translate_request
from siwc_proxy.errors import OpenAIError
from support import BODY


def test_history_images_tools_and_controls():
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    body = {**BODY, "messages": [
        {"role": "system", "content": "First instruction"},
        {"role": "developer", "content": "Second instruction"},
        {"role": "user", "content": [{"type": "text", "text": "Picture"},
             {"type": "image_url", "image_url": {"url": "https://example.com/p.png", "detail": "high"}},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGlj"}}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            {"id": "call-b", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-b", "content": "Result B"},
        {"role": "tool", "tool_call_id": "call-a", "content": [{"type": "text", "text": "Result A"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Thanks"}]},
    ], "tools": [tool], "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "max_completion_tokens": 100, "reasoning_effort": "low", "verbosity": "low",
        "parallel_tool_calls": True, "service_tier": "auto"}
    result, ignored = translate_request(body)
    assert ignored == []
    assert [m.get("role", m.get("type")) for m in result["input"]] == [
        "system", "developer", "user", "function_call", "function_call", "function_call_output", "function_call_output",
        "assistant"]
    assert result["input"][2]["content"][1] == {"type": "input_image", "image_url": "https://example.com/p.png",
                                                "detail": "high"}
    assert result["input"][5]["call_id"] == "call-b"
    assert result["input"][6]["output"] == "Result A"
    assert result["input"][7]["content"][0]["type"] == "output_text"
    assert result["tools"][0]["strict"] is False
    assert result["tool_choice"] == {"type": "function", "name": "lookup"}
    assert result["max_output_tokens"] == 100
    assert result["reasoning"] == {"effort": "low"}
    assert result["text"] == {"verbosity": "low"}
    assert result["stream"] is True and result["store"] is False
    assert tool["function"].get("strict") is None  # Input dictionaries are unchanged.


@pytest.mark.parametrize("strict", [True, False])
def test_structured_output_and_tool_strictness(strict):
    schema = {"name": "answer", "schema": {"type": "object"}, "strict": strict}
    result, _ = translate_request({**BODY, "tools": [{"type": "function", "function": {"name": "f", "strict": strict}}],
                                   "response_format": {"type": "json_schema", "json_schema": schema}})
    assert result["tools"][0]["strict"] is strict
    assert result["text"]["format"] == {"type": "json_schema", **schema}
    for kind in ("text", "json_object"):
        assert translate_request({**BODY, "response_format": {"type": kind}})[0]["text"]["format"] == {"type": kind}


LENIENT = [("user", "u-1"), ("safety_identifier", "s"), ("seed", 42), ("stop", ["end"]),
           ("presence_penalty", 1), ("frequency_penalty", -1), ("logit_bias", {"1": 2}), ("store", True),
           ("temperature", 0), ("top_p", 1), ("metadata", {"a": "b"})]


def test_prompt_cache_key_is_kept_for_the_cache_session():
    assert translate_request({**BODY, "prompt_cache_key": "k"}, strict=True) == (
        {"model": "test-model", "input": [{"role": "user", "content": "Hello"}], "store": False, "stream": True,
         "prompt_cache_key": "k"}, [])


@pytest.mark.parametrize("option,value", LENIENT)
def test_lenient_drops_and_reports_options(option, value):
    body = {**BODY, option: value}
    result, ignored = translate_request(body)
    assert ignored == [option] and result.get(option, False) is False
    assert body[option] == value


@pytest.mark.parametrize("option,value", LENIENT)
def test_strict_rejects_the_same_options(option, value):
    with pytest.raises(OpenAIError) as caught:
        translate_request({**BODY, option: value}, strict=True)
    assert caught.value.status == 400 and caught.value.payload["error"]["param"] == option
    assert f"'{option}'" in caught.value.payload["error"]["message"]  # Many clients show only the message.


def test_lenient_message_name_and_stream_options():
    body = {**BODY, "stream_options": {"include_usage": True}, "messages": [
        {"role": "user", "content": "a", "name": "alice"}, {"role": "assistant", "content": "b", "name": "bot"}]}
    result, ignored = translate_request(body)
    assert ignored == ["stream_options", "messages[].name"]
    assert all("name" not in m for m in result["input"])
    with pytest.raises(OpenAIError) as caught:
        translate_request(body, strict=True)
    assert caught.value.payload["error"]["param"] == "stream_options"


def test_web_search_options_become_a_tool():
    result, _ = translate_request({**BODY, "web_search_options": {"search_context_size": "low"}})
    assert result["tools"][-1]["type"] == "web_search"
    assert result["tools"][-1]["search_context_size"] == "low"


@pytest.mark.parametrize("option,value", [("n", 2), ("n", True), ("logprobs", True), ("top_logprobs", 2),
    ("functions", []), ("function_call", "auto"), ("audio", {}), ("stream", "true"), ("max_tokens", True),
    ("max_tokens", 0)])
def test_unsupported_options_are_rejected_in_both_modes(option, value):
    for strict in (False, True):
        with pytest.raises(OpenAIError) as caught:
            translate_request({**BODY, option: value}, strict=strict)
        assert caught.value.status == 400
        assert caught.value.payload["error"]["param"] == option


def test_legacy_functions_explain_the_replacement():
    with pytest.raises(OpenAIError, match="use tools and tool_choice"):
        translate_request({**BODY, "functions": [{"name": "f"}]})


@pytest.mark.parametrize("body,param", [
    ({**BODY, "max_tokens": 4, "max_completion_tokens": 4}, "max_completion_tokens"),
    ({**BODY, "messages": [{"role": "tool", "tool_call_id": "orphan", "content": "x"}]}, "messages[0].tool_call_id"),
    ({**BODY, "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]},
     "messages[0].content[0].type"),
    ({**BODY, "messages": ["not-a-message"]}, "messages[0]"),
])
def test_invalid_requests(body, param):
    with pytest.raises(OpenAIError) as caught:
        translate_request(body)
    assert caught.value.payload["error"]["param"] == param


def test_neutral_options_and_token_alias():
    result, ignored = translate_request({**BODY, "n": 1, "stop": None, "seed": None, "logprobs": False,
                                         "presence_penalty": 0, "frequency_penalty": 0, "store": False,
                                         "max_tokens": 4})
    assert result["max_output_tokens"] == 4 and ignored == []
    assert "n" not in result
