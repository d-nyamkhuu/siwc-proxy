"""The stock openai SDK, used the way frameworks use it, against the proxy."""

import httpx
import openai
import pytest
from openai import AsyncOpenAI, OpenAI

from siwc_proxy.config import Settings
from siwc_proxy.server import create_app
from support import BASE, BODY, KEY, SSE_TYPE, Chunks, running, sse, terminal, text_events, tool_events


@pytest.fixture
def sdk(directory, fake, monkeypatch):
    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    with running(directory, fake) as (client, fake, app):
        with OpenAI(api_key=KEY, base_url=BASE + "/v1", http_client=client, max_retries=0) as sdk:
            yield sdk, fake


def test_responses_create_without_stream_or_store(sdk):
    sdk, fake = sdk
    response = sdk.responses.create(model="test-model", input="hi")
    assert response.output_text == "Hi 世界"
    assert response.usage.total_tokens == 14
    assert "openai-organization" not in fake.calls[-1].headers


def realistic_events():
    """Text events with the item and part lifecycle the SDK's stream helper tracks."""
    message = {"id": "msg-1", "type": "message", "role": "assistant", "status": "in_progress", "content": []}
    part = {"type": "output_text", "text": "", "annotations": []}
    deltas = [{**e, "item_id": "msg-1"} for e in text_events()[2:5]]
    return [text_events()[0],
            {"type": "response.output_item.added", "output_index": 0, "item": message},
            {"type": "response.content_part.added", "output_index": 0, "content_index": 0, "item_id": "msg-1",
             "part": part},
            *deltas, text_events()[-1]]


def test_responses_streaming_and_helper(sdk):
    sdk, fake = sdk
    fake.api = lambda r: httpx.Response(200, headers=SSE_TYPE, stream=Chunks(sse(realistic_events())))
    events = list(sdk.responses.create(model="test-model", input="hi", stream=True))
    assert events[-1].type == "response.completed"
    with sdk.responses.stream(model="test-model", input="hi") as stream:
        assert "".join(e.delta for e in stream if e.type == "response.output_text.delta") == "Hi 世界"
        assert stream.get_final_response().output_text == "Hi 世界"


def test_chat_with_framework_defaults(sdk):
    sdk, _ = sdk
    raw = sdk.chat.completions.with_raw_response.create(
        **BODY, user="me", seed=7, stop=["\n\n"], presence_penalty=0.5)
    assert raw.headers["x-siwc-ignored-params"] == "user,seed,stop,presence_penalty"
    completion = raw.parse()
    assert completion.choices[0].message.content == "Hi 世界"
    assert completion.usage.prompt_tokens_details.cached_tokens == 2
    assert completion._request_id == "req-test"


def test_chat_stream_helper_accumulates_tool_arguments(sdk):
    sdk, fake = sdk
    fake.api = lambda r: httpx.Response(200, headers=SSE_TYPE, stream=Chunks(sse(tool_events())))
    with sdk.chat.completions.stream(**BODY) as stream:
        list(stream)
        message = stream.get_final_completion().choices[0].message
    assert [(t.id, t.function.name, t.function.arguments) for t in message.tool_calls] == [
        ("call-a", "a", '{"x":1}'), ("call-b", "b", "{}")]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_plan_limit_is_a_rate_limit_error(sdk, endpoint):
    sdk, fake = sdk
    fake.api = lambda r: httpx.Response(200, headers=SSE_TYPE, stream=Chunks(sse([{"type": "response.failed", "response": terminal(
        status="failed", error={"message": "Limit", "code": "subscription_sharing_usage_limit_exceeded"})}])))
    with pytest.raises(openai.RateLimitError) as caught:
        if endpoint == "chat":
            sdk.chat.completions.create(**BODY)
        else:
            sdk.responses.create(model="test-model", input="hi")
    assert caught.value.code == "subscription_sharing_usage_limit_exceeded"


def test_admission_detail_error(sdk):
    sdk, fake = sdk
    fake.api = lambda r: httpx.Response(403, json={"detail": {"message": "Not eligible",
                                                              "code": "subscription_sharing_user_not_eligible"}})
    with pytest.raises(openai.PermissionDeniedError) as caught:
        sdk.chat.completions.create(**BODY)
    assert caught.value.code == "subscription_sharing_user_not_eligible"
    assert caught.value.body["message"] == "Not eligible"


def test_unknown_route_and_models(sdk):
    sdk, fake = sdk
    with pytest.raises(openai.NotFoundError):
        sdk.embeddings.create(model="x", input="hi")
    fake.api = lambda r: httpx.Response(200, json={"models": [{"slug": "test-model", "visibility": "list"}]})
    assert [m.id for m in sdk.models.list()] == ["test-model"]
    assert sdk.models.retrieve("test-model").id == "test-model"
    with pytest.raises(openai.NotFoundError):
        sdk.models.retrieve("missing")


def test_signed_out_is_an_authentication_error(tmp_path, fake):
    with running(tmp_path / "empty", fake) as (client, *_):
        with OpenAI(api_key=KEY, base_url=BASE + "/v1", http_client=client, max_retries=0) as sdk:
            with pytest.raises(openai.AuthenticationError) as caught:
                sdk.responses.create(model="m", input="hi")
    assert caught.value.code == "chatgpt_login_required"


def test_async_sdk(directory, fake):
    import anyio

    async def exercise():
        app = create_app(Settings(data_dir=directory, api_key=KEY), transport=fake.transport)
        async with app.router.lifespan_context(app):
            async with AsyncOpenAI(api_key=KEY, base_url=BASE + "/v1", max_retries=0,
                                   http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app))) as sdk:
                response = await sdk.responses.create(model="test-model", input="hi")
                assert response.output_text == "Hi 世界"
                fake.api = lambda r: httpx.Response(200, headers=SSE_TYPE, stream=Chunks(sse(text_events())))
                stream = await sdk.chat.completions.create(**BODY, stream=True)
                text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
                assert text == "Hi 世界"

    anyio.run(exercise)
