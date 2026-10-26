"""Provider payload compatibility tests; no HTTP requests or live credentials."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict, messages_from_dict
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from interview_prep.config import Settings
from interview_prep.provider import CompatibleChatOpenAI, make_chat_model, native_response_format
from interview_prep.research import RESEARCH_TOOLS


class SampleResult(BaseModel):
    answer: str


def settings(provider="fireworks"):
    return Settings(api_key="test-only-key", model="accounts/fireworks/models/glm-5p3-flash", provider=provider,
                    base_url="https://api.fireworks.ai/inference/v1" if provider == "fireworks" else "https://api.tokenfactory.nebius.com/v1")


def response(reasoning="opaque provider reasoning field"):
    return {
        "id": "completion-controlled", "object": "chat.completion", "created": 1,
        "model": "accounts/fireworks/models/glm-5p3-flash",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": reasoning,
            "tool_calls": [{"id": "call-source-1", "type": "function", "function": {
                "name": "search_sources", "arguments": json.dumps({"query": "public documentation", "topic_id": "t0",
                "reason": "Find suitable primary support.", "assessment": None})}}],
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
    }


@pytest.mark.parametrize("typed_response", [False, True])
def test_reasoning_and_tool_call_survive_raw_response_checkpoint_and_next_payload(typed_response):
    model = make_chat_model(settings(), max_tokens=1800, timeout=20)
    raw = response()
    if typed_response:
        raw = ChatCompletion.model_validate(raw)
    message = model._create_chat_result(raw).generations[0].message
    assert message.content == ""  # Provider reasoning is not mixed into visible content.
    assert message.additional_kwargs["reasoning_content"] == "opaque provider reasoning field"
    assert message.tool_calls[0]["id"] == "call-source-1"
    assert message.usage_metadata["total_tokens"] == 32
    restored = messages_from_dict([message_to_dict(message)])[0]
    tool = ToolMessage(content='{"results":[]}', tool_call_id="call-source-1")
    payload = model._get_request_payload([HumanMessage(content="Find topic sources."), restored, tool], tools=RESEARCH_TOOLS)
    assert payload["messages"][1]["reasoning_content"] == message.additional_kwargs["reasoning_content"]
    assert payload["messages"][1]["tool_calls"][0]["id"] == "call-source-1"
    assert payload["messages"][2]["tool_call_id"] == "call-source-1"
    assert payload["parallel_tool_calls"] is False
    assert "reasoning_content" not in payload["messages"][0]
    assert "reasoning_content" not in payload["messages"][2]


@pytest.mark.parametrize("reasoning", ["", None])
def test_empty_or_absent_reasoning_does_not_invent_content(reasoning):
    model = make_chat_model(settings(), max_tokens=1800, timeout=20)
    message = model._create_chat_result(response(reasoning)).generations[0].message
    payload = model._get_request_payload([message])
    if reasoning is None:
        assert "reasoning_content" not in payload["messages"][0]
    else:
        assert payload["messages"][0]["reasoning_content"] == ""


def test_native_schema_is_sent_to_provider_with_no_retrieval_tools():
    model = make_chat_model(settings(), max_tokens=4000, timeout=45, schema=SampleResult)
    payload = model._get_request_payload([HumanMessage(content="Return a JSON answer.")])
    assert payload["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "SampleResult", "schema": SampleResult.model_json_schema()}}
    assert "tools" not in payload and "parallel_tool_calls" not in payload
    assert payload["extra_body"]["context_length_exceeded_behavior"] == "error"
    assert payload["max_tokens"] == 4000 and "max_completion_tokens" not in payload
    assert model.max_retries == 0 and model.disable_streaming is True and model.use_responses_api is False
    assert model.request_timeout == 45
    assert "test-only-key" not in str(payload)


def test_nebius_receives_standard_schema_without_fireworks_only_options():
    model = make_chat_model(settings("nebius"), max_tokens=2500, timeout=30, schema=SampleResult)
    payload = model._get_request_payload([HumanMessage(content="Return JSON.")])
    assert payload["response_format"]["type"] == "json_schema"
    assert not payload.get("extra_body")
    assert payload["max_tokens"] == 2500


def test_schema_dict_is_copied_and_invalid_schema_is_rejected():
    schema = {"title": "Original", "type": "object", "properties": {"answer": {"type": "string"}}}
    value = native_response_format(schema)
    value["json_schema"]["schema"]["properties"]["answer"]["type"] = "integer"
    assert schema["properties"]["answer"]["type"] == "string"
    with pytest.raises(TypeError):
        native_response_format("invalid schema")


def test_exact_fireworks_glm_uses_low_reasoning_without_other_toggles():
    model = make_chat_model(settings(), max_tokens=1800, timeout=20)
    payload = model._get_request_payload([HumanMessage(content="Prepare useful questions.")])
    assert payload["model"] == settings().model
    assert payload["reasoning_effort"] == "low" and "thinking" not in payload.get("extra_body", {})
    assert "response_format" not in payload


@pytest.mark.parametrize("provider,model,cap,effort", [
    ("fireworks", "accounts/fireworks/models/glm-5p3-flash", 8192, "low"),
    ("fireworks", "accounts/fireworks/models/glm-5p3", 5000, None),
    ("fireworks", "accounts/fireworks/models/gpt-oss-120b", 5000, None),
    ("nebius", "accounts/fireworks/models/glm-5p3-flash", 5000, None),
    ("nebius", "other-provider-model", 5000, None),
])
@pytest.mark.parametrize("request_kind", ["schema", "tools"])
def test_environment_policy_reaches_real_payload_only_for_selected_glm(monkeypatch, provider, model, cap, effort, request_kind):
    import interview_prep.config as config

    # Isolate configuration from the real .env and every credential variable.
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name, value in {
        "MODEL_PROVIDER": provider, "FIREWORKS_MODEL": model, "NEBIUS_MODEL": model,
        "FIREWORKS_API_KEY": "test-only-fireworks-key", "NEBIUS_API_KEY": "test-only-nebius-key",
        "TAVILY_API_KEY": "test-only-tavily-key",
    }.items():
        monkeypatch.setenv(name, value)
    resolved = Settings.from_env()
    assert resolved.model_max_tokens == cap and resolved.model_timeout == 75
    adapter = make_chat_model(resolved, max_tokens=resolved.model_max_tokens, timeout=resolved.model_timeout,
                              schema=SampleResult if request_kind == "schema" else None)
    payload = adapter._get_request_payload([HumanMessage(content="Return the requested public-topic result.")],
                                          **({"tools": RESEARCH_TOOLS} if request_kind == "tools" else {}))
    assert payload["model"] == model and payload["max_tokens"] == cap
    if effort is None:
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning_effort"] == effort
    assert "thinking" not in payload and "thinking" not in payload.get("extra_body", {})
    assert adapter.max_retries == 0 and adapter.request_timeout == 75
    if request_kind == "schema":
        assert payload["response_format"]["type"] == "json_schema" and "tools" not in payload
    else:
        assert payload["tools"] == RESEARCH_TOOLS and payload["parallel_tool_calls"] is False
        assert "response_format" not in payload


def test_raw_assistant_dict_reasoning_also_survives_conversion():
    model = make_chat_model(settings(), max_tokens=1800, timeout=20)
    payload = model._get_request_payload([{"role": "assistant", "content": "Ready", "reasoning_content": "opaque"}])
    assert payload["messages"][0]["reasoning_content"] == "opaque"


def test_missing_provider_configuration_is_blocked_locally():
    with pytest.raises(ValueError, match="Configure"):
        make_chat_model(Settings(), max_tokens=1800, timeout=20)
    with pytest.raises(ValueError, match="positive"):
        make_chat_model(settings(), max_tokens=0, timeout=20)
