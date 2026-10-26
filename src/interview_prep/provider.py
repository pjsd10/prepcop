"""Small ChatOpenAI compatibility adapter for the selected hosted provider.

The application uses synchronous, non-streaming Chat Completions. Fireworks'
reasoning_content must survive an assistant tool call and its next request. The
installed langchain-openai conversion helpers omit that field in both directions,
so this adapter restores it without moving it into user-facing message content.

Provider references (checked 25 October 2026):
https://docs.fireworks.ai/api-reference/post-chatcompletions
https://docs.fireworks.ai/guides/reasoning
https://docs.fireworks.ai/structured-responses/structured-response-formatting
https://huggingface.co/zai-org/GLM-5.3-Flash
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class CompatibleChatOpenAI(ChatOpenAI):
    """Preserve the provider's opaque reasoning field through native tool turns.

    This uses two protected LangChain hooks; focused compatibility tests cover
    them. Re-run those tests when changing the pinned langchain-openai version.
    Streaming is disabled by the factory because this adapter does not implement
    reasoning delta aggregation. The factory applies an explicit low-effort
    calibration only to Fireworks' selected GLM-5.3-Flash model.
    """

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump(warnings=False)
        for choice, generation in zip(raw.get("choices", []), result.generations):
            message = choice.get("message", {})
            reasoning = message.get("reasoning_content")
            if isinstance(generation.message, AIMessage) and isinstance(reasoning, str):
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs: Any) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        for message, serialized in zip(messages, payload.get("messages", [])):
            if isinstance(message, AIMessage) and "reasoning_content" in message.additional_kwargs:
                reasoning = message.additional_kwargs["reasoning_content"]
                if isinstance(reasoning, str):
                    serialized["reasoning_content"] = reasoning
        # Both providers use the standard completion cap. Never send both aliases.
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        if payload.get("tools"):
            payload["parallel_tool_calls"] = False
        return payload


def native_response_format(schema: type[BaseModel] | dict) -> dict:
    """Use provider-native JSON Schema while retaining local Pydantic validation."""
    if isinstance(schema, dict):
        definition = deepcopy(schema)
        name = str(definition.get("title", "StructuredResponse"))
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        definition = schema.model_json_schema()
        name = schema.__name__
    else:
        raise TypeError("Structured output needs a Pydantic model class or JSON Schema object.")
    return {"type": "json_schema", "json_schema": {"name": name, "schema": definition}}


def make_chat_model(settings, *, max_tokens: int, timeout: float, schema=None) -> CompatibleChatOpenAI:
    """Build an unbound adapter. RunContext remains responsible for all attempts.

    Native tools can be bound by the caller. Schema calls return AIMessage so the
    existing durable ledger and shared malformed-output repair remain in control.
    Keys are passed only to the SDK, never inserted into prompts or event records.
    """
    if settings.provider not in ("fireworks", "nebius"):
        raise ValueError("Unsupported model provider.")
    if not settings.api_key or not settings.model:
        raise ValueError("Configure the provider API key and exact model ID first.")
    if max_tokens < 1 or timeout <= 0:
        raise ValueError("A positive model token cap and timeout are required.")
    model_kwargs = {"response_format": native_response_format(schema)} if schema is not None else {}
    extra_body = {"context_length_exceeded_behavior": "error"} if settings.provider == "fireworks" else None
    # GLM-5.3-Flash defaults to max effort; low/high/max are the vendor's named
    # controls. Low reduces reasoning demand but is not a hard token guarantee.
    reasoning_effort = "low" if settings.provider == "fireworks" and settings.model == "accounts/fireworks/models/glm-5p3-flash" else None
    return CompatibleChatOpenAI(
        model=settings.model, api_key=settings.api_key, base_url=settings.base_url,
        max_tokens=max_tokens, timeout=timeout, max_retries=0, temperature=0.2,
        reasoning_effort=reasoning_effort,
        model_kwargs=model_kwargs, extra_body=extra_body, streaming=False,
        disable_streaming=True, use_responses_api=False,
    )
