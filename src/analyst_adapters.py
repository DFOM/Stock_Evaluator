"""Concrete provider adapters.

Two adapters cover the whole market: Anthropic-format, and OpenAI-compatible
chat completions (OpenAI, Groq, DeepSeek, Mistral, xAI, Ollama, etc. — the
provider is selected purely by base_url + model in config/analysts.yaml).
"""
from __future__ import annotations

import anthropic
from openai import OpenAI

from src.analyst_base import AnalystAdapter, AnalystConfig

REQUEST_TIMEOUT_SECONDS = 180.0
# Generous cap: reasoning models (GPT-5.x etc.) burn internal reasoning tokens
# against the same completion budget as the visible JSON — a tight cap
# truncates the JSON mid-object even though the answer itself is short.
MAX_OUTPUT_TOKENS = 8192


class AnthropicAnalyst(AnalystAdapter):
    """Calls the Anthropic Messages API."""

    def __init__(self, config: AnalystConfig, api_key: str):
        super().__init__(config, api_key)
        self._client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)

    def _complete(self, system: str, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=messages,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.usage.input_tokens, response.usage.output_tokens


class OpenAICompatibleAnalyst(AnalystAdapter):
    """Calls any OpenAI-compatible chat completions endpoint via base_url."""

    def __init__(self, config: AnalystConfig, api_key: str):
        super().__init__(config, api_key)
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,  # None -> api.openai.com
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    def _complete(self, system: str, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        full_messages = [{"role": "system", "content": system}, *messages]
        # Start with the modern param names + JSON mode (which guarantees
        # syntactically valid JSON — some models otherwise emit near-JSON with
        # mismatched brackets). Compatible providers that reject either
        # parameter get it dropped/renamed and the request retried, so the
        # same adapter works against OpenAI, Groq, DeepSeek, Ollama, etc.
        kwargs: dict = {
            "model": self.config.model,
            "messages": full_messages,
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
        }
        for _ in range(3):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                msg = str(e)
                if "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                elif "response_format" in msg and "response_format" in kwargs:
                    del kwargs["response_format"]
                else:
                    raise
        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return text, input_tokens, output_tokens


ADAPTERS: dict[str, type[AnalystAdapter]] = {
    "anthropic": AnthropicAnalyst,
    "openai_compatible": OpenAICompatibleAnalyst,
}
