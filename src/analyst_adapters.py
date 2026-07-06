"""Concrete provider adapters.

Two adapters cover the whole market: Anthropic-format, and OpenAI-compatible
chat completions (OpenAI, Groq, DeepSeek, Mistral, xAI, Ollama, etc. — the
provider is selected purely by base_url + model in config/analysts.yaml).
"""
from __future__ import annotations

import anthropic
from openai import OpenAI

from src.analyst_base import SYSTEM_PROMPT, AnalystAdapter, AnalystConfig

REQUEST_TIMEOUT_SECONDS = 120.0
MAX_OUTPUT_TOKENS = 2048


class AnthropicAnalyst(AnalystAdapter):
    """Calls the Anthropic Messages API."""

    def __init__(self, config: AnalystConfig, api_key: str):
        super().__init__(config, api_key)
        self._client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)

    def _complete(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
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

    def _complete(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                messages=full_messages,
            )
        except Exception as e:
            # Older OpenAI-compatible servers only know the legacy param name.
            if "max_completion_tokens" not in str(e):
                raise
            response = self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=full_messages,
            )
        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return text, input_tokens, output_tokens


ADAPTERS: dict[str, type[AnalystAdapter]] = {
    "anthropic": AnthropicAnalyst,
    "openai_compatible": OpenAICompatibleAnalyst,
}
