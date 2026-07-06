"""Phase 2 interface stub: pluggable AI analyst adapters.

Nothing in this module makes network calls yet. It exists so Phase 2 can add
real Anthropic-format and OpenAI-compatible adapters without touching the
Phase 1 pipeline: every analyst will take the same briefing packet (see
briefing.py) and return the same AnalystResponse shape, so the ensemble can
combine N analysts' opinions into a consensus range and a disagreement index
without caring which provider produced which opinion.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AnalystConfig:
    """Mirrors the future DB row for one configured analyst.

    api_key is expected to be decrypted by the caller before being placed
    here (Phase 2 will encrypt it at rest using SECRET_KEY) — this stub does
    not handle encryption itself.
    """

    name: str
    provider_type: str  # "anthropic" | "openai_compatible"
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    active: bool = True
    weight: float = 1.0


@dataclass
class AnalystResponse:
    """The strict schema every analyst adapter must return, regardless of
    provider, so responses are directly comparable across the ensemble."""

    analyst_name: str
    fair_value_low: Optional[float]
    fair_value_high: Optional[float]
    confidence: float
    key_risks: list[str] = field(default_factory=list)
    pillar_scores: dict[str, float] = field(default_factory=dict)
    archetype_agreement: Optional[bool] = None
    raw_response: Optional[str] = None


class AnalystAdapter(ABC):
    """Base class for one pluggable AI analyst. Subclasses implement the
    provider-specific request/response handling; the pipeline only ever
    calls `analyze()`."""

    def __init__(self, config: AnalystConfig):
        self.config = config

    @abstractmethod
    def analyze(self, briefing_packet: dict[str, Any]) -> AnalystResponse:
        """Send the briefing packet to the analyst and return its opinion in
        the strict AnalystResponse schema. Must not raise on a well-formed
        packet — provider/network errors should be caught and surfaced as a
        low-confidence AnalystResponse so one flaky analyst can't take down
        the whole ensemble run."""
        raise NotImplementedError


class AnthropicAnalyst(AnalystAdapter):
    """Phase 2: calls the Anthropic Messages API. Not implemented yet."""

    def analyze(self, briefing_packet: dict[str, Any]) -> AnalystResponse:
        raise NotImplementedError("AnthropicAnalyst is a Phase 2 stub — no API calls in Phase 1.")


class OpenAICompatibleAnalyst(AnalystAdapter):
    """Phase 2: calls any OpenAI-compatible chat completions endpoint
    (OpenAI, Groq, DeepSeek, Mistral, xAI, Ollama, etc. via base_url). Not
    implemented yet."""

    def analyze(self, briefing_packet: dict[str, Any]) -> AnalystResponse:
        raise NotImplementedError("OpenAICompatibleAnalyst is a Phase 2 stub — no API calls in Phase 1.")
