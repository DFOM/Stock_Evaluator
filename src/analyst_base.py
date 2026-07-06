"""Pluggable AI analyst adapters: base class, config, and response schema.

Every analyst — regardless of provider — receives the same briefing packet
and must return the same strict AnalystResponse, so the ensemble can combine
N opinions into a consensus range and a disagreement index without caring
which provider produced which opinion. Provider-specific request handling
lives in analyst_adapters.py; everything shared (prompt construction,
defensive JSON parsing, the retry-once policy, cost math) lives here.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

PILLARS = ("valuation", "quality", "growth", "financial_health", "macro", "momentum")

SYSTEM_PROMPT = """You are a professional equity analyst. You will receive a briefing packet \
(JSON) containing a company's fundamentals, computed ratios, a rule-based archetype \
classification, pillar scores, red flags, and a rule-based fair value estimate.

Form your own independent opinion. Do not simply echo the rule-based numbers — challenge \
them where the data warrants it.

Respond with ONLY a single JSON object, no markdown fences, no prose before or after, \
matching exactly this schema:
{
  "fair_value_low": <number, low end of your fair value estimate per share, in the company's listing currency>,
  "fair_value_high": <number, high end of your fair value estimate per share>,
  "confidence": <number between 0 and 1, how confident you are in this range>,
  "pillar_scores": {"valuation": <0-10>, "quality": <0-10>, "growth": <0-10>, "financial_health": <0-10>, "macro": <0-10>, "momentum": <0-10>},
  "key_risks": [<up to 5 short strings, the most important risks>],
  "archetype_agreement": <true if you agree with the briefing's archetype classification, else false>,
  "suggested_archetype": <null if you agree; otherwise one of "cash_cow", "growth", "financial", "asset_play", "turnaround">,
  "one_paragraph_thesis": <string, one paragraph summarizing your investment thesis>
}"""

RETRY_PROMPT = (
    "Your previous response was not valid JSON matching the required schema. "
    "Respond again with ONLY the JSON object — no markdown fences, no explanation, no text outside the JSON."
)


@dataclass
class AnalystConfig:
    """One analyst's configuration. Mirrors the future DB row exactly
    (config/analysts.yaml is the interim store) so the Phase 3 migration is
    a straight copy. api_key_env names the environment variable holding the
    key — keys themselves live in a local .env, never in this config."""

    name: str
    provider_type: str  # "anthropic" | "openai_compatible"
    model: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    active: bool = True
    weight: float = 1.0
    cost_per_mtok_input: float = 0.0
    cost_per_mtok_output: float = 0.0


@dataclass
class AnalystResponse:
    """The strict schema every adapter must return. ok=False marks a failed
    run (network error, invalid JSON after retry, missing key) — failures are
    recorded and displayed, never silently dropped, and never crash the run."""

    analyst_name: str
    ok: bool = False
    error: Optional[str] = None

    fair_value_low: Optional[float] = None
    fair_value_high: Optional[float] = None
    confidence: float = 0.0
    pillar_scores: dict[str, float] = field(default_factory=dict)
    key_risks: list[str] = field(default_factory=list)
    archetype_agreement: Optional[bool] = None
    suggested_archetype: Optional[str] = None
    one_paragraph_thesis: Optional[str] = None

    weight: float = 1.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    raw_response: Optional[str] = None

    @property
    def midpoint(self) -> Optional[float]:
        if self.fair_value_low is None or self.fair_value_high is None:
            return None
        return (self.fair_value_low + self.fair_value_high) / 2

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "analyst_name": self.analyst_name,
            "ok": self.ok,
            "error": self.error,
            "fair_value_low": self.fair_value_low,
            "fair_value_high": self.fair_value_high,
            "confidence": self.confidence,
            "pillar_scores": self.pillar_scores,
            "key_risks": self.key_risks,
            "archetype_agreement": self.archetype_agreement,
            "suggested_archetype": self.suggested_archetype,
            "one_paragraph_thesis": self.one_paragraph_thesis,
            "weight": self.weight,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": round(self.estimated_cost, 6),
        }


def extract_json(text: str) -> dict[str, Any]:
    """Parse a model response into a dict, defensively: strip markdown fences,
    then fall back to the outermost {...} span. Raises ValueError if nothing
    parseable is found."""
    if not text:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(cleaned[start : end + 1])


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class AnalystAdapter(ABC):
    """Base class for one pluggable AI analyst. Subclasses implement a single
    provider-specific method (_complete); the shared analyze() flow — prompt
    construction, parse, one retry, cost accounting, error capture — lives here."""

    def __init__(self, config: AnalystConfig, api_key: str):
        self.config = config
        self.api_key = api_key

    @abstractmethod
    def _complete(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        """Send a chat exchange to the provider. `messages` alternates
        user/assistant roles (the system prompt is handled per-provider).
        Returns (response_text, input_tokens, output_tokens). May raise —
        analyze() catches everything."""
        raise NotImplementedError

    def analyze(self, briefing_packet: dict[str, Any]) -> AnalystResponse:
        resp = AnalystResponse(analyst_name=self.config.name, weight=self.config.weight)
        packet_json = json.dumps(briefing_packet, default=str)
        messages = [{"role": "user", "content": f"Analyze this company:\n\n{packet_json}"}]

        total_in = total_out = 0
        try:
            text, tin, tout = self._complete(messages)
            total_in, total_out = tin, tout
            try:
                data = extract_json(text)
            except (ValueError, json.JSONDecodeError):
                # One corrective retry, then give up and mark the run failed.
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": RETRY_PROMPT},
                ]
                text, tin, tout = self._complete(messages)
                total_in += tin
                total_out += tout
                data = extract_json(text)

            resp.raw_response = text
            resp.fair_value_low = _num(data.get("fair_value_low"))
            resp.fair_value_high = _num(data.get("fair_value_high"))
            if resp.fair_value_low is not None and resp.fair_value_high is not None and resp.fair_value_low > resp.fair_value_high:
                resp.fair_value_low, resp.fair_value_high = resp.fair_value_high, resp.fair_value_low
            conf = _num(data.get("confidence"))
            resp.confidence = min(1.0, max(0.0, conf)) if conf is not None else 0.0
            raw_pillars = data.get("pillar_scores") or {}
            resp.pillar_scores = {
                p: s for p in PILLARS if (s := _num(raw_pillars.get(p))) is not None
            }
            risks = data.get("key_risks")
            resp.key_risks = [str(r) for r in risks][:5] if isinstance(risks, list) else []
            agreement = data.get("archetype_agreement")
            resp.archetype_agreement = bool(agreement) if agreement is not None else None
            suggested = data.get("suggested_archetype")
            resp.suggested_archetype = str(suggested) if suggested else None
            thesis = data.get("one_paragraph_thesis")
            resp.one_paragraph_thesis = str(thesis) if thesis else None

            if resp.fair_value_low is None or resp.fair_value_high is None:
                resp.ok = False
                resp.error = "response parsed but fair_value_low/high missing or non-numeric"
            else:
                resp.ok = True
        except Exception as e:
            resp.ok = False
            resp.error = f"{type(e).__name__}: {e}"

        resp.input_tokens = total_in
        resp.output_tokens = total_out
        resp.estimated_cost = (
            total_in / 1e6 * self.config.cost_per_mtok_input
            + total_out / 1e6 * self.config.cost_per_mtok_output
        )
        return resp
