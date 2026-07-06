"""One entry point for a full per-ticker analysis.

Both consumers — the CLI (analyze.py) and the web app (webapp/) — call
run_full_analysis() and render from the result, so the engine stays a plain
importable package and the two surfaces can't drift apart.

Also home to two cross-cutting features that need the whole picture:
  - the unanimous-dispute rule: when every blind AI analyst rejects the
    classifier's archetype and the classifier itself wasn't confident,
    fair value is re-estimated under the AI-suggested archetype and BOTH
    valuations are surfaced;
  - the position-sizing hint (documented heuristic, not advice).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.archetype_classifier import classify
from src.briefing import build_briefing_packet, redact_for_blind
from src.data_fetcher import fetch_company_data
from src.ensemble import RULE_BASED_NAME, DualPassResult, run_dual_pass
from src.fair_value import estimate_fair_value
from src.models import ArchetypeResult, CompanyData, FairValueResult, RatioSet, ScoreResult
from src.ratio_engine import compute_ratios
from src.scoring import score_company

# The unanimous-dispute rule only fires when the classifier was genuinely
# unsure AND at least this many blind AI analysts all disagreed with it.
ALT_VALUATION_MAX_CLASSIFIER_CONFIDENCE = 0.5
ALT_VALUATION_MIN_DISPUTERS = 2


@dataclass
class AltValuation:
    archetype: str
    reason: str
    fair_value: FairValueResult


@dataclass
class PipelineResult:
    ticker: str
    data: CompanyData
    ratios: RatioSet
    archetype: ArchetypeResult
    scores: ScoreResult
    fair_value: FairValueResult
    dual: Optional[DualPassResult] = None
    alt_valuation: Optional[AltValuation] = None
    position_hint: Optional[dict[str, Any]] = None

    @property
    def total_cost(self) -> float:
        return self.dual.total_cost if self.dual else 0.0

    def to_payload(self) -> dict[str, Any]:
        """Full card as a JSON-serializable dict — what the web app stores in
        the analyses table and renders from."""
        d, fv = self.data, self.fair_value
        payload: dict[str, Any] = {
            "ticker": self.ticker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "company": {
                "name": d.long_name,
                "sector": d.sector,
                "industry": d.industry,
                "country": d.country,
                "currency": d.currency,
                "current_price": d.current_price,
                "market_cap": d.market_cap,
                "enterprise_value": d.enterprise_value,
            },
            "archetype": {
                "name": self.archetype.archetype,
                "confidence": self.archetype.confidence,
                "reasons": self.archetype.reasons,
                "candidate_scores": self.archetype.scores,
            },
            "ratios": self.ratios.as_dict(),
            "ratio_completeness": round(self.ratios.completeness, 2),
            "pillar_scores": {
                name: {"score": ps.score, "confidence": ps.confidence, "notes": ps.notes}
                for name, ps in self.scores.pillar_scores.items()
            },
            "investor_score": self.scores.investor_score,
            "trader_score": self.scores.trader_score,
            "quadrant": {"verdict": self.scores.quadrant, "reason": self.scores.quadrant_reason},
            "red_flags": [
                {"name": f.name, "description": f.description, "score_cap": f.score_cap}
                for f in self.scores.red_flags
            ],
            "info_notes": self.scores.info_notes,
            "fair_value": _fv_dict(fv),
            "alt_valuation": (
                {
                    "archetype": self.alt_valuation.archetype,
                    "reason": self.alt_valuation.reason,
                    "fair_value": _fv_dict(self.alt_valuation.fair_value),
                }
                if self.alt_valuation
                else None
            ),
            "position_hint": self.position_hint,
            "data_warnings": d.warnings,
            "ensemble": None,
        }
        if self.dual:
            payload["ensemble"] = {
                "mode": self.dual.mode,
                "consensus": self.dual.consensus.as_log_dict(),
                "contributors": [r.as_log_dict() for r in self.dual.consensus.responses],
                "critic": (
                    {
                        "summary": self.dual.critic.as_log_dict(),
                        "analysts": [r.as_log_dict() for r in self.dual.critic.responses],
                    }
                    if self.dual.critic
                    else None
                ),
                "total_cost": round(self.dual.total_cost, 6),
            }
        return payload


def _fv_dict(fv: FairValueResult) -> dict[str, Any]:
    return {
        "method": fv.method,
        "low": fv.low,
        "mid": fv.mid,
        "high": fv.high,
        "currency": fv.currency,
        "confidence": fv.confidence,
        "margin_of_safety_price": fv.margin_of_safety_price,
        "assumptions": fv.assumptions,
        "notes": fv.notes,
    }


def maybe_alt_valuation(
    data: CompanyData,
    ratios: RatioSet,
    archetype: ArchetypeResult,
    dual: Optional[DualPassResult],
    config: dict[str, Any],
) -> Optional[AltValuation]:
    """The unanimous-dispute rule (motivated by JJSF): blind-pass archetype
    dispute is unanimous across 2+ AI analysts AND the classifier's own
    confidence was <= 0.5 -> re-run fair value under the modal AI-suggested
    archetype so the card shows both valuations side by side."""
    if dual is None or archetype.confidence > ALT_VALUATION_MAX_CLASSIFIER_CONFIDENCE:
        return None

    blind_ai = [
        r for r in dual.consensus.responses
        if r.ok and r.analyst_name != RULE_BASED_NAME
    ]
    if len(blind_ai) < ALT_VALUATION_MIN_DISPUTERS:
        return None
    if not all(r.archetype_agreement is False for r in blind_ai):
        return None

    suggestions = [r.suggested_archetype for r in blind_ai if r.suggested_archetype]
    if not suggestions:
        return None
    counts = Counter(suggestions).most_common(2)
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None  # tied suggestions — ambiguous, don't guess
    suggested = counts[0][0]
    if suggested == archetype.archetype:
        return None

    alt_archetype = ArchetypeResult(
        archetype=suggested,
        confidence=archetype.confidence,
        reasons=[f"AI-suggested alternative (unanimous blind-pass dispute of '{archetype.archetype}')"],
        scores={},
    )
    alt_fv = estimate_fair_value(data, ratios, alt_archetype, config)
    if alt_fv.mid is None:
        return None
    reason = (
        f"All {len(blind_ai)} blind AI analysts disputed the '{archetype.archetype}' classification "
        f"(classifier confidence {archetype.confidence:.0%}) and suggested '{suggested}' — "
        f"fair value re-estimated under that archetype for comparison."
    )
    return AltValuation(archetype=suggested, reason=reason, fair_value=alt_fv)


def position_hint(
    scores: ScoreResult,
    fair_value: FairValueResult,
    dual: Optional[DualPassResult],
) -> dict[str, Any]:
    """Suggested MAXIMUM position size as % of portfolio. A documented
    heuristic, not advice: 10% ceiling x ensemble confidence x
    (investor score / 10), halved when any red flag caps the score."""
    conf = None
    if dual and dual.consensus.overall_confidence is not None:
        conf = dual.consensus.overall_confidence
    elif fair_value.confidence:
        conf = fair_value.confidence
    conf = conf or 0.0
    inv = scores.investor_score or 0.0

    pct = 10.0 * conf * (inv / 10.0)
    flagged = bool(scores.red_flags)
    if flagged:
        pct *= 0.5
    pct = round(pct, 1)

    rationale = (
        f"10% ceiling x {conf:.0%} ensemble confidence x investor score {inv:.1f}/10"
        + (" x 0.5 (red flags cap the score)" if flagged else "")
    )
    return {
        "max_position_pct": pct,
        "entry_price": fair_value.margin_of_safety_price,
        "rationale": rationale,
        "disclaimer": "Heuristic sizing hint, not financial advice.",
    }


def run_full_analysis(
    ticker: str,
    config: dict[str, Any],
    use_ai: bool = True,
    only_analysts: Optional[list[str]] = None,
    mode: str = "dual",
) -> PipelineResult:
    """fetch -> ratios -> archetype -> scoring -> rule-based fair value ->
    dual-pass ensemble -> unanimous-dispute alt valuation -> position hint.

    Raises ValueError for an unresolvable ticker; analyst/API failures are
    absorbed by the ensemble layer and reported in the result instead."""
    data = fetch_company_data(ticker)
    if not data.current_price and not data.info:
        raise ValueError(f"Could not fetch any data for '{ticker}' — check the symbol.")

    ratios = compute_ratios(data)
    archetype = classify(data, ratios)
    scores = score_company(data, ratios, archetype, config)
    fair_value = estimate_fair_value(data, ratios, archetype, config)

    dual = None
    if use_ai:
        briefing = build_briefing_packet(data, ratios, archetype, scores, fair_value)
        blind_briefing = redact_for_blind(briefing)
        dual = run_dual_pass(
            briefing, blind_briefing, fair_value,
            only_names=only_analysts, mode=mode, current_price=data.current_price,
        )

    alt = maybe_alt_valuation(data, ratios, archetype, dual, config)
    hint = position_hint(scores, fair_value, dual)

    return PipelineResult(
        ticker=data.ticker,
        data=data,
        ratios=ratios,
        archetype=archetype,
        scores=scores,
        fair_value=fair_value,
        dual=dual,
        alt_valuation=alt,
        position_hint=hint,
    )
