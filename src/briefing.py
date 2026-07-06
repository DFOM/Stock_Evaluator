"""Builds the standardized briefing packet.

This is the one artifact every downstream consumer agrees on: the console
report (Phase 1) renders it, and the AI analyst ensemble (Phase 2) will send
the same packet to every analyst so responses are comparable apples-to-apples.
Keeping it as a plain JSON-serializable dict (not a dataclass) is deliberate —
it's meant to cross a process/network boundary eventually.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.models import ArchetypeResult, CompanyData, FairValueResult, RatioSet, ScoreResult


def build_briefing_packet(
    data: CompanyData,
    ratios: RatioSet,
    archetype: ArchetypeResult,
    scores: ScoreResult,
    fair_value: FairValueResult,
) -> dict[str, Any]:
    return {
        "ticker": data.ticker,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "company": {
            "name": data.long_name,
            "sector": data.sector,
            "industry": data.industry,
            "country": data.country,
            "currency": data.currency,
            "current_price": data.current_price,
            "market_cap": data.market_cap,
            "enterprise_value": data.enterprise_value,
        },
        "archetype": {
            "name": archetype.archetype,
            "confidence": archetype.confidence,
            "reasons": archetype.reasons,
            "candidate_scores": archetype.scores,
        },
        "ratios": ratios.as_dict(),
        "ratio_completeness": round(ratios.completeness, 2),
        "pillar_scores": {
            name: {"score": ps.score, "confidence": ps.confidence, "notes": ps.notes}
            for name, ps in scores.pillar_scores.items()
        },
        "investor_score": scores.investor_score,
        "trader_score": scores.trader_score,
        "quadrant": {"verdict": scores.quadrant, "reason": scores.quadrant_reason},
        "red_flags": [
            {"name": f.name, "description": f.description, "score_cap": f.score_cap}
            for f in scores.red_flags
        ],
        "info_notes": scores.info_notes,
        "fair_value_rule_based": {
            "method": fair_value.method,
            "low": fair_value.low,
            "mid": fair_value.mid,
            "high": fair_value.high,
            "currency": fair_value.currency,
            "confidence": fair_value.confidence,
            "margin_of_safety_price": fair_value.margin_of_safety_price,
            "assumptions": fair_value.assumptions,
            "notes": fair_value.notes,
        },
        # Peer/comparables data isn't sourced yet in Phase 1 — this is the
        # slot the AI analyst ensemble (Phase 2) will populate so analysts
        # can weigh in on relative/peer valuation, not just this packet.
        "peers": [],
        "data_warnings": data.warnings,
    }
