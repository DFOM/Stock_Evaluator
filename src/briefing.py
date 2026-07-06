"""Builds the standardized briefing packet.

This is the one artifact every downstream consumer agrees on: the console
report (Phase 1) renders it, and the AI analyst ensemble (Phase 2) will send
the same packet to every analyst so responses are comparable apples-to-apples.
Keeping it as a plain JSON-serializable dict (not a dataclass) is deliberate —
it's meant to cross a process/network boundary eventually.
"""
from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.models import ArchetypeResult, CompanyData, FairValueResult, RatioSet, ScoreResult

_INCOME_ROWS = ("Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA", "Interest Expense", "Diluted EPS")
_BALANCE_ROWS = ("Total Assets", "Total Debt", "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Stockholders Equity", "Current Assets", "Current Liabilities")
_CASHFLOW_ROWS = ("Operating Cash Flow", "Capital Expenditure", "Free Cash Flow", "Depreciation And Amortization")


def _stmt_rows(df: pd.DataFrame, labels: tuple[str, ...], periods: int = 3) -> dict[str, list]:
    """Extract key statement rows (most recent annual period first) as plain
    floats/None so the packet stays JSON-serializable."""
    out: dict[str, list] = {}
    if df is None or df.empty:
        return out
    for label in labels:
        if label not in df.index:
            continue
        vals = []
        for v in df.loc[label].iloc[:periods]:
            try:
                f = float(v)
                vals.append(None if math.isnan(f) or math.isinf(f) else f)
            except (TypeError, ValueError):
                vals.append(None)
        out[label] = vals
    return out


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
        # Absolute statement figures + share count: what a blind analyst needs
        # to derive per-share fair value without any price context (and useful
        # grounding for sighted analysts too).
        "fundamentals": {
            "shares_outstanding": data.shares_outstanding,
            "statement_order": "most recent annual period first",
            "income_statement": _stmt_rows(data.income_stmt, _INCOME_ROWS),
            "balance_sheet": _stmt_rows(data.balance_sheet, _BALANCE_ROWS),
            "cash_flow": _stmt_rows(data.cashflow, _CASHFLOW_ROWS),
        },
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


# Everything the market's opinion touches. Ratios: all price-derived
# multiples and yields; beta is estimated from price history so it goes too.
_BLIND_COMPANY_FIELDS = ("current_price", "market_cap", "enterprise_value")
_BLIND_RATIO_FIELDS = (
    "pe_trailing", "pe_forward", "price_to_book", "price_to_sales",
    "ev_to_ebitda", "ev_to_revenue", "peg_ratio", "dividend_yield",
    "fcf_yield", "market_cap", "enterprise_value", "beta",
)
# Valuation pillar is scored on multiples; momentum on price history.
_BLIND_PILLARS = ("valuation", "momentum")


def redact_for_blind(packet: dict[str, Any]) -> dict[str, Any]:
    """Price-blind variant of the briefing packet (deep copy; the original is
    untouched). Strips everything derived from the market price so an analyst
    must construct fair value from fundamentals alone:
      - price, market cap, EV, and all price-derived multiples/yields
      - valuation + momentum pillars, composite scores, and the quadrant
        verdict (all contain price-derived inputs)
      - the rule-based fair value range (our own target number would simply
        replace the market price as the anchor)
      - archetype reasons/candidate scores (vote text can quote multiples);
        the archetype name + confidence stay
    Kept: statements, growth, margins, returns, debt metrics, FCF, share
    count, archetype, country, red flags."""
    p = copy.deepcopy(packet)
    for f in _BLIND_COMPANY_FIELDS:
        p["company"].pop(f, None)
    for f in _BLIND_RATIO_FIELDS:
        p["ratios"].pop(f, None)
    p["pillar_scores"] = {
        k: v for k, v in p["pillar_scores"].items() if k.lower() not in _BLIND_PILLARS
    }
    for key in ("investor_score", "trader_score", "quadrant", "fair_value_rule_based"):
        p.pop(key, None)
    p["archetype"] = {"name": p["archetype"]["name"], "confidence": p["archetype"]["confidence"]}
    p["price_blind"] = True
    return p
