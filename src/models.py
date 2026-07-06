"""Shared data contracts passed between pipeline stages.

Keeping these in one module means every stage (fetch -> ratios -> archetype ->
scoring -> fair value -> briefing -> report) speaks the same language and can
be unit-tested independently of yfinance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class CompanyData:
    """Raw-ish data pulled from the data layer for one ticker."""

    ticker: str
    long_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None

    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    shares_outstanding: Optional[float] = None
    beta: Optional[float] = None

    info: dict[str, Any] = field(default_factory=dict)
    income_stmt: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    warnings: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass
class RatioSet:
    """All computed ratios. Every field is Optional — missing data is normal,
    not an error, and downstream stages must treat None as 'unknown' rather
    than 0."""

    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None

    # Growth
    revenue_growth_yoy: Optional[float] = None
    revenue_cagr_3y: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    fcf_growth_yoy: Optional[float] = None

    # Leverage / liquidity
    debt_to_equity: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None

    # Cash flow
    free_cash_flow: Optional[float] = None
    fcf_margin: Optional[float] = None
    fcf_yield: Optional[float] = None
    capex_to_revenue: Optional[float] = None

    # Valuation multiples
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    peg_ratio: Optional[float] = None

    # Dividend
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

    # Size / risk
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    beta: Optional[float] = None

    # Provenance: which fields had to fall back to statement math, or
    # couldn't be computed at all (used to degrade confidence downstream).
    missing_fields: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "missing_fields"}
        return d

    @property
    def completeness(self) -> float:
        """Fraction of ratio fields that were successfully computed."""
        vals = self.as_dict()
        if not vals:
            return 0.0
        known = sum(1 for v in vals.values() if v is not None)
        return known / len(vals)


ARCHETYPES = ("cash_cow", "growth", "financial", "asset_play", "turnaround")


@dataclass
class ArchetypeResult:
    archetype: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RedFlag:
    name: str
    description: str
    score_cap: float  # maximum /10 score this flag permits, regardless of pillars


@dataclass
class PillarScore:
    name: str
    score: Optional[float]  # 0-10, None if it couldn't be assessed at all
    confidence: float  # 0-1
    notes: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    pillar_scores: dict[str, PillarScore]
    investor_score: Optional[float]
    trader_score: Optional[float]
    quadrant: str
    quadrant_reason: str
    red_flags: list[RedFlag] = field(default_factory=list)
    # Findings worth surfacing that do NOT cap the score (e.g. a red flag
    # that was investigated and downgraded, like buyback-driven negative equity).
    info_notes: list[str] = field(default_factory=list)


@dataclass
class FairValueResult:
    method: str
    low: Optional[float]
    mid: Optional[float]
    high: Optional[float]
    currency: Optional[str]
    confidence: float
    margin_of_safety_price: Optional[float]
    assumptions: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
