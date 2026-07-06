"""Pillar scoring, red-flag veto layer, investor/trader scores, and the
quality-vs-price quadrant verdict.

Design notes:
  - Every pillar is scored 0-10 from a weighted blend of ratios. A ratio that
    is None (missing, or structurally not applicable to this archetype, e.g.
    current_ratio for a bank) is simply left out of the blend — it lowers
    that pillar's *confidence* but does not drag the score toward 0.
  - Red flags CAP the final score rather than subtract from it: a company
    can't average its way out of a going-concern-shaped risk with strong
    scores elsewhere.
  - Investor vs trader scores are the same six pillars combined with
    different weight profiles (config/weights.yaml), per archetype.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.models import ArchetypeResult, CompanyData, PillarScore, RatioSet, RedFlag, ScoreResult

# Approximate sovereign risk scores (0-10, higher = safer), used as a stand-in
# for a full Damodaran country-risk-premium table. Deliberately coarse for
# Phase 1 — country field on the info dict is what's available from yfinance.
COUNTRY_RISK_SCORE: dict[str, float] = {
    "United States": 9.5, "Switzerland": 9.5, "Singapore": 9.3, "Australia": 9.2,
    "Denmark": 9.2, "Canada": 9.2, "Netherlands": 9.0, "Sweden": 9.0, "Germany": 9.0,
    "United Kingdom": 8.9, "France": 8.7, "Ireland": 8.7, "Japan": 8.7, "Taiwan": 7.2,
    "South Korea": 8.0, "Israel": 6.8, "China": 6.2, "India": 6.4, "Indonesia": 5.8,
    "Mexico": 5.6, "Brazil": 5.0, "South Africa": 4.5, "Turkey": 3.5, "Nigeria": 3.0,
    "Argentina": 2.5, "Russia": 1.5,
}
DEFAULT_COUNTRY_RISK = 6.0


class _PillarBuilder:
    """Accumulates (score, weight) pairs for one pillar and produces a
    weighted-average PillarScore. Confidence = achieved weight / attempted
    weight, so a pillar built entirely from unavailable metrics naturally
    ends up with score=None, confidence=0 instead of a misleading score."""

    def __init__(self, name: str):
        self.name = name
        self.notes: list[str] = []
        self._num = 0.0
        self._den = 0.0
        self._possible = 0.0

    def add(self, score: Optional[float], weight: float, note: str) -> None:
        self._possible += weight
        if score is not None:
            self._num += score * weight
            self._den += weight
            self.notes.append(note)
        else:
            self.notes.append(f"{note} — unavailable, excluded")

    def build(self) -> PillarScore:
        if self._den == 0:
            return PillarScore(self.name, None, 0.0, self.notes)
        score = round(self._num / self._den, 2)
        confidence = round(self._den / self._possible, 2) if self._possible else 0.0
        return PillarScore(self.name, score, confidence, self.notes)


def higher_is_better(value: Optional[float], floor: float, ceiling: float) -> Optional[float]:
    if value is None:
        return None
    t = (value - floor) / (ceiling - floor)
    return round(max(0.0, min(1.0, t)) * 10, 2)


def lower_is_better(value: Optional[float], floor: float, ceiling: float) -> Optional[float]:
    if value is None:
        return None
    t = (value - floor) / (ceiling - floor)
    t = max(0.0, min(1.0, t))
    return round((1 - t) * 10, 2)


# --------------------------------------------------------------------------
# Pillars
# --------------------------------------------------------------------------

def _quality_pillar(ratios: RatioSet) -> PillarScore:
    b = _PillarBuilder("quality")
    b.add(higher_is_better(ratios.net_margin, 0.0, 0.20), 3, f"net margin {_pct(ratios.net_margin)}")
    b.add(higher_is_better(ratios.roe, 0.0, 0.25), 3, f"ROE {_pct(ratios.roe)}")
    b.add(higher_is_better(ratios.roic, 0.0, 0.15), 2, f"ROIC {_pct(ratios.roic)}")
    b.add(higher_is_better(ratios.gross_margin, 0.20, 0.60), 1, f"gross margin {_pct(ratios.gross_margin)}")
    b.add(higher_is_better(ratios.operating_margin, 0.0, 0.25), 2, f"operating margin {_pct(ratios.operating_margin)}")
    b.add(higher_is_better(ratios.roa, 0.0, 0.10), 1, f"ROA {_pct(ratios.roa)}")
    return b.build()


def _growth_pillar(ratios: RatioSet, archetype: str) -> PillarScore:
    b = _PillarBuilder("growth")
    b.add(higher_is_better(ratios.revenue_growth_yoy, -0.05, 0.25), 3, f"revenue growth YoY {_pct(ratios.revenue_growth_yoy)}")
    b.add(higher_is_better(ratios.revenue_cagr_3y, -0.05, 0.20), 2, f"3yr revenue CAGR {_pct(ratios.revenue_cagr_3y)}")
    b.add(higher_is_better(ratios.eps_growth_yoy, -0.10, 0.30), 2, f"EPS growth YoY {_pct(ratios.eps_growth_yoy)}")
    if archetype not in ("financial", "asset_play"):
        # FCF growth is noisy/not comparable for banks (deposit/loan flows
        # dominate) and REITs (FFO, not FCF, is the relevant cash metric).
        b.add(higher_is_better(ratios.fcf_growth_yoy, -0.20, 0.20), 2, f"FCF growth YoY {_pct(ratios.fcf_growth_yoy)}")
    return b.build()


_FH_BANDS = {
    # archetype: (d2e_floor, d2e_ceiling, current_ratio_floor, current_ratio_ceiling,
    #             interest_cov_floor, interest_cov_ceiling, ndebt_ebitda_floor, ndebt_ebitda_ceiling)
    "cash_cow": (0.3, 2.5, 0.8, 2.0, 1.5, 8.0, 0.5, 5.0),
    "growth": (0.1, 2.0, 0.8, 2.5, 2.0, 10.0, 0.0, 4.0),
    "asset_play": (0.5, 3.5, 0.6, 1.8, 1.5, 6.0, 1.0, 8.0),
    "turnaround": (0.3, 3.0, 0.6, 1.5, 1.0, 5.0, 1.0, 8.0),
}


def _financial_health_pillar(ratios: RatioSet, archetype: str) -> PillarScore:
    b = _PillarBuilder("financial_health")
    if archetype == "financial":
        # Banks/insurers run on regulatory capital ratios we don't have from
        # yfinance; standard corporate leverage/liquidity ratios don't apply
        # (leverage is the business model, and there's no current/non-current
        # split on a bank balance sheet). Lean on leverage + ROA as a coarse
        # solvency/profitability proxy instead of penalizing structural leverage.
        b.add(lower_is_better(ratios.debt_to_equity, 3.0, 14.0), 4, f"leverage (bank-scaled) D/E={_ratio(ratios.debt_to_equity)}")
        b.add(higher_is_better(ratios.roa, 0.004, 0.018), 3, f"ROA {_pct(ratios.roa)} (bank solvency/profitability proxy)")
        return b.build()

    floor_d2e, ceil_d2e, floor_cr, ceil_cr, floor_ic, ceil_ic, floor_nde, ceil_nde = _FH_BANDS.get(
        archetype, _FH_BANDS["cash_cow"]
    )
    # Negative D/E means negative book equity, not "less debt than zero" —
    # scoring it on the lower-is-better band would award a perfect 10.
    # Exclude it and let coverage/net-debt metrics carry the pillar.
    d2e = ratios.debt_to_equity if (ratios.debt_to_equity is None or ratios.debt_to_equity >= 0) else None
    d2e_note = f"debt/equity {_ratio(ratios.debt_to_equity)}" if d2e is not None else "debt/equity not meaningful (negative book equity)"
    b.add(lower_is_better(d2e, floor_d2e, ceil_d2e), 3, d2e_note)
    b.add(higher_is_better(ratios.current_ratio, floor_cr, ceil_cr), 2, f"current ratio {_ratio(ratios.current_ratio)}")
    b.add(higher_is_better(ratios.interest_coverage, floor_ic, ceil_ic), 3, f"interest coverage {_ratio(ratios.interest_coverage)}x")
    b.add(lower_is_better(ratios.net_debt_to_ebitda, floor_nde, ceil_nde), 2, f"net debt/EBITDA {_ratio(ratios.net_debt_to_ebitda)}x")
    return b.build()


def _valuation_pillar(ratios: RatioSet, archetype: str) -> PillarScore:
    b = _PillarBuilder("valuation")
    if archetype == "cash_cow":
        b.add(lower_is_better(ratios.pe_trailing, 10, 30), 3, f"trailing P/E {_ratio(ratios.pe_trailing)}")
        b.add(lower_is_better(ratios.ev_to_ebitda, 8, 20), 3, f"EV/EBITDA {_ratio(ratios.ev_to_ebitda)}")
        b.add(higher_is_better(ratios.fcf_yield, 0.02, 0.08), 3, f"FCF yield {_pct(ratios.fcf_yield)}")
        b.add(lower_is_better(ratios.peg_ratio, 1.0, 3.0), 1, f"PEG {_ratio(ratios.peg_ratio)}")
    elif archetype == "growth":
        b.add(lower_is_better(ratios.ev_to_revenue, 2, 20), 4, f"EV/Revenue {_ratio(ratios.ev_to_revenue)}")
        b.add(lower_is_better(ratios.price_to_sales, 3, 25), 3, f"P/S {_ratio(ratios.price_to_sales)}")
        b.add(lower_is_better(ratios.peg_ratio, 1.0, 3.5), 3, f"PEG {_ratio(ratios.peg_ratio)}")
    elif archetype == "financial":
        b.add(lower_is_better(ratios.price_to_book, 0.8, 3.0), 6, f"P/B {_ratio(ratios.price_to_book)}")
        b.add(lower_is_better(ratios.pe_trailing, 8, 20), 4, f"trailing P/E {_ratio(ratios.pe_trailing)}")
    elif archetype == "asset_play":
        b.add(lower_is_better(ratios.price_to_book, 0.7, 2.0), 5, f"P/B (NAV proxy) {_ratio(ratios.price_to_book)}")
        b.add(higher_is_better(ratios.dividend_yield, 0.02, 0.07), 5, f"dividend yield {_pct(ratios.dividend_yield)}")
    elif archetype == "turnaround":
        b.add(lower_is_better(ratios.ev_to_revenue, 0.3, 3.0), 5, f"EV/Revenue (distress pricing) {_ratio(ratios.ev_to_revenue)}")
        b.add(lower_is_better(ratios.price_to_book, 0.3, 1.5), 5, f"P/B (distress pricing) {_ratio(ratios.price_to_book)}")
    return b.build()


def _macro_pillar(data: CompanyData) -> PillarScore:
    b = _PillarBuilder("macro")
    country = data.country
    score = COUNTRY_RISK_SCORE.get(country, DEFAULT_COUNTRY_RISK) if country else None
    note = f"country risk proxy for '{country}'" if country in COUNTRY_RISK_SCORE else f"country '{country}' not in lookup table, using neutral default"
    b.add(score, 5, note)
    # Beta as a rough macro-sensitivity signal: high beta = more exposed to
    # broad market/macro swings, penalized lightly (not a quality judgment).
    beta = data.beta
    beta_score = lower_is_better(beta, 0.7, 2.0) if beta is not None else None
    b.add(beta_score, 2, f"beta {_ratio(beta)} (market sensitivity)")
    return b.build()


def _momentum_pillar(price_history: pd.DataFrame) -> PillarScore:
    b = _PillarBuilder("momentum")
    if price_history is None or price_history.empty or "Close" not in price_history.columns:
        b.add(None, 10, "price history unavailable")
        return b.build()

    closes = price_history["Close"].dropna()
    last = float(closes.iloc[-1]) if len(closes) else None

    def _return(days: int) -> Optional[float]:
        if last is None or len(closes) <= days:
            return None
        past = float(closes.iloc[-days - 1])
        return (last - past) / past if past else None

    r6 = _return(126)
    r12 = _return(252)
    b.add(higher_is_better(r6, -0.20, 0.30), 3, f"6-month return {_pct(r6)}")
    b.add(higher_is_better(r12, -0.30, 0.50), 2, f"12-month return {_pct(r12)}")

    if last is not None and len(closes) >= 20:
        window_high = float(closes.tail(min(len(closes), 252)).max())
        pct_from_high = (last / window_high) - 1 if window_high else None
        b.add(higher_is_better(pct_from_high, -0.30, 0.0), 3, f"{_pct(pct_from_high)} from 52wk high")
    else:
        b.add(None, 3, "52wk high unavailable")

    if last is not None and len(closes) >= 200:
        ma50 = float(closes.tail(50).mean())
        ma200 = float(closes.tail(200).mean())
        if last > ma50 > ma200:
            trend_score = 10.0
        elif last > ma200:
            trend_score = 6.0
        elif last > ma50:
            trend_score = 4.0
        else:
            trend_score = 1.0
        b.add(trend_score, 2, f"trend: price {'>' if last > ma50 else '<'} 50d MA, {'>' if last > ma200 else '<'} 200d MA")
    else:
        b.add(None, 2, "insufficient history for moving averages")

    return b.build()


def _pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "n/a"


def _ratio(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


# --------------------------------------------------------------------------
# Red-flag veto layer
# --------------------------------------------------------------------------

def detect_red_flags(ratios: RatioSet, archetype: str, config: dict[str, Any]) -> tuple[list[RedFlag], list[str]]:
    """Returns (capping red flags, informational notes). A finding is only a
    veto when the underlying risk is real — e.g. negative book equity caused
    by years of buybacks at a company that comfortably services its debt is
    surfaced as a note, not a score cap."""
    rf = config["red_flags"]
    flags: list[RedFlag] = []
    info_notes: list[str] = []

    if ratios.debt_to_equity is not None and ratios.debt_to_equity < 0:
        healthy_coverage = (
            ratios.interest_coverage is not None
            and ratios.interest_coverage >= rf["negative_equity_healthy_coverage"]
        )
        positive_fcf = ratios.free_cash_flow is not None and ratios.free_cash_flow > 0
        if healthy_coverage and positive_fcf:
            info_notes.append(
                f"Negative book equity, but interest coverage is healthy ({ratios.interest_coverage:.1f}x) and "
                "free cash flow is positive — consistent with buyback-driven negative equity, not distress. No score cap applied."
            )
        else:
            flags.append(RedFlag(
                "negative_equity",
                "Negative shareholder equity without the debt-service strength to explain it away — "
                "the balance sheet is technically insolvent on a book-value basis.",
                rf["negative_equity_cap"],
            ))

    if archetype != "financial" and ratios.interest_coverage is not None and ratios.interest_coverage < rf["interest_coverage_critical"]:
        flags.append(RedFlag(
            "thin_interest_coverage",
            f"Interest coverage of {ratios.interest_coverage:.1f}x is below {rf['interest_coverage_critical']:.1f}x — "
            "operating earnings barely cover, or don't cover, interest expense.",
            rf["interest_coverage_cap"],
        ))

    if archetype != "financial" and ratios.current_ratio is not None and ratios.current_ratio < rf["current_ratio_critical"]:
        flags.append(RedFlag(
            "liquidity_risk",
            f"Current ratio of {ratios.current_ratio:.2f} is below {rf['current_ratio_critical']:.1f} — "
            "short-term liabilities may exceed liquid assets.",
            rf["current_ratio_cap"],
        ))

    if archetype != "financial" and ratios.net_debt_to_ebitda is not None and ratios.net_debt_to_ebitda > rf["net_debt_to_ebitda_critical"]:
        flags.append(RedFlag(
            "heavy_debt_load",
            f"Net debt is {ratios.net_debt_to_ebitda:.1f}x EBITDA, above the {rf['net_debt_to_ebitda_critical']:.1f}x threshold — "
            "a heavy refinancing burden.",
            rf["net_debt_to_ebitda_cap"],
        ))

    if ratios.net_margin is not None and ratios.net_margin < 0 and ratios.free_cash_flow is not None and ratios.free_cash_flow < 0:
        flags.append(RedFlag(
            "burning_cash",
            "Negative net income AND negative free cash flow simultaneously — burning cash on both an "
            "accounting and cash basis.",
            rf["negative_fcf_and_earnings_cap"],
        ))

    return flags, info_notes


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _weighted_score(pillars: dict[str, PillarScore], weights: dict[str, float]) -> Optional[float]:
    total_w = 0.0
    total = 0.0
    for name, w in weights.items():
        ps = pillars.get(name)
        if ps and ps.score is not None:
            total += ps.score * w
            total_w += w
    if total_w == 0:
        return None
    return round(total / total_w, 2)


def compute_quadrant(pillars: dict[str, PillarScore], config: dict[str, Any]) -> tuple[str, str]:
    qt = config["quadrant"]["quality_threshold"]
    vt = config["quadrant"]["valuation_threshold"]

    quality_inputs = [pillars[p].score for p in ("quality", "financial_health") if pillars[p].score is not None]
    valuation_score = pillars["valuation"].score

    if not quality_inputs or valuation_score is None:
        return "Insufficient Data", "Not enough ratio coverage to place this company in the quality-vs-price quadrant."

    quality_axis = sum(quality_inputs) / len(quality_inputs)

    if quality_axis >= qt and valuation_score >= vt:
        return (
            "Strong Buy Candidate",
            f"High quality ({quality_axis:.1f}/10) at an attractive valuation ({valuation_score:.1f}/10) — good company, good price.",
        )
    if quality_axis >= qt and valuation_score < vt:
        return (
            "Watchlist (Quality, Rich Price)",
            f"High quality ({quality_axis:.1f}/10) but valuation is unattractive ({valuation_score:.1f}/10) — good company, wait for a better price.",
        )
    if quality_axis < qt and valuation_score >= vt:
        return (
            "Value Trap Risk",
            f"Looks cheap ({valuation_score:.1f}/10) but quality/financial health is weak ({quality_axis:.1f}/10) — verify it isn't cheap for a good reason.",
        )
    return (
        "Avoid",
        f"Weak quality ({quality_axis:.1f}/10) and unattractive valuation ({valuation_score:.1f}/10).",
    )


def score_company(data: CompanyData, ratios: RatioSet, archetype: ArchetypeResult, config: dict[str, Any]) -> ScoreResult:
    arch = archetype.archetype

    pillars: dict[str, PillarScore] = {
        "quality": _quality_pillar(ratios),
        "growth": _growth_pillar(ratios, arch),
        "financial_health": _financial_health_pillar(ratios, arch),
        "valuation": _valuation_pillar(ratios, arch),
        "macro": _macro_pillar(data),
        "momentum": _momentum_pillar(data.price_history),
    }

    weights = config["archetypes"][arch]["pillar_weights"]
    investor_score = _weighted_score(pillars, weights["investor"])
    trader_score = _weighted_score(pillars, weights["trader"])

    red_flags, info_notes = detect_red_flags(ratios, arch, config)
    if red_flags:
        cap = min(f.score_cap for f in red_flags)
        if investor_score is not None:
            investor_score = min(investor_score, cap)
        if trader_score is not None:
            trader_score = min(trader_score, cap)

    quadrant, quadrant_reason = compute_quadrant(pillars, config)

    return ScoreResult(
        pillar_scores=pillars,
        investor_score=investor_score,
        trader_score=trader_score,
        quadrant=quadrant,
        quadrant_reason=quadrant_reason,
        red_flags=red_flags,
        info_notes=info_notes,
    )
