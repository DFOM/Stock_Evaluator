"""Classifies a company into a business-shape archetype.

Why this exists: a single valuation method applied to every company is wrong
more often than it's right. A bank's cash flow statement is not comparable to
an industrial's; a pre-profit SaaS company can't be DCF'd sensibly. So we
score each archetype with independent heuristic signals (sector/industry
keywords + ratio evidence) and pick the highest, with a confidence measure
based on how clear the win was.

This is a rule-based heuristic, not a trained classifier — it's meant to be
obviously-correct on clear-cut cases (a bank, a REIT, a mature dividend payer,
a pre-profit high-growth name) and transparent (every point is explained)
rather than clever on ambiguous ones.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models import ArchetypeResult, CompanyData, RatioSet
from src.ratio_engine import compute_cash_runway

# A cash-burning company counts as a funded pre-profit grower (not a
# turnaround) when it's still growing and holds at least this many years of
# cash at the current burn rate.
MIN_RUNWAY_YEARS = 2.0

FINANCIAL_INDUSTRY_KEYWORDS = (
    "bank", "insurance", "capital markets", "asset management", "credit services",
    "financial conglomerates", "financial data", "mortgage", "reinsurance",
    "insurance brokers", "shell companies",
)
REIT_INDUSTRY_KEYWORDS = ("reit",)
ASSET_HEAVY_SECTOR_KEYWORDS = ("energy", "basic materials", "utilities")
ASSET_HEAVY_INDUSTRY_KEYWORDS = ("oil & gas", "mining", "metals", "coal", "steel", "gold", "copper")


@dataclass
class _Vote:
    score: float = 0.0
    reasons: list[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []

    def add(self, points: float, reason: str) -> None:
        self.score += points
        self.reasons.append(f"+{points:g}: {reason}")


def _contains_any(text: str | None, keywords: tuple[str, ...]) -> bool:
    if not text:
        return False
    text = text.lower()
    return any(kw in text for kw in keywords)


def classify(data: CompanyData, ratios: RatioSet) -> ArchetypeResult:
    sector = data.sector or ""
    industry = data.industry or ""

    votes: dict[str, _Vote] = {
        "financial": _Vote(),
        "asset_play": _Vote(),
        "growth": _Vote(),
        "turnaround": _Vote(),
        "cash_cow": _Vote(),
    }

    # ---------------- financial ----------------
    fin = votes["financial"]
    if sector.lower() == "financial services":
        fin.add(5, f"sector is '{sector}'")
    if _contains_any(industry, FINANCIAL_INDUSTRY_KEYWORDS):
        fin.add(3, f"industry '{industry}' matches financial-institution keywords")
    if ratios.gross_margin is None and ratios.roe is not None and ratios.price_to_book is not None:
        fin.add(1, "no meaningful gross margin but ROE/P-B are present (bank-shaped income statement)")

    # ---------------- asset_play ----------------
    ap = votes["asset_play"]
    if sector.lower() == "real estate":
        ap.add(5, f"sector is '{sector}'")
    if _contains_any(industry, REIT_INDUSTRY_KEYWORDS):
        ap.add(4, f"industry '{industry}' is a REIT")
    if _contains_any(sector, ASSET_HEAVY_SECTOR_KEYWORDS) or _contains_any(industry, ASSET_HEAVY_INDUSTRY_KEYWORDS):
        ap.add(2, f"sector/industry ('{sector}'/'{industry}') is asset-heavy (energy, mining, materials, utilities)")
    # Income-shape signals (high payout, high yield) only CORROBORATE an
    # asset play whose sector already says so — on their own they must not
    # classify. A snack company whose payout ratio spiked because earnings
    # collapsed is not an income/NAV vehicle (JJSF), it's a company in
    # trouble; the earnings-collapse case is handled as a turnaround signal
    # below instead.
    if ap.score > 0:
        if ratios.payout_ratio is not None and ratios.payout_ratio > 0.75:
            ap.add(1, f"high payout ratio ({ratios.payout_ratio:.0%}) typical of income/NAV vehicles")
        if ratios.dividend_yield is not None and ratios.dividend_yield > 0.04:
            ap.add(1, f"high dividend yield ({ratios.dividend_yield:.1%})")

    # Pre-profit vs distressed disambiguation: burning cash while growing
    # fast with years of runway is the normal shape of a young growth
    # company, not distress. Turnaround signals below are gated on this.
    rg = ratios.revenue_growth_yoy
    runway = compute_cash_runway(data, ratios)
    burning_cash = ratios.free_cash_flow is not None and ratios.free_cash_flow < 0
    still_growing = (rg is not None and rg > 0.10) or (
        ratios.revenue_cagr_3y is not None and ratios.revenue_cagr_3y > 0.20
    )
    pre_profit_growth = (
        burning_cash and still_growing and runway is not None and runway >= MIN_RUNWAY_YEARS
    )

    # ---------------- growth ----------------
    gr = votes["growth"]
    if rg is not None:
        if rg > 0.40:
            gr.add(5, f"revenue growth {rg:.1%} > 40% (hyper-growth)")
        elif rg > 0.20:
            gr.add(3, f"revenue growth {rg:.1%} > 20%")
        elif rg > 0.15:
            gr.add(2, f"revenue growth {rg:.1%} > 15%")
        elif rg > 0.10:
            gr.add(1, f"revenue growth {rg:.1%} > 10%")
    if ratios.net_margin is not None and ratios.net_margin < 0:
        gr.add(2, f"negative net margin ({ratios.net_margin:.1%}) — reinvesting/pre-profit rather than distressed given growth")
    if ratios.price_to_sales is not None and ratios.price_to_sales > 15:
        gr.add(2, f"extremely rich revenue multiple (P/S {ratios.price_to_sales:.1f}x) — priced on future growth, re-rating risk")
    elif ratios.price_to_sales is not None and ratios.price_to_sales > 8 and (ratios.net_margin is None or ratios.net_margin < 0.10):
        gr.add(2, f"high P/S ({ratios.price_to_sales:.1f}x) priced on revenue, not earnings")
    if not ratios.dividend_yield:
        gr.add(1, "pays no dividend (reinvesting for growth)")
    if sector.lower() == "technology" and rg is not None and rg > 0.10:
        gr.add(1, f"technology sector with double-digit growth")
    if pre_profit_growth:
        gr.add(3, f"pre-profit but well funded: ~{runway:.1f} years of cash runway at current burn")
    if ratios.revenue_cagr_3y is not None and ratios.revenue_cagr_3y > 0.10 and rg is not None and rg > 0.05:
        gr.add(2, f"sustained multi-year compounding (3y revenue CAGR {ratios.revenue_cagr_3y:.1%}, still growing)")

    # ---------------- turnaround ----------------
    tu = votes["turnaround"]
    # Negative equity alone isn't distress — mature companies buy back stock
    # past book equity while comfortably servicing debt (DPZ, MCD). Only
    # treat it as a distress vote when debt service or cash generation is
    # actually strained.
    healthy_coverage = ratios.interest_coverage is not None and ratios.interest_coverage >= 4.0
    positive_fcf = ratios.free_cash_flow is not None and ratios.free_cash_flow > 0
    if ratios.debt_to_equity is not None and ratios.debt_to_equity < 0 and not (healthy_coverage and positive_fcf):
        tu.add(4, "negative shareholder equity (balance sheet distress)")
    if rg is not None and rg < -0.05:
        tu.add(2, f"revenue declining ({rg:.1%} YoY)")
    if ratios.revenue_cagr_3y is not None and ratios.revenue_cagr_3y < -0.03 and (rg is None or rg < 0.05):
        tu.add(2, f"multi-year revenue deterioration (3y CAGR {ratios.revenue_cagr_3y:.1%})")
    if not pre_profit_growth:
        if ratios.net_margin is not None and ratios.net_margin < 0 and burning_cash:
            tu.add(3, "negative net margin AND negative free cash flow simultaneously")
        if ratios.interest_coverage is not None and ratios.interest_coverage < 1.5:
            tu.add(2, f"interest coverage {ratios.interest_coverage:.1f}x is dangerously thin")
    if burning_cash and runway is not None and runway < MIN_RUNWAY_YEARS:
        tu.add(3, f"only ~{runway:.1f} years of cash runway at current burn")
    if ratios.current_ratio is not None and ratios.current_ratio < 1 and ratios.net_debt_to_ebitda is not None and ratios.net_debt_to_ebitda > 5:
        tu.add(2, "current ratio < 1 combined with net debt > 5x EBITDA")
    if (
        ratios.payout_ratio is not None and ratios.payout_ratio > 1.0
        and ratios.eps_growth_yoy is not None and ratios.eps_growth_yoy < -0.30
    ):
        tu.add(1, f"dividend exceeds earnings (payout {ratios.payout_ratio:.0%}) while EPS is collapsing ({ratios.eps_growth_yoy:.0%}) — sustainability risk")

    # ---------------- cash_cow ----------------
    cc = votes["cash_cow"]
    if rg is not None and 0 <= rg <= 0.15:
        cc.add(2, f"stable, moderate revenue growth ({rg:.1%})")
    if ratios.net_margin is not None and ratios.net_margin > 0.10:
        cc.add(2, f"consistently profitable (net margin {ratios.net_margin:.1%})")
    if ratios.free_cash_flow is not None and ratios.free_cash_flow > 0 and ratios.fcf_margin is not None and ratios.fcf_margin > 0.05:
        cc.add(2, f"solid positive free cash flow (margin {ratios.fcf_margin:.1%})")
    if ratios.dividend_yield:
        cc.add(1, "pays a dividend")
    if ratios.market_cap is not None and ratios.market_cap > 10e9:
        cc.add(1, "large, established market cap")

    # Financial and asset_play are near-disqualifying for cash_cow/growth/turnaround
    # framing when their sector signal is strong, since valuation approach is what
    # matters here, not just profitability shape.
    if fin.score >= 5:
        for name in ("growth", "turnaround", "cash_cow"):
            votes[name].score *= 0.4
    if ap.score >= 5:
        for name in ("growth", "turnaround", "cash_cow"):
            votes[name].score *= 0.5

    # Hyper-growth crowds out the cash_cow framing even if currently
    # profitable: a rich revenue multiple plus fast growth means the story is
    # about future growth durability, not steady-state cash generation, so
    # revenue-multiple/Rule-of-40 valuation is more relevant than a steady DCF.
    if rg is not None and rg > 0.30:
        cc.score *= 0.5

    ranked = sorted(votes.items(), key=lambda kv: kv[1].score, reverse=True)
    winner, winner_vote = ranked[0]
    runner_up_score = ranked[1][1].score if len(ranked) > 1 else 0.0

    if winner_vote.score <= 0:
        # No signal fired at all (heavily missing data) — default to cash_cow
        # as the least valuation-method-specific fallback, but flag low confidence.
        winner = "cash_cow"
        winner_vote = votes["cash_cow"]
        winner_vote.reasons.append("no classification signal fired — defaulting to cash_cow")
        confidence = 0.25
    else:
        margin = (winner_vote.score - runner_up_score) / winner_vote.score
        confidence = max(0.3, min(0.95, 0.5 + margin * 0.5))
        confidence *= 0.7 + 0.3 * ratios.completeness

    scores = {name: round(v.score, 2) for name, v in votes.items()}

    return ArchetypeResult(
        archetype=winner,
        confidence=round(confidence, 2),
        reasons=winner_vote.reasons,
        scores=scores,
    )
