"""Rule-based fair value estimation, one method per archetype.

This is intentionally simple math (Phase 1, no AI analysts yet) — the point
is a transparent, sanity-checkable range with explicit assumptions, not a
precise number. Every result is a low/mid/high range with a confidence score
and the assumptions used, never a single "magic number." A margin of safety
(config/weights.yaml, per archetype) is applied to the midpoint to produce a
suggested buy-below target.
"""
from __future__ import annotations

from typing import Any, Optional

from src.models import ArchetypeResult, CompanyData, FairValueResult, RatioSet


def _cost_of_equity(beta: Optional[float], config: dict[str, Any]) -> float:
    va = config["valuation_assumptions"]
    beta = beta if beta is not None else 1.0
    coe = va["risk_free_rate"] + beta * va["equity_risk_premium"]
    return max(0.06, min(0.16, coe))


def _shares_outstanding(data: CompanyData) -> Optional[float]:
    if data.shares_outstanding:
        return data.shares_outstanding
    if data.market_cap and data.current_price:
        return data.market_cap / data.current_price
    return None


def _net_debt(data: CompanyData) -> Optional[float]:
    info = data.info or {}
    total_debt = info.get("totalDebt")
    total_cash = info.get("totalCash")
    if total_debt is None or total_cash is None:
        return None
    return total_debt - total_cash


def _normalized_fcf(data: CompanyData, ratios: RatioSet) -> tuple[Optional[float], str]:
    """Averages up to 3 years of annual free cash flow instead of using the
    raw TTM figure. yfinance's TTM `freeCashflow` can swing hard on one noisy
    quarter (working-capital timing, one-off investing items); a single lumpy
    year shouldn't dominate a 10-year DCF projection."""
    cf = data.cashflow
    if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
        raw = cf.loc["Free Cash Flow"].iloc[:3]
        vals = [float(v) for v in raw if v == v]  # drop NaN
        if vals:
            return sum(vals) / len(vals), f"average of last {len(vals)} annual free cash flow figures"
    if ratios.free_cash_flow:
        return ratios.free_cash_flow, "TTM free cash flow (annual statement history unavailable)"
    return None, ""


def _book_value_per_share(data: CompanyData, ratios: RatioSet) -> Optional[float]:
    bv = (data.info or {}).get("bookValue")
    if bv:
        return bv
    if data.current_price and ratios.price_to_book:
        return data.current_price / ratios.price_to_book
    return None


def _project_fcf_pv(fcf0: float, growth: float, discount: float, terminal_growth: float, years: int) -> float:
    discount = max(discount, terminal_growth + 0.02)  # avoid blow-up if rates converge
    pv = 0.0
    fcf = fcf0
    for yr in range(1, years + 1):
        fcf *= (1 + growth)
        pv += fcf / (1 + discount) ** yr
    terminal_value = fcf * (1 + terminal_growth) / (discount - terminal_growth)
    pv += terminal_value / (1 + discount) ** years
    return pv


def _dcf_cash_cow(data: CompanyData, ratios: RatioSet, config: dict[str, Any]) -> FairValueResult:
    va = config["valuation_assumptions"]
    shares = _shares_outstanding(data)
    fcf, fcf_note = _normalized_fcf(data, ratios)

    if not fcf or fcf <= 0 or not shares:
        return _pe_fallback(data, ratios, "cash_cow DCF needs positive free cash flow and share count")

    base_growth = ratios.revenue_cagr_3y if ratios.revenue_cagr_3y is not None else ratios.revenue_growth_yoy
    base_growth = base_growth if base_growth is not None else 0.03
    base_growth = max(-0.02, min(base_growth, va["fcf_growth_cap"]))

    discount = _cost_of_equity(ratios.beta, config)
    years = va["dcf_projection_years"]
    tg = va["terminal_growth_rate"]

    mid_pv = _project_fcf_pv(fcf, base_growth, discount, tg, years)
    low_pv = _project_fcf_pv(fcf, max(base_growth - 0.03, -0.05), discount + 0.01, tg, years)
    high_pv = _project_fcf_pv(fcf, min(base_growth + 0.03, va["fcf_growth_cap"] + 0.03), max(discount - 0.01, tg + 0.02), tg, years)

    net_debt = _net_debt(data) or 0.0
    low = (low_pv - net_debt) / shares
    mid = (mid_pv - net_debt) / shares
    high = (high_pv - net_debt) / shares
    low, high = min(low, high), max(low, high)

    return FairValueResult(
        method="DCF on free cash flow (2-stage: explicit growth + Gordon terminal value)",
        low=round(low, 2), mid=round(mid, 2), high=round(high, 2),
        currency=data.currency,
        confidence=0.65 if ratios.revenue_cagr_3y is not None else 0.5,
        margin_of_safety_price=None,
        assumptions={
            "base_fcf": round(fcf, 0), "growth_rate": round(base_growth, 4),
            "discount_rate": round(discount, 4), "terminal_growth": tg,
            "projection_years": years, "net_debt": round(net_debt, 0), "shares_outstanding": round(shares, 0),
        },
        notes=[
            "Treats reported free cash flow as equity-owner cash flow (a common simplification for a quick DCF).",
            f"Base FCF is {fcf_note}, not a single trailing-twelve-month snapshot.",
        ],
    )


def _pe_fallback(data: CompanyData, ratios: RatioSet, reason: str) -> FairValueResult:
    eps = (data.info or {}).get("trailingEps")
    if not eps or eps <= 0:
        return FairValueResult(
            method="none (insufficient data)", low=None, mid=None, high=None,
            currency=data.currency, confidence=0.0, margin_of_safety_price=None,
            notes=[reason, "No positive EPS available either — cannot estimate fair value."],
        )
    return FairValueResult(
        method="fallback: trailing P/E band (15x-20x)",
        low=round(eps * 15, 2), mid=round(eps * 17.5, 2), high=round(eps * 20, 2),
        currency=data.currency, confidence=0.3, margin_of_safety_price=None,
        assumptions={"trailing_eps": eps},
        notes=[reason, "Fell back to a generic fair-multiple band on trailing EPS — low confidence."],
    )


def _multiples_growth(data: CompanyData, ratios: RatioSet, config: dict[str, Any]) -> FairValueResult:
    revenue = (data.info or {}).get("totalRevenue")
    shares = _shares_outstanding(data)
    if not revenue or not shares:
        return FairValueResult(
            method="none (insufficient data)", low=None, mid=None, high=None,
            currency=data.currency, confidence=0.0, margin_of_safety_price=None,
            notes=["Growth multiples valuation needs revenue and share count, neither fully available."],
        )

    growth = ratios.revenue_growth_yoy or 0.0
    fcf_margin = ratios.fcf_margin if ratios.fcf_margin is not None else (ratios.operating_margin or 0.0)
    rule_of_40 = (growth * 100) + (fcf_margin * 100)

    # Heuristic multiple ladder: no peer database yet in Phase 1 (that arrives
    # with the briefing packet feeding the AI analyst ensemble) so the
    # benchmark EV/Revenue multiple is derived from the Rule of 40 score
    # rather than actual comparables. Documented, not empirically fit.
    base_multiple = 4.0 + (rule_of_40 - 40) * 0.15
    base_multiple = max(1.0, min(base_multiple, 25.0))

    net_debt = _net_debt(data) or 0.0
    mid_ev = revenue * base_multiple
    low_ev = revenue * base_multiple * 0.7
    high_ev = revenue * base_multiple * 1.3

    mid = (mid_ev - net_debt) / shares
    low = (low_ev - net_debt) / shares
    high = (high_ev - net_debt) / shares

    return FairValueResult(
        method="Revenue-multiple valuation (Rule-of-40-scaled EV/Revenue benchmark)",
        low=round(low, 2), mid=round(mid, 2), high=round(high, 2),
        currency=data.currency, confidence=0.4,
        margin_of_safety_price=None,
        assumptions={
            "revenue": round(revenue, 0), "revenue_growth_yoy": round(growth, 4),
            "fcf_or_op_margin": round(fcf_margin, 4), "rule_of_40_score": round(rule_of_40, 1),
            "benchmark_ev_to_revenue": round(base_multiple, 2), "net_debt": round(net_debt, 0),
        },
        notes=[
            f"Rule of 40 = {rule_of_40:.0f} ({'passes' if rule_of_40 >= 40 else 'fails'} the growth+profitability bar).",
            "No peer/comparables data in Phase 1 — benchmark multiple is a documented heuristic, not a market comp. "
            "Real peer multiples arrive once the AI analyst ensemble (Phase 2) is wired up.",
        ],
    )


def _pb_financial(data: CompanyData, ratios: RatioSet, config: dict[str, Any]) -> FairValueResult:
    bvps = _book_value_per_share(data, ratios)
    roe = ratios.roe
    if not bvps or roe is None:
        return FairValueResult(
            method="none (insufficient data)", low=None, mid=None, high=None,
            currency=data.currency, confidence=0.0, margin_of_safety_price=None,
            notes=["Justified P/B needs book value per share and ROE, not both available."],
        )

    coe = _cost_of_equity(ratios.beta, config)
    payout = ratios.payout_ratio if ratios.payout_ratio is not None else 0.3
    # Gordon-growth-style P/B is unstable as g approaches the cost of equity
    # (the denominator shrinks toward zero and the multiple blows up), so g
    # is capped a full 3 points below cost of equity rather than the more
    # typical 1-2 point buffer used for stage-one DCF growth.
    g = max(0.0, min(roe * (1 - payout), coe - 0.03))

    def justified_pb(_coe: float) -> float:
        return max(0.3, min((roe - g) / (_coe - g), 5.0))

    mid_pb = justified_pb(coe)
    low_pb = justified_pb(coe + 0.015)
    # Floor here (g + 0.015) is a plain minimum-spread guard, independent of
    # the g cap above — using the same constant as that cap would make this
    # collapse to the mid case whenever g is pinned at its ceiling.
    high_pb = justified_pb(max(coe - 0.015, g + 0.015))

    return FairValueResult(
        method="Justified P/B = (ROE - g) / (cost of equity - g)",
        low=round(bvps * low_pb, 2), mid=round(bvps * mid_pb, 2), high=round(bvps * high_pb, 2),
        currency=data.currency, confidence=0.55,
        margin_of_safety_price=None,
        assumptions={
            "book_value_per_share": round(bvps, 2), "roe": round(roe, 4),
            "sustainable_growth_g": round(g, 4), "cost_of_equity": round(coe, 4),
            "justified_pb_mid": round(mid_pb, 2),
        },
        notes=["Cash-flow-based valuation is not meaningful for a financial institution — P/B vs ROE is the standard lens."],
    )


def _nav_asset_play(data: CompanyData, ratios: RatioSet, config: dict[str, Any]) -> FairValueResult:
    navps = _book_value_per_share(data, ratios)
    if not navps:
        return FairValueResult(
            method="none (insufficient data)", low=None, mid=None, high=None,
            currency=data.currency, confidence=0.0, margin_of_safety_price=None,
            notes=["NAV proxy needs a book-value-per-share figure, not available."],
        )

    return FairValueResult(
        method="NAV proxy (book value per share x P/NAV band)",
        low=round(navps * 0.85, 2), mid=round(navps * 1.0, 2), high=round(navps * 1.15, 2),
        currency=data.currency, confidence=0.45,
        margin_of_safety_price=None,
        assumptions={"book_value_per_share_as_nav_proxy": round(navps, 2), "pnav_band": "0.85x - 1.15x"},
        notes=[
            "Uses reported book value as a NAV proxy — no independent property/asset appraisal data available "
            "(REITs/miners/energy often trade at a real premium or discount to book NAV that this can't see).",
        ],
    )


def _survival_turnaround(data: CompanyData, ratios: RatioSet, config: dict[str, Any]) -> FairValueResult:
    tbv = (data.info or {}).get("bookValue")
    shares = _shares_outstanding(data)
    if tbv is None or ratios.debt_to_equity is None:
        return FairValueResult(
            method="none (insufficient data)", low=None, mid=None, high=None,
            currency=data.currency, confidence=0.0, margin_of_safety_price=None,
            notes=["Survival analysis needs book value and leverage data, not fully available."],
        )

    if ratios.debt_to_equity < 0 or tbv <= 0:
        return FairValueResult(
            method="balance-sheet survival analysis",
            low=0.0, mid=0.0, high=round(max(tbv, 0) * 0.3, 2),
            currency=data.currency, confidence=0.3, margin_of_safety_price=None,
            notes=["Negative or near-zero book equity — equity value is highly uncertain and could be near zero in a restructuring."],
        )

    return FairValueResult(
        method="balance-sheet survival analysis (haircut on book value)",
        low=round(tbv * 0.3, 2), mid=round(tbv * 0.55, 2), high=round(tbv * 0.85, 2),
        currency=data.currency, confidence=0.3, margin_of_safety_price=None,
        assumptions={"book_value_per_share": round(tbv, 2), "haircut_band": "30%-85% of book, reflecting distress"},
        notes=[
            "Distressed situations are survival-dependent, not cash-flow-valuable in the normal sense — this range is "
            "wide on purpose. Treat it as a sanity floor/ceiling, not a target price.",
        ],
    )


_DISPATCH = {
    "cash_cow": _dcf_cash_cow,
    "growth": _multiples_growth,
    "financial": _pb_financial,
    "asset_play": _nav_asset_play,
    "turnaround": _survival_turnaround,
}


def estimate_fair_value(data: CompanyData, ratios: RatioSet, archetype: ArchetypeResult, config: dict[str, Any]) -> FairValueResult:
    fn = _DISPATCH[archetype.archetype]
    result = fn(data, ratios, config)
    if result.mid is not None:
        mos = config["archetypes"][archetype.archetype]["margin_of_safety"]
        result.margin_of_safety_price = round(result.mid * (1 - mos), 2)
        result.assumptions["margin_of_safety"] = mos
    return result
