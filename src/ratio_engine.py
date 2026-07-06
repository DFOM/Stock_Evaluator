"""Computes the RatioSet from a CompanyData.

Preference order for every ratio: use yfinance's `.info` field when present
(it's usually TTM-based and pre-cleaned), fall back to computing it directly
from the financial statements, and if neither works leave it as None and
record it in `missing_fields`. Nothing here raises on missing data — a
ticker missing half its balance sheet still produces a partial RatioSet.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.models import CompanyData, RatioSet

DEFAULT_TAX_RATE = 0.21


def _clean(x) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    a, b = _clean(a), _clean(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _row(df: pd.DataFrame, label: str, col: int = 0) -> Optional[float]:
    if df is None or df.empty or label not in df.index:
        return None
    try:
        return _clean(df.loc[label].iloc[col])
    except (IndexError, KeyError):
        return None


def _yoy(latest: Optional[float], prior: Optional[float]) -> Optional[float]:
    latest, prior = _clean(latest), _clean(prior)
    if latest is None or prior is None or prior == 0:
        return None
    return (latest - prior) / abs(prior)


def _cagr(latest: Optional[float], oldest: Optional[float], years: int) -> Optional[float]:
    latest, oldest = _clean(latest), _clean(oldest)
    if latest is None or oldest is None or oldest <= 0 or latest <= 0 or years <= 0:
        return None
    return (latest / oldest) ** (1 / years) - 1


def compute_cash_runway(data: CompanyData, ratios: RatioSet) -> Optional[float]:
    """Years of cash left at the current burn rate. Only meaningful for
    cash-burning companies — returns None when FCF is positive/unknown or no
    cash balance is available. Kept out of RatioSet so a profitable company's
    'not applicable' doesn't read as missing data in the completeness score."""
    fcf = _clean(ratios.free_cash_flow)
    if fcf is None or fcf >= 0:
        return None
    info = data.info or {}
    cash = _clean(info.get("totalCash"))
    if cash is None:
        cash = _row(data.balance_sheet, "Cash Cash Equivalents And Short Term Investments")
    if cash is None:
        cash = _row(data.balance_sheet, "Cash And Cash Equivalents")
    if cash is None:
        return None
    return cash / abs(fcf)


class _Tracker:
    """Wraps RatioSet field assignment so we can record which fields were
    never resolved, for the confidence-degradation logic downstream."""

    def __init__(self, ratios: RatioSet):
        self.ratios = ratios
        self.missing: list[str] = []

    def set(self, field: str, value: Optional[float]) -> None:
        setattr(self.ratios, field, value)
        if value is None:
            self.missing.append(field)


def compute_ratios(data: CompanyData) -> RatioSet:
    info = data.info or {}
    inc, bs, cf = data.income_stmt, data.balance_sheet, data.cashflow

    ratios = RatioSet()
    tr = _Tracker(ratios)

    revenue = info.get("totalRevenue") or _row(inc, "Total Revenue")
    revenue_prior = _row(inc, "Total Revenue", 1)
    revenue_3y = _row(inc, "Total Revenue", 3)

    # --- Profitability ---
    tr.set("gross_margin", info.get("grossMargins") or _safe_div(_row(inc, "Gross Profit"), revenue))
    tr.set("operating_margin", info.get("operatingMargins") or _safe_div(_row(inc, "Operating Income"), revenue))
    tr.set("net_margin", info.get("profitMargins") or _safe_div(_row(inc, "Net Income"), revenue))

    ebitda = info.get("ebitda") or _row(inc, "EBITDA")
    tr.set("ebitda_margin", _safe_div(ebitda, revenue))

    equity = _row(bs, "Stockholders Equity")
    total_assets = _row(bs, "Total Assets")
    net_income = _row(inc, "Net Income")
    tr.set("roe", info.get("returnOnEquity") or _safe_div(net_income, equity))
    tr.set("roa", info.get("returnOnAssets") or _safe_div(net_income, total_assets))

    ebit = _row(inc, "EBIT") or _row(inc, "Operating Income")
    total_debt_bs = _row(bs, "Total Debt")
    cash_bs = _row(bs, "Cash And Cash Equivalents")
    invested_capital = _row(bs, "Invested Capital")
    if invested_capital is None and equity is not None and total_debt_bs is not None:
        invested_capital = equity + total_debt_bs - (cash_bs or 0)
    tax_rate = _row(inc, "Tax Rate For Calcs") or DEFAULT_TAX_RATE
    nopat = ebit * (1 - tax_rate) if ebit is not None else None
    tr.set("roic", _safe_div(nopat, invested_capital))

    # --- Growth ---
    tr.set("revenue_growth_yoy", info.get("revenueGrowth") or _yoy(revenue, revenue_prior))
    tr.set("revenue_cagr_3y", _cagr(revenue, revenue_3y, 3))

    eps_latest = info.get("trailingEps") or _row(inc, "Diluted EPS")
    eps_prior = _row(inc, "Diluted EPS", 1)
    tr.set("eps_growth_yoy", info.get("earningsGrowth") or _yoy(eps_latest, eps_prior))

    fcf_latest = info.get("freeCashflow") or _row(cf, "Free Cash Flow")
    fcf_prior = _row(cf, "Free Cash Flow", 1)
    tr.set("fcf_growth_yoy", _yoy(fcf_latest, fcf_prior))

    # --- Leverage / liquidity ---
    d2e_info = info.get("debtToEquity")
    tr.set("debt_to_equity", (d2e_info / 100) if d2e_info is not None else _safe_div(total_debt_bs, equity))

    total_cash_info = info.get("totalCash") or cash_bs
    total_debt_info = info.get("totalDebt") or total_debt_bs
    net_debt = None
    if total_debt_info is not None:
        net_debt = total_debt_info - (total_cash_info or 0)
    tr.set("net_debt_to_ebitda", _safe_div(net_debt, ebitda))

    tr.set("current_ratio", info.get("currentRatio") or _safe_div(_row(bs, "Current Assets"), _row(bs, "Current Liabilities")))
    quick_assets = None
    ca, inv = _row(bs, "Current Assets"), _row(bs, "Inventory")
    if ca is not None:
        quick_assets = ca - (inv or 0)
    tr.set("quick_ratio", info.get("quickRatio") or _safe_div(quick_assets, _row(bs, "Current Liabilities")))

    interest_expense = _row(inc, "Interest Expense")
    tr.set("interest_coverage", _safe_div(ebit, abs(interest_expense) if interest_expense is not None else None))

    # --- Cash flow ---
    if fcf_latest is None:
        ocf = info.get("operatingCashflow") or _row(cf, "Operating Cash Flow")
        capex = _row(cf, "Capital Expenditure")
        if ocf is not None and capex is not None:
            fcf_latest = ocf + capex  # capex is stored negative
    tr.set("free_cash_flow", fcf_latest)
    tr.set("fcf_margin", _safe_div(fcf_latest, revenue))
    tr.set("fcf_yield", _safe_div(fcf_latest, data.market_cap))
    capex_val = _row(cf, "Capital Expenditure")
    tr.set("capex_to_revenue", _safe_div(abs(capex_val) if capex_val is not None else None, revenue))

    # --- Valuation multiples ---
    tr.set("pe_trailing", info.get("trailingPE"))
    tr.set("pe_forward", info.get("forwardPE"))
    tr.set("price_to_book", info.get("priceToBook"))
    tr.set("price_to_sales", info.get("priceToSalesTrailing12Months"))
    tr.set("ev_to_ebitda", info.get("enterpriseToEbitda") or _safe_div(data.enterprise_value, ebitda))
    tr.set("ev_to_revenue", info.get("enterpriseToRevenue") or _safe_div(data.enterprise_value, revenue))
    tr.set("peg_ratio", info.get("trailingPegRatio") or info.get("pegRatio"))

    # --- Dividend ---
    div_yield_info = info.get("dividendYield")
    div_rate = info.get("dividendRate")
    tr.set("dividend_yield", (div_yield_info / 100) if div_yield_info is not None else _safe_div(div_rate, data.current_price))
    tr.set("payout_ratio", info.get("payoutRatio"))

    # --- Size / risk ---
    tr.set("market_cap", data.market_cap)
    tr.set("enterprise_value", data.enterprise_value)
    tr.set("beta", data.beta)

    ratios.missing_fields = tr.missing
    return ratios
