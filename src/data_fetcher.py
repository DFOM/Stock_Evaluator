"""Fetches raw fundamentals from yfinance and packs them into a CompanyData.

Everything here is defensive: yfinance's `.info` dict, and even which rows
exist in the financial statements, varies by ticker (foreign listings, ETFs,
newly-IPO'd companies, financials vs industrials all differ). We never raise
on missing fields — we record a warning and let downstream stages degrade
confidence instead of crashing.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.models import CompanyData


def fetch_company_data(ticker: str) -> CompanyData:
    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)

    data = CompanyData(ticker=ticker)

    try:
        info = t.info or {}
    except Exception as e:  # yfinance network / parsing failures
        info = {}
        data.note(f"Could not fetch .info: {e}")
    data.info = info

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        data.note("Ticker info looks empty or invalid — check the symbol.")

    data.long_name = info.get("longName") or info.get("shortName")
    data.sector = info.get("sector")
    data.industry = info.get("industry")
    data.country = info.get("country")
    data.currency = info.get("currency")
    data.current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    data.market_cap = info.get("marketCap")
    data.enterprise_value = info.get("enterpriseValue")
    data.shares_outstanding = info.get("sharesOutstanding")
    data.beta = info.get("beta")

    for label, getter in (
        ("income_stmt", lambda: t.income_stmt),
        ("balance_sheet", lambda: t.balance_sheet),
        ("cashflow", lambda: t.cashflow),
    ):
        try:
            df = getter()
            if df is None or df.empty:
                data.note(f"{label} unavailable from yfinance.")
                df = pd.DataFrame()
        except Exception as e:
            data.note(f"Failed to fetch {label}: {e}")
            df = pd.DataFrame()
        setattr(data, label, df)

    try:
        hist = t.history(period="5y")
        if hist is None or hist.empty:
            data.note("Price history unavailable.")
            hist = pd.DataFrame()
    except Exception as e:
        data.note(f"Failed to fetch price history: {e}")
        hist = pd.DataFrame()
    data.price_history = hist

    if not data.sector:
        data.note("Sector/industry missing — archetype classification will rely on financial ratios only.")

    return data
