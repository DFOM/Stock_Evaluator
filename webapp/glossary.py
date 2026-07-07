"""Loads config/glossary.yaml and exposes it to templates.

A term missing from the file simply renders as plain text (the Jinja macro
guards on presence), so cards never break when a slug is absent.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "config" / "glossary.yaml"


@lru_cache(maxsize=1)
def load_glossary() -> dict[str, dict[str, str]]:
    with open(GLOSSARY_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    terms = data.get("terms", data)
    # Normalize: collapse the folded-scalar whitespace in `long`, ensure a title.
    out: dict[str, dict[str, str]] = {}
    for key, entry in terms.items():
        if not isinstance(entry, dict):
            continue
        out[key] = {
            "title": entry.get("title", key.replace("_", " ").title()),
            "short": " ".join((entry.get("short") or "").split()),
            "long": " ".join((entry.get("long") or "").split()),
        }
    return out


def sorted_terms() -> list[tuple[str, dict[str, str]]]:
    """All entries sorted by display title, for the /glossary page."""
    return sorted(load_glossary().items(), key=lambda kv: kv[1]["title"].lower())


# Maps used by the card template to attach the right glossary slug to a
# label. Kept here (not in the template) so they're easy to extend.

PILLAR_GLOSS = {
    "valuation": "pillar_valuation",
    "quality": "pillar_quality",
    "growth": "pillar_growth",
    "financial_health": "pillar_financial_health",
    "macro": "pillar_macro",
    "momentum": "pillar_momentum",
}

# Ratio field name -> glossary slug (most are 1:1; a few differ).
RATIO_GLOSS = {
    "gross_margin": "gross_margin", "operating_margin": "operating_margin",
    "net_margin": "net_margin", "ebitda_margin": "ebitda_margin",
    "roe": "roe", "roa": "roa", "roic": "roic",
    "revenue_growth_yoy": "revenue_growth_yoy", "revenue_cagr_3y": "revenue_cagr_3y",
    "eps_growth_yoy": "eps_growth_yoy", "fcf_growth_yoy": "fcf_growth_yoy",
    "debt_to_equity": "debt_to_equity", "net_debt_to_ebitda": "net_debt_to_ebitda",
    "current_ratio": "current_ratio", "quick_ratio": "quick_ratio",
    "interest_coverage": "interest_coverage",
    "free_cash_flow": "free_cash_flow", "fcf_margin": "fcf_margin",
    "fcf_yield": "fcf_yield", "capex_to_revenue": "capex_to_revenue",
    "pe_trailing": "pe_trailing", "pe_forward": "pe_forward",
    "price_to_book": "price_to_book", "price_to_sales": "price_to_sales",
    "ev_to_ebitda": "ev_to_ebitda", "ev_to_revenue": "ev_to_revenue", "peg_ratio": "peg_ratio",
    "dividend_yield": "dividend_yield", "payout_ratio": "payout_ratio",
}


def quadrant_slug(verdict: str) -> str:
    """Map a quadrant verdict string to its glossary slug (falls back to the
    generic 'quadrant' entry)."""
    v = (verdict or "").lower()
    if "strong buy" in v:
        return "strong_buy"
    if "watchlist" in v:
        return "watchlist"
    if "value trap" in v:
        return "value_trap"
    if "avoid" in v:
        return "avoid"
    return "quadrant"
