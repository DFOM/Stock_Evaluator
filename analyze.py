#!/usr/bin/env python3
"""CLI entrypoint: python analyze.py <TICKER> [<TICKER> ...]

Runs the full Phase 1 pipeline (fetch -> ratios -> archetype -> scoring ->
fair value) and prints an analysis card per ticker. Also builds the
standardized briefing packet in memory (the artifact Phase 2's AI analyst
ensemble will consume) so the interface is exercised end-to-end even though
nothing sends it anywhere yet.
"""
from __future__ import annotations

import argparse
import sys

from src.archetype_classifier import classify
from src.briefing import build_briefing_packet
from src.config import load_config
from src.data_fetcher import fetch_company_data
from src.fair_value import estimate_fair_value
from src.ratio_engine import compute_ratios
from src.report import console, render_report
from src.scoring import score_company


def analyze_ticker(ticker: str, config: dict) -> None:
    console.rule(f"[bold]{ticker.upper()}[/bold]")
    data = fetch_company_data(ticker)

    if not data.current_price and not data.info:
        console.print(f"[bold red]Could not fetch any data for '{ticker}'. Check the symbol and try again.[/bold red]")
        return

    ratios = compute_ratios(data)
    archetype = classify(data, ratios)
    scores = score_company(data, ratios, archetype, config)
    fair_value = estimate_fair_value(data, ratios, archetype, config)

    # Built now so the Phase 2 interface is exercised, even though nothing
    # consumes it yet in Phase 1.
    _ = build_briefing_packet(data, ratios, archetype, scores, fair_value)

    render_report(data, ratios, archetype, scores, fair_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal stock evaluation tool (Phase 1: local, rule-based).")
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols, e.g. AAPL KO JPM O PLTR")
    args = parser.parse_args()

    config = load_config()
    for ticker in args.tickers:
        try:
            analyze_ticker(ticker, config)
        except Exception as e:
            console.print(f"[bold red]Failed to analyze {ticker}: {e}[/bold red]")
        console.print()


if __name__ == "__main__":
    sys.exit(main())
