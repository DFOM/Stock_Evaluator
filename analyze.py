#!/usr/bin/env python3
"""CLI entrypoint.

    python analyze.py <TICKER> [<TICKER> ...]          full analysis (rule-based + AI ensemble)
    python analyze.py <TICKER> --no-ai                 rule-based only, free, no API calls
    python analyze.py <TICKER> --analysts a,b          run only the named analysts from config/analysts.yaml

Pipeline: fetch -> ratios -> archetype -> scoring -> rule-based fair value ->
briefing packet -> AI analyst ensemble (consensus + disagreement index) ->
console card + runs.jsonl cost log.
"""
from __future__ import annotations

import argparse
import sys

from src.archetype_classifier import classify
from src.briefing import build_briefing_packet
from src.config import load_config
from src.data_fetcher import fetch_company_data
from src.ensemble import run_ensemble
from src.fair_value import estimate_fair_value
from src.ratio_engine import compute_ratios
from src.report import console, render_ensemble, render_report
from src.run_log import append_run
from src.scoring import score_company


def analyze_ticker(ticker: str, config: dict, use_ai: bool, only_analysts: list[str] | None) -> None:
    console.rule(f"[bold]{ticker.upper()}[/bold]")
    data = fetch_company_data(ticker)

    if not data.current_price and not data.info:
        console.print(f"[bold red]Could not fetch any data for '{ticker}'. Check the symbol and try again.[/bold red]")
        return

    ratios = compute_ratios(data)
    archetype = classify(data, ratios)
    scores = score_company(data, ratios, archetype, config)
    fair_value = estimate_fair_value(data, ratios, archetype, config)

    render_report(data, ratios, archetype, scores, fair_value)

    if not use_ai:
        return

    briefing = build_briefing_packet(data, ratios, archetype, scores, fair_value)
    with console.status("Consulting AI analyst ensemble..."):
        ensemble = run_ensemble(briefing, fair_value, only_analysts)
    render_ensemble(ensemble, data.currency)
    append_run(data.ticker, ensemble)


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal stock evaluation tool.")
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols, e.g. AAPL KO JPM O PLTR")
    parser.add_argument("--no-ai", action="store_true", help="Rule-based analysis only — no AI calls, no cost.")
    parser.add_argument(
        "--analysts",
        metavar="NAME1,NAME2",
        help="Comma-separated subset of analysts from config/analysts.yaml to consult (overrides 'active' flags).",
    )
    args = parser.parse_args()

    only_analysts = [a.strip() for a in args.analysts.split(",") if a.strip()] if args.analysts else None

    config = load_config()
    for ticker in args.tickers:
        try:
            analyze_ticker(ticker, config, use_ai=not args.no_ai, only_analysts=only_analysts)
        except Exception as e:
            console.print(f"[bold red]Failed to analyze {ticker}: {e}[/bold red]")
        console.print()


if __name__ == "__main__":
    sys.exit(main())
