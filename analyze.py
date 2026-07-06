#!/usr/bin/env python3
"""CLI entrypoint.

    python analyze.py <TICKER> [<TICKER> ...]          dual-pass analysis (default):
                                                         pass A blind -> fair-value consensus
                                                         pass B sighted -> critic notes only
    python analyze.py <TICKER> --no-ai                 rule-based only, free, no API calls
    python analyze.py <TICKER> --blind-only            debug: consensus pass only, no critic
    python analyze.py <TICKER> --sighted-only          debug: pre-2.7 sighted ensemble (anchoring-prone)
    python analyze.py <TICKER> --analysts a,b          run only the named analysts from config/analysts.yaml

Pipeline: fetch -> ratios -> archetype -> scoring -> rule-based fair value ->
briefing packet (sighted + price-blind variants) -> dual-pass AI ensemble ->
console card + runs.jsonl cost log.
"""
from __future__ import annotations

import argparse
import sys

from src.archetype_classifier import classify
from src.briefing import build_briefing_packet, redact_for_blind
from src.config import load_config
from src.data_fetcher import fetch_company_data
from src.ensemble import run_dual_pass
from src.fair_value import estimate_fair_value
from src.ratio_engine import compute_ratios
from src.report import console, render_report
from src.run_log import append_run
from src.scoring import score_company


def analyze_ticker(ticker: str, config: dict, use_ai: bool, only_analysts: list[str] | None, mode: str = "dual") -> None:
    console.rule(f"[bold]{ticker.upper()}[/bold]")
    data = fetch_company_data(ticker)

    if not data.current_price and not data.info:
        console.print(f"[bold red]Could not fetch any data for '{ticker}'. Check the symbol and try again.[/bold red]")
        return

    ratios = compute_ratios(data)
    archetype = classify(data, ratios)
    scores = score_company(data, ratios, archetype, config)
    fair_value = estimate_fair_value(data, ratios, archetype, config)

    dual = None
    if use_ai:
        briefing = build_briefing_packet(data, ratios, archetype, scores, fair_value)
        blind_briefing = redact_for_blind(briefing)
        status = {
            "dual": "Consulting AI analysts (blind consensus pass + sighted critic pass)...",
            "blind_only": "Consulting AI analysts (blind pass only)...",
            "sighted_only": "Consulting AI analysts (sighted pass only)...",
        }[mode]
        with console.status(status):
            dual = run_dual_pass(
                briefing, blind_briefing, fair_value,
                only_names=only_analysts, mode=mode, current_price=data.current_price,
            )

    render_report(data, ratios, archetype, scores, fair_value, dual)
    if dual is not None:
        append_run(data.ticker, dual)


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal stock evaluation tool.")
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols, e.g. AAPL KO JPM O PLTR")
    parser.add_argument("--no-ai", action="store_true", help="Rule-based analysis only — no AI calls, no cost.")
    parser.add_argument(
        "--analysts",
        metavar="NAME1,NAME2",
        help="Comma-separated subset of analysts from config/analysts.yaml to consult (overrides 'active' flags).",
    )
    pass_group = parser.add_mutually_exclusive_group()
    pass_group.add_argument(
        "--blind-only", action="store_true",
        help="Debug: run only the price-blind consensus pass (no sighted critic pass).",
    )
    pass_group.add_argument(
        "--sighted-only", action="store_true",
        help="Debug: run only the sighted pass and build the consensus from it "
        "(pre-2.7 behavior; anchoring-prone).",
    )
    args = parser.parse_args()

    only_analysts = [a.strip() for a in args.analysts.split(",") if a.strip()] if args.analysts else None
    mode = "blind_only" if args.blind_only else "sighted_only" if args.sighted_only else "dual"

    config = load_config()
    for ticker in args.tickers:
        try:
            analyze_ticker(ticker, config, use_ai=not args.no_ai, only_analysts=only_analysts, mode=mode)
        except Exception as e:
            console.print(f"[bold red]Failed to analyze {ticker}: {e}[/bold red]")
        console.print()


if __name__ == "__main__":
    sys.exit(main())
