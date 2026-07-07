
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

from src.config import load_config
from src.pipeline import run_full_analysis
from src.report import console, render_report
from src.run_log import append_run


def analyze_ticker(ticker: str, config: dict, use_ai: bool, only_analysts: list[str] | None, mode: str = "dual") -> None:
    console.rule(f"[bold]{ticker.upper()}[/bold]")
    status = {
        "dual": "Analyzing (blind consensus pass + sighted critic pass)...",
        "blind_only": "Analyzing (blind pass only)...",
        "sighted_only": "Analyzing (sighted pass only)...",
    }[mode]
    try:
        with console.status(status):
            result = run_full_analysis(ticker, config, use_ai=use_ai, only_analysts=only_analysts, mode=mode)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        return

    render_report(
        result.data, result.ratios, result.archetype, result.scores, result.fair_value,
        result.dual, alt_valuation=result.alt_valuation, position_hint=result.position_hint,
    )
    if result.dual is not None:
        append_run(result.ticker, result.dual)


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
