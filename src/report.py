"""Renders the full analysis card to the console with `rich`."""
from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.ensemble import RULE_BASED_NAME, EnsembleResult
from src.models import ArchetypeResult, CompanyData, FairValueResult, RatioSet, ScoreResult

console = Console()


def _pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "n/a"


def _num(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


def _num_x(v: Optional[float]) -> str:
    return f"{v:.2f}x" if v is not None else "n/a"


def _money(v: Optional[float], currency: Optional[str]) -> str:
    if v is None:
        return "n/a"
    cur = currency or ""
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f}B {cur}".strip()
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M {cur}".strip()
    return f"{v:,.2f} {cur}".strip()


def _score_color(score: Optional[float]) -> str:
    if score is None:
        return "dim"
    if score >= 7:
        return "bold green"
    if score >= 4:
        return "bold yellow"
    return "bold red"


def _header(data: CompanyData) -> Panel:
    lines = [
        f"[bold]{data.long_name or data.ticker}[/bold] ({data.ticker})",
        f"{data.sector or 'n/a'} / {data.industry or 'n/a'}  |  {data.country or 'n/a'}",
        f"Price: {_money(data.current_price, data.currency)}   "
        f"Market Cap: {_money(data.market_cap, data.currency)}   "
        f"Enterprise Value: {_money(data.enterprise_value, data.currency)}",
    ]
    return Panel("\n".join(lines), title="Company", border_style="cyan")


def _archetype_panel(archetype: ArchetypeResult) -> Panel:
    lines = [f"[bold]{archetype.archetype.upper()}[/bold]  (confidence {archetype.confidence:.0%})", ""]
    lines.append("Why:")
    for r in archetype.reasons:
        lines.append(f"  • {r}")
    lines.append("")
    lines.append("Candidate scores: " + ", ".join(f"{k}={v:g}" for k, v in archetype.scores.items()))
    return Panel("\n".join(lines), title="Archetype Classification", border_style="magenta")


def _ratios_table(ratios: RatioSet) -> Table:
    table = Table(title="Key Ratios", show_lines=False)
    table.add_column("Category", style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    groups: list[tuple[str, list[tuple[str, str]]]] = [
        ("Profitability", [
            ("Gross margin", _pct(ratios.gross_margin)),
            ("Operating margin", _pct(ratios.operating_margin)),
            ("Net margin", _pct(ratios.net_margin)),
            ("EBITDA margin", _pct(ratios.ebitda_margin)),
            ("ROE", _pct(ratios.roe)),
            ("ROA", _pct(ratios.roa)),
            ("ROIC", _pct(ratios.roic)),
        ]),
        ("Growth", [
            ("Revenue growth (YoY)", _pct(ratios.revenue_growth_yoy)),
            ("Revenue CAGR (3y)", _pct(ratios.revenue_cagr_3y)),
            ("EPS growth (YoY)", _pct(ratios.eps_growth_yoy)),
            ("FCF growth (YoY)", _pct(ratios.fcf_growth_yoy)),
        ]),
        ("Leverage / Liquidity", [
            ("Debt / Equity", _num(ratios.debt_to_equity)),
            ("Net debt / EBITDA", _num_x(ratios.net_debt_to_ebitda)),
            ("Current ratio", _num(ratios.current_ratio)),
            ("Quick ratio", _num(ratios.quick_ratio)),
            ("Interest coverage", _num_x(ratios.interest_coverage)),
        ]),
        ("Cash Flow", [
            ("Free cash flow", _money(ratios.free_cash_flow, None)),
            ("FCF margin", _pct(ratios.fcf_margin)),
            ("FCF yield", _pct(ratios.fcf_yield)),
            ("Capex / Revenue", _pct(ratios.capex_to_revenue)),
        ]),
        ("Valuation", [
            ("P/E (trailing)", _num(ratios.pe_trailing)),
            ("P/E (forward)", _num(ratios.pe_forward)),
            ("Price / Book", _num(ratios.price_to_book)),
            ("Price / Sales", _num(ratios.price_to_sales)),
            ("EV / EBITDA", _num(ratios.ev_to_ebitda)),
            ("EV / Revenue", _num(ratios.ev_to_revenue)),
            ("PEG ratio", _num(ratios.peg_ratio)),
        ]),
        ("Dividend", [
            ("Dividend yield", _pct(ratios.dividend_yield)),
            ("Payout ratio", _pct(ratios.payout_ratio)),
        ]),
    ]

    for i, (cat, rows) in enumerate(groups):
        for j, (metric, value) in enumerate(rows):
            table.add_row(cat if j == 0 else "", metric, value)
        if i < len(groups) - 1:
            table.add_row("", "", "")

    return table


def _pillar_table(scores: ScoreResult) -> Table:
    table = Table(title="Pillar Scores (0-10)")
    table.add_column("Pillar", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")

    for name, ps in scores.pillar_scores.items():
        score_txt = f"{ps.score:.1f}" if ps.score is not None else "n/a"
        table.add_row(
            name.replace("_", " ").title(),
            Text(score_txt, style=_score_color(ps.score)),
            f"{ps.confidence:.0%}",
        )
    return table


def _scores_panel(scores: ScoreResult) -> Panel:
    inv = f"{scores.investor_score:.1f}/10" if scores.investor_score is not None else "n/a"
    trd = f"{scores.trader_score:.1f}/10" if scores.trader_score is not None else "n/a"
    lines = [
        f"[bold]Investor Score:[/bold] [{_score_color(scores.investor_score)}]{inv}[/{_score_color(scores.investor_score)}]  "
        "(years horizon: valuation, quality, financial health dominate)",
        f"[bold]Trader Score:[/bold]   [{_score_color(scores.trader_score)}]{trd}[/{_score_color(scores.trader_score)}]  "
        "(weeks-months horizon: momentum dominates)",
        "",
        f"[bold]Quality-vs-Price Quadrant:[/bold] {scores.quadrant}",
        f"  {scores.quadrant_reason}",
    ]
    return Panel("\n".join(lines), title="Scores", border_style="green")


def _red_flags_panel(scores: ScoreResult) -> Panel:
    lines = []
    for f in scores.red_flags:
        lines.append(f"[bold red]● {f.name}[/bold red] (caps score at {f.score_cap:.1f}/10)")
        lines.append(f"    {f.description}")
    for note in scores.info_notes:
        lines.append(f"[yellow]ℹ {note}[/yellow]")
    if not lines:
        return Panel("No red flags detected.", title="Red Flags", border_style="green")
    style = "red" if scores.red_flags else "yellow"
    return Panel("\n".join(lines), title="Red Flags & Notes", border_style=style)


def _fair_value_panel(fv: FairValueResult) -> Panel:
    lines = [f"[bold]Method:[/bold] {fv.method}", f"[bold]Confidence:[/bold] {fv.confidence:.0%}", ""]
    if fv.mid is not None:
        lines.append(f"Fair value range: {_money(fv.low, fv.currency)}  —  {_money(fv.mid, fv.currency)} (mid)  —  {_money(fv.high, fv.currency)}")
        if fv.margin_of_safety_price is not None:
            mos = fv.assumptions.get("margin_of_safety")
            mos_txt = f" ({mos:.0%} margin of safety)" if mos is not None else ""
            lines.append(f"Suggested buy-below target: {_money(fv.margin_of_safety_price, fv.currency)}{mos_txt}")
    else:
        lines.append("[dim]Could not compute a fair value range — insufficient data.[/dim]")
    if fv.assumptions:
        lines.append("")
        lines.append("[bold]Assumptions:[/bold]")
        for k, v in fv.assumptions.items():
            lines.append(f"  • {k}: {v}")
    if fv.notes:
        lines.append("")
        lines.append("[bold]Notes:[/bold]")
        for n in fv.notes:
            lines.append(f"  • {n}")
    return Panel("\n".join(lines), title="Rule-Based Fair Value", border_style="blue")


def _ai_disagreement_label(di: Optional[float]) -> str:
    if di is None:
        return "n/a (needs 2+ AI analysts)"
    if di < 0.10:
        return f"[bold green]{di:.1%} (AI analysts broadly agree)[/bold green]"
    if di < 0.25:
        return f"[bold yellow]{di:.1%} (moderate — some genuine valuation uncertainty)[/bold yellow]"
    return f"[bold red]{di:.1%} (HIGH — AI analysts see very different fair values; hard to value)[/bold red]"


def _model_gap_label(gap: Optional[float]) -> str:
    if gap is None:
        return "n/a (needs rule-based estimate + AI consensus)"
    if gap < 0.20:
        return f"[bold green]{gap:.1%} (rule-based engine and AI judgment broadly aligned)[/bold green]"
    if gap < 0.50:
        return f"[bold yellow]{gap:.1%} (notable — review the rule-based assumptions)[/bold yellow]"
    return f"[bold red]{gap:.1%} (LARGE — rule-based assumptions and LLM judgment diverge sharply)[/bold red]"


def _ensemble_panel(ensemble: EnsembleResult, currency: Optional[str]) -> Panel:
    lines: list[str] = []

    if ensemble.consensus_mid is not None:
        lines.append(
            f"[bold]Consensus fair value:[/bold] {_money(ensemble.consensus_low, currency)}  —  "
            f"{_money(ensemble.consensus_mid, currency)} (mid)  —  {_money(ensemble.consensus_high, currency)}"
        )
        conf = ensemble.overall_confidence
        lines.append(f"[bold]Overall confidence:[/bold] {conf:.0%}" if conf is not None else "")
        lines.append(f"[bold]AI disagreement:[/bold] {_ai_disagreement_label(ensemble.ai_disagreement)}")
        lines.append(f"[bold]Model gap:[/bold]       {_model_gap_label(ensemble.model_gap)}")
        if ensemble.price_deviation is not None:
            lines.append(f"[bold]Price deviation:[/bold] {ensemble.price_deviation:.1%} (AI consensus mid vs market price)")
        if ensemble.anchored_warning:
            lines.append(
                "[bold yellow]⚠ AI consensus may be anchored to market price — treat agreement as low-information.[/bold yellow]"
            )
    else:
        lines.append("[dim]No usable analyst responses — consensus unavailable.[/dim]")

    lines.append("")
    lines.append(f"[bold]Analysts ({ensemble.contributing_analysts} contributing):[/bold]")
    for r in ensemble.responses:
        if r.ok and r.midpoint is not None:
            if r.archetype_agreement is False:
                stance = f"[yellow]disputes archetype → {r.suggested_archetype or '?'}[/yellow]"
            else:
                stance = "agrees with archetype"
            tag = " [dim](rule-based)[/dim]" if r.analyst_name == RULE_BASED_NAME else ""
            lines.append(
                f"  • {r.analyst_name}{tag}: {_money(r.fair_value_low, currency)} — {_money(r.fair_value_high, currency)}"
                f" (mid {_money(r.midpoint, currency)}), conf {r.confidence:.0%}, {stance}"
            )
        else:
            lines.append(f"  • [red]{r.analyst_name}: FAILED — {r.error}[/red]")

    if ensemble.archetype_disputes:
        lines.append("")
        lines.append("[bold yellow]Archetype disputes:[/bold yellow]")
        for d in ensemble.archetype_disputes:
            lines.append(f"  ⚠ {d}")

    theses = [r for r in ensemble.responses if r.ok and r.one_paragraph_thesis and r.analyst_name != RULE_BASED_NAME]
    if theses:
        lines.append("")
        lines.append("[bold]Theses:[/bold]")
        for r in theses:
            lines.append(f"  [bold]{r.analyst_name}:[/bold] {r.one_paragraph_thesis}")

    risks: list[str] = []
    for r in ensemble.responses:
        for risk in r.key_risks:
            if risk not in risks:
                risks.append(risk)
    if risks:
        lines.append("")
        lines.append("[bold]Key risks flagged by analysts:[/bold]")
        for risk in risks[:8]:
            lines.append(f"  • {risk}")

    if ensemble.skipped:
        lines.append("")
        for msg in ensemble.skipped:
            lines.append(f"[yellow]⚠ {msg}[/yellow]")

    lines.append("")
    lines.append(f"[bold]Estimated AI cost for this analysis:[/bold] ${ensemble.total_cost:.4f}")

    title = "AI Analyst Ensemble (PRICE-BLIND)" if ensemble.blind else "AI Analyst Ensemble"
    return Panel("\n".join(l for l in lines if l is not None), title=title, border_style="magenta")


def render_ensemble(ensemble: EnsembleResult, currency: Optional[str]) -> None:
    console.print(_ensemble_panel(ensemble, currency))


def render_report(
    data: CompanyData,
    ratios: RatioSet,
    archetype: ArchetypeResult,
    scores: ScoreResult,
    fair_value: FairValueResult,
) -> None:
    console.print(_header(data))
    if data.warnings:
        console.print(Panel("\n".join(f"⚠ {w}" for w in data.warnings), title="Data Warnings", border_style="yellow"))
    console.print(_archetype_panel(archetype))
    console.print(_ratios_table(ratios))
    console.print(f"[dim]Ratio data completeness: {ratios.completeness:.0%}[/dim]")
    console.print(_pillar_table(scores))
    console.print(_scores_panel(scores))
    console.print(_red_flags_panel(scores))
    console.print(_fair_value_panel(fair_value))
