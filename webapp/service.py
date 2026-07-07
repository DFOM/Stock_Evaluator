"""Analysis orchestration for the web app: cache lookups, the concurrency
guard, and the background runner that wraps the engine pipeline."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from sqlalchemy import select

from src.config import load_config
from src.pipeline import PipelineResult, run_full_analysis
from src.run_log import append_run
from webapp import settings
from webapp.db import Analysis, AnalystRun, SessionLocal, utcnow

log = logging.getLogger("webapp.service")

ACTIVE_STATUSES = ("queued", "running")


def find_cached(ticker: str) -> Optional[Analysis]:
    """Most recent completed analysis younger than ANALYSIS_CACHE_HOURS."""
    cutoff = utcnow() - timedelta(hours=settings.ANALYSIS_CACHE_HOURS)
    with SessionLocal() as db:
        return db.execute(
            select(Analysis)
            .where(Analysis.ticker == ticker, Analysis.status == "done", Analysis.finished_at >= cutoff)
            .order_by(Analysis.finished_at.desc())
            .limit(1)
        ).scalar_one_or_none()


def find_active(ticker: str) -> Optional[Analysis]:
    with SessionLocal() as db:
        return db.execute(
            select(Analysis)
            .where(Analysis.ticker == ticker, Analysis.status.in_(ACTIVE_STATUSES))
            .order_by(Analysis.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()


def active_count() -> int:
    with SessionLocal() as db:
        return len(db.execute(select(Analysis.id).where(Analysis.status.in_(ACTIVE_STATUSES))).all())


def create_analysis(ticker: str) -> Analysis:
    with SessionLocal() as db:
        analysis = Analysis(ticker=ticker, status="queued")
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis


def get_analysis(analysis_id: int) -> Optional[Analysis]:
    with SessionLocal() as db:
        return db.get(Analysis, analysis_id)


def recent_analyses(limit: int = 20) -> list[Analysis]:
    with SessionLocal() as db:
        return list(
            db.execute(select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)).scalars()
        )


def _store_runs(db, analysis: Analysis, result: PipelineResult) -> None:
    if not result.dual:
        return
    passes = [("blind", result.dual.consensus.responses)]
    if result.dual.critic:
        passes.append(("sighted", result.dual.critic.responses))
    for pass_name, responses in passes:
        for r in responses:
            db.add(AnalystRun(
                analysis_id=analysis.id,
                pass_name=pass_name,
                analyst_name=r.analyst_name,
                ok=r.ok,
                fair_value_low=r.fair_value_low,
                fair_value_high=r.fair_value_high,
                confidence=r.confidence,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                estimated_cost=r.estimated_cost,
                error=r.error,
                payload=r.as_log_dict(),
            ))


def run_analysis_task(analysis_id: int) -> None:
    """Background worker: runs the full engine pipeline and lands the result
    (or the failure) in the analyses row. Never raises."""
    with SessionLocal() as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis is None or analysis.status not in ACTIVE_STATUSES:
            return
        analysis.status = "running"
        db.commit()
        ticker = analysis.ticker

    try:
        config = load_config()
        result = run_full_analysis(ticker, config, use_ai=True, mode="dual")
        payload = result.to_payload()

        with SessionLocal() as db:
            analysis = db.get(Analysis, analysis_id)
            analysis.status = "done"
            analysis.finished_at = utcnow()
            analysis.payload = payload
            analysis.total_cost = result.total_cost
            analysis.investor_score = result.scores.investor_score
            analysis.trader_score = result.scores.trader_score
            if result.dual:
                cons = result.dual.consensus
                analysis.consensus_low = cons.consensus_low
                analysis.consensus_mid = cons.consensus_mid
                analysis.consensus_high = cons.consensus_high
                analysis.overall_confidence = cons.overall_confidence
                analysis.ai_disagreement = cons.ai_disagreement
                analysis.model_gap = cons.model_gap
                if result.dual.critic:
                    analysis.sighted_mid = result.dual.critic.ai_consensus_mid
                    analysis.anchored_warning = result.dual.critic.anchored_warning
            _store_runs(db, analysis, result)
            db.commit()

        # Keep the file-based log in parallel for now (Phase 3 spec).
        if result.dual:
            try:
                append_run(result.ticker, result.dual)
            except OSError as e:
                log.warning("runs.jsonl append failed: %s", e)
    except Exception as e:
        log.exception("analysis %s (%s) failed", analysis_id, ticker)
        with SessionLocal() as db:
            analysis = db.get(Analysis, analysis_id)
            if analysis is not None:
                analysis.status = "failed"
                analysis.finished_at = utcnow()
                analysis.error = f"{type(e).__name__}: {e}"
                db.commit()
