"""SQLAlchemy models + session plumbing. DATABASE_URL-driven: SQLite for
local dev, Postgres on Railway (whose postgres:// scheme is normalized to
the postgresql:// SQLAlchemy expects)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from webapp import settings

_url = settings.DATABASE_URL
if _url.startswith("postgres://"):
    _url = _url.replace("postgres://", "postgresql://", 1)

_engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
if _url.startswith("sqlite"):
    # The analysis runner updates rows from a worker thread.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(_url, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|done|failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Full card as JSON (PipelineResult.to_payload()).
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Queryable headline columns, denormalized from the payload.
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    investor_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trader_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    consensus_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    consensus_mid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    consensus_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ai_disagreement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_gap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sighted_mid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    anchored_warning: Mapped[bool] = mapped_column(Boolean, default=False)

    runs: Mapped[list["AnalystRun"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")


class AnalystRun(Base):
    """Per-analyst rows — the runs.jsonl concept as a table (the jsonl file
    keeps being written in parallel for now)."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    pass_name: Mapped[str] = mapped_column(String(16))  # blind|sighted
    analyst_name: Mapped[str] = mapped_column(String(64))
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    fair_value_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fair_value_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # full response (risks, thesis, ...)

    analysis: Mapped[Analysis] = relationship(back_populates="runs")


def init_db() -> None:
    Base.metadata.create_all(engine)
