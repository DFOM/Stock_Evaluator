"""File-based run log (runs.jsonl) — the Phase 3 DB run-log table's interim
form. One JSON line per analysis run: timestamp, ticker, per-analyst
responses with token usage and estimated cost, and the ensemble consensus.
Kept append-only so it can back cost tracking and backtesting later."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ensemble import EnsembleResult

RUNS_LOG_PATH = Path(__file__).resolve().parent.parent / "runs.jsonl"


def append_run(ticker: str, ensemble: EnsembleResult) -> None:
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "analysts": [r.as_log_dict() for r in ensemble.responses],
        "consensus": ensemble.as_log_dict(),
        "total_input_tokens": sum(r.input_tokens for r in ensemble.responses),
        "total_output_tokens": sum(r.output_tokens for r in ensemble.responses),
        "total_cost": round(ensemble.total_cost, 6),
    }
    with open(RUNS_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
