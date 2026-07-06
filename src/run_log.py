"""File-based run log (runs.jsonl) — the Phase 3 DB run-log table's interim
form. One JSON line per analysis run: timestamp, ticker, mode, per-analyst
responses for BOTH passes with token usage and estimated cost, and the
consensus. Sighted-pass fair values are logged here even though they are
excluded from consensus math (anchoring-prone — see ensemble.py). Kept
append-only so it can back cost tracking and backtesting later."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ensemble import DualPassResult

RUNS_LOG_PATH = Path(__file__).resolve().parent.parent / "runs.jsonl"


def append_run(ticker: str, dual: DualPassResult) -> None:
    all_responses = dual.consensus.responses + (dual.critic.responses if dual.critic else [])
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "mode": dual.mode,
        "blind": dual.consensus.blind,
        "consensus": dual.consensus.as_log_dict(),
        "analysts": [r.as_log_dict() for r in dual.consensus.responses],
        "total_input_tokens": sum(r.input_tokens for r in all_responses),
        "total_output_tokens": sum(r.output_tokens for r in all_responses),
        "total_cost": round(dual.total_cost, 6),
    }
    if dual.critic is not None:
        record["sighted_consensus"] = dual.critic.as_log_dict()
        record["sighted_analysts"] = [r.as_log_dict() for r in dual.critic.responses]
    with open(RUNS_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
