"""The AI analyst ensemble: roster loading, fan-out, and consensus math.

Core principle: disagreement is signal, not noise. The ensemble reports a
weighted consensus range AND a disagreement index (how far the analysts'
midpoints spread relative to the consensus midpoint). High disagreement
lowers overall confidence and is displayed prominently — it is never hidden
by the averaging.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from src.analyst_adapters import ADAPTERS
from src.analyst_base import AnalystConfig, AnalystResponse
from src.models import FairValueResult

ANALYSTS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "analysts.yaml"

RULE_BASED_NAME = "rule-based-engine"

# Overall confidence = weighted avg analyst confidence x this disagreement
# penalty floor. A disagreement index of 0 leaves confidence untouched; an
# index of 0.75+ (midpoints spreading 75% of the consensus mid) bottoms out
# at the floor rather than zeroing confidence entirely.
DISAGREEMENT_PENALTY_FLOOR = 0.25


@dataclass
class EnsembleResult:
    responses: list[AnalystResponse] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # human-readable skip/warn messages

    consensus_low: Optional[float] = None
    consensus_mid: Optional[float] = None
    consensus_high: Optional[float] = None
    disagreement_index: Optional[float] = None
    overall_confidence: Optional[float] = None

    archetype_disputes: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    contributing_analysts: int = 0

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "consensus_low": self.consensus_low,
            "consensus_mid": self.consensus_mid,
            "consensus_high": self.consensus_high,
            "disagreement_index": self.disagreement_index,
            "overall_confidence": self.overall_confidence,
            "archetype_disputes": self.archetype_disputes,
            "contributing_analysts": self.contributing_analysts,
            "skipped": self.skipped,
            "total_cost": round(self.total_cost, 6),
        }


@lru_cache(maxsize=1)
def load_analyst_config() -> dict[str, Any]:
    with open(ANALYSTS_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_roster(only_names: Optional[list[str]] = None) -> tuple[list[tuple[AnalystConfig, str]], list[str]]:
    """Returns ([(config, api_key)] for runnable analysts, [skip warnings]).

    An analyst is runnable when it's active (or explicitly requested via
    --analysts), its provider type is known, and its API key env var is set.
    Anything else is skipped with a warning — never an exception.
    """
    load_dotenv()
    raw = load_analyst_config()
    runnable: list[tuple[AnalystConfig, str]] = []
    skipped: list[str] = []

    known_names = set()
    for entry in raw.get("analysts", []):
        cfg = AnalystConfig(
            name=entry["name"],
            provider_type=entry["provider_type"],
            model=entry["model"],
            api_key_env=entry.get("api_key_env"),
            base_url=entry.get("base_url"),
            active=entry.get("active", True),
            weight=float(entry.get("weight", 1.0)),
            cost_per_mtok_input=float(entry.get("cost_per_mtok_input", 0.0)),
            cost_per_mtok_output=float(entry.get("cost_per_mtok_output", 0.0)),
        )
        known_names.add(cfg.name)

        if only_names is not None:
            if cfg.name not in only_names:
                continue  # not requested — silent
        elif not cfg.active:
            continue  # inactive and not explicitly requested — silent

        if cfg.provider_type not in ADAPTERS:
            skipped.append(f"{cfg.name}: unknown provider_type '{cfg.provider_type}' — skipped.")
            continue
        api_key = os.environ.get(cfg.api_key_env or "", "")
        if not api_key:
            skipped.append(
                f"{cfg.name}: API key env var '{cfg.api_key_env}' is not set (check .env) — skipped."
            )
            continue
        runnable.append((cfg, api_key))

    if only_names is not None:
        for name in only_names:
            if name not in known_names:
                skipped.append(f"'{name}': no analyst with this name in config/analysts.yaml — skipped.")

    return runnable, skipped


def _rule_based_response(fair_value: FairValueResult, weight: float) -> Optional[AnalystResponse]:
    if fair_value.low is None or fair_value.high is None:
        return None
    return AnalystResponse(
        analyst_name=RULE_BASED_NAME,
        ok=True,
        fair_value_low=fair_value.low,
        fair_value_high=fair_value.high,
        confidence=fair_value.confidence,
        archetype_agreement=True,  # it produced the archetype, by definition
        one_paragraph_thesis=f"Rule-based estimate via: {fair_value.method}",
        weight=weight,
    )


def _consensus(result: EnsembleResult, contributors: list[AnalystResponse]) -> None:
    """Weighted consensus + disagreement index over analysts that returned a
    usable range. Effective weight = configured weight x analyst confidence,
    so a shaky opinion moves the consensus less but still shows in the spread."""
    weighted = [
        (r, max(r.weight * r.confidence, 1e-6))
        for r in contributors
        if r.midpoint is not None
    ]
    if not weighted:
        return

    total_w = sum(w for _, w in weighted)
    result.consensus_low = round(sum(r.fair_value_low * w for r, w in weighted) / total_w, 2)
    result.consensus_high = round(sum(r.fair_value_high * w for r, w in weighted) / total_w, 2)
    result.consensus_mid = round(sum(r.midpoint * w for r, w in weighted) / total_w, 2)
    result.contributing_analysts = len(weighted)

    # Disagreement index: weighted std dev of midpoints / |consensus midpoint|
    # (a coefficient of variation). 0 = full agreement; 0.30 means midpoints
    # spread ~30% around the consensus.
    if len(weighted) >= 2 and result.consensus_mid:
        mean_mid = sum(r.midpoint * w for r, w in weighted) / total_w
        variance = sum(w * (r.midpoint - mean_mid) ** 2 for r, w in weighted) / total_w
        result.disagreement_index = round(math.sqrt(variance) / abs(result.consensus_mid), 3)
    else:
        # A single opinion can't disagree with itself — report n/a, not 0.
        result.disagreement_index = None

    avg_conf = sum(r.confidence * w for r, w in weighted) / total_w
    di = result.disagreement_index or 0.0
    penalty = max(DISAGREEMENT_PENALTY_FLOOR, 1.0 - di)
    result.overall_confidence = round(min(1.0, avg_conf * penalty), 2)


def run_ensemble(
    briefing_packet: dict[str, Any],
    fair_value: FairValueResult,
    only_names: Optional[list[str]] = None,
) -> EnsembleResult:
    result = EnsembleResult()
    roster, skipped = load_roster(only_names)
    result.skipped = skipped

    adapters = [ADAPTERS[cfg.provider_type](cfg, key) for cfg, key in roster]
    if adapters:
        with ThreadPoolExecutor(max_workers=min(len(adapters), 6)) as pool:
            result.responses = list(pool.map(lambda a: a.analyze(briefing_packet), adapters))

    result.total_cost = sum(r.estimated_cost for r in result.responses)

    contributors = [r for r in result.responses if r.ok]

    rule_based = _rule_based_response(
        fair_value, float(load_analyst_config().get("rule_based_weight", 1.0))
    )
    if rule_based is not None:
        contributors.append(rule_based)
        result.responses.append(rule_based)

    briefing_archetype = (briefing_packet.get("archetype") or {}).get("name")
    for r in contributors:
        if r.archetype_agreement is False:
            suggestion = r.suggested_archetype or "unspecified"
            result.archetype_disputes.append(
                f"{r.analyst_name} disputes '{briefing_archetype}' — suggests '{suggestion}'"
            )

    _consensus(result, contributors)
    return result
