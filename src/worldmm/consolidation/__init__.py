"""Offline memory consolidation.  [Objective 2]"""

from .canonicalise import canonical_key, dedupe_exact, is_degenerate, normalise_text
from .interval_store import ConsolidatedTriple, IntervalTripleStore
from .sleep_cycle import (
    ConsolidationReport,
    SleepCycleConfig,
    SleepCycleConsolidator,
    cumulative_baseline_size,
)

__all__ = [
    "ConsolidatedTriple",
    "ConsolidationReport",
    "IntervalTripleStore",
    "SleepCycleConfig",
    "SleepCycleConsolidator",
    "canonical_key",
    "cumulative_baseline_size",
    "dedupe_exact",
    "is_degenerate",
    "normalise_text",
]
