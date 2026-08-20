"""
Group chunks into fixed-duration semantic nodes.

WorldMM builds a semantic node from every 10 consecutive chunks
(`extract_semantic_triples.DEFAULT_PERIOD`). On the 30s grid that is a 5-minute
node, but the count is only a proxy for duration and it stops being one the
moment chunk length varies: at the event condition's ~76s mean, ten chunks span
roughly 13 minutes. Comparing a 5-minute node against a 13-minute node would
confound the consolidation results outright, so nodes are built by elapsed time
instead and both conditions get genuine 5-minute windows.

Node keys follow the existing convention - the DHHMMSSFF timestamp of the last
chunk in the window - so downstream `SemanticMemory.index()` is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .timestamps import EgoTimestamp, chunk_timestamp_key

logger = logging.getLogger(__name__)

DEFAULT_PERIOD_SECONDS = 300.0


@dataclass
class SemanticNode:
    """One window of the timeline, and the chunks that fall inside it."""

    key: str                                  # DHHMMSSFF of the last member
    date: str
    start_seconds: float
    end_seconds: float
    chunk_ids: List[str] = field(default_factory=list)
    triples: List[List[str]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)


def _chunk_bounds(chunk: Dict[str, Any]) -> tuple:
    start = EgoTimestamp.from_parts(chunk["date"], chunk["start_time"]).seconds
    end = EgoTimestamp.from_parts(chunk["date"], chunk["end_time"]).seconds
    return start, max(end, start)


def build_time_nodes(
    chunks: Sequence[Dict[str, Any]],
    triples_by_chunk: Dict[str, List[List[str]]],
    *,
    period_seconds: float = DEFAULT_PERIOD_SECONDS,
) -> List[SemanticNode]:
    """
    Partition chunks into consecutive windows of `period_seconds`.

    A window closes when adding the next chunk would push it past the period,
    so a single chunk longer than the period becomes a node on its own rather
    than being split - chunk boundaries are the atom here, having already been
    chosen by the chunking strategy.

    Days never share a node: a window is closed at a day change so that an
    overnight gap cannot be folded into one node.
    """
    if period_seconds <= 0:
        raise ValueError("period_seconds must be positive")

    ordered = sorted(chunks, key=lambda c: _chunk_bounds(c)[0])
    nodes: List[SemanticNode] = []

    current: List[Dict[str, Any]] = []
    window_start = 0.0

    def close() -> None:
        if not current:
            return
        last = current[-1]
        start = _chunk_bounds(current[0])[0]
        end = _chunk_bounds(last)[1]
        node = SemanticNode(
            key=chunk_timestamp_key(last["date"], last["end_time"]),
            date=last["date"],
            start_seconds=start,
            end_seconds=end,
            chunk_ids=[c["chunk_id"] for c in current],
        )
        for chunk in current:
            node.triples.extend(triples_by_chunk.get(chunk["chunk_id"], []))
        nodes.append(node)

    for chunk in ordered:
        start, end = _chunk_bounds(chunk)
        if not current:
            current, window_start = [chunk], start
            continue
        crosses_day = chunk["date"] != current[-1]["date"]
        if crosses_day or (end - window_start) > period_seconds:
            close()
            current, window_start = [chunk], start
        else:
            current.append(chunk)
    close()

    if nodes:
        spans = [n.duration_seconds for n in nodes]
        logger.info(
            "Built %d semantic node(s) at %.0fs: mean span %.0fs, mean %.1f chunks",
            len(nodes), period_seconds, sum(spans) / len(spans),
            sum(n.n_chunks for n in nodes) / len(nodes),
        )
    return nodes
