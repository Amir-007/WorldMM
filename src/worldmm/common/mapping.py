"""
The openie <-> timestamp mapping, rebuilt properly.

`openie_results_*.json` is keyed by a content hash of the caption text
(`chunk-<md5>`), not by timestamp. Counts happen to match at 6,223 and
insertion order happens to line up, but joining on dict position is fragile
and would fail silently and late. This module reconstructs the real mapping
from the caption file instead.

The hash is `compute_mdhash_id` from `worldmm.memory.episodic.utils`, applied
to the caption `text`. It is reimplemented here as a three-line function so
this module stays importable without torch/igraph; `tests/test_stage0.py`
pins it against the real inherited artifacts rather than against the original
function, which is the stronger check.

Verified on A1_JAKE / qwen3vl-30b: all 6,223 timestamps reproduce their
episodic triples exactly, with zero hash misses and zero collisions.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .timestamps import EgoTimestamp

logger = logging.getLogger(__name__)

# DAY6 of A1_JAKE is excluded by default. The caption file shipped in
# data/EgoLife/caption.zip predates upstream commit 3a55b65 ("fix date handling
# in memory generation"), and in that pre-fix release every DAY6 caption is
# rotated by 21 slots (~10.5 minutes) against its video segment and timestamp.
# DAY1-5 and DAY7 are byte-identical to the fixed release. Excluding DAY6 keeps
# both experimental conditions on clean data at the cost of 9 of 52 hours;
# repairing it instead means rebuilding DAY6 openie plus all of consolidation.
DEFAULT_EXCLUDED_DAYS: Tuple[int, ...] = (6,)


def chunk_key(text: str) -> str:
    """
    The OpenIE chunk id for a caption text.

    Mirrors `worldmm.memory.episodic.utils.compute_mdhash_id(text, prefix="chunk-")`.
    """
    return "chunk-" + md5(text.encode()).hexdigest()


@dataclass(frozen=True)
class CaptionChunk:
    """One caption entry, with both of its identities attached."""

    index: int
    timestamp_key: str
    chunk_key: str
    text: str
    date: str
    start_time: str
    end_time: str
    video_path: str

    @property
    def day(self) -> int:
        return int(self.timestamp_key[0])

    @property
    def start(self) -> EgoTimestamp:
        return EgoTimestamp.from_parts(self.date, self.start_time)

    @property
    def end(self) -> EgoTimestamp:
        return EgoTimestamp.from_parts(self.date, self.end_time)

    @property
    def duration_seconds(self) -> float:
        return self.end.seconds - self.start.seconds


@dataclass
class VerificationReport:
    """Outcome of checking a rebuilt mapping against the inherited artifacts."""

    total: int = 0
    hash_hits: int = 0
    hash_misses: List[str] = field(default_factory=list)
    collisions: Dict[str, List[str]] = field(default_factory=dict)
    triples_reproduced: int = 0
    triples_mismatched: List[str] = field(default_factory=list)
    unclaimed_openie_keys: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.total > 0
            and not self.hash_misses
            and not self.collisions
            and not self.triples_mismatched
            and not self.unclaimed_openie_keys
        )

    def summary(self) -> str:
        lines = [
            f"caption chunks:            {self.total}",
            f"hashes found in openie:    {self.hash_hits}/{self.total}",
            f"hash misses:               {len(self.hash_misses)}",
            f"chunk_key collisions:      {len(self.collisions)}",
            f"openie keys unclaimed:     {len(self.unclaimed_openie_keys)}",
        ]
        if self.triples_reproduced or self.triples_mismatched:
            lines.append(
                f"episodic triples reproduced: {self.triples_reproduced}/{self.total} "
                f"(mismatched={len(self.triples_mismatched)})"
            )
        lines.append(f"VERDICT: {'OK' if self.ok else 'FAILED'}")
        return "\n".join(lines)


def _normalise_excluded_days(days: Optional[Iterable[Any]]) -> Set[int]:
    if days is None:
        return set()
    out: Set[int] = set()
    for day in days:
        if isinstance(day, int):
            out.add(day)
        else:
            digits = "".join(ch for ch in str(day) if ch.isdigit())
            if not digits:
                raise ValueError(f"cannot read a day number from {day!r}")
            out.add(int(digits))
    return out


def load_caption_entries(path: str) -> List[Dict[str, Any]]:
    """Load raw caption entries from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_caption_entries_from_zip(zip_path: str, person: str) -> List[Dict[str, Any]]:
    """
    Load raw caption entries straight out of `data/EgoLife/caption.zip`.

    The repo ships captions as a zip, and this is the archive whose texts key
    the inherited openie results, so reading it directly avoids depending on a
    manual extraction step.
    """
    member = f"data/EgoLife/EgoLifeCap/{person}/{person}_30sec.json"
    with zipfile.ZipFile(zip_path) as archive:
        try:
            with archive.open(member) as f:
                return json.loads(f.read().decode("utf-8"))
        except KeyError as exc:
            available = [n for n in archive.namelist() if n.endswith("_30sec.json")]
            raise FileNotFoundError(
                f"{member} not in {zip_path}. Available: {available}"
            ) from exc


def build_caption_chunks(
    entries: Sequence[Dict[str, Any]],
    *,
    exclude_days: Optional[Iterable[Any]] = DEFAULT_EXCLUDED_DAYS,
) -> List[CaptionChunk]:
    """
    Turn raw caption entries into `CaptionChunk`s carrying both identities.

    `index` is the position in the *original* file, preserved across exclusion
    so it still lines up with the inherited artifacts.
    """
    excluded = _normalise_excluded_days(exclude_days)
    chunks: List[CaptionChunk] = []
    dropped = 0

    for index, entry in enumerate(entries):
        try:
            timestamp = EgoTimestamp.from_parts(entry["date"], entry["end_time"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"caption entry {index} has an unusable timestamp: {exc}") from exc

        if timestamp.day in excluded:
            dropped += 1
            continue

        chunks.append(
            CaptionChunk(
                index=index,
                timestamp_key=timestamp.key,
                chunk_key=chunk_key(entry["text"]),
                text=entry["text"],
                date=entry["date"],
                start_time=entry["start_time"],
                end_time=entry["end_time"],
                video_path=entry.get("video_path", ""),
            )
        )

    if dropped:
        logger.info(
            "Excluded %d caption chunk(s) from day(s) %s; %d remain",
            dropped, sorted(excluded), len(chunks),
        )
    return chunks


def load_caption_chunks(
    path: str,
    *,
    person: Optional[str] = None,
    exclude_days: Optional[Iterable[Any]] = DEFAULT_EXCLUDED_DAYS,
) -> List[CaptionChunk]:
    """Load caption chunks from a JSON file or from `caption.zip`."""
    if zipfile.is_zipfile(path):
        if not person:
            raise ValueError("person is required when loading captions from a zip")
        entries = load_caption_entries_from_zip(path, person)
    else:
        entries = load_caption_entries(path)
    return build_caption_chunks(entries, exclude_days=exclude_days)


def timestamp_to_chunk_key(chunks: Sequence[CaptionChunk]) -> Dict[str, str]:
    """Mapping from timestamp key to OpenIE chunk id."""
    return {c.timestamp_key: c.chunk_key for c in chunks}


def chunk_key_to_timestamp(chunks: Sequence[CaptionChunk]) -> Dict[str, str]:
    """
    Mapping from OpenIE chunk id back to timestamp key.

    Identical caption texts collapse to one chunk id, so this direction can
    lose entries. `find_collisions` reports when that happens.
    """
    return {c.chunk_key: c.timestamp_key for c in chunks}


def find_collisions(chunks: Sequence[CaptionChunk]) -> Dict[str, List[str]]:
    """Chunk ids claimed by more than one timestamp (i.e. repeated caption text)."""
    by_key: Dict[str, List[str]] = {}
    for chunk in chunks:
        by_key.setdefault(chunk.chunk_key, []).append(chunk.timestamp_key)
    return {k: v for k, v in by_key.items() if len(v) > 1}


def triples_by_timestamp(
    chunks: Sequence[CaptionChunk],
    openie_data: Dict[str, Any],
    *,
    field_name: str = "triple_results",
) -> Dict[str, List[List[str]]]:
    """Re-key OpenIE results from content hash to timestamp."""
    results = openie_data[field_name]
    return {c.timestamp_key: results[c.chunk_key] for c in chunks if c.chunk_key in results}


def verify(
    chunks: Sequence[CaptionChunk],
    openie_data: Dict[str, Any],
    episodic_data: Optional[Dict[str, Any]] = None,
    *,
    expect_full_coverage: bool = True,
) -> VerificationReport:
    """
    Check a rebuilt mapping against the inherited artifacts.

    Args:
        chunks: Caption chunks to verify.
        openie_data: Parsed `openie_results_*.json`.
        episodic_data: Parsed `episodic_triple_results_*.json`. When given, the
            mapping must reproduce its triples exactly, which is the real test.
        expect_full_coverage: Whether every openie key should be claimed by a
            chunk. False when days have been excluded.
    """
    report = VerificationReport(total=len(chunks))
    triple_results = openie_data.get("triple_results", {})

    report.collisions = find_collisions(chunks)

    for chunk in chunks:
        if chunk.chunk_key in triple_results:
            report.hash_hits += 1
        else:
            report.hash_misses.append(chunk.timestamp_key)

    if expect_full_coverage:
        claimed = {c.chunk_key for c in chunks}
        report.unclaimed_openie_keys = [k for k in triple_results if k not in claimed]

    if episodic_data is not None:
        episodic_triples = episodic_data["episodic_triples"]
        for chunk in chunks:
            if chunk.chunk_key not in triple_results:
                continue
            expected = episodic_triples.get(chunk.timestamp_key)
            if expected == triple_results[chunk.chunk_key]:
                report.triples_reproduced += 1
            else:
                report.triples_mismatched.append(chunk.timestamp_key)

    return report


def normalise_excluded_days(days: Optional[Iterable[Any]]) -> Set[int]:
    """Public wrapper for day-list normalisation, used when reporting config."""
    return _normalise_excluded_days(days)
