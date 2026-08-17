"""
Chunking strategies: fixed windows versus detected event boundaries.

Both conditions run through this one module, differing only in where the
segment edges come from:

    fixed  - edges at the pre-sliced mp4 boundaries (or every N of them),
             reproducing the native 30s grid that WorldMM and EgoLife use
    event  - edges at cut times from visual boundary detection, free to land
             anywhere on the timeline

Everything after the edges is shared: the same source entries, the same
rendering, the same chunk id scheme. So a difference in the results is a
difference in chunking, which is the whole point of the ablation.

Chunk ids are `chunk-<md5 of text>`, the same scheme OpenIE computes
internally, which makes a chunk id directly usable as a checkpoint key and as
a join key against openie output.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..common.mapping import chunk_key
from ..common.timestamps import EgoTimestamp
from .config import EVENT, FIXED, ChunkingConfig
from .sources import SourceEntry, render_entries

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One unit of memory: a span of timeline, its text, and its provenance."""

    chunk_id: str
    date: str
    start_time: str                       # HHMMSSFF
    end_time: str                         # HHMMSSFF
    text: str
    video_path: str                       # first constituent, for compatibility
    video_paths: List[str] = field(default_factory=list)
    n_source_entries: int = 0
    start_seconds: float = 0.0
    end_seconds: float = 0.0

    @property
    def timestamp_key(self) -> str:
        """The DHHMMSSFF key downstream stages index by (built from end time)."""
        return EgoTimestamp.from_parts(self.date, self.end_time).key

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def spans_multiple_videos(self) -> bool:
        return len(self.video_paths) > 1

    def to_caption_entry(self) -> Dict[str, Any]:
        """
        The dict shape the existing pipeline consumes.

        `video_path` stays a single string so `load_clips_from_data` and
        `extract_episodic_triples` keep working unchanged; `video_paths`
        carries the full list for the visual encoder, which samples frames
        across every file an event touches.
        """
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "date": self.date,
            "text": self.text,
            "video_path": self.video_path,
            "video_paths": list(self.video_paths),
            "chunk_id": self.chunk_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _video_path_for(video_file: str, date: str, person: str) -> str:
    """Mirror `generate_fine_caption_egolife.create_video_path`."""
    return f"data/EgoLife/{person}/{date}/{video_file}" if video_file else ""


def _build_chunk(
    entries: Sequence[SourceEntry],
    person: str,
    *,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
) -> Optional[Chunk]:
    """Assemble one chunk from a run of source entries, or None if it is empty."""
    if not entries:
        return None

    text = render_entries(entries, person)
    if not text.strip():
        return None

    start = start_seconds if start_seconds is not None else entries[0].start_seconds
    end = end_seconds if end_seconds is not None else entries[-1].end_seconds
    if end <= start:
        end = entries[-1].end_seconds

    start_ts = EgoTimestamp.from_seconds(start)
    end_ts = EgoTimestamp.from_seconds(end)
    date = f"DAY{entries[0].day}"

    # Distinct source files in the order the event touched them.
    video_files = list(dict.fromkeys(e.video_file for e in entries if e.video_file))
    video_paths = [_video_path_for(f, date, person) for f in video_files]

    return Chunk(
        chunk_id=chunk_key(text),
        date=date,
        start_time=start_ts.time_str,
        end_time=end_ts.time_str,
        text=text,
        video_path=video_paths[0] if video_paths else "",
        video_paths=video_paths,
        n_source_entries=len(entries),
        start_seconds=start,
        end_seconds=end,
    )


class ChunkingStrategy(ABC):
    """Turns source entries into chunks."""

    name: str

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    @abstractmethod
    def segment(self, entries: Sequence[SourceEntry], **kwargs) -> List[Chunk]:
        ...


class FixedWindowStrategy(ChunkingStrategy):
    """
    The baseline: one chunk per pre-sliced mp4, or per N of them.

    EgoLife ships ~30s segments and WorldMM indexes one chunk per segment, so
    grouping by source file reproduces the native grid exactly rather than
    imposing a wall-clock window that would drift against it. `window_seconds`
    is expressed against that 30s native segment, so 30 gives one file per
    chunk and 60 gives two.
    """

    name = FIXED

    def segment(self, entries: Sequence[SourceEntry], **kwargs) -> List[Chunk]:
        files_per_chunk = max(1, int(round(self.config.window_seconds / 30.0)))

        chunks: List[Chunk] = []
        current: List[SourceEntry] = []
        current_files: List[str] = []

        for entry in entries:
            if entry.video_file not in current_files:
                if len(current_files) >= files_per_chunk:
                    chunk = _build_chunk(current, self.config.person)
                    if chunk:
                        chunks.append(chunk)
                    current, current_files = [], []
                current_files.append(entry.video_file)
            current.append(entry)

        chunk = _build_chunk(current, self.config.person)
        if chunk:
            chunks.append(chunk)

        logger.info("Fixed windows (%d file(s) each): %d chunks", files_per_chunk, len(chunks))
        return chunks


class EventBoundaryStrategy(ChunkingStrategy):
    """
    Objective 1: segment on detected semantic state change.

    Cut times come from visual boundary detection and can land anywhere, so a
    chunk routinely spans several mp4 files and never splits an action at an
    arbitrary 30s tick. Entries are assigned by midpoint, so one straddling a
    cut lands on the side where most of it occurred.
    """

    name = EVENT

    def segment(
        self,
        entries: Sequence[SourceEntry],
        *,
        cuts: Sequence[float] = (),
        **kwargs,
    ) -> List[Chunk]:
        if not entries:
            return []

        edges = self._edges(entries, cuts)
        buckets: List[List[SourceEntry]] = [[] for _ in range(len(edges) - 1)]

        index = 0
        for entry in sorted(entries, key=lambda e: e.midpoint):
            while index + 1 < len(buckets) and entry.midpoint >= edges[index + 1]:
                index += 1
            buckets[index].append(entry)

        chunks: List[Chunk] = []
        for i, bucket in enumerate(buckets):
            chunk = _build_chunk(
                bucket, self.config.person,
                start_seconds=edges[i], end_seconds=edges[i + 1],
            )
            if chunk:
                chunks.append(chunk)

        empty = len(buckets) - len(chunks)
        logger.info(
            "Event boundaries: %d chunks from %d segment(s)%s",
            len(chunks), len(buckets),
            f" ({empty} had no text and were dropped)" if empty else "",
        )
        return chunks

    @staticmethod
    def _edges(entries: Sequence[SourceEntry], cuts: Sequence[float]) -> List[float]:
        start = min(e.start_seconds for e in entries)
        end = max(e.end_seconds for e in entries)
        inside = sorted({c for c in cuts if start < c < end})
        return [start, *inside, end]


def build_strategy(config: ChunkingConfig) -> ChunkingStrategy:
    """Instantiate the strategy named in the config."""
    if config.strategy == FIXED:
        return FixedWindowStrategy(config)
    if config.strategy == EVENT:
        return EventBoundaryStrategy(config)
    raise ValueError(f"unknown strategy: {config.strategy!r}")


def chunk_statistics(chunks: Sequence[Chunk]) -> Dict[str, Any]:
    """Distribution summary for the Obj 3 chunk-length comparison."""
    if not chunks:
        return {"n_chunks": 0}

    import statistics

    lengths = sorted(c.duration_seconds for c in chunks)
    texts = [len(c.text) for c in chunks]

    def pct(values: List[float], p: float) -> float:
        return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]

    return {
        "n_chunks": len(chunks),
        "total_seconds": sum(lengths),
        "mean_seconds": statistics.fmean(lengths),
        "median_seconds": statistics.median(lengths),
        "p95_seconds": pct(lengths, 95),
        "min_seconds": lengths[0],
        "max_seconds": lengths[-1],
        "mean_text_chars": statistics.fmean(texts),
        "median_text_chars": statistics.median(texts),
        "max_text_chars": max(texts),
        "multi_video_chunks": sum(1 for c in chunks if c.spans_multiple_videos),
        "duplicate_chunk_ids": len(chunks) - len({c.chunk_id for c in chunks}),
    }
