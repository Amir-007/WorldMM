"""
The finest-grained text available: EgoLife Sync entries.

`generate_sync.py` merges translated DenseCaption lines with Transcript
subtitles into per-hour files under `EgoLifeCap/Sync/`, each a list of
`{"video_file", "data": [{"start", "end", "text", "type"}]}` where start/end
are HHMMSS integers at roughly 2-second granularity.

This is the source both experimental conditions are built from. Using it
rather than the 30s fused captions is what lets event boundaries land where
detection actually put them: grouping pre-fused 30s captions would quantise
every boundary to the 30s grid, costing +/-15s, which is longer than many of
the actions Objective 1 claims not to split.

Because both conditions assemble their text from these same entries with the
same code, the ablation isolates the chunking strategy and introduces no
captioner difference.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ..common.timestamps import SECONDS_PER_DAY

logger = logging.getLogger(__name__)

CAPTION = "caption"
TRANSCRIPT = "transcript"

_SYNC_NAME_RE = re.compile(r"_DAY(\d+)_", re.IGNORECASE)


@dataclass(frozen=True)
class SourceEntry:
    """One dense-caption line or transcript subtitle."""

    day: int
    start_seconds: float          # absolute, from the start of day 1
    end_seconds: float
    text: str
    type: str
    video_file: str

    @property
    def midpoint(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def _hhmmss_to_seconds(value: object, day: int) -> float:
    """
    Convert an HHMMSS integer to absolute seconds.

    Transcript timestamps are built by adding the file's hour to the subtitle
    offset, so an hour field of 24 or more is legitimate near midnight and
    rolls into the following day rather than being an error.
    """
    digits = str(int(value)).zfill(6)
    hours, minutes, seconds = int(digits[:-4]), int(digits[-4:-2]), int(digits[-2:])
    if minutes > 59 or seconds > 59:
        raise ValueError(f"malformed HHMMSS value {value!r}")
    return day * SECONDS_PER_DAY + hours * 3600 + minutes * 60 + seconds


def _day_from_filename(path: str) -> int:
    match = _SYNC_NAME_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"cannot read a DAY from sync filename: {path!r}")
    return int(match.group(1))


def load_sync_entries(
    sync_dir: str,
    person: str,
    *,
    days: Optional[Iterable[int]] = None,
    exclude_days: Sequence[int] = (),
) -> List[SourceEntry]:
    """
    Load every Sync entry for a person, ordered by time.

    Args:
        sync_dir: Directory of `<person>_DAY<n>_<HHMMSSFF>.json` files.
        days: Restrict to these days. None means all.
        exclude_days: Days to drop (the DAY6 exclusion lives here).
    """
    paths = sorted(glob.glob(os.path.join(sync_dir, f"{person}_*.json")))
    if not paths:
        raise FileNotFoundError(f"no sync files for {person} in {sync_dir}")

    wanted: Optional[Set[int]] = set(days) if days is not None else None
    excluded = set(exclude_days)

    entries: List[SourceEntry] = []
    skipped_files = 0
    malformed = 0

    for path in paths:
        day = _day_from_filename(path)
        if day in excluded or (wanted is not None and day not in wanted):
            skipped_files += 1
            continue

        with open(path, "r", encoding="utf-8") as f:
            blocks = json.load(f)

        for block in blocks:
            video_file = block.get("video_file", "")
            for item in block.get("data", []):
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                try:
                    start = _hhmmss_to_seconds(item["start"], day)
                    end = _hhmmss_to_seconds(item["end"], day)
                except (KeyError, TypeError, ValueError):
                    malformed += 1
                    continue
                if end < start:
                    # A subtitle straddling midnight; the end belongs to the
                    # following day.
                    end += SECONDS_PER_DAY
                entries.append(SourceEntry(
                    day=day,
                    start_seconds=start,
                    end_seconds=end,
                    text=text,
                    type=item.get("type", CAPTION),
                    video_file=video_file,
                ))

    entries.sort(key=lambda e: (e.start_seconds, e.end_seconds))
    logger.info(
        "Loaded %d sync entries for %s across %d file(s)%s%s",
        len(entries), person, len(paths) - skipped_files,
        f", skipped {skipped_files} file(s)" if skipped_files else "",
        f", dropped {malformed} malformed entr(ies)" if malformed else "",
    )
    return entries


def speaker_name(person: str) -> str:
    """`A1_JAKE` -> `Jake`, matching how transcripts name the camera wearer."""
    _, _, name = person.partition("_")
    return (name or person).replace("_", " ").title()


def normalize_first_person(text: str, person: str) -> str:
    """
    Rewrite the camera wearer's own name into first person.

    Mirrors `generate_fine_caption_egolife.normalize_camera_wearer_text`, which
    cannot be imported here because that module loads an LLM at import time.
    Applied so concatenated source text reads in the same voice as the shipped
    30s captions, keeping the two conditions stylistically comparable.
    """
    name = re.escape(speaker_name(person))
    text = re.sub(rf"(?<!\w){name}:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?<!\w){name}'s(?!\w)", "my", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?<!\w){name}(?!\w)", "I", text, flags=re.IGNORECASE)
    return text


def render_entries(entries: Sequence[SourceEntry], person: str) -> str:
    """
    Turn a run of source entries into one chunk's text.

    Deterministic and model-free: entries are joined in time order and the
    camera wearer is rewritten to first person. Identical for both conditions,
    so any difference downstream comes from which entries were grouped, not
    from how they were rendered.
    """
    parts: List[str] = []
    seen: Set[str] = set()
    for entry in entries:
        rendered = normalize_first_person(entry.text, person).strip()
        if not rendered:
            continue
        # Dense captions and transcripts overlap and repeat verbatim more often
        # than not; keeping duplicates would inflate longer chunks specifically.
        if rendered in seen:
            continue
        seen.add(rendered)
        parts.append(rendered)
    return " ".join(parts)


def entries_by_video(entries: Sequence[SourceEntry]) -> Dict[str, List[SourceEntry]]:
    """Group entries by their source video file, preserving time order."""
    grouped: Dict[str, List[SourceEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.video_file, []).append(entry)
    return grouped
