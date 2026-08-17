"""
EgoLife DHHMMSSFF timestamp handling.

Timestamp keys look like `111095800` = day 1, 11:09:58, frame 00.

These must never be diffed as plain integers. The seconds and minutes fields
wrap at 60, so a raw integer delta of 3000 across a minute boundary is 30
seconds of wall-clock, not 3000 of anything. Parse to `EgoTimestamp` (or
straight to `to_seconds`) before any arithmetic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

# EgoLife capture rate, used to interpret the trailing frame field. Every
# timestamp in the shipped caption files uses FF=00, so this only matters once
# boundary detection starts emitting sub-second cut points.
FRAMES_PER_SECOND = 30

SECONDS_PER_DAY = 86400

_KEY_RE = re.compile(r"^(\d)(\d{2})(\d{2})(\d{2})(\d{2})$")
_TIME_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})(\d{2})$")

TimestampLike = Union[str, int, "EgoTimestamp"]


@dataclass(frozen=True, order=True)
class EgoTimestamp:
    """A single point on the EgoLife recording timeline."""

    day: int
    hour: int
    minute: int
    second: int
    frame: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.day <= 9:
            raise ValueError(f"day out of range (EgoLife keys hold one digit): {self.day}")
        if not 0 <= self.hour <= 23:
            raise ValueError(f"hour out of range: {self.hour}")
        if not 0 <= self.minute <= 59:
            raise ValueError(f"minute out of range: {self.minute}")
        if not 0 <= self.second <= 59:
            raise ValueError(f"second out of range: {self.second}")
        if not 0 <= self.frame < FRAMES_PER_SECOND:
            raise ValueError(f"frame out of range: {self.frame}")

    @property
    def key(self) -> str:
        """The canonical 9-character DHHMMSSFF key."""
        return f"{self.day}{self.hour:02d}{self.minute:02d}{self.second:02d}{self.frame:02d}"

    @property
    def time_str(self) -> str:
        """The 8-character HHMMSSFF form used by caption `start_time`/`end_time`."""
        return f"{self.hour:02d}{self.minute:02d}{self.second:02d}{self.frame:02d}"

    @property
    def date_str(self) -> str:
        """The `DAYn` form used by the caption `date` field."""
        return f"DAY{self.day}"

    @property
    def seconds(self) -> float:
        """Absolute seconds from the start of day 1."""
        return (
            self.day * SECONDS_PER_DAY
            + self.hour * 3600
            + self.minute * 60
            + self.second
            + self.frame / FRAMES_PER_SECOND
        )

    @classmethod
    def from_key(cls, key: TimestampLike) -> "EgoTimestamp":
        if isinstance(key, EgoTimestamp):
            return key
        text = str(key).strip()
        if len(text) == 8:
            # Bare HHMMSSFF with no day digit is ambiguous; callers must supply one.
            raise ValueError(
                f"{text!r} is an 8-digit HHMMSSFF time with no day digit. "
                "Use from_parts(date, time_str) instead."
            )
        text = text.zfill(9)
        match = _KEY_RE.match(text)
        if not match:
            raise ValueError(f"not a DHHMMSSFF timestamp key: {key!r}")
        day, hour, minute, second, frame = (int(g) for g in match.groups())
        return cls(day=day, hour=hour, minute=minute, second=second, frame=frame)

    @classmethod
    def from_parts(cls, date: str, time_str: str) -> "EgoTimestamp":
        """Build from a caption entry's `date` ("DAY1") and `start_time`/`end_time`."""
        day_match = re.search(r"(\d+)", date)
        if not day_match:
            raise ValueError(f"cannot read a day number from date {date!r}")
        match = _TIME_RE.match(str(time_str).zfill(8))
        if not match:
            raise ValueError(f"not an HHMMSSFF time: {time_str!r}")
        hour, minute, second, frame = (int(g) for g in match.groups())
        return cls(day=int(day_match.group(1)), hour=hour, minute=minute,
                   second=second, frame=frame)

    @classmethod
    def from_seconds(cls, total: float) -> "EgoTimestamp":
        """Inverse of `.seconds`."""
        if total < 0:
            raise ValueError(f"negative absolute time: {total}")
        whole = int(total)
        frame = int(round((total - whole) * FRAMES_PER_SECOND))
        if frame >= FRAMES_PER_SECOND:  # rounded up past a second boundary
            whole += 1
            frame = 0
        day, rem = divmod(whole, SECONDS_PER_DAY)
        hour, rem = divmod(rem, 3600)
        minute, second = divmod(rem, 60)
        return cls(day=day, hour=hour, minute=minute, second=second, frame=frame)

    def __str__(self) -> str:
        return f"DAY{self.day} {self.hour:02d}:{self.minute:02d}:{self.second:02d}"


def parse(key: TimestampLike) -> EgoTimestamp:
    """Parse a DHHMMSSFF key into an `EgoTimestamp`."""
    return EgoTimestamp.from_key(key)


def to_seconds(key: TimestampLike) -> float:
    """Absolute seconds from the start of day 1."""
    return EgoTimestamp.from_key(key).seconds


def duration_seconds(start: TimestampLike, end: TimestampLike) -> float:
    """Wall-clock seconds between two timestamps. Never subtract the keys directly."""
    return to_seconds(end) - to_seconds(start)


def day_of(key: TimestampLike) -> int:
    """The day number a timestamp key belongs to."""
    return EgoTimestamp.from_key(key).day


def chunk_timestamp_key(date: str, end_time: str) -> str:
    """
    Build the episodic-memory timestamp key for a caption entry.

    Mirrors `extract_episodic_triples.create_episodic_triples_results` exactly
    (`date[-1] + end_time.zfill(8)`), so keys stay byte-identical to the
    inherited artifacts. Going through `EgoTimestamp` also validates the fields,
    which the original does not.
    """
    return EgoTimestamp.from_parts(date, end_time).key


def format_key(key: TimestampLike) -> str:
    """Human-readable rendering, e.g. `DAY1 11:09:58`."""
    return str(EgoTimestamp.from_key(key))
