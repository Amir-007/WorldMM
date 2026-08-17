"""
Chunking configuration.

Both experimental conditions run from the same code path, so the strategy and
its parameters live in a config file rather than in the call site. The resolved
config is written into every artifact this pipeline produces, so any chunk file
can be traced back to the settings that made it - which the methodology chapter
needs, and which makes a threshold sweep self-documenting.

JSON rather than YAML: pyyaml is not a declared dependency of this project, and
the config is small enough that the extra syntax buys nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Tuple

FIXED = "fixed"
EVENT = "event"


@dataclass
class ChunkingConfig:
    """Everything that determines how a video stream becomes chunks."""

    # --- which strategy ----------------------------------------------------
    strategy: str = FIXED
    person: str = "A1_JAKE"
    exclude_days: Tuple[int, ...] = (6,)

    # --- fixed-window strategy --------------------------------------------
    # 30s reproduces the inherited grid exactly and is the baseline condition.
    window_seconds: float = 30.0

    # --- event strategy: frame features -----------------------------------
    # 2 fps over 52 h is ~374k frames. Cheap on CPU, and cached so that a
    # threshold sweep never touches video again.
    sample_fps: float = 2.0
    hist_bins: Tuple[int, int, int] = (8, 8, 8)
    gray_size: int = 32

    # --- event strategy: boundary detection -------------------------------
    # Colour histogram catches scene/lighting change; downsampled grayscale
    # catches layout change. Neither alone is reliable in egocentric video.
    hist_weight: float = 0.5
    # Threshold is a robust z-score (median/MAD), not a raw distance, so it
    # transfers across days with different lighting without retuning.
    threshold: float = 3.0
    # A z-score is scale-free, so in near-static video it will always find
    # "outliers" in sensor noise. A candidate must also clear this absolute
    # distance. Both component distances are bounded in [0, 1]: noise sits
    # below ~0.01 and a genuine scene change is an order of magnitude above,
    # so this is a noise gate, not the sensitivity knob.
    min_absolute_distance: float = 0.02
    rolling_window_seconds: float = 60.0
    min_event_seconds: float = 10.0
    # 60s cap: at ~3x the 30s text length, OpenIE triple output lands near the
    # 512-token generation cap, beyond which output truncates and the fork's
    # parser silently degrades it to an empty result.
    max_event_seconds: float = 60.0
    # Larger than this between consecutive sampled frames means unobserved
    # video, which forces a boundary rather than being read as a scene change.
    gap_tolerance_seconds: float = 2.0

    def __post_init__(self) -> None:
        self.exclude_days = tuple(int(d) for d in self.exclude_days)
        self.hist_bins = tuple(int(b) for b in self.hist_bins)
        self.validate()

    def validate(self) -> None:
        if self.strategy not in (FIXED, EVENT):
            raise ValueError(f"strategy must be {FIXED!r} or {EVENT!r}, got {self.strategy!r}")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if len(self.hist_bins) != 3 or any(b < 1 for b in self.hist_bins):
            raise ValueError("hist_bins must be three positive integers (H, S, V)")
        if self.gray_size < 2:
            raise ValueError("gray_size must be at least 2")
        if not 0.0 <= self.hist_weight <= 1.0:
            raise ValueError("hist_weight must be in [0, 1]")
        if not 0.0 <= self.min_absolute_distance <= 1.0:
            raise ValueError("min_absolute_distance must be in [0, 1]")
        if self.min_event_seconds <= 0:
            raise ValueError("min_event_seconds must be positive")
        if self.max_event_seconds < self.min_event_seconds:
            raise ValueError("max_event_seconds must be >= min_event_seconds")
        if self.rolling_window_seconds < self.min_event_seconds:
            raise ValueError("rolling_window_seconds should span at least one minimum event")
        if self.gap_tolerance_seconds < 1.0 / self.sample_fps:
            raise ValueError(
                "gap_tolerance_seconds is below the frame sampling interval, so every "
                "frame pair would look like a gap"
            )

    @property
    def frame_interval_seconds(self) -> float:
        return 1.0 / self.sample_fps

    @property
    def rolling_window_frames(self) -> int:
        """Rolling statistics window, in sampled frames. Always odd, always >= 5."""
        frames = int(round(self.rolling_window_seconds * self.sample_fps))
        frames = max(5, frames)
        return frames + 1 if frames % 2 == 0 else frames

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config key(s): {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str) -> "ChunkingConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["exclude_days"] = list(self.exclude_days)
        data["hist_bins"] = list(self.hist_bins)
        return data

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
