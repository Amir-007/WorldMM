"""
Per-frame visual features, computed once and cached.

Boundary detection needs a cheap, stable description of each sampled frame.
Two complementary signals, because neither is reliable alone in egocentric
video: an HSV colour histogram (catches scene, room and lighting change but is
blind to rearrangement within a scene) and a downsampled grayscale thumbnail
(catches layout and viewpoint change but is fooled by lighting).

No VLM is involved, by design.

Features are cached per video file as .npz. That is the expensive pass over 52
hours of video; once it exists, re-running boundary detection at a different
threshold costs seconds and never touches video again, which is what makes the
Obj 3 threshold sweep affordable.

Reader backends: decord when available (matches the rest of the pipeline and
the cluster), OpenCV otherwise, so this is testable on a laptop.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..common.timestamps import EgoTimestamp

logger = logging.getLogger(__name__)

# Outcome of trying to read one video file.
OK = "ok"
ABSENT = "absent"
UNREADABLE = "unreadable"

_VIDEO_NAME_RE = re.compile(r"DAY(\d+)_.*?_(\d{8})\.mp4$", re.IGNORECASE)


@dataclass
class FrameFeatures:
    """Sampled-frame features for one video file."""

    video_path: str
    status: str
    times: np.ndarray          # (N,) absolute seconds on the EgoLife timeline
    hist: np.ndarray           # (N, prod(hist_bins)) L1-normalised HSV histogram
    gray: np.ndarray           # (N, gray_size**2) grayscale thumbnail in [0, 1]
    source_fps: float = 0.0
    total_frames: int = 0
    error: Optional[str] = None

    @property
    def n(self) -> int:
        return int(self.times.shape[0])

    @property
    def usable(self) -> bool:
        return self.status == OK and self.n > 0

    @classmethod
    def empty(cls, video_path: str, status: str, error: Optional[str] = None) -> "FrameFeatures":
        return cls(
            video_path=video_path,
            status=status,
            times=np.zeros(0, dtype=np.float64),
            hist=np.zeros((0, 0), dtype=np.float32),
            gray=np.zeros((0, 0), dtype=np.float32),
            error=error,
        )


def parse_video_start(video_path: str) -> EgoTimestamp:
    """
    Recover a video file's start time from its name.

    EgoLife names segments `DAY1_A1_JAKE_11094208.mp4`, i.e. day plus HHMMSSFF.
    """
    match = _VIDEO_NAME_RE.search(os.path.basename(video_path))
    if not match:
        raise ValueError(f"cannot read a DAY/timestamp from video name: {video_path!r}")
    return EgoTimestamp.from_parts(f"DAY{match.group(1)}", match.group(2))


def _frame_features(
    frame_rgb: np.ndarray,
    hist_bins: Tuple[int, int, int],
    gray_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """HSV histogram and grayscale thumbnail for a single RGB frame."""
    import cv2

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(hist_bins), [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).ravel()
    total = float(hist.sum())
    if total > 0:
        hist /= total  # L1-normalised, so histogram distance is scale-free

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (gray_size, gray_size), interpolation=cv2.INTER_AREA)
    return hist, (gray.astype(np.float32) / 255.0).ravel()


def _sample_indices(total_frames: int, source_fps: float, sample_fps: float) -> List[int]:
    """Frame indices to sample, evenly spaced at the requested rate."""
    if total_frames <= 0:
        return []
    if source_fps <= 0:
        return [0]
    stride = max(1, int(round(source_fps / sample_fps)))
    return list(range(0, total_frames, stride))


def _read_decord(video_path: str, sample_fps: float):
    from decord import VideoReader, cpu

    reader = VideoReader(video_path, ctx=cpu(0))
    total = len(reader)
    fps = float(reader.get_avg_fps())
    indices = _sample_indices(total, fps, sample_fps)
    if not indices:
        return [], [], fps, total
    batch = reader.get_batch(indices).asnumpy()  # RGB
    return list(batch), indices, fps, total


def _read_opencv(video_path: str, sample_fps: float):
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the file")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        wanted = set(_sample_indices(total, fps, sample_fps))

        # Sequential decode and skip: seeking per frame is slower and less
        # reliable across codecs than reading straight through.
        frames, indices, index = [], [], 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if index in wanted:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                indices.append(index)
            index += 1
        return frames, indices, fps, (total if total > 0 else index)
    finally:
        capture.release()


def extract_features(
    video_path: str,
    *,
    sample_fps: float = 2.0,
    hist_bins: Tuple[int, int, int] = (8, 8, 8),
    gray_size: int = 32,
    start_seconds: Optional[float] = None,
    prefer_decord: bool = True,
) -> FrameFeatures:
    """
    Sample one video and describe each sampled frame. Never raises.

    Args:
        start_seconds: Absolute timeline offset for frame 0. Derived from the
            filename when omitted.
    """
    if not os.path.exists(video_path):
        return FrameFeatures.empty(video_path, ABSENT, "file not found")
    try:
        if os.path.getsize(video_path) == 0:
            return FrameFeatures.empty(video_path, UNREADABLE, "zero bytes")
    except OSError as exc:
        return FrameFeatures.empty(video_path, UNREADABLE, str(exc))

    if start_seconds is None:
        try:
            start_seconds = parse_video_start(video_path).seconds
        except ValueError as exc:
            return FrameFeatures.empty(video_path, UNREADABLE, str(exc))

    readers = []
    if prefer_decord:
        readers.append(("decord", _read_decord))
    readers.append(("opencv", _read_opencv))

    frames = indices = None
    source_fps = 0.0
    total = 0
    last_error: Optional[str] = None

    for name, reader in readers:
        try:
            frames, indices, source_fps, total = reader(video_path, sample_fps)
            break
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001 - any decode failure falls through
            last_error = f"{name}: {type(exc).__name__}: {exc}"
            frames = None

    if frames is None:
        return FrameFeatures.empty(video_path, UNREADABLE, last_error or "no usable video reader")
    if not frames:
        return FrameFeatures.empty(video_path, UNREADABLE, "decoded zero frames")

    hists, grays = [], []
    for frame in frames:
        hist, gray = _frame_features(frame, hist_bins, gray_size)
        hists.append(hist)
        grays.append(gray)

    effective_fps = source_fps if source_fps > 0 else sample_fps
    times = np.array([start_seconds + idx / effective_fps for idx in indices], dtype=np.float64)

    return FrameFeatures(
        video_path=video_path,
        status=OK,
        times=times,
        hist=np.asarray(hists, dtype=np.float32),
        gray=np.asarray(grays, dtype=np.float32),
        source_fps=source_fps,
        total_frames=total,
    )


class FeatureCache:
    """On-disk store of per-video features, one .npz per video."""

    def __init__(self, root: str) -> None:
        self.root = root

    def path_for(self, video_path: str) -> str:
        """Cache path mirroring the video's DAY directory, to keep it browsable."""
        base = os.path.basename(video_path)
        stem = base[:-4] if base.lower().endswith(".mp4") else base
        day = os.path.basename(os.path.dirname(video_path)) or "unknown"
        return os.path.join(self.root, day, f"{stem}.npz")

    def has(self, video_path: str) -> bool:
        return os.path.exists(self.path_for(video_path))

    def save(self, features: FrameFeatures) -> str:
        path = self.path_for(features.video_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        meta = np.array(
            [features.video_path, features.status, str(features.source_fps),
             str(features.total_frames), features.error or ""],
            dtype=object,
        )
        # Write through a file handle: given a path, numpy appends ".npz" to
        # any name that lacks it, which would leave the rename with no source.
        with open(tmp, "wb") as handle:
            np.savez_compressed(
                handle,
                times=features.times,
                hist=features.hist,
                gray=features.gray,
                meta=meta,
            )
        os.replace(tmp, path)
        return path

    def load(self, video_path: str) -> Optional[FrameFeatures]:
        path = self.path_for(video_path)
        if not os.path.exists(path):
            return None
        try:
            with np.load(path, allow_pickle=True) as data:
                meta = list(data["meta"])
                return FrameFeatures(
                    video_path=str(meta[0]),
                    status=str(meta[1]),
                    times=data["times"],
                    hist=data["hist"],
                    gray=data["gray"],
                    source_fps=float(meta[2]),
                    total_frames=int(meta[3]),
                    error=str(meta[4]) or None,
                )
        except Exception as exc:  # noqa: BLE001 - a corrupt cache entry is recomputable
            logger.warning("Discarding unreadable cache entry %s: %s", path, exc)
            return None

    def load_many(self, video_paths: Sequence[str]) -> List[FrameFeatures]:
        """Load in the given order, skipping videos with no cache entry."""
        out = []
        for video_path in video_paths:
            features = self.load(video_path)
            if features is not None:
                out.append(features)
        return out
