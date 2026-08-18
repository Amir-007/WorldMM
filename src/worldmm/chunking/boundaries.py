"""
Event-boundary detection by visual feature differencing.  [Objective 1]

Frames arrive as a single continuous timeline stitched across mp4 files,
because EgoLife ships ~30s pre-sliced segments and an event routinely runs
across several of them. Detection therefore never looks at file boundaries;
it looks at time.

Pipeline:

    frame features -> adjacent-frame distance -> robust rolling z-score
                   -> threshold -> min/max length constraints -> cut times

The z-score is computed against a rolling median/MAD rather than a global
mean/σ, so a threshold tuned on one day transfers to a day with different
lighting without retuning, and a single dramatic transition cannot raise the
bar for the rest of the recording.

The expensive parts (features, distances, z-scores) are computed once.
Changing the threshold only re-runs `cuts_from_zscores`, so the Obj 3
sensitivity sweep costs seconds rather than another pass over 52 hours.

No VLM is involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ChunkingConfig
from .features import FrameFeatures

logger = logging.getLogger(__name__)


@dataclass
class Timeline:
    """Frame features from many files, stitched into one ordered timeline."""

    times: np.ndarray               # (N,) absolute seconds, strictly increasing
    hist: np.ndarray                # (N, H)
    gray: np.ndarray                # (N, G)
    video_paths: List[str]          # (N,) source file per frame
    gaps: np.ndarray                # (N-1,) True where the pair spans unobserved video

    @property
    def n(self) -> int:
        return int(self.times.shape[0])

    @property
    def span(self) -> Tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])


# Why a cut exists. Only DETECTED is evidence of a semantic state change; the
# other two are structural, and reporting them separately keeps the Objective 1
# claim honest about how much of the segmentation the detector actually drove.
DETECTED = "detected"    # a visual-change outlier cleared the threshold
GAP = "gap"              # unobserved video, so continuity cannot be assumed
CAP = "cap"              # max_event_seconds forced a split with no evidence


@dataclass
class BoundaryResult:
    """Detected cuts, plus the intermediates needed for a cheap threshold sweep."""

    timeline: Timeline
    distances: np.ndarray           # (N-1,) NaN across gaps
    zscores: np.ndarray             # (N-1,) NaN across gaps
    cuts: List[float]               # absolute seconds, sorted, strictly inside the span
    origins: Dict[float, str] = field(default_factory=dict)
    config: Optional[ChunkingConfig] = None

    @property
    def forced_cuts(self) -> List[float]:
        """Cuts imposed by a gap in observed video."""
        return sorted(c for c in self.cuts if self.origins.get(c) == GAP)

    def origin_counts(self) -> Dict[str, int]:
        counts = {DETECTED: 0, GAP: 0, CAP: 0}
        for cut in self.cuts:
            counts[self.origins.get(cut, DETECTED)] += 1
        return counts

    @property
    def segments(self) -> List[Tuple[float, float]]:
        """Half-open [start, end) event spans covering the whole timeline."""
        start, end = self.timeline.span
        edges = [start, *self.cuts, end]
        return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]

    def stats(self) -> dict:
        lengths = np.array([b - a for a, b in self.segments], dtype=np.float64)
        if lengths.size == 0:
            return {"n_segments": 0}
        counts = self.origin_counts()
        return {
            "n_segments": int(lengths.size),
            "n_forced_cuts": counts[GAP],
            "n_detected_cuts": counts[DETECTED],
            "n_cap_cuts": counts[CAP],
            "pct_cuts_detected": (
                100.0 * counts[DETECTED] / len(self.cuts) if self.cuts else 0.0
            ),
            "total_seconds": float(lengths.sum()),
            "mean_seconds": float(lengths.mean()),
            "median_seconds": float(np.median(lengths)),
            "p95_seconds": float(np.percentile(lengths, 95)),
            "min_seconds": float(lengths.min()),
            "max_seconds": float(lengths.max()),
        }


def build_timeline(
    features: Sequence[FrameFeatures],
    *,
    gap_tolerance_seconds: float,
) -> Timeline:
    """
    Stitch per-file features into one ordered timeline.

    Unusable files are dropped, which leaves a hole; the hole is recorded as a
    gap so detection forces a cut there. Continuity cannot be asserted across
    video that was never observed.
    """
    usable = [f for f in features if f.usable]
    if not usable:
        raise ValueError("no usable frame features")

    times = np.concatenate([f.times for f in usable])
    hist = np.concatenate([f.hist for f in usable])
    gray = np.concatenate([f.gray for f in usable])
    paths = [f.video_path for f in usable for _ in range(f.n)]

    order = np.argsort(times, kind="stable")
    times, hist, gray = times[order], hist[order], gray[order]
    paths = [paths[i] for i in order]

    # Identical timestamps would produce a zero-length interval downstream.
    keep = np.ones(times.shape[0], dtype=bool)
    keep[1:] = np.diff(times) > 0
    if not keep.all():
        logger.warning("Dropping %d frame(s) with duplicate timestamps", int((~keep).sum()))
        times, hist, gray = times[keep], hist[keep], gray[keep]
        paths = [p for p, k in zip(paths, keep) if k]

    gaps = np.diff(times) > gap_tolerance_seconds
    if gaps.any():
        logger.info(
            "Timeline has %d gap(s) in observed video; each forces an event boundary",
            int(gaps.sum()),
        )
    return Timeline(times=times, hist=hist, gray=gray, video_paths=paths, gaps=gaps)


def frame_distances(timeline: Timeline, *, hist_weight: float) -> np.ndarray:
    """
    Distance between consecutive frames, NaN across gaps.

    Histogram distance is half the L1 distance between L1-normalised
    histograms, so it lands in [0, 1]. Grayscale distance is mean absolute
    difference of thumbnails in [0, 1]. Both are bounded the same way, so the
    weighted sum is meaningful without further scaling.
    """
    hist_d = 0.5 * np.abs(np.diff(timeline.hist, axis=0)).sum(axis=1)
    gray_d = np.abs(np.diff(timeline.gray, axis=0)).mean(axis=1)
    distances = hist_weight * hist_d + (1.0 - hist_weight) * gray_d
    # A gap is missing observation, not evidence of change; excluding it keeps
    # the rolling statistics honest.
    distances[timeline.gaps] = np.nan
    return distances


def rolling_zscore(distances: np.ndarray, window: int) -> np.ndarray:
    """
    Robust z-score against a rolling median/MAD, NaN-aware.

    A hard floor is applied to the scale: in genuinely static video the MAD
    collapses to zero and any flicker would otherwise score arbitrarily high.
    """
    n = distances.shape[0]
    if n == 0:
        return distances.copy()

    half = max(1, window // 2)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return np.full(n, np.nan)

    # Floor the scale at a small fraction of the overall typical distance, so
    # a locally flat window cannot manufacture boundaries out of noise.
    scale_floor = max(float(np.median(finite)) * 0.05, 1e-6)

    zscores = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isfinite(distances[i]):
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        local = distances[lo:hi]
        local = local[np.isfinite(local)]
        if local.size < 3:
            continue
        median = np.median(local)
        mad = np.median(np.abs(local - median))
        scale = max(1.4826 * mad, scale_floor)
        zscores[i] = (distances[i] - median) / scale
    return zscores


def cuts_from_zscores(
    timeline: Timeline,
    distances: np.ndarray,
    zscores: np.ndarray,
    config: ChunkingConfig,
) -> Tuple[List[float], List[float]]:
    """
    Turn z-scores into cut times under the length constraints.

    Cheap by design: this is the only step a threshold sweep needs to repeat.

    Returns:
        (cuts, origins) where origins maps each cut time to DETECTED, GAP or
        CAP, so the write-up can state how much of the segmentation the
        detector actually drove.
    """
    times = timeline.times
    start, end = timeline.span

    # A cut sits at the *later* frame of the pair that triggered it.
    forced = [float(times[i + 1]) for i in np.flatnonzero(timeline.gaps)]

    # A candidate must be a local outlier *and* an absolute change, so sensor
    # noise in static footage cannot clear the bar on z-score alone.
    eligible = (
        np.isfinite(zscores)
        & (zscores > config.threshold)
        & np.isfinite(distances)
        & (distances > config.min_absolute_distance)
    )
    candidates = [(float(times[i + 1]), float(zscores[i])) for i in np.flatnonzero(eligible)]

    forced_set = set(forced)

    # Gap-forced cuts are hard discontinuities, not evidence to be weighed:
    # an event cannot span video that was never observed, so they are kept
    # unconditionally even if that leaves a short segment.
    accepted: List[float] = [t for t in forced if start < t < end]

    # Pass 1: non-maximum suppression by evidence strength, so the minimum
    # event length is spent on the best available boundary. Accepting in time
    # order instead would let a marginal early candidate suppress a much
    # stronger one a few seconds later.
    #
    # The span endpoints act as fixed neighbours, so the first and last
    # segments obey the minimum length like every other one.
    ranked = sorted(
        (c for c in candidates if c[0] not in forced_set),
        key=lambda item: -item[1],
    )
    for time, _ in ranked:
        if time <= start or time >= end:
            continue
        neighbours = [start, end, *accepted]
        if all(abs(time - other) >= config.min_event_seconds for other in neighbours):
            accepted.append(time)
    accepted.sort()

    origins: Dict[float, str] = {}
    for cut in accepted:
        origins[cut] = GAP if cut in forced_set else DETECTED

    # Pass 2: honour the maximum event length, splitting at the strongest
    # available evidence inside the over-long stretch rather than at an
    # arbitrary tick.
    z_by_time = {float(times[i + 1]): float(zscores[i])
                 for i in np.flatnonzero(np.isfinite(zscores))}
    result: List[float] = []
    last = start
    for time in [*accepted, end]:
        while time - last > config.max_event_seconds:
            window_lo = last + config.min_event_seconds
            # Leave at least a minimum event on the far side too, so the
            # remainder of an over-long stretch is not cut into a sliver.
            window_hi = min(last + config.max_event_seconds,
                            time - config.min_event_seconds)
            if window_hi < window_lo:
                window_hi = last + config.max_event_seconds
            inside = [(t, z) for t, z in z_by_time.items() if window_lo <= t <= window_hi]
            if inside:
                split = max(inside, key=lambda item: item[1])[0]
            else:
                # No frame evidence available (e.g. sparse sampling): fall back
                # to the nearest sampled frame at or before the cap.
                eligible = times[(times >= window_lo) & (times <= window_hi)]
                split = float(eligible[-1]) if eligible.size else window_hi
            if split <= last:
                break
            result.append(split)
            origins.setdefault(split, CAP)
            last = split
        if time < end:
            result.append(time)
            last = time

    cuts = sorted({c for c in result if start < c < end})
    return cuts, {c: origins.get(c, DETECTED) for c in cuts}


def detect_boundaries(
    features: Sequence[FrameFeatures],
    config: ChunkingConfig,
) -> BoundaryResult:
    """Run the whole detection pipeline over one day's features."""
    timeline = build_timeline(features, gap_tolerance_seconds=config.gap_tolerance_seconds)
    distances = frame_distances(timeline, hist_weight=config.hist_weight)
    zscores = rolling_zscore(distances, config.rolling_window_frames)
    cuts, origins = cuts_from_zscores(timeline, distances, zscores, config)
    return BoundaryResult(
        timeline=timeline,
        distances=distances,
        zscores=zscores,
        cuts=cuts,
        origins=origins,
        config=config,
    )


def sweep_thresholds(
    result: BoundaryResult,
    thresholds: Sequence[float],
) -> List[dict]:
    """
    Re-cut an existing detection at several thresholds.

    Reuses the cached distances and z-scores, so this is the cheap Obj 3
    ablation: no video is read and no features are recomputed.
    """
    if result.config is None:
        raise ValueError("BoundaryResult has no config to vary")

    rows = []
    for threshold in thresholds:
        config = ChunkingConfig.from_dict({**result.config.to_dict(), "threshold": threshold})
        cuts, origins = cuts_from_zscores(
            result.timeline, result.distances, result.zscores, config
        )
        candidate = BoundaryResult(
            timeline=result.timeline,
            distances=result.distances,
            zscores=result.zscores,
            cuts=cuts,
            origins=origins,
            config=config,
        )
        rows.append({"threshold": float(threshold), **candidate.stats()})
    return rows
