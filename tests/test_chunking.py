#!/usr/bin/env python3
"""
Stage 1 tests: frame features and event-boundary detection.

    python3 tests/test_chunking.py
    pytest tests/test_chunking.py

The detection algorithm is tested against synthetic feature timelines with
known cut points, so the logic is fully covered without any video. The decode
path is then tested against a real mp4 synthesised with OpenCV, which is what
makes this verifiable on a laptop rather than only on the cluster.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.chunking.boundaries import (  # noqa: E402
    build_timeline,
    cuts_from_zscores,
    detect_boundaries,
    frame_distances,
    rolling_zscore,
    sweep_thresholds,
)
from worldmm.chunking.config import ChunkingConfig  # noqa: E402
from worldmm.chunking.features import (  # noqa: E402
    OK,
    FeatureCache,
    FrameFeatures,
    extract_features,
    parse_video_start,
)

SEED = 20260817


class SkipTest(Exception):
    """Raised when a test's dependency is missing."""


# --------------------------------------------------------------------------
# synthetic timelines
# --------------------------------------------------------------------------

def make_features(
    scene_lengths,
    *,
    sample_fps=2.0,
    start=0.0,
    n_hist=64,
    n_gray=16,
    noise=0.01,
    video_path="data/EgoLife/A1_JAKE/DAY1/DAY1_A1_JAKE_11000000.mp4",
):
    """
    Build a feature timeline of consecutive visually distinct scenes.

    Each scene gets its own histogram mode and grayscale level, plus noise, so
    within-scene frames are similar and scene transitions are sharp.
    """
    rng = np.random.default_rng(SEED)
    times, hists, grays = [], [], []
    t = start

    for scene_index, length in enumerate(scene_lengths):
        base_hist = np.zeros(n_hist, dtype=np.float32)
        base_hist[(scene_index * 7) % n_hist] = 1.0
        base_gray = np.full(n_gray, 0.15 + 0.7 * ((scene_index * 3) % 5) / 5.0, dtype=np.float32)

        for _ in range(int(round(length * sample_fps))):
            hist = np.abs(base_hist + rng.normal(0, noise, n_hist).astype(np.float32))
            hist /= hist.sum()
            gray = np.clip(base_gray + rng.normal(0, noise, n_gray).astype(np.float32), 0, 1)
            times.append(t)
            hists.append(hist)
            grays.append(gray)
            t += 1.0 / sample_fps

    return FrameFeatures(
        video_path=video_path,
        status=OK,
        times=np.asarray(times, dtype=np.float64),
        hist=np.asarray(hists, dtype=np.float32),
        gray=np.asarray(grays, dtype=np.float32),
        source_fps=30.0,
        total_frames=len(times),
    )


def config(**overrides) -> ChunkingConfig:
    base = {
        "strategy": "event",
        "sample_fps": 2.0,
        "threshold": 3.0,
        "rolling_window_seconds": 60.0,
        "min_event_seconds": 10.0,
        "max_event_seconds": 60.0,
        "gap_tolerance_seconds": 2.0,
    }
    base.update(overrides)
    return ChunkingConfig.from_dict(base)


def assert_close_to_any(value, targets, tolerance):
    assert any(abs(value - t) <= tolerance for t in targets), \
        f"{value} is not within {tolerance}s of any of {targets}"


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_detects_known_scene_changes():
    """Five 30s scenes must yield cuts at the four transitions."""
    features = make_features([30, 30, 30, 30, 30])
    result = detect_boundaries([features], config())

    expected = [30.0, 60.0, 90.0, 120.0]
    for boundary in expected:
        assert any(abs(c - boundary) <= 1.0 for c in result.cuts), \
            f"missed the transition at {boundary}s; got {result.cuts}"
    assert len(result.cuts) <= len(expected) + 1, \
        f"too many spurious cuts: {result.cuts}"


def test_segments_tile_the_timeline():
    result = detect_boundaries([make_features([30, 30, 30])], config())
    segments = result.segments
    start, end = result.timeline.span
    assert segments[0][0] == start
    assert segments[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(segments, segments[1:]):
        assert prev_end == next_start, "segments must tile with no overlap or hole"


def test_min_event_length_suppresses_rapid_cuts():
    """Scenes shorter than min_event_seconds must not each become a chunk."""
    features = make_features([4] * 15)            # a transition every 4s
    result = detect_boundaries([features], config(min_event_seconds=20.0))
    lengths = [b - a for a, b in result.segments]
    # The final segment is whatever remains, so only check the constrained ones.
    for length in lengths[:-1]:
        assert length >= 20.0 - 1e-6, f"segment of {length}s violates the 20s minimum"


def test_min_event_length_applies_at_the_timeline_edges():
    """A cut moments after the start would leave a sliver opening segment."""
    features = make_features([3, 57])                 # transition only 3s in
    result = detect_boundaries(
        [features], config(min_event_seconds=10.0, max_event_seconds=1e6)
    )
    assert result.cuts == [], f"the 3s transition should be suppressed, got {result.cuts}"


def test_only_gaps_may_produce_short_segments():
    """Detected cuts always respect the minimum; forced ones may not."""
    features = make_features([12, 12, 12, 12, 12])
    result = detect_boundaries(
        [features], config(min_event_seconds=10.0, max_event_seconds=1e6)
    )
    assert not result.forced_cuts
    for start, end in result.segments:
        assert end - start >= 10.0 - 1e-6, f"segment of {end - start}s is under the minimum"


def test_max_event_length_forces_a_split():
    """A long featureless stretch must still be cut at the cap."""
    features = make_features([300])               # one static scene, no transitions
    result = detect_boundaries([features], config(max_event_seconds=60.0))
    lengths = [b - a for a, b in result.segments]
    assert len(lengths) >= 5, f"expected the 60s cap to force splits, got {lengths}"
    for length in lengths:
        assert length <= 60.0 + 1e-6, f"segment of {length}s exceeds the 60s cap"


def test_gap_forces_a_boundary():
    """Unobserved video is a discontinuity, not evidence of visual similarity."""
    first = make_features([30], start=0.0)
    second = make_features([30], start=300.0)     # a 270s hole
    result = detect_boundaries([first, second], config())
    assert any(abs(c - 300.0) <= 1e-6 for c in result.cuts), \
        f"the gap at 300s did not force a cut; got {result.cuts}"
    assert 300.0 in result.forced_cuts


def test_gap_is_excluded_from_distance_statistics():
    first = make_features([20], start=0.0)
    second = make_features([20], start=300.0)
    timeline = build_timeline([first, second], gap_tolerance_seconds=2.0)
    distances = frame_distances(timeline, hist_weight=0.5)
    assert np.isnan(distances[timeline.gaps]).all()
    assert np.isfinite(distances[~timeline.gaps]).all()


def test_gap_survives_min_event_suppression():
    """A forced cut must not be swallowed by the minimum-length rule."""
    first = make_features([30], start=0.0)
    second = make_features([30], start=32.0)      # gap lands 2s after a transition
    result = detect_boundaries([first, second], config(min_event_seconds=30.0))
    assert any(abs(c - 32.0) <= 1e-6 for c in result.cuts), \
        f"forced cut was suppressed; got {result.cuts}"


def test_timeline_sorts_and_deduplicates():
    later = make_features([10], start=100.0)
    earlier = make_features([10], start=0.0)
    timeline = build_timeline([later, earlier], gap_tolerance_seconds=2.0)
    assert np.all(np.diff(timeline.times) > 0)
    assert timeline.times[0] == 0.0


def test_unusable_features_are_dropped():
    good = make_features([20])
    bad = FrameFeatures.empty("missing.mp4", "absent", "file not found")
    timeline = build_timeline([good, bad], gap_tolerance_seconds=2.0)
    assert timeline.n == good.n


def test_rolling_zscore_ignores_nan():
    values = np.array([0.1, 0.1, np.nan, 0.1, 5.0, 0.1, 0.1], dtype=np.float64)
    zscores = rolling_zscore(values, window=5)
    assert np.isnan(zscores[2])
    assert zscores[4] > 3.0
    assert np.nanmax(np.abs(zscores[[0, 1, 3, 5, 6]])) < zscores[4]


def test_static_video_produces_no_spurious_cuts():
    """A flat MAD must not turn sensor noise into boundaries."""
    features = make_features([120], noise=1e-6)
    cfg = config(max_event_seconds=1e6)           # disable the cap so only detection speaks
    result = detect_boundaries([features], cfg)
    assert result.cuts == [], f"static video produced cuts: {result.cuts}"


def test_threshold_sweep_is_monotonic_and_free():
    features = make_features([30, 30, 30, 30, 30])
    result = detect_boundaries([features], config())
    rows = sweep_thresholds(result, [1.0, 3.0, 6.0, 12.0, 50.0])
    counts = [row["n_segments"] for row in rows]
    assert counts == sorted(counts, reverse=True), \
        f"raising the threshold should not add segments: {counts}"


def test_sweep_reuses_cached_zscores():
    """The sweep must not recompute features or distances."""
    result = detect_boundaries([make_features([30, 30, 30])], config())
    before = result.zscores.copy()
    sweep_thresholds(result, [2.0, 4.0, 8.0])
    assert np.array_equal(before, result.zscores, equal_nan=True)


def test_cuts_are_strictly_inside_the_span():
    result = detect_boundaries([make_features([30, 30, 30])], config())
    start, end = result.timeline.span
    for cut in result.cuts:
        assert start < cut < end


def test_stats_report_segment_distribution():
    result = detect_boundaries([make_features([30, 30, 30, 30])], config())
    stats = result.stats()
    assert stats["n_segments"] == len(result.segments)
    assert stats["min_seconds"] <= stats["median_seconds"] <= stats["max_seconds"]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_config_rejects_bad_values():
    for bad in ({"strategy": "nonsense"},
                {"max_event_seconds": 5.0, "min_event_seconds": 10.0},
                {"hist_weight": 1.5},
                {"sample_fps": 0},
                {"gap_tolerance_seconds": 0.1, "sample_fps": 2.0}):
        try:
            config(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")


def test_config_rejects_unknown_keys():
    try:
        ChunkingConfig.from_dict({"strategy": "event", "thresold": 3.0})   # typo
    except ValueError as exc:
        assert "thresold" in str(exc)
    else:
        raise AssertionError("a mistyped key must not be silently ignored")


def test_config_round_trips_through_disk():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.json")
        original = config(threshold=4.5, max_event_seconds=45.0)
        original.write(path)
        assert ChunkingConfig.from_file(path).to_dict() == original.to_dict()


def test_rolling_window_frames_is_odd_and_bounded():
    assert config(rolling_window_seconds=60.0, sample_fps=2.0).rolling_window_frames == 121
    sparse = config(rolling_window_seconds=10.0, sample_fps=0.1,
                    min_event_seconds=10.0, gap_tolerance_seconds=20.0)
    assert sparse.rolling_window_frames >= 5
    assert sparse.rolling_window_frames % 2 == 1


# --------------------------------------------------------------------------
# features and cache
# --------------------------------------------------------------------------

def test_parse_video_start():
    ts = parse_video_start("data/EgoLife/A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4")
    assert (ts.day, ts.hour, ts.minute, ts.second, ts.frame) == (1, 11, 9, 42, 8)
    assert abs(ts.seconds - (86400 + 11 * 3600 + 9 * 60 + 42 + 8 / 30)) < 1e-6


def test_parse_video_start_rejects_unknown_names():
    try:
        parse_video_start("clip.mp4")
    except ValueError:
        return
    raise AssertionError("an unparseable video name must raise")


def test_missing_video_is_reported_not_raised():
    features = extract_features("does/not/exist/DAY1_A1_JAKE_11000000.mp4")
    assert features.status == "absent"
    assert not features.usable


def test_feature_cache_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        cache = FeatureCache(tmp)
        original = make_features([10])
        assert not cache.has(original.video_path)
        cache.save(original)
        assert cache.has(original.video_path)

        loaded = cache.load(original.video_path)
        assert loaded is not None
        assert loaded.status == OK
        assert np.allclose(loaded.times, original.times)
        assert np.allclose(loaded.hist, original.hist)
        assert np.allclose(loaded.gray, original.gray)


def test_feature_cache_survives_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        cache = FeatureCache(tmp)
        features = make_features([10])
        path = cache.save(features)
        with open(path, "wb") as f:
            f.write(b"not an npz")
        assert cache.load(features.video_path) is None


# --------------------------------------------------------------------------
# real decode path
# --------------------------------------------------------------------------

def _write_test_video(path, scene_lengths, fps=10, size=(160, 120)):
    """Synthesise an mp4 of visually distinct scenes."""
    try:
        import cv2
    except ImportError as exc:
        raise SkipTest("opencv not installed") from exc

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SkipTest("no mp4v encoder available")

    rng = np.random.default_rng(SEED)
    colours = [(220, 30, 30), (30, 220, 30), (30, 30, 220), (220, 220, 30), (30, 220, 220)]
    try:
        for index, length in enumerate(scene_lengths):
            colour = colours[index % len(colours)]
            for _ in range(int(length * fps)):
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                frame[:, :] = colour
                frame = np.clip(
                    frame.astype(np.int16) + rng.integers(-6, 7, frame.shape), 0, 255
                ).astype(np.uint8)
                writer.write(frame)
    finally:
        writer.release()
    return path


def test_extract_features_from_a_real_video():
    with tempfile.TemporaryDirectory() as tmp:
        day_dir = os.path.join(tmp, "DAY1")
        os.makedirs(day_dir)
        path = os.path.join(day_dir, "DAY1_A1_JAKE_11000000.mp4")
        _write_test_video(path, [10, 10, 10])

        features = extract_features(path, sample_fps=2.0)
        assert features.usable, f"status={features.status} error={features.error}"
        assert features.hist.shape[0] == features.gray.shape[0] == features.n
        assert features.hist.shape[1] == 8 * 8 * 8
        assert features.gray.shape[1] == 32 * 32
        assert 50 <= features.n <= 70, f"expected ~60 frames at 2fps over 30s, got {features.n}"

        # Absolute times must start at the timestamp encoded in the filename.
        assert abs(features.times[0] - parse_video_start(path).seconds) < 0.5
        assert np.all(np.diff(features.times) > 0)

        # Histograms are L1-normalised.
        assert np.allclose(features.hist.sum(axis=1), 1.0, atol=1e-4)


def test_detects_boundaries_in_a_real_video():
    """End to end on real pixels: decode, feature, detect."""
    with tempfile.TemporaryDirectory() as tmp:
        day_dir = os.path.join(tmp, "DAY1")
        os.makedirs(day_dir)
        path = os.path.join(day_dir, "DAY1_A1_JAKE_11000000.mp4")
        _write_test_video(path, [20, 20, 20])

        features = extract_features(path, sample_fps=2.0)
        if not features.usable:
            raise SkipTest(f"decode failed: {features.error}")

        result = detect_boundaries([features], config(min_event_seconds=5.0))
        start = result.timeline.span[0]
        relative = [c - start for c in result.cuts]
        for expected in (20.0, 40.0):
            assert_close_to_any(expected, relative, tolerance=1.5)


# --------------------------------------------------------------------------

def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed = failed = skipped = 0

    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            print(f"SKIP {name}: {exc}")
            skipped += 1
        except AssertionError as exc:
            print(f"FAIL {name}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
            failed += 1
        else:
            print(f"ok   {name}")
            passed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
