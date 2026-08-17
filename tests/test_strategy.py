#!/usr/bin/env python3
"""
Stage 1b tests: source loading and chunk assembly.

    python3 tests/test_strategy.py
    pytest tests/test_strategy.py

The Sync files live on the cluster, so these build synthetic ones with the same
structure `generate_sync.py` writes. The invariant that matters most is that
both strategies consume identical source entries through identical rendering,
so any downstream difference is attributable to chunking alone.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.chunking.config import ChunkingConfig  # noqa: E402
from worldmm.chunking.sources import (  # noqa: E402
    SourceEntry,
    load_sync_entries,
    normalize_first_person,
    render_entries,
    speaker_name,
)
from worldmm.chunking.strategy import (  # noqa: E402
    EventBoundaryStrategy,
    FixedWindowStrategy,
    build_strategy,
    chunk_statistics,
)
from worldmm.common.timestamps import SECONDS_PER_DAY  # noqa: E402

PERSON = "A1_JAKE"


class SkipTest(Exception):
    pass


def config(**overrides) -> ChunkingConfig:
    base = {"strategy": "event", "person": PERSON, "max_event_seconds": 60.0}
    base.update(overrides)
    return ChunkingConfig.from_dict(base)


def make_entries(n_files=4, per_file=10, day=1, base_hhmmss=110000, step=3):
    """Source entries spread across consecutive video files, 30s per file."""
    entries = []
    for f in range(n_files):
        video_file = f"DAY{day}_{PERSON}_{base_hhmmss + f * 30:06d}00.mp4"
        for i in range(per_file):
            offset = f * 30 + i * step
            start = day * SECONDS_PER_DAY + 11 * 3600 + offset
            entries.append(SourceEntry(
                day=day,
                start_seconds=float(start),
                end_seconds=float(start + step),
                text=f"I do action {f}-{i}.",
                type="caption",
                video_file=video_file,
            ))
    return entries


def write_sync(directory, day=1, hour=11, blocks=2, per_block=3):
    """Write a Sync file in the shape generate_sync.py produces."""
    path = os.path.join(directory, f"{PERSON}_DAY{day}_{hour:02d}000000.json")
    data = []
    for b in range(blocks):
        entries = []
        for i in range(per_block):
            second = b * 30 + i * 5
            hhmmss = int(f"{hour:02d}{second // 60:02d}{second % 60:02d}")
            entries.append({
                "start": hhmmss,
                "end": hhmmss + 2,
                "text": f"Jake: block {b} line {i}",
                "type": "caption" if i % 2 == 0 else "transcript",
            })
        data.append({"video_file": f"DAY{day}_{PERSON}_{hour:02d}{b * 30 // 60:02d}"
                                   f"{b * 30 % 60:02d}00.mp4", "data": entries})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def test_speaker_name():
    assert speaker_name("A1_JAKE") == "Jake"
    assert speaker_name("A5_KATRINA") == "Katrina"


def test_normalize_first_person():
    assert normalize_first_person("Jake: hello", PERSON) == "hello"
    assert normalize_first_person("Jake's phone", PERSON) == "my phone"
    assert normalize_first_person("Jake walks in", PERSON) == "I walks in"
    # Other speakers keep their names.
    assert "Shure" in normalize_first_person("Shure: hi", PERSON)


def test_normalize_does_not_match_inside_words():
    assert normalize_first_person("Jakes", PERSON) == "Jakes"


def test_load_sync_entries():
    with tempfile.TemporaryDirectory() as tmp:
        write_sync(tmp, day=1)
        entries = load_sync_entries(tmp, PERSON, exclude_days=())
        assert len(entries) == 6
        assert all(e.day == 1 for e in entries)
        seconds = [e.start_seconds for e in entries]
        assert seconds == sorted(seconds)


def test_load_sync_entries_respects_day_filters():
    with tempfile.TemporaryDirectory() as tmp:
        write_sync(tmp, day=1)
        write_sync(tmp, day=6)
        assert len(load_sync_entries(tmp, PERSON, exclude_days=(6,))) == 6
        assert len(load_sync_entries(tmp, PERSON, exclude_days=())) == 12
        assert len(load_sync_entries(tmp, PERSON, days=[6], exclude_days=())) == 6


def test_load_sync_entries_errors_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_sync_entries(tmp, PERSON)
        except FileNotFoundError:
            return
        raise AssertionError("missing sync files must raise")


def test_render_deduplicates_repeated_lines():
    """Dense captions and transcripts repeat verbatim; longer chunks would
    otherwise accumulate duplicates and look artificially content-rich."""
    entries = [
        SourceEntry(1, 0.0, 2.0, "I open the drawer.", "caption", "a.mp4"),
        SourceEntry(1, 2.0, 4.0, "I open the drawer.", "transcript", "a.mp4"),
        SourceEntry(1, 4.0, 6.0, "I grab the scissors.", "caption", "a.mp4"),
    ]
    rendered = render_entries(entries, PERSON)
    assert rendered.count("I open the drawer.") == 1
    assert "I grab the scissors." in rendered


# --------------------------------------------------------------------------
# fixed-window strategy (the baseline)
# --------------------------------------------------------------------------

def test_fixed_window_is_one_chunk_per_video_file():
    entries = make_entries(n_files=4, per_file=10)
    chunks = FixedWindowStrategy(config(strategy="fixed", window_seconds=30.0)).segment(entries)
    assert len(chunks) == 4
    for chunk in chunks:
        assert not chunk.spans_multiple_videos


def test_fixed_window_groups_multiple_files():
    entries = make_entries(n_files=4, per_file=10)
    chunks = FixedWindowStrategy(config(strategy="fixed", window_seconds=60.0)).segment(entries)
    assert len(chunks) == 2
    for chunk in chunks:
        assert len(chunk.video_paths) == 2


def test_fixed_window_handles_overlapping_video_files():
    """
    Real EgoLife files overlap: starts drift under 30s apart while each clip
    runs a full 30s, so entries from adjacent files interleave in time order.
    Grouping must follow file membership, not the time-sorted stream, or the
    grid shatters into fragments.
    """
    base = SECONDS_PER_DAY + 11 * 3600
    entries = []
    for f in range(3):
        video_file = f"DAY1_{PERSON}_file{f}.mp4"
        for i in range(10):
            start = base + f * 18 + i * 3        # files start only 18s apart
            entries.append(SourceEntry(1, float(start), float(start + 2),
                                       f"line {f}-{i}", "caption", video_file))
    entries.sort(key=lambda e: e.start_seconds)   # interleaved across files

    chunks = FixedWindowStrategy(config(strategy="fixed")).segment(entries)
    assert len(chunks) == 3, f"expected one chunk per file, got {len(chunks)}"
    for chunk in chunks:
        assert chunk.n_source_entries == 10
        assert not chunk.spans_multiple_videos
    assert sum(c.n_source_entries for c in chunks) == len(entries)


def test_fixed_window_covers_every_entry():
    entries = make_entries(n_files=3, per_file=8)
    chunks = FixedWindowStrategy(config(strategy="fixed")).segment(entries)
    assert sum(c.n_source_entries for c in chunks) == len(entries)


# --------------------------------------------------------------------------
# event strategy (Objective 1)
# --------------------------------------------------------------------------

def test_event_chunks_span_video_file_boundaries():
    """The point of Obj 1: an event is not cut at an arbitrary mp4 edge."""
    entries = make_entries(n_files=4, per_file=10)
    start = min(e.start_seconds for e in entries)
    chunks = EventBoundaryStrategy(config()).segment(entries, cuts=[start + 45.0])
    assert len(chunks) == 2
    assert any(c.spans_multiple_videos for c in chunks), \
        "no chunk crossed a file boundary, so nothing was tested"


def test_event_cut_times_are_honoured_exactly():
    entries = make_entries(n_files=4, per_file=10)
    start = min(e.start_seconds for e in entries)
    cut = start + 45.0
    chunks = EventBoundaryStrategy(config()).segment(entries, cuts=[cut])
    assert abs(chunks[0].end_seconds - cut) < 1e-6
    assert abs(chunks[1].start_seconds - cut) < 1e-6


def test_event_boundaries_are_not_quantised_to_the_30s_grid():
    """A cut at 47.5s must stay at 47.5s, not snap to 30 or 60."""
    entries = make_entries(n_files=4, per_file=10, step=1)
    start = min(e.start_seconds for e in entries)
    chunks = EventBoundaryStrategy(config()).segment(entries, cuts=[start + 47.5])
    assert abs(chunks[0].duration_seconds - 47.5) < 1e-6


def test_event_assigns_entries_by_midpoint():
    """An entry straddling a cut goes where most of it happened."""
    base = SECONDS_PER_DAY + 11 * 3600          # DAY1 11:00:00
    entries = [
        SourceEntry(1, base + 100.0, base + 110.0, "mostly before", "caption", "a.mp4"),
        SourceEntry(1, base + 108.0, base + 118.0, "mostly after", "caption", "a.mp4"),
    ]
    chunks = EventBoundaryStrategy(config()).segment(entries, cuts=[base + 109.0])
    assert "mostly before" in chunks[0].text
    assert "mostly after" in chunks[1].text


def test_event_with_no_cuts_is_a_single_chunk():
    entries = make_entries(n_files=3, per_file=5)
    chunks = EventBoundaryStrategy(config()).segment(entries, cuts=[])
    assert len(chunks) == 1
    assert chunks[0].spans_multiple_videos


def test_event_covers_every_entry():
    entries = make_entries(n_files=4, per_file=10)
    start = min(e.start_seconds for e in entries)
    chunks = EventBoundaryStrategy(config()).segment(
        entries, cuts=[start + 20, start + 55, start + 90]
    )
    assert sum(c.n_source_entries for c in chunks) == len(entries)


def test_cuts_outside_the_span_are_ignored():
    entries = make_entries(n_files=2, per_file=5)
    chunks = EventBoundaryStrategy(config()).segment(entries, cuts=[0.0, 1e9])
    assert len(chunks) == 1


# --------------------------------------------------------------------------
# parity between conditions
# --------------------------------------------------------------------------

def test_both_conditions_render_identical_text_for_identical_grouping():
    """
    With cuts placed exactly on the file boundaries, the event strategy must
    reproduce the fixed strategy's text character for character. This is the
    ablation's core assumption: only the grouping differs.
    """
    entries = make_entries(n_files=4, per_file=10)
    fixed = FixedWindowStrategy(config(strategy="fixed")).segment(entries)

    start = min(e.start_seconds for e in entries)
    file_edges = [start + 30.0 * i for i in range(1, 4)]
    event = EventBoundaryStrategy(config()).segment(entries, cuts=file_edges)

    assert len(fixed) == len(event)
    for a, b in zip(fixed, event):
        assert a.text == b.text
        assert a.chunk_id == b.chunk_id, "identical text must give identical chunk ids"


def test_chunk_id_matches_openie_scheme():
    from worldmm.common.mapping import chunk_key
    entries = make_entries(n_files=1, per_file=4)
    chunk = FixedWindowStrategy(config(strategy="fixed")).segment(entries)[0]
    assert chunk.chunk_id == chunk_key(chunk.text)


def test_caption_entry_shape_is_backward_compatible():
    entries = make_entries(n_files=2, per_file=5)
    chunk = EventBoundaryStrategy(config()).segment(entries, cuts=[])[0]
    entry = chunk.to_caption_entry()
    for key in ("start_time", "end_time", "date", "text", "video_path"):
        assert key in entry, f"downstream requires {key}"
    assert isinstance(entry["video_path"], str), "video_path must stay a single string"
    assert len(entry["start_time"]) == 8 and len(entry["end_time"]) == 8
    assert entry["date"].startswith("DAY")


def test_timestamp_key_is_derivable():
    entries = make_entries(n_files=1, per_file=4)
    chunk = FixedWindowStrategy(config(strategy="fixed")).segment(entries)[0]
    assert len(chunk.timestamp_key) == 9
    assert chunk.timestamp_key.startswith("1")


def test_build_strategy_dispatches():
    assert isinstance(build_strategy(config(strategy="fixed")), FixedWindowStrategy)
    assert isinstance(build_strategy(config(strategy="event")), EventBoundaryStrategy)


def test_chunk_statistics():
    entries = make_entries(n_files=4, per_file=10)
    chunks = FixedWindowStrategy(config(strategy="fixed")).segment(entries)
    stats = chunk_statistics(chunks)
    assert stats["n_chunks"] == 4
    assert stats["duplicate_chunk_ids"] == 0
    assert stats["min_seconds"] <= stats["median_seconds"] <= stats["max_seconds"]


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
