#!/usr/bin/env python3
"""
Stage 0 regression tests.

Runnable two ways:
    python3 tests/test_stage0.py      (no pytest needed)
    pytest tests/test_stage0.py

The mapping tests pin `chunk_key` against the real inherited artifacts rather
than against the original `compute_mdhash_id`, which is the stronger check:
if the hash ever drifts, the reproduction count drops and these fail loudly
instead of the pipeline failing silently and late.

Tests that need the inherited metadata skip cleanly when it is absent.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.common.checkpoint import CheckpointStore  # noqa: E402
from worldmm.common.mapping import (  # noqa: E402
    DEFAULT_EXCLUDED_DAYS,
    build_caption_chunks,
    chunk_key,
    find_collisions,
    load_caption_entries_from_zip,
    verify,
)
from worldmm.common.timestamps import (  # noqa: E402
    EgoTimestamp,
    chunk_timestamp_key,
    duration_seconds,
    to_seconds,
)

CAPTION_ZIP = os.path.join(REPO_ROOT, "data", "EgoLife", "caption.zip")
METADATA_DIR = os.path.join(REPO_ROOT, "output", "metadata")
PERSON = "A1_JAKE"
MODEL = "qwen3vl-30b"

TOTAL_CHUNKS = 6223
DAY6_CHUNKS = 1067
CHUNKS_AFTER_EXCLUSION = TOTAL_CHUNKS - DAY6_CHUNKS


class SkipTest(Exception):
    """Raised when a test's data dependency is missing."""


def _require(path: str) -> str:
    if not os.path.exists(path):
        raise SkipTest(f"missing {os.path.relpath(path, REPO_ROOT)}")
    return path


def _load_entries():
    return load_caption_entries_from_zip(_require(CAPTION_ZIP), PERSON)


def _load_metadata(name: str):
    path = os.path.join(METADATA_DIR, name)
    with open(_require(path), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------

def test_timestamp_roundtrip():
    ts = EgoTimestamp.from_key("111095800")
    assert (ts.day, ts.hour, ts.minute, ts.second, ts.frame) == (1, 11, 9, 58, 0)
    assert ts.key == "111095800"
    assert str(ts) == "DAY1 11:09:58"
    assert EgoTimestamp.from_seconds(ts.seconds) == ts


def test_timestamp_never_diff_as_integers():
    """The wrap artifact the handoff warns about: raw deltas of 3000 are 30s."""
    a, b = "111095800", "111102800"          # 11:09:58 -> 11:10:28, i.e. 30s apart
    assert int(b) - int(a) == 7000           # the misleading raw integer delta
    assert duration_seconds(a, b) == 30.0    # the real answer

    a, b = "111100000", "111103000"          # 11:10:00 -> 11:10:30
    assert int(b) - int(a) == 3000
    assert duration_seconds(a, b) == 30.0


def test_timestamp_spans_days():
    assert duration_seconds("123595900", "200000100") == 2.0


def test_timestamp_rejects_malformed():
    for bad in ("11109990", "1abc09580", "199995800"):
        try:
            EgoTimestamp.from_key(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not have parsed")

    try:
        EgoTimestamp.from_key("11095800")  # 8 digits: ambiguous, no day
    except ValueError as exc:
        assert "day digit" in str(exc)
    else:
        raise AssertionError("bare HHMMSSFF should be rejected")


def test_chunk_timestamp_key_matches_original_formula():
    """Must stay byte-identical to extract_episodic_triples.py."""
    entries = _load_entries()
    for entry in entries:
        expected = entry["date"][-1] + entry["end_time"].zfill(8)
        assert chunk_timestamp_key(entry["date"], entry["end_time"]) == expected


def test_all_inherited_timestamps_parse():
    episodic = _load_metadata(f"episodic_memory/{PERSON}/episodic_triple_results_{MODEL}.json")
    keys = list(episodic["episodic_triples"])
    assert len(keys) == TOTAL_CHUNKS
    for key in keys:
        EgoTimestamp.from_key(key)  # raises on anything malformed


def test_day6_rotation_breaks_chronological_order():
    """
    Documents the DAY6 defect from a third angle.

    Upstream 3a55b65 rotated DAY6 by 21 slots. In the pre-fix release we build
    on, that rotation wraps 21 entries from the end of DAY6 back to its start,
    leaving exactly one backwards jump in the whole 6,223-entry corpus. If this
    ever stops failing, the caption file has been swapped for the fixed one and
    the DAY6 exclusion should be revisited.
    """
    episodic = _load_metadata(f"episodic_memory/{PERSON}/episodic_triple_results_{MODEL}.json")
    keys = list(episodic["episodic_triples"])
    regressions = [
        (keys[i - 1], keys[i])
        for i in range(1, len(keys))
        if to_seconds(keys[i]) < to_seconds(keys[i - 1])
    ]
    assert len(regressions) == 1, f"expected exactly the DAY6 seam, got {regressions}"
    before, after = regressions[0]
    assert before.startswith("6") and after.startswith("6")


def test_inherited_timestamps_monotonic_once_day6_excluded():
    """The data we actually build on must be strictly chronological."""
    chunks = build_caption_chunks(_load_entries())
    seconds = [to_seconds(c.timestamp_key) for c in chunks]
    assert seconds == sorted(seconds)
    assert len(set(seconds)) == len(seconds), "duplicate timestamps after exclusion"


# --------------------------------------------------------------------------
# mapping  (Priority A)
# --------------------------------------------------------------------------

def test_chunk_key_shape():
    key = chunk_key("hello")
    assert key.startswith("chunk-") and len(key) == len("chunk-") + 32


def test_mapping_reproduces_inherited_artifacts_exactly():
    """The whole point of Priority A: no positional join, exact reproduction."""
    entries = _load_entries()
    openie = _load_metadata(f"episodic_memory/{PERSON}/openie_results_{MODEL}.json")
    episodic = _load_metadata(f"episodic_memory/{PERSON}/episodic_triple_results_{MODEL}.json")

    chunks = build_caption_chunks(entries, exclude_days=())
    assert len(chunks) == TOTAL_CHUNKS

    report = verify(chunks, openie, episodic, expect_full_coverage=True)
    assert report.hash_hits == TOTAL_CHUNKS, report.summary()
    assert not report.hash_misses, report.summary()
    assert not report.collisions, report.summary()
    assert not report.unclaimed_openie_keys, report.summary()
    assert report.triples_reproduced == TOTAL_CHUNKS, report.summary()
    assert report.ok, report.summary()


def test_no_duplicate_caption_texts():
    """Repeated caption text would collapse two timestamps onto one chunk id."""
    chunks = build_caption_chunks(_load_entries(), exclude_days=())
    assert find_collisions(chunks) == {}


def test_day6_excluded_by_default():
    entries = _load_entries()
    assert DEFAULT_EXCLUDED_DAYS == (6,)

    chunks = build_caption_chunks(entries)
    assert len(chunks) == CHUNKS_AFTER_EXCLUSION
    assert all(c.day != 6 for c in chunks)

    all_chunks = build_caption_chunks(entries, exclude_days=())
    assert sum(1 for c in all_chunks if c.day == 6) == DAY6_CHUNKS


def test_excluded_chunks_keep_original_index():
    """Indices must still line up with the inherited artifacts after exclusion."""
    entries = _load_entries()
    chunks = build_caption_chunks(entries)
    for chunk in chunks[:50] + chunks[-50:]:
        assert entries[chunk.index]["text"] == chunk.text


def test_verify_still_passes_with_day6_excluded():
    openie = _load_metadata(f"episodic_memory/{PERSON}/openie_results_{MODEL}.json")
    episodic = _load_metadata(f"episodic_memory/{PERSON}/episodic_triple_results_{MODEL}.json")
    chunks = build_caption_chunks(_load_entries())
    report = verify(chunks, openie, episodic, expect_full_coverage=False)
    assert report.ok, report.summary()
    assert report.triples_reproduced == CHUNKS_AFTER_EXCLUSION


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def test_checkpoint_records_and_resumes():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "progress.jsonl")
        with CheckpointStore(path) as store:
            store.record("a", {"triples": [["I", "hold", "phone"]]})
            store.record("b", {"triples": []})
        reopened = CheckpointStore(path)
        assert reopened.completed() == {"a", "b"}
        assert reopened.get("a")["triples"] == [["I", "hold", "phone"]]
        assert len(reopened) == 2


def test_checkpoint_pending_skips_completed():
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(os.path.join(tmp, "p.jsonl"))
        store.record("k1", 1)
        items = [{"k": "k1"}, {"k": "k2"}, {"k": "k3"}]
        remaining = list(store.pending(items, key=lambda i: i["k"]))
        assert [i["k"] for i in remaining] == ["k2", "k3"]


def test_checkpoint_survives_truncated_final_line():
    """A hard kill mid-write must cost one unit of work, not the whole run."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.jsonl")
        with CheckpointStore(path) as store:
            store.record("a", 1)
            store.record("b", 2)
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"key": "c", "val')  # killed mid-write, no trailing newline

        recovered = CheckpointStore(path)
        assert recovered.completed() == {"a", "b"}

        # The partial bytes must be gone, not merely ignored: otherwise the
        # next append lands on the same line and destroys that record too.
        with open(path, encoding="utf-8") as f:
            assert f.read().endswith("\n")

        recovered.record("c", 3)
        recovered.close()
        assert CheckpointStore(path).completed() == {"a", "b", "c"}


def test_checkpoint_compact_dedups():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.jsonl")
        store = CheckpointStore(path)
        store.record("a", 1)
        store.record("a", 2)   # re-run overwrites
        store.record("b", 3)
        assert store.compact() == 2
        with open(path, encoding="utf-8") as f:
            assert len(f.readlines()) == 2
        assert CheckpointStore(path).get("a") == 2


def test_checkpoint_write_json_respects_order():
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(os.path.join(tmp, "p.jsonl"))
        for key in ("c", "a", "b"):
            store.record(key, key.upper())
        out = os.path.join(tmp, "out.json")
        store.write_json(out, order=["a", "b", "c"])
        with open(out, encoding="utf-8") as f:
            assert list(json.load(f)) == ["a", "b", "c"]


def test_checkpoint_write_json_transform():
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(os.path.join(tmp, "p.jsonl"))
        store.record("chunk-1", {"ner": ["phone"], "triples": [["I", "hold", "phone"]]})
        out = os.path.join(tmp, "openie.json")
        store.write_json(out, transform=lambda r: {
            "ner_results": {k: v["ner"] for k, v in r.items()},
            "triple_results": {k: v["triples"] for k, v in r.items()},
        })
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["ner_results"]["chunk-1"] == ["phone"]
        assert data["triple_results"]["chunk-1"] == [["I", "hold", "phone"]]


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
