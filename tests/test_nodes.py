#!/usr/bin/env python3
"""
Time-based semantic node construction.

The invariant these protect: both chunking conditions must yield nodes of the
same wall-clock duration, because WorldMM's count-based period silently
produces 5-minute nodes on the 30s grid and ~13-minute nodes at the event
condition's ~76s mean, which would confound the consolidation comparison.

    python3 tests/test_nodes.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.common.nodes import build_time_nodes  # noqa: E402


class SkipTest(Exception):
    pass


def chunk(cid, day, h, m, s, dur):
    tot = h * 3600 + m * 60 + s
    end = tot + dur
    fmt = lambda t: f"{t // 3600:02d}{(t % 3600) // 60:02d}{t % 60:02d}00"
    return {"chunk_id": cid, "date": f"DAY{day}",
            "start_time": fmt(tot), "end_time": fmt(end)}


def test_thirty_second_chunks_give_five_minute_nodes():
    chunks = [chunk(f"c{i}", 1, 11, 0, i * 30, 30) for i in range(40)]
    triples = {f"c{i}": [["I", "do", str(i)]] for i in range(40)}
    nodes = build_time_nodes(chunks, triples, period_seconds=300)
    assert len(nodes) == 4, [n.duration_seconds for n in nodes]
    for node in nodes:
        assert node.duration_seconds <= 300
        assert node.n_chunks == 10


def test_long_chunks_give_the_same_node_duration():
    """The whole point: 76s chunks must still yield ~5-minute nodes."""
    chunks = [chunk(f"c{i}", 1, 11, 0, i * 76, 76) for i in range(20)]
    triples = {f"c{i}": [["I", "do", str(i)]] for i in range(20)}
    nodes = build_time_nodes(chunks, triples, period_seconds=300)
    for node in nodes:
        assert node.duration_seconds <= 300, f"{node.duration_seconds}s exceeds the period"
    # A count-based period of 10 would have produced 760s nodes.
    assert max(n.duration_seconds for n in nodes) < 400


def test_nodes_never_span_a_day_boundary():
    chunks = [chunk("a", 1, 23, 59, 0, 30), chunk("b", 2, 0, 0, 0, 30)]
    nodes = build_time_nodes(chunks, {"a": [], "b": []}, period_seconds=3600)
    assert len(nodes) == 2
    assert nodes[0].date == "DAY1" and nodes[1].date == "DAY2"


def test_triples_are_collected_from_member_chunks():
    chunks = [chunk("a", 1, 11, 0, 0, 30), chunk("b", 1, 11, 0, 30, 30)]
    triples = {"a": [["I", "hold", "phone"]], "b": [["I", "open", "drawer"]]}
    nodes = build_time_nodes(chunks, triples, period_seconds=300)
    assert len(nodes) == 1
    assert nodes[0].triples == [["I", "hold", "phone"], ["I", "open", "drawer"]]


def test_chunk_longer_than_the_period_becomes_its_own_node():
    chunks = [chunk("a", 1, 11, 0, 0, 600), chunk("b", 1, 11, 10, 0, 30)]
    nodes = build_time_nodes(chunks, {"a": [], "b": []}, period_seconds=300)
    assert len(nodes) == 2
    assert nodes[0].chunk_ids == ["a"]


def test_node_key_is_the_last_member_timestamp():
    chunks = [chunk("a", 1, 11, 0, 0, 30), chunk("b", 1, 11, 0, 30, 30)]
    nodes = build_time_nodes(chunks, {"a": [], "b": []}, period_seconds=300)
    assert nodes[0].key == "111010000"


def test_every_chunk_lands_in_exactly_one_node():
    chunks = [chunk(f"c{i}", 1, 11, 0, i * 45, 45) for i in range(30)]
    nodes = build_time_nodes(chunks, {}, period_seconds=300)
    seen = [cid for n in nodes for cid in n.chunk_ids]
    assert sorted(seen) == sorted(c["chunk_id"] for c in chunks)
    assert len(seen) == len(set(seen))


def test_unordered_input_is_sorted():
    chunks = [chunk("b", 1, 11, 5, 0, 30), chunk("a", 1, 11, 0, 0, 30)]
    nodes = build_time_nodes(chunks, {"a": [], "b": []}, period_seconds=3600)
    assert nodes[0].chunk_ids == ["a", "b"]


def test_rejects_bad_period():
    try:
        build_time_nodes([], {}, period_seconds=0)
    except ValueError:
        return
    raise AssertionError("period_seconds=0 must raise")


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            print(f"SKIP {name}: {exc}"); skipped += 1
        except AssertionError as exc:
            print(f"FAIL {name}: {exc}"); failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {name}: {type(exc).__name__}: {exc}"); failed += 1
        else:
            print(f"ok   {name}"); passed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
