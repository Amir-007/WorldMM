#!/usr/bin/env python3
"""
Tests for evaluation checkpoint and resume.

A 500-question run costs ~30 minutes of model loading before it even starts, so
an interrupted run must keep everything it completed.

    python tests/test_eval_checkpoint.py
"""

import json
import os
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE = open(os.path.join(_ROOT, "eval", "eval_egolife.py"), encoding="utf-8").read()


def _load(*names, extra=None):
    """Lift functions out by source; eval_egolife.py imports torch at module scope."""
    ns: Dict[str, Any] = {"json": json, "os": os, "re": re,
                          "Dict": Dict, "List": List, "Any": Any}
    ns.update(extra or {})
    import logging
    ns["logger"] = logging.getLogger("test")
    for name in names:
        start = _SOURCE.index(f"def {name}(")
        tail = _SOURCE[start:]
        end = len(tail)
        for marker in ("\ndef ", "\nclass "):
            idx = tail.find(marker, 1)
            if idx != -1:
                end = min(end, idx)
        exec(compile(tail[:end], f"<{name}>", "exec"), ns)
    return tuple(ns[n] for n in names)


# FAILURE_RESPONSES is a module constant the helpers rely on.
_FR = re.search(r'FAILURE_RESPONSES = \(([^)]*)\)', _SOURCE).group(1)
is_failed_entry, load_checkpoint, append_checkpoint, summarise = _load(
    "is_failed_entry", "load_checkpoint", "append_checkpoint", "summarise",
    extra={"FAILURE_RESPONSES": tuple(eval(f"({_FR})"))})

_FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        _FAILURES.append(label)


def entry(qid, evaluate=None, abstained=False):
    return {"ID": qid, "evaluate": evaluate, "abstained": abstained, "response": "A"}


def test_roundtrip():
    print("\n--- append and reload ---")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "run.jsonl")
        check("missing file reads as empty", load_checkpoint(path) == [])
        for i in range(5):
            append_checkpoint(path, entry(f"q{i}", evaluate=(i % 2 == 0)))
        loaded = load_checkpoint(path)
        check("all entries survive", len(loaded) == 5, f"{len(loaded)}")
        check("IDs preserved", {e["ID"] for e in loaded} == {f"q{i}" for i in range(5)})
    finally:
        shutil.rmtree(tmp)


def test_truncated_line_is_survivable():
    print("\n--- crash mid-write ---")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "run.jsonl")
        for i in range(3):
            append_checkpoint(path, entry(f"q{i}", evaluate=True))
        # Simulate being killed part-way through writing the fourth record.
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"ID": "q3", "evaluate": tr')
        loaded = load_checkpoint(path)
        check("keeps the complete entries", len(loaded) == 3, f"{len(loaded)}")
        check("drops the truncated one", "q3" not in {e["ID"] for e in loaded})
    finally:
        shutil.rmtree(tmp)


def test_duplicate_ids_take_the_latest():
    print("\n--- duplicate IDs ---")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "run.jsonl")
        append_checkpoint(path, entry("q1", evaluate=False))
        append_checkpoint(path, entry("q1", evaluate=True))
        loaded = load_checkpoint(path)
        check("deduplicated", len(loaded) == 1, f"{len(loaded)}")
        check("latest wins", loaded[0]["evaluate"] is True)
    finally:
        shutil.rmtree(tmp)


def test_resume_skips_completed():
    print("\n--- resume skips completed questions ---")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "run.jsonl")
        all_questions = [{"ID": f"q{i}"} for i in range(10)]
        for i in range(4):
            append_checkpoint(path, entry(f"q{i}", evaluate=True))
        done = {r["ID"] for r in load_checkpoint(path)}
        pending = [q for q in all_questions if q["ID"] not in done]
        check("4 complete, 6 pending", len(pending) == 6, f"{len(pending)}")
        check("resumes at the first unanswered", pending[0]["ID"] == "q4", pending[0]["ID"])
    finally:
        shutil.rmtree(tmp)


def test_metrics_are_restart_invariant():
    print("\n--- metrics identical regardless of restarts ---")
    results = ([entry(f"c{i}", evaluate=True) for i in range(6)] +
               [entry(f"w{i}", evaluate=False) for i in range(3)] +
               [entry(f"a{i}", evaluate=None, abstained=True) for i in range(2)])
    stats = summarise(results)
    check("total counts everything", stats["total"] == 11, str(stats["total"]))
    check("abstentions excluded from answered", stats["answered"] == 9, str(stats["answered"]))
    check("correct counted", stats["correct"] == 6, str(stats["correct"]))
    check("wrong excludes abstentions", stats["wrong"] == 3, str(stats["wrong"]))
    check("accuracy over answered only", abs(stats["accuracy_answered"] - 6/9) < 1e-9,
          f"{stats['accuracy_answered']:.4f}")
    check("accuracy overall dilutes by abstentions", abs(stats["accuracy_overall"] - 6/11) < 1e-9,
          f"{stats['accuracy_overall']:.4f}")
    check("abstention rate", abs(stats["abstention_rate"] - 2/11) < 1e-9)
    check("hallucination rate excludes abstentions", abs(stats["hallucination_rate"] - 3/11) < 1e-9)

    # Order must not matter, since a resumed file interleaves old and new.
    import random
    shuffled = results[:]
    random.shuffle(shuffled)
    check("order independent", summarise(shuffled) == stats)

    # Splitting the same work across three restarts must give the same numbers.
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "run.jsonl")
        for chunk in (results[:4], results[4:7], results[7:]):
            for e in chunk:
                append_checkpoint(path, e)
        check("three restarts equal one pass", summarise(load_checkpoint(path)) == stats)
    finally:
        shutil.rmtree(tmp)


def test_run_tag_separates_arms():
    print("\n--- ablation arms do not collide ---")
    tags = []
    for enable_abstention, enable_spatial, threshold in (
            (False, False, 0.75), (False, True, 0.75), (True, True, 0.75), (True, True, 0.6)):
        if enable_abstention:
            tag = f"abstain{threshold:g}"
        elif enable_spatial:
            tag = "spatial"
        else:
            tag = "baseline"
        tags.append(f"egolife_eval_A1_JAKE_{tag}.jsonl")
    check("all four arms get distinct files", len(set(tags)) == 4, str(tags))
    check("threshold appears in the tag", "egolife_eval_A1_JAKE_abstain0.6.jsonl" in tags)


def test_failures_are_retried_not_frozen():
    print("\n--- OOM failures must be retried, not baked in ---")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "run.jsonl")
        append_checkpoint(path, {"ID": "q1", "response": "A", "evaluate": True,
                                 "abstained": False, "failed": False})
        # memory.py's fallback when generation raises
        append_checkpoint(path, {"ID": "q2", "response": "Unable to generate answer",
                                 "evaluate": None, "abstained": False, "failed": True})
        # the eval loop's own fallback when answer() raises
        append_checkpoint(path, {"ID": "q3", "response": "Error",
                                 "evaluate": None, "abstained": False, "failed": True})
        loaded = load_checkpoint(path)
        completed = {r["ID"] for r in loaded if not is_failed_entry(r)}
        check("successful question counts as done", "q1" in completed)
        check("'Unable to generate answer' is retryable", "q2" not in completed)
        check("'Error' is retryable", "q3" not in completed)

        all_q = [{"ID": f"q{i}"} for i in (1, 2, 3, 4)]
        pending = [q for q in all_q if q["ID"] not in completed]
        check("failed questions reappear as pending",
              [q["ID"] for q in pending] == ["q2", "q3", "q4"],
              str([q["ID"] for q in pending]))

        # A successful retry must supersede the failure.
        append_checkpoint(path, {"ID": "q2", "response": "B", "evaluate": True,
                                 "abstained": False, "failed": False})
        loaded = load_checkpoint(path)
        check("successful retry supersedes the failure",
              "q2" in {r["ID"] for r in loaded if not is_failed_entry(r)})
    finally:
        shutil.rmtree(tmp)


def test_failures_excluded_from_accuracy():
    print("\n--- failures must not deflate accuracy ---")
    results = [
        {"ID": "a", "response": "A", "evaluate": True,  "abstained": False, "failed": False},
        {"ID": "b", "response": "B", "evaluate": False, "abstained": False, "failed": False},
        {"ID": "c", "response": "Unable to generate answer", "evaluate": None,
         "abstained": False, "failed": True},
        {"ID": "d", "response": "Error", "evaluate": None, "abstained": False, "failed": True},
    ]
    stats = summarise(results)
    check("failures counted separately", stats["failed"] == 2, str(stats["failed"]))
    check("answered excludes failures", stats["answered"] == 2, str(stats["answered"]))
    check("accuracy not deflated by OOM", abs(stats["accuracy_answered"] - 0.5) < 1e-9,
          f"{stats['accuracy_answered']:.4f} (would be 0.25 if failures counted wrong)")
    check("wrong excludes failures", stats["wrong"] == 1, str(stats["wrong"]))


def main():
    print("=" * 62)
    print("EVAL CHECKPOINT / RESUME TESTS")
    print("=" * 62)
    test_roundtrip()
    test_truncated_line_is_survivable()
    test_duplicate_ids_take_the_latest()
    test_resume_skips_completed()
    test_metrics_are_restart_invariant()
    test_run_tag_separates_arms()
    test_failures_are_retried_not_frozen()
    test_failures_excluded_from_accuracy()
    print("\n" + "=" * 62)
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): " + "; ".join(_FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
