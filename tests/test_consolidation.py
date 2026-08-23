#!/usr/bin/env python3
"""
Objective 2: consolidation correctness and storage behaviour.

The embedder is injected, so these run without torch: a deterministic
stand-in maps triples that share a canonical form to identical vectors and
everything else to orthogonal ones, which exercises the clustering logic
exactly while keeping the test hermetic.

    python3 tests/test_consolidation.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.consolidation import (  # noqa: E402
    ConsolidatedTriple,
    IntervalTripleStore,
    SleepCycleConfig,
    SleepCycleConsolidator,
    cumulative_baseline_size,
    dedupe_exact,
)


class SkipTest(Exception):
    pass


def topic_embedder(topics):
    """
    Embed by declared topic: same topic -> identical vector, else orthogonal.

    `topics` maps a substring to a dimension, so a test can state which triples
    are meant to be near-duplicates without depending on a real model.
    """
    dim = max(len(topics), 1) + 1

    def embed(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            lowered = text.lower()
            for topic, axis in topics.items():
                if topic in lowered:
                    out[i, axis] = 1.0
                    break
            else:
                out[i, dim - 1] = 1.0
                out[i, i % max(1, dim - 1)] = 0.001 * (i + 1)
        return out

    return embed


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------

def test_exact_dedupe_folds_case_and_articles():
    out = dedupe_exact([["I", "hold", "the phone"], ["i", "HOLD", "phone"]])
    assert len(out) == 1


def test_only_contentless_triples_are_dropped():
    """No subject, nothing asserted, or a self-loop."""
    out = dedupe_exact([["", "hold", "phone"], ["I", "is", "me"], ["I", "", ""]])
    assert out == []


def test_empty_object_triples_are_kept():
    """
    Regression guard on a bug that would have deleted three quarters of the
    semantic memory. The extractor routinely packs predicate and object into
    the predicate slot, leaving the object empty:

        ["I", "participates in problem-solving activities", ""]

    That still asserts something and is still retrievable by similarity, so it
    must survive. Dropping these shrank the store 4x and would have been
    reported as compression.
    """
    triples = [["I", "participates in problem-solving activities", ""],
               ["I", "sways", ""]]
    assert len(dedupe_exact(triples)) == 2


# --------------------------------------------------------------------------
# interval store
# --------------------------------------------------------------------------

def test_store_folds_identical_triples_and_widens_the_interval():
    store = IntervalTripleStore()
    store.add(["I", "drink", "coffee"], "111000000")
    store.add(["i", "drink", "the coffee"], "115000000")
    assert len(store) == 1
    only = next(iter(store))
    assert only.support == 2
    assert only.first_seen == "111000000"
    assert only.last_seen == "115000000"


def test_snapshot_respects_first_seen():
    store = IntervalTripleStore()
    store.add(["I", "drink", "coffee"], "111000000")
    store.add(["I", "read", "book"], "115000000")
    assert len(store.snapshot_at("112000000")) == 1
    assert len(store.snapshot_at("116000000")) == 2


def test_materialised_format_matches_worldmm_layout():
    store = IntervalTripleStore()
    store.add(["I", "drink", "coffee"], "111000000")
    store.add(["I", "read", "book"], "115000000")
    out = store.materialise_worldmm_format(["111000000", "113000000", "116000000"])
    assert list(out) == ["111000000", "113000000", "116000000"]
    assert len(out["111000000"]["consolidated_semantic_triples"]) == 1
    assert len(out["113000000"]["consolidated_semantic_triples"]) == 1
    assert len(out["116000000"]["consolidated_semantic_triples"]) == 2


def test_store_round_trips_through_disk():
    store = IntervalTripleStore()
    store.add(["I", "drink", "coffee"], "111000000")
    store.add(["I", "drink", "coffee"], "115000000")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "store.json")
        store.write(path, metadata={"condition": "fixed"})
        again = IntervalTripleStore.read(path)
    assert len(again) == 1
    assert next(iter(again)).support == 2


def test_storage_is_linear_not_quadratic():
    """
    The Objective 2 storage claim, made executable.

    WorldMM writes the running state at every node; this writes each fact once.
    With a state that grows to N over N nodes the difference is O(N^2) vs O(N).
    """
    nodes = {f"1{i:08d}": [["I", "act", str(i)]] for i in range(200)}
    baseline = cumulative_baseline_size(nodes)
    assert baseline["final_state_triples"] == 200
    assert baseline["total_triples_written_upper_bound"] == 200 * 201 // 2
    assert baseline["write_amplification_upper_bound"] > 100

    store = IntervalTripleStore()
    for key, triples in nodes.items():
        for triple in triples:
            store.add(triple, key)
    assert len(store.to_dict()["triples"]) == 200


# --------------------------------------------------------------------------
# sleep-cycle consolidation
# --------------------------------------------------------------------------

def test_repeated_routine_collapses_to_one_supported_triple():
    """A daily routine should become one fact with support, not seven copies."""
    nodes = {}
    for day in range(1, 8):
        nodes[f"{day}08000000"] = [["I", "drink", "coffee"]]
    consolidator = SleepCycleConsolidator(topic_embedder({"coffee": 0}))
    store = consolidator.consolidate(nodes)
    assert len(store) == 1
    only = next(iter(store))
    assert only.support == 7
    assert only.first_seen == "108000000"
    assert only.last_seen == "708000000"


def test_near_duplicates_merge_but_distinct_facts_do_not():
    nodes = {
        "111000000": [["I", "drink", "coffee"], ["I", "drink", "hot coffee"]],
        "112000000": [["I", "read", "book"]],
    }
    embedder = topic_embedder({"coffee": 0, "book": 1})
    store = SleepCycleConsolidator(embedder).consolidate(nodes)
    assert len(store) == 2, [t.triple for t in store]
    supports = sorted(t.support for t in store)
    assert supports == [1, 2]


def test_consolidation_uses_no_llm_calls():
    """The cost claim: embeddings only, no generation in the loop."""
    phrasings = ["drink coffee", "drink hot coffee", "sip coffee",
                 "am drinking coffee", "have coffee"]
    nodes = {f"1{i:08d}": [["I", phrasings[i % len(phrasings)].split()[0],
                            " ".join(phrasings[i % len(phrasings)].split()[1:])]]
             for i in range(50)}
    consolidator = SleepCycleConsolidator(topic_embedder({"coffee": 0}))
    consolidator.consolidate(nodes)
    assert consolidator.report.n_llm_calls == 0
    assert consolidator.report.n_embedding_calls > 0, \
        "distinct phrasings should have required an embedding pass"


def test_identical_triples_need_no_embedding_at_all():
    """
    Exact duplicates fold before the embedding pass, so a purely repetitive
    stretch costs nothing to consolidate. WorldMM issues an LLM call per new
    triple regardless.
    """
    nodes = {f"1{i:08d}": [["I", "drink", "coffee"]] for i in range(50)}
    consolidator = SleepCycleConsolidator(topic_embedder({"coffee": 0}))
    store = consolidator.consolidate(nodes)
    assert len(store) == 1
    assert next(iter(store)).support == 50
    assert consolidator.report.n_embedding_calls == 0
    assert consolidator.report.n_llm_calls == 0


def test_compression_ratio_is_reported():
    nodes = {f"{d}08000000": [["I", "drink", "coffee"]] for d in range(1, 8)}
    consolidator = SleepCycleConsolidator(topic_embedder({"coffee": 0}))
    consolidator.consolidate(nodes)
    report = consolidator.report.to_dict()
    assert report["n_input_triples"] == 7
    assert report["n_final"] == 1
    assert report["compression_ratio"] == 7.0


def test_cycles_run_periodically():
    nodes = {f"1{i:08d}": [["I", "act", str(i)]] for i in range(20)}
    config = SleepCycleConfig(cycle_every=5)
    consolidator = SleepCycleConsolidator(topic_embedder({}), config)
    consolidator.consolidate(nodes)
    assert consolidator.report.n_cycles >= 4


def test_working_set_cap_forces_an_early_cycle():
    nodes = {f"1{i:08d}": [["I", "act", str(i)]] for i in range(30)}
    config = SleepCycleConfig(cycle_every=10_000, max_working_set=10)
    consolidator = SleepCycleConsolidator(topic_embedder({}), config)
    consolidator.consolidate(nodes)
    assert consolidator.report.n_cycles >= 2, "the cap never triggered a cycle"


def test_contentless_triples_never_enter_the_store():
    nodes = {"111000000": [["", "hold", "phone"], ["I", "is", "me"],
                           ["I", "drink", "coffee"]]}
    store = SleepCycleConsolidator(topic_embedder({"coffee": 0})).consolidate(nodes)
    assert len(store) == 1


def test_snapshot_after_consolidation_is_monotonic():
    """Retrieval semantics must survive: state never shrinks as time advances."""
    nodes = {f"1{i:08d}": [["I", "act", str(i)]] for i in range(30)}
    store = SleepCycleConsolidator(topic_embedder({}),
                                   SleepCycleConfig(cycle_every=7)).consolidate(nodes)
    sizes = [len(store.snapshot_at(k)) for k in sorted(nodes)]
    assert sizes == sorted(sizes), sizes
    assert sizes[-1] == len(store)


def test_empty_input_is_handled():
    store = SleepCycleConsolidator(topic_embedder({})).consolidate({})
    assert len(store) == 0


def test_config_validation():
    for bad in ({"similarity_threshold": 0.0}, {"similarity_threshold": 1.5},
                {"cycle_every": 0}, {"max_working_set": 0}):
        try:
            SleepCycleConfig(**bad).validate()
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")


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
