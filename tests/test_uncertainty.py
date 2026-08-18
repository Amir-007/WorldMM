#!/usr/bin/env python3
"""
Tests for the uncertainty-aware retrieval loop.

Covers the confidence function, the abstention rule, question phrasing, and the
end-to-end decision against the real Entity ID bank.

WorldMemory itself is not instantiated here: it pulls in torch and an embedding
model. What is verified is the decision logic it delegates to, plus a static
check that the loop is wired so a disabled bank cannot change baseline behaviour.

    python tests/test_uncertainty.py
"""

import os
import re
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _import_spatial():
    for name, path in (
        ("worldmm", os.path.join(_SRC, "worldmm")),
        ("worldmm.memory", os.path.join(_SRC, "worldmm", "memory")),
        ("worldmm.memory.spatial", os.path.join(_SRC, "worldmm", "memory", "spatial")),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module
    from worldmm.memory.spatial import confidence as conf
    from worldmm.memory.spatial.memory import SpatialMemory
    return conf, SpatialMemory


C, SpatialMemory = _import_spatial()

BANK = os.path.join(_ROOT, "output/metadata/spatial_memory/A1_JAKE/entity_ids.json")
MEMORY_PY = os.path.join(_SRC, "worldmm", "memory", "memory.py")

_FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        _FAILURES.append(label)


def test_confidence_function():
    print("\n--- confidence function ---")
    f = C.compute_confidence
    check("single candidate is always certain", f(1, 0.0) == 1.0)
    check("zero candidates is certain", f(0, 0.0) == 1.0)
    check("two candidates, zero margin is minimal", f(2, 0.0) == 0.0, str(f(2, 0.0)))
    check("confidence rises with margin", f(2, 0.05) < f(2, 0.15) < f(2, 0.30),
          f"{f(2,0.05):.3f} < {f(2,0.15):.3f} < {f(2,0.30):.3f}")
    check("saturates at the saturation margin", f(2, C.MARGIN_SATURATION) == 1.0)
    check("does not exceed 1.0", f(2, 10.0) == 1.0)
    check("more rivals lowers confidence at equal margin", f(5, 0.30) < f(2, 0.30),
          f"{f(5,0.30):.3f} < {f(2,0.30):.3f}")
    check("negative margin clamps to zero", f(2, -0.5) == 0.0)


def test_abstention_rule():
    print("\n--- abstention rule ---")
    ambiguous_low = C.AmbiguityAssessment(confidence=0.20, n_matching_entity_ids=3, margin=0.03)
    ambiguous_high = C.AmbiguityAssessment(confidence=0.95, n_matching_entity_ids=3, margin=0.30)
    single_low = C.AmbiguityAssessment(confidence=0.10, n_matching_entity_ids=1, margin=0.0)

    check("abstains on low confidence with 2+ matches", C.should_abstain(ambiguous_low))
    check("answers on high confidence", not C.should_abstain(ambiguous_high))
    check("does not abstain on a single match even at low confidence",
          not C.should_abstain(single_low),
          "both conditions are required, otherwise weak matches over-trigger")
    check("threshold is configurable",
          C.should_abstain(ambiguous_high, threshold=0.99))
    check("default threshold is the scoped 0.75", C.DEFAULT_CONFIDENCE_THRESHOLD == 0.75)


def test_question_phrasing():
    print("\n--- disambiguation question ---")
    two = C.AmbiguityAssessment(confidence=0.1, n_matching_entity_ids=2, margin=0.01,
                                surface_form="knife", locations=["kitchen", "living_room"])
    text = C.format_disambiguation_question(two)
    check("names the entity", "knife" in text, text)
    check("names both locations", "kitchen" in text and "living room" in text, text)
    check("underscores are humanised", "living_room" not in text, text)
    check("is phrased as a question", text.strip().endswith("?"), text)

    three = C.AmbiguityAssessment(confidence=0.1, n_matching_entity_ids=3, margin=0.01,
                                  surface_form="plate",
                                  locations=["kitchen", "living_room", "store"])
    text3 = C.format_disambiguation_question(three)
    check("handles three locations", all(p in text3 for p in ("kitchen", "living room", "store")), text3)


def test_no_false_trigger_on_substrings():
    print("\n--- substring false-trigger regression ---")
    if not os.path.exists(BANK):
        print("  [SKIP] entity bank not built")
        return
    memory = SpatialMemory()
    memory.load_entities_from_file(BANK)
    memory.index(10 ** 9)

    # "hat" must not match inside "what". This fired disambiguation on questions
    # containing no such entity before word-level matching was introduced.
    for query in ("what did I say to Tasha", "what was the weather like",
                  "did I finish the report"):
        candidates = memory.disambiguation_candidates(query)
        locations = [memory.entities[e].location_label for e, _ in candidates]
        surface = memory.entities[candidates[0][0]].surface_form if candidates else None
        assessment = C.assess(candidates, locations, surface)
        check(f"no abstention on {query!r}", not C.should_abstain(assessment),
              f"n={assessment.n_matching_entity_ids} surface={surface!r}")


def test_end_to_end_decisions():
    print("\n--- end-to-end decisions on the real bank ---")
    if not os.path.exists(BANK):
        print("  [SKIP] entity bank not built")
        return
    memory = SpatialMemory()
    memory.load_entities_from_file(BANK)
    memory.index(10 ** 9)

    def decide(query):
        candidates = memory.disambiguation_candidates(query)
        pool = memory.indexed_entities
        locations = [pool[e].location_label for e, _ in candidates]
        surface = pool[candidates[0][0]].surface_form if candidates else None
        assessment = C.assess(candidates, locations, surface)
        return C.should_abstain(assessment), assessment

    for query in ("I need the plate", "where did I leave the knife", "the mango"):
        abstain, assessment = decide(query)
        check(f"abstains on unqualified {query!r}", abstain,
              f"conf={assessment.confidence:.3f} n={assessment.n_matching_entity_ids}")

    for query in ("where is the plate in the kitchen",
                  "where is the knife in the kitchen",
                  "where is the mango in the kitchen"):
        abstain, assessment = decide(query)
        check(f"answers when location is named {query!r}", not abstain,
              f"conf={assessment.confidence:.3f}")


def test_baseline_is_unaffected():
    print("\n--- baseline arm safety (static wiring checks) ---")
    source = open(MEMORY_PY, encoding="utf-8").read()
    check("abstention defaults to off",
          re.search(r"enable_abstention:\s*bool\s*=\s*False", source) is not None)
    check("assess returns None when disabled",
          "if not self.enable_abstention or not self.spatial_memory.entities:" in source)
    check("spatial index is skipped when no bank is loaded",
          "if self.spatial_memory.entities:" in source)
    check("abstention is checked before answering",
          source.index("should_abstain") < source.index("Generating answer from accumulated context"))
    check("QAResult still carries the original fields",
          all(f in source for f in ("retrieved_items=retrieved_items",
                                    "round_history=round_history",
                                    "num_rounds=round_num")))


def main():
    print("=" * 62)
    print("UNCERTAINTY-AWARE LOOP TESTS")
    print("=" * 62)
    test_confidence_function()
    test_abstention_rule()
    test_question_phrasing()
    test_no_false_trigger_on_substrings()
    test_end_to_end_decisions()
    test_baseline_is_unaffected()
    print("\n" + "=" * 62)
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): " + ", ".join(_FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
