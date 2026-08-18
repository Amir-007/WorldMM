#!/usr/bin/env python3
"""
Tests for the spatial (Entity ID) memory bank.

Self-contained: no pytest, no GPU, no model. Run directly.

    python tests/test_spatial_memory.py

Unit checks always run. Integration checks need the A1_JAKE captions and OpenIE
results, which are not in git; if those are absent the integration block is
skipped rather than failed, so a fresh clone still passes.
"""

import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _import_spatial():
    """
    Import the spatial submodules without executing the parent packages.

    worldmm/memory/__init__.py imports WorldMemory, which needs torch and
    tenacity. The spatial bank is pure Python by design so it can be tested on a
    laptop, so we stand in for the parents and load the real submodules under them.
    """
    for name, path in (
        ("worldmm", os.path.join(_SRC, "worldmm")),
        ("worldmm.memory", os.path.join(_SRC, "worldmm", "memory")),
        ("worldmm.memory.spatial", os.path.join(_SRC, "worldmm", "memory", "spatial")),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module
    from worldmm.memory.spatial import build as build_mod
    from worldmm.memory.spatial import location_vocab as vocab_mod
    from worldmm.memory.spatial import utils as utils_mod
    from worldmm.memory.spatial.memory import SpatialMemory
    return build_mod, vocab_mod, utils_mod, SpatialMemory


build, vocab, utils, SpatialMemory = _import_spatial()

CAPTION_FILE = os.path.join(_ROOT, "data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json")
OPENIE_FILE = os.path.join(_ROOT, "output/metadata/episodic_memory/A1_JAKE/"
                                  "openie_results_qwen3vl-30b.json")

_FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        _FAILURES.append(label)


# --------------------------------------------------------------------- units

def test_surface_form_normalisation():
    print("\n--- surface form normalisation ---")
    n = utils.normalise_surface_form
    check("strips leading determiners", n("the Power Bank") == "power bank", n("the Power Bank"))
    check("strips possessives", n("my backpack") == "backpack", n("my backpack"))
    check("strips stacked determiners", n("the my  knife") == "knife", repr(n("the my  knife")))
    check("strips quotes and punctuation", n('"hard drive."') == "hard drive", repr(n('"hard drive."')))
    check("rejects pronouns", n("it") is None)
    check("rejects body parts", n("my hands") is None)
    check("rejects mass nouns", n("the time") is None)
    check("rejects long clause fragments", n("items on the dining table over there") is None)
    check("rejects empties", n("   ") is None)


def test_location_cue():
    print("\n--- location cue ---")
    check("single keyword matches", vocab.location_cue("I walk into the kitchen.") == "kitchen")
    check("is case insensitive", vocab.location_cue("The KITCHEN is warm.") == "kitchen")
    check("no keyword gives None", vocab.location_cue("I pick up the thing.") is None)
    check("two locations give None (ambiguous)",
          vocab.location_cue("I leave the kitchen and enter the bedroom.") is None)
    check("synonyms map to one label", vocab.location_cue("I sit on the sofa.") == "living_room")


def test_timestamps():
    print("\n--- timestamps ---")
    ts = utils.caption_timestamp
    check("builds day + zero-padded time", ts("DAY1", "11094300") == 111094300, str(ts("DAY1", "11094300")))
    check("zero-pads short times", ts("DAY2", "94300") == 200094300, str(ts("DAY2", "94300")))
    # The upstream episodic reformat builds this key with date[-1], the last
    # character of the date string, so DAY10 degrades to "0". We use the full
    # numeric day. EgoLife has 7 days so both agree today, but the difference
    # matters if the data grows.
    check("uses the full day number, not date[-1]",
          ts("DAY10", "11094300") == 1011094300, str(ts("DAY10", "11094300")))
    check("DAY10 and DAY1 stay distinct for valid 8-digit times",
          ts("DAY10", "11094300") != ts("DAY1", "11094300"),
          f"{ts('DAY10','11094300')} vs {ts('DAY1','11094300')}")
    check("renders human readable", utils.transform_timestamp(111094300) == "DAY1 11:09:43",
          utils.transform_timestamp(111094300))


def test_mention_extraction():
    print("\n--- mention extraction from triples ---")
    extract = build.extract_mentions
    check("keeps ego manipulation triples",
          extract([["I", "pick up", "the knife"]]) == ["knife"])
    check("drops speech predicates",
          extract([["I", "says", "the knife"]]) == [])
    check("drops non-ego subjects",
          extract([["Tasha", "pick up", "the knife"]]) == [])
    check("accepts 'me' as ego", extract([["me", "hold", "a mango"]]) == ["mango"])
    check("is case insensitive", extract([["I", "PICK UP", "The Knife"]]) == ["knife"])
    check("drops malformed triples", extract([["I", "pick up"], [], ["a", "b", "c", "d"]]) == [])
    check("drops normalised-away objects", extract([["I", "hold", "it"]]) == [])


def test_carry_forward():
    print("\n--- carry-forward labelling ---")
    caps = [
        {"text": "I walk into the kitchen."},   # 0 kitchen
        {"text": "I pick up the power bank."},  # 1 inherits
        {"text": "I plug it in."},              # 2 inherits
        {"text": "I wait."},                    # 3 inherits (gap 3)
        {"text": "I keep waiting."},            # 4 gap 4
        {"text": "I still wait."},              # 5 gap 5 -> beyond window 4
    ]
    labels = build.assign_locations(caps, carry_forward=4)
    check("direct match labelled", labels.get(0) == "kitchen")
    check("adjacent caption inherits", labels.get(1) == "kitchen")
    check("inherits within window", labels.get(4) == "kitchen")
    check("stops beyond window", 5 not in labels, str(labels.get(5)))

    strict = build.assign_locations(caps, carry_forward=0)
    check("window 0 labels only direct matches", set(strict) == {0}, str(sorted(strict)))

    ambiguous = build.assign_locations([
        {"text": "I am in the kitchen."},
        {"text": "I move from the kitchen to the bedroom."},
        {"text": "I sit down."},
    ], carry_forward=4)
    check("ambiguous caption clears the carried label",
          2 not in ambiguous, str(sorted(ambiguous)))


# -------------------------------------------------------------- integration

def test_integration():
    print("\n--- integration (real A1_JAKE data) ---")
    bank, stats = build.build_entity_bank(CAPTION_FILE, OPENIE_FILE, carry_forward=4)
    check("bank is non-empty", len(bank) > 0, f"{len(bank)} entity ids")
    check("usable ambiguity budget clears 20",
          stats["usable_ambiguity_budget"] >= 20, str(stats["usable_ambiguity_budget"]))

    memory = SpatialMemory()
    memory.load_entities_from_data({"stats": stats,
                                    "entities": {k: v.to_dict() for k, v in bank.items()}})
    memory.index(10 ** 9)

    ranked = memory.query("where did I leave the knife?")
    check("query returns hits", len(ranked) > 0, f"{len(ranked)} hits")
    check("best hit is the queried object",
          memory.entities[ranked[0][0]].surface_form == "knife",
          memory.entities[ranked[0][0]].surface_form)

    candidates = memory.disambiguation_candidates("where is the knife?")
    forms = {memory.entities[e].surface_form for e, _ in candidates}
    locations = {memory.entities[e].location_label for e, _ in candidates}
    check("candidates share one surface form", len(forms) == 1, str(forms))
    check("candidates span multiple locations", len(locations) >= 2, str(sorted(locations)))

    # Naming the location must sharpen the margin; this is the Objective 5 signal.
    vague = memory.disambiguation_candidates("I need the plate")
    precise = memory.disambiguation_candidates("where is the plate in the kitchen")
    vague_margin = vague[0][1] - vague[1][1]
    precise_margin = precise[0][1] - precise[1][1]
    check("naming the location widens the top1-top2 margin",
          precise_margin > vague_margin,
          f"vague={vague_margin:.4f} precise={precise_margin:.4f}")

    # Time gate: nothing observed after the boundary may be visible.
    records = list(bank.values())
    earliest = min(r.first_seen for r in records)
    memory.reset_index()
    memory.index(earliest)
    leaked = [r for r in memory.indexed_entities.values() if r.first_seen > earliest]
    check("no post-boundary entity is visible", not leaked, f"{len(leaked)} leaked")
    check("early boundary exposes fewer than full", memory.get_indexed_count() < len(bank),
          f"{memory.get_indexed_count()} < {len(bank)}")

    # Mention counts must also be truncated, not just membership.
    busiest = max(records, key=lambda r: len(r.mention_times))
    midpoint = busiest.mention_times[len(busiest.mention_times) // 2]
    memory.reset_index()
    memory.index(midpoint)
    expected = sum(1 for t in busiest.mention_times if t <= midpoint)
    check("mention count truncated to the past",
          memory.indexed_mention_counts.get(busiest.entity_id) == expected,
          f"got {memory.indexed_mention_counts.get(busiest.entity_id)}, expected {expected}, "
          f"total {busiest.n_mentions}")


def main():
    print("=" * 62)
    print("SPATIAL MEMORY TESTS")
    print("=" * 62)

    test_surface_form_normalisation()
    test_location_cue()
    test_timestamps()
    test_mention_extraction()
    test_carry_forward()

    if os.path.exists(CAPTION_FILE) and os.path.exists(OPENIE_FILE):
        test_integration()
    else:
        print("\n--- integration (real A1_JAKE data) ---")
        print("  [SKIP] data not present locally; unit checks still ran")
        for path in (CAPTION_FILE, OPENIE_FILE):
            if not os.path.exists(path):
                print(f"         missing: {os.path.relpath(path, _ROOT)}")

    print("\n" + "=" * 62)
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): " + ", ".join(_FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
