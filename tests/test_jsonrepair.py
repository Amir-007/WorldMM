#!/usr/bin/env python3
"""
JSON salvage tests, built from real Qwen3-VL-30B-A3B failures.

The three fixtures below are taken from the warnings emitted during a 50-chunk
calibration run on Eureka. Each one previously produced zero triples, and 4 of
50 chunks (8%) hit one of them; across the inherited 5,156-chunk build set,
680 chunks (13.2%) came back empty for the same reasons.

    python3 tests/test_jsonrepair.py
    pytest tests/test_jsonrepair.py
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.common.jsonrepair import fix_broken_generated_json, salvage_json  # noqa: E402
from worldmm.common.schema_coercion import coerce_to_schema  # noqa: E402


# --- fixture 1: generation hit the 512-token cap mid-element ---------------
TRUNCATED = '''```json
{
  "triples": [
    ["I", "turns head to", "right"],
    ["I", "rocks body", ""],
    ["Shure", "says", "\\"Anshan Steel\\""],
    ["I", "watches everyone", ""],
    ["I", "says", "\\"Yeah, you can tell it's Chinese,\\""],
    ["Shure", "proposes'''

# --- fixture 2: stray ')' where a ']' belonged ------------------------------
MALFORMED = '''```json
{
  "triples": [
    ["Lucia", "says", "Athletes also need to check that."],
    ["Tasha", "laughs"],
    ["I", "hands", "tape measure to Alice"],
    ["Alice", "takes", "tape measure"],
    ["Tasha", "confirms", "there was a batch of...")
  ]
}
```'''

# --- fixture 3: the model reasoned aloud instead of answering --------------
PROSE = '''Since there are no explicit named entities provided in the input (`[""]` implies
none listed), we will infer relevant concepts based on context while ensuring all
elements remain grounded in the text.

However, since **the instruction says** *"Each triple should contain at least one,
but preferably two, of the named entities"* and *no named entities were supplied*,
this creates ambiguity.

Thus, strictly speaking, **there are zero valid named entities**, so any inclusion
would be specu'''

# --- fixture 4: prose wrapped around a usable object -----------------------
PROSE_WITH_JSON = '''Here is my analysis of the passage. The camera wearer is
clearly the subject throughout, so I will use "I".

{"triples": [["I", "hold", "phone"], ["I", "open", "drawer"]]}

Let me know if you would like these refined further.'''


class SkipTest(Exception):
    pass


def test_clean_json_is_untouched():
    value, how = salvage_json('{"triples": [["I", "hold", "phone"]]}')
    assert how == "clean"
    assert value["triples"] == [["I", "hold", "phone"]]


def test_fenced_json_is_unwrapped():
    value, how = salvage_json('```json\n{"triples": [["I", "hold", "phone"]]}\n```')
    assert how == "fenced"
    assert len(value["triples"]) == 1


def test_truncated_output_is_repaired_not_discarded():
    """Fixture 1: recover the complete elements instead of returning nothing."""
    value, how = salvage_json(TRUNCATED)
    assert how == "repaired", f"expected repair, got {how}"
    triples = value["triples"]
    assert len(triples) >= 4, f"salvaged too few: {triples}"
    assert ["I", "turns head to", "right"] in triples
    assert ["Shure", "says", '"Anshan Steel"'] in triples
    # The half-written final element must not survive.
    assert all(isinstance(t, list) for t in triples)
    assert not any("proposes" in str(t) for t in triples)


def test_malformed_syntax_is_repaired():
    """Fixture 2: one bad element must not cost the other four."""
    value, how = salvage_json(MALFORMED)
    assert how == "repaired", f"expected repair, got {how}"
    triples = value["triples"]
    well_formed = [t for t in triples if isinstance(t, list) and len(t) == 3]
    assert len(well_formed) >= 3, f"salvaged too few well-formed triples: {triples}"
    assert ["Alice", "takes", "tape measure"] in well_formed


def test_prose_without_json_fails_loudly():
    """Fixture 3: nothing to salvage, so the caller must be told to retry."""
    value, how = salvage_json(PROSE)
    assert how == "failed"
    assert value is None


def test_json_buried_in_prose_is_extracted():
    value, how = salvage_json(PROSE_WITH_JSON)
    assert how in ("sliced", "repaired"), f"got {how}"
    assert value["triples"] == [["I", "hold", "phone"], ["I", "open", "drawer"]]


def test_trailing_comma_is_tolerated():
    value, how = salvage_json('{"triples": [["I", "hold", "phone"],]}')
    assert how != "failed"
    assert value["triples"][0] == ["I", "hold", "phone"]


def test_line_comments_are_stripped():
    value, how = salvage_json('{\n  "named_entities": ["phone"] // the device\n}')
    assert how != "failed"
    assert value["named_entities"] == ["phone"]


def test_empty_input_fails():
    for bad in ("", "   ", None):
        value, how = salvage_json(bad)
        assert how == "failed" and value is None


def test_fix_broken_generated_json_leaves_valid_input_alone():
    valid = '{"a": [1, 2, 3]}'
    assert fix_broken_generated_json(valid) == valid


def test_fix_broken_generated_json_ignores_brackets_inside_strings():
    """
    A '[' inside a string literal must not be counted as an open bracket.

    The half-written element survives as a short list rather than being
    dropped here; discarding it is `filter_invalid_triples`' job, since this
    layer repairs syntax and must not invent or apply semantics.
    """
    text = '{"triples": [["I", "say", "a [bracket] here"], ["I", "wave", "'
    parsed = json.loads(fix_broken_generated_json(text))
    triples = parsed["triples"]
    assert ["I", "say", "a [bracket] here"] in triples, \
        f"the complete triple was lost: {triples}"
    for triple in triples:
        assert len(triple) <= 3, f"repair fabricated content: {triple}"
    assert [t for t in triples if len(t) == 3] == [["I", "say", "a [bracket] here"]]


def test_repair_is_idempotent():
    once = fix_broken_generated_json(TRUNCATED.split("```json\n")[1])
    twice = fix_broken_generated_json(once)
    assert json.loads(once) == json.loads(twice)


# --------------------------------------------------------------------------
# schema coercion
# --------------------------------------------------------------------------

def test_bare_list_is_wrapped():
    out = coerce_to_schema([["I", "hold", "phone"]], ["triples"])
    assert out == {"triples": [["I", "hold", "phone"]]}


def test_bare_list_left_alone_for_multi_field_schema():
    data = [["I", "hold", "phone"]]
    assert coerce_to_schema(data, ["triples", "evidence"]) == data


def test_entities_wrapped_in_dicts_are_unwrapped():
    for wrapper in ("entity", "name", "value", "text"):
        out = coerce_to_schema({"named_entities": [{wrapper: "phone"}]}, ["named_entities"])
        assert out["named_entities"] == ["phone"], f"failed for {wrapper!r}"


def test_dict_element_with_extra_keys_prefers_the_known_key():
    out = coerce_to_schema(
        {"named_entities": [{"entity": "phone", "type": "device"}]}, ["named_entities"]
    )
    assert out["named_entities"] == ["phone"]


def test_dict_element_with_no_known_key_falls_back_to_first_value():
    out = coerce_to_schema({"named_entities": [{"thing": "phone"}]}, ["named_entities"])
    assert out["named_entities"] == ["phone"]


def test_wrong_single_key_name_is_renamed():
    out = coerce_to_schema({"entities": ["phone"]}, ["named_entities"])
    assert out == {"named_entities": ["phone"]}


def test_correct_key_is_not_renamed():
    data = {"named_entities": ["phone"]}
    assert coerce_to_schema(dict(data), ["named_entities"]) == data


def test_nulls_in_triples_become_empty_strings():
    out = coerce_to_schema({"triples": [["I", "hold", None]]}, ["triples"])
    assert out["triples"] == [["I", "hold", ""]]


def test_nulls_in_flat_lists_become_empty_strings():
    out = coerce_to_schema({"named_entities": ["phone", None]}, ["named_entities"])
    assert out["named_entities"] == ["phone", ""]


def test_integer_fields_are_not_corrupted():
    """Quirk 4 must only touch lists whose elements are all lists."""
    data = {"episodic_evidence": [0, 1, 2]}
    assert coerce_to_schema(dict(data), ["episodic_evidence"]) == data


def test_well_formed_input_passes_through_untouched():
    data = {"triples": [["I", "hold", "phone"], ["I", "open", "drawer"]]}
    assert coerce_to_schema(dict(data), ["triples"]) == data


def test_coercion_does_not_invent_fields():
    out = coerce_to_schema({"triples": [], "extra": 1}, ["triples"])
    assert set(out) == {"triples", "extra"}


def test_end_to_end_salvage_then_coerce():
    """
    The two stages compose: repair the syntax, then fix the shape.

    The fixture is truncated mid-element with a null earlier in the list, so
    repair must drop only the unfinished tail while coercion still normalises
    the surviving null. A wrong key name exercises the rename at the same time.
    """
    value, how = salvage_json('```json\n{"entities": [null, "phone", "dining tab')
    assert how == "repaired", f"got {how}"
    out = coerce_to_schema(value, ["named_entities"])
    assert out == {"named_entities": ["", "phone"]}, out


def test_truncated_tail_is_discarded_not_guessed():
    """Content after the final comma was never finished and must not survive."""
    value, how = salvage_json('{"entities": ["phone", nul')
    assert how == "repaired"
    assert value == {"entities": ["phone"]}


def test_nested_list_in_a_triple_slot_is_flattened():
    """
    Observed on Eureka: the model wraps an object phrase one level too deep,
    e.g. ["Katrina", "mentions", ["Katrina", "mentions buying items"]].
    That failed schema validation and cost the chunk all of its triples.
    """
    data = {"triples": [
        ["Katrina", "mentions", ["Katrina", "mentions buying items after lunchtime"]],
        ["I", "hold", "phone"],
    ]}
    out = coerce_to_schema(data, ["triples"])
    assert out["triples"][0] == [
        "Katrina", "mentions", "Katrina mentions buying items after lunchtime"
    ]
    assert out["triples"][1] == ["I", "hold", "phone"], "well-formed rows must not change"
    for row in out["triples"]:
        assert all(isinstance(cell, str) for cell in row)


def test_nested_list_with_nulls_is_flattened_cleanly():
    out = coerce_to_schema({"triples": [["I", "say", [None, "hello", None]]]}, ["triples"])
    assert out["triples"][0][2] == "hello"


def test_deeply_nested_cell_is_flattened():
    out = coerce_to_schema({"triples": [["I", "see", [["a", "b"], "c"]]]}, ["triples"])
    assert out["triples"][0][2] == "a b c"


def test_flattening_does_not_touch_integer_evidence_lists():
    data = {"episodic_evidence": [0, 1, 2]}
    assert coerce_to_schema(dict(data), ["episodic_evidence"]) == data


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
