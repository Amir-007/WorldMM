#!/usr/bin/env python3
"""
Tests for multiple-choice answer extraction in the EgoLifeQA evaluation.

Motivated by a baseline run that reported 12.16% accuracy because correct
answers were being scored wrong. The original matcher anchored with re.match,
so "Final Answer: $\\boxed{A}$" captured "F" from "Final".

    python tests/test_answer_extraction.py
"""

import os
import re
import sys
from typing import Iterable, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# eval_egolife.py imports torch at module scope, so lift the two functions out
# by source rather than importing the module.
_SOURCE = open(os.path.join(_ROOT, "eval", "eval_egolife.py"), encoding="utf-8").read()


def _load(name, extra=None):
    start = _SOURCE.index(f"def {name}(")
    tail = _SOURCE[start:]
    end = len(tail)
    for marker in ("\ndef ", "\nclass "):
        idx = tail.find(marker, 1)
        if idx != -1:
            end = min(end, idx)
    namespace = {"re": re, "Optional": Optional, "Iterable": Iterable}
    namespace.update(extra or {})
    exec(compile(tail[:end], f"<{name}>", "exec"), namespace)
    return namespace[name]


extract_choice_letter = _load("extract_choice_letter")

_FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        _FAILURES.append(label)


def expect(response, expected, note=""):
    got = extract_choice_letter(response, "ABCD")
    label = f"{response!r} -> {expected}"
    check(label, got == expected, note or (f"got {got}" if got != expected else ""))


def test_shapes_seen_in_the_log():
    print("\n--- shapes observed in the baseline run log ---")
    # The exact string from the log line that read: Gold: A, Correct: False
    expect(r"Final Answer: $\boxed{A}$", "A", "the regression that caused 12.16%")
    expect(r"$\boxed{B}$", "B")
    expect(r"\boxed{C}", "C")
    expect("Final Answer: C", "C")
    expect("**(D)**", "D")
    expect("**A**", "A")


def test_bare_letter():
    print("\n--- bare letter, which is what the QA prompt asks for ---")
    # memory.py appends: "Please provide only the final answer from the
    # choices given (e.g., A, B, C, or D)." So a lone letter is expected output.
    for letter in ("A", "B", "C", "D"):
        expect(letter, letter)
    expect("A\n", "A")
    expect("  D  ", "D")
    expect("(C)", "C")
    expect("B.", "B")


def test_letter_with_text():
    print("\n--- letter followed by the option text ---")
    expect("A. The keys are on the desk", "A")
    expect("(B) The kitchen counter", "B")
    expect("The answer is C.", "C")
    expect("The correct answer is (D).", "D")
    expect("Answer: The keys are in the kitchen, so B.", "B")


def test_does_not_invent_answers():
    print("\n--- must not extract a letter that is not a choice ---")
    # "answer" appears but no option is named. Returning "T" from "this" would
    # silently score as wrong rather than being visible as unparseable.
    expect("I cannot answer this question.", None)
    expect("To answer this, the correct option is C.", "C", "picks C, not T from 'this'")
    expect("", None)
    expect("   ", None)
    check("respects the valid-letter set",
          extract_choice_letter("The answer is E.", "ABCD") is None,
          "E is not among the choices")
    check("honours a restricted choice set",
          extract_choice_letter("A", "AB") == "A")


def test_case_insensitive():
    print("\n--- case handling ---")
    expect("final answer: a", "A")
    expect("answer is d", "D")


def main():
    print("=" * 62)
    print("ANSWER EXTRACTION TESTS")
    print("=" * 62)
    test_shapes_seen_in_the_log()
    test_bare_letter()
    test_letter_with_text()
    test_does_not_invent_answers()
    test_case_insensitive()
    print("\n" + "=" * 62)
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): " + "; ".join(_FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
