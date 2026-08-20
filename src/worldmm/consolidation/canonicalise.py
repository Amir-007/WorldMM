"""
Light surface normalisation of triples.

Deliberately shallow. Entity disambiguation is Track B's contribution, so this
does not attempt coreference, alias resolution or entity linking - it only
folds together strings that differ by casing, whitespace, punctuation or a
leading article, which is enough to catch the exact-duplicate mass before the
embedding pass has to look at anything.

Measured on the rebuilt baseline: 15.3% of extracted triples are exact
duplicates once normalised, so this alone removes a sixth of the store at zero
model cost.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

__all__ = ["normalise_text", "canonical_key", "is_degenerate"]

_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\s\"'`.,;:!?()\[\]{}-]+|[\s\"'`.,;:!?()\[\]{}-]+$")
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)

# First-person surface forms the captioner uses interchangeably for the camera
# wearer. Folding them is safe here because every chunk is from one wearer.
_FIRST_PERSON = {"i", "me", "my", "myself", "mine"}


def normalise_text(value: object) -> str:
    """Collapse whitespace, strip edge punctuation and a leading article."""
    text = _WS_RE.sub(" ", str(value)).strip()
    text = _EDGE_PUNCT_RE.sub("", text)
    text = _ARTICLE_RE.sub("", text)
    return text.strip()


def _fold(value: object) -> str:
    text = normalise_text(value).lower()
    return "i" if text in _FIRST_PERSON else text


def canonical_key(triple: Sequence[object]) -> Tuple[str, str, str]:
    """
    A comparison key for exact-duplicate detection.

    Case-folded and punctuation-stripped, with first-person pronouns unified.
    The original surface form is kept separately; this is only for grouping.
    """
    parts = list(triple) + ["", "", ""]
    return (_fold(parts[0]), _fold(parts[1]), _fold(parts[2]))


def is_degenerate(triple: Sequence[object]) -> bool:
    """
    True only for triples that carry no retrievable content at all.

    Deliberately permissive about an empty object. Measured on the inherited
    semantic extraction, 76.9% of triples look like

        ["I", "participates in problem-solving activities", ""]

    where the model packed predicate and object into the predicate slot. Those
    still assert something and are still retrievable by embedding similarity -
    they simply contribute an isolated vertex rather than an edge to the PPR
    graph. Discarding them would delete three quarters of the semantic memory
    and show up as a spuriously good compression ratio.

    So a triple is dropped only when it asserts nothing: no subject, or no
    predicate and no object, or a self-loop.
    """
    key = canonical_key(triple)
    if not key[0]:
        return True
    if not key[1] and not key[2]:
        return True
    # A self-loop asserts nothing: ("I", "is", "I").
    return bool(key[2]) and key[0] == key[2]


def dedupe_exact(triples: Iterable[Sequence[object]]) -> List[List[str]]:
    """Drop degenerate and duplicate triples, preserving first-seen order."""
    seen = set()
    out: List[List[str]] = []
    for triple in triples:
        if is_degenerate(triple):
            continue
        key = canonical_key(triple)
        if key in seen:
            continue
        seen.add(key)
        out.append([normalise_text(p) for p in list(triple)[:3]])
    return out
