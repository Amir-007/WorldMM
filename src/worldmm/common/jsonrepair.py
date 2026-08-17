"""
Salvage JSON from local-VLM output.

Qwen3-VL emits JSON that is usually valid and occasionally is not. Three
failure modes were observed in a 50-chunk calibration run against
Qwen3-VL-30B-A3B, all three of which previously produced an empty result:

1. Truncation - the generation hits `max_new_tokens` mid-array, leaving
   unclosed brackets and a half-written element.
2. Malformed syntax - a stray `)` or an unbalanced quote inside an otherwise
   well-formed object.
3. Prose instead of JSON - the model reasons aloud about the task and buries
   the object (or emits none at all).

`salvage_json` walks escalating repairs and reports whether it had to work for
the result, so callers can distinguish a clean parse from a rescued one and
count how often rescue was needed.

The brace-balancing repair is adapted from
`worldmm.memory.episodic.utils.fix_broken_generated_json`, which upstream wrote
but never wired into any call path - it has no callers anywhere in the repo,
which is why truncated output was being discarded rather than repaired. That
copy is left in place untouched; this is the one that actually runs.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple

__all__ = ["fix_broken_generated_json", "salvage_json"]

_LINE_COMMENT_RE = re.compile(r"//[^\n\"]*(?=\n|$)")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _find_unclosed(json_str: str) -> List[str]:
    """Brackets opened and never closed, outermost first, ignoring string literals."""
    unclosed: List[str] = []
    inside_string = False
    escape_next = False

    for char in json_str:
        if inside_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"':
                inside_string = False
        else:
            if char == '"':
                inside_string = True
            elif char in "{[":
                unclosed.append(char)
            elif char in "}]":
                if unclosed and ((char == "}" and unclosed[-1] == "{")
                                 or (char == "]" and unclosed[-1] == "[")):
                    unclosed.pop()

    return unclosed


def fix_broken_generated_json(json_str: str) -> str:
    """
    Close a truncated JSON string by discarding the partial tail and balancing
    brackets. Returns the input untouched if it already parses.
    """
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    # The text after the final comma is the element that was still being
    # written when generation stopped, so it cannot be recovered.
    last_comma = json_str.rfind(",")
    if last_comma != -1:
        json_str = json_str[:last_comma]

    closing = {"{": "}", "[": "]"}
    for open_char in reversed(_find_unclosed(json_str)):
        json_str += closing[open_char]

    return json_str


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1:] if newline != -1 else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _outermost_object(text: str) -> Optional[str]:
    """The widest {...} span, for output with prose wrapped around the JSON."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return None


def salvage_json(response: str) -> Tuple[Optional[Any], str]:
    """
    Recover a JSON value from raw model output.

    Returns:
        (value, how) where `how` is one of "clean", "fenced", "sliced",
        "repaired", or "failed". Anything other than "clean" or "fenced" means
        the model misbehaved and the result may be partial.
    """
    if response is None:
        return None, "failed"

    raw = response.strip()
    if not raw:
        return None, "failed"

    # 1. As-is.
    try:
        return json.loads(raw), "clean"
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences and JSON-illegal line comments.
    text = _strip_fences(raw)
    text = _LINE_COMMENT_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    try:
        return json.loads(text), "fenced"
    except json.JSONDecodeError:
        pass

    # 3. Cut prose away from the outermost object.
    sliced = _outermost_object(text)
    if sliced is not None:
        candidate = _TRAILING_COMMA_RE.sub(r"\1", sliced)
        try:
            return json.loads(candidate), "sliced"
        except json.JSONDecodeError:
            pass

    # 4. Repair truncation. Try the slice first: on truncated output there is
    #    no closing brace, so the slice fails and the fenced text is all we have.
    for candidate in filter(None, (sliced, text)):
        repaired = fix_broken_generated_json(candidate)
        repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
        try:
            return json.loads(repaired), "repaired"
        except json.JSONDecodeError:
            continue

    # 5. Last resort: the opening brace with everything after it repaired.
    start = text.find("{")
    if start != -1:
        repaired = fix_broken_generated_json(text[start:])
        repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
        try:
            return json.loads(repaired), "repaired"
        except json.JSONDecodeError:
            pass

    return None, "failed"
