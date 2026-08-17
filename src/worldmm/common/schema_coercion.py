"""
Normalise local-VLM JSON into the shape a schema expects.

Qwen3-VL returns structurally valid JSON that often does not match the
requested schema: a bare list instead of the wrapper object, entities as
`{"entity": "phone"}` instead of `"phone"`, a plausible-but-wrong key name,
nulls where strings belong.

These coercions were developed against real output and live here, decoupled
from Pydantic, so they can be tested without torch or pydantic installed -
which matters because the cluster is the only place the full stack runs and
the only place this code would otherwise be exercised.

The caller supplies the schema's field names and validates the result itself.
"""

from __future__ import annotations

from typing import Any, List, Sequence

__all__ = ["coerce_to_schema"]

# Keys a model tends to wrap a bare string in, most specific first.
_UNWRAP_KEYS = ("entity", "name", "value", "text")


def coerce_to_schema(json_data: Any, field_names: Sequence[str]) -> Any:
    """
    Reshape parsed JSON toward a schema with the given field names.

    Conservative by design: anything already well-formed is passed through
    untouched, and no value is invented. Returns the input unchanged when no
    rule applies, leaving the caller's validation to reject it.
    """
    fields: List[str] = list(field_names)

    # 1. A bare list where a single-field object was requested.
    if isinstance(json_data, list) and len(fields) == 1:
        json_data = {fields[0]: json_data}

    if not isinstance(json_data, dict):
        return json_data

    # 2. List elements wrapped in single-entry dicts.
    for key, value in list(json_data.items()):
        if not isinstance(value, list):
            continue
        unwrapped = []
        for element in value:
            if isinstance(element, dict):
                for candidate in _UNWRAP_KEYS:
                    if candidate in element:
                        unwrapped.append(element[candidate])
                        break
                else:
                    unwrapped.append(next(iter(element.values())) if element else "")
            else:
                unwrapped.append(element)
        json_data[key] = unwrapped

    # 3. Single-field schema, single-key object, wrong key name.
    if len(fields) == 1 and fields[0] not in json_data and len(json_data) == 1:
        json_data = {fields[0]: next(iter(json_data.values()))}

    # 4. Nulls inside a list of lists (triples). Only touched when every
    #    element is itself a list, so integer-typed fields stay intact.
    for key, value in list(json_data.items()):
        if isinstance(value, list) and value and all(isinstance(v, list) for v in value):
            json_data[key] = [
                ["" if element is None else element for element in row] for row in value
            ]

    # 5. Nulls inside a flat list (named entities).
    for key, value in list(json_data.items()):
        if isinstance(value, list):
            json_data[key] = ["" if element is None else element for element in value]

    return json_data
