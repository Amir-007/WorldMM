"""
Location vocabulary for the spatial memory bank.

Location cues are derived by keyword-matching caption text. This file is kept
separate so the vocabulary can be widened without touching build or query logic.

Measured coverage on A1_JAKE (6,223 x 30sec captions): 44.8% of captions contain
at least one cue, 28.0% contain exactly one. Captions matching two or more
distinct labels are treated as ambiguous and left unlabelled, because an
egocentric caption spanning two rooms cannot be pinned to either.
"""

from typing import Dict, List, Optional

# label -> surface keywords. Matched case-insensitively as substrings.
LOCATION_VOCAB: Dict[str, List[str]] = {
    "kitchen":     ["kitchen"],
    "bedroom":     ["bedroom", "my room"],
    "living_room": ["living room", "sofa", "couch"],
    "dining":      ["dining table", "dining room"],
    "bathroom":    ["bathroom", "toilet", "shower"],
    "balcony":     ["balcony", "terrace"],
    "desk":        ["desk"],
    "garden":      ["garden", "yard", "courtyard"],
    "stairs":      ["stairs", "staircase", "second floor", "upstairs", "downstairs"],
    "entrance":    ["entrance", "front door", "hallway"],
    "store":       ["supermarket", "store", "shop", "market", "restaurant"],
    "street":      ["street", "road", "sidewalk"],
}


def match_location_labels(text: str) -> List[str]:
    """Return every distinct location label whose keywords appear in text."""
    lowered = text.lower()
    return [
        label for label, keywords in LOCATION_VOCAB.items()
        if any(keyword in lowered for keyword in keywords)
    ]


def location_cue(text: str) -> Optional[str]:
    """
    Return the single unambiguous location label for a caption, else None.

    Returns None both when no keyword matches and when two or more distinct
    labels match, since neither case identifies one place.
    """
    labels = match_location_labels(text)
    return labels[0] if len(labels) == 1 else None
