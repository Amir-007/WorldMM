"""
Shared types and helpers for the spatial memory bank.
"""

import re
from dataclasses import dataclass, field, asdict
from hashlib import md5
from typing import Any, Dict, List, Optional

# Predicates denoting physical manipulation of an object by the camera wearer.
# The object slot of "I <verb> X" is a thing that was physically handled, which
# is what makes its location meaningful. Speech predicates (says, asks, mentions)
# are deliberately excluded: captions transcribe conversation, so their objects
# are things talked about, not things present.
MANIPULATION_PREDICATES = frozenset({
    "hold", "holds", "holding", "hold up", "holds up",
    "pick up", "picks up", "picked up", "picking up",
    "put", "puts", "putting", "put down", "puts down", "put on", "puts on",
    "place", "places", "placed", "set down", "sets down",
    "grab", "grabs", "grabbed", "take", "takes", "took",
    "carry", "carries", "carrying", "lift", "lifts",
    "open", "opens", "opened", "close", "closes", "closed",
    "use", "uses", "using", "touch", "touches",
    "reach for", "reaches for", "move", "moves", "drop", "drops",
    "plug in", "plugs in", "unplug", "connect", "connects",
    "insert", "inserts", "remove", "removes",
    "wash", "washes", "cut", "cuts", "pour", "pours",
    "eat", "eats", "drink", "drinks", "wear", "wears",
    "push", "pushes", "pull", "pulls", "hand", "hands",
    "give", "gives", "bring", "brings", "fill", "fills",
    "press", "presses", "turn on", "turns on", "turn off", "turns off",
})

# Subjects identifying the camera wearer in the egocentric captions.
EGO_SUBJECTS = frozenset({"i", "me"})

# Leading determiners and possessives stripped during surface-form normalisation.
_LEADING = ("the ", "a ", "an ", "my ", "his ", "her ", "their ",
            "our ", "your ", "some ", "this ", "that ", "these ", "those ")

# Referents carrying no identity, and body parts / mass nouns that are not
# discrete objects a user could disambiguate between.
_STOPWORDS = frozenset({
    "it", "them", "him", "her", "me", "you", "us", "they", "he", "she", "i",
    "one", "thing", "something", "anything", "everything", "someone",
    "everyone", "anyone", "nothing", "others", "we",
    "hand", "hands", "head", "hair", "face", "eyes", "arm", "arms",
    "leg", "legs", "foot", "feet", "body", "finger", "fingers",
    "time", "way", "bit", "lot", "part", "side", "end",
})

_PUNCT = re.compile(r"[^\w\s'-]")
_WS = re.compile(r"\s+")


def compute_chunk_id(text: str) -> str:
    """
    Recompute the OpenIE chunk id for a caption.

    Must stay identical to worldmm.memory.episodic.utils.compute_mdhash_id with
    prefix "chunk-", since that is what keys the openie results file. The
    passage hashed is the caption's 'text' field verbatim.
    """
    return "chunk-" + md5(text.encode()).hexdigest()


def normalise_surface_form(raw: str, max_words: int = 4) -> Optional[str]:
    """
    Normalise an entity surface form, or return None if it is not usable.

    Lowercases, strips quotes/punctuation and leading determiners, and rejects
    pronouns, body parts, mass nouns, and phrases longer than max_words (which
    are clause fragments rather than object names).
    """
    text = raw.strip().strip('"“”\'').lower()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()

    changed = True
    while changed:
        changed = False
        for prefix in _LEADING:
            if text.startswith(prefix):
                text = text[len(prefix):]
                changed = True

    if not text or text in _STOPWORDS:
        return None
    if len(text) < 3 or len(text.split()) > max_words:
        return None
    return text


def caption_timestamp(date: str, time_str: str) -> int:
    """
    Build the integer timestamp used across WorldMM: day digit + time zero-padded to 8.

    Mirrors CaptionEntry.timestamp_int in the episodic module. Uses the full
    numeric part of the date rather than its last character, so DAY10 and beyond
    do not collide with DAY1 (see the note in the module docstring of build.py).
    """
    day = date.replace("DAY", "").replace("Day", "").strip()
    return int(day + str(time_str).zfill(8))


def transform_timestamp(ts: int) -> str:
    """Render an integer timestamp as DAY<d> HH:MM:SS."""
    text = str(ts)
    day, rest = text[0], text[1:]
    return f"DAY{day} {rest[0:2]}:{rest[2:4]}:{rest[4:6]}"


@dataclass
class EntityRecord:
    """
    One persistent Entity ID: a surface form observed at one location.

    The same surface form seen at two locations yields two EntityRecords with
    distinct entity_ids, which is the disambiguation signal Objective 5 consumes.
    """
    entity_id: str
    surface_form: str
    location_label: str
    first_seen: int
    last_seen: int
    mention_chunks: List[str] = field(default_factory=list)
    n_mentions: int = 0
    mention_times: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityRecord":
        return cls(**data)

    def to_display_str(self) -> str:
        return (f"{self.surface_form} @ {self.location_label} "
                f"[{transform_timestamp(self.first_seen)} - "
                f"{transform_timestamp(self.last_seen)}, {self.n_mentions} mentions]")
