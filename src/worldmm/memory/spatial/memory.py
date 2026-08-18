"""
Spatial Memory module for WorldMM.

Fourth memory bank alongside episodic, semantic and visual. Holds persistent
Entity IDs that bind a surface form to a location, so that "the phone in the
kitchen" and "the phone on the desk" are distinct retrievable entities.

Retrieval is lexical rather than embedding-based: it runs on CPU in
microseconds, which keeps the agent's retrieval loop cheap, and it needs no
model to be resident. If lexical matching proves too weak, the query path is
the only thing that would need to change.
"""

import json
import logging
import math
import os
import re
from bisect import bisect_right
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import EntityRecord, normalise_surface_form, transform_timestamp

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9']+")


def _tokenise(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _contains_phrase(tokens: List[str], phrase: List[str]) -> bool:
    """
    True when phrase appears as a contiguous run of whole tokens in tokens.

    Word-level rather than substring matching. Plain `in` on the raw string made
    "hat" match inside "what", which fired disambiguation on queries containing
    no such entity at all.
    """
    span = len(phrase)
    if span == 0 or span > len(tokens):
        return False
    return any(tokens[i:i + span] == phrase for i in range(len(tokens) - span + 1))


class SpatialMemory:
    """
    Spatial memory over persistent Entity IDs.

    Like the sibling banks, retrieval is gated by index(until_time) so a query
    can never see entities observed after the question's timestamp. EgoLifeQA is
    time-gated, so skipping that gate would leak future information and
    invalidate any comparison against the baseline.

    Attributes:
        entities: entity_id -> EntityRecord, the full unfiltered bank
        indexed_entities: entity_id -> EntityRecord visible at indexed_time
        indexed_time: timestamp boundary; 0 means nothing indexed yet
    """

    def __init__(self) -> None:
        self.entities: Dict[str, EntityRecord] = {}
        self.surface_to_ids: Dict[str, List[str]] = defaultdict(list)

        self.indexed_entities: Dict[str, EntityRecord] = {}
        self.indexed_mention_counts: Dict[str, int] = {}
        self.indexed_time: int = 0
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ load

    def load_entities_from_file(self, file_path: str) -> None:
        """Load an entity bank written by the spatial build step."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Entity bank not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.load_entities_from_data(payload)
        logger.info("Loaded %d entity ids from %s", len(self.entities), file_path)

    def load_entities_from_data(self, payload: Dict[str, Any]) -> None:
        """Load an entity bank from an in-memory payload."""
        entities = payload.get("entities", payload)
        self.stats = payload.get("stats", {})
        for entity_id, record in entities.items():
            entry = record if isinstance(record, EntityRecord) else EntityRecord.from_dict(record)
            self.entities[entity_id] = entry
            self.surface_to_ids[entry.surface_form].append(entity_id)

    # ----------------------------------------------------------------- index

    def index(self, until_time: int) -> None:
        """
        Expose only what had been observed by until_time.

        An entity is visible once first seen at or before the boundary, and its
        mention count is truncated to mentions at or before it, so evidence
        strength also reflects only the past.
        """
        if self.indexed_time >= until_time and self.indexed_entities:
            logger.debug("Already indexed up to %d, skipping %d", self.indexed_time, until_time)
            return

        self.indexed_entities = {}
        self.indexed_mention_counts = {}
        for entity_id, record in self.entities.items():
            if record.first_seen > until_time:
                continue
            visible = bisect_right(record.mention_times, until_time) if record.mention_times \
                else record.n_mentions
            if visible <= 0:
                continue
            self.indexed_entities[entity_id] = record
            self.indexed_mention_counts[entity_id] = visible

        self.indexed_time = until_time
        logger.info("Indexed %d/%d entity ids up to %s",
                    len(self.indexed_entities), len(self.entities),
                    transform_timestamp(until_time))

    # ----------------------------------------------------------------- query

    def query(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank Entity IDs against a free-text query.

        Returns (entity_id, score) descending, score in (0, 1]. The caller reads
        both the number of results and the top1 minus top2 margin to gauge how
        ambiguous the query was, so scores are comparable within one call rather
        than calibrated globally.
        """
        pool = self.indexed_entities or self.entities
        if not pool:
            return []

        query_token_list = _tokenise(text)
        query_tokens = set(query_token_list)
        if not query_tokens:
            return []
        lowered = text.lower()

        scored: List[Tuple[str, float]] = []
        for entity_id, record in pool.items():
            surface = record.surface_form
            surface_token_list = _tokenise(surface)
            surface_tokens = set(surface_token_list)
            if not surface_tokens:
                continue

            if _contains_phrase(query_token_list, surface_token_list):
                # Whole surface form present verbatim; longer phrases are stronger.
                base = 1.0 if len(surface_tokens) > 1 else 0.9
            else:
                overlap = len(surface_tokens & query_tokens)
                if overlap == 0:
                    continue
                base = 0.75 * (overlap / len(surface_tokens))

            # Mild evidence prior so a well-attested entity outranks a one-off
            # without letting frequency dominate lexical fit.
            mentions = self.indexed_mention_counts.get(entity_id, record.n_mentions)
            prior = 1.0 + 0.08 * math.log1p(max(mentions, 0))
            if _contains_phrase(query_token_list, _tokenise(record.location_label.replace("_", " "))):
                prior *= 1.25  # query names the location explicitly

            scored.append((entity_id, base * prior))

        if not scored:
            return []

        # Normalise against the best raw score rather than clamping to 1.0.
        # Clamping made same-surface-form entities tie at the ceiling, which
        # zeroed the top1-top2 margin, which is what the caller reads. After
        # normalisation the top hit is always 1.0 and the margin reflects how
        # much better it fits than its closest rival.
        best = max(score for _, score in scored)
        scored = [(entity_id, score / best) for entity_id, score in scored]

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]

    def disambiguation_candidates(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Ranked entities that share the best-matching surface form.

        Two or more Entity IDs sharing a name at different locations means the
        query is genuinely ambiguous, whereas several unrelated entities merely
        means the query was broad. Only the former is worth asking the user about.
        """
        ranked = self.query(text, top_k=max(top_k, 10))
        if not ranked:
            return []
        pool = self.indexed_entities or self.entities
        best_surface = pool[ranked[0][0]].surface_form
        return [(eid, score) for eid, score in ranked
                if pool[eid].surface_form == best_surface][:top_k]

    # --------------------------------------------------------------- helpers

    def retrieve(self, query: str, top_k: int = 5, as_context: bool = True):
        """Retrieve matching entities, as a context string or as EntityRecords."""
        ranked = self.query(query, top_k=top_k)
        pool = self.indexed_entities or self.entities
        records = [pool[eid] for eid, _ in ranked]
        return self.retrieve_entities_as_str(records) if as_context else records

    def retrieve_entities_as_str(self, records: Sequence[EntityRecord]) -> str:
        return "\n".join(record.to_display_str() for record in records)

    def get_entity(self, entity_id: str) -> Optional[EntityRecord]:
        return self.entities.get(entity_id)

    def get_entity_ids_for_surface(self, surface_form: str) -> List[str]:
        """Entity IDs sharing a surface form, restricted to what is indexed."""
        normalised = normalise_surface_form(surface_form) or surface_form.strip().lower()
        pool = self.indexed_entities or self.entities
        return [eid for eid in self.surface_to_ids.get(normalised, []) if eid in pool]

    def get_indexed_time(self) -> str:
        return transform_timestamp(self.indexed_time) if self.indexed_time else "not indexed"

    def get_indexed_count(self) -> int:
        return len(self.indexed_entities)

    def reset_index(self) -> None:
        self.indexed_entities = {}
        self.indexed_mention_counts = {}
        self.indexed_time = 0
        logger.info("Spatial memory index reset")

    def cleanup(self) -> None:
        """Present for parity with the sibling banks; this one holds no GPU state."""
        return None
