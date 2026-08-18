"""
Offline construction of the spatial (Entity ID) memory bank.

Runs on data already produced by the episodic pipeline. Pure text processing,
no GPU and no model calls.

Pipeline:
  1. Join captions to OpenIE results by md5 of the caption text. The chunk id in
     openie_results_*.json is "chunk-" + md5(caption["text"]), so the join is
     exact and reproducible.
  2. Derive a location cue per caption from the location vocabulary, optionally
     carrying the last known location forward across captions that name no room.
  3. Take entity mentions from the object slot of "I <manipulation verb> X"
     triples, rather than from the raw NER list. NER on these captions is
     dominated by housemate names and by places discussed in conversation,
     because the captions transcribe speech.
  4. Group mentions by (surface form, location) into persistent Entity IDs.

Known upstream quirk, preserved deliberately: the episodic reformat in
preprocess/episodic_memory/extract_episodic_triples.py builds its timestamp with
date[-1], the last character of the date string, which breaks at DAY10 and
beyond. This module uses the full numeric day instead. EgoLife has 7 days so the
two agree on this dataset, but the difference is worth knowing if the data grows.
"""

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .location_vocab import location_cue
from .utils import (EGO_SUBJECTS, MANIPULATION_PREDICATES, EntityRecord,
                    caption_timestamp, compute_chunk_id, normalise_surface_form)

logger = logging.getLogger(__name__)

DEFAULT_CARRY_FORWARD = 4


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sort_captions(captions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order captions chronologically, which carry-forward depends on."""
    return sorted(captions, key=lambda c: caption_timestamp(c["date"], c["start_time"]))


def assign_locations(captions: List[Dict[str, Any]], carry_forward: int) -> Dict[int, str]:
    """
    Map caption index -> location label.

    A caption naming exactly one location is labelled directly. With
    carry_forward > 0, a caption naming none inherits the most recent label,
    provided no more than carry_forward unlabelled captions have passed. A
    caption naming two or more locations clears the carried label rather than
    guessing, so an ambiguous caption does not poison what follows.
    """
    labels: Dict[int, str] = {}
    last_label: Optional[str] = None
    gap = 0

    for index, caption in enumerate(captions):
        cue = location_cue(caption["text"])
        if cue is not None:
            labels[index] = cue
            last_label, gap = cue, 0
            continue

        # location_cue returns None for both "no match" and "several matches";
        # only the latter should invalidate the carried label.
        from .location_vocab import match_location_labels
        if len(match_location_labels(caption["text"])) > 1:
            last_label, gap = None, 0
            continue

        gap += 1
        if last_label is not None and gap <= carry_forward:
            labels[index] = last_label

    return labels


def extract_mentions(triples: List[List[str]]) -> List[str]:
    """
    Return normalised object surface forms from ego manipulation triples.

    Filters to triples of the form "I <manipulation verb> X" and returns the
    normalised X. Triples that are malformed, non-ego, non-manipulation, or
    whose object normalises away are dropped.
    """
    mentions: List[str] = []
    for triple in triples:
        if not isinstance(triple, list) or len(triple) != 3:
            continue
        subject, predicate, obj = (str(part).strip().lower() for part in triple)
        if subject not in EGO_SUBJECTS or predicate not in MANIPULATION_PREDICATES:
            continue
        normalised = normalise_surface_form(obj)
        if normalised:
            mentions.append(normalised)
    return mentions


def build_entity_bank(
    caption_file: str,
    openie_file: str,
    carry_forward: int = DEFAULT_CARRY_FORWARD,
    use_ner: bool = False,
) -> Tuple[Dict[str, EntityRecord], Dict[str, Any]]:
    """
    Build the Entity ID bank.

    Args:
        caption_file: caption JSON, e.g. A1_JAKE_30sec.json
        openie_file: openie_results_<model>.json from the episodic pipeline
        carry_forward: unlabelled captions a location label may span; 0 disables
        use_ner: use raw NER entities instead of triple objects. Retained for
            comparison; produces a far noisier bank, since NER on these captions
            picks up people and places mentioned in speech rather than objects
            actually handled.

    Returns:
        (entity_id -> EntityRecord, stats dict)
    """
    captions = sort_captions(load_json(caption_file))
    openie = load_json(openie_file)
    ner_results = openie.get("ner_results", {})
    triple_results = openie.get("triple_results", {})

    matched = sum(1 for c in captions if compute_chunk_id(c["text"]) in triple_results)
    if matched == 0:
        raise ValueError(
            f"No caption hashed to any chunk id in {openie_file}. The caption file "
            f"is almost certainly the wrong granularity for this OpenIE run."
        )
    if matched < len(captions):
        logger.warning("Only %d/%d captions matched a chunk id (%.1f%%)",
                       matched, len(captions), 100 * matched / len(captions))

    labels = assign_locations(captions, carry_forward)

    # (surface_form, location) -> accumulating record
    grouped: Dict[Tuple[str, str], EntityRecord] = {}
    localised_mentions = 0

    for index, caption in enumerate(captions):
        location = labels.get(index)
        if location is None:
            continue

        chunk_id = compute_chunk_id(caption["text"])
        if use_ner:
            raw = [normalise_surface_form(e) for e in ner_results.get(chunk_id, [])]
            mentions = [m for m in raw if m]
        else:
            mentions = extract_mentions(triple_results.get(chunk_id, []))
        if not mentions:
            continue

        start = caption_timestamp(caption["date"], caption["start_time"])
        end = caption_timestamp(caption["date"], caption["end_time"])

        for surface in mentions:
            localised_mentions += 1
            key = (surface, location)
            record = grouped.get(key)
            if record is None:
                grouped[key] = EntityRecord(
                    entity_id="",  # assigned after grouping, once counts are final
                    surface_form=surface,
                    location_label=location,
                    first_seen=start,
                    last_seen=end,
                    mention_chunks=[chunk_id],
                    n_mentions=1,
                    mention_times=[start],
                )
            else:
                record.first_seen = min(record.first_seen, start)
                record.last_seen = max(record.last_seen, end)
                record.n_mentions += 1
                record.mention_times.append(start)
                if chunk_id not in record.mention_chunks:
                    record.mention_chunks.append(chunk_id)

    # Assign stable ids: surface form, then location ordered by descending
    # mention count so the primary location of an entity is consistently _0.
    by_surface: Dict[str, List[EntityRecord]] = defaultdict(list)
    for record in grouped.values():
        by_surface[record.surface_form].append(record)

    bank: Dict[str, EntityRecord] = {}
    for surface, records in sorted(by_surface.items()):
        records.sort(key=lambda r: (-r.n_mentions, r.location_label))
        slug = surface.replace(" ", "_")
        for ordinal, record in enumerate(records):
            record.entity_id = f"ent-{slug}-{ordinal}"
            record.mention_times.sort()
            bank[record.entity_id] = record

    stats = compute_stats(bank, captions, labels, localised_mentions, carry_forward)
    return bank, stats


def compute_stats(
    bank: Dict[str, EntityRecord],
    captions: List[Dict[str, Any]],
    labels: Dict[int, str],
    localised_mentions: int,
    carry_forward: int,
) -> Dict[str, Any]:
    """Compute the ambiguity budget and supporting coverage figures."""
    by_surface: Dict[str, List[EntityRecord]] = defaultdict(list)
    for record in bank.values():
        by_surface[record.surface_form].append(record)

    split = {s: r for s, r in by_surface.items() if len(r) >= 2}

    # A split is only usable as evaluation signal when each side has real
    # support. One stray mention at a second location is noise, not ambiguity.
    solid: Dict[str, List[EntityRecord]] = {}
    for surface, records in split.items():
        supported = [r for r in records if r.n_mentions >= 2]
        if 2 <= len(supported) <= 3:
            solid[surface] = supported

    return {
        "n_captions": len(captions),
        "n_captions_localised": len(labels),
        "caption_location_coverage": round(len(labels) / len(captions), 4) if captions else 0.0,
        "carry_forward_window": carry_forward,
        "n_localised_mentions": localised_mentions,
        "n_entity_ids": len(bank),
        "n_surface_forms": len(by_surface),
        "ambiguity_budget": len(split),
        "usable_ambiguity_budget": len(solid),
        "location_distribution": dict(Counter(r.location_label for r in bank.values()).most_common()),
        "usable_entities": {
            surface: {r.location_label: r.n_mentions for r in records}
            for surface, records in sorted(solid.items(), key=lambda kv: -sum(r.n_mentions for r in kv[1]))
        },
    }


def save_entity_bank(bank: Dict[str, EntityRecord], stats: Dict[str, Any], output_dir: str,
                     filename: str = "entity_ids.json") -> str:
    """Write the bank plus its stats to JSON and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    payload = {
        "stats": stats,
        "entities": {eid: record.to_dict() for eid, record in bank.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote %d entity ids to %s", len(bank), path)
    return path
