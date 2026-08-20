"""
Interval-encoded semantic memory.  [Objective 2, storage]

WorldMM serialises the *entire* accumulated triple set at every semantic node.
Measured on the inherited A1_JAKE run: 623 nodes, a final state of 1,582
triples, and 611,350 triples written to a 56.2 MB file - the same facts
rewritten once per node. Storage is O(nodes x state), quadratic in recording
length, while the information content is linear.

Here each triple is stored once with the interval over which it holds. The
state at any node is reconstructed by filtering on `first_seen`, which is what
`SemanticMemory.index(until_time)` actually needs, so retrieval semantics are
preserved exactly while the file stops growing quadratically.

`materialise_worldmm_format` regenerates the original nested layout when a
downstream consumer insists on it, so this is a storage change rather than a
protocol change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .canonicalise import canonical_key, normalise_text

# Surface variants kept per consolidated triple. Enough to show what was
# merged in the write-up without letting provenance dominate the file.
MAX_VARIANTS = 8


@dataclass
class ConsolidatedTriple:
    """One fact, the window it was observed over, and how often."""

    subject: str
    predicate: str
    object: str
    first_seen: str                                  # DHHMMSSFF node key
    last_seen: str
    support: int = 1                                 # source triples merged in
    variants: List[List[str]] = field(default_factory=list)

    @property
    def triple(self) -> List[str]:
        return [self.subject, self.predicate, self.object]

    @property
    def key(self):
        return canonical_key(self.triple)

    @property
    def text(self) -> str:
        return " ".join(self.triple)

    def absorb(self, other: "ConsolidatedTriple") -> None:
        """Fold another triple into this one, keeping the widest interval."""
        self.support += other.support
        self.first_seen = min(self.first_seen, other.first_seen)
        self.last_seen = max(self.last_seen, other.last_seen)
        for variant in [other.triple, *other.variants]:
            if variant != self.triple and variant not in self.variants:
                if len(self.variants) < MAX_VARIANTS:
                    self.variants.append(variant)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triple": self.triple,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "support": self.support,
            "variants": self.variants,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsolidatedTriple":
        subject, predicate, obj = (list(data["triple"]) + ["", "", ""])[:3]
        return cls(subject=subject, predicate=predicate, object=obj,
                   first_seen=data["first_seen"], last_seen=data["last_seen"],
                   support=int(data.get("support", 1)),
                   variants=[list(v) for v in data.get("variants", [])])


class IntervalTripleStore:
    """Consolidated triples, stored once each with their validity interval."""

    def __init__(self) -> None:
        self._triples: Dict[Any, ConsolidatedTriple] = {}

    def __len__(self) -> int:
        return len(self._triples)

    def __iter__(self):
        return iter(self._triples.values())

    def add(self, triple: Sequence[str], node_key: str, support: int = 1) -> ConsolidatedTriple:
        """Insert a triple, folding it into an existing identical one."""
        parts = [normalise_text(p) for p in (list(triple) + ["", "", ""])[:3]]
        candidate = ConsolidatedTriple(
            subject=parts[0], predicate=parts[1], object=parts[2],
            first_seen=node_key, last_seen=node_key, support=support,
        )
        existing = self._triples.get(candidate.key)
        if existing is None:
            self._triples[candidate.key] = candidate
            return candidate
        existing.absorb(candidate)
        return existing

    def replace_all(self, triples: Iterable[ConsolidatedTriple]) -> None:
        """Swap the contents, e.g. after a consolidation pass."""
        self._triples = {}
        for triple in triples:
            existing = self._triples.get(triple.key)
            if existing is None:
                self._triples[triple.key] = triple
            else:
                existing.absorb(triple)

    def snapshot_at(self, node_key: str) -> List[ConsolidatedTriple]:
        """
        The knowledge state as of a node.

        Everything first observed at or before `node_key`, which mirrors
        WorldMM's own semantics: its per-node snapshot is the accumulation up
        to that node. Triples are not expired at `last_seen`; a fact stays
        known once learned.
        """
        return [t for t in self._triples.values() if t.first_seen <= node_key]

    def triples_at(self, node_key: str) -> List[List[str]]:
        return [t.triple for t in self.snapshot_at(node_key)]

    def materialise_worldmm_format(
        self, node_keys: Sequence[str]
    ) -> Dict[str, Dict[str, List[List[str]]]]:
        """
        Regenerate WorldMM's nested per-node layout.

        Only for feeding a consumer that expects it - writing this to disk
        reintroduces exactly the quadratic blow-up the interval store exists to
        avoid, so it is built in memory on demand.
        """
        ordered = sorted(self._triples.values(), key=lambda t: t.first_seen)
        out: Dict[str, Dict[str, List[List[str]]]] = {}
        running: List[List[str]] = []
        index = 0
        for key in sorted(node_keys):
            while index < len(ordered) and ordered[index].first_seen <= key:
                running.append(ordered[index].triple)
                index += 1
            out[key] = {"consolidated_semantic_triples": list(running)}
        return out

    # --- persistence ------------------------------------------------------

    def to_dict(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "format": "interval_v1",
            "metadata": metadata or {},
            "triples": [t.to_dict() for t in
                        sorted(self._triples.values(), key=lambda t: t.first_seen)],
        }

    def write(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(metadata), f, indent=2, ensure_ascii=False)

    @classmethod
    def read(cls, path: str) -> "IntervalTripleStore":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store = cls()
        store.replace_all(ConsolidatedTriple.from_dict(d) for d in data["triples"])
        return store

    # --- reporting --------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        if not self._triples:
            return {"n_triples": 0}
        supports = [t.support for t in self._triples.values()]
        subjects = {t.key[0] for t in self._triples.values()}
        predicates = {t.key[1] for t in self._triples.values()}
        objects = {t.key[2] for t in self._triples.values()}
        return {
            "n_triples": len(self._triples),
            "total_support": sum(supports),
            "compression_ratio": sum(supports) / len(self._triples),
            "mean_support": sum(supports) / len(supports),
            "max_support": max(supports),
            "singletons": sum(1 for s in supports if s == 1),
            "distinct_subjects": len(subjects),
            "distinct_predicates": len(predicates),
            "distinct_objects": len(objects),
        }
