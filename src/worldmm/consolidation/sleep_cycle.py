"""
Sleep-cycle memory consolidation.  [Objective 2]

WorldMM consolidates online and one triple at a time: at every semantic node it
embeds the new triples plus the whole accumulated state, then issues an LLM
call per new triple to decide what to merge. On the inherited run that is
~4,440 LLM calls and ~245,000 embedding operations, and the state still grows
monotonically to 1,582 triples over 52 hours with no collapse of repeated
routine.

This runs offline in periodic passes instead, in the spirit of the biological
consolidation the objective refers to: episodic detail accumulates during the
"day", and at intervals a "sleep cycle" sweeps the working set and collapses
repetition into single semantic entries carrying a support count.

Two structural differences from WorldMM:

  * No LLM in the consolidation loop. Merging is decided by embedding
    similarity over the whole triple, so the cost is one embedding pass per
    cycle rather than one generation per triple.
  * Repetition is rewarded rather than duplicated. A routine that recurs on
    six days becomes one triple with support 6, not six triples.

The embedder is injected, so the same model used elsewhere in the pipeline is
used here, and tests can substitute a deterministic stand-in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .canonicalise import is_degenerate
from .interval_store import ConsolidatedTriple, IntervalTripleStore

logger = logging.getLogger(__name__)

Embedder = Callable[[List[str]], np.ndarray]


@dataclass
class SleepCycleConfig:
    """Parameters of the consolidation algorithm."""

    # Cosine similarity above which two triples are the same fact. Well above
    # WorldMM's 0.6 retrieval threshold: that value selects *relevant* triples
    # to reason about, whereas this one asserts identity, and merging at 0.6
    # would collapse genuinely distinct facts.
    similarity_threshold: float = 0.86
    # Nodes per sleep cycle. At 5-minute nodes, 144 is roughly one waking day,
    # which is the analogy the objective asks for and keeps each pass small.
    cycle_every: int = 144
    # Cap on the working set held between cycles; a cycle is forced early if
    # it is exceeded, bounding the O(n^2) similarity matrix.
    max_working_set: int = 6000
    batch_size: int = 256

    def validate(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if self.cycle_every < 1:
            raise ValueError("cycle_every must be at least 1")
        if self.max_working_set < 1:
            raise ValueError("max_working_set must be at least 1")


@dataclass
class ConsolidationReport:
    """What a run did, for the Objective 3 comparison."""

    n_input_triples: int = 0
    n_after_exact_dedupe: int = 0
    n_final: int = 0
    n_cycles: int = 0
    n_embedding_calls: int = 0
    n_llm_calls: int = 0
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        ratio = (self.n_input_triples / self.n_final) if self.n_final else 0.0
        return {
            "n_input_triples": self.n_input_triples,
            "n_after_exact_dedupe": self.n_after_exact_dedupe,
            "n_final": self.n_final,
            "compression_ratio": ratio,
            "n_cycles": self.n_cycles,
            "n_embedding_calls": self.n_embedding_calls,
            "n_llm_calls": self.n_llm_calls,
            "seconds": self.seconds,
        }


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SleepCycleConsolidator:
    """Collapses repetitive semantic triples into supported, interval-tagged facts."""

    def __init__(self, embedder: Embedder, config: Optional[SleepCycleConfig] = None) -> None:
        self.embedder = embedder
        self.config = config or SleepCycleConfig()
        self.config.validate()
        self.report = ConsolidationReport()

    # --- one consolidation pass -------------------------------------------

    def _cluster(self, triples: List[ConsolidatedTriple]) -> List[ConsolidatedTriple]:
        """
        Merge near-duplicate triples by embedding similarity.

        Greedy, seeded by support descending, so a fact already observed many
        times becomes the representative and rarer phrasings fold into it
        rather than the other way round.
        """
        if len(triples) < 2:
            return triples

        embeddings = _normalise_rows(
            np.asarray(self.embedder([t.text for t in triples]), dtype=np.float32)
        )
        self.report.n_embedding_calls += len(triples)
        similarity = embeddings @ embeddings.T

        order = sorted(range(len(triples)),
                       key=lambda i: (-triples[i].support, len(triples[i].text)))
        assigned = np.zeros(len(triples), dtype=bool)
        merged: List[ConsolidatedTriple] = []

        for index in order:
            if assigned[index]:
                continue
            members = np.flatnonzero(
                (~assigned) & (similarity[index] >= self.config.similarity_threshold)
            )
            assigned[members] = True
            representative = triples[index]
            for member in members:
                if member != index:
                    representative.absorb(triples[member])
            merged.append(representative)

        return merged

    # --- the run ----------------------------------------------------------

    def consolidate(
        self,
        nodes: Dict[str, Sequence[Sequence[str]]],
    ) -> IntervalTripleStore:
        """
        Consolidate a whole recording.

        Args:
            nodes: node key -> semantic triples extracted at that node.

        Returns:
            An `IntervalTripleStore` holding the consolidated memory.
        """
        import time

        started = time.perf_counter()
        store = IntervalTripleStore()
        self.report = ConsolidationReport()

        # How many distinct facts survive exact deduplication, before any
        # embedding merge. Computed up front because the streaming design
        # interleaves adds and cycles, so there is no single moment afterwards
        # at which the store holds exactly that set.
        from .canonicalise import canonical_key
        self.report.n_after_exact_dedupe = len({
            canonical_key(t)
            for triples in nodes.values() for t in triples if not is_degenerate(t)
        })

        working: List[ConsolidatedTriple] = []
        since_cycle = 0

        for node_key in sorted(nodes):
            for triple in nodes[node_key]:
                self.report.n_input_triples += 1
                if is_degenerate(triple):
                    continue
                # Exact duplicates fold here for free, before any embedding.
                working.append(store.add(triple, node_key))

            since_cycle += 1
            distinct = len(store)
            if since_cycle >= self.config.cycle_every or distinct >= self.config.max_working_set:
                self._run_cycle(store)
                since_cycle = 0
                working = []

        # A final pass so the tail after the last cycle is consolidated too.
        self._run_cycle(store)

        self.report.n_final = len(store)
        self.report.seconds = time.perf_counter() - started
        logger.info(
            "Consolidated %d triples -> %d (%.1fx) in %d cycle(s), %d embeddings, 0 LLM calls",
            self.report.n_input_triples, self.report.n_final,
            self.report.n_input_triples / max(1, self.report.n_final),
            self.report.n_cycles, self.report.n_embedding_calls,
        )
        return store

    def _run_cycle(self, store: IntervalTripleStore) -> None:
        if len(store) < 2:
            return
        before = len(store)
        store.replace_all(self._cluster(list(store)))
        self.report.n_cycles += 1
        logger.debug("Sleep cycle %d: %d -> %d triples",
                     self.report.n_cycles, before, len(store))


def cumulative_baseline_size(nodes: Dict[str, Sequence[Sequence[str]]]) -> Dict[str, Any]:
    """
    What WorldMM's serialisation costs on the same input.

    Reproduces its accounting without re-running it: every node stores the full
    accumulation to that point, so the written total is the sum of the running
    size. This is the figure the Objective 3 comparison is made against.
    """
    running = 0
    written = 0
    for node_key in sorted(nodes):
        running += len(nodes[node_key])
        written += running
    return {
        "n_nodes": len(nodes),
        "final_state_triples": running,
        "total_triples_written": written,
        "write_amplification": written / max(1, running),
    }
