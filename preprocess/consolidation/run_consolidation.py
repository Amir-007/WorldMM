#!/usr/bin/env python3
"""
Sleep-cycle consolidation of semantic memory.  [Objective 2]

Consolidates one condition's semantic extraction into an interval store, and
reports the comparison against WorldMM's cumulative serialisation on the same
input - both the semantic compression (how many facts survive) and the storage
compression (how many times each fact is written).

    python preprocess/consolidation/run_consolidation.py \
        --semantic output/metadata/semantic_memory/A1_JAKE_fixed/semantic_extraction_results_qwen3vl-30b.json \
        --output-dir output/consolidation/A1_JAKE_fixed \
        --sweep 0.80,0.84,0.86,0.90,0.94

Embeddings are computed once and cached by triple text, so a threshold sweep
after the first pass costs no model time - the same arrangement that made the
chunking sweep free.

`--materialise` additionally writes the nested per-node layout WorldMM uses, for
feeding the unmodified retrieval stack. That file is the quadratic one; its size
is reported either way as the storage baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.consolidation import (  # noqa: E402
    SleepCycleConfig,
    SleepCycleConsolidator,
    cumulative_baseline_size,
)

logger = logging.getLogger(__name__)


class CachingEmbedder:
    """Embeds triple text once and reuses it across threshold settings."""

    def __init__(self, model_name: str, batch_size: int = 256) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._cache: Dict[str, np.ndarray] = {}
        self._model = None
        self.calls = 0

    def _load(self):
        if self._model is None:
            from worldmm.embedding import EmbeddingModel
            started = time.perf_counter()
            self._model = EmbeddingModel(text_model_name=self.model_name)
            self._model.load_model(model_type="text")
            print(f"embedding model load: {time.perf_counter() - started:.0f}s")
        return self._model

    def __call__(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            model = self._load()
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                vectors = model.encode(batch, modality="text")
                for text, vector in zip(batch, np.asarray(vectors)):
                    self._cache[text] = np.asarray(vector, dtype=np.float32)
                self.calls += len(batch)
        return np.stack([self._cache[t] for t in texts])


def _size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--cycle-every", type=int, default=144)
    parser.add_argument("--sweep", default=None,
                        help="Comma-separated thresholds to report, e.g. '0.80,0.86,0.92'.")
    parser.add_argument("--materialise", action="store_true",
                        help="Also write WorldMM's nested per-node layout.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.semantic, "r", encoding="utf-8") as f:
        nodes = json.load(f)["semantic_triples"]

    baseline = cumulative_baseline_size(nodes)
    print("=== WORLDMM CUMULATIVE BASELINE (same input) ===")
    print(f"  nodes                    {baseline['n_nodes']:>12,}")
    print(f"  final state triples      {baseline['final_state_triples']:>12,}")
    print(f"  triples written to disk  {baseline['total_triples_written']:>12,}")
    print(f"  write amplification      {baseline['write_amplification']:>12.1f}x")
    print(f"  source file size         {_size_mb(args.semantic):>12.1f} MB")

    embedder = CachingEmbedder(args.embedding_model)
    thresholds = ([float(t) for t in args.sweep.split(",")] if args.sweep
                  else [args.threshold])
    if args.threshold not in thresholds:
        thresholds.append(args.threshold)

    rows = []
    best_store = None
    for threshold in sorted(thresholds):
        config = SleepCycleConfig(similarity_threshold=threshold,
                                  cycle_every=args.cycle_every)
        consolidator = SleepCycleConsolidator(embedder, config)
        store = consolidator.consolidate(nodes)
        row = {"threshold": threshold, **consolidator.report.to_dict(),
               **store.statistics()}
        rows.append(row)
        if threshold == args.threshold:
            best_store = store

    print("\n=== SLEEP-CYCLE CONSOLIDATION ===")
    print(f"  {'thresh':>8s}{'final':>10s}{'ratio':>9s}{'mean sup':>10s}"
          f"{'singles':>10s}{'preds':>9s}{'sec':>8s}")
    for row in rows:
        print(f"  {row['threshold']:>8.2f}{row['n_final']:>10,}"
              f"{row['compression_ratio']:>9.2f}{row['mean_support']:>10.2f}"
              f"{row['singletons']:>10,}{row['distinct_predicates']:>9,}"
              f"{row['seconds']:>8.1f}")
    print(f"\n  embedding calls: {embedder.calls:,}   LLM calls: 0   "
          f"(WorldMM issues one generation per new triple)")

    assert best_store is not None
    store_path = os.path.join(args.output_dir, "consolidated_interval.json")
    best_store.write(store_path, metadata={
        "source": args.semantic,
        "threshold": args.threshold,
        "cycle_every": args.cycle_every,
        "embedding_model": args.embedding_model,
        "worldmm_baseline": baseline,
    })

    print("\n=== STORAGE ===")
    print(f"  interval store           {_size_mb(store_path):>12.1f} MB")

    if args.materialise:
        nested = best_store.materialise_worldmm_format(sorted(nodes))
        nested_path = os.path.join(args.output_dir, "consolidated_worldmm_format.json")
        with open(nested_path, "w", encoding="utf-8") as f:
            json.dump(nested, f, indent=2, ensure_ascii=False)
        written = sum(len(v["consolidated_semantic_triples"]) for v in nested.values())
        print(f"  materialised (nested)    {_size_mb(nested_path):>12.1f} MB "
              f"({written:,} triples written)")
        print(f"  storage reduction        "
              f"{_size_mb(nested_path) / max(1e-9, _size_mb(store_path)):>12.1f}x")

    with open(os.path.join(args.output_dir, "consolidation_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({"baseline": baseline, "sweep": rows,
                   "embedding_calls": embedder.calls,
                   "interval_store_mb": _size_mb(store_path)}, f, indent=2)

    print(f"\nwrote {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
