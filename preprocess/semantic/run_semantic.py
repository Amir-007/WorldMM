#!/usr/bin/env python3
"""
Semantic extraction over time-based nodes, checkpointed and shardable.

Nodes are 5-minute windows of the timeline rather than fixed counts of chunks,
so both chunking conditions produce comparable nodes. See
`worldmm.common.nodes` for why that matters.

    python preprocess/semantic/run_semantic.py \
        --chunks output/chunks/A1_JAKE/fixed/chunks.json \
        --openie output/metadata/episodic_memory/A1_JAKE_fixed/openie_results_qwen3vl-30b.json \
        --output-dir output/metadata/semantic_memory/A1_JAKE_fixed \
        --model qwen3vl-30b

    # then
    python preprocess/semantic/run_semantic.py ... --merge

Emits `semantic_extraction_results_<model>.json` in the layout the existing
consolidation stage expects.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.common.checkpoint import CheckpointStore  # noqa: E402
from worldmm.common.nodes import DEFAULT_PERIOD_SECONDS, build_time_nodes  # noqa: E402

logger = logging.getLogger(__name__)


def load_nodes(args) -> List:
    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(args.openie, "r", encoding="utf-8") as f:
        triples = json.load(f)["triple_results"]
    return build_time_nodes(chunks, triples, period_seconds=args.period_seconds)


def progress_path(output_dir: str, model: str, shard_id: int) -> str:
    return os.path.join(output_dir, f"semantic_progress_{model}_shard{shard_id}.jsonl")


def run_shard(args) -> int:
    nodes = load_nodes(args)
    assigned = [n for i, n in enumerate(nodes) if i % args.num_shards == args.shard_id]
    os.makedirs(args.output_dir, exist_ok=True)

    store = CheckpointStore(progress_path(args.output_dir, args.model, args.shard_id))
    pending = [n for n in store.pending(assigned, key=lambda n: n.key) if n.triples]
    skipped_empty = sum(1 for n in assigned if not n.triples)

    print(f"nodes total:     {len(nodes):,}")
    print(f"this shard:      {len(assigned):,}  (shard {args.shard_id}/{args.num_shards})")
    print(f"empty (skipped): {skipped_empty:,}")
    print(f"to process:      {len(pending):,}")
    if not pending:
        print("Nothing to do.")
        return 0

    from worldmm.llm import LLMModel
    from worldmm.memory.semantic import SemanticExtraction

    load_start = time.perf_counter()
    llm_model = LLMModel(model_name=args.model)
    print(f"model load:      {time.perf_counter() - load_start:.0f}s")
    extractor = SemanticExtraction(llm_model)
    backend = getattr(llm_model, "model", None)

    started = time.perf_counter()
    empty_out = 0

    for done, node in enumerate(pending, start=1):
        result = extractor.semantic_extraction(node.key, node.triples)
        if not result.semantic_triples:
            empty_out += 1
        store.record(node.key, {
            "semantic_triples": result.semantic_triples,
            "n_input_triples": len(node.triples),
            "n_chunks": node.n_chunks,
            "span_seconds": node.duration_seconds,
        })
        if done % 25 == 0 or done == len(pending):
            elapsed = time.perf_counter() - started
            remaining = (len(pending) - done) * elapsed / done
            line = (f"[{done}/{len(pending)}] {elapsed / done:.1f}s/node, "
                    f"~{remaining / 60:.0f} min remaining, empty={empty_out}")
            if backend is not None:
                line += (f", repairs={getattr(backend, 'parse_repairs', 0)}"
                         f", failures={getattr(backend, 'parse_failures', 0)}")
            print(line, flush=True)

    store.compact()
    store.close()
    elapsed = time.perf_counter() - started
    print(f"\n=== SHARD {args.shard_id} DONE in {elapsed / 60:.1f} min ===")
    print(f"  nodes with input but no semantic triples: {empty_out}/{len(pending)}")
    return 0


def merge(args) -> int:
    nodes = load_nodes(args)

    records: Dict[str, Any] = {}
    for shard_id in range(args.num_shards):
        path = progress_path(args.output_dir, args.model, shard_id)
        if os.path.exists(path):
            records.update(CheckpointStore(path).results())
        else:
            logger.warning("No progress file for shard %d", shard_id)

    # Empty nodes are legitimately absent from the checkpoints; they are
    # materialised here so the node sequence stays complete and the two
    # conditions have the same timeline coverage.
    semantic: Dict[str, List[List[str]]] = {}
    for node in nodes:
        record = records.get(node.key)
        semantic[node.key] = record["semantic_triples"] if record else []

    processed = sum(1 for n in nodes if n.triples)
    missing = [n.key for n in nodes if n.triples and n.key not in records]
    print(f"nodes:          {len(nodes):,}")
    print(f"with input:     {processed:,}")
    print(f"results:        {len(records):,}")
    print(f"missing:        {len(missing):,}")
    if missing and not args.allow_partial:
        print(f"  e.g. {missing[:3]}")
        print("\nRefusing to write a partial result. Finish the shards, or pass "
              "--allow-partial.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"semantic_extraction_results_{args.model}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"semantic_triples": semantic}, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in semantic.values())
    nonempty = sum(1 for v in semantic.values() if v)
    print(f"\nnodes:              {len(semantic):,}")
    print(f"non-empty:          {nonempty:,}")
    print(f"total triples:      {total:,}")
    print(f"mean per non-empty: {total / max(1, nonempty):.1f}")
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--openie", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen3vl-30b")
    parser.add_argument("--period-seconds", type=float, default=DEFAULT_PERIOD_SECONDS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be within [0, num-shards)")
    return merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
