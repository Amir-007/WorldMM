#!/usr/bin/env python3
"""
Run OpenIE over a chunk set, sharded across GPUs and checkpointed per chunk.

The stock `batch_openie` holds every result in memory and writes one JSON at
the end, so a crash 20 hours in costs 20 hours. This records each chunk the
moment it finishes, so a killed or requeued job resumes where it stopped, and
several shards can work the same chunk set concurrently on separate GPUs.

    # one process per full A100, stride-sharded
    python preprocess/openie/run_openie.py --chunks output/chunks/A1_JAKE/fixed/chunks.json \
        --output-dir output/metadata/episodic_memory/A1_JAKE_fixed \
        --model qwen3vl-30b --shard-id 0 --num-shards 2

    # once every shard is done
    python preprocess/openie/run_openie.py --chunks ... --output-dir ... --merge

Sharding is by stride, not by contiguous block: chunk cost varies with content
and content varies by time of day, so striding balances the shards.

Checkpoint keys are chunk ids, which are the md5 of the chunk text - the same
value OpenIE computes internally - so resuming is exact and a chunk can never
be processed twice under a different name.
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
from worldmm.common.timestamps import chunk_timestamp_key  # noqa: E402

logger = logging.getLogger(__name__)


def load_chunks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    for chunk in chunks:
        if "chunk_id" not in chunk:
            raise ValueError(
                f"{path} predates chunk ids; rebuild it with build_chunks.py"
            )
    return chunks


def shard_of(chunks: List[Dict[str, Any]], shard_id: int, num_shards: int):
    return [c for i, c in enumerate(chunks) if i % num_shards == shard_id]


def progress_path(output_dir: str, model: str, shard_id: int) -> str:
    return os.path.join(output_dir, f"openie_progress_{model}_shard{shard_id}.jsonl")


def run_shard(args) -> int:
    from worldmm.llm import LLMModel
    from worldmm.memory.episodic.openie import (
        NER_MAX_TOKENS, TRIPLE_MAX_TOKENS, OpenIE,
    )

    chunks = load_chunks(args.chunks)
    assigned = shard_of(chunks, args.shard_id, args.num_shards)
    os.makedirs(args.output_dir, exist_ok=True)

    store = CheckpointStore(progress_path(args.output_dir, args.model, args.shard_id))
    pending = list(store.pending(assigned, key=lambda c: c["chunk_id"]))

    print(f"chunks total:    {len(chunks):,}")
    print(f"this shard:      {len(assigned):,}  (shard {args.shard_id}/{args.num_shards})")
    print(f"already done:    {len(assigned) - len(pending):,}")
    print(f"to process:      {len(pending):,}")
    print(f"token budgets:   ner={NER_MAX_TOKENS} triples={TRIPLE_MAX_TOKENS}")
    if not pending:
        print("Nothing to do.")
        return 0

    load_start = time.perf_counter()
    llm_model = LLMModel(model_name=args.model)
    print(f"model load:      {time.perf_counter() - load_start:.0f}s")
    openie = OpenIE(llm_model)
    backend = getattr(llm_model, "model", None)

    started = time.perf_counter()
    empty_from_text = 0

    for done, chunk in enumerate(pending, start=1):
        text = chunk["text"]
        ner_output = openie.ner(chunk_key=chunk["chunk_id"], passage=text)
        triple_output = openie.triple_extraction(
            chunk_key=chunk["chunk_id"],
            passage=text,
            named_entities=ner_output.unique_entities,
        )

        if text.strip() and not triple_output.triples:
            empty_from_text += 1

        store.record(chunk["chunk_id"], {
            "ner": ner_output.unique_entities,
            "triples": triple_output.triples,
            "ner_error": ner_output.metadata.get("error"),
            "triple_error": triple_output.metadata.get("error"),
        })

        if done % 25 == 0 or done == len(pending):
            elapsed = time.perf_counter() - started
            rate = done / elapsed
            remaining = (len(pending) - done) / rate if rate else 0
            line = (f"[{done}/{len(pending)}] {elapsed / done:.1f}s/chunk, "
                    f"~{remaining / 3600:.1f} h remaining, empty={empty_from_text}")
            if backend is not None:
                line += (f", repairs={getattr(backend, 'parse_repairs', 0)}"
                         f", failures={getattr(backend, 'parse_failures', 0)}")
            print(line, flush=True)

    store.compact()
    store.close()

    elapsed = time.perf_counter() - started
    print(f"\n=== SHARD {args.shard_id} DONE in {elapsed / 3600:.2f} h "
          f"({elapsed / len(pending):.1f}s/chunk) ===")
    print(f"  chunks with text but no triples: {empty_from_text}/{len(pending)}")
    if backend is not None:
        print(f"  parse repairs:  {getattr(backend, 'parse_repairs', 0)}")
        print(f"  parse failures: {getattr(backend, 'parse_failures', 0)}")
    return 0


def merge(args) -> int:
    """Combine shard checkpoints into the artifacts downstream stages expect."""
    chunks = load_chunks(args.chunks)

    records: Dict[str, Any] = {}
    found_shards = 0
    for shard_id in range(args.num_shards):
        path = progress_path(args.output_dir, args.model, shard_id)
        if not os.path.exists(path):
            logger.warning("No progress file for shard %d (%s)", shard_id, path)
            continue
        found_shards += 1
        records.update(CheckpointStore(path).results())

    missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in records]
    print(f"shards found:   {found_shards}/{args.num_shards}")
    print(f"chunks:         {len(chunks):,}")
    print(f"results:        {len(records):,}")
    print(f"missing:        {len(missing):,}")
    if missing:
        print(f"  e.g. {missing[:3]}")
        if not args.allow_partial:
            print("\nRefusing to write a partial result. Finish the shards, or pass "
                  "--allow-partial if the gap is understood and intended.")
            return 1

    # Ordered by the chunk sequence, so the artifacts are stable and diffable.
    ner_results, triple_results = {}, {}
    episodic_triples, raw_video = {}, {}
    for chunk in chunks:
        record = records.get(chunk["chunk_id"])
        if record is None:
            continue
        ner_results[chunk["chunk_id"]] = record["ner"]
        triple_results[chunk["chunk_id"]] = record["triples"]
        key = chunk_timestamp_key(chunk["date"], chunk["end_time"])
        episodic_triples[key] = record["triples"]
        raw_video[key] = chunk["video_path"]

    os.makedirs(args.output_dir, exist_ok=True)
    openie_path = os.path.join(args.output_dir, f"openie_results_{args.model}.json")
    with open(openie_path, "w", encoding="utf-8") as f:
        json.dump({"ner_results": ner_results, "triple_results": triple_results},
                  f, indent=2, ensure_ascii=False)

    episodic_path = os.path.join(
        args.output_dir, f"episodic_triple_results_{args.model}.json")
    with open(episodic_path, "w", encoding="utf-8") as f:
        json.dump({"episodic_triples": episodic_triples, "raw_video": raw_video},
                  f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in triple_results.values())
    empty = sum(1 for v in triple_results.values() if not v)
    print(f"\ntotal triples:  {total:,}")
    print(f"empty chunks:   {empty:,} ({empty / max(1, len(triple_results)):.1%})")
    print(f"mean per chunk: {total / max(1, len(triple_results)):.1f}")
    print(f"\nwrote {openie_path}")
    print(f"wrote {episodic_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--chunks", required=True, help="chunks.json from build_chunks.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen3vl-30b")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="Combine finished shards.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Merge even if some chunks have no result.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be within [0, num-shards)")

    return merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
