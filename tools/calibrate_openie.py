#!/usr/bin/env python3
"""
Time the OpenIE stage so the build estimates stop being guesses.

Every wall-clock figure in the Track A plan rests on one unmeasured number:
Qwen3-VL-30B-A3B decode throughput on Eureka. The pipeline uses HF
`transformers.generate()` with `device_map="auto"` and `WORLDMM_WORKERS=1`, so
there is no batching and no continuous scheduling - chunks are processed one at
a time. A ~20 minute run over 50 chunks pins the per-chunk cost and turns the
whole schedule into arithmetic.

    python tools/calibrate_openie.py --model qwen3vl-30b --n 50 \
        --output output/calibration/openie_30b.json

Chunks are sampled evenly across the corpus, not taken from the front, so the
sample is not biased toward one day or time of day.

This also counts a specific failure the plan flags as a landmine: the fork's
`_parse_structured_response` degrades truncated or unparseable model output to
an *empty* result instead of raising. A chunk with real caption text that comes
back with zero triples is very likely a silent truncation, and at event-chunk
lengths (~3x the text, ~44 triples against a 512-token cap) that would suppress
triple counts and flatter the compression ratio.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from worldmm.common.mapping import DEFAULT_EXCLUDED_DAYS, load_caption_chunks  # noqa: E402

logger = logging.getLogger(__name__)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _describe(label: str, values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    stats = {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }
    print(f"  {label:22s} mean={stats['mean']:6.2f}s  p50={stats['p50']:6.2f}s  "
          f"p95={stats['p95']:6.2f}s  max={stats['max']:6.2f}s")
    return stats


def _token_counter(llm_model) -> Optional[Any]:
    """Best-effort access to the underlying tokenizer, for real tok/s figures."""
    processor = getattr(getattr(llm_model, "model", None), "processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        logger.info("No tokenizer reachable; reporting a 4-chars/token approximation")
    return tokenizer


def _count_tokens(tokenizer, text: str) -> float:
    if tokenizer is None:
        return len(text) / 4
    try:
        return len(tokenizer.encode(text))
    except Exception:  # noqa: BLE001
        return len(text) / 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--model", default="qwen3vl-30b")
    parser.add_argument("--caption-file", default="data/EgoLife/caption.zip")
    parser.add_argument("--n", type=int, default=50, help="Chunks to time.")
    parser.add_argument("--output", default="output/calibration/openie.json")
    parser.add_argument("--baseline-chunks", type=int, default=None,
                        help="Chunks in the baseline condition. Defaults to the build set size.")
    parser.add_argument("--event-chunks", type=int, default=None,
                        help="Expected event count. Defaults to baseline/2 (a 60s cap).")
    parser.add_argument("--event-text-scale", type=float, default=2.0,
                        help="Mean event length / 30s, i.e. how much longer event text is.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks = load_caption_chunks(args.caption_file, person=args.person,
                                 exclude_days=DEFAULT_EXCLUDED_DAYS)
    baseline_chunks = args.baseline_chunks or len(chunks)
    event_chunks = args.event_chunks or max(1, baseline_chunks // 2)

    if args.n >= len(chunks):
        sample = chunks
    else:
        stride = len(chunks) / args.n
        sample = [chunks[int(i * stride)] for i in range(args.n)]

    print(f"Corpus: {len(chunks)} chunks (days {DEFAULT_EXCLUDED_DAYS} excluded)")
    print(f"Timing {len(sample)} chunks with {args.model}\n")

    from worldmm.llm import LLMModel
    from worldmm.memory.episodic.openie import OpenIE

    load_start = time.perf_counter()
    llm_model = LLMModel(model_name=args.model)
    load_seconds = time.perf_counter() - load_start
    print(f"Model load: {load_seconds:.1f}s\n")

    tokenizer = _token_counter(llm_model)
    openie = OpenIE(llm_model)

    records: List[Dict[str, Any]] = []
    try:
        for i, chunk in enumerate(sample, start=1):
            t0 = time.perf_counter()
            ner_output = openie.ner(chunk_key=chunk.chunk_key, passage=chunk.text)
            t1 = time.perf_counter()
            triple_output = openie.triple_extraction(
                chunk_key=chunk.chunk_key,
                passage=chunk.text,
                named_entities=ner_output.unique_entities,
            )
            t2 = time.perf_counter()

            triples = triple_output.triples
            record = {
                "timestamp_key": chunk.timestamp_key,
                "date": chunk.date,
                "text_chars": len(chunk.text),
                "text_tokens": _count_tokens(tokenizer, chunk.text),
                "ner_seconds": t1 - t0,
                "triple_seconds": t2 - t1,
                "chunk_seconds": t2 - t0,
                "n_entities": len(ner_output.unique_entities),
                "n_triples": len(triples),
                "triple_out_tokens": _count_tokens(tokenizer, json.dumps({"triples": triples})),
                "ner_error": ner_output.metadata.get("error"),
                "triple_error": triple_output.metadata.get("error"),
                # Real text in, zero triples out: almost certainly a truncated
                # generation silently degraded to an empty result.
                "suspect_silent_empty": bool(chunk.text.strip()) and not triples,
            }
            records.append(record)
            print(f"[{i}/{len(sample)}] {chunk.timestamp_key} "
                  f"ner={record['ner_seconds']:5.2f}s triples={record['triple_seconds']:5.2f}s "
                  f"({record['n_triples']} triples)"
                  + ("  <-- EMPTY" if record["suspect_silent_empty"] else ""))
    except KeyboardInterrupt:
        print("\nInterrupted; reporting on what completed so far.\n")

    if not records:
        print("No chunks completed.")
        return 1

    print(f"\n=== PER-CALL TIMING (n={len(records)}) ===")
    ner_stats = _describe("NER", [r["ner_seconds"] for r in records])
    triple_stats = _describe("triple extraction", [r["triple_seconds"] for r in records])
    chunk_stats = _describe("per chunk (both)", [r["chunk_seconds"] for r in records])

    out_tokens = [r["triple_out_tokens"] for r in records]
    triple_times = [r["triple_seconds"] for r in records]
    decode_rate = sum(out_tokens) / sum(triple_times) if sum(triple_times) else float("nan")
    print(f"\napprox decode rate (triple call): {decode_rate:.1f} output tok/s"
          f"{'' if tokenizer is not None else '  [4-chars/token approximation]'}")
    print(f"mean triples/chunk: {statistics.fmean(r['n_triples'] for r in records):.1f}   "
          f"mean output tokens: {statistics.fmean(out_tokens):.0f} (cap 512)")

    errors = sum(1 for r in records if r["ner_error"] or r["triple_error"])
    empties = sum(1 for r in records if r["suspect_silent_empty"])
    print(f"\nreported errors: {errors}/{len(records)}")
    print(f"silent empty results: {empties}/{len(records)}"
          + ("   <-- investigate before the full run" if empties else ""))

    per_chunk = chunk_stats["mean"]
    baseline_hours = baseline_chunks * per_chunk / 3600
    # Longer event chunks mean proportionally more output tokens, and decode
    # dominates, so scale the per-chunk cost by the text-length ratio.
    event_per_chunk = per_chunk * args.event_text_scale
    event_hours = event_chunks * event_per_chunk / 3600

    print("\n=== EXTRAPOLATION ===")
    print(f"  baseline ({baseline_chunks:,} chunks @ {per_chunk:.2f}s):        "
          f"{baseline_hours:5.1f} h   [inherited - not being rebuilt]")
    print(f"  event    ({event_chunks:,} chunks @ {event_per_chunk:.2f}s):        "
          f"{event_hours:5.1f} h   <-- the run you actually pay for")
    print(f"\n  p95-based worst case for the event condition:  "
          f"{event_chunks * chunk_stats['p95'] * args.event_text_scale / 3600:5.1f} h")
    print("\n  Excludes semantic extraction and consolidation; time those separately "
          "once the event chunks exist.")

    payload = {
        "model": args.model,
        "person": args.person,
        "sample_size": len(records),
        "model_load_seconds": load_seconds,
        "timing": {"ner": ner_stats, "triple": triple_stats, "chunk": chunk_stats},
        "decode_tokens_per_second": decode_rate,
        "tokenizer_exact": tokenizer is not None,
        "errors": errors,
        "silent_empty": empties,
        "extrapolation": {
            "baseline_chunks": baseline_chunks,
            "baseline_hours": baseline_hours,
            "event_chunks": event_chunks,
            "event_text_scale": args.event_text_scale,
            "event_hours": event_hours,
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
