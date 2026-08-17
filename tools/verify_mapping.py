#!/usr/bin/env python3
"""
Rebuild the openie <-> timestamp mapping and check it against the inherited data.

`openie_results_*.json` is keyed by a content hash of the caption text, not by
timestamp. This reconstructs the real mapping from the caption file and proves
it by reproducing `episodic_triple_results_*.json` exactly, rather than joining
on dict insertion order.

    python tools/verify_mapping.py --person A1_JAKE --model qwen3vl-30b
    python tools/verify_mapping.py --output output/metadata/mapping_A1_JAKE.json

Exit status is non-zero if verification fails, so this is safe to run as a
pre-flight gate before a long build.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from worldmm.common.mapping import (  # noqa: E402
    DEFAULT_EXCLUDED_DAYS,
    load_caption_chunks,
    normalise_excluded_days,
    timestamp_to_chunk_key,
    verify,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--model", default="qwen3vl-30b")
    parser.add_argument("--caption-file", default="data/EgoLife/caption.zip",
                        help="Caption JSON, or caption.zip (requires --person).")
    parser.add_argument("--metadata-dir", default="output/metadata")
    parser.add_argument("--output", default=None, help="Write the mapping as JSON.")
    parser.add_argument(
        "--include-all-days", action="store_true",
        help=f"Keep days excluded by default {DEFAULT_EXCLUDED_DAYS}. Use this to "
             "reproduce the full inherited artifact end to end.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    exclude = () if args.include_all_days else DEFAULT_EXCLUDED_DAYS
    chunks = load_caption_chunks(args.caption_file, person=args.person, exclude_days=exclude)

    episodic_dir = os.path.join(args.metadata_dir, "episodic_memory", args.person)
    openie_path = os.path.join(episodic_dir, f"openie_results_{args.model}.json")
    with open(openie_path, encoding="utf-8") as f:
        openie_data = json.load(f)

    episodic_path = os.path.join(episodic_dir, f"episodic_triple_results_{args.model}.json")
    episodic_data = None
    if os.path.exists(episodic_path):
        with open(episodic_path, encoding="utf-8") as f:
            episodic_data = json.load(f)
    else:
        print(f"NOTE: {episodic_path} not found; checking hashes only, "
              "which is the weaker test.")

    report = verify(chunks, openie_data, episodic_data,
                    expect_full_coverage=args.include_all_days)
    print(report.summary())

    if report.hash_misses:
        print(f"\nfirst hash misses: {report.hash_misses[:5]}")
    if report.collisions:
        print(f"\ncolliding chunk ids (repeated caption text): "
              f"{list(report.collisions.items())[:3]}")
    if report.triples_mismatched:
        print(f"\nfirst mismatched timestamps: {report.triples_mismatched[:5]}")

    if args.output:
        payload = {
            "person": args.person,
            "model": args.model,
            "excluded_days": sorted(normalise_excluded_days(exclude)),
            "chunk_count": len(chunks),
            "timestamp_to_chunk_key": timestamp_to_chunk_key(chunks),
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nMapping written to {args.output}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
