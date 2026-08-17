#!/usr/bin/env python3
"""
Build chunks for one experimental condition.

Reads the config, assembles chunks from the EgoLife Sync entries, and writes
artifacts the rest of the pipeline consumes. The strategy is chosen entirely by
config, so both conditions come from this one command:

    python preprocess/chunking/build_chunks.py --config configs/fixed30.json
    python preprocess/chunking/build_chunks.py --config configs/event.json

The fixed condition needs nothing but the Sync files. The event condition also
needs the cached frame features from `extract_frame_features.py`, which it uses
to detect boundaries per day - cuts never span midnight, and unobserved video
always forces a boundary.

Outputs, under `output/chunks/<person>/<strategy>/`:
    chunks.json       caption-entry list, the shape existing stages consume
    chunks_meta.json  full chunk records, config, and distribution statistics
    boundaries.json   per-day cut times and detection stats (event only)

`--sweep` re-cuts at several thresholds from the cached features and reports
the resulting chunk-length distributions without rebuilding anything. That is
the Objective 3 sensitivity ablation and it costs seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.chunking.boundaries import detect_boundaries, sweep_thresholds  # noqa: E402
from worldmm.chunking.config import EVENT, ChunkingConfig  # noqa: E402
from worldmm.chunking.features import FeatureCache  # noqa: E402
from worldmm.chunking.sources import SourceEntry, load_sync_entries  # noqa: E402
from worldmm.chunking.strategy import build_strategy, chunk_statistics  # noqa: E402
from worldmm.common.mapping import load_caption_chunks  # noqa: E402

logger = logging.getLogger(__name__)


def _videos_for_day(caption_file: str, person: str, day: int, exclude_days) -> List[str]:
    """Video files for one day, in timeline order, from the caption grid."""
    chunks = load_caption_chunks(caption_file, person=person, exclude_days=exclude_days)
    return list(dict.fromkeys(c.video_path for c in chunks if c.day == day and c.video_path))


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", required=True, help="Chunking config JSON.")
    parser.add_argument("--sync-dir", default="data/EgoLife/EgoLifeCap/Sync")
    parser.add_argument("--caption-file", default="data/EgoLife/caption.zip")
    parser.add_argument("--cache-dir", default=None,
                        help="Feature cache. Defaults to output/features/<person>.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to output/chunks/<person>/<strategy>.")
    parser.add_argument("--days", default=None, help="Comma-separated days, e.g. '1,2'.")
    parser.add_argument("--sweep", default=None,
                        help="Comma-separated thresholds to report, e.g. '2,3,4,5,6'.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = ChunkingConfig.from_file(args.config)
    person = config.person

    # Fail fast with something readable rather than a traceback three frames
    # deep, since this is normally run from a submit script.
    for label, path in (("sync directory", args.sync_dir),
                        ("caption file", args.caption_file)):
        if config.strategy != EVENT and label == "caption file":
            continue     # only the event path needs the video list
        if not os.path.exists(path):
            logger.error("%s not found: %s", label, path)
            logger.error("Run from the repo root, or pass an explicit path.")
            return 1
    cache_dir = args.cache_dir or f"output/features/{person}"
    output_dir = args.output_dir or f"output/chunks/{person}/{config.strategy}"

    days = [int(d) for d in args.days.split(",")] if args.days else None
    entries = load_sync_entries(
        args.sync_dir, person, days=days, exclude_days=config.exclude_days
    )
    if not entries:
        logger.error("No source entries loaded")
        return 1

    by_day: Dict[int, List[SourceEntry]] = defaultdict(list)
    for entry in entries:
        by_day[entry.day].append(entry)

    strategy = build_strategy(config)
    cache = FeatureCache(cache_dir)

    all_chunks = []
    boundary_report: Dict[str, object] = {}
    sweep_rows: List[dict] = []

    for day in sorted(by_day):
        day_entries = by_day[day]

        if config.strategy != EVENT:
            chunks = strategy.segment(day_entries)
            all_chunks.extend(chunks)
            print(f"DAY{day}: {len(day_entries)} source entries -> {len(chunks)} chunks")
            continue

        video_paths = _videos_for_day(args.caption_file, person, day, config.exclude_days)
        features = cache.load_many(video_paths)
        if not features:
            logger.error(
                "DAY%d: no cached features in %s. Run extract_frame_features.py first.",
                day, cache_dir,
            )
            return 1
        if len(features) < len(video_paths):
            logger.warning(
                "DAY%d: %d/%d videos have features; the rest force boundaries",
                day, len(features), len(video_paths),
            )

        result = detect_boundaries(features, config)
        chunks = strategy.segment(day_entries, cuts=result.cuts)
        all_chunks.extend(chunks)

        stats = result.stats()
        boundary_report[f"DAY{day}"] = {
            "n_videos_with_features": len(features),
            "n_videos_expected": len(video_paths),
            "cuts": result.cuts,
            "forced_cuts": result.forced_cuts,
            **stats,
        }
        print(f"DAY{day}: {len(day_entries)} entries, {len(features)} videos -> "
              f"{stats['n_segments']} segments "
              f"(mean {stats['mean_seconds']:.0f}s, {stats['n_forced_cuts']} forced) "
              f"-> {len(chunks)} chunks with text")

        if args.sweep:
            thresholds = [float(t) for t in args.sweep.split(",")]
            for row in sweep_thresholds(result, thresholds):
                sweep_rows.append({"day": day, **row})

    if not all_chunks:
        logger.error("No chunks produced")
        return 1

    stats = chunk_statistics(all_chunks)
    print(f"\n=== {config.strategy.upper()} CONDITION ===")
    for key in ("n_chunks", "mean_seconds", "median_seconds", "p95_seconds",
                "min_seconds", "max_seconds", "mean_text_chars", "max_text_chars",
                "multi_video_chunks", "duplicate_chunk_ids"):
        value = stats[key]
        print(f"  {key:22s} {value:,.1f}" if isinstance(value, float)
              else f"  {key:22s} {value:,}")

    if stats["duplicate_chunk_ids"]:
        logger.warning(
            "%d chunk(s) share an id because their text is identical; they will "
            "collapse to one entry in openie results",
            stats["duplicate_chunk_ids"],
        )

    # The number that sets the whole GPU budget.
    per_chunk_seconds = 14.51                      # measured, 30s chunks
    scale = stats["mean_text_chars"] / 526.0       # inherited 30s mean text length
    estimate = stats["n_chunks"] * per_chunk_seconds * max(1.0, scale) / 3600
    print(f"\n  estimated openie cost: {estimate:.1f} h "
          f"({stats['n_chunks']:,} chunks, text {scale:.2f}x the 30s baseline)")

    _write_json(os.path.join(output_dir, "chunks.json"),
                [c.to_caption_entry() for c in all_chunks])
    _write_json(os.path.join(output_dir, "chunks_meta.json"),
                {"config": config.to_dict(), "statistics": stats,
                 "chunks": [c.to_dict() for c in all_chunks]})
    if boundary_report:
        _write_json(os.path.join(output_dir, "boundaries.json"), boundary_report)
    if sweep_rows:
        _write_json(os.path.join(output_dir, "threshold_sweep.json"), sweep_rows)
        print("\n=== THRESHOLD SWEEP ===")
        merged: Dict[float, Dict[str, float]] = defaultdict(
            lambda: {"n_segments": 0, "total_seconds": 0.0})
        for row in sweep_rows:
            merged[row["threshold"]]["n_segments"] += row["n_segments"]
            merged[row["threshold"]]["total_seconds"] += row["total_seconds"]
        for threshold in sorted(merged):
            entry = merged[threshold]
            mean = entry["total_seconds"] / entry["n_segments"] if entry["n_segments"] else 0
            print(f"  threshold={threshold:5.1f}  segments={int(entry['n_segments']):6,}  "
                  f"mean={mean:6.1f}s")

    print(f"\nWritten to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
