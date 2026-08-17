#!/usr/bin/env python3
"""
Extract and cache per-frame visual features for event-boundary detection.

This is the expensive pass over the video: CPU-bound decode at a low sample
rate, with no model in the loop. It gates the whole event condition, and once
the cache exists, boundary detection and the entire Obj 3 threshold sweep run
from it in seconds without touching video again.

Checkpointed per mp4 and safe to re-run: completed files are skipped, so an
interrupted job resumes where it stopped.

    # smoke test on two days first
    python preprocess/chunking/extract_frame_features.py \
        --person A1_JAKE --days 1,2 --workers 8

    # then the full build set
    python preprocess/chunking/extract_frame_features.py \
        --person A1_JAKE --workers 16

Runs on the `shared` CPU partition; it needs no GPU and can proceed while the
A100s are busy with the LLM stages.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from worldmm.chunking.config import ChunkingConfig  # noqa: E402
from worldmm.chunking.features import FeatureCache, extract_features  # noqa: E402
from worldmm.common.checkpoint import CheckpointStore  # noqa: E402
from worldmm.common.mapping import DEFAULT_EXCLUDED_DAYS, load_caption_chunks  # noqa: E402

logger = logging.getLogger(__name__)


def _process_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker: extract one video's features and cache them.

    Runs in a separate process because decode is CPU-bound and does not give
    the GIL back reliably. Returns a small status record rather than the
    feature arrays, so nothing large crosses the process boundary.
    """
    video_path = task["video_path"]
    started = time.perf_counter()
    try:
        features = extract_features(
            video_path,
            sample_fps=task["sample_fps"],
            hist_bins=tuple(task["hist_bins"]),
            gray_size=task["gray_size"],
            prefer_decord=task["prefer_decord"],
        )
        if features.usable:
            FeatureCache(task["cache_dir"]).save(features)
        return {
            "video_path": video_path,
            "status": features.status,
            "n_frames": features.n,
            "source_fps": features.source_fps,
            "error": features.error,
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the pool
        return {
            "video_path": video_path,
            "status": "worker_error",
            "n_frames": 0,
            "source_fps": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.perf_counter() - started,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--caption-file", default="data/EgoLife/caption.zip")
    parser.add_argument("--config", default=None,
                        help="Chunking config JSON. Defaults to built-in event settings.")
    parser.add_argument("--cache-dir", default=None,
                        help="Defaults to output/features/<person>")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--days", default=None,
                        help="Comma-separated days to process, e.g. '1,2'. Default: the build set.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many videos (smoke test).")
    parser.add_argument("--include-all-days", action="store_true",
                        help=f"Include days excluded by default {DEFAULT_EXCLUDED_DAYS}.")
    parser.add_argument("--no-decord", action="store_true",
                        help="Force the OpenCV reader (decord is preferred when present).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = (ChunkingConfig.from_file(args.config) if args.config
              else ChunkingConfig(strategy="event"))
    cache_dir = args.cache_dir or f"output/features/{args.person}"

    exclude = () if args.include_all_days else DEFAULT_EXCLUDED_DAYS
    chunks = load_caption_chunks(args.caption_file, person=args.person, exclude_days=exclude)

    if args.days:
        wanted = {int(d) for d in args.days.split(",") if d.strip()}
        chunks = [c for c in chunks if c.day in wanted]
        logger.info("Restricted to day(s) %s: %d chunks", sorted(wanted), len(chunks))

    # One entry per distinct video file, in timeline order. The caption grid
    # is one chunk per mp4, but dedup keeps this correct regardless.
    video_paths: List[str] = list(dict.fromkeys(c.video_path for c in chunks if c.video_path))
    if args.limit:
        video_paths = video_paths[: args.limit]
    if not video_paths:
        logger.error("No videos selected")
        return 1

    os.makedirs(cache_dir, exist_ok=True)
    config.write(os.path.join(cache_dir, "chunking_config.json"))

    store = CheckpointStore(os.path.join(cache_dir, "progress.jsonl"))
    cache = FeatureCache(cache_dir)

    # Resume only where both the ledger and the cache agree; a cache file
    # deleted by hand should be recomputed rather than silently assumed.
    pending = [
        p for p in video_paths
        if p not in store or not cache.has(p)
    ]
    print(f"videos in scope: {len(video_paths)}")
    print(f"already done:    {len(video_paths) - len(pending)}")
    print(f"to process:      {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return 0

    task_template = {
        "sample_fps": config.sample_fps,
        "hist_bins": list(config.hist_bins),
        "gray_size": config.gray_size,
        "cache_dir": cache_dir,
        "prefer_decord": not args.no_decord,
    }
    tasks = [{**task_template, "video_path": p} for p in pending]

    counts: Counter = Counter()
    started = time.perf_counter()
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one, t): t["video_path"] for t in tasks}
        for future in as_completed(futures):
            record = future.result()
            store.record(record["video_path"], record)
            counts[record["status"]] += 1
            done += 1

            if done % 25 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0
                remaining = (len(tasks) - done) / rate if rate else 0
                print(f"[{done}/{len(tasks)}] {rate:.1f} videos/s, "
                      f"~{remaining / 60:.1f} min remaining, "
                      + ", ".join(f"{k}={v}" for k, v in counts.most_common()))

    store.compact()
    store.close()

    elapsed = time.perf_counter() - started
    print(f"\n=== DONE in {elapsed / 60:.1f} min ===")
    for status, count in counts.most_common():
        print(f"  {status:14s} {count:5d}")

    unusable = sum(v for k, v in counts.items() if k != "ok")
    if unusable:
        print(f"\n{unusable} video(s) produced no features. Each becomes a forced event "
              "boundary, since continuity cannot be assumed across unobserved video.")
    print(f"\nFeature cache: {cache_dir}")
    print("Next: preprocess/chunking/build_chunks.py to detect boundaries and emit chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
