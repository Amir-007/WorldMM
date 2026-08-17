#!/usr/bin/env python3
"""
Classify the caption chunks that have no visual embedding.  [Priority B]

5,673 of 6,223 chunks in the inherited `visual_embeddings.pkl` have an
embedding; 550 do not. Analysis of the pickle alone ruled out the cheap
explanations: there are 6,223 *distinct* video paths so it is not key
collision, the gaps are scattered across all seven days in 368 runs (291 of
them single chunks) so it is not a crashed split worker, and they show no
filename or grid pattern.

That leaves the two silent `continue` branches in
`preprocess/visual_memory/extract_visual_features.py::process_videos_sequentially`
- the missing-file check and the bare `except Exception` around `encode_video`.
Telling them apart needs the video files, so this runs on the cluster.

    python tools/diagnose_missing_embeddings.py --person A1_JAKE

Run from the repo root: the stored paths are relative (`data/EgoLife/...`).
A control sample of chunks that *do* have embeddings is probed first, so a
wrong working directory fails loudly instead of reporting 6,223 absent files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from worldmm.common.mapping import DEFAULT_EXCLUDED_DAYS, load_caption_chunks  # noqa: E402

logger = logging.getLogger(__name__)

# What went wrong with a given video, worst first.
ABSENT = "absent"          # file not on disk -> hit the os.path.exists guard
EMPTY = "empty"            # zero bytes -> truncated copy or interrupted download
UNREADABLE = "unreadable"  # decord cannot open it, or it decodes to zero frames
SHORT = "short"            # opens, but too few frames to sample 16 from
READABLE = "readable"      # nothing wrong now -> the original failure was transient


def probe(path: str, *, min_frames: int) -> Dict[str, Any]:
    """Stat and probe-decode one video. Never raises."""
    result: Dict[str, Any] = {"path": path, "status": ABSENT, "size": None,
                              "frames": None, "fps": None, "error": None}

    if not os.path.exists(path):
        return result

    try:
        result["size"] = os.path.getsize(path)
    except OSError as exc:
        result["error"] = str(exc)
        return result

    if result["size"] == 0:
        result["status"] = EMPTY
        return result

    try:
        from decord import VideoReader, cpu
    except ImportError:
        # No decord: report what stat alone can tell us rather than crashing.
        result["status"] = "present_unprobed"
        return result

    try:
        reader = VideoReader(path, ctx=cpu(0))
        frames = len(reader)
        result["frames"] = frames
        result["fps"] = float(reader.get_avg_fps())
        if frames == 0:
            result["status"] = UNREADABLE
            result["error"] = "decodes to zero frames"
        elif frames < min_frames:
            result["status"] = SHORT
        else:
            reader[0].asnumpy()          # a header can parse while frames do not
            reader[frames - 1].asnumpy()
            result["status"] = READABLE
    except Exception as exc:  # noqa: BLE001 - mirrors the pipeline's own catch-all
        result["status"] = UNREADABLE
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def probe_all(paths: List[str], *, workers: int, min_frames: int) -> List[Dict[str, Any]]:
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda p: probe(p, min_frames=min_frames), paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--caption-file", default="data/EgoLife/caption.zip")
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Defaults to output/metadata/visual_memory/<person>/visual_embeddings.pkl",
    )
    parser.add_argument("--output", default=None, help="Write the full per-file report as JSON.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=16,
                        help="Frames the embedding step samples per clip.")
    parser.add_argument("--control-sample", type=int, default=50,
                        help="Chunks with embeddings to probe as a sanity check.")
    parser.add_argument(
        "--build-days-only", action="store_true",
        help=f"Restrict to the days we build on, dropping {DEFAULT_EXCLUDED_DAYS}. "
             "Off by default: this diagnoses the inherited pickle, which covers all days.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    embeddings_path = args.embeddings or (
        f"output/metadata/visual_memory/{args.person}/visual_embeddings.pkl"
    )
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    logger.info("Loaded %d embeddings from %s", len(embeddings), embeddings_path)

    exclude_days = DEFAULT_EXCLUDED_DAYS if args.build_days_only else ()
    chunks = load_caption_chunks(args.caption_file, person=args.person,
                                 exclude_days=exclude_days)
    logger.info("Loaded %d caption chunks", len(chunks))

    have = [c for c in chunks if c.video_path in embeddings]
    missing = [c for c in chunks if c.video_path not in embeddings]
    print(f"\nchunks with embedding:    {len(have)}")
    print(f"chunks without embedding: {len(missing)}")
    if not missing:
        print("Nothing to diagnose.")
        return 0

    # Control first: if videos that DID embed are now 'absent', the paths are
    # wrong (wrong cwd or a moved dataset) and the real report would be noise.
    control = [c.video_path for c in have[:: max(1, len(have) // max(1, args.control_sample))]]
    control = control[: args.control_sample]
    control_results = probe_all(control, workers=args.workers, min_frames=args.min_frames)
    control_absent = sum(1 for r in control_results if r["status"] == ABSENT)
    print(f"\ncontrol sample: {len(control_results)} chunks that DO have embeddings")
    print(f"  of these, absent on disk: {control_absent}")
    if control_absent > len(control_results) // 2:
        print(
            "\nABORTING: most control videos are missing too, so the video paths do not "
            "resolve from here. Run from the repo root, or point --caption-file at the "
            "caption file whose paths match this machine."
        )
        return 2

    results = probe_all([c.video_path for c in missing], workers=args.workers,
                        min_frames=args.min_frames)

    by_status = Counter(r["status"] for r in results)
    print(f"\n=== {len(results)} chunks with no embedding ===")
    for status, count in by_status.most_common():
        print(f"  {status:18s} {count:5d}  ({count / len(results):.1%})")

    by_day: Dict[str, Counter] = {}
    for chunk, result in zip(missing, results):
        by_day.setdefault(chunk.date, Counter())[result["status"]] += 1
    print("\nby day:")
    for date in sorted(by_day):
        row = ", ".join(f"{s}={n}" for s, n in by_day[date].most_common())
        print(f"  {date}: {row}")

    for status in (UNREADABLE, EMPTY, SHORT):
        examples = [r for r in results if r["status"] == status][:3]
        if examples:
            print(f"\nexample {status}:")
            for r in examples:
                print(f"  {os.path.basename(r['path'])}  size={r['size']} "
                      f"frames={r['frames']} err={r['error']}")

    retryable = by_status[READABLE] + by_status.get("present_unprobed", 0)
    print("\n=== VERDICT ===")
    if retryable == len(results):
        print(f"All {retryable} are readable now: the original failures were transient "
              "(OOM / CUDA / decode races). The rebuild recovers them at no extra cost.")
    elif retryable:
        print(f"{retryable}/{len(results)} are readable now and will be recovered by the "
              f"rebuild. The remaining {len(results) - retryable} are genuinely damaged "
              "or absent and must be excluded from both conditions to keep them comparable.")
    else:
        print(f"None are readable: all {len(results)} are absent or damaged on disk. "
              "Exclude them from both conditions.")

    if args.output:
        payload = {
            "person": args.person,
            "embeddings_path": embeddings_path,
            "total_chunks": len(chunks),
            "with_embedding": len(have),
            "without_embedding": len(missing),
            "status_counts": dict(by_status),
            "control_absent": control_absent,
            "results": [
                {"timestamp_key": c.timestamp_key, "date": c.date, **r}
                for c, r in zip(missing, results)
            ],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
