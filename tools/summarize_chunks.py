#!/usr/bin/env python3
"""
Compare the two chunking conditions side by side.

Reads the artifacts `build_chunks.py` wrote and reports the chunk-count and
chunk-length distributions Objective 3 asks for, plus a GPU-cost projection
built from the measured calibration rather than an assumption.

    python tools/summarize_chunks.py --person A1_JAKE

Run after both conditions have been built. Safe to re-run; reads only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Measured on Eureka, Qwen3-VL-30B-A3B, 50 chunks of the 30s grid. Overridden
# by output/calibration/openie_30b.json when that file is present.
FALLBACK_SECONDS_PER_CHUNK = 14.51
BASELINE_TEXT_CHARS = 526.0          # mean caption length of the inherited grid


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _seconds_per_chunk(calibration_path: str) -> tuple:
    data = _load(calibration_path)
    if not data:
        return FALLBACK_SECONDS_PER_CHUNK, "assumed"
    try:
        return float(data["timing"]["chunk"]["mean"]), "measured"
    except (KeyError, TypeError, ValueError):
        return FALLBACK_SECONDS_PER_CHUNK, "assumed"


def _histogram(values: List[float], bins: int = 10, width: int = 40) -> List[str]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [f"  {lo:6.1f}s  {'#' * width}  {len(values)}"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - lo) / step))
        counts[index] += 1
    peak = max(counts) or 1
    rows = []
    for i, count in enumerate(counts):
        edge = lo + i * step
        bar = "#" * max(0, round(width * count / peak))
        rows.append(f"  {edge:6.1f}s {bar:<{width}} {count:,}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--chunks-root", default="output/chunks")
    parser.add_argument("--calibration", default="output/calibration/openie_30b.json")
    parser.add_argument("--shards", type=int, default=2,
                        help="Concurrent GPU processes for the wall-clock projection.")
    args = parser.parse_args()

    root = os.path.join(args.chunks_root, args.person)
    conditions = {}
    for name in ("fixed", "event"):
        meta = _load(os.path.join(root, name, "chunks_meta.json"))
        if meta:
            conditions[name] = meta
        else:
            print(f"NOTE: no {name} condition at {root}/{name}/chunks_meta.json")

    if not conditions:
        print(f"Nothing found under {root}. Run build_chunks.py first.")
        return 1

    per_chunk, provenance = _seconds_per_chunk(args.calibration)

    print(f"\n{'':24s}" + "".join(f"{n.upper():>16s}" for n in conditions))
    print("-" * (24 + 16 * len(conditions)))

    rows = [
        ("chunks", "n_chunks", "{:,.0f}"),
        ("hours covered", None, None),
        ("mean length (s)", "mean_seconds", "{:,.1f}"),
        ("median length (s)", "median_seconds", "{:,.1f}"),
        ("p95 length (s)", "p95_seconds", "{:,.1f}"),
        ("min length (s)", "min_seconds", "{:,.1f}"),
        ("max length (s)", "max_seconds", "{:,.1f}"),
        ("mean text (chars)", "mean_text_chars", "{:,.0f}"),
        ("max text (chars)", "max_text_chars", "{:,.0f}"),
        ("multi-video chunks", "multi_video_chunks", "{:,.0f}"),
        ("duplicate chunk ids", "duplicate_chunk_ids", "{:,.0f}"),
    ]

    for label, key, fmt in rows:
        cells = []
        for name, meta in conditions.items():
            stats = meta["statistics"]
            if key is None:
                cells.append(f"{stats.get('total_seconds', 0) / 3600:,.1f}")
            else:
                cells.append(fmt.format(stats.get(key, 0)))
        print(f"{label:24s}" + "".join(f"{c:>16s}" for c in cells))

    # --- cost projection ---------------------------------------------------
    print(f"\n=== OPENIE COST ({provenance} at {per_chunk:.2f}s per 30s-equivalent chunk) ===")
    total_hours = 0.0
    for name, meta in conditions.items():
        stats = meta["statistics"]
        scale = max(1.0, stats["mean_text_chars"] / BASELINE_TEXT_CHARS)
        hours = stats["n_chunks"] * per_chunk * scale / 3600
        total_hours += hours
        print(f"  {name:8s} {stats['n_chunks']:>7,} chunks x {scale:4.2f}x text "
              f"-> {hours:6.1f} h")
    print(f"  {'TOTAL':8s} {'':>7s}         {'':>9s} -> {total_hours:6.1f} h serial")
    print(f"  across {args.shards} shard(s):{' ':>22}{total_hours / args.shards:6.1f} h wall-clock")

    if total_hours / args.shards > 40:
        print("\n  WARNING: over 40h wall-clock. Cut days before starting, not after.")

    # --- length distributions ---------------------------------------------
    for name, meta in conditions.items():
        lengths = [c["duration_seconds"] for c in meta["chunks"]]
        print(f"\n=== {name.upper()} chunk-length distribution ===")
        for line in _histogram(lengths):
            print(line)

    # --- boundary detail ---------------------------------------------------
    boundaries = _load(os.path.join(root, "event", "boundaries.json"))
    if boundaries:
        print("\n=== BOUNDARY DETECTION BY DAY ===")
        print(f"  {'day':6s}{'videos':>10s}{'segments':>10s}{'mean s':>10s}{'forced':>9s}")
        total_forced = 0
        for day in sorted(boundaries):
            entry = boundaries[day]
            total_forced += entry.get("n_forced_cuts", 0)
            print(f"  {day:6s}{entry.get('n_videos_with_features', 0):>10,}"
                  f"{entry.get('n_segments', 0):>10,}{entry.get('mean_seconds', 0):>10.1f}"
                  f"{entry.get('n_forced_cuts', 0):>9,}")
        if total_forced:
            print(f"\n  {total_forced} boundary(ies) forced by unobserved video "
                  "(missing or unreadable clips).")

    sweep = _load(os.path.join(root, "event", "threshold_sweep.json"))
    if sweep:
        print("\n=== THRESHOLD SWEEP (Obj 3 ablation) ===")
        merged: Dict[float, Dict[str, float]] = {}
        for row in sweep:
            entry = merged.setdefault(row["threshold"], {"n": 0, "total": 0.0})
            entry["n"] += row["n_segments"]
            entry["total"] += row["total_seconds"]
        print(f"  {'threshold':>10s}{'segments':>12s}{'mean s':>10s}")
        for threshold in sorted(merged):
            entry = merged[threshold]
            mean = entry["total"] / entry["n"] if entry["n"] else 0.0
            print(f"  {threshold:>10.1f}{int(entry['n']):>12,}{mean:>10.1f}")
        counts = [merged[t]["n"] for t in sorted(merged)]
        if len(set(counts)) == 1:
            print("\n  NOTE: every threshold gives the same result, so the sweep is not "
                  "discriminating. Check the z-score distribution before trusting it.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
