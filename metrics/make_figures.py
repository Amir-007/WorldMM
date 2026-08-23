#!/usr/bin/env python3
"""
Render the Objective 3 figures from the CSVs `collect_metrics.py` writes.

Optional. matplotlib is not a project dependency, so if it is absent this
reports what it would have drawn and exits cleanly - every figure's data is
already in CSV and can be plotted anywhere.

    pip install matplotlib          # only if you want the PNGs from here
    python metrics/make_figures.py --metrics-dir output/metrics

Deliberately plain: no styling, no colour choices, one chart per file, so the
output slots into whatever template the dissertation uses.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str, default=None):
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--metrics-dir", default="output/metrics")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    out = args.output_dir or os.path.join(args.metrics_dir, "figures")
    src = args.metrics_dir

    figures = {
        "fig_chunk_lengths.csv": "chunk-length distribution, both conditions",
        "fig_growth_curve.csv": "semantic state size per node (growth curve)",
        "fig_consolidation_sweep.csv": "consolidation: compression vs similarity threshold",
        "fig_boundary_sweep.csv": "boundary detection: segments vs z-score threshold",
        "fig_database_size.csv": "database size by component",
    }

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so no images were rendered.\n")
        print("The data for every figure is already in CSV:")
        for name, description in figures.items():
            path = os.path.join(src, name)
            status = f"{len(read_csv(path)):,} rows" if os.path.exists(path) else "MISSING"
            print(f"  {name:34s} {status:>12s}   {description}")
        print("\nEither plot these in your tool of choice, or `pip install matplotlib` "
              "and re-run.")
        return 0

    os.makedirs(out, exist_ok=True)
    written = []

    # 1. chunk-length distribution -----------------------------------------
    rows = read_csv(os.path.join(src, "fig_chunk_lengths.csv"))
    if rows:
        by_condition = defaultdict(list)
        for row in rows:
            value = _f(row, "duration_seconds")
            if value is not None:
                by_condition[row["condition"]].append(value)
        fig, ax = plt.subplots(figsize=(7, 4))
        for condition, values in by_condition.items():
            ax.hist(values, bins=60, alpha=0.55, label=f"{condition} (n={len(values):,})")
        ax.set_xlabel("chunk duration (s)")
        ax.set_ylabel("chunks")
        ax.set_title("Chunk-length distribution")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out, "chunk_lengths.png")
        fig.savefig(path, dpi=args.dpi); plt.close(fig); written.append(path)

    # 2. growth curve -------------------------------------------------------
    rows = read_csv(os.path.join(src, "fig_growth_curve.csv"))
    if rows:
        series = defaultdict(lambda: ([], []))
        for row in rows:
            index = _f(row, "node_index")
            worldmm = _f(row, "worldmm_state")
            interval = _f(row, "interval_state")
            if index is None:
                continue
            if worldmm is not None:
                series[f"{row['condition']}: state"][0].append(index)
                series[f"{row['condition']}: state"][1].append(worldmm)
            if interval is not None:
                series[f"{row['condition']}: interval"][0].append(index)
                series[f"{row['condition']}: interval"][1].append(interval)
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, (xs, ys) in sorted(series.items()):
            if xs:
                ax.plot(xs, ys, label=label, linewidth=1.2)
        ax.set_xlabel("semantic node index (time order)")
        ax.set_ylabel("triples in state")
        ax.set_title("Semantic state growth per node")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out, "growth_curve.png")
        fig.savefig(path, dpi=args.dpi); plt.close(fig); written.append(path)

    # 3. consolidation sweep ------------------------------------------------
    rows = read_csv(os.path.join(src, "fig_consolidation_sweep.csv"))
    if rows:
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
        by_condition = defaultdict(list)
        for row in rows:
            by_condition[row["condition"]].append(row)
        for condition, entries in by_condition.items():
            entries.sort(key=lambda r: _f(r, "threshold", 0))
            thresholds = [_f(r, "threshold") for r in entries]
            left.plot(thresholds, [_f(r, "compression_ratio") for r in entries],
                      marker="o", label=condition)
            right.plot(thresholds, [_f(r, "n_final") for r in entries],
                       marker="o", label=condition)
        left.set_xlabel("similarity threshold"); left.set_ylabel("compression ratio")
        right.set_xlabel("similarity threshold"); right.set_ylabel("triples retained")
        left.set_title("Consolidation compression"); right.set_title("Store size")
        left.legend(); right.legend()
        fig.tight_layout()
        path = os.path.join(out, "consolidation_sweep.png")
        fig.savefig(path, dpi=args.dpi); plt.close(fig); written.append(path)

    # 4. boundary sweep -----------------------------------------------------
    rows = read_csv(os.path.join(src, "fig_boundary_sweep.csv"))
    if rows:
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
        by_condition = defaultdict(list)
        for row in rows:
            by_condition[row["condition"]].append(row)
        for condition, entries in by_condition.items():
            entries.sort(key=lambda r: _f(r, "threshold", 0))
            thresholds = [_f(r, "threshold") for r in entries]
            left.plot(thresholds, [_f(r, "segments") for r in entries],
                      marker="o", label=condition)
            right.plot(thresholds, [_f(r, "mean_seconds") for r in entries],
                       marker="o", label=condition)
        left.set_xlabel("z-score threshold"); left.set_ylabel("segments")
        right.set_xlabel("z-score threshold"); right.set_ylabel("mean segment length (s)")
        left.set_title("Boundary sensitivity"); right.set_title("Segment length")
        left.legend(); right.legend()
        fig.tight_layout()
        path = os.path.join(out, "boundary_sweep.png")
        fig.savefig(path, dpi=args.dpi); plt.close(fig); written.append(path)

    # 5. database size ------------------------------------------------------
    rows = read_csv(os.path.join(src, "fig_database_size.csv"))
    if rows:
        components = ["episodic_mb", "semantic_mb", "consolidated_interval_mb", "visual_mb"]
        conditions = [r["condition"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        bottoms = [0.0] * len(rows)
        for component in components:
            values = [_f(r, component, 0.0) for r in rows]
            ax.bar(conditions, values, bottom=bottoms,
                   label=component.replace("_mb", "").replace("_", " "))
            bottoms = [b + v for b, v in zip(bottoms, values)]
        for i, row in enumerate(rows):
            ax.scatter([i], [_f(row, "total_worldmm_mb", 0.0)], marker="_", s=400,
                       zorder=5, label="WorldMM format total" if i == 0 else None)
        ax.set_ylabel("MB")
        ax.set_title("Database size by component")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out, "database_size.png")
        fig.savefig(path, dpi=args.dpi); plt.close(fig); written.append(path)

    print(f"wrote {len(written)} figure(s) to {out}/")
    for path in written:
        print(f"  {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
