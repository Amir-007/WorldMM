#!/usr/bin/env python3
"""
Collect the Objective 3 measurements.  [compression, DB size, indexing latency]

Walks the artifacts both conditions produced and emits one `metrics.json` plus
a CSV per figure, so the numbers are usable whether or not a plotting library
is available.

    python metrics/collect_metrics.py --person A1_JAKE
    python metrics/collect_metrics.py --person A1_JAKE --with-model   # + index() timing

Missing inputs are reported and skipped rather than raising, so this is safe to
run while stages are still finishing.

The core latency benchmark is model-free on purpose: it measures loading a
database and reconstructing the knowledge state at a query time, which is
exactly what the interval encoding changes. `--with-model` additionally times
the full `SemanticMemory.index()`, which needs a GPU and the embedding model.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536          # VLM2Vec-V2 visual embedding width
FLOAT_BYTES = 4


def _read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _size_bytes(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


class Paths:
    """Where each stage leaves its output."""

    def __init__(self, root: str, person: str, condition: str, model: str) -> None:
        self.chunks_dir = os.path.join(root, "chunks", person, condition)
        self.episodic_dir = os.path.join(root, "metadata", "episodic_memory",
                                         f"{person}_{condition}")
        self.semantic_dir = os.path.join(root, "metadata", "semantic_memory",
                                         f"{person}_{condition}")
        self.consolidation_dir = os.path.join(root, "consolidation", f"{person}_{condition}")
        self.model = model

    @property
    def chunks_meta(self): return os.path.join(self.chunks_dir, "chunks_meta.json")
    @property
    def chunk_sweep(self): return os.path.join(self.chunks_dir, "threshold_sweep.json")
    @property
    def boundaries(self): return os.path.join(self.chunks_dir, "boundaries.json")
    @property
    def openie(self):
        return os.path.join(self.episodic_dir, f"openie_results_{self.model}.json")
    @property
    def episodic(self):
        return os.path.join(self.episodic_dir, f"episodic_triple_results_{self.model}.json")
    @property
    def semantic(self):
        return os.path.join(self.semantic_dir,
                            f"semantic_extraction_results_{self.model}.json")
    @property
    def interval(self):
        return os.path.join(self.consolidation_dir, "consolidated_interval.json")
    @property
    def worldmm_format(self):
        return os.path.join(self.consolidation_dir, "consolidated_worldmm_format.json")
    @property
    def consolidation_metrics(self):
        return os.path.join(self.consolidation_dir, "consolidation_metrics.json")


# --------------------------------------------------------------------------
# per-stage collection
# --------------------------------------------------------------------------

def collect_chunking(paths: Paths) -> Dict[str, Any]:
    meta = _read_json(paths.chunks_meta)
    if not meta:
        return {"available": False}

    lengths = [c["duration_seconds"] for c in meta["chunks"]]
    texts = [len(c["text"]) for c in meta["chunks"]]
    out = {
        "available": True,
        "statistics": meta["statistics"],
        "config": meta["config"],
        "length_seconds": {
            "mean": statistics.fmean(lengths), "median": statistics.median(lengths),
            "p05": _pct(lengths, 5), "p95": _pct(lengths, 95),
            "min": min(lengths), "max": max(lengths),
        },
        "text_chars": {
            "mean": statistics.fmean(texts), "median": statistics.median(texts),
            "p95": _pct(texts, 95), "max": max(texts),
        },
        "total_hours": sum(lengths) / 3600.0,
        "_lengths": lengths,
    }

    boundaries = _read_json(paths.boundaries)
    if boundaries:
        detected = sum(d.get("n_detected_cuts", 0) for d in boundaries.values())
        cap = sum(d.get("n_cap_cuts", 0) for d in boundaries.values())
        gap = sum(d.get("n_forced_cuts", 0) for d in boundaries.values())
        total = detected + cap + gap
        out["cut_origins"] = {
            "detected": detected, "cap": cap, "gap": gap,
            "pct_detected": 100.0 * detected / total if total else 0.0,
        }
    return out


def collect_episodic(paths: Paths) -> Dict[str, Any]:
    openie = _read_json(paths.openie)
    if not openie:
        return {"available": False}

    triples = openie["triple_results"]
    counts = [len(v) for v in triples.values()]
    unique = {tuple(str(x).strip().lower() for x in t)
              for v in triples.values() for t in v if len(t) == 3}
    entities = {e.strip().lower() for v in openie["ner_results"].values()
                for e in v if str(e).strip()}
    return {
        "available": True,
        "n_chunks": len(triples),
        "total_triples": sum(counts),
        "unique_triples": len(unique),
        "duplicate_pct": 100.0 * (1 - len(unique) / max(1, sum(counts))),
        "empty_chunks": sum(1 for c in counts if c == 0),
        "empty_pct": 100.0 * sum(1 for c in counts if c == 0) / max(1, len(counts)),
        "mean_per_chunk": statistics.fmean(counts) if counts else 0.0,
        "distinct_entities": len(entities),
        "bytes_openie": _size_bytes(paths.openie),
        "bytes_episodic": _size_bytes(paths.episodic),
    }


def collect_semantic(paths: Paths) -> Dict[str, Any]:
    data = _read_json(paths.semantic)
    if not data:
        return {"available": False}
    nodes = data["semantic_triples"]
    counts = [len(v) for v in nodes.values()]
    return {
        "available": True,
        "n_nodes": len(nodes),
        "non_empty_nodes": sum(1 for c in counts if c),
        "total_triples": sum(counts),
        "mean_per_node": statistics.fmean(counts) if counts else 0.0,
        "bytes": _size_bytes(paths.semantic),
    }


def collect_consolidation(paths: Paths) -> Dict[str, Any]:
    metrics = _read_json(paths.consolidation_metrics)
    if not metrics:
        return {"available": False}

    interval_bytes = _size_bytes(paths.interval)
    nested_bytes = _size_bytes(paths.worldmm_format)
    operating = None
    store = _read_json(paths.interval)
    if store:
        operating = store.get("metadata", {}).get("threshold")

    return {
        "available": True,
        "baseline": metrics["baseline"],
        "sweep": metrics["sweep"],
        "embedding_calls": metrics["embedding_calls"],
        "llm_calls": 0,
        "operating_threshold": operating,
        "bytes_interval": interval_bytes,
        "bytes_worldmm_format": nested_bytes,
        "storage_reduction": nested_bytes / interval_bytes if interval_bytes else 0.0,
    }


def collect_worldmm_reference(root: str, person: str, model: str) -> Dict[str, Any]:
    """
    WorldMM's own consolidation output, measured.

    The inherited artifact is the only direct evidence of how the reference
    implementation actually behaves, since re-running its per-triple LLM
    consolidation on the rebuilt data would cost ~10 GPU-hours per condition.
    It ran on a different (smaller) input, so it anchors the *behaviour* -
    monotonic growth and per-node rewriting - rather than serving as a
    like-for-like size comparison.
    """
    path = os.path.join(root, "metadata", "semantic_memory", person,
                        f"semantic_consolidation_results_{model}.json")
    data = _read_json(path)
    if not data:
        return {"available": False}
    sizes = [len(v.get("consolidated_semantic_triples", [])) for v in data.values()]
    if not sizes:
        return {"available": False}
    non_decreasing = sum(1 for a, b in zip(sizes, sizes[1:]) if b >= a)
    return {
        "available": True,
        "note": "inherited artifact; different input from the rebuilt conditions",
        "n_nodes": len(sizes),
        "final_state": sizes[-1],
        "peak_state": max(sizes),
        "total_triples_written": sum(sizes),
        "write_amplification": sum(sizes) / max(1, max(sizes)),
        "monotonic_steps": non_decreasing,
        "monotonic_pct": 100.0 * non_decreasing / max(1, len(sizes) - 1),
        "bytes": _size_bytes(path),
        "_curve": sizes,
    }


def collect_visual(paths: Paths, n_chunks: int, inherited_pkl: str) -> Dict[str, Any]:
    """
    Visual memory size.

    Computed analytically rather than measured: one VLM2Vec vector per chunk at
    a fixed width, so the size follows from the chunk count exactly. The
    inherited pickle is measured alongside to calibrate serialisation overhead
    against the raw array size.
    """
    raw = n_chunks * EMBEDDING_DIM * FLOAT_BYTES
    out = {"n_vectors": n_chunks, "dim": EMBEDDING_DIM, "raw_bytes": raw,
           "analytic": True}
    if os.path.exists(inherited_pkl):
        measured = _size_bytes(inherited_pkl)
        try:
            import pickle
            with open(inherited_pkl, "rb") as f:
                vectors = len(pickle.load(f))
            out["reference_pkl_bytes"] = measured
            out["reference_pkl_vectors"] = vectors
            overhead = measured / (vectors * EMBEDDING_DIM * FLOAT_BYTES)
            out["serialisation_overhead"] = overhead
            out["estimated_bytes"] = int(raw * overhead)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", inherited_pkl, exc)
    out.setdefault("estimated_bytes", raw)
    return out


# --------------------------------------------------------------------------
# indexing latency
# --------------------------------------------------------------------------

def benchmark_indexing(paths: Paths, n_queries: int = 20,
                       repeats: int = 3) -> Dict[str, Any]:
    """
    Time loading a semantic database and reconstructing the state at a query.

    Two formats on identical content: WorldMM's nested per-node layout, where a
    query reads the whole file to reach one snapshot, and the interval store,
    where the state is filtered out of a single flat table. Model-free, so this
    isolates the storage format rather than embedding throughput.
    """
    from worldmm.consolidation import IntervalTripleStore

    result: Dict[str, Any] = {"n_queries": n_queries, "repeats": repeats}

    nested_path, interval_path = paths.worldmm_format, paths.interval
    if not (os.path.exists(nested_path) and os.path.exists(interval_path)):
        return {"available": False}

    nested = _read_json(nested_path)
    query_keys = sorted(nested)
    step = max(1, len(query_keys) // n_queries)
    queries = query_keys[::step][:n_queries]

    # --- WorldMM nested format ---
    load_times, query_times, state_sizes = [], [], []
    for _ in range(repeats):
        started = time.perf_counter()
        with open(nested_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        load_times.append(time.perf_counter() - started)
        for key in queries:
            started = time.perf_counter()
            state = data[key]["consolidated_semantic_triples"]
            query_times.append(time.perf_counter() - started)
            state_sizes.append(len(state))
    result["worldmm_format"] = {
        "load_seconds": statistics.fmean(load_times),
        "query_seconds_mean": statistics.fmean(query_times),
        "bytes": _size_bytes(nested_path),
        "mean_state_triples": statistics.fmean(state_sizes),
    }

    # --- interval store ---
    load_times, query_times, state_sizes = [], [], []
    for _ in range(repeats):
        started = time.perf_counter()
        store = IntervalTripleStore.read(interval_path)
        load_times.append(time.perf_counter() - started)
        for key in queries:
            started = time.perf_counter()
            state = store.triples_at(key)
            query_times.append(time.perf_counter() - started)
            state_sizes.append(len(state))
    result["interval_store"] = {
        "load_seconds": statistics.fmean(load_times),
        "query_seconds_mean": statistics.fmean(query_times),
        "bytes": _size_bytes(interval_path),
        "mean_state_triples": statistics.fmean(state_sizes),
    }

    w, i = result["worldmm_format"], result["interval_store"]
    result["load_speedup"] = w["load_seconds"] / max(1e-9, i["load_seconds"])
    result["state_size_reduction"] = (
        w["mean_state_triples"] / max(1e-9, i["mean_state_triples"]))
    result["available"] = True
    return result


def benchmark_with_model(paths: Paths, n_queries: int = 5) -> Dict[str, Any]:
    """Time the real `SemanticMemory.index()` on both formats. Needs a GPU."""
    from worldmm.embedding import EmbeddingModel
    from worldmm.memory.semantic import SemanticMemory
    from worldmm.consolidation import IntervalTripleStore

    embedding_model = EmbeddingModel(text_model_name="Qwen/Qwen3-Embedding-4B")
    embedding_model.load_model(model_type="text")

    nested = _read_json(paths.worldmm_format)
    keys = sorted(nested)
    queries = keys[::max(1, len(keys) // n_queries)][:n_queries]

    out: Dict[str, Any] = {"available": True, "n_queries": len(queries)}
    for label, data in (
        ("worldmm_format", nested),
        ("interval_store",
         IntervalTripleStore.read(paths.interval).materialise_worldmm_format(keys)),
    ):
        memory = SemanticMemory(embedding_model=embedding_model)
        memory.load_triples_from_data(data)
        timings = []
        for key in queries:
            memory.reset_index()
            started = time.perf_counter()
            memory.index(int(key))
            timings.append(time.perf_counter() - started)
        out[label] = {"index_seconds_mean": statistics.fmean(timings),
                      "index_seconds_max": max(timings)}
    out["index_speedup"] = (out["worldmm_format"]["index_seconds_mean"]
                            / max(1e-9, out["interval_store"]["index_seconds_mean"]))
    return out


# --------------------------------------------------------------------------

def write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--conditions", default="fixed,event")
    parser.add_argument("--model", default="qwen3vl-30b")
    parser.add_argument("--root", default="output")
    parser.add_argument("--output-dir", default="output/metrics")
    parser.add_argument("--with-model", action="store_true",
                        help="Also time the real SemanticMemory.index() (needs a GPU).")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    report: Dict[str, Any] = {"person": args.person, "model": args.model,
                              "conditions": {}}
    report["worldmm_reference"] = collect_worldmm_reference(
        args.root, args.person, args.model)

    for condition in conditions:
        paths = Paths(args.root, args.person, condition, args.model)
        entry: Dict[str, Any] = {}
        entry["chunking"] = collect_chunking(paths)
        entry["episodic"] = collect_episodic(paths)
        entry["semantic"] = collect_semantic(paths)
        entry["consolidation"] = collect_consolidation(paths)
        n_chunks = entry["chunking"].get("statistics", {}).get("n_chunks", 0)
        entry["visual"] = collect_visual(
            paths, n_chunks,
            os.path.join(args.root, "metadata", "visual_memory", args.person,
                         "visual_embeddings.pkl"))
        if not args.skip_latency:
            try:
                entry["latency"] = benchmark_indexing(paths)
            except Exception as exc:  # noqa: BLE001
                logger.warning("latency benchmark skipped for %s: %s", condition, exc)
                entry["latency"] = {"available": False, "error": str(exc)}
        if args.with_model:
            try:
                entry["latency_with_model"] = benchmark_with_model(paths)
            except Exception as exc:  # noqa: BLE001
                logger.warning("model latency skipped for %s: %s", condition, exc)
                entry["latency_with_model"] = {"available": False, "error": str(exc)}

        # Total on-disk database, the Objective 3 headline size.
        total = (entry["episodic"].get("bytes_openie", 0)
                 + entry["episodic"].get("bytes_episodic", 0)
                 + entry["semantic"].get("bytes", 0)
                 + entry["consolidation"].get("bytes_interval", 0)
                 + entry["visual"].get("estimated_bytes", 0))
        baseline_total = (entry["episodic"].get("bytes_openie", 0)
                          + entry["episodic"].get("bytes_episodic", 0)
                          + entry["semantic"].get("bytes", 0)
                          + entry["consolidation"].get("bytes_worldmm_format", 0)
                          + entry["visual"].get("estimated_bytes", 0))
        entry["database"] = {
            "total_bytes_interval": total,
            "total_bytes_worldmm": baseline_total,
            "reduction": baseline_total / total if total else 0.0,
        }
        report["conditions"][condition] = entry
        missing = [k for k, v in entry.items()
                   if isinstance(v, dict) and v.get("available") is False]
        print(f"{condition}: collected"
              + (f"  (missing: {', '.join(missing)})" if missing else ""))

    # --- CSVs, one per figure -------------------------------------------
    out = args.output_dir

    rows = []
    for condition in conditions:
        lengths = report["conditions"][condition]["chunking"].get("_lengths", [])
        rows.extend({"condition": condition, "duration_seconds": round(v, 2)}
                    for v in lengths)
    write_csv(os.path.join(out, "fig_chunk_lengths.csv"), rows,
              ["condition", "duration_seconds"])

    rows = []
    for condition in conditions:
        for row in report["conditions"][condition]["consolidation"].get("sweep", []):
            rows.append({"condition": condition, **row})
    write_csv(os.path.join(out, "fig_consolidation_sweep.csv"), rows,
              ["condition", "threshold", "n_final", "compression_ratio",
               "mean_support", "singletons", "distinct_predicates", "seconds"])

    rows = []
    for condition in conditions:
        sweep = _read_json(Paths(args.root, args.person, condition,
                                 args.model).chunk_sweep) or []
        merged: Dict[float, Dict[str, float]] = {}
        for row in sweep:
            acc = merged.setdefault(row["threshold"], {"n": 0, "total": 0.0})
            acc["n"] += row["n_segments"]
            acc["total"] += row["total_seconds"]
        for threshold in sorted(merged):
            acc = merged[threshold]
            rows.append({"condition": condition, "threshold": threshold,
                         "segments": int(acc["n"]),
                         "mean_seconds": round(acc["total"] / max(1, acc["n"]), 2)})
    write_csv(os.path.join(out, "fig_boundary_sweep.csv"), rows,
              ["condition", "threshold", "segments", "mean_seconds"])

    # Triples-per-node growth: WorldMM's cumulative store vs the interval store.
    rows = []
    for condition in conditions:
        paths = Paths(args.root, args.person, condition, args.model)
        nested = _read_json(paths.worldmm_format)
        if not nested:
            continue
        try:
            from worldmm.consolidation import IntervalTripleStore
            store = IntervalTripleStore.read(paths.interval)
        except Exception:  # noqa: BLE001
            store = None
        for i, key in enumerate(sorted(nested)):
            row = {"condition": condition, "node_index": i, "node_key": key,
                   "worldmm_state": len(nested[key]["consolidated_semantic_triples"])}
            if store is not None:
                row["interval_state"] = len(store.snapshot_at(key))
            rows.append(row)
    reference = report.get("worldmm_reference", {})
    for i, size in enumerate(reference.pop("_curve", [])):
        rows.append({"condition": "worldmm_inherited", "node_index": i,
                     "node_key": "", "worldmm_state": size, "interval_state": ""})
    write_csv(os.path.join(out, "fig_growth_curve.csv"), rows,
              ["condition", "node_index", "node_key", "worldmm_state", "interval_state"])

    rows = []
    for condition in conditions:
        entry = report["conditions"][condition]
        rows.append({
            "condition": condition,
            "episodic_mb": (entry["episodic"].get("bytes_openie", 0)
                            + entry["episodic"].get("bytes_episodic", 0)) / 1e6,
            "semantic_mb": entry["semantic"].get("bytes", 0) / 1e6,
            "consolidated_interval_mb": entry["consolidation"].get("bytes_interval", 0) / 1e6,
            "consolidated_worldmm_mb": entry["consolidation"].get("bytes_worldmm_format", 0) / 1e6,
            "visual_mb": entry["visual"].get("estimated_bytes", 0) / 1e6,
            "total_interval_mb": entry["database"]["total_bytes_interval"] / 1e6,
            "total_worldmm_mb": entry["database"]["total_bytes_worldmm"] / 1e6,
        })
    write_csv(os.path.join(out, "fig_database_size.csv"), rows,
              list(rows[0]) if rows else [])

    for entry in report["conditions"].values():
        entry["chunking"].pop("_lengths", None)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report, conditions)
    print(f"\nwrote {out}/metrics.json and {len(os.listdir(out))} file(s)")
    return 0


def _print_summary(report: Dict[str, Any], conditions: List[str]) -> None:
    def cell(condition, *keys, default=0):
        node = report["conditions"][condition]
        for key in keys:
            node = node.get(key, {}) if isinstance(node, dict) else default
        return node if not isinstance(node, dict) else default

    print(f"\n{'':32s}" + "".join(f"{c.upper():>16s}" for c in conditions))
    print("-" * (32 + 16 * len(conditions)))
    lines = [
        ("chunks", ("chunking", "statistics", "n_chunks"), "{:,.0f}"),
        ("mean chunk length (s)", ("chunking", "length_seconds", "mean"), "{:,.1f}"),
        ("median chunk length (s)", ("chunking", "length_seconds", "median"), "{:,.1f}"),
        ("cuts detected (%)", ("chunking", "cut_origins", "pct_detected"), "{:,.1f}"),
        ("episodic triples", ("episodic", "total_triples"), "{:,.0f}"),
        ("episodic empty (%)", ("episodic", "empty_pct"), "{:,.2f}"),
        ("semantic nodes", ("semantic", "n_nodes"), "{:,.0f}"),
        ("semantic triples", ("semantic", "total_triples"), "{:,.0f}"),
        ("consolidated triples", ("consolidation", "sweep"), None),
        ("storage reduction (x)", ("consolidation", "storage_reduction"), "{:,.1f}"),
        ("DB total, interval (MB)", ("database", "total_bytes_interval"), None),
        ("DB total, WorldMM (MB)", ("database", "total_bytes_worldmm"), None),
        ("DB reduction (x)", ("database", "reduction"), "{:,.1f}"),
    ]
    for label, keys, fmt in lines:
        cells = []
        for condition in conditions:
            value = cell(condition, *keys)
            if label.startswith("consolidated"):
                sweep = report["conditions"][condition]["consolidation"].get("sweep", [])
                thresh = report["conditions"][condition]["consolidation"].get(
                    "operating_threshold")
                match = next((r for r in sweep if r["threshold"] == thresh), None)
                cells.append(f"{match['n_final']:,}" if match else "-")
            elif label.startswith("cuts detected") and not report["conditions"][
                    condition]["chunking"].get("cut_origins"):
                # The fixed grid has no detector, so this is inapplicable
                # rather than zero.
                cells.append("n/a")
            elif "MB" in label:
                cells.append(f"{value / 1e6:,.1f}")
            else:
                cells.append(fmt.format(value) if fmt else str(value))
        print(f"{label:32s}" + "".join(f"{c:>16s}" for c in cells))

    reference = report.get("worldmm_reference", {})
    if reference.get("available"):
        print(f"\nWorldMM reference (inherited artifact, different input): "
              f"{reference['n_nodes']:,} nodes, final state {reference['final_state']:,}, "
              f"{reference['total_triples_written']:,} triples written "
              f"({reference['write_amplification']:.0f}x), "
              f"{reference['monotonic_pct']:.0f}% of steps non-decreasing, "
              f"{reference['bytes'] / 1e6:.1f} MB")

    for condition in conditions:
        modelled = report["conditions"][condition].get("latency_with_model", {})
        if modelled.get("available"):
            w = modelled["worldmm_format"]["index_seconds_mean"]
            i = modelled["interval_store"]["index_seconds_mean"]
            print(f"\n{condition}: SemanticMemory.index() {w:.2f}s -> {i:.2f}s "
                  f"({modelled['index_speedup']:.2f}x, n={modelled['n_queries']})")

    for condition in conditions:
        latency = report["conditions"][condition].get("latency", {})
        if latency.get("available"):
            w, i = latency["worldmm_format"], latency["interval_store"]
            print(f"\n{condition}: DB load {w['load_seconds']:.2f}s -> "
                  f"{i['load_seconds']:.2f}s ({latency['load_speedup']:.1f}x), "
                  f"state {w['mean_state_triples']:.0f} -> "
                  f"{i['mean_state_triples']:.0f} triples "
                  f"({latency['state_size_reduction']:.2f}x)")


if __name__ == "__main__":
    raise SystemExit(main())
