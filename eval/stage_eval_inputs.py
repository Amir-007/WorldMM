#!/usr/bin/env python3
"""
Stage one chunking condition into the layout `eval_egolife.py` expects.

That script resolves every input from --data-dir, --metadata-dir and --subject,
so a condition can be evaluated by building those trees rather than by editing
the evaluation itself. Nothing in eval/eval_egolife.py is modified, which keeps
Track B's evaluation and this comparison independent.

    python eval/stage_eval_inputs.py --condition fixed
    python eval/stage_eval_inputs.py --condition event

Three things are bridged:

  captions   Our chunks become the "30sec" granularity. The coarser levels are
             written as copies, because the pre-flight check requires all four
             to exist - but retrieval is restricted to the base level via
             --episodic-candidates "30sec:5". The shipped 3min/10min/1h files
             describe the fixed 30s grid, so using them would leak baseline
             structure into the event condition.

  semantic   The consolidated store, materialised into WorldMM's nested layout.

  visual     Chunk embeddings are the mean of their constituent segment
             embeddings, taken from the inherited per-mp4 bank. For a fixed
             chunk that is one segment, so the mean is the identity - the same
             rule applied to both conditions rather than two different ones,
             and it needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

logger = logging.getLogger(__name__)

GRANULARITIES = ["30sec", "3min", "10min", "1h"]


def stage_captions(chunks: List[Dict[str, Any]], out_dir: str, subject: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    entries = [
        {"start_time": c["start_time"], "end_time": c["end_time"], "date": c["date"],
         "text": c["text"], "video_path": c["video_path"]}
        for c in chunks
    ]
    base = os.path.join(out_dir, f"{subject}_30sec.json")
    with open(base, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    for granularity in GRANULARITIES[1:]:
        shutil.copyfile(base, os.path.join(out_dir, f"{subject}_{granularity}.json"))
    logger.info("captions: %d entries -> %s (+%d copies for the pre-flight check)",
                len(entries), base, len(GRANULARITIES) - 1)


def stage_semantic(interval_path: str, nested_path: str, node_keys: List[str],
                   destination: str) -> None:
    from worldmm.consolidation import IntervalTripleStore

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(nested_path):
        shutil.copyfile(nested_path, destination)
        source = "materialised file"
    else:
        store = IntervalTripleStore.read(interval_path)
        with open(destination, "w", encoding="utf-8") as f:
            json.dump(store.materialise_worldmm_format(node_keys), f, ensure_ascii=False)
        source = "interval store"
    with open(destination, encoding="utf-8") as f:
        nodes = json.load(f)
    logger.info("semantic: %d node(s) from the %s -> %s", len(nodes), source, destination)


def stage_visual(chunks: List[Dict[str, Any]], bank_path: str, destination: str) -> None:
    """
    Mean-pool the inherited per-segment embeddings onto chunks.

    Keyed by start_time rather than video_path: `load_clips_from_data` tries
    clip_id, then video_path, then start_time, and an event chunk can share its
    first video file with its neighbour, so video_path is not a unique key.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(bank_path):
        logger.warning("no visual bank at %s; writing an empty one "
                       "(visual retrieval will return nothing)", bank_path)
        with open(destination, "wb") as f:
            pickle.dump({}, f)
        return

    with open(bank_path, "rb") as f:
        bank = pickle.load(f)

    out: Dict[str, np.ndarray] = {}
    covered = partial = missing = 0
    for chunk in chunks:
        paths = chunk.get("video_paths") or [chunk.get("video_path", "")]
        vectors = [bank[p] for p in paths if p in bank]
        if not vectors:
            missing += 1
            continue
        if len(vectors) < len(paths):
            partial += 1
        out[str(chunk["start_time"])] = np.mean(np.stack(vectors), axis=0)
        covered += 1

    with open(destination, "wb") as f:
        pickle.dump(out, f)
    logger.info("visual: %d/%d chunk(s) embedded (%d partially covered, %d with no "
                "segment in the bank) -> %s",
                covered, len(chunks), partial, missing, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--condition", required=True, choices=["fixed", "event"])
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--model", default="qwen3vl-30b")
    parser.add_argument("--root", default="output")
    parser.add_argument("--data-dir", default="data/EgoLife",
                        help="Source of EgoLifeQA, symlinked into the staged tree.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions. Both conditions "
                             "take the same slice, so the comparison holds.")
    parser.add_argument("--stage-dir", default=None,
                        help="Defaults to output/eval/<condition>.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    person, condition = args.person, args.condition
    stage = args.stage_dir or os.path.join(args.root, "eval", condition)

    chunks_path = os.path.join(args.root, "chunks", person, condition, "chunks.json")
    if not os.path.exists(chunks_path):
        logger.error("chunks not found: %s", chunks_path)
        return 1
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("condition=%s  chunks=%d", condition, len(chunks))

    # --- QA set -----------------------------------------------------------
    qa_source = os.path.join(args.data_dir, "EgoLifeQA", f"EgoLifeQA_{person}.json")
    if not os.path.exists(qa_source):
        logger.error("EgoLifeQA not found: %s", qa_source)
        logger.error("  the evaluation cannot run without it")
        return 1
    qa_dir = os.path.join(stage, "data", "EgoLifeQA")
    os.makedirs(qa_dir, exist_ok=True)
    qa_dest = os.path.join(qa_dir, f"EgoLifeQA_{person}.json")
    if os.path.lexists(qa_dest):
        os.remove(qa_dest)
    with open(qa_source, encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        # Truncate here rather than adding a flag to eval_egolife.py, so that
        # file stays untouched. Both conditions take the same leading slice of
        # the same source, so they are answering identical questions.
        questions = questions[:args.limit]
        with open(qa_dest, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False)
        logger.info("EgoLifeQA: %d question(s) (capped from the full set)",
                    len(questions))
    else:
        os.symlink(os.path.abspath(qa_source), qa_dest)
        logger.info("EgoLifeQA: %d question(s)", len(questions))

    # --- captions, semantic, visual --------------------------------------
    stage_captions(chunks, os.path.join(stage, "data", "EgoLifeCap", person), person)

    consolidation_dir = os.path.join(args.root, "consolidation", f"{person}_{condition}")
    semantic_source = os.path.join(
        args.root, "metadata", "semantic_memory", f"{person}_{condition}",
        f"semantic_extraction_results_{args.model}.json")
    node_keys: List[str] = []
    if os.path.exists(semantic_source):
        with open(semantic_source, encoding="utf-8") as f:
            node_keys = sorted(json.load(f)["semantic_triples"])
    stage_semantic(
        os.path.join(consolidation_dir, "consolidated_interval.json"),
        os.path.join(consolidation_dir, "consolidated_worldmm_format.json"),
        node_keys,
        os.path.join(stage, "metadata", "semantic_memory", person,
                     f"semantic_consolidation_results_{args.model}.json"))

    stage_visual(
        chunks,
        os.path.join(args.root, "metadata", "visual_memory", person,
                     "visual_embeddings.pkl"),
        os.path.join(stage, "metadata", "visual_memory", person,
                     "visual_embeddings.pkl"))

    print(f"\nstaged {condition} -> {stage}")
    print(f"  --data-dir     {os.path.join(stage, 'data')}")
    print(f"  --metadata-dir {os.path.join(stage, 'metadata')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
