#!/usr/bin/env python3
"""
Rebuild HippoRAG's OpenIE cache from surviving worldmm OpenIE results.

Two different files live side by side in each cache directory:

  openie_results_ner_<model>.json   HippoRAG's cache.  {"docs": [{idx, passage,
                                    extracted_entities, extracted_triples}, ...]}
  openie_results_<model>.json       written by OpenIE.save_results.
                                    {"ner_results": {...}, "triple_results": {...}}

Only the first is read by HippoRAG.load_existing_openie. If it is missing,
every indexed caption is re-extracted with the LLM, which costs hours. The
second file holds the same extractions but is keyed by chunk id and carries no
passage text, so this script recovers the passages by re-hashing the captions:
chunk id is "chunk-" + md5(caption["text"]), the same identity HippoRAG uses.

    python tools/rebuild_hipporag_openie_cache.py \
        --cache-dir .cache/episodic_memory \
        --caption-dir data/EgoLife/EgoLifeCap/A1_JAKE \
        --subject A1_JAKE --model qwen3vl-30b

Add --write to actually create the files; the default is a dry run.
"""

import argparse
import json
import os
from hashlib import md5

GRANULARITIES = ["30sec", "3min", "10min", "1h"]


def chunk_id(text: str) -> str:
    return "chunk-" + md5(text.encode()).hexdigest()


def rebuild_one(cache_dir, caption_file, granularity, model, write, extra_sources=(), merge=False):
    """
    Merge every available source, then join to captions by md5 of the text.

    Sources are merged rather than picked because the copy inside .cache is
    rewritten on every batch and therefore holds only the final batch. The
    authoritative extraction for 30sec lives in the preprocess output. Feeding a
    source for the wrong granularity is harmless: no caption will hash to its
    keys, so it contributes nothing.
    """
    target = os.path.join(cache_dir, granularity, f"openie_results_ner_{model}.json")

    candidates = [os.path.join(cache_dir, granularity, f"openie_results_{model}.json")]
    candidates.extend(extra_sources)
    candidates = [c for c in candidates if os.path.exists(c)]

    if not os.path.exists(caption_file):
        return (granularity, "no caption file", 0, 0)

    # An interrupted eval leaves a partial cache: HippoRAG accumulates as it
    # indexes, so the file is real but only covers captions reached so far.
    # Merging tops it up instead of discarding it or re-extracting from scratch.
    existing = {}
    if os.path.exists(target):
        if not merge:
            return (granularity, "target exists, use --merge to top it up", 0, 0)
        try:
            with open(target, encoding="utf-8") as handle:
                for doc in json.load(handle).get("docs", []):
                    if doc.get("passage"):
                        existing[chunk_id(doc["passage"])] = doc
        except (json.JSONDecodeError, OSError) as exc:
            return (granularity, f"target unreadable ({type(exc).__name__})", 0, 0)

    if not candidates and not existing:
        return (granularity, "no source file", 0, 0)

    ner, triples = {}, {}
    for candidate in candidates:
        with open(candidate, encoding="utf-8") as handle:
            data = json.load(handle)
        ner.update(data.get("ner_results", {}))
        triples.update(data.get("triple_results", {}))

    with open(caption_file, encoding="utf-8") as handle:
        captions = json.load(handle)

    docs, added = [], 0
    seen = set()
    for entry in captions:
        text = entry.get("text", "")
        if not text:
            continue
        key = chunk_id(text)
        if key in seen:
            continue
        seen.add(key)
        if key in existing:
            docs.append(existing[key])          # keep what the run already extracted
            continue
        if key not in ner and key not in triples:
            continue
        added += 1
        docs.append({
            "idx": key,
            "passage": text,
            "extracted_entities": ner.get(key, []),
            "extracted_triples": triples.get(key, []),
        })

    total_keys = len(captions)
    if write and docs and (added or not existing):
        if existing:
            backup = target + ".bak"
            if not os.path.exists(backup):
                os.replace(target, backup)
        entity_count = sum(len(d["extracted_entities"]) for d in docs)
        chars = sum(len(e) for d in docs for e in d["extracted_entities"])
        words = sum(len(e.split()) for d in docs for e in d["extracted_entities"])
        payload = {
            "docs": docs,
            "avg_ent_chars": round(chars / entity_count, 4) if entity_count else 0,
            "avg_ent_words": round(words / entity_count, 4) if entity_count else 0,
        }
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    shortfall = total_keys - len(docs)
    if added:
        status = f"merged (+{added})" if write else f"would merge (+{added})"
    elif existing:
        # No source could top this one up. Say so plainly rather than implying it
        # is finished: a partial cache still costs LLM calls at eval time.
        status = "complete" if shortfall <= 0 else f"no source, {shortfall} still uncached"
    else:
        status = "rebuilt" if write else "would rebuild"
    if shortfall > 0 and added:
        status += f", {shortfall} short"
    return (granularity, status, len(docs), total_keys)


def main():
    parser = argparse.ArgumentParser(description="Rebuild HippoRAG OpenIE cache from worldmm results.")
    parser.add_argument("--cache-dir", default=".cache/episodic_memory")
    parser.add_argument("--caption-dir", default="data/EgoLife/EgoLifeCap/A1_JAKE")
    parser.add_argument("--subject", default="A1_JAKE")
    parser.add_argument("--model", default="qwen3vl-30b")
    parser.add_argument("--extra-source", action="append", default=[],
                        help="Additional openie_results_<model>.json to merge in. Repeatable. "
                             "Use the preprocess build, whose .cache copy is only ever "
                             "the last batch written.")
    parser.add_argument("--metadata-dir", default="output/metadata",
                        help="Used to locate the preprocess OpenIE build automatically.")
    parser.add_argument("--merge", action="store_true", help="Top up an existing cache instead of skipping it. The original is kept as <file>.bak.")
    parser.add_argument("--write", action="store_true", help="Actually write; default is a dry run.")
    args = parser.parse_args()

    extra = list(args.extra_source)
    preprocess_build = os.path.join(args.metadata_dir, "episodic_memory", args.subject,
                                    f"openie_results_{args.model}.json")
    if os.path.exists(preprocess_build) and preprocess_build not in extra:
        extra.append(preprocess_build)
        print(f"merging preprocess build: {preprocess_build}\n")

    print(f"{'granularity':<12} {'status':<34} {'cached':>10} {'captions':>10}")
    print("-" * 70)
    total = 0
    for granularity in GRANULARITIES:
        caption_file = os.path.join(args.caption_dir, f"{args.subject}_{granularity}.json")
        g, status, matched, keys = rebuild_one(
            args.cache_dir, caption_file, granularity, args.model, args.write, extra, args.merge)
        total += matched
        print(f"{g:<12} {status:<34} {matched:>10} {keys:>10}")
    print("-" * 70)
    print(f"{'total':<12} {'':<34} {total:>10}")
    if not args.write:
        print("\nDry run. Re-run with --write to create the files.")


if __name__ == "__main__":
    main()
