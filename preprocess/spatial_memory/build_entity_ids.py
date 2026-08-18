#!/usr/bin/env python3
"""
Build the spatial (Entity ID) memory bank from captions and OpenIE results.

Pure text processing: no GPU, no model, runs locally in seconds.

  python preprocess/spatial_memory/build_entity_ids.py \
      --caption-file data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json \
      --openie-file output/metadata/episodic_memory/A1_JAKE/openie_results_qwen3vl-30b.json \
      --output-dir output/metadata/spatial_memory/A1_JAKE

Prints the ambiguity budget, the count of surface forms splitting into two or
more Entity IDs. That count is how much genuine ambiguity the data contains,
and so how much signal there is for the retrieval loop to act on.
Use --sweep to compare carry-forward windows without writing anything.
"""

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _import_spatial_build():
    """
    Import worldmm.memory.spatial.build without executing the parent packages.

    worldmm/memory/__init__.py pulls in WorldMemory, which needs torch and
    tenacity. The spatial bank is deliberately pure Python so it can be built and
    tested on a laptop, so we register lightweight stand-ins for the parent
    packages and let the real submodules load underneath them.
    """
    import types
    for name, path in (
        ("worldmm", os.path.join(_SRC, "worldmm")),
        ("worldmm.memory", os.path.join(_SRC, "worldmm", "memory")),
        ("worldmm.memory.spatial", os.path.join(_SRC, "worldmm", "memory", "spatial")),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module
    from worldmm.memory.spatial.build import (DEFAULT_CARRY_FORWARD, build_entity_bank,
                                              save_entity_bank)
    return DEFAULT_CARRY_FORWARD, build_entity_bank, save_entity_bank


DEFAULT_CARRY_FORWARD, build_entity_bank, save_entity_bank = _import_spatial_build()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def report(stats, entity_source):
    print()
    print("=" * 68)
    print(f"SPATIAL MEMORY BANK  ({entity_source}, carry-forward="
          f"{stats['carry_forward_window']})")
    print("=" * 68)
    print(f"  captions                     : {stats['n_captions']}")
    print(f"  captions with a location     : {stats['n_captions_localised']} "
          f"({100 * stats['caption_location_coverage']:.1f}%)")
    print(f"  localised entity mentions    : {stats['n_localised_mentions']}")
    print(f"  distinct surface forms       : {stats['n_surface_forms']}")
    print(f"  Entity IDs                   : {stats['n_entity_ids']}")
    print()
    print(f"  AMBIGUITY BUDGET             : {stats['ambiguity_budget']}"
          "   (surface forms with >=2 Entity IDs)")
    print(f"  usable budget                : {stats['usable_ambiguity_budget']}"
          "   (2-3 locations, each with >=2 mentions)")
    threshold = 20
    if stats["usable_ambiguity_budget"] < threshold:
        print(f"  WARNING: usable budget below {threshold}. Widen the location "
              "vocabulary or raise --carry-forward.")
    else:
        print(f"  OK: usable budget clears the threshold of {threshold}.")
    print()
    print("  locations:", ", ".join(f"{k}={v}" for k, v in stats["location_distribution"].items()))
    usable = stats["usable_entities"]
    if usable:
        print()
        print(f"  usable ambiguous entities (top {min(len(usable), 20)} of {len(usable)}):")
        for surface, locs in list(usable.items())[:20]:
            spread = ", ".join(f"{loc}:{n}" for loc, n in sorted(locs.items(), key=lambda x: -x[1]))
            print(f"    {surface:<28} {spread}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Build the spatial Entity ID memory bank.")
    parser.add_argument("--subject", type=str, default="A1_JAKE", help="Subject ID.")
    parser.add_argument("--caption-file", type=str, default=None,
                        help="Caption JSON. Defaults to the subject's 30sec file.")
    parser.add_argument("--openie-file", type=str, default=None,
                        help="OpenIE results JSON. Defaults from --metadata-dir and --model.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to <metadata-dir>/spatial_memory/<subject>.")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife", help="Data directory.")
    parser.add_argument("--metadata-dir", type=str, default="output/metadata",
                        help="Root metadata directory.")
    parser.add_argument("--model", type=str, default="qwen3vl-30b",
                        help="Model whose OpenIE results to read.")
    parser.add_argument("--carry-forward", type=int, default=DEFAULT_CARRY_FORWARD,
                        help="Unlabelled captions a location may span. 0 disables.")
    parser.add_argument("--use-ner", action="store_true",
                        help="Ablation: use raw NER entities instead of triple objects.")
    parser.add_argument("--sweep", action="store_true",
                        help="Report the budget across carry-forward windows, write nothing.")
    args = parser.parse_args()

    caption_file = args.caption_file or os.path.join(
        args.data_dir, "EgoLifeCap", args.subject, f"{args.subject}_30sec.json")
    openie_file = args.openie_file or os.path.join(
        args.metadata_dir, "episodic_memory", args.subject,
        f"openie_results_{args.model}.json")
    output_dir = args.output_dir or os.path.join(
        args.metadata_dir, "spatial_memory", args.subject)

    missing = [p for p in (caption_file, openie_file) if not os.path.exists(p)]
    if missing:
        for path in missing:
            logger.error("Missing required input: %s", path)
        raise SystemExit(1)

    source = "NER entities" if args.use_ner else "triple objects, manipulation verbs"

    if args.sweep:
        print(f"\ncarry-forward sweep ({source})")
        print(f"{'window':>8} {'localised':>10} {'entity ids':>11} {'budget':>8} {'usable':>8}")
        for window in (0, 2, 4, 6, 10):
            _, stats = build_entity_bank(caption_file, openie_file,
                                         carry_forward=window, use_ner=args.use_ner)
            print(f"{window:>8} {stats['n_localised_mentions']:>10} "
                  f"{stats['n_entity_ids']:>11} {stats['ambiguity_budget']:>8} "
                  f"{stats['usable_ambiguity_budget']:>8}")
        return

    bank, stats = build_entity_bank(caption_file, openie_file,
                                    carry_forward=args.carry_forward,
                                    use_ner=args.use_ner)
    path = save_entity_bank(bank, stats, output_dir)
    report(stats, source)
    print(f"\nWritten to: {path}\n")


if __name__ == "__main__":
    main()
