#!/usr/bin/env python3
"""
Compare evaluation arms and produce the figures a write-up needs.

Reads the per-question result files two runs produce and reports each arm on its
own, then the paired comparison that only a join can give: how often the system
asked a clarifying question when the baseline already knew the answer
(over-triggering), and how often it asked when the baseline was wrong
(a hallucination avoided).

    python tools/compare_arms.py \
        --baseline output/qwen3vl_30b_qwen3vl_30b/egolife_eval_A1_JAKE_baseline.json \
        --system   output/qwen3vl_30b_qwen3vl_30b/egolife_eval_A1_JAKE_abstain0.75.json

Accepts .json or .jsonl. Add --markdown for a table to paste into a document,
or --csv PATH to write the per-question join for further analysis.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

FAILURE_RESPONSES = ("Error", "Unable to generate answer")


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load a results file, tolerating both the aggregated JSON and the JSONL checkpoint."""
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}")
    with open(path, encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            by_id = {}
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and "ID" in entry:
                    by_id[entry["ID"]] = entry      # later entries supersede
            return list(by_id.values())
        data = json.load(handle)
    return data if isinstance(data, list) else data.get("results", [])


def is_failed(entry: Dict[str, Any]) -> bool:
    if entry.get("failed"):
        return True
    return str(entry.get("response", "")).strip() in FAILURE_RESPONSES


def wilson(correct: int, n: int, z: float = 1.96):
    """
    Wilson score interval for a proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly for small counts, which matters for per-type breakdowns
    where a cell may hold only a few dozen questions.
    """
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(b: int, c: int):
    """
    Exact two-sided McNemar test on the discordant pairs.

    b and c are the counts that changed in each direction. Only discordant pairs
    carry information, so under the null each is a coin flip. Exact rather than
    chi-square because the discordant counts here are often small.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    failed = sum(1 for r in results if is_failed(r))
    abstained = sum(1 for r in results if r.get("abstained") and not is_failed(r))
    answered = total - abstained - failed
    correct = sum(1 for r in results if r.get("evaluate") is True)
    wrong = answered - correct
    lo, hi = wilson(correct, answered)
    return {
        "total": total, "answered": answered, "abstained": abstained,
        "failed": failed, "correct": correct, "wrong": wrong,
        "accuracy_answered": correct / answered if answered else 0.0,
        "accuracy_ci": (lo, hi),
        "accuracy_overall": correct / total if total else 0.0,
        "abstention_rate": abstained / total if total else 0.0,
        "hallucination_rate": wrong / total if total else 0.0,
    }


def arm_table(name: str, s: Dict[str, Any]) -> List[str]:
    lo, hi = s["accuracy_ci"]
    return [
        f"  {name}",
        f"    questions            {s['total']}",
        f"    answered             {s['answered']}",
        f"    abstained            {s['abstained']}   ({s['abstention_rate']:.1%})",
        f"    failed (infra)       {s['failed']}",
        f"    correct              {s['correct']}",
        f"    wrong                {s['wrong']}",
        f"    accuracy (answered)  {s['accuracy_answered']:.1%}  95% CI [{lo:.1%}, {hi:.1%}]",
        f"    accuracy (overall)   {s['accuracy_overall']:.1%}",
        f"    hallucination rate   {s['hallucination_rate']:.1%}",
    ]


def main():
    ap = argparse.ArgumentParser(description="Compare two evaluation arms.")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--baseline-name", default="baseline")
    ap.add_argument("--system-name", default="with abstention")
    ap.add_argument("--csv", default=None, help="Write the per-question join here.")
    ap.add_argument("--markdown", action="store_true", help="Emit a markdown table too.")
    ap.add_argument("--examples", type=int, default=5, help="Disambiguation questions to show.")
    args = ap.parse_args()

    base = {r["ID"]: r for r in load_results(args.baseline)}
    syst = {r["ID"]: r for r in load_results(args.system)}
    shared = sorted(set(base) & set(syst), key=str)

    print("=" * 74)
    print("ARM COMPARISON")
    print("=" * 74)
    print(f"  {args.baseline_name:<22} {len(base)} questions")
    print(f"  {args.system_name:<22} {len(syst)} questions")
    print(f"  compared on            {len(shared)} shared IDs")
    if len(shared) < max(len(base), len(syst)):
        print(f"  WARNING: {max(len(base), len(syst)) - len(shared)} question(s) appear in only one arm "
              "and are excluded from the paired comparison.")
    print()

    bs, ss = summarise(list(base.values())), summarise(list(syst.values()))
    print("\n".join(arm_table(args.baseline_name, bs)));  print()
    print("\n".join(arm_table(args.system_name, ss)));    print()

    # Paired outcomes on shared questions, excluding infrastructure failures in
    # either arm: those say nothing about the mechanism under test.
    cells = Counter()
    rows = []
    for qid in shared:
        b, s = base[qid], syst[qid]
        if is_failed(b) or is_failed(s):
            cells["excluded_failure"] += 1
            continue
        b_ok = b.get("evaluate") is True
        s_abst = bool(s.get("abstained"))
        s_ok = s.get("evaluate") is True
        if s_abst:
            key = "abstain_baseline_right" if b_ok else "abstain_baseline_wrong"
        elif b_ok and s_ok:   key = "both_right"
        elif b_ok and not s_ok: key = "system_lost"
        elif not b_ok and s_ok: key = "system_gained"
        else:                 key = "both_wrong"
        cells[key] += 1
        rows.append({
            "ID": qid, "type": b.get("type"),
            "baseline_correct": b_ok,
            "system_abstained": s_abst, "system_correct": s_ok,
            "confidence": s.get("confidence"),
            "surface_form": (s.get("ambiguity") or {}).get("surface_form"),
            "n_entity_ids": (s.get("ambiguity") or {}).get("n_matching_entity_ids"),
            "outcome": key,
        })

    n = sum(v for k, v in cells.items() if k != "excluded_failure")
    print("-" * 74)
    print("  PAIRED OUTCOMES  (shared questions, infra failures excluded)")
    print("-" * 74)
    over = cells["abstain_baseline_right"]
    avoid = cells["abstain_baseline_wrong"]
    print(f"    both correct                       {cells['both_right']:>5}")
    print(f"    both wrong                         {cells['both_wrong']:>5}")
    print(f"    system gained (base wrong -> right){cells['system_gained']:>5}")
    print(f"    system lost   (base right -> wrong){cells['system_lost']:>5}")
    print(f"    ABSTAINED, baseline was wrong      {avoid:>5}   <- hallucination avoided")
    print(f"    ABSTAINED, baseline was right      {over:>5}   <- over-trigger")
    print(f"    excluded (infra failure)           {cells['excluded_failure']:>5}")
    print()
    if n:
        print(f"    over-triggering rate    {over/n:.1%}   (abstained when baseline knew)")
        print(f"    hallucinations avoided  {avoid/n:.1%}   (abstained when baseline was wrong)")
        if over + avoid:
            print(f"    abstention precision    {avoid/(over+avoid):.1%}   "
                  "(of abstentions, share where baseline was wrong)")
            print("      above 50% means abstaining beats the baseline's guess more often than not")
    p = mcnemar(cells["system_lost"], cells["system_gained"])
    print(f"\n    McNemar (answered pairs only): gained={cells['system_gained']} "
          f"lost={cells['system_lost']}  p={p:.4f}"
          f"  {'significant at 0.05' if p < 0.05 else 'not significant at 0.05'}")

    # Per question type, where the effect often concentrates.
    by_type = defaultdict(lambda: Counter())
    for r in rows:
        t = r["type"] or "unknown"
        by_type[t]["n"] += 1
        by_type[t]["b_ok"] += r["baseline_correct"]
        by_type[t]["s_abst"] += r["system_abstained"]
        by_type[t]["s_ok"] += r["system_correct"]
    if by_type:
        print("\n" + "-" * 74)
        print("  BY QUESTION TYPE")
        print("-" * 74)
        print(f"    {'type':<22}{'n':>5}{'base acc':>10}{'sys acc':>9}{'abstain':>9}")
        for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
            ans = c["n"] - c["s_abst"]
            print(f"    {t:<22}{c['n']:>5}{c['b_ok']/c['n']:>10.1%}"
                  f"{(c['s_ok']/ans if ans else 0):>9.1%}{c['s_abst']/c['n']:>9.1%}")

    # Qualitative evidence: what the system actually asked.
    examples = [(qid, syst[qid]) for qid in shared
                if syst[qid].get("abstained") and not is_failed(syst[qid])]
    if examples:
        print("\n" + "-" * 74)
        print(f"  SAMPLE DISAMBIGUATION QUESTIONS ({min(args.examples, len(examples))} of {len(examples)})")
        print("-" * 74)
        for qid, s in examples[:args.examples]:
            amb = s.get("ambiguity") or {}
            print(f"    Q{qid}  conf={s.get('confidence')}  "
                  f"entity={amb.get('surface_form')!r}  locations={amb.get('locations')}")
            print(f"      asked: {s.get('response','')[:150]}")
            print(f"      baseline was: {'correct' if base[qid].get('evaluate') is True else 'wrong'}")

    if args.markdown:
        print("\n" + "-" * 74)
        print("  MARKDOWN")
        print("-" * 74)
        print(f"| metric | {args.baseline_name} | {args.system_name} |")
        print("|---|---|---|")
        print(f"| accuracy (answered) | {bs['accuracy_answered']:.1%} | {ss['accuracy_answered']:.1%} |")
        print(f"| accuracy (overall) | {bs['accuracy_overall']:.1%} | {ss['accuracy_overall']:.1%} |")
        print(f"| abstention rate | {bs['abstention_rate']:.1%} | {ss['abstention_rate']:.1%} |")
        print(f"| hallucination rate | {bs['hallucination_rate']:.1%} | {ss['hallucination_rate']:.1%} |")
        if n:
            print(f"| over-triggering rate | n/a | {over/n:.1%} |")
            print(f"| hallucinations avoided | n/a | {avoid/n:.1%} |")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["ID"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n  per-question join written to {args.csv}")
    print("=" * 74)


if __name__ == "__main__":
    main()
