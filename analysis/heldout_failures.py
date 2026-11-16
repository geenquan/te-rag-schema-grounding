"""Held-out-table disagreement and failure analysis (paper Sec. 4.6).

Categories over the held-out protocol for a (method_a, method_b) pair:
both_success / a_only / b_only / both_fail, with per-question gold-table
ranks inside the saved top-5 and a deterministic case-selection rule
(one per category, distinct target tables when the data allows).

All method names and question ids are parameters — nothing from the paper's
private runs is hardcoded.
"""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from analysis.saved_prefix import load_gold, load_run
from eval.full_gold import full_gold_metrics


def _covered(record, gold) -> bool:
    """Covered@5 against the full gold mapping (see eval.full_gold)."""
    return bool(full_gold_metrics(record, gold)[1])


def gold_table_ranks(record, gold_tables) -> Dict[str, Optional[int]]:
    """1-based rank of each gold table within the saved top-5 (None if absent)."""
    returned = list(record.get("retrieved_tables") or [])[:5]
    ranks = {}
    for t in gold_tables:
        simple = t.split(".")[-1].lower()
        ranks[t] = (returned.index(simple) + 1
                    if simple in returned else None)
    return ranks


def analyze(run_dir: str, gold_path: str, output_dir: Path,
            method_a: str = "TERAG-Full", method_b: str = "MetadataMatched",
            method_c: Optional[str] = None,
            valid_qids=None) -> Path:
    gold = load_gold(gold_path)
    pool = load_run(run_dir)

    def predictions(variant):
        key = ("table_v2", variant)
        if key not in pool:
            raise KeyError(f"run lacks table_v2 predictions for {variant}")
        return pool[key]

    pa, pb = predictions(method_a), predictions(method_b)
    pc = predictions(method_c) if (method_c and
                                   ("table_v2", method_c) in pool) else None

    qids = sorted(valid_qids) if valid_qids is not None else sorted(pa)
    rows = []
    counts = Counter()
    for qid in qids:
        g = gold[qid]
        a_cov, b_cov = _covered(pa[qid], gold), _covered(pb[qid], gold)
        category = ("both_success" if a_cov and b_cov
                    else f"{method_a}_only" if a_cov
                    else f"{method_b}_only" if b_cov
                    else "both_fail")
        counts[category] += 1
        row = {
            "qid": qid, "original_table": g.get("original_table", ""),
            "category": category,
            f"{method_a}_covered": a_cov, f"{method_b}_covered": b_cov,
            f"{method_a}_gold_ranks": json.dumps(
                gold_table_ranks(pa[qid], g["gold_tables"])),
            f"{method_b}_gold_ranks": json.dumps(
                gold_table_ranks(pb[qid], g["gold_tables"])),
        }
        if pc is not None:
            c_cov = _covered(pc[qid], gold)
            row[f"{method_c}_covered"] = c_cov
            row[f"{method_c}_gold_ranks"] = json.dumps(
                gold_table_ranks(pc[qid], g["gold_tables"]))
            row[f"{method_c}_recovered"] = (not a_cov) and c_cov
            row[f"{method_c}_lost"] = a_cov and (not c_cov)
        rows.append(row)

    # Deterministic case selection: lowest qid per category, preferring
    # distinct original target tables.
    cases, used_tables = [], set()
    for category in (f"{method_b}_only", "both_fail", f"{method_a}_only",
                     "both_success"):
        candidates = [r for r in rows if r["category"] == category]
        pick = next((r for r in candidates
                     if r["original_table"] not in used_tables), None)
        if pick is None and candidates:
            pick = candidates[0]
        if pick is not None:
            cases.append({"qid": pick["qid"], "category": category,
                          "selection_rule": "lowest qid in category, distinct "
                                            "target table preferred",
                          "original_table": pick["original_table"]})
            used_tables.add(pick["original_table"])

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "heldout_disagreements.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n": len(rows, ), "categories": dict(counts),
        "methods": [m for m in (method_a, method_b, method_c) if m],
        "cases": cases,
        "note": ("score components of the original runs were not saved; "
                 "rank/category evidence only, no per-component attribution"),
    }
    with open(output_dir / "heldout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return output_dir / "heldout_disagreements.csv"
