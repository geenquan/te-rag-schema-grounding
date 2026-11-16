"""Column saved-prefix analysis (paper Table `tab:column_budget`).

Truncates SAVED outputs — never re-retrieves. For each prediction record the
top-5 tables are fixed and the per-table column list is cut to the first
``budget`` RAW slots: original order kept, duplicate slots retained, metrics
computed over normalized sets. The gold-feasible population (<= budget gold
columns on every gold table and <= 5 gold tables) is recomputed from gold.

Ports the review-stage analysis of the private runs with parameterized I/O;
no question ids, method names, or denominators are hardcoded.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

from eval.retrieval_metrics import normalize_column_key, normalize_table
from eval.full_gold import full_gold_metrics, index_predictions

BUDGETS = (5, 10)


def load_run(run_dir: str):
    """Load predictions.jsonl from a runner output directory."""
    path = Path(run_dir) / "predictions.jsonl"
    records = [json.loads(line) for line in
               open(path, encoding="utf-8") if line.strip()]
    groups: Dict[tuple, Dict[str, Dict]] = {}
    for rec in records:
        groups.setdefault((rec["setting"], rec["variant"]), {})[rec["qid"]] = rec
    return {k: index_predictions(list(v.values())) for k, v in groups.items()}


def load_gold(gold_path: str) -> Dict[str, Dict]:
    return {r["qid"]: r for r in
            (json.loads(line) for line in
             open(gold_path, encoding="utf-8") if line.strip())}


def eligible(gold_record: Dict, budget: int, max_tables: int = 5) -> bool:
    """Gold-only budget feasibility for a given per-table column budget."""
    tables = {normalize_table(t) for t in gold_record["gold_tables"]}
    per_table: Dict[str, int] = {}
    for c in gold_record["gold_columns"]:
        key = normalize_column_key(c)
        t = key.rsplit(".", 1)[0]
        per_table[t] = per_table.get(t, 0) + 1
    return (len(tables) <= max_tables
            and max(per_table.values(), default=0) <= budget)


def crop(record: Dict, budget: int) -> Dict:
    """Copy of ``record`` truncated to top-5 tables / budget column slots.

    ``status == 'error'`` records keep empty outputs. retrieved_column_keys is
    rebuilt from the truncated detail so downstream metric code sees exactly
    the cropped view.
    """
    if record.get("status") == "error":
        return dict(record)
    out = dict(record)
    detail = []
    keys: List[str] = []
    for table_entry in (record.get("retrieved_columns_detail") or [])[:5]:
        cols = table_entry.get("columns", [])[:budget]
        detail.append({**table_entry, "columns": cols})
        for col in cols:
            key = normalize_column_key(col.get("key", ""))
            if key and key not in keys:
                keys.append(key)
    out["retrieved_tables"] = (record.get("retrieved_tables") or [])[:5]
    out["retrieved_columns_detail"] = detail
    out["retrieved_column_keys"] = keys
    return out


def analyze(run_dir: str, gold_path: str, output_dir: Path,
            methods: List[str], valid_qids, feasible_by_budget) -> Path:
    """Write per-question error rows + per-method/budget summary CSV.

    ``feasible_by_budget`` maps budget -> set(qid) (recompute with
    :func:`eligible` — never hardcode paper denominators).
    """
    gold = load_gold(gold_path)
    pool = load_run(run_dir)
    rows: List[Dict] = []
    summary: List[Dict] = []

    for (setting, variant), predictions in sorted(pool.items()):
        if variant not in methods or setting != "question_v1":
            continue
        for budget in BUDGETS:
            per_q = []
            for qid in sorted(valid_qids):
                record = crop(predictions[qid], budget)
                tr, cov, p, r, f1, fgcs = full_gold_metrics(record, gold)
                g = gold[qid]
                per_q.append({
                    "setting": setting, "variant": variant, "qid": qid,
                    "budget_per_table": budget,
                    "gold_columns_n": len(g["gold_columns"]),
                    "predicted_columns_raw_n": sum(
                        len(t["columns"]) for t in
                        record.get("retrieved_columns_detail") or []),
                    "predicted_columns_unique_n": len(
                        set(record.get("retrieved_column_keys") or [])),
                    "colp": p, "colrec": r, "colf1": f1,
                    "covered": bool(cov), "fgcs": bool(fgcs),
                    "gold_budget_feasible": qid in feasible_by_budget[budget],
                })
            rows.extend(per_q)
            n = len(per_q)
            feasible_rows = [x for x in per_q if x["gold_budget_feasible"]]
            summary.append({
                "setting": setting, "variant": variant,
                "budget_per_table": budget, "n": n,
                "covered@5": sum(x["covered"] for x in per_q) / n,
                "colp": sum(x["colp"] or 0 for x in per_q) / n,
                "colrec": sum(x["colrec"] or 0 for x in per_q) / n,
                "colf1": sum(x["colf1"] or 0 for x in per_q) / n,
                "fgcs": sum(x["fgcs"] for x in per_q) / n,
                "fgcs_complete_questions": sum(x["fgcs"] for x in per_q),
                "n_gold_feasible": len(feasible_rows),
                "fgcs_feasible": (sum(x["fgcs"] for x in feasible_rows)
                                  / len(feasible_rows)) if feasible_rows else None,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "column_budget.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with open(output_dir / "column_budget_per_question.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_dir / "column_budget.csv"
