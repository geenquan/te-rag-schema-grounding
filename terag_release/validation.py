"""Input validation for the documented data contract (docs/data_format.md).

Checks structural validity only — no retrievers are imported, nothing is
fitted. Gold and split checks are performed when the corresponding paths are
supplied.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

GOLD_STATUSES = {"verified_complete", "needs_review", "parse_failed"}


def _check_columns(df: pd.DataFrame, required: List[str], kind: str,
                   errors: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"{kind}: missing required columns {missing}")


def _check_gold_record(rec: Dict, qid: str, errors: List[str]):
    for key in ("qid", "gold_status", "gold_tables", "gold_columns",
                "original_table", "sql", "ast_parsed"):
        if key not in rec:
            errors.append(f"gold[{qid}]: missing key '{key}'")
            return
    if rec["gold_status"] not in GOLD_STATUSES:
        errors.append(f"gold[{qid}]: unknown gold_status "
                      f"{rec['gold_status']!r}")
    if rec["gold_status"] == "verified_complete":
        if not rec["gold_tables"]:
            errors.append(f"gold[{qid}]: verified_complete but no gold_tables")
        if not rec["gold_columns"]:
            errors.append(f"gold[{qid}]: verified_complete but no gold_columns")
        for c in rec["gold_columns"]:
            if "." not in str(c):
                errors.append(f"gold[{qid}]: column key lacks '.' : {c!r}")


def validate_dataset(field_csv: str, table_csv: str, qa_csv: str,
                     gold: Optional[str] = None,
                     splits_manifest: Optional[str] = None) -> Dict:
    errors: List[str] = []
    warnings: List[str] = []
    summary_parts: List[str] = []

    field_df = pd.read_csv(field_csv)
    table_df = pd.read_csv(table_csv)
    qa_df = pd.read_csv(qa_csv)

    _check_columns(field_df, ["table", "field_name"], "field CSV", errors)
    _check_columns(table_df, ["table"], "table CSV", errors)
    _check_columns(qa_df, ["question", "sql", "table"], "QA CSV", errors)
    if "field" not in qa_df.columns:
        warnings.append("QA CSV: no 'field' column (original SELECT "
                        "annotation) — held-out grouping still works, the "
                        "legacy annotated-field metric will be empty")

    tables_in_fields = set(field_df["table"].astype(str))
    tables_in_tables = set(table_df["table"].astype(str))
    if tables_in_fields != tables_in_tables:
        warnings.append(
            f"table sets differ between field/table CSVs: "
            f"{sorted(tables_in_fields ^ tables_in_tables)[:10]}")
    dup_field_rows = int(field_df.duplicated(
        subset=["table", "field_name"]).sum())
    if dup_field_rows:
        warnings.append(
            f"field CSV has {dup_field_rows} duplicated (table, field_name) "
            f"rows — the algorithm tolerates them (physical-target caps are "
            f"cumulative across duplicates) but they are worth reviewing")
    summary_parts.append(f"{len(table_df)} tables, {len(field_df)} field rows, "
                         f"{len(qa_df)} questions")

    qids = [f"q{i:03d}" for i in range(len(qa_df))]
    gold_records = {}
    if gold:
        path = Path(gold)
        if not path.exists():
            errors.append(f"gold file not found: {gold}")
        else:
            seen = set()
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(f"gold line {line_no}: invalid JSON ({e})")
                        continue
                    qid = rec.get("qid")
                    if qid in seen:
                        errors.append(f"gold line {line_no}: duplicate qid {qid}")
                    seen.add(qid)
                    _check_gold_record(rec, qid, errors)
                    gold_records[qid] = rec
            expected = set(qids)
            if gold_records and set(gold_records) != expected:
                errors.append(
                    f"gold qids do not match the QA row count "
                    f"({len(gold_records)} records vs {len(qa_df)} rows); "
                    f"rebuild gold with the same qid prefix and row order")
            n_valid = sum(1 for r in gold_records.values()
                          if r.get("gold_status") == "verified_complete")
            summary_parts.append(f"gold: {len(gold_records)} records, "
                                 f"{n_valid} verified_complete")

    if splits_manifest:
        path = Path(splits_manifest)
        if not path.exists():
            errors.append(f"splits manifest not found: {splits_manifest}")
        else:
            manifest = json.load(open(path, encoding="utf-8"))
            if manifest.get("n_rows") != len(qa_df):
                errors.append("splits manifest n_rows does not match the QA "
                              "row count; rebuild folds for this dataset")
            failed_checks = [k for k, v in
                             (manifest.get("checks") or {}).items() if not v]
            if failed_checks:
                errors.append(f"splits manifest failed checks: {failed_checks}")
            summary_parts.append("folds: manifest verified")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "summary": "; ".join(summary_parts)}
