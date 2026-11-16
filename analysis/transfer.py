"""Transfer (C) evidence summary from diagnostic exports.

Consumes the per-fold artifacts written by ``terag_release run-experiments
--diagnostics`` (DiagnosticExport): ``annotations.json`` (direct B +
accepted/rejected transfer) and ``index.json`` (merged index inputs, column
terms). Verifies, on this NEW computation:

* every accepted record's source_qids are inside that fold's training set,
* per-normalized-target acceptance <= 3 and per-table distinct C texts <= 8
  (the physical caps of `transfer_v3_target_cap`),
* accepted TERM texts appear in the final column-terms map and accepted
  texts appear among the table's merged index inputs.

This is new-computation evidence about the CURRENT run; it is not a
reconstruction of the paper's original per-record traces (those were never
saved — see docs/experiment_catalog.md §7).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.retrieval_metrics import normalize_column_key


def _iter_fold_dirs(run_dir: Path):
    diag_root = run_dir / "diagnostics"
    if not diag_root.exists():
        return
    for setting_dir in sorted(diag_root.iterdir()):
        if not setting_dir.is_dir():
            continue
        for fold_dir in sorted(setting_dir.iterdir()):
            if fold_dir.is_dir():
                for variant_dir in sorted(fold_dir.iterdir()):
                    if (variant_dir / "annotations.json").exists():
                        yield variant_dir


def analyze(run_dir: str, output_dir: Path) -> Path:
    run_dir = Path(run_dir)
    accepted_rows, violations = [], []
    totals = Counter()

    for variant_dir in _iter_fold_dirs(run_dir):
        setting = variant_dir.parent.parent.name
        fold = variant_dir.parent.name
        variant = variant_dir.name
        # Caps apply PER FOLD/CONFIG — never accumulate across them.
        per_target, per_table_texts = Counter(), defaultdict(set)
        ann = json.loads((variant_dir / "annotations.json")
                         .read_text(encoding="utf-8"))
        idx = json.loads((variant_dir / "index.json")
                         .read_text(encoding="utf-8"))
        fit = json.loads((variant_dir / "fit.json").read_text(encoding="utf-8"))
        train_qids = set(fit["training_qids"])

        merged = {}
        for table, info in (idx.get("injected_tables") or {}).items():
            merged[table] = {str(t) for t, _w in
                             (info.get("merged_candidates") or [])}
        column_terms = idx.get("column_terms") or {}

        provenance = ann.get("transfer", {}).get("provenance", {})
        totals[f"{setting}/{variant}/folds"] += 1
        totals[f"{setting}/{variant}/n_accepted"] += len(
            ann.get("transfer", {}).get("accepted", []))
        for stage, count in (provenance.get("decision_counts") or {}).items():
            totals[f"decisions/{stage}"] += count

        for record in ann.get("transfer", {}).get("accepted", []):
            target = record.get("target_element") or ""
            table = normalize_column_key(target).rsplit(".", 1)[0] \
                or target.split(".")[0]
            key = normalize_column_key(target)
            per_target[key] += 1
            if record.get("adapted_content"):
                per_table_texts[table].add(str(record["adapted_content"]))
            row = {
                "setting": setting, "fold": fold, "variant": variant,
                "annotation_type": record.get("source_type"),
                "method_version": record.get("method_version"),
                "source_element": record.get("source_element"),
                "target_element": target,
                "source_qids": json.dumps(record.get("source_qids", [])),
                "adapted_content": record.get("adapted_content"),
                "similarity": record.get("similarity"),
                "in_column_terms": str(record.get("adapted_content")) in
                                   column_terms.get(key, []),
                "in_merged_index_input": str(record.get("adapted_content")) in
                                         merged.get(table, set()),
            }
            accepted_rows.append(row)
            outside = set(record.get("source_qids", [])) - train_qids
            if outside:
                violations.append({"setting": setting, "fold": fold,
                                   "target": target,
                                   "kind": "source_qid_outside_train",
                                   "detail": sorted(outside)})

        for target, count in per_target.items():
            if count > 3:
                violations.append({"setting": setting, "fold": fold,
                                   "target": target,
                                   "kind": "max_annotations_per_target>3",
                                   "count": count})
        for table, texts in per_table_texts.items():
            if len(texts) > 8:
                violations.append({"setting": setting, "fold": fold,
                                   "table": table,
                                   "kind": "max_texts_per_table>8",
                                   "count": len(texts)})

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "transfer_records.jsonl", "w",
              encoding="utf-8") as f:
        for row in accepted_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "identity": "new diagnostic computation; not recovered historical "
                    "records",
        "n_accepted": len(accepted_rows),
        "by_type": dict(Counter(r["annotation_type"]
                                for r in accepted_rows)),
        "totals": dict(totals),
        "violations": violations,
        "caps": {"max_annotations_per_target": 3,
                 "max_texts_per_table": 8},
    }
    with open(output_dir / "transfer_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return output_dir / "transfer_summary.json"
