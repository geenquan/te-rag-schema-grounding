# Data Format Contract

All inputs are plain CSV/JSONL with UTF-8 encoding. `terag_release
validate-data` checks everything below mechanically.

## Field schema CSV — `field_csv`

| Column | Required | Meaning |
|---|---|---|
| `table` | ✓ | simple table identifier (no dots needed; last dot-segment is used if present) |
| `field_name` | ✓ | physical column name |
| `field_name_desc` | recommended | column comment/business description (drives the enriched index and similarity) |
| `field_type` | recommended | type string (`varchar`/`int`/`decimal`/`date`/...) — feeds transfer type-compatibility gates |

Duplicate `(table, field_name)` rows are tolerated (real exports contain
them): the transfer caps accumulate across duplicates per *normalized
physical* column.

## Table schema CSV — `table_csv`

| Column | Required | Meaning |
|---|---|---|
| `table` | ✓ | table identifier |
| `table_desc` | recommended | table comment |

## QA CSV — `qa_csv`

Row order defines the qids: `<qid-prefix><row-index:03d>` (default prefix
`q` → `q000, q001, ...`). Keep the file and order stable across steps.

| Column | Required | Meaning |
|---|---|---|
| `question` | ✓ | natural-language question |
| `sql` | ✓ | reference SQL (training supervision / gold / isolation; never executed) |
| `table` | ✓ | originally annotated target table (simple name or `db.table`; last segment used) — drives held-out-table fold grouping |
| `field` | recommended | original SELECT annotation, `\|`-separated bare column names (kept as a *separate* caliber from the derived full gold) |

## Full gold — `full_gold_v2.jsonl`

One JSON record per QA row, produced by `scripts/build_full_gold.py`
(sqlglot AST, MySQL→Hive dialect chain; **parsing, not execution**):

```json
{
  "qid": "q000",
  "sql": "SELECT ...",
  "dialect": "mysql",
  "parser": "terag.sql_role_parser.SQLRoleParser(sqlglot 30.3.0)",
  "original_table": "wms_in_lines",
  "original_fields": ["qty"],
  "gold_tables": ["wms_in_lines", "wms_in_orders"],
  "gold_columns": ["wms_in_lines.qty", "wms_in_orders.warehouse_code"],
  "projection_columns": ["wms_in_lines.qty"],
  "column_roles": {"wms_in_lines.qty": ["SELECT", "GROUP_BY"]},
  "star_handling": {}, "unresolved_items": [], "ambiguous_items": [],
  "tables_missing_in_schema": [],
  "ast_parsed": true,
  "gold_status": "verified_complete",
  "review_reasons": [],
  "annotation_diff": {"orig_not_in_derived_select": [], "derived_select_extra": [], "n_derived_all_role": 2}
}
```

Rules: `gold_columns` = every physical column the SQL needs (all clause
roles); unresolved/ambiguous/out-of-schema references are **recorded, not
written into gold**; `gold_status` ∈ `verified_complete` / `needs_review` /
`parse_failed`. Only `verified_complete` questions form the valid evaluation
denominator. Budget feasibility is derived from gold only (≤5 distinct gold
tables and ≤ budget gold columns per table; recomputed, never hardcoded).

## Splits manifest — `splits_manifest.json`

Written by `terag_release make-folds` (and by every run into its `run_meta`);
loadable via `data.splits_manifest` in a dataset config.

```json
{
  "n_rows": 49,
  "dup_groups": [[0], [1, 2]],
  "question_folds": {"0": [0, 5], "1": [1, 6]},
  "table_folds":     {"0": {"test_tables": ["t_a"], "test": [3], "train": [0, 1]}},
  "ast_refined_table_folds": {"0": {"targets": ["t_a"], "train": ["q000"], "test": ["q003"]}},
  "ast_exclusions": [{"fold": 0, "excluded_qids": ["q007"],
                      "exclusion_reasons": [{"qid": "q007", "reason": "references_target_table"}]}],
  "checks": {"question_coverage_complete": true, "table_train_refs_isolated": true}
}
```

Duplicate-question groups (exact normalized match, or masked-token Jaccard
≥ 0.8) never cross question folds. Table folds: each gold target table is
tested in exactly one fold; training questions whose AST gold tables
intersect the fold's targets are excluded, as are questions with unparseable
SQL (conservative). Target-table isolation does not imply auxiliary-table
isolation — auxiliary tables may legitimately remain visible.

## Training supervision (internal)

History-consuming methods (ExampleRRF, CooccurRRF, TE-RAG variants) receive
training frames with `gold_tables` (JSON list) and `gold_fields` (JSON
`[table, field]` pairs) injected from the **full gold of the training fold
only** (`full_sql` arm). Test frames are never injected.

## Predictions — `predictions.jsonl`

One record per (question, method, protocol, fold):

```json
{
  "qid": "q003", "setting": "question_v1", "fold": 0,
  "variant": "TERAG-Full", "method": "TE-RAG-V2", "status": "ok",
  "question": "...",
  "retrieved_tables": ["wms_stock", "base_warehouse"],
  "retrieved_column_keys": ["wms_stock.sku_id", "wms_stock.stock_qty"],
  "retrieved_columns_detail": [
    {"table": "wms_stock", "table_score": 1.42,
     "columns": [{"key": "wms_stock.sku_id", "score": 0.9}]}],
  "method_evidence": {...}, "query_time": 0.012, "memory_mb": 1.2,
  "exec_accuracy": "not_run"
}
```

`status == "error"` records carry `error_type`/`error_message`, keep empty
outputs, and **stay in every denominator**. Budget contract: ≤5 tables, ≤10
columns per table (verified by `evaluate`). Later prediction files supplied
to `evaluate` replace earlier ones per complete (setting, variant) group —
missing qids are never patched question-by-question from another source.

## Budget feasibility — `budget_feasibility.json`

Written by runs into `run_meta`; consumed by `evaluate` (which independently
recomputes and cross-validates it against the gold):

```json
{"rule": "...", "max_tables": 5, "max_columns_per_table": 10,
 "n_verified": 44, "n_feasible": 44, "infeasible_qids": [],
 "per_question": [{"qid": "q000", "n_gold_tables": 1, "max_cols_per_table": 2, "budget_feasible": true}]}
```

---

## The BIRD data tree (`data/bird/`)

Public BIRD inputs (CC BY-SA 4.0; provenance, hashes, and the derivation
chain in `data/bird/SOURCES.md`; license text in
`data/bird/LICENSE-BIRD-DATA.txt`):

```
data/bird/
├── raw/train/{train.json,train_tables.json}   # official Train split: 9428 questions / 69 databases
├── raw/dev/{dev.json,dev_tables.json}         # official Dev split (2024-06-27): 1534 questions / 11 databases
├── processed/                                  # frozen Dev five-fold evaluation inputs
│   ├── gold/gold.json                          # derived gold, 1534 entries (1530 verified; db-local views)
│   ├── q5_manifest.json / h5_manifest.json     # question / heldout five-fold manifests (55 / 50 applicable cells,
│   │                                           #   per-cell train/test qids, heldout tables, unverified records)
│   ├── schema/<db>/{field_schema.csv,table_schema.csv}   # 11 Dev databases
│   └── train/{question,heldout}_f<fold>_<db>.jsonl       # per-cell supervised training rows
├── train_dev/                                  # frozen Train development inputs
│   ├── opt_gold.json / opt_manifest.json       # 9428 questions, 9068 verified, duplicate-group 80/20 split
│   ├── schema/<db>/...                         # 69 Train databases
│   ├── train_rows/<db>/{opt_train,opt_dev}.jsonl
│   └── h_splits/h_manifest.json (+train_rows)  # Train-H development split: 55/69 databases, 4387 dev questions
```

* Dev main evaluation population = the **1530 verified** questions; the 4
  unverified questions are kept in `gold.json` and excluded by the
  manifests (recorded per database; never deleted silently).
* Train population = **9068 verified** of 9428 raw; the 360 unverified are
  excluded from every development split by `opt_manifest.json`.
* Train row records carry `qid, question, sql, db_id, gold_tables,
  gold_fields` (full-SQL supervision; the same injection semantics as the
  grid protocol).
* Manifests are **re-validated** at run time (`terag_release.bird_data`)
  and never rebuilt; a mismatch with the shipped data is a hard error.
* SQLite database archives are not shipped and not required (SQL is parsed,
  never executed).
