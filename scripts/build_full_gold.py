"""
Full schema gold construction.

Emit one derived record per question: a stable qid, source checksum, the raw
SQL, dialect/parser versions, the original annotations, the AST physical
table set, the normalized table.column set (with each column's SQL roles),
the basis for star expansion and derived columns, unresolved/ambiguous/
schema-mismatch items, and the parse and review status.

Rules:
- Use only terag/sql_role_parser.py (sqlglot AST) and the real schema CSVs;
  never fill gold from predictions.
- "AST parsed successfully" and "gold verified complete" are marked
  separately; unlocatable/ambiguous items are recorded as-is instead of
  being forced into a completeness claim (candidate fields are not blindly
  dumped into gold, and the SQL/schema is not modified).
- The original SELECT annotation is kept as a separate accounting
  (original_fields) and is neither overwritten nor deleted.

Usage:
  python scripts/build_full_gold.py --field-csv F.csv --table-csv T.csv \
      --qa-csv Q.csv --output-dir derived/ [--qid-prefix q] [--dialect mysql]

SQL is parsed, never executed.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import sqlglot

from terag.sql_role_parser import SQLRoleParser, SQLGLOT_AVAILABLE


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build(field_csv, table_csv, qa_csv, output_dir, qid_prefix='pg',
          dialect='mysql'):
    field_csv = str(field_csv)
    table_csv = str(table_csv)
    qa_csv = str(qa_csv)

    field_df = pd.read_csv(field_csv)
    table_df = pd.read_csv(table_csv)
    qa_df = pd.read_csv(qa_csv)

    schema_columns = {t: set(g['field_name'].astype(str))
                      for t, g in field_df.groupby('table')}
    schema_tables = set(table_df['table'].astype(str))
    table_missing_in_schema = sorted(
        {t for t in schema_columns} ^ schema_tables)  # should be empty

    parser = SQLRoleParser(dialect=dialect)

    records = []
    status_counts = {'verified_complete': 0, 'needs_review': 0, 'parse_failed': 0}
    for i, row in qa_df.iterrows():
        qid = f"{qid_prefix}{i:03d}"
        sql = str(row['sql'])
        pr = parser.parse(sql, schema_columns=schema_columns)

        gold_tables = sorted(pr.get_gold_tables())
        gold_cols = pr.get_gold_columns()          # "table.column" (lowercased table names)
        col_roles = {}
        for t, fields in pr.field_roles.items():
            for f, info in fields.items():
                col_roles[f"{t}.{f}".lower()] = sorted(info.roles)

        # original annotations (kept as a separate accounting)
        orig_table = str(row['table']).split('.')[-1]
        orig_fields = ([f.strip() for f in str(row['field']).split('|')
                        if f.strip()]
                       if pd.notna(row['field']) and str(row['field']).strip() else [])

        # status decision (AST success != gold completeness)
        review_reasons = []
        if not pr.ast_parsed:
            status = 'parse_failed'
            review_reasons.append(f"AST parse failed: {pr.error}")
        else:
            missing_tables = [t for t in gold_tables if t not in schema_columns]
            if missing_tables:
                review_reasons.append(f"SQL references tables not present in schema: {missing_tables}")
            if pr.unresolved:
                kinds = sorted({u['kind'] for u in pr.unresolved})
                review_reasons.append(f"unresolved items: {len(pr.unresolved)} ({kinds})")
            if pr.ambiguous:
                review_reasons.append(f"ambiguous columns: {len(pr.ambiguous)} found")
            if not gold_cols:
                review_reasons.append("no physical columns extracted")
            status = 'verified_complete' if not review_reasons else 'needs_review'

        # Diff against the original annotations (original SELECT fields vs
        # derived SELECT-role columns vs all-role columns)
        derived_select_cols = {c for c, r in col_roles.items() if 'SELECT' in r}
        orig_keys = {f"{orig_table.lower()}.{f.lower()}" for f in orig_fields}
        diff = {
            'orig_not_in_derived_select': sorted(
                {k.split('.', 1)[1] for k in orig_keys - derived_select_cols}),
            'derived_select_extra': sorted(
                {c for c in derived_select_cols - orig_keys}),
            'n_derived_all_role': len(gold_cols),
        }

        records.append({
            'qid': qid,
            'source_qa_sha256': _sha256(qa_csv),
            'sql': sql,
            'dialect': pr.parse_method,
            'parser': f"terag.sql_role_parser.SQLRoleParser(sqlglot {sqlglot.__version__})",
            'original_table': orig_table,
            'original_fields': orig_fields,
            'gold_tables': gold_tables,
            'gold_columns': sorted(gold_cols),
            # clean_projection accounting: physical table columns resolved in
            # the outermost SELECT output (star expansion included; columns
            # selected inside subqueries excluded)
            'projection_columns': sorted(f"{t}.{c}" for t, c in pr.projection_fields),
            'column_roles': col_roles,
            'star_handling': pr.star_expansions,
            'unresolved_items': pr.unresolved,
            'ambiguous_items': pr.ambiguous,
            'tables_missing_in_schema': [t for t in gold_tables
                                         if t not in schema_columns],
            'ast_parsed': pr.ast_parsed,
            'gold_status': status,
            'review_reasons': review_reasons,
            'annotation_diff': diff,
            'built_at': datetime.now().isoformat(timespec='seconds'),
        })
        status_counts[status] += 1

    os.makedirs(output_dir, exist_ok=True)
    gold_path = os.path.join(output_dir, 'full_gold_v2.jsonl')
    with open(gold_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    manifest = {
        'built_at': records[0]['built_at'],
        'inputs': {p: _sha256(p) for p in [field_csv, table_csv, qa_csv]},
        'parser': f"terag.sql_role_parser.SQLRoleParser (sqlglot {sqlglot.__version__})",
        'sqlglot_available': SQLGLOT_AVAILABLE,
        'dialect_chain': (f"{dialect} → hive → {dialect} "
                          f"(regex fallback does not count as success)"),
        'output': {'file': gold_path, 'sha256': _sha256(gold_path)},
        'status_counts': status_counts,
        'n_records': len(records),
        'schema_tables_missing_from_field_csv': table_missing_in_schema,
        'rules': ('gold_columns = all locatable physical table columns the SQL needs '
                  '(union of the per-clause roles); unresolved/ambiguous/out-of-schema '
                  'references are not written into gold, only recorded; original_fields '
                  'keeps the original SELECT annotation as a separate accounting'),
    }
    manifest_path = os.path.join(output_dir, 'full_gold_v1_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: manifest[k] for k in
                      ('status_counts', 'n_records', 'output')},
                     ensure_ascii=False, indent=2))
    print('Needs review:', [r['qid'] for r in records if r['gold_status'] != 'verified_complete'])
    return records


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--field-csv', required=True,
                    help='field schema CSV (table, field_name, ...)')
    ap.add_argument('--table-csv', required=True,
                    help='table schema CSV (table, table_desc)')
    ap.add_argument('--qa-csv', required=True,
                    help='QA CSV (question, sql, table, field)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--qid-prefix', default='pg')
    ap.add_argument('--dialect', default='mysql')
    args = ap.parse_args()
    build(args.field_csv, args.table_csv, args.qa_csv, args.output_dir,
          qid_prefix=args.qid_prefix, dialect=args.dialect)
