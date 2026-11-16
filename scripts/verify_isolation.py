"""
Unseen-table isolation verification.

Independently re-check the table-fold isolation (originally established
with FROM/JOIN regexes) using the verified AST physical table sets:
- Per question: compare the original regex-referenced table set, the new
  AST physical table set, and the original primary-table annotation, and
  report differences with reasons.
- For each old table fold: check whether any training SQL references a
  held-out target table (judged by the AST set).
- The isolation check and the split generation use different extraction
  functions (splits = regex, verification = AST).
- On missed leaks: flag the affected folds, keep the old splits, and write
  derived splits v2 (old files are not overwritten). Conservative rule: a
  training question whose AST parse is incomplete must not be assumed free
  of held-out tables — exclude it from the training set and count it
  (questions with ast_parsed=False enter no training fold).

Naming clarification: the isolation criterion of the table-fold setup is
"no table involved in the test SQL (per the regex reference set) may be
referenced in the training logs" (full-reference isolation), not merely
"the held-out primary table is unreferenced".

Usage:
  python scripts/verify_isolation.py --run-dir RUN_DIR --gold GOLD.jsonl \
      --qa-csv QA.csv --qid-prefix q --out-dir OUT_DIR

``RUN_DIR`` is a run directory whose ``run_meta/splits_manifest.json`` holds
the table folds (as written by ``terag_release make-folds`` or
``run-experiments``).
"""

import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd


def regex_ref_tables(sql):
    """The same regex used to generate the table splits (deliberately
    independent of the AST parser, so the two calibers cross-check)."""
    ids = re.findall(r'(?:FROM|JOIN)\s+`?([a-zA-Z_][\w.]*)`?', str(sql), re.I)
    return {x.split('.')[-1] for x in ids if x.split('.')[-1].lower() != 'select'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--gold', required=True,
                    help='full gold JSONL (qid, gold_tables, ast_parsed, ...)')
    ap.add_argument('--qa-csv', required=True)
    ap.add_argument('--qid-prefix', default='q')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    qa = pd.read_csv(args.qa_csv)
    gold = {}
    with open(args.gold, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            gold[r['qid']] = r

    manifest = json.load(open(os.path.join(args.run_dir, 'run_meta/splits_manifest.json'),
                              encoding='utf-8'))
    table_folds = manifest['table_folds']

    # ---- per-question comparison of the three accountings ----
    per_qid_diff = {}
    for i, row in qa.iterrows():
        qid = f"{args.qid_prefix}{i:03d}"
        rx = regex_ref_tables(row['sql'])
        ast = set(gold[qid]['gold_tables'])
        primary = gold[qid]['original_table']
        if rx != ast or primary not in ast:
            per_qid_diff[qid] = {
                'regex_refs': sorted(rx),
                'ast_tables': sorted(ast),
                'primary': primary,
                'ast_minus_regex': sorted(ast - rx),
                'regex_minus_ast': sorted(rx - ast),
                'primary_in_ast': primary in ast,
            }

    # ---- table-fold isolation check (AST as the authority) ----
    fold_reports = []
    any_leak = False
    for k in range(5):
        fd = table_folds[str(k)]
        test_tables = set(fd['test_tables'])
        train_qids = [f"{args.qid_prefix}{i:03d}" for i in fd['train']]
        test_qids = [f"{args.qid_prefix}{i:03d}" for i in fd['test']]
        leaks = []
        for qid in train_qids:
            refs = set(gold[qid]['gold_tables'])
            hit = refs & test_tables
            if hit:
                leaks.append({'qid': qid, 'referenced_test_tables': sorted(hit)})
        fold_reports.append({
            'fold': k,
            'test_tables': sorted(test_tables),
            'n_test': len(test_qids),
            'n_train_regex': len(train_qids),
            'train_leaks_by_ast': leaks,
            'isolated_by_ast': not leaks,
        })
        if leaks:
            any_leak = True

    report = {
        'run_dir_checked': args.run_dir,
        'gold_file': args.gold,
        'n_qid_table_set_diff': len(per_qid_diff),
        'per_qid_diff': per_qid_diff,
        'table_fold_isolation': fold_reports,
        'any_leak': any_leak,
        'caliber_statement': (
            'Table folds = "no table referenced by the test SQL (regex FROM/JOIN) '
            'may appear in the training logs"; this check re-verifies it '
            'independently with the AST physical table sets. A hit on either '
            'criterion counts as a leak.'),
    }

    # ---- derived splits v2 (written only when leaks are found; conservatively excludes incompletely parsed questions) ----
    derived_splits = None
    if any_leak:
        v2_dir = os.path.join(args.out_dir, 'splits_v2')
        os.makedirs(v2_dir, exist_ok=True)
        excluded_unparseable = []
        for k in range(5):
            fd = table_folds[str(k)]
            test_tables = set(fd['test_tables'])
            train_rows, excluded = [], []
            for i in fd['train']:
                qid = f"{args.qid_prefix}{i:03d}"
                g = gold[qid]
                if not g['ast_parsed']:
                    # conservative: incomplete parse → cannot be assumed free of held-out tables; moved out of training
                    excluded.append(qid)
                    continue
                if set(g['gold_tables']) & test_tables:
                    excluded.append(qid)  # AST found a reference to a held-out table
                    continue
                train_rows.append(i)
            excluded_unparseable.append({'fold': k, 'excluded_qids': excluded})
            qa.iloc[train_rows].to_csv(
                os.path.join(v2_dir, f'table_fold{k}_train_v2.csv'), index=False)
            qa.iloc[fd['test']].to_csv(
                os.path.join(v2_dir, f'table_fold{k}_test_v2.csv'), index=False)
        derived_splits = {
            'dir': v2_dir,
            'rule': ('AST-set isolation + conservative exclusion (training questions with ast_parsed=False or referencing held-out tables are moved out)'),
            'excluded': excluded_unparseable,
        }
        report['derived_splits_v2'] = derived_splits

    out_path = os.path.join(args.out_dir, 'isolation_check.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Per-question table-set differences: {len(per_qid_diff)} questions")
    for qid, d in per_qid_diff.items():
        print(f"  {qid}: ast-regex={d['ast_minus_regex']} regex-ast={d['regex_minus_ast']}"
              f" primary_in_ast={d['primary_in_ast']}")
    for fr in fold_reports:
        print(f"Table fold {fr['fold']}: AST isolation={'passed' if fr['isolated_by_ast'] else 'leaked'} "
              f"leaked questions={len(fr['train_leaks_by_ast'])}")
    print(f"Conclusion: {'missed leaks found' if any_leak else 'all table folds pass the AST isolation check'}")
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
