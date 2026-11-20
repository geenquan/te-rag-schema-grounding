"""Offline recomputation check for the shipped BIRD reference results.

Reads the sanitized reference predictions (results/reference/bird/), whose
only inputs are the frozen gold under data/bird/processed/, recomputes the
nine metrics per question with the frozen metric implementation, and
compares them to the shipped per-question CSVs and aggregate summaries. Any
mismatch fails. No model, no API, no network.
"""
import csv
import gzip
import json
from pathlib import Path

from terag_release.bird_data import bird_data_root, load_dev_gold
from terag_release.config import REPO_ROOT
from terag_release import reporting

REF = REPO_ROOT / 'results' / 'reference' / 'bird'
METRIC_KEYS = ['tr@1', 'tr@3', 'tr@5', 'covered@5', 'mrr',
               'colp', 'colrec', 'colf1', 'fgcs']
PER_Q_TOL = 1e-9      # shipped CSVs carry 10 decimals
AGG_TOL = 5e-5        # shipped aggregates carry 4 decimals (percent)


def _recompute(protocol, method, gold):
    from terag.engines.evalkit.metrics_core import metric_values
    path = REF / 'predictions' / protocol / f'{method}.jsonl.gz'
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            g = gold[r['qid']]
            rows[r['qid']] = metric_values(
                r, {'gold_tables': g['gold_tables'],
                    'gold_columns': [f'{t}.{c}'
                                     for t, c in g['gold_columns']]})
    return rows


def check_reference_results(output):
    out = reporting.require_new_directory(Path(output))
    gold = load_dev_gold()
    problems = []
    manifest = json.loads((REF / 'MANIFEST.json').read_text(encoding='utf-8'))
    checked = 0
    for key in sorted(manifest['methods']):
        protocol, method = key.split(':', 1)
        rows = _recompute(protocol, method, gold)
        # population: 1530 verified dev questions
        if len(rows) != 1530:
            problems.append(f'{key}: {len(rows)} records, expected 1530')
        # per-question comparison
        csv_path = (REF / 'metrics' / protocol
                    / f'{method}_per_question.csv')
        shipped = {}
        with csv_path.open(encoding='utf-8') as f:
            for row in csv.DictReader(f):
                shipped[row['qid']] = row
        if set(shipped) != set(rows):
            problems.append(f'{key}: qid set differs from shipped metrics')
        for qid, m in rows.items():
            s = shipped.get(qid)
            if s is None:
                continue
            for k in METRIC_KEYS:
                if abs(float(s[k]) - m[k]) > PER_Q_TOL:
                    problems.append(f'{key} {qid} {k}: shipped '
                                    f'{s[k]} vs recomputed {m[k]:.10f}')
        # aggregate comparison (percent, 4 decimals)
        means = {k: sum(m[k] for m in rows.values()) / len(rows)
                 for k in METRIC_KEYS}
        agg_path = REF / 'aggregates' / f'{protocol}_summary.csv'
        with agg_path.open(encoding='utf-8') as f:
            agg = {r['method']: r for r in csv.DictReader(f)}
        a = agg.get(method)
        if a is None:
            problems.append(f'{key}: missing from aggregate summary')
        else:
            for k in METRIC_KEYS:
                if abs(float(a[k]) - means[k] * 100) > AGG_TOL:
                    problems.append(
                        f'{key} {k}: aggregate {a[k]} vs recomputed '
                        f'{means[k] * 100:.4f}')
        checked += 1
        print(f'{key}: {len(rows)} questions recomputed', flush=True)
    report = {'checked': checked, 'problems': problems,
              'ok': not problems}
    reporting.write_json_exclusive(out / 'reference_check.json', report)
    print(('PASS' if report['ok'] else 'FAIL')
          + f' — {checked} method-protocol groups, '
          + f'{len(problems)} problem(s)')
    for p in problems[:20]:
        print(f'  problem: {p}')
    return report['ok']
