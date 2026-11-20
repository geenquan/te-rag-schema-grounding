"""Offline recomputation check for the shipped BIRD reference results.

Reads the sanitized reference predictions (results/reference/bird/), whose
only inputs are the frozen gold under data/bird/processed/, recomputes the
nine metrics per question with the frozen metric implementation, and
compares them to the shipped per-question CSVs and aggregate summaries. Any
mismatch fails. No model, no API, no network.

Guarded inputs (independent-review fixes): the MANIFEST's per-file SHA256
is verified, per-question files are loaded with duplicate-qid rejection,
the population is checked against the EXACT frozen OOF set per protocol
(never just len(dict) == 1530), statuses must be legal, and the set of
error records is reconciled against the machine-readable expected-frozen-
failure list — the 33 historically frozen errors are disclosed as such,
while any NEW error (or any disappearance of an expected one) is reported
as a regression problem.
"""
import csv
import json
from pathlib import Path

from terag_release.bird_data import load_dev_gold
from terag_release.config import REPO_ROOT
from terag_release import reporting
from terag_release.input_guards import (DuplicateQidError,
                                        InvalidMetricError,
                                        expected_population,
                                        load_metric_rows,
                                        load_prediction_rows)

REF = REPO_ROOT / 'results' / 'reference' / 'bird'
EXPECTED_FAILURES_PATH = REF / 'EXPECTED_FROZEN_FAILURES.json'
METRIC_KEYS = ['tr@1', 'tr@3', 'tr@5', 'covered@5', 'mrr',
               'colp', 'colrec', 'colf1', 'fgcs']
PER_Q_TOL = 1e-9      # shipped CSVs carry 10 decimals
AGG_TOL = 5e-5        # shipped aggregates carry 4 decimals (percent)


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _expected_failures():
    if not EXPECTED_FAILURES_PATH.is_file():
        return {}
    data = json.loads(EXPECTED_FAILURES_PATH.read_text(encoding='utf-8'))
    return {(e['protocol'], e['method'], e['qid']): e
            for e in data.get('failures', [])}


def check_reference_results(output):
    out = reporting.require_new_directory(Path(output))
    gold = load_dev_gold()
    problems = []
    expected_failures = _expected_failures()
    seen_failures = set()
    manifest = json.loads((REF / 'MANIFEST.json').read_text(encoding='utf-8'))
    checked = 0
    for key in sorted(manifest['methods']):
        protocol, method = key.split(':', 1)
        # ---- manifest hash verification (file identity, not just size) ---
        gz_path = REF / 'predictions' / protocol / f'{method}.jsonl.gz'
        if not gz_path.is_file():
            problems.append(f'{key}: predictions file missing: {gz_path}')
            continue
        recorded = manifest['methods'][key].get('sha256_gzip')
        actual = _sha256_file(gz_path)
        if recorded != actual:
            problems.append(f'{key}: predictions SHA256 mismatch (manifest '
                            f'{recorded} != actual {actual})')
        # ---- guarded load: duplicate qids rejected before any dict -------
        try:
            rows = load_prediction_rows(gz_path)
        except DuplicateQidError as exc:
            problems.append(f'{key}: {exc}')
            continue
        # ---- exact frozen population (per protocol, from the manifests) --
        expected_qids = expected_population(protocol)
        if set(rows) != expected_qids:
            problems.append(
                f'{key}: population mismatch — {len(rows)} records vs the '
                f'frozen {len(expected_qids)}-question OOF population '
                f'(missing {len(expected_qids - set(rows))}, extra '
                f'{len(set(rows) - expected_qids)})')
        # ---- statuses + expected-failure reconciliation -------------------
        for qid, record in rows.items():
            status = record.get('status')
            if status not in ('ok', 'error', 'timeout'):
                problems.append(f'{key} {qid}: unknown status {status!r}')
            if status != 'ok':
                seen_failures.add((protocol, method, qid))
        if manifest['methods'][key]['n_questions'] != len(rows):
            problems.append(f'{key}: manifest n_questions '
                            f'{manifest["methods"][key]["n_questions"]} != '
                            f'loaded records {len(rows)}')

        # ---- per-question recomputation -----------------------------------
        from terag.engines.evalkit.metrics_core import metric_values
        recomputed = {}
        for qid, r in rows.items():
            g = gold[r['qid']]
            recomputed[r['qid']] = metric_values(
                r, {'gold_tables': g['gold_tables'],
                    'gold_columns': [f'{t}.{c}'
                                     for t, c in g['gold_columns']]})
        # ---- guarded per-question metrics CSV -----------------------------
        csv_path = (REF / 'metrics' / protocol / f'{method}_per_question.csv')
        try:
            shipped = load_metric_rows(csv_path)
        except (DuplicateQidError, InvalidMetricError, ValueError) as exc:
            problems.append(f'{key}: {exc}')
            shipped = {}
        if set(shipped) != expected_qids:
            problems.append(f'{key}: shipped metrics population differs '
                            'from the frozen OOF set')
        for qid, m in recomputed.items():
            s = shipped.get(qid)
            if s is None:
                continue
            for k in METRIC_KEYS:
                if abs(float(s[k]) - m[k]) > PER_Q_TOL:
                    problems.append(f'{key} {qid} {k}: shipped '
                                    f'{s[k]} vs recomputed {m[k]:.10f}')
        # ---- aggregate comparison (percent, 4 decimals) --------------------
        means = {k: sum(m[k] for m in recomputed.values()) / len(recomputed)
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
        print(f'{key}: {len(rows)} questions recomputed '
              f'({sum(1 for r in rows.values() if r["status"] != "ok")} '
              f'error records)', flush=True)

    # ---- expected-failure reconciliation ----------------------------------
    unexpected = sorted(seen_failures - set(expected_failures))
    disappeared = sorted(set(expected_failures) - seen_failures)
    for u in unexpected:
        problems.append(f'unexpected error record (regression, NOT an '
                        f'expected frozen failure): {u}')
    for d in disappeared:
        problems.append(f'expected frozen failure disappeared (reference '
                        f'records changed?): {d}')
    report = {'checked': checked, 'problems': problems,
              'expected_frozen_failures': len(seen_failures
                                              & set(expected_failures)),
              'unexpected_failures': len(unexpected),
              'ok': not problems}
    reporting.write_json_exclusive(out / 'reference_check.json', report)
    print(('PASS' if report['ok'] else 'FAIL')
          + f' — {checked} method-protocol groups, '
          + f'{len(seen_failures & set(expected_failures))} expected '
          + 'frozen failures disclosed, '
          + f'{len(problems)} problem(s)')
    for p in problems[:20]:
        print(f'  problem: {p}')
    return report['ok']
