"""Population-exact OOF evaluation of one method's predictions against the
frozen Dev five-fold manifests.

Hard gates (any violation => the method is REFUSED and no metric row is
produced; a refused method never enters any comparison):
  1. one predictions.jsonl per applicable cell (question 55, heldout 50);
  2. qid uniqueness within a cell;
  3. the cell qid set equals the manifest fold test_qids EXACTLY;
  4. budget: <=5 retrieved tables, <=10 columns per table per record;
  5. db scope: every retrieved table is in the db's table_schema.csv.
Error/timeout records score zero on every metric and stay in the denominator.

Library use (the public port): call collect_cell_records to load a
results-like tree, then evaluate_records / summarize / write_results.
The research module's argparse CLI and its hash-verified reuse mode are not
ported (the public release ships reviewed reference predictions instead).

--predictions-root layout: <root>/<protocol>/fold<k>/<db>/<method>/predictions.jsonl
"""
import json
import statistics
from pathlib import Path

from terag.engines.evalkit.metrics_core import METRIC_KEYS, actual_counts, gate_problems, metric_values

EXIT_OK, EXIT_GATE, EXIT_REUSE = 0, 2, 3


# ---------------------------------------------------------------- core
def collect_cell_records(predictions_root, protocol, method, pop):
    """Load every applicable cell from a results-like tree; a missing cell is
    a gate violation (population exactness)."""
    root = Path(predictions_root)
    out = {}
    for cell in pop.cells(protocol):
        db, fold = cell['db'], cell['fold']
        path = root / protocol / f'fold{fold}' / db / method / 'predictions.jsonl'
        if not path.is_file():
            raise GateError(f'missing predictions file: {path} '
                            f'(population-exact evaluation requires every cell)')
        records = [json.loads(line) for line in
                   path.read_text(encoding='utf-8').splitlines() if line.strip()]
        out[(protocol, fold, db)] = records
    return out


class GateError(RuntimeError):
    exit_code = EXIT_GATE


def evaluate_records(records_by_cell, pop, raw_gold, method):
    """Gate + score a full method. Returns {'per_question': rows, 'summary':
    {...}}; raises GateError listing every violation (all cells are checked;
    problems are aggregated, not first-fail, so one report shows everything)."""
    problems = []
    for (protocol, fold, db), records in sorted(records_by_cell.items()):
        cell = next(c for c in pop.cells(protocol)
                    if c['db'] == db and c['fold'] == fold)
        problems += [f'{protocol} f{fold} {db}/{method}: {p}'
                     for p in gate_problems(records, cell['test_qids'],
                                            pop.schema_tables(db))]
    if problems:
        raise GateError(f'{method}: {len(problems)} gate violation(s):\n  '
                        + '\n  '.join(problems[:50])
                        + ('' if len(problems) <= 50 else
                           f'\n  ... and {len(problems) - 50} more'))

    rows = []
    for (protocol, fold, db), records in sorted(records_by_cell.items()):
        for r in records:
            qid = r['qid']
            m = metric_values(r, pop.gold_entry(raw_gold, qid))
            n_t, n_c, per_table = actual_counts(r)
            rows.append({'protocol': protocol, 'fold': fold, 'db': db,
                         'qid': qid, 'status': r.get('status'),
                         'n_tables': n_t, 'n_columns': n_c,
                         'max_columns_per_table': max(per_table.values()) if per_table else 0,
                         **m})
    summary = summarize(rows, method)
    return {'per_question': rows, 'summary': summary}


def summarize(rows, method):
    """Aggregates per protocol (and per db within protocol). All metrics are
    means over the FIXED denominator (errors included as zeros)."""
    def agg(subset):
        if not subset:
            return None
        out = {'n': len(subset),
               'n_error': sum(1 for r in subset if r['status'] != 'ok'),
               'n_ok': sum(1 for r in subset if r['status'] == 'ok')}
        for k in METRIC_KEYS:
            out[k] = sum(r[k] for r in subset) / len(subset)
        for k in ('n_tables', 'n_columns', 'max_columns_per_table'):
            vals = [r[k] for r in subset]
            out[f'mean_{k}'] = sum(vals) / len(vals)
            out[f'median_{k}'] = statistics.median(vals)
        return out

    summary = {'method': method, 'protocols': {}}
    for protocol in sorted({r['protocol'] for r in rows}):
        sub = [r for r in rows if r['protocol'] == protocol]
        entry = {'overall': agg(sub), 'per_db': {}, 'per_fold': {}}
        for db in sorted({r['db'] for r in sub}):
            entry['per_db'][db] = agg([r for r in sub if r['db'] == db])
        for fold in sorted({r['fold'] for r in sub}):
            entry['per_fold'][str(fold)] = agg(
                [r for r in sub if r['fold'] == fold])
        summary['protocols'][protocol] = entry
    return summary


# ---------------------------------------------------------------- output
def write_results(result, out_dir, extra=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ['protocol', 'fold', 'db', 'qid', 'status', 'n_tables',
            'n_columns', 'max_columns_per_table'] + METRIC_KEYS
    lines = [','.join(cols)]
    for r in result['per_question']:
        lines.append(','.join(
            str(r[c]) if not isinstance(r[c], float) else f'{r[c]:.10f}'
            for c in cols))
    (out / 'per_question.csv').write_text('\n'.join(lines) + '\n',
                                          encoding='utf-8')
    payload = dict(result['summary'])
    if extra:
        payload.update(extra)
    (out / 'summary.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + '\n',
        encoding='utf-8')
