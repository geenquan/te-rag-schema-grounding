"""Final-grid replay runner: execute the accepted GridEngine on authorized
private-grid data with the frozen per-fold parameters.

This is the RUNNABLE entry point of the final paper profile's grid side
(the engine itself is ``terag.engines.grid.GridEngine``; the legacy
``run-experiments`` path still runs the original 25-group matrix and both
remain available — this command never substitutes one for the other).

Inputs
------
* a dataset config (name under configs/datasets/ or a YAML path) declaring
  ``data.field_csv`` / ``data.table_csv`` / ``data.qa_csv`` / ``data.gold``
  — the documented private-data format (docs/data_format.md);
* a frozen splits manifest from ``terag_release make-folds``
  (``question_folds`` for the ``question_v1`` setting,
  ``ast_refined_table_folds`` for ``table_v2``);
* the frozen parameter table
  ``configs/paper/final_20261120/grid_selected_configurations.csv``.

Tracks
------
* ``full``     — per (setting, fold) the frozen Full parameters;
* ``fixed``    — the SAME per-fold Full parameters with ONE module switch
                 turned off at predict time (params identical, only the
                 engine variant changes — the fixed-ablation definition);
* ``optimized``— the frozen parameters already selected FOR THAT VARIANT
                 in that fold (independent-optimization replay; no search
                 is executed, the recorded ``chosen`` config is replayed).

Every run writes a NEW output directory with per-cell predictions (the
same record shape as the other runners: qid/setting/fold/method/variant/
status/retrieved_tables/retrieved_column_keys/retrieved_columns_detail),
per-cell metrics recomputed from the final records against the dataset
gold, effective parameters + their source (row ``chosen`` id) per cell,
and a run_meta provenance block (code hashes, parameter-table SHA256,
manifest SHA256, input checksums, statuses, timings). Failed questions
become error records that stay in the denominator.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

from terag_release import reporting
from terag_release.config import REPO_ROOT, load_dataset
from terag_release.supervision import inject_supervision

PARAMS_CSV = (REPO_ROOT / 'configs' / 'paper' / 'final_20261120'
              / 'grid_selected_configurations.csv')
SETTINGS = ('question_v1', 'table_v2')
STOCK_SWITCHES = ['noBIndex', 'noBHit', 'noB', 'noTableGraph',
                  'noFieldGraph', 'noGraph', 'noPattern', 'noC']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_selected_params(csv_path=PARAMS_CSV):
    """(setting, fold, variant) -> {params, chosen} from the frozen table.

    The frozen table carries one row per (setting, fold, variant): each
    variant's ``chosen`` config is the independently optimized selection
    FOR THAT VARIANT in that fold; the Full row is the final Full.
    """
    table = {}
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['setting'], int(row['fold']), row['variant'])
            if key in table:
                raise ValueError(f'duplicate frozen parameter row: {key}')
            table[key] = {'params': json.loads(row['params']),
                          'chosen': row['chosen']}
    return table


def resolve_track_params(track, member, setting, fold, table):
    """Effective (variant, params, source) for one cell of one track."""
    if track == 'full':
        entry = table.get((setting, fold, 'Full'))
        if entry is None:
            raise ValueError(f'no frozen Full parameters for '
                             f'{setting} fold {fold}')
        return 'Full', entry['params'], f'full:{entry["chosen"]}'
    if track == 'fixed':
        if member not in STOCK_SWITCHES:
            raise ValueError(f'unknown ablation member {member!r} '
                             f'(one of {STOCK_SWITCHES})')
        entry = table.get((setting, fold, 'Full'))
        if entry is None:
            raise ValueError(f'no frozen Full parameters for '
                             f'{setting} fold {fold}')
        # fixed ablation: the Full parameters, only the module switch off
        return member, entry['params'], f'fixed:{entry["chosen"]}'
    if track == 'optimized':
        if member not in STOCK_SWITCHES:
            raise ValueError(f'unknown variant {member!r} '
                             f'(one of {STOCK_SWITCHES})')
        entry = table.get((setting, fold, member))
        if entry is None:
            raise ValueError(f'no frozen optimized parameters for '
                             f'{setting} fold {fold} variant {member}')
        return member, entry['params'], f'optimized:{entry["chosen"]}'
    raise ValueError(f'unknown track {track!r} (full|fixed|optimized)')


def load_splits(manifest_path, ordered_qids):
    """{setting: {fold: {'train': [...], 'test': [...]}}} from a make-folds
    manifest (question_folds / ast_refined_table_folds).

    ``question_folds`` entries are 0-BASED ROW INDICES over the ordered qid
    list (the manifest's documented semantics); ``ast_refined_table_folds``
    entries already carry qid lists. The training side of a question fold is
    the union of the other folds (each question is tested exactly once).
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    splits = {}
    q = manifest.get('question_folds') or {}
    splits['question_v1'] = {}
    all_q = list(ordered_qids)
    for k, indices in q.items():
        test = [all_q[int(i)] for i in indices]
        test_set = set(test)
        splits['question_v1'][int(k)] = {
            'test': test,
            'train': [x for x in all_q if x not in test_set]}
    t = manifest.get('ast_refined_table_folds') or {}
    splits['table_v2'] = {int(k): {'train': list(v['train']),
                                   'test': list(v['test'])}
                          for k, v in t.items()}
    for setting in SETTINGS:
        if not splits.get(setting):
            raise ValueError(f'manifest has no folds for {setting}: '
                             f'{manifest_path}')
    return splits


def run_grid_replay(dataset, splits_manifest, track, member=None,
                    settings=None, fold=None, output=None, params_csv=None,
                    qid_prefix='q'):
    import pandas as pd

    dataset_cfg, _ = load_dataset(dataset)
    data = dataset_cfg['data']
    for key in ('field_csv', 'table_csv', 'qa_csv', 'gold'):
        if not data.get(key):
            raise ValueError(f'dataset config misses data.{key}')
    for key in ('field_csv', 'table_csv', 'qa_csv'):
        if not Path(data[key]).is_file():
            raise FileNotFoundError(f'dataset data.{key}: {data[key]}')
    gold_path = Path(data['gold'])
    if not gold_path.is_file():
        raise FileNotFoundError(f'dataset gold: {gold_path}')

    run_dir = reporting.require_new_directory(Path(output))
    params_csv = Path(params_csv) if params_csv else PARAMS_CSV
    table = load_selected_params(params_csv)
    settings = list(settings) if settings else list(SETTINGS)

    rows = pd.read_csv(data['qa_csv'])
    if 'qid' not in rows.columns:
        # documented positional qid contract (docs/data_format.md); the
        # prefix is configurable (the research protocol used pg, the
        # documented public format uses q)
        rows['qid'] = [f'{qid_prefix}{i:03d}' for i in range(len(rows))]
    rows['qid'] = rows['qid'].astype(str)
    question_text = dict(zip(rows.qid, rows.question.astype(str)))
    splits = load_splits(splits_manifest, rows.qid.tolist())
    gold = {}
    if gold_path.suffix == '.jsonl':
        for line in gold_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                gold[str(rec['qid'])] = rec
    else:
        gold = {str(k): v for k, v in json.loads(
            gold_path.read_text(encoding='utf-8')).items()}
    if set(question_text) != set(gold):
        raise ValueError('QA qids and gold qids differ — refusing to run')

    from terag.engines.grid import GridEngine, make_grid_config

    statuses = []
    t0 = time.perf_counter()
    for setting in settings:
        for f in sorted(splits[setting]):
            if fold is not None and f != fold:
                continue
            spec = splits[setting][f]
            variant, params, source = resolve_track_params(
                track, member, setting, f, table)
            cell_dir = run_dir / setting / f'fold{f}' / variant
            cell_dir.mkdir(parents=True, exist_ok=True)
            t_cell = time.perf_counter()
            records, n_err = [], 0
            fit_stats = None
            try:
                train = inject_supervision(
                    rows[rows.qid.astype(str).isin(spec['train'])]
                    .reset_index(drop=True), gold, 'full_sql')
                cache = run_dir / 'engine_cache' / f'{setting}_fold{f}'
                cache.mkdir(parents=True, exist_ok=True)
                cfg = make_grid_config(
                    field_csv=data['field_csv'],
                    table_csv=data['table_csv'],
                    artifacts_dir=cache)
                gold_local = {q: gold[q] for q in spec['test']}
                engine = GridEngine(cfg, setting, f, train, gold_local,
                                    question_text, cache_dir=cache)
                fit_stats = engine.fit_stats()
                for q in spec['test']:
                    t1 = time.perf_counter()
                    try:
                        rec, _metric, _trace = engine.predict(
                            q, params, variant)
                        rec['setting'] = setting
                        rec['fold'] = f
                        rec['method'] = variant
                        rec['query_time'] = time.perf_counter() - t1
                        records.append(rec)
                    except Exception as exc:  # per-question isolation
                        n_err += 1
                        records.append(
                            {'qid': q, 'setting': setting, 'fold': f,
                             'method': variant, 'variant': variant,
                             'status': 'error',
                             'error': str(exc)[:300],
                             'retrieved_tables': [],
                             'retrieved_column_keys': [],
                             'retrieved_columns_detail': [],
                             'query_time': time.perf_counter() - t1})
            except Exception as exc:  # construction failure -> error records
                n_err += len(spec['test'])
                records = [{'qid': q, 'setting': setting, 'fold': f,
                            'method': variant, 'variant': variant,
                            'status': 'error', 'error': str(exc)[:300],
                            'retrieved_tables': [],
                            'retrieved_column_keys': [],
                            'retrieved_columns_detail': [],
                            'query_time': 0.0}
                           for q in spec['test']]
            with (cell_dir / 'predictions.jsonl').open('w',
                                                       encoding='utf-8') as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False,
                                        default=str) + '\n')
            # per-cell metrics recomputed from the FINAL records
            from eval.full_gold import full_gold_metrics
            per_q = []
            for r in records:
                m = full_gold_metrics(r, gold)
                per_q.append({'qid': r['qid'],
                              'status': r.get('status'),
                              'covered@5': float(m[1]),
                              'tr@1': float(m[0][1]),
                              'tr@5': float(m[0][5]),
                              'colp': float(m[2]), 'colrec': float(m[3]),
                              'colf1': float(m[4]), 'fgcs': float(m[5])})
            summary = {'setting': setting, 'fold': f, 'variant': variant,
                       'track': track, 'n': len(records),
                       'n_error': n_err,
                       'covered@5': (sum(p['covered@5'] for p in per_q)
                                     / len(per_q)) if per_q else 0.0,
                       'colf1': (sum(p['colf1'] for p in per_q)
                                 / len(per_q)) if per_q else 0.0,
                       'fgcs': (sum(p['fgcs'] for p in per_q)
                                / len(per_q)) if per_q else 0.0}
            cell_meta = {'setting': setting, 'fold': f, 'method': variant,
                         'track': track,
                         'status': ('failed' if n_err == len(records)
                                    else 'completed_with_errors'
                                    if n_err else 'ok'),
                         'n': len(records), 'n_error': n_err,
                         'params_source': source,
                         'params_sha256': digest(params),
                         'effective_params': params,
                         'wall_seconds': time.perf_counter() - t_cell,
                         'fit_stats': fit_stats}
            (cell_dir / 'cell.json').write_text(
                json.dumps(cell_meta, ensure_ascii=False, indent=1,
                           default=str) + '\n', encoding='utf-8')
            (cell_dir / 'per_question_metrics.json').write_text(
                json.dumps(per_q, ensure_ascii=False, indent=1) + '\n',
                encoding='utf-8')
            statuses.append(cell_meta)
            print(f'{setting} f{f} {variant} ({track}): n={len(records)} '
                  f'err={n_err}', flush=True)

    run_status = 'completed'
    if any(s['status'] != 'ok' for s in statuses):
        run_status = ('failed' if statuses and all(s['status'] == 'failed'
                                                   for s in statuses)
                      else 'completed_with_failures')
    meta = {'run_kind': 'grid_replay', 'run_status': run_status,
            'track': track, 'member': member,
            'dataset': {k: str(v) for k, v in data.items()},
            'params_table': {'path': str(params_csv),
                             'sha256': sha256_file(params_csv)},
            'splits_manifest': {'path': str(splits_manifest),
                                'sha256': sha256_file(splits_manifest)},
            'input_sha256': {k: sha256_file(Path(v))
                             for k, v in data.items()},
            'cells': statuses, 'wall_seconds': time.perf_counter() - t0}
    (run_dir / 'run_meta').mkdir(parents=True, exist_ok=True)
    reporting.write_json_exclusive(run_dir / 'run_meta' / 'run_meta.json',
                                   meta)
    print(f"grid replay '{run_status}': {run_dir}")
    return meta
