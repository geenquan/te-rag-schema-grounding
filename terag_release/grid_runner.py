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

Acceptance and execution-plan semantics
---------------------------------------
* The raw manifest is FULLY re-accepted on the current QA/gold/qids via
  ``terag_release.protocols.validate_manifest`` BEFORE any engine is
  constructed (row/qid identity and order, five-fold completeness/
  exclusivity/coverage, duplicate groups never crossing folds, train/
  test disjointness, held-out target membership, AST isolation recomputed
  from the CURRENT gold, conservative unparseable-SQL exclusions). The
  cached ``checks`` block inside the manifest is never trusted, and the
  format conversion in ``load_splits`` never replaces this acceptance.
* An execution plan is built up front from the ACCEPTED manifest; an
  invalid fold (out of the manifest's folds, or negative), an unknown
  setting, or an empty plan is a hard error — a zero-unit run can never
  be written out as a success.
* Run statuses: ``completed`` (the full plan, both settings × all folds,
  no question-level errors), ``partial`` (a legitimately narrowed clean
  run — e.g. ``--fold N`` — whose whole non-empty plan executed; the real
  outputs are kept and clearly labeled), ``completed_with_failures`` /
  ``completed_with_failures_partial`` / ``failed`` (question/cell
  failures; the CLI exits non-zero), ``incomplete`` (planned units
  missing — never downgraded to a success wording).

Every run writes a NEW output directory with per-cell predictions (the
same record shape as the other runners: qid/setting/fold/method/variant/
status/retrieved_tables/retrieved_column_keys/retrieved_columns_detail),
per-cell metrics recomputed from the final records against the dataset
gold, effective parameters + their source (row ``chosen`` id) per cell,
and a run_meta provenance block (code hashes, parameter-table SHA256,
manifest SHA256, input checksums, execution plan, statuses, timings).
Failed questions become error records that stay in the denominator.
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


def load_splits(manifest, ordered_qids):
    """{setting: {fold: {'train': [...], 'test': [...]}}} from an
    ALREADY-VALIDATED make-folds manifest dict (question_folds /
    ast_refined_table_folds).

    This is a pure FORMAT conversion on top of a manifest that has passed
    ``terag_release.protocols.validate_manifest`` (the caller's job — the
    conversion never replaces or weakens that acceptance), so the fold
    membership is trusted exactly as validated. ``question_folds`` entries
    are 0-BASED ROW INDICES over the ordered qid list (the manifest's
    documented semantics); ``ast_refined_table_folds`` entries already
    carry qid lists. The training side of a question fold is the union of
    the other folds (each question is tested exactly once).
    """
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
            raise ValueError(f'manifest has no folds for {setting}')
    return splits


def validate_manifest_full(manifest_path, rows, gold, qids):
    """Re-acceptance of the raw manifest on the CURRENT data, BEFORE any
    engine is constructed: the existing framework check
    (``terag_release.protocols.validate_manifest``), never trusting the
    manifest's cached ``checks`` block. Covers QA row count, qid identity
    and order, five-fold completeness/exclusivity/coverage, duplicate
    question groups never crossing folds, train/test disjointness,
    held-out target-table vs test membership, AST isolation recomputed
    from the current gold, and the conservative unparseable-train-SQL
    exclusion. Returns the validated manifest dict (raises
    ManifestValidationError on any tampering).
    """
    from terag_release import protocols
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    protocols.validate_manifest(manifest, rows, gold, list(qids))
    return manifest


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

    params_csv = Path(params_csv) if params_csv else PARAMS_CSV
    table = load_selected_params(params_csv)
    settings = list(SETTINGS) if settings is None else list(settings)

    rows = pd.read_csv(data['qa_csv'])
    if 'qid' not in rows.columns:
        # documented positional qid contract (docs/data_format.md); the
        # prefix is configurable (the research protocol used pg, the
        # documented public format uses q)
        rows['qid'] = [f'{qid_prefix}{i:03d}' for i in range(len(rows))]
    rows['qid'] = rows['qid'].astype(str)
    question_text = dict(zip(rows.qid, rows.question.astype(str)))
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

    # ---- FULL manifest re-acceptance on the CURRENT data, BEFORE any
    # engine is constructed (train/test overlap, leak tampering, fold
    # integrity, AST isolation recomputed from this gold — the cached
    # ``checks`` block inside the manifest is never trusted) ----
    manifest = validate_manifest_full(splits_manifest, rows, gold,
                                      rows.qid.tolist())
    splits = load_splits(manifest, rows.qid.tolist())

    # ---- execution plan: validate settings/fold against the ACCEPTED
    # manifest and require a non-empty plan before anything executes ----
    if not settings:
        raise ValueError('no settings requested — the execution plan is '
                         'empty; refusing to report an empty run')
    unknown = [s for s in settings if s not in SETTINGS]
    if unknown:
        raise ValueError(f'unknown settings {unknown} (legal: '
                         f'{list(SETTINGS)})')
    available_folds = sorted({f for s in settings for f in splits[s]})
    if fold is not None:
        if fold < 0:
            raise ValueError(f'invalid fold {fold}: folds are 0-based '
                             'non-negative integers')
        if fold not in available_folds:
            raise ValueError(f'invalid fold {fold}: not present in the '
                             f'accepted manifest (available folds: '
                             f'{available_folds})')
    plan = [(setting, f) for setting in settings
            for f in sorted(splits[setting])
            if fold is None or f == fold]
    if not plan:
        raise ValueError('empty execution plan: no (setting, fold) unit '
                         'matches the request — refusing to write a '
                         'success status for a zero-unit run')

    # only NOW is the output directory created: an invalid request (bad
    # manifest, invalid fold, empty plan) leaves no artifacts behind
    run_dir = reporting.require_new_directory(Path(output))

    from terag.engines.grid import GridEngine, make_grid_config

    statuses = []
    t0 = time.perf_counter()
    for setting, f in plan:
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

    # ---- run status: an EMPTY or INCOMPLETE execution is never "completed"
    # ----
    # partial: a legitimately narrowed run (single fold via --fold or a
    #   subset of settings) that executed its whole non-empty plan — the
    #   real outputs are kept and clearly labeled, never presented as the
    #   full five-fold run;
    # completed: every unit of the FULL plan (both settings, all folds)
    #   executed without question-level errors;
    # completed_with_failures / failed: question/cell failures occurred
    #   (non-zero exit at the CLI);
    # incomplete: planned units are missing from the executed set — never
    #   downgraded to a success wording.
    planned_keys = [(s, f) for s, f in plan]
    executed_keys = [(s['setting'], s['fold']) for s in statuses]
    missing_units = [k for k in planned_keys if k not in executed_keys]
    full_plan = [(s, f) for s in SETTINGS for f in sorted(splits[s])]
    is_narrowed = sorted(planned_keys) != sorted(full_plan)
    if missing_units:
        run_status = 'incomplete'
    elif not statuses:
        run_status = 'incomplete'
    elif all(s['status'] == 'failed' for s in statuses):
        run_status = 'failed'
    elif any(s['status'] != 'ok' for s in statuses):
        run_status = ('completed_with_failures_partial' if is_narrowed
                      else 'completed_with_failures')
    elif is_narrowed:
        run_status = 'partial'
    else:
        run_status = 'completed'
    meta = {'run_kind': 'grid_replay', 'run_status': run_status,
            'track': track, 'member': member,
            'dataset': {k: str(v) for k, v in data.items()},
            'params_table': {'path': str(params_csv),
                             'sha256': sha256_file(params_csv)},
            'splits_manifest': {'path': str(splits_manifest),
                                'sha256': sha256_file(splits_manifest),
                                'revalidated': True},
            'input_sha256': {k: sha256_file(Path(v))
                             for k, v in data.items()},
            'execution_plan': [{'setting': s, 'fold': f}
                               for s, f in planned_keys],
            'n_planned_units': len(planned_keys),
            'n_executed_units': len(executed_keys),
            'missing_units': [{'setting': s, 'fold': f}
                              for s, f in missing_units],
            'full_plan_units': len(full_plan),
            'is_partial': is_narrowed,
            'cells': statuses, 'wall_seconds': time.perf_counter() - t0}
    (run_dir / 'run_meta').mkdir(parents=True, exist_ok=True)
    reporting.write_json_exclusive(run_dir / 'run_meta' / 'run_meta.json',
                                   meta)
    print(f"grid replay '{run_status}': {run_dir}")
    return meta
