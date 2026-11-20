"""BIRD experiment runner: the public execution path for the final-paper
BIRD methods on the frozen Dev five-fold population.

Execution semantics are those of the frozen final evaluation (unified entry):
per question, for a method whose frozen params declare ``field_rerank`` or
``hg``, the layers MUST execute (a declared-but-not-executed mechanism is a
hard error, never a silent no-op); budgets (5 tables / 10 columns per table)
are enforced after every mechanism; metrics are computed from the FINAL
record. Engine construction follows the frozen per-database construction
(schema CSVs under data/bird/processed, supervised train rows per fold, gold
in the db-local simple-name view).

Method sources
--------------
* ``Core`` (paper name for the original full run) — BirdEngine, variant
  ``Full``, the frozen ``DEFAULT`` parameter vector (verified equal to the
  original run's recorded effective params).
* frozen-config methods — every entry of
  ``configs/paper/final_20261120/bird_final_configs.json`` (selected Q/H,
  fixed and optimized ablations, adaptive variants, candidates), executed
  with their recorded ``engine`` kind (``bird``/``vnext``) and params.
* offline baselines — BM25, MetadataMatched, ExampleRRF (both protocols) and
  TFIDF, Hybrid, CooccurRRF (both protocols) through the frozen workspace
  retrievers, exactly as the original adapters wired them.
* model/API methods — Dense (MiniLM download) and DenseDeepSeek /
  KaSLA-Adapted / LinkAlign-SingleDB-Adapted (DeepSeek API) run ONLY with
  ``--allow-network``; API methods additionally require an explicit budget
  cap and DEEPSEEK_API_KEY. Question protocol only, as in the paper.

Every run writes a NEW output directory with predictions, per-question
execution evidence, per-cell status, per-method recomputed metrics and a
run_meta provenance block (code hashes, config hash, data identity, planned
and executed units, statuses, timings). Failed questions become error
records that stay in the fixed denominator; the command exits non-zero when
any question fails.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from terag_release.bird_data import (applicable_cells, bird_data_root,
                                     identity_report, load_dev_gold,
                                     load_manifest, schema_dir, train_rows)
from terag_release.config import REPO_ROOT
from terag_release import reporting

FROZEN_CONFIG_PATH = (REPO_ROOT / 'configs' / 'paper' / 'final_20261120'
                      / 'bird_final_configs.json')
FROZEN_FAMILIES_PATH = (REPO_ROOT / 'configs' / 'paper' / 'final_20261120'
                        / 'bird_families.json')
FROZEN_FAMILIES_SHA = (REPO_ROOT / 'configs' / 'paper' / 'final_20261120'
                       / 'bird_families.sha256')

PROTOCOLS = ('question', 'heldout')
PER_QUESTION_TIMEOUT_S = 300.0

# protocol availability, exactly as the paper ran the methods (absent groups
# are never zero-filled; Dense/API never ran the heldout protocol)
BASELINE_PROTOCOLS = {
    'BM25': ('question', 'heldout'),
    'MetadataMatched': ('question', 'heldout'),
    'ExampleRRF': ('question', 'heldout'),
    'TFIDF': ('question', 'heldout'),
    'Hybrid': ('question', 'heldout'),
    'CooccurRRF': ('question', 'heldout'),
    'Dense': ('question',),
    'DenseDeepSeek': ('question',),
    'KaSLA-Adapted': ('question',),
    'LinkAlign-SingleDB-Adapted': ('question',),
}


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


def require_hashseed():
    """The frozen engine requires PYTHONHASHSEED=0 (cross-process identity);
    re-exec once with it set rather than running nondeterministically."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = '0'
        raise SystemExit(subprocess.run(
            [sys.executable] + sys.argv, env=env, cwd=os.getcwd()).returncode)


# ---------------------------------------------------------------- methods
def load_frozen_methods():
    """name -> {engine, engine_variant, params, group} from the frozen
    configs. Duplicate names across groups are a hard error (identity)."""
    data = json.loads(FROZEN_CONFIG_PATH.read_text(encoding='utf-8'))
    out = {}
    def register(entry, group):
        name = entry['name']
        if name in out:
            raise ValueError(f'duplicate frozen method name: {name}')
        out[name] = {'engine': entry.get('engine') or 'bird',
                     'engine_variant': entry.get('engine_variant') or 'Full',
                     'params': entry['params'], 'group': group}
    for proto in PROTOCOLS:
        register(data['selected'][proto], f'selected.{proto}')
    for group in ('other_candidates', 'fixed_ablation_members',
                  'optimized_ablations', 'adaptive_variants'):
        for entry in data.get(group) or []:
            register(entry, group)
    tb = data.get('tradeoff_branch')
    if isinstance(tb, dict):
        register(tb, 'tradeoff_branch')
    return out


def engine_methods_for_protocol(frozen, protocol):
    """Engine methods run on BOTH protocols (final-run semantics)."""
    return {n: e for n, e in frozen.items()}


def resolve_methods(requested, experiment_methods=None):
    """Resolve a method list into execution entries.

    Names may be: 'Core' (DEFAULT-parameter full engine), a frozen-config
    method name, or an offline/network baseline name. Experiment YAML lists
    expand the same way; unknown names are hard errors.
    """
    frozen = load_frozen_methods()
    names = list(requested) if requested else list(experiment_methods or [])
    if not names:
        raise ValueError('no methods requested')
    resolved = {}
    for name in names:
        if name == 'Core':
            from terag.engines.bird.study_defaults import DEFAULT
            resolved[name] = {'kind': 'engine', 'engine': 'bird',
                              'engine_variant': 'Full',
                              'params': dict(DEFAULT), 'group': 'core'}
        elif name in frozen:
            e = frozen[name]
            resolved[name] = {'kind': 'engine', **e}
        elif name in BASELINE_PROTOCOLS:
            resolved[name] = {'kind': 'baseline', 'group': 'baseline'}
        else:
            raise ValueError(
                f'unknown BIRD method {name!r}: expected Core, a '
                'configs/paper/final_20261120/bird_final_configs.json '
                'method name, or one of '
                f'{sorted(BASELINE_PROTOCOLS)}')
    return resolved


def method_requires_network(name):
    return name in ('Dense', 'DenseDeepSeek', 'KaSLA-Adapted',
                    'LinkAlign-SingleDB-Adapted')


# ---------------------------------------------------------------- cells
def gold_local_for(gold, db, qids):
    out, qtext = {}, {}
    for qid in qids:
        g = gold[qid]
        out[qid] = {'gold_status': g.get('parse_status', g.get('gold_status')),
                    'gold_tables': list(g['gold_tables']),
                    'gold_columns': [f'{t}.{c}' for t, c in g['gold_columns']]}
        qtext[qid] = g['question']
    return out, qtext


def cell_iter(protocols, fold_filter=None):
    for protocol in protocols:
        for cell in applicable_cells(protocol):
            if fold_filter is not None and cell['fold'] != fold_filter:
                continue
            yield protocol, cell


# ---------------------------------------------------------------- run loop
def error_record(protocol, fold, db, qid, method, reason, elapsed=0.0):
    return {'protocol': protocol, 'fold': fold, 'db_id': db, 'qid': qid,
            'variant': method, 'status': 'error',
            'error': str(reason)[:300], 'retrieved_tables': [],
            'retrieved_column_keys': [], 'retrieved_columns_detail': [],
            'query_time': elapsed}


def _engine_class(kind):
    if kind == 'vnext':
        from terag.engines.bird.vnext_engine import VnextEngine
        return VnextEngine
    from terag.engines.bird.bird_engine import BirdEngine
    return BirdEngine


def run_cell_methods(protocol, cell, methods, run_dir, cache_root,
                     allow_network, budget_cap_cny, minilm_cache_dir,
                     statuses):
    """Run every engine/baseline method of the selection on one cell.

    Engines are built once per engine kind and shared by all methods of that
    kind (per-method ``modules`` masks live in the params, not the engine).
    """
    import pandas as pd
    from terag.engines.bird.config_factory import make_bird_config

    db, fold = cell['db'], cell['fold']
    qids = list(cell['test_qids'])
    rows = train_rows(protocol, fold, db)
    train_df = pd.DataFrame(rows)
    manifest_train = set(cell['train_qids'])
    file_train = {str(r['qid']) for r in rows}
    if manifest_train != file_train:
        raise RuntimeError(
            f'{protocol} f{fold} {db}: train qid set differs from the frozen '
            'manifest (data identity broken)')
    gold = load_dev_gold()
    gold_local, qtext = gold_local_for(gold, db, qids)
    sdir = schema_dir(db, 'processed')
    field_csv = str(sdir / 'field_schema.csv')
    table_csv = str(sdir / 'table_schema.csv')
    heldout_tables = (list(cell.get('heldout_tables') or [])
                      if protocol == 'heldout' else None)

    from terag.engines.bird.unified_entry import run_one
    from terag.engines.hgate.gating import predict_gated

    engines = {}

    def get_engine(kind):
        if kind not in engines:
            cache = cache_root / f'{protocol}_f{fold}_{db}'
            cache.mkdir(parents=True, exist_ok=True)
            cfgdb = make_bird_config(db, schema_dir=sdir,
                                     artifacts_dir=cache)
            cls = _engine_class(kind)
            engines[kind] = cls(cfgdb, db, train_df, qids, gold_local, qtext,
                                cache_dir=cache,
                                heldout_tables=heldout_tables)
        return engines[kind]

    for name, spec in sorted(methods.items()):
        outdir = run_dir / protocol / f'fold{fold}' / db / name
        if (outdir / 'predictions.jsonl').exists():
            raise RuntimeError(f'refusing to overwrite {outdir} (every '
                               'execution needs a NEW output directory)')
        outdir.mkdir(parents=True, exist_ok=True)
        t_cell = time.perf_counter()
        records, evidences, n_err = [], [], 0
        try:
            if spec['kind'] == 'engine':
                engine = get_engine(spec['engine'])
                params = spec['params']
                variant = spec['engine_variant']
                for q in qids:
                    t1 = time.perf_counter()
                    try:
                        rec, _met, ev = run_one(engine, q, params, variant,
                                                gold_local[q],
                                                predict_gated=predict_gated)
                        rec['protocol'] = protocol
                        rec['fold'] = fold
                        rec['db_id'] = db
                        rec['variant'] = name
                        rec['query_time'] = time.perf_counter() - t1
                        if rec['query_time'] > PER_QUESTION_TIMEOUT_S:
                            rec['status'] = 'timeout'
                            n_err += 1
                        records.append(rec)
                        evidences.append(ev)
                    except Exception as exc:  # per-question isolation
                        traceback.print_exc()
                        n_err += 1
                        records.append(error_record(protocol, fold, db, q,
                                                    name, exc,
                                                    time.perf_counter() - t1))
                        evidences.append({'qid': q,
                                          'error': str(exc)[:300]})
            else:
                records, n_err = run_baseline_cell(
                    name, protocol, fold, db, qids, qtext, train_df,
                    field_csv, table_csv, cache_root,
                    allow_network=allow_network,
                    budget_cap_cny=budget_cap_cny,
                    minilm_cache_dir=minilm_cache_dir)
        except Exception as exc:  # construction failure -> all-error cell
            traceback.print_exc()
            records = [error_record(protocol, fold, db, q, name, exc)
                       for q in qids]
            evidences = [{'qid': q, 'error': str(exc)[:300]} for q in qids]
            n_err = len(qids)
        with (outdir / 'predictions.jsonl').open('w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str)
                        + '\n')
        with (outdir / 'execution_evidence.jsonl').open('w',
                                                        encoding='utf-8') as f:
            for e in evidences:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + '\n')
        status = ('failed' if n_err == len(qids)
                  else 'completed_with_errors' if n_err else 'ok')
        cell_meta = {'protocol': protocol, 'fold': fold, 'db_id': db,
                     'method': name, 'status': status, 'n': len(records),
                     'n_error': n_err,
                     'wall_seconds': time.perf_counter() - t_cell,
                     'engine_kind': (spec.get('engine')
                                     if spec['kind'] == 'engine' else None),
                     'engine_variant': spec.get('engine_variant'),
                     'config_sha256': digest(
                         {'parameters': spec.get('params'),
                          'variant': spec.get('engine_variant', name)})}
        (outdir / 'cell.json').write_text(
            json.dumps(cell_meta, ensure_ascii=False, indent=1) + '\n',
            encoding='utf-8')
        statuses.append(cell_meta)
        print(f'{protocol} f{fold} {db} {name}: n={len(qids)} err={n_err}',
              flush=True)


def run_baseline_cell(name, protocol, fold, db, qids, qtext, train_df,
                      field_csv, table_csv, cache_root, allow_network,
                      budget_cap_cny, minilm_cache_dir):
    """Offline/model/API baselines on one cell (frozen adapter wiring)."""
    cache_root = Path(cache_root)
    cache = cache_root / f'{protocol}_f{fold}_{db}' / 'baselines'
    cache.mkdir(parents=True, exist_ok=True)
    if name in ('Dense', 'DenseDeepSeek', 'KaSLA-Adapted',
                'LinkAlign-SingleDB-Adapted'):
        if not allow_network:
            raise RuntimeError(
                f'{name} requires a model download or the DeepSeek API: '
                'pass --allow-network (and set DEEPSEEK_API_KEY + '
                '--budget-cap-cny for API methods)')
    if name in ('DenseDeepSeek', 'KaSLA-Adapted',
                'LinkAlign-SingleDB-Adapted'):
        return _run_api_cell(name, protocol, fold, db, qids, qtext,
                             train_df, field_csv, table_csv, cache_root,
                             cache, allow_network, budget_cap_cny,
                             minilm_cache_dir)
    from terag.engines.bird.baselines import run_baseline
    from terag.engines.bird.config_factory import make_bird_config
    cfg_b = make_bird_config(db, schema_dir=Path(field_csv).parent,
                             artifacts_dir=cache)
    records = run_baseline(name, cfg_b, db, train_df, qids, qtext,
                           field_csv, table_csv, cache_root=cache_root)
    n_err = 0
    for r in records:
        r['protocol'] = protocol
        r['fold'] = fold
        n_err += r.get('status') != 'ok'
    return records, n_err


def _run_api_cell(name, protocol, fold, db, qids, qtext, train_df,
                  field_csv, table_csv, cache_root, cache, allow_network,
                  budget_cap_cny, minilm_cache_dir):
    """One DeepSeek API method on one cell (frozen research call semantics:
    adapter fit, per-question retrieve, budget check, error records in the
    denominator)."""
    import time as _time
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise RuntimeError('DEEPSEEK_API_KEY is not set (env-only; '
                           'there is no file fallback)')
    if budget_cap_cny is None:
        raise RuntimeError('API methods need an explicit user budget cap '
                           '(--budget-cap-cny or TERAG_BIRD_BUDGET_CNY); '
                           'the historical research budget is never '
                           'reused')
    from terag.engines.api import config as api_config
    api_config.configure(run_dir=cache.parent.parent,
                         budget_cap_cny=budget_cap_cny,
                         minilm_cache_dir=minilm_cache_dir,
                         allow_download=allow_network)
    unit_label = f'q_fold{fold}_{db}'
    if name == 'DenseDeepSeek':
        from terag.engines.api.densedeepseek_bird import DenseDeepSeekBird
        adapter = DenseDeepSeekBird(field_csv, table_csv,
                                    unit_label=unit_label)
        adapter.fit(None)
    elif name == 'KaSLA-Adapted':
        from terag.engines.api.kasla_bird import KaSLABird
        adapter = KaSLABird(field_csv, table_csv, unit_label=unit_label)
        adapter.fit(train_df)  # fold-trained classifiers + capacities
    elif name == 'LinkAlign-SingleDB-Adapted':
        from terag.engines.api.linkalign_bird import LinkAlignBird
        adapter = LinkAlignBird(field_csv, table_csv, unit_label=unit_label,
                                cache_dir=api_config.RUNTIME.dense_cache_dir)
        adapter.fit(None)
    else:
        raise ValueError(name)

    def strip_c(key):
        return key[2:] if str(key).startswith('C:') else str(key)

    records, n_err = [], 0
    for qid in qids:
        t1 = _time.perf_counter()
        adapter.api_qid = qid
        adapter.calls = []
        try:
            results = adapter.retrieve(str(qtext[qid]), k=5)
            if len(results) > 5 or any(len(r.columns) > 10
                                       for r in results):
                raise RuntimeError(
                    f'budget violation: {len(results)} tables / per-table '
                    f'{[len(r.columns) for r in results]}')
            detail = [{'table': r.table, 'table_score': float(r.table_score),
                       'columns': [{'key': strip_c(c), 'score': float(s)}
                                   for c, s in r.columns]}
                      for r in results]
            keys = []
            for r in results:
                for c, _s in r.columns:
                    k = strip_c(c)
                    if k not in keys:
                        keys.append(k)
            records.append({'protocol': protocol, 'fold': fold,
                            'db_id': db, 'qid': qid, 'variant': name,
                            'status': 'ok',
                            'retrieved_tables': [r.table for r in results],
                            'retrieved_column_keys': keys,
                            'retrieved_columns_detail': detail,
                            'query_time': _time.perf_counter() - t1})
        except Exception as exc:  # per-question isolation
            n_err += 1
            records.append(error_record(protocol, fold, db, qid, name, exc,
                                        _time.perf_counter() - t1))
    return records, n_err


# ---------------------------------------------------------------- evaluation
def evaluate_run(run_dir, methods, protocols, statuses):
    """Population-exact evaluation of every method (gates + metrics), same
    semantics as the frozen evaluator: qid sets must equal the manifests',
    budgets and db scope enforced, errors zero-scored but kept."""
    from terag.engines.evalkit.metrics_core import (METRIC_KEYS,
                                                    actual_counts,
                                                    gate_problems,
                                                    metric_values)
    from terag.engines.evalkit.population import BirdPopulation

    pop = BirdPopulation()
    raw_gold = pop.gold()
    problems_all = []
    for name in sorted(methods):
        ran = {p for p in protocols
               if (run_dir / p / 'fold0').exists()
               or any((run_dir / p / f'fold{k}').exists()
                      for k in range(5))}
        for protocol in sorted(ran):
            records_by_cell = {}
            for cell in pop.cells(protocol):
                path = (run_dir / protocol / f'fold{cell["fold"]}' / cell['db']
                        / name / 'predictions.jsonl')
                if not path.is_file():
                    problems_all.append(
                        f'{protocol}/{name}: missing cell {cell["db"]} '
                        f'f{cell["fold"]}')
                    continue
                records = [json.loads(l) for l in
                           path.read_text(encoding='utf-8').splitlines()
                           if l.strip()]
                records_by_cell[(protocol, cell['fold'], cell['db'])] = records
                for p in gate_problems(records, cell['test_qids'],
                                       pop.schema_tables(cell['db'])):
                    problems_all.append(
                        f'{protocol} f{cell["fold"]} {cell["db"]}/{name}: {p}')
            if not records_by_cell:
                continue
            rows = []
            for (protocol_, fold, db), records in sorted(
                    records_by_cell.items()):
                for r in records:
                    qid = r['qid']
                    g = raw_gold[qid]
                    m = metric_values(r, {'gold_tables': g['gold_tables'],
                                          'gold_columns': g['gold_columns']})
                    n_t, n_c, per_table = actual_counts(r)
                    rows.append({'protocol': protocol_, 'fold': fold,
                                 'db': db, 'qid': qid,
                                 'status': r.get('status'), 'n_tables': n_t,
                                 'n_columns': n_c,
                                 'max_columns_per_table':
                                     max(per_table.values())
                                     if per_table else 0, **m})
            out = run_dir / 'metrics' / name
            out.mkdir(parents=True, exist_ok=True)
            cols = ['protocol', 'fold', 'db', 'qid', 'status', 'n_tables',
                    'n_columns', 'max_columns_per_table'] + METRIC_KEYS
            lines = [','.join(cols)]
            for r in rows:
                lines.append(','.join(
                    str(r[c]) if not isinstance(r[c], float)
                    else f'{r[c]:.10f}' for c in cols))
            (out / f'{protocol}_per_question.csv').write_text(
                '\n'.join(lines) + '\n', encoding='utf-8')
            summary = {'method': name, 'protocol': protocol,
                       'n': len(rows),
                       'n_error': sum(1 for r in rows if r['status'] != 'ok')}
            for k in METRIC_KEYS:
                summary[k] = (sum(r[k] for r in rows) / len(rows)
                              if rows else 0.0)
            (out / f'{protocol}_summary.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=1) + '\n',
                encoding='utf-8')
            statuses.append({'evaluation': f'{protocol}/{name}', 'n': len(rows),
                             'n_error': summary['n_error'],
                             'covered@5': round(summary['covered@5'] * 100, 4),
                             'colf1': round(summary['colf1'] * 100, 4),
                             'fgcs': round(summary['fgcs'] * 100, 4)})
    return problems_all


# ---------------------------------------------------------------- entry
def run_bird(methods=None, experiment_methods=None, protocols=None,
             output=None, fold=None, allow_network=False,
             budget_cap_cny=None, minilm_cache_dir=None):
    require_hashseed()
    resolved = resolve_methods(methods, experiment_methods)
    protocols = list(protocols) if protocols else list(PROTOCOLS)
    run_dir = Path(output)
    reporting.require_new_directory(run_dir)

    network = {n for n in resolved if method_requires_network(n)}
    if network and not allow_network:
        raise RuntimeError(f'methods {sorted(network)} require a model '
                           'download or the DeepSeek API; pass --allow-network')

    statuses = []
    cache_root = run_dir / 'engine_cache'
    t0 = time.perf_counter()
    planned = []
    for protocol, cell in cell_iter(protocols, fold):
        cell_methods = {}
        for name, spec in resolved.items():
            if spec['kind'] == 'engine':
                cell_methods[name] = spec
            elif protocol in BASELINE_PROTOCOLS.get(name, ()):
                cell_methods[name] = spec
        if cell_methods:
            planned.append({'protocol': protocol, 'fold': cell['fold'],
                            'db': cell['db'],
                            'methods': sorted(cell_methods)})
        run_cell_methods(protocol, cell, cell_methods, run_dir, cache_root,
                         allow_network, budget_cap_cny, minilm_cache_dir,
                         statuses)
    problems = evaluate_run(run_dir, resolved, protocols, statuses)

    run_status = 'completed'
    cell_statuses = [s['status'] for s in statuses if 'status' in s]
    if any(st != 'ok' for st in cell_statuses):
        run_status = ('failed' if cell_statuses
                      and all(st == 'failed' for st in cell_statuses)
                      else 'completed_with_failures')
    if problems:
        run_status = 'failed'
    meta = {'run_kind': 'bird',
            'run_status': run_status,
            'methods': {n: {'group': s['group'],
                            'engine': s.get('engine'),
                            'engine_variant': s.get('engine_variant'),
                            'params_sha256': digest(s.get('params'))}
                        for n, s in resolved.items()},
            'frozen_config_sha256': sha256_file(FROZEN_CONFIG_PATH),
            'frozen_families_sha256': sha256_file(FROZEN_FAMILIES_PATH),
            'data_identity': identity_report(),
            'planned_units': planned,
            'statuses': statuses,
            'evaluation_gate_problems': problems,
            'wall_seconds': time.perf_counter() - t0,
            'environment': {'python': sys.version.split()[0],
                            'PYTHONHASHSEED': os.environ.get(
                                'PYTHONHASHSEED')},
            'allow_network': allow_network,
            'budget_cap_cny': budget_cap_cny}
    (run_dir / 'run_meta').mkdir(parents=True, exist_ok=True)
    reporting.write_json_exclusive(run_dir / 'run_meta' / 'run_meta.json',
                                   meta)
    print(f"bird run '{run_status}': {run_dir}")
    return meta
