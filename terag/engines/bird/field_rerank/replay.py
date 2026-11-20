"""Offline replay of the frozen engine's field-axis scoring from snapshots.

Replicates bird_engine.py predict() field arithmetic EXACTLY (same operation
order, same tie-breaks) for the axes that only touch column scoring inside
the fixed 5-table candidate set:

  field_weights[4], b_hit_weight, c_hit_weight, direct_scale,
  description_scale, field_normalization, field_graph_norm,
  field_pattern_norm

Source of truth (byte-frozen engine/bird_engine.py, predict):
  :607-611 description/direct scaling of v['direct_match']
  :612-614 fg/pat transform() over graph_propagation / role_prior
  :615-619 field_normalization=='max' within-table division
  :623    raw_scores = sum(w*v[k]) + BHit + CHit
  :627    ordered by (-raw, field)
transform() below is a semantic copy of bird_engine.transform (lines 96-110).

With AXES at engine DEFAULT the replayed raw score is bit-identical to the
saved raw_total (enforced by tests). Metrics come from the frozen
eval.full_gold_metrics — no redefinition.
"""
import math
from pathlib import Path

from terag.engines.bird_runtime.eval.retrieval_metrics import normalize_column_key

ENGINE_DEFAULT_AXES = {
    'field_weights': [.40, .20, .15, .25],
    'b_hit_weight': 1., 'c_hit_weight': 1.,
    'direct_scale': 1., 'description_scale': 1.,
    'field_normalization': 'raw', 'field_graph_norm': 'none', 'field_pattern_norm': 'none',
}
AXES_KEYS = tuple(ENGINE_DEFAULT_AXES)

FIELD_NAMES = ['direct_match', 'graph_propagation', 'role_prior', 'train_recommend']


def _clip01_ok(mode, clip):
    return mode in ('none', 'raw', 'max', 'minmax', 'zscore', 'rank')


def transform_scores(values, mode, clip=1000000.):
    """Semantic copy of engine/bird_engine.py transform() for dict inputs."""
    out = {k: min(max(0., float(v)), clip) for k, v in values.items()}
    positive = [v for v in out.values() if v > 0]
    if not positive or mode in ['none', 'raw']:
        return out
    if mode in ['max', 'minmax']:
        lo = min(positive) if mode == 'minmax' else 0.
        hi = max(positive)
        return {k: ((v - lo) / (hi - lo) if hi > lo else 1.) if v > 0 else 0. for k, v in out.items()}
    if mode == 'zscore':
        n = len(positive)
        mean = math.fsum(positive) / n
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in positive) / n)
        return {k: max(0., (v - mean) / sd) if v > 0 and sd > 1e-12 else (1. if v > 0 else 0.)
                for k, v in out.items()}
    if mode == 'rank':
        unique = sorted(set(positive))
        ranks = {v: (i + 1) / len(unique) for i, v in enumerate(unique)}
        return {k: ranks[v] if v > 0 else 0. for k, v in out.items()}
    raise ValueError('Unknown evidence normalization: ' + mode)


class ColSnap:
    __slots__ = ('field', 'raw_total', 'direct_match', 'graph_propagation',
                 'role_prior', 'train_recommend', 'BHit', 'CHit', 'name_only')

    def __init__(self, d):
        self.field = d['field']
        self.raw_total = float.fromhex(d['raw_total'])
        f = d['feats']
        self.direct_match = float.fromhex(f['direct_match'])
        self.graph_propagation = float.fromhex(f['graph_propagation'])
        self.role_prior = float.fromhex(f['role_prior'])
        self.train_recommend = float.fromhex(f['train_recommend'])
        self.BHit = float.fromhex(f['BHit'])
        self.CHit = float.fromhex(f['CHit'])
        self.name_only = float.fromhex(d['name_only'])


class TableSnap:
    __slots__ = ('table', 'table_score', 'table_channels', 'columns', 'engine_top10')

    def __init__(self, d):
        self.table = d['table']
        self.table_score = float.fromhex(d['table_score'])
        ch = d.get('table_channels') or {}
        self.table_channels = {k: float.fromhex(v) for k, v in ch.items()}
        self.columns = [ColSnap(c) for c in d['columns']]
        self.engine_top10 = list(d['engine_top10'])


class SnapshotQuestion:
    __slots__ = ('qid', 'db_id', 'valid', 'gold_status', 'gold_tables',
                 'gold_columns', 'question', 'tables', 'engine_record_keys',
                 'engine_metric')

    def __init__(self, d):
        self.qid = d['qid']
        self.db_id = d['db_id']
        self.valid = bool(d['valid'])
        self.gold_status = d['gold_status']
        self.gold_tables = list(d['gold_tables'])
        self.gold_columns = list(d['gold_columns'])
        self.question = d.get('question', '')
        self.tables = [TableSnap(t) for t in d['tables']]
        self.engine_record_keys = list(d['engine_record_keys'])
        self.engine_metric = dict(d['engine_metric'])

    def gold_local(self):
        return {'gold_status': self.gold_status, 'gold_tables': self.gold_tables,
                'gold_columns': self.gold_columns}


def merge_axes(overrides=None):
    axes = dict(ENGINE_DEFAULT_AXES)
    for k, v in (overrides or {}).items():
        if k not in axes:
            raise KeyError(f'unknown field axis: {k}')
        axes[k] = v
    if len(axes['field_weights']) != 4:
        raise ValueError('field_weights must have 4 entries')
    return axes


def replay_table_scores(tsnap, axes):
    """Exact replay of predict() column scores for one table.

    Returns {field: raw_score} with engine-identical float arithmetic:
    sum(w*v[k] for k,w in zip(FIELD_NAMES, fw)) + BHit*bhw + CHit*chw,
    after the engine-ordered scaling/normalization pipeline.
    """
    ds = axes['description_scale']
    dsc = axes['direct_scale']
    fw = list(axes['field_weights'])
    bhw = axes['b_hit_weight']
    chw = axes['c_hit_weight']
    cols = tsnap.columns
    direct = {}
    for c in cols:
        dm = c.direct_match
        if ds != 1.:
            dm = c.name_only + ds * (dm - c.name_only)
        direct[c.field] = dm * dsc
    graph = transform_scores({c.field: c.graph_propagation for c in cols},
                             axes['field_graph_norm'])
    pat = transform_scores({c.field: c.role_prior for c in cols},
                           axes['field_pattern_norm'])
    if axes['field_normalization'] == 'max':
        maximum = {}
        for name, values in (('direct_match', direct), ('graph_propagation', graph),
                             ('role_prior', pat),
                             ('train_recommend', {c.field: c.train_recommend for c in cols})):
            m = max(values.values(), default=0.) or 1.
            maximum[name] = m
        direct = {f: v / maximum['direct_match'] for f, v in direct.items()}
        graph = {f: v / maximum['graph_propagation'] for f, v in graph.items()}
        pat = {f: v / maximum['role_prior'] for f, v in pat.items()}
        train = {c.field: c.train_recommend / maximum['train_recommend'] for c in cols}
    else:
        train = {c.field: c.train_recommend for c in cols}
    raw = {}
    for c in cols:
        v = {'direct_match': direct[c.field], 'graph_propagation': graph[c.field],
             'role_prior': pat[c.field], 'train_recommend': train[c.field]}
        score = sum(w * v[k] for k, w in zip(FIELD_NAMES, fw)) + c.BHit * bhw + c.CHit * chw
        raw[c.field] = score
    return raw


def order_fields(raw):
    """Engine tie-break: ascending field name on equal scores (predict :627)."""
    return sorted(raw, key=lambda f: (-raw[f], f))


def normalize_key_of(table):
    """Simple table-name normalization for internal map keys."""
    return str(table).strip().lower()


def build_record(q, entries, variant='field_layer'):
    """Record in the frozen engine's shape.

    entries: [(TableSnap, [field, ...])] in output order; <=5 tables,
    <=10 columns each (enforced).
    """
    details = []
    keys = []
    for tsnap, cols in entries:
        details.append({'table': tsnap.table, 'table_score': tsnap.table_score,
                        'columns': [{'key': normalize_column_key(tsnap.table + '.' + f)}
                                    for f in cols]})
        keys.extend(normalize_column_key(tsnap.table + '.' + f) for f in cols)
    if len(details) > 5 or any(len(t['columns']) > 10 for t in details):
        raise RuntimeError('Budget expanded')
    return {'qid': q.qid, 'fold': None, 'setting': None, 'variant': variant,
            'status': 'ok',
            'retrieved_tables': [t['table'] for t in details],
            'retrieved_column_keys': keys,
            'retrieved_columns_detail': details}


def replay_record(q, axes_overrides=None, ks=10):
    """Control-equivalent replay: order by replayed raw, top-ks per table."""
    axes = merge_axes(axes_overrides)
    entries = []
    for t in q.tables:
        raw = replay_table_scores(t, axes)
        entries.append((t, order_fields(raw)[:ks]))
    return build_record(q, entries)


def record_metrics(record, q):
    """Frozen metric path on a replayed record (eval.full_gold_metrics)."""
    from eval.full_gold import full_gold_metrics
    m = full_gold_metrics(record, {record['qid']: q.gold_local()})
    return {'covered@5': float(m[1]), 'colp': float(m[2]), 'colrec': float(m[3]),
            'colf1': float(m[4]), 'fgcs': float(m[5])}
