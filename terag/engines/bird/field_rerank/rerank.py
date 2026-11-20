"""Separable field-rerank layer over frozen-engine trace features (variant 2).

The layer consumes ONLY inference-visible per-column features (engine trace
values + schema/train-side attributes from the snapshot meta) and produces an
additive adjustment to the replayed engine score:

  blend = replayed_raw + lambda * sum(w_f * x_f)

Identity config (lambda=0 or all weights 0) reproduces the control ordering
exactly. Weights are fitted on Train opt-train questions only (see
search/q/fit_layer.py); nothing here is fitted on Dev.

Layer features (within-table, monotone transforms):
  ratio_top   raw / max(raw) in table (0 if max<=0)
  rank_norm   1 - position/(n-1), position by control order (1.0 if n==1)
  name_share  name_only / direct_match (0 if direct_match<=0) — description-
              inflation detector
  log_degree  log1p(bipartite-graph column degree)
  log_tcount  log1p(#opt-train questions whose gold selects this column)
  pk_like     1 if field name is 'id' or ends '_id'
  log_width   log1p(#columns of the table)
  log_tscore  log1p(table_score of the candidate table)
  bh_ind      1 if BHit>0 ; ch_ind 1 if CHit>0
"""
import math

from .replay import (build_record, merge_axes, order_fields,
                     replay_table_scores)

LAYER_FEATURES = ['ratio_top', 'rank_norm', 'name_share', 'log_degree',
                  'log_tcount', 'pk_like', 'log_width', 'log_tscore',
                  'bh_ind', 'ch_ind']

RERANK_DEFAULTS = {
    'lambda': 0.0,               # 0 = layer off (identity)
    'weights': {f: 0.0 for f in LAYER_FEATURES},
    'blend_with': 'raw',         # blend base: 'raw' (replayed engine score)
}


def merge_layer(overrides=None):
    layer = {'lambda': RERANK_DEFAULTS['lambda'],
             'weights': dict(RERANK_DEFAULTS['weights']),
             'blend_with': 'raw'}
    for k, v in (overrides or {}).items():
        if k == 'weights':
            for f, w in v.items():
                if f not in LAYER_FEATURES:
                    raise KeyError(f'unknown layer feature: {f}')
                layer['weights'][f] = float(w)
        elif k in layer:
            layer[k] = v
        else:
            raise KeyError(f'unknown layer key: {k}')
    return layer


def _meta_lookup(meta, table):
    fields = (meta or {}).get('fields', {}).get(table, [])
    by_name = {f['field']: f for f in fields}
    tc = (meta or {}).get('train_count', {})
    deg = (meta or {}).get('degree', {})
    return by_name, tc, deg


def layer_features(q, meta):
    """Per-table [{field: {feature: value}}] on the CONTROL ordering.

    Uses the snapshot's control raw scores (DEFAULT axes); the layer is
    deliberately separable from the engine-axis replay.
    """
    out = []
    for t in q.tables:
        t_by_name, t_tc, t_deg = _meta_lookup(meta, t.table)
        raws = {c.field: c.raw_total for c in t.columns}
        top = max(raws.values()) if raws else 0.
        n = len(t.columns)
        order = [c.field for c in t.columns]  # control order (engine ordering)
        pos = {f: i for i, f in enumerate(order)}
        feats = {}
        for c in t.columns:
            fld = t_by_name.get(c.field, {})
            key = f'{t.table}.{c.field}'
            feats[c.field] = {
                'ratio_top': (c.raw_total / top) if top > 0 else 0.,
                'rank_norm': 1. if n <= 1 else 1. - pos[c.field] / (n - 1),
                'name_share': (c.name_only / c.direct_match) if c.direct_match > 0 else 0.,
                'log_degree': math.log1p(t_deg.get(key, 0)),
                'log_tcount': math.log1p(t_tc.get(key, 0)),
                'pk_like': 1. if fld.get('pk_like') else 0.,
                'log_width': math.log1p(n),
                'log_tscore': math.log1p(max(t.table_score, 0.)),
                'bh_ind': 1. if c.BHit > 0 else 0.,
                'ch_ind': 1. if c.CHit > 0 else 0.,
            }
        out.append(feats)
    return out


def blended_scores(tsnap, raw, feats, layer):
    """blend = raw + lambda * w.x ; deterministic float order (sorted feats)."""
    lam = float(layer['lambda'])
    w = layer['weights']
    if lam == 0. or not any(w.values()):
        return dict(raw)
    out = {}
    for f, base in raw.items():
        x = feats.get(f) or {}
        out[f] = base + lam * sum(w[k] * x[k] for k in LAYER_FEATURES if k in x)
    return out


def rerank_question(q, meta, axes_overrides=None, layer_overrides=None):
    """Full variant-2 pipeline: engine-axis replay + separable layer blend.

    Returns (record, per_table) where per_table = [{'table', 'ordered',
    'raw', 'blend', 'conf', 'tsnap'}] for downstream filtering and
    honest-count reporting ('conf' == blend by default; the supervised
    pipeline substitutes calibrated P(gold)).
    """
    axes = merge_axes(axes_overrides)
    layer = merge_layer(layer_overrides)
    feat_tables = layer_features(q, meta) if layer['lambda'] else None
    entries = []
    per_table = []
    for t, feats in zip(q.tables, feat_tables or [None] * len(q.tables)):
        raw = replay_table_scores(t, axes)
        blend = blended_scores(t, raw, feats, layer) if feats else dict(raw)
        fo = order_fields(blend)
        entries.append((t, fo[:10]))
        per_table.append({'table': t.table, 'ordered': fo, 'raw': raw,
                          'blend': blend, 'conf': blend, 'tsnap': t})
    return build_record(q, entries), per_table
