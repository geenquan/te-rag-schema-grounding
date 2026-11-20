"""Field-confidence FILTER variant (variant 3, named field_filter).

Tables stay exactly as the engine returned them; per table we return only
columns above a confidence threshold and/or a top-k' cap (k' <= 10). Real
output counts are always reported; no empty-shell-table tricks:

  empty_policy='keep_best'  a table with zero confident columns still appears
                            with its single best-scoring column;
  empty_policy='drop_table' the table is dropped from the record (honest
                            table-count reduction, chosen only on dev
                            evidence and documented).

Confidence semantics (rule key):
  'tau_rel'   keep columns with conf >= tau * (table top conf)  — the
              per-table relative cut (a table never empties below its top);
  'tau_abs'   keep columns with conf >= tau absolutely — can empty a table
              (then empty_policy applies). Used with calibrated confidences
              (supervised P(gold)) or absolute scores.
  'k_cap'     keep the top k columns of the (possibly reranked) order.

Options compose: k_cap caps the kept list; min_k enforces a floor for kept
tables; k_low optionally applies a smaller cap to low-confidence tables
(table_gate feature + threshold) — the table-aware asymmetry that protects
recall in gold-looking tables while cutting columns elsewhere.
"""
from .replay import build_record, normalize_key_of

FILTER_DEFAULTS = {
    'enabled': False,
    'rule': 'k_cap',        # 'k_cap' | 'tau_rel' | 'tau_abs'
    'tau': 0.0,             # threshold for tau_rel / tau_abs
    'k_cap': 10,            # per-table cap (<=10)
    'k_low': None,          # smaller cap for low-confidence tables (or None)
    'table_gate': 'table_score',  # table feature for k_low gating
    'table_gate_op': 'lt',  # apply k_low when feature < tau_t
    'table_gate_tau': 0.0,
    'min_k': 1,
    'empty_policy': 'keep_best',   # or 'drop_table'
}


def merge_filter(overrides=None):
    flt = dict(FILTER_DEFAULTS)
    for k, v in (overrides or {}).items():
        if k not in flt:
            raise KeyError(f'unknown filter key: {k}')
        flt[k] = v
    if flt['k_cap'] > 10:
        raise ValueError('filter k_cap exceeds per-table budget')
    if flt['rule'] not in ('k_cap', 'tau_rel', 'tau_abs'):
        raise ValueError('unknown filter rule')
    if flt['empty_policy'] not in ('keep_best', 'drop_table'):
        raise ValueError('unknown empty_policy')
    return flt


def _table_gate_applies(tsnap, flt, tscore_top=None):
    feat = {'table_score': tsnap.table_score,
            'bm25_score': tsnap.table_channels.get('bm25_score', 0.),
            'graph_score': tsnap.table_channels.get('graph_score', 0.),
            'pattern_score': tsnap.table_channels.get('pattern_score', 0.),
            'tscore_ratio': (tsnap.table_score / tscore_top) if tscore_top else 0.
            }.get(flt['table_gate'], 0.)
    return feat < flt['table_gate_tau'] if flt['table_gate_op'] == 'lt' else \
        feat >= flt['table_gate_tau']


def filter_table(tsnap, ordered, conf, flt, tscore_top=None):
    """Keep list for one table under the active rule (never > k_cap)."""
    if not ordered:
        return []
    k = flt['k_cap']
    if flt['k_low'] is not None and _table_gate_applies(tsnap, flt, tscore_top):
        k = min(k, flt['k_low'])
    if flt['rule'] == 'k_cap':
        kept = ordered[:k]
    elif flt['rule'] == 'tau_rel':
        top = conf.get(ordered[0], 0.) if ordered else 0.
        kept = [f for f in ordered if top <= 0 or conf.get(f, 0.) >= flt['tau'] * top][:k]
    else:  # tau_abs
        kept = [f for f in ordered if conf.get(f, 0.) >= flt['tau']][:k]
    # NOTE: no min_k floor here — a rule-emptied table is handled by the
    # empty_policy in filter_question (keep_best floors to min_k/best column;
    # drop_table drops it honestly).
    return kept


def filter_question(q, per_table, flt_overrides=None):
    """Apply the filter to pipeline output.

    per_table entries: {'table', 'ordered' (ranked fields), 'conf' (dict
    field->confidence, e.g. blend or calibrated P(gold)), 'tsnap'}.
    Returns (record, counts) with honest budget_actual.
    """
    flt = merge_filter(flt_overrides)
    if not flt['enabled']:
        flt = dict(FILTER_DEFAULTS)
    tables_map = {normalize_key_of(t.table): t for t in q.tables}
    tscore_top = max((t.table_score for t in q.tables), default=0.)
    entries = []
    counts = {'tables_in': len(per_table), 'tables_out': 0,
              'columns_by_table': {}, 'dropped_tables': []}
    for tinfo in per_table:
        tsnap = tinfo.get('tsnap') or tables_map[normalize_key_of(tinfo['table'])]
        kept = filter_table(tsnap, tinfo['ordered'], tinfo['conf'], flt, tscore_top) \
            if flt['enabled'] else tinfo['ordered'][:10]
        if not kept:
            if flt['empty_policy'] == 'drop_table':
                counts['dropped_tables'].append(tinfo['table'])
                continue
            kept = tinfo['ordered'][:max(1, flt['min_k'])]
        elif flt['empty_policy'] == 'keep_best' and len(kept) < flt['min_k']:
            kept = tinfo['ordered'][:flt['min_k']]
        entries.append((tsnap, kept))
        counts['tables_out'] += 1
        counts['columns_by_table'][tinfo['table']] = len(kept)
    record = build_record(q, entries,
                          variant='field_filter' if flt['enabled'] else 'field_layer')
    record['budget_actual'] = {'tables': counts['tables_out'],
                               'columns_by_table': counts['columns_by_table'],
                               'dropped_tables': counts['dropped_tables'],
                               'mean_columns_per_q': (sum(counts['columns_by_table'].values())
                                                      / max(len(per_table), 1))}
    return record, counts
