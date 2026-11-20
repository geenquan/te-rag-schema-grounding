"""Lightweight supervised field-confidence model (variant 4, optional).

Logistic / HistGradientBoosting over frozen-engine trace features + schema /
train-side attributes. Labels come from Train opt-train question gold columns
ONLY (Query-SQL derived); negatives are the non-gold candidate columns of
those same questions' engine candidate sets. No cross-question candidates are
generated, so no cross-fitting is required (protocol V2 direction B).

Honest caveat recorded in identity(): opt-train questions participate in the
engine fit (graph/patterns/B mined on them), so train-question features carry
self-retrieval inflation (e.g. train_recommend); thresholds are therefore
re-tuned on Train opt-DEV and the model is never fitted on Dev data.
"""
import math
import time

import numpy as np

FEATURES = [
    # table-level (inference-visible)
    'log_tscore', 'tscore_ratio', 'table_bm25', 'table_graph', 'table_pattern',
    'log_width', 'table_rank', 'is_top_table',
    # column-level engine channels
    'raw_total', 'direct_match', 'graph_prop', 'role_prior', 'train_recommend',
    'BHit', 'CHit', 'name_only', 'rank_norm', 'ratio_top',
    # schema / train-side
    'log_degree', 'log_tcount', 'pk_like',
    'type_text', 'type_numeric', 'type_date', 'type_other',
]


def _type_flags(ftype):
    t = (ftype or '').lower()
    return (float('char' in t or 'text' in t or 'clob' in t),
            float(t in ('int', 'integer', 'real', 'float', 'double', 'numeric',
                        'decimal', 'number', 'bigint', 'smallint')),
            float('date' in t or 'time' in t or 'year' in t),
            float(not (('char' in t or 'text' in t or 'clob' in t)
                       or t in ('int', 'integer', 'real', 'float', 'double',
                                'numeric', 'decimal', 'number', 'bigint', 'smallint')
                       or ('date' in t or 'time' in t or 'year' in t))))


def column_features(q, meta, max_cols=10):
    """Feature rows for the engine-returned (top max_cols) columns per table.

    Returns list of (table, field, feature_list) in engine order.
    """
    from terag.engines.bird_runtime.eval.retrieval_metrics import normalize_column_key
    rows = []
    tscore_top = max((t.table_score for t in q.tables), default=0.)
    for rank, t in enumerate(q.tables):
        cols = t.columns
        top = cols[0].raw_total if cols else 0.
        n = len(cols)
        ch = t.table_channels
        for pos, c in enumerate(cols[:max_cols]):
            key = f'{t.table}.{c.field}'
            fld = None
            for x in (meta.get('fields', {}).get(t.table) or []):
                if x['field'] == c.field:
                    fld = x
                    break
            ftype = fld['type'] if fld else ''
            tt, tn, td, to = _type_flags(ftype)
            rows.append((t.table, c.field, [
                math.log1p(max(t.table_score, 0.)),
                (t.table_score / tscore_top) if tscore_top > 0 else 0.,
                ch.get('bm25_score', 0.), ch.get('graph_score', 0.),
                ch.get('pattern_score', 0.),
                math.log1p(n), rank / 4., 1. if rank == 0 else 0.,
                c.raw_total, c.direct_match, c.graph_propagation, c.role_prior,
                c.train_recommend, c.BHit, c.CHit, c.name_only,
                1. if n <= 1 else 1. - pos / (n - 1) if pos < max_cols else 0.,
                (c.raw_total / top) if top > 0 else 0.,
                math.log1p(meta.get('degree', {}).get(key, 0)),
                math.log1p(meta.get('train_count', {}).get(key, 0)),
                1. if (fld and fld.get('pk_like')) else 0.,
                tt, tn, td, to]))
    return rows


def labels_for(q):
    from terag.engines.bird_runtime.eval.retrieval_metrics import normalize_column_key
    return {normalize_column_key(k) for k in q.gold_columns}


class SupervisedFieldModel:
    """Fitted P(gold | column features). fit() on opt-train questions only."""

    def __init__(self, kind='gbdt', seed=42):
        if kind not in ('logit', 'gbdt'):
            raise ValueError('kind must be logit or gbdt')
        self.kind = kind
        self.seed = seed
        self.model = None
        self.fit_stats = {}

    def fit(self, questions, metas, max_cols=10):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        X, y = [], []
        for q in questions:
            gset = labels_for(q)
            meta = metas.get(q.db_id, {})
            for table, field, feats in column_features(q, meta, max_cols):
                X.append(feats)
                y.append(1.0 if normalize(table, field) in gset else 0.0)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        t0 = time.perf_counter()
        if self.kind == 'logit':
            self.model = LogisticRegression(max_iter=3000, C=1.0, random_state=self.seed)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=150, max_leaf_nodes=31, learning_rate=0.1,
                random_state=self.seed)
        self.model.fit(X, y)
        self.fit_stats = {
            'kind': self.kind, 'n_rows': int(len(y)), 'n_positive': int(y.sum()),
            'positive_rate': float(y.mean()), 'n_features': len(FEATURES),
            'fit_seconds': round(time.perf_counter() - t0, 3),
            'trained_on': 'Train opt-train questions (per-db snapshots)',
            'labels': 'gold columns of opt-train questions (Train Query-SQL only)',
            'negatives': 'non-gold candidate columns of the same questions',
            'cross_question_candidates': False,
            'caveat': 'self-retrieval inflation on train-question history channels; '
                      'thresholds re-tuned on Train opt-dev, model never fit on Dev',
        }
        return self

    def predict_conf(self, q, meta, max_cols=10):
        """Per-table {field: P(gold)} for the engine top-max_cols columns."""
        rows = column_features(q, meta, max_cols)
        if not rows:
            return {}
        X = np.asarray([r[2] for r in rows], dtype=float)
        p = self.model.predict_proba(X)[:, 1]
        out = {}
        for (table, field, _), pi in zip(rows, p):
            out.setdefault(table, {})[field] = float(pi)
        return out

    def identity(self):
        return {'model': self.kind, 'features': FEATURES, 'fit_stats': self.fit_stats}


def normalize(table, field):
    from terag.engines.bird_runtime.eval.retrieval_metrics import normalize_column_key
    return normalize_column_key(f'{table}.{field}')
