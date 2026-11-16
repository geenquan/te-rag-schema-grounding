"""
Term–Schema Cooccurrence retrieval.

Baseline B. Exact identity: accumulate cooccurrence statistics between
tokens of training questions and gold tables/fields, and score by
conditional probability. Serves as the "simple annotation enhancement"
control — no Phrase Alignment, no template mining, no template transfer,
no graph propagation.

Supervision source: only (question, table, field) from the current
training fold. It never reads test data or global artifacts. Token-level
evidence (token→schema weights and the source training qids) is stored in
metadata['evidence'] of the retrieval result and persisted per question
for inspection.

Scoring formulas (fixed; this file is the implementation):
- For each training question take its deduplicated token set (repeated
  tokens within one question are counted once); c(t,T) = number of
  training questions containing token t whose gold table is T; c(t) =
  number of questions containing t; c(t,C) = number of questions
  containing t that referenced field C under table T (C written as
  "T.C").
- Table evidence score E(T|q) = Σ_{t ∈ tokens(q)} P(T|t), where
  P(T|t) = c(t,T)/c(t).
- Field evidence score E(C|t) likewise: Σ_t c(t,C)/c(t).
- Ranking key = (E(T|q) desc, BM25 score desc, table name asc):
  cooccurrence evidence first, zero-evidence tables fall back to the
  plain-schema BM25 (reusing retrievers/bm25_retriever.py), ties broken
  stably by table name (gold is never used). With zero evidence (no
  known token in the query) it degrades to pure BM25, so unseen tables
  are never structurally excluded.
- Per-table column budget of 10: cooccurrence fields first (E desc), the
  rest filled by query-term/field overlap (same field scoring as
  bm25_retriever), deduplicated, and fewer are returned if the budget
  cannot be filled.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig
from retrievers.bm25_retriever import BM25Retriever
from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields

COLUMN_BUDGET = 10        # fixed: per-table column budget (unified protocol)
BM25_RANK_DEPTH = 20      # pre-registered: BM25 list depth for RRF (separate from the final k)

from retrievers.example_rag_retriever import _rrf_rank  # same pre-registered corrected rule


def _rrf_rank_co(candidates, evidence_scores, bm25_order):
    """Same rule as example_rag._rrf_rank (only positive evidence is
    ranked + independent BM25 depth)."""
    return _rrf_rank(candidates, evidence_scores, bm25_order)


class CooccurrenceRetriever(BaseRetriever):
    """Term–Schema Cooccurrence (see module docstring)."""

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='Cooccurrence',
                description='Term–Schema Cooccurrence (cooccurrence conditional probability, simple annotation control)')
        super().__init__(field_csv, table_csv, config)

        self._fallback = BM25Retriever(field_csv, table_csv)

        # Statistics populated by fit
        self.token_total: Dict[str, int] = {}                 # c(t)
        self.token_table: Dict[str, Dict[str, int]] = {}      # t -> {T: c(t,T)}
        self.token_field: Dict[str, Dict[str, int]] = {}      # t -> {"T.C": c(t,C)}
        self.token_qids: Dict[str, List[str]] = {}            # t -> source training qids
        self.train_qids: List[str] = []

        # Fusion mode: evidence_first (original) | rank_equal (pre-registered RRF)
        self.fusion_mode = ((config.extra_params or {}).get('fusion_mode')
                            if config else None) or 'evidence_first'

    def _rank_candidates(self, candidates, evidence_scores, bm25_order, bm25_scores):
        """Same pre-registered rule as ExampleRAG._rank_candidates
        (rank_equal_v2)."""
        if self.fusion_mode == 'rank_equal':
            return _rrf_rank_co(candidates, evidence_scores, bm25_order)
        ordered = sorted(candidates,
                         key=lambda t: (-evidence_scores.get(t, 0.0),
                                        -bm25_scores.get(t, 0.0), t))
        scores = {t: (evidence_scores.get(t, 0.0) or bm25_scores.get(t, 0.0))
                  for t in candidates}
        return ordered, scores

    # ---------------- fit ----------------

    def fit(self, train_data: pd.DataFrame = None):
        if train_data is None or len(train_data) == 0:
            self._fallback.fit(train_data)
            self._is_fitted = True
            return

        self.train_qids = (
            [str(q) for q in train_data['qid'].tolist()]
            if 'qid' in train_data.columns
            else [f"ex{i:03d}" for i in range(len(train_data))])

        for i, (_, row) in enumerate(train_data.iterrows()):
            tokens = {t for t in self.tokenize(str(row['question'])) if len(t) > 1}
            tables = {t.split('.')[-1] for t in parse_ground_truth_tables(row)}
            fields = {f"{t.split('.')[-1]}.{f}"
                      for t, f in parse_ground_truth_fields(row)}
            qid = self.train_qids[i]
            for t in tokens:  # deduplicated within the question: each token counts once per question
                self.token_total[t] = self.token_total.get(t, 0) + 1
                tt = self.token_table.setdefault(t, {})
                for T in tables:
                    tt[T] = tt.get(T, 0) + 1
                tf = self.token_field.setdefault(t, {})
                for C in fields:
                    tf[C] = tf.get(C, 0) + 1
                self.token_qids.setdefault(t, []).append(qid)

        self._fallback.fit(train_data)
        self._is_fitted = True

    # ---------------- retrieve ----------------

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        depth = BM25_RANK_DEPTH if self.fusion_mode == 'rank_equal' else k
        fallback_results = self._fallback.retrieve(query, k=depth)
        bm25_score = {r.table: float(r.table_score) for r in fallback_results}
        bm25_order = [r.table for r in fallback_results]

        tokens = [t for t in self.tokenize(query) if len(t) > 1]
        table_evidence: Dict[str, float] = defaultdict(float)
        field_evidence: Dict[str, float] = defaultdict(float)
        token_ev = {}
        for t in tokens:
            c = self.token_total.get(t)
            if not c:
                continue
            p_tables = {T: cnt / c for T, cnt in self.token_table.get(t, {}).items()}
            p_fields = {C: cnt / c for C, cnt in self.token_field.get(t, {}).items()}
            for T, p in p_tables.items():
                table_evidence[T] += p
            for C, p in p_fields.items():
                field_evidence[C] += p
            token_ev[t] = {
                'count': c,
                'top_tables': dict(sorted(p_tables.items(), key=lambda x: (-x[1], x[0]))[:5]),
                'qids': self.token_qids.get(t, [])[:20],
            }

        candidates = set(table_evidence) | set(bm25_order)
        ordered, rank_scores = self._rank_candidates(
            candidates, table_evidence, bm25_order, bm25_score)

        evidence = {
            'tokens': dict(sorted(token_ev.items(), key=lambda x: -x[1]['count'])[:10]),
            'table_evidence': {T: round(v, 6) for T, v in table_evidence.items()},
            'train_qid_pool_size': len(self.train_qids),
            'fusion_mode': ('rank_equal_v2' if self.fusion_mode == 'rank_equal'
                            else self.fusion_mode),
        }

        results = []
        for table in ordered[:k]:
            columns = self._columns_for(query, table, field_evidence)
            results.append(RetrievalResult(
                table=table,
                table_score=rank_scores.get(
                    table, table_evidence.get(table, bm25_score.get(table, 0.0))),
                columns=columns,
                metadata={'method': 'Cooccurrence', 'evidence': evidence},
            ))
        return results

    def _columns_for(self, query: str, table: str,
                     field_evidence: Dict[str, float]) -> List[Tuple[str, float]]:
        schema_cols = self.field_df[self.field_df['table'] == table]
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        scored = []
        for _, row in schema_cols.iterrows():
            fname = str(row['field_name'])
            fdesc = str(row.get('field_name_desc', '') or '')
            ev = field_evidence.get(f"{table}.{fname}", 0.0)
            overlap = len(query_term_set & set(
                t for t in self.tokenize(f"{fname} {fdesc}") if len(t) > 1))
            overlap_score = overlap / len(query_terms) if query_terms else 0.0
            if any(t in fname for t in query_term_set):
                overlap_score += 0.5
            scored.append((fname, ev, overlap_score))

        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [(f"C:{table}.{f}", e + o) for f, e, o in scored[:COLUMN_BUDGET]]
