"""
Historical Example Schema Retrieval.

Baseline A. Exact identity: retrieve the historical questions (NLQ→SQL
examples) most similar to the current question, and vote with the
tables/fields they referenced to obtain schema candidates. **There is
currently no SQL generator**: this method performs schema retrieval only
and is not, and must not be described as, a completed end-to-end
Example-RAG.

Supervision source: only (question, table, field) triples from the
current training fold (the train_data passed to fit). It never reads
test SQL, test annotations, or global artifacts.

Processing kept consistent with the TF-IDF baseline: tokenization
(BaseRetriever.tokenize) and the TF-IDF formula (log-TF ×
idf=log(N/(df+1)), L2 normalization, cosine similarity, as in
retrievers/vector_retriever.py) — here applied to historical questions
rather than to schema fields.

Fixed rules (declared up front in the protocol, not chosen by test
performance):
1. Take at most 5 historical examples with similarity > 0 (fixed m=5).
2. Table vote vote(T) = Σ_e sim(e), over examples e whose gold tables
   include T; field vote vote(T.c) = Σ_e sim(e), over examples e that
   referenced field c under table T.
3. Ranking key = (vote desc, BM25 score desc, table name asc):
   historical evidence first, zero-vote tables fall back to the
   plain-schema BM25 ranking (reusing an instance of
   retrievers/bm25_retriever.py); ties are broken stably by table name
   (gold is never used to break ties).
4. With zero similarity (no votes at all) it degrades to pure BM25, so
   unseen tables are never structurally excluded.
5. Per-table column budget of 10: fields voted by examples come first
   (vote desc); the rest are filled with the BM25-style query-term/field
   overlap (same field scoring as bm25_retriever), deduplicated, and
   fewer are returned if the budget cannot be filled.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig
from retrievers.bm25_retriever import BM25Retriever
from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields

MAX_EXAMPLES = 5          # fixed: retrieve at most 5 historical examples
COLUMN_BUDGET = 10        # fixed: per-table column budget (unified protocol)
BM25_RANK_DEPTH = 20      # pre-registered: BM25 list depth for RRF (kept
                          # separate from the final returned-table budget
                          # k=5; fixed value, not chosen by results)


def _rrf_rank(candidates, evidence_scores, bm25_order):
    """Corrected RRF (rank_equal_v2).

    Differences from the earlier rank_equal version (bug fixes, not
    tuning):
    - The historical-evidence list contains only tables with actual
      positive evidence (score>0), ranked by (score desc, table name);
      zero-evidence tables get no rank or contribution on that list (the
      old version ranked every candidate, so zero-evidence tables earned
      a spurious historical contribution that competed with the BM25
      list).
    - The BM25 list uses the independent fixed depth BM25_RANK_DEPTH (the
      old version used depth = the final k).
    Formula unchanged: score(T)=Σ 1/(60+rank_l), a missing list
    contributes 0, ties are broken by table name ascending.
    """
    ev_list = sorted((t for t in candidates if evidence_scores.get(t, 0.0) > 0),
                     key=lambda t: (-evidence_scores.get(t, 0.0), t))
    ev_rank = {t: i + 1 for i, t in enumerate(ev_list)}
    bm_rank = {t: i + 1 for i, t in enumerate(bm25_order)}

    def rrf(t):
        s = 0.0
        if t in ev_rank:
            s += 1.0 / (60 + ev_rank[t])
        if t in bm_rank:
            s += 1.0 / (60 + bm_rank[t])
        return s

    scores = {t: rrf(t) for t in candidates}
    return sorted(candidates, key=lambda t: (-scores[t], t)), scores


class ExampleRAGRetriever(BaseRetriever):
    """Historical Example Schema Retrieval (see module docstring)."""

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='ExampleRAG',
                description='Historical Example Schema Retrieval (historical-example voting, not end-to-end Example-RAG)')
        super().__init__(field_csv, table_csv, config)

        # Plain-schema BM25 fallback (reuse the existing implementation)
        self._fallback = BM25Retriever(field_csv, table_csv)

        # Populated by fit
        self.train_qids: List[str] = []          # example qids (training-row order)
        self._questions: List[str] = []
        self._gold_tables: List[List[str]] = []  # per-example gold table short names
        self._gold_fields: List[Dict[str, List[str]]] = {}  # idx -> {table: [fields]}
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._q_vectors: List[np.ndarray] = []

        # Fusion mode: evidence_first (original) | rank_equal (pre-registered RRF)
        self.fusion_mode = ((config.extra_params or {}).get('fusion_mode')
                            if config else None) or 'evidence_first'

    # ---------------- fit ----------------

    def fit(self, train_data: pd.DataFrame = None):
        """Uses only the current training fold. train_data must contain
        question/table(/field) and may contain a qid column (for evidence
        tracing; when missing, row numbers of the form ex000 are used)."""
        if train_data is None or len(train_data) == 0:
            # No historical supervision: degrade entirely to BM25
            # (unseen tables remain reachable by construction)
            self._fallback.fit(train_data)
            self._is_fitted = True
            return

        self._questions = [str(q) for q in train_data['question'].tolist()]
        self.train_qids = (
            [str(q) for q in train_data['qid'].tolist()]
            if 'qid' in train_data.columns
            else [f"ex{i:03d}" for i in range(len(train_data))])

        gold_tables, gold_fields = [], {}
        for i, (_, row) in enumerate(train_data.iterrows()):
            tables = [t.split('.')[-1] for t in parse_ground_truth_tables(row)]
            fields_by_table = defaultdict(list)
            for t, f in parse_ground_truth_fields(row):
                fields_by_table[t.split('.')[-1]].append(f)
            gold_tables.append(tables)
            gold_fields[i] = fields_by_table
        self._gold_tables = gold_tables
        self._gold_fields = gold_fields

        # TF-IDF over historical questions (same formula as vector_retriever.py)
        docs = [[w for w in self.tokenize(q) if len(w) > 1] for q in self._questions]
        df_count = defaultdict(int)
        for words in docs:
            for w in set(words):
                df_count[w] += 1
        n_docs = len(docs)
        self._vocab = {w: i for i, w in enumerate(sorted(df_count))}
        self._idf = {w: math.log(n_docs / (c + 1)) for w, c in df_count.items()}
        self._q_vectors = [self._tfidf_vector(words) for words in docs]

        # Fallback BM25 (depends only on the schema CSVs)
        self._fallback.fit(train_data)
        self._is_fitted = True

    def _rank_candidates(self, candidates, evidence_scores, bm25_order, bm25_scores):
        """Rank candidate tables according to the fusion mode; returns
        (ordered list, ranking-score map).

        - evidence_first (original, kept for traceability): (evidence
          score desc, BM25 score desc, table name asc); output score =
          evidence score (BM25 score for zero-vote tables).
        - rank_equal (pre-registered equal-weight RRF rank fusion):
          score(T) = Σ_{l ∈ {evidence list, BM25 list}} 1/(60+rank_l(T)),
          a missing list contributes 0; candidates = evidence tables ∪
          BM25 top-k; ties by table name. Rank fusion avoids adding raw
          scores on different scales; the BM25 list participates
          independently (unseen tables can reach the top on schema signal
          alone), so this is not a re-ranking within historical
          candidates only; output score = the RRF score (consistent with
          the ordering).
        """
        if self.fusion_mode == 'rank_equal':
            return _rrf_rank(candidates, evidence_scores, bm25_order)
        ordered = sorted(candidates,
                         key=lambda t: (-evidence_scores.get(t, 0.0),
                                        -bm25_scores.get(t, 0.0), t))
        scores = {t: (evidence_scores.get(t, 0.0) or bm25_scores.get(t, 0.0))
                  for t in candidates}
        return ordered, scores

    def _tfidf_vector(self, words: List[str]) -> np.ndarray:
        vec = np.zeros(len(self._vocab))
        count = defaultdict(int)
        for w in words:
            count[w] += 1
        for w, c in count.items():
            if w in self._vocab:
                tf = 1 + math.log(c)
                vec[self._vocab[w]] = tf * self._idf.get(w, 1.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ---------------- retrieve ----------------

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        # Fallback BM25 list: evidence_first keeps the original depth k
        # (behavior unchanged); rank_equal_v2 uses the independent fixed
        # depth BM25_RANK_DEPTH
        depth = BM25_RANK_DEPTH if self.fusion_mode == 'rank_equal' else k
        fallback_results = self._fallback.retrieve(query, k=depth)
        bm25_order = [r.table for r in fallback_results]
        bm25_score = {r.table: float(r.table_score) for r in fallback_results}

        evidence = {'examples': [], 'table_votes': {}, 'field_votes': {}}
        rank_scores: Dict[str, float] = {}

        if self._q_vectors:
            q_vec = self._tfidf_vector([w for w in self.tokenize(query) if len(w) > 1])
            sims = []
            for i, v in enumerate(self._q_vectors):
                s = float(np.dot(q_vec, v)) if q_vec.any() and v.any() else 0.0
                if s > 0:
                    sims.append((s, i))
            # Fixed rule: similarity desc, example index asc, keep the
            # first MAX_EXAMPLES
            sims.sort(key=lambda x: (-x[0], x[1]))
            sims = sims[:MAX_EXAMPLES]

            table_votes: Dict[str, float] = defaultdict(float)
            field_votes: Dict[Tuple[str, str], float] = defaultdict(float)
            for s, i in sims:
                evidence['examples'].append(
                    {'qid': self.train_qids[i], 'sim': round(s, 6),
                     'tables': self._gold_tables[i]})
                for t in self._gold_tables[i]:
                    table_votes[t] += s
                    for f in self._gold_fields.get(i, {}).get(t, []):
                        field_votes[(t, f)] += s
            evidence['table_votes'] = dict(table_votes)
            evidence['field_votes'] = {f"{t}.{c}": v for (t, c), v in field_votes.items()}

            # Rank by fusion_mode (constructor parameter, default
            # evidence_first)
            candidates = set(table_votes) | set(bm25_order)
            ordered, rank_scores = self._rank_candidates(
                candidates, table_votes, bm25_order, bm25_score)
            evidence['fusion_mode'] = ('rank_equal_v2' if self.fusion_mode == 'rank_equal'
                                       else self.fusion_mode)
        else:
            ordered = bm25_order

        results = []
        for table in ordered[:k]:
            columns = self._columns_for(query, table, evidence['field_votes'],
                                        fallback_results)
            results.append(RetrievalResult(
                table=table,
                table_score=rank_scores.get(
                    table, evidence['table_votes'].get(
                        table, bm25_score.get(table, 0.0))),
                columns=columns,
                metadata={'method': 'ExampleRAG', 'evidence': evidence},
            ))
        return results

    def _columns_for(self, query: str, table: str, field_votes: Dict[str, float],
                     fallback_results) -> List[Tuple[str, float]]:
        """Per-table column budget of 10: voted fields first, the rest
        filled by query-term/field overlap (same field scoring as
        BM25)."""
        schema_cols = self.field_df[self.field_df['table'] == table]
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        scored = []
        for _, row in schema_cols.iterrows():
            fname = str(row['field_name'])
            fdesc = str(row.get('field_name_desc', '') or '')
            vote = field_votes.get(f"{table}.{fname}", 0.0)
            overlap = len(query_term_set & set(
                t for t in self.tokenize(f"{fname} {fdesc}") if len(t) > 1))
            overlap_score = overlap / len(query_terms) if query_terms else 0.0
            if any(t in fname for t in query_term_set):
                overlap_score += 0.5
            scored.append((fname, vote, overlap_score))

        # Order: vote desc -> overlap desc -> field name asc (schema-row
        # uniqueness guarantees dedup)
        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [(f"C:{table}.{f}", v + o) for f, v, o in scored[:COLUMN_BUDGET]]
