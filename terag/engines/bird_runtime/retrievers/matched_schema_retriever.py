"""Matched Schema control retriever (P2-C).

Purpose: under EXACTLY the same schema documents, tokenization and index
scoring as the current TE-RAG enhanced index, provide a retrieval control
WITHOUT historical supervision - no history graph, no pattern matching, no
historical field recommendation, no code-table mining. fit() only builds
the schema index (insensitive to training logs); table ranking = that
index's BM25 score; column ranking = the same query-term/field overlap as
the baselines (per-table budget 10).

It is not a new method and does not duplicate V2: it reuses
terag/index_builder.IndexBuilder and its BM25Retriever for document
construction and scoring; it exists only as a factor control inside
retrievers/. Index-document variants (field descriptions / field weights)
are controlled through the TERAGConfig index knobs, sharing one
implementation with the TE-RAG variants.
"""

from typing import Dict, List, Tuple

import pandas as pd

from terag.engines.bird_runtime.retrievers.base_retriever import BaseRetriever, RetrievalResult, RetrieverConfig

COLUMN_BUDGET = 10  # unified protocol column budget
from terag.engines.bird_runtime.retrievers.annotation_miner import W_ANN as W_ANN_D1  # C class reuses the same bonus


class MatchedSchemaRetriever(BaseRetriever):
    """Matched Schema control (see the module docstring)."""

    def __init__(self, field_csv: str, table_csv: str,
                 config: RetrieverConfig = None):
        if config is None:
            config = RetrieverConfig(
                name='MatchedSchema',
                description='Schema index control (no historical supervision; same TE-RAG documents/tokenization/scoring)')
        super().__init__(field_csv, table_csv, config)
        self._field_csv = field_csv
        self._table_csv = table_csv
        self._bm25 = None
        self._index = None
        self.use_history = False
        self.miner = None

    def fit(self, train_data: pd.DataFrame = None):
        """Build the schema index; with use_history=True also mine direct
        annotations from the CURRENT training fold and inject them into the
        index (passed in memory, B class) + the column-ranking bonus; with
        False, bitwise identical to the pure-metadata control (same
        postings). Test SQL/gold never enters this method."""
        from terag.engines.bird_runtime.config import TERAGConfig
        from terag.engines.bird_runtime.index_builder import IndexBuilder, BM25Retriever

        injected = (self.config.extra_params or {}).get('terag_config')
        cfg = injected or TERAGConfig()
        cfg = cfg.with_data_paths(field_csv=self._field_csv,
                                  table_csv=self._table_csv,
                                  artifacts_dir=cfg.output.artifacts_dir)

        self.use_history = bool((self.config.extra_params or {}).get('use_history'))
        self.use_transfer = bool((self.config.extra_params or {}).get('use_transfer'))
        self.miner = None
        self.transfer_result = None
        extra = None
        if self.use_history:
            if train_data is None or len(train_data) == 0:
                raise ValueError('use_history=True needs the current training fold data (B-class annotation source)')
            from terag.engines.bird_runtime.retrievers.annotation_miner import DirectAnnotationMiner
            self.miner = DirectAnnotationMiner(self.field_df, self.table_df,
                                               dialect='mysql').mine(train_data)
            extra = self.miner.table_annotation_texts(
                cfg.index.field_weights.pattern)

        # C-class transfer annotations (separate channel; can be switched
        # off alone to restore A+B)
        self._transfer_texts = {}
        self._transfer_terms = {}
        if self.use_history and self.use_transfer:
            from terag.engines.bird_runtime.retrievers.annotation_transfer import (
                SchemaInfo, transfer_annotations, transfer_index_texts,
                transfer_column_bonus_map)
            columns = {}
            for t, g in self.field_df.groupby('table'):
                columns[t] = g[['field_name', 'field_name_desc', 'field_type']] \
                    .rename(columns={'field_name_desc': 'field_desc'}) \
                    .to_dict('records')
            table_desc = {r['table']: r.get('table_desc', '')
                          for _, r in self.table_df.iterrows()}
            schema = SchemaInfo(columns=columns, table_desc=table_desc)
            self.transfer_result = transfer_annotations(
                self.miner.annotations, schema)
            self._transfer_texts = transfer_index_texts(
                self.transfer_result, cfg.index.field_weights.pattern)
            self._transfer_terms = transfer_column_bonus_map(self.transfer_result)
            merged = dict(extra or {})
            for t, texts in self._transfer_texts.items():
                merged.setdefault(t, [])
                merged[t] = list(merged.get(t, [])) + list(texts)
            extra = merged

        index = IndexBuilder(cfg).build(extra_table_texts=extra)
        self._bm25 = BM25Retriever(index)
        self._index = index
        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        results = self._bm25.search(query, k=k)
        out = []
        for doc_id, score in results:
            table = doc_id.replace('T:', '')
            out.append(RetrievalResult(
                table=table,
                table_score=float(score),
                columns=self._columns_for(query, table),
                metadata=self._metadata(table),
            ))
        return out

    def _metadata(self, table: str) -> Dict:
        meta = {'method': 'MatchedSchema',
                'use_history': self.use_history}
        if self.miner is not None:
            meta['annotation_summary'] = self.miner.summary()
        return meta

    def _columns_for(self, query: str, table: str) -> List[Tuple[str, float]]:
        """Column scoring: query-term/field overlap + field-name substring
        bonus (same as the baselines) + the B-class annotation hit bonus
        (with use_history, w_ann=0.5 fixed), budget 10."""
        schema_cols = self.field_df[self.field_df['table'] == table]
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)
        scored = []
        for _, row in schema_cols.iterrows():
            fname = str(row['field_name'])
            fdesc = str(row.get('field_name_desc', '') or '')
            field_terms = set(t for t in self.tokenize(f"{fname} {fdesc}") if len(t) > 1)
            overlap = len(query_term_set & field_terms)
            s = overlap / len(query_terms) if query_terms else 0.0
            if any(t in fname for t in query_term_set):
                s += 0.5
            if self.miner is not None:
                s += self.miner.column_bonus(query_term_set, table, fname)
                for term in self._transfer_terms.get(
                        f"{table.lower()}.{fname.lower()}", []):
                    if term in query_term_set:
                        s += W_ANN_D1
            scored.append((fname, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(f"C:{table}.{f}", s) for f, s in scored[:COLUMN_BUDGET]]
