"""Hybrid retriever.

Reuses the BM25 and Vector retrievers with linear score fusion.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from terag.engines.bird_runtime.retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)
from terag.engines.bird_runtime.retrievers.bm25_retriever import BM25Retriever
from terag.engines.bird_runtime.retrievers.vector_retriever import VectorRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever.

    Reuses the BM25 and Vector retrievers with linear score fusion:

    final_score = alpha * bm25_score + (1 - alpha) * vector_score
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None,
                 alpha: float = 0.5):
        """Initialize the Hybrid retriever.

        Args:
            field_csv: field CSV path.
            table_csv: table CSV path.
            config: retriever configuration.
            alpha: BM25 score weight (0-1); the default 0.5 means equal
                   weight. Suggested values: 0.5 (balanced), 0.6
                   (keyword-leaning), 0.4 (semantic-leaning).
        """
        if config is None:
            config = RetrieverConfig(
                name="Hybrid",
                description="Hybrid retriever combining BM25 and Vector"
            )

        super().__init__(field_csv, table_csv, config)

        # Fusion weight
        self.alpha = alpha

        # Create the sub-retrievers (tokenization language forwarded; the
        # BIRD English setting does not use jieba)
        sub_tokenizer = self.config.use_chinese_tokenizer
        self.bm25_retriever = BM25Retriever(
            field_csv, table_csv,
            config=RetrieverConfig(name="BM25", use_chinese_tokenizer=sub_tokenizer))
        self.vector_retriever = VectorRetriever(
            field_csv, table_csv,
            config=RetrieverConfig(name="Vector", use_chinese_tokenizer=sub_tokenizer))

    def fit(self, train_data: pd.DataFrame = None):
        """Train/build the indexes.

        Builds the BM25 and Vector indexes separately.

        Args:
            train_data: training data (BM25 and Vector do not need it; the
                parameter keeps the interface uniform).
        """
        print("  Hybrid: building the BM25 index...")
        self.bm25_retriever.fit(train_data)

        print("  Hybrid: building the Vector index...")
        self.vector_retriever.fit(train_data)

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Hybrid retrieval.

        1. Call BM25 and Vector retrieval separately.
        2. Align the candidate results by table name.
        3. Compute the fused score.
        4. Re-rank by the fused score.
        5. Return the top-k results.

        Args:
            query: query text.
            k: top-k results to return.

        Returns:
            List of retrieval results.
        """
        # Get both retrievers' results (more candidates for the fusion)
        retrieve_k = min(k * 3, 20)  # extra candidates for fusion

        bm25_results = self.bm25_retriever.retrieve(query, retrieve_k)
        vector_results = self.vector_retriever.retrieve(query, retrieve_k)

        # Collect all candidate table names
        all_tables = set()
        bm25_scores: Dict[str, float] = {}
        vector_scores: Dict[str, float] = {}
        bm25_columns: Dict[str, List[tuple]] = {}
        vector_columns: Dict[str, List[tuple]] = {}

        for r in bm25_results:
            all_tables.add(r.table)
            bm25_scores[r.table] = r.table_score
            bm25_columns[r.table] = r.columns

        for r in vector_results:
            all_tables.add(r.table)
            vector_scores[r.table] = r.table_score
            vector_columns[r.table] = r.columns

        # Score normalization (min-max)
        def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
            if not scores:
                return {}
            values = list(scores.values())
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k: 1.0 for k in scores}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

        norm_bm25 = normalize_scores(bm25_scores)
        norm_vector = normalize_scores(vector_scores)

        # Compute the fused scores
        final_scores: Dict[str, float] = {}
        for table in all_tables:
            bm25_s = norm_bm25.get(table, 0.0)
            vector_s = norm_vector.get(table, 0.0)
            # Linear weighted fusion
            final_scores[table] = self.alpha * bm25_s + (1 - self.alpha) * vector_s

        # Rank by the fused score
        sorted_tables = sorted(final_scores.items(), key=lambda x: (-x[1], x[0]))[:k]

        # Build the results
        results = []
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        for table_name, score in sorted_tables:
            # Merge the column lists (prefer BM25's columns, else Vector's)
            if table_name in bm25_columns:
                columns = bm25_columns[table_name]
            elif table_name in vector_columns:
                columns = vector_columns[table_name]
            else:
                # Neither retriever has column info: derive from the raw data
                columns = self._get_default_columns(table_name, query_term_set, query_terms)

            results.append(RetrievalResult(
                table=table_name,
                table_score=score,
                columns=columns,
                metadata={
                    'method': 'Hybrid',
                    'alpha': self.alpha,
                    'bm25_score': bm25_scores.get(table_name, 0.0),
                    'vector_score': vector_scores.get(table_name, 0.0)
                }
            ))

        return results

    def _get_default_columns(self, table_name: str, query_term_set: set, query_terms: list) -> List[tuple]:
        """Get the default column list."""
        columns = self.field_df[self.field_df['table'] == table_name]

        column_scores = []
        for _, row in columns.iterrows():
            field_name = row['field_name']
            field_desc = str(row.get('field_name_desc', ''))

            field_text = f"{field_name} {field_desc}"
            field_terms = set(t for t in self.tokenize(field_text) if len(t) > 1)

            overlap = len(query_term_set & field_terms)
            col_score = overlap / len(query_terms) if query_terms else 0

            if any(term in field_name for term in query_term_set):
                col_score += 0.5

            column_scores.append((f"C:{table_name}.{field_name}", col_score))

        column_scores.sort(key=lambda x: x[1], reverse=True)
        return column_scores[:10]  # P1-B unified protocol column budget
