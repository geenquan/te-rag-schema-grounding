"""
BM25 retriever implementation.

Table retriever based on the BM25 algorithm.
"""

import pandas as pd
import numpy as np
import jieba
import math
from collections import defaultdict
from typing import Dict, List, Optional

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever.

    Retrieves tables with the BM25 algorithm, i.e. keyword matching.
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        Initialize the BM25 retriever.

        Args:
            field_csv: Path to the field CSV.
            table_csv: Path to the table CSV.
            config: Retriever configuration.
        """
        if config is None:
            config = RetrieverConfig(
                name="BM25",
                description="BM25 keyword-based retriever"
            )

        super().__init__(field_csv, table_csv, config)

        # BM25 index
        self.table_inverted_index: Dict[str, List[tuple]] = {}
        self.table_doc_freq: Dict[str, int] = {}
        self.table_doc_lengths: List[int] = []
        self.total_table_docs: int = 0
        self.avg_table_doc_length: float = 1.0

    def fit(self, train_data: pd.DataFrame = None):
        """
        Build the BM25 index.

        Args:
            train_data: Training data (unused by BM25; kept for interface consistency).
        """
        # Build the inverted index
        self.table_inverted_index = {}
        self.table_doc_lengths = []

        for idx, row in self.table_df.iterrows():
            # Table name plus description forms the document
            text = f"{row['table']} {row.get('table_desc', '')}"
            words = self.tokenize(text)
            self.table_doc_lengths.append(len(words))

            # Count term frequencies
            word_count = defaultdict(int)
            for word in words:
                if len(word) > 1:
                    word_count[word] += 1

            # Add to the inverted index
            for word, count in word_count.items():
                if word not in self.table_inverted_index:
                    self.table_inverted_index[word] = []
                self.table_inverted_index[word].append((idx, count))

        # Compute document frequencies
        self.table_doc_freq = {}
        for word, postings in self.table_inverted_index.items():
            self.table_doc_freq[word] = len(postings)

        # Compute corpus statistics
        self.total_table_docs = len(self.table_df)
        self.avg_table_doc_length = np.mean(self.table_doc_lengths) if self.table_doc_lengths else 1.0

        self._is_fitted = True

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        BM25 retrieval.

        Args:
            query: Query text.
            k: Number of top results to return.

        Returns:
            List of retrieval results.
        """
        query_terms = [t for t in self.tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)
        scores: Dict[int, float] = {}

        # Compute BM25 scores
        for term in query_terms:
            if term not in self.table_inverted_index:
                continue

            df = self.table_doc_freq[term]
            idf = math.log((self.total_table_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_idx, tf in self.table_inverted_index[term]:
                doc_length = self.table_doc_lengths[doc_idx]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_table_doc_length)
                score = idf * numerator / denominator

                if doc_idx not in scores:
                    scores[doc_idx] = 0
                scores[doc_idx] += score

        # Sort for top-k (ties broken stably by doc_idx)
        sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

        results = []
        for doc_idx, score in sorted_scores:
            table_name = self.table_df.iloc[doc_idx]['table']

            # Get the table's columns, ranked by relevance to the query
            columns = self.field_df[self.field_df['table'] == table_name]

            # Score each field's relevance to the query
            column_scores = []
            for _, row in columns.iterrows():
                field_name = row['field_name']
                field_desc = str(row.get('field_name_desc', ''))

                # Match the field name and description against the query
                field_text = f"{field_name} {field_desc}"
                field_terms = set(t for t in self.tokenize(field_text) if len(t) > 1)

                # Match score: intersection size / number of query terms
                overlap = len(query_term_set & field_terms)
                col_score = overlap / len(query_terms) if query_terms else 0

                # Bonus when the query contains a field-name keyword
                if any(term in field_name for term in query_term_set):
                    col_score += 0.5

                column_scores.append((f"C:{table_name}.{field_name}", col_score))

            # Per-table column budget: at most 10 deduplicated fields (unified protocol)
            column_scores.sort(key=lambda x: x[1], reverse=True)
            column_list = column_scores[:10]

            results.append(RetrievalResult(
                table=table_name,
                table_score=score,
                columns=column_list,
                metadata={'method': 'BM25'}
            ))

        return results
