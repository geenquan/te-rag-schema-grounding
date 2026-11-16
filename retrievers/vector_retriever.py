"""
Vector retriever implementation.

Semantic-similarity table retriever (field-level semantic matching).
Deliberately complements BM25: BM25 leans on table-name keywords,
Vector leans on field semantics.
"""

import pandas as pd
import numpy as np
import jieba
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, RetrievalResult
)


class VectorRetriever(BaseRetriever):
    """
    Vector retriever.

    Retrieves tables via semantic similarity.
    Difference from BM25:
    - BM25: keyword matching over table names and descriptions.
    - Vector: similarity matching over field semantics.
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """
        Initialize the vector retriever.

        Args:
            field_csv: Path to the field CSV.
            table_csv: Path to the table CSV.
            config: Retriever configuration.
        """
        if config is None:
            config = RetrieverConfig(
                name="Vector",
                description="Semantic vector retriever (field-level)"
            )

        super().__init__(field_csv, table_csv, config)

        # Vector state
        self.table_vectors: Dict[str, np.ndarray] = {}
        self.field_vectors: Dict[str, np.ndarray] = {}
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, train_data: pd.DataFrame = None):
        """
        Build the vector index.

        Args:
            train_data: Training data (unused by this method; kept for interface consistency).
        """
        # Build the vocabulary from field descriptions, not table names
        all_words = set()
        field_docs = []  # each field is one document
        table_field_map = defaultdict(list)  # table name -> list of field indices

        for idx, row in self.field_df.iterrows():
            # Field name and field description form the document content
            field_name = str(row.get('field_name', ''))
            field_desc = str(row.get('field_name_desc', ''))
            table_name = str(row.get('table', ''))

            # Combine the text with the description first (it carries the semantics)
            text = f"{field_desc} {field_name}"
            words = [w for w in self.tokenize(text) if len(w) > 1]

            all_words.update(words)
            field_docs.append(words)
            table_field_map[table_name].append(len(field_docs) - 1)

        self.vocab = {word: idx for idx, word in enumerate(all_words)}

        # Compute IDF over field documents
        n_docs = len(field_docs)
        word_doc_count = defaultdict(int)
        for doc_words in field_docs:
            for word in set(doc_words):  # set avoids double counting
                word_doc_count[word] += 1

        self.idf = {
            word: math.log(n_docs / (count + 1))
            for word, count in word_doc_count.items()
        }

        # Build field vectors
        for idx, doc_words in enumerate(field_docs):
            vector = self._build_tfidf_vector(doc_words)
            table_name = self.field_df.iloc[idx]['table']
            field_name = self.field_df.iloc[idx]['field_name']
            self.field_vectors[f"{table_name}.{field_name}"] = vector

        # Build table vectors by aggregating the table's field vectors
        for table_name in self.table_df['table'].unique():
            table_fields = self.field_df[self.field_df['table'] == table_name]
            if len(table_fields) == 0:
                continue

            # Aggregate all field vectors (weighted average)
            field_vecs = []
            for _, row in table_fields.iterrows():
                key = f"{table_name}.{row['field_name']}"
                if key in self.field_vectors:
                    field_vecs.append(self.field_vectors[key])

            if field_vecs:
                # Weighted average: fields with longer descriptions weigh more
                weights = []
                for _, row in table_fields.iterrows():
                    desc = str(row.get('field_name_desc', ''))
                    weight = 1.0 + 0.5 * min(len(desc) / 10, 1.0)  # longer description -> higher weight
                    weights.append(weight)

                weights = np.array(weights)
                weights = weights / weights.sum()

                table_vector = np.zeros(len(self.vocab))
                for vec, w in zip(field_vecs, weights):
                    table_vector += vec * w

                # L2 normalization
                norm = np.linalg.norm(table_vector)
                if norm > 0:
                    table_vector = table_vector / norm

                self.table_vectors[table_name] = table_vector

        self._is_fitted = True

    def _build_tfidf_vector(self, words: List[str]) -> np.ndarray:
        """Build a TF-IDF vector."""
        vector = np.zeros(len(self.vocab))

        # Compute TF
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1

        # Compute TF-IDF
        for word, count in word_count.items():
            if word in self.vocab:
                # Log TF dampens the influence of high-frequency terms
                tf = 1 + math.log(count) if count > 0 else 0
                vector[self.vocab[word]] = tf * self.idf.get(word, 1.0)

        # L2 normalization
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Vector retrieval.

        Args:
            query: Query text.
            k: Number of top results to return.

        Returns:
            List of retrieval results.
        """
        # Build the query vector
        query_words = [w for w in self.tokenize(query) if len(w) > 1]
        query_vector = self._build_tfidf_vector(query_words)
        query_term_set = set(query_words)

        # Cosine similarity against every table
        scores = {}
        for table_name, vector in self.table_vectors.items():
            similarity = np.dot(query_vector, vector)
            scores[table_name] = similarity

        # Sort for top-k
        sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

        results = []
        for table_name, score in sorted_scores:
            # Get the table's fields, ranked by semantic similarity
            table_fields = self.field_df[self.field_df['table'] == table_name]

            # Semantic similarity of each field to the query
            column_scores = []
            for _, row in table_fields.iterrows():
                field_name = row['field_name']
                field_desc = str(row.get('field_name_desc', ''))

                # Similarity via the field's vector
                key = f"{table_name}.{field_name}"
                if key in self.field_vectors:
                    field_vec = self.field_vectors[key]
                    field_sim = np.dot(query_vector, field_vec)
                else:
                    # Fall back to plain vocabulary overlap
                    field_text = f"{field_name} {field_desc}"
                    field_terms = set(t for t in self.tokenize(field_text) if len(t) > 1)
                    overlap = len(query_term_set & field_terms)
                    field_sim = overlap / len(query_words) if query_words else 0

                column_scores.append((f"C:{table_name}.{field_name}", field_sim))

            # Per-table column budget: at most 10 deduplicated fields (unified protocol)
            column_scores.sort(key=lambda x: x[1], reverse=True)
            column_list = column_scores[:10]

            results.append(RetrievalResult(
                table=table_name,
                table_score=float(score),
                columns=column_list,
                metadata={'method': 'Vector', 'similarity': 'semantic'}
            ))

        return results
