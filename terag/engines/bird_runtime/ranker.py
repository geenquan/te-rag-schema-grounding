"""Rankers.

Table ranker and field ranker.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from terag.engines.bird_runtime.config import TERAGConfig
from terag.engines.bird_runtime.feature_extractor import FeatureExtractor


@dataclass
class RankingResult:
    """A ranking result."""
    table: str
    table_score: float
    columns: List[Tuple[str, float]]  # [(column_name, score), ...]


class TableRanker:
    """Table ranker.

    Ranks tables using learned weights.
    """

    def __init__(self, config: TERAGConfig):
        """Initialize the table ranker.

        Args:
            config: TE-RAG configuration.
        """
        self.config = config
        self.weights = config.feature.table_weights.copy()
        self._model = None

    def set_weights(self, weights: Dict[str, float]):
        """Set the weights."""
        self.weights = weights.copy()

    def get_weights(self) -> Dict[str, float]:
        """Get the weights."""
        return self.weights.copy()

    def rank(
        self,
        table_features: Dict[str, Dict[str, float]],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Rank tables.

        Args:
            table_features: {table_node: {'bm25_score': float, ...}}
            k: top-k results to return.

        Returns:
            [(table_node, score), ...]
        """
        scores = {}

        for table_node, features in table_features.items():
            score = 0.0
            for feature_name, feature_value in features.items():
                weight = self.weights.get(feature_name, 0.0)
                score += weight * feature_value
            scores[table_node] = score

        # Sort and return top-k (ties broken stably by table name, not by
        # insertion order)
        sorted_results = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return sorted_results[:k]

    def save(self, output_path: str):
        """Save the ranker."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'model': self._model,
            }, f)
        print(f"Table ranker saved to: {output_path}")

    def load(self, input_path: str):
        """Load the ranker."""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self._model = data.get('model')
        print(f"Table ranker loaded from {input_path}")


class FieldRanker:
    """Field ranker.

    Ranks fields using learned weights.
    """

    def __init__(self, config: TERAGConfig):
        """Initialize the field ranker.

        Args:
            config: TE-RAG configuration.
        """
        self.config = config
        self.weights = config.feature.field_weights.copy()
        self._model = None

    def set_weights(self, weights: Dict[str, float]):
        """Set the weights."""
        self.weights = weights.copy()

    def get_weights(self) -> Dict[str, float]:
        """Get the weights."""
        return self.weights.copy()

    def rank(
        self,
        field_features: Dict[str, Dict[str, float]],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Rank fields.

        Args:
            field_features: {field_name: {'direct_match': float, ...}}
            k: top-k results to return.

        Returns:
            [(field_name, score), ...]
        """
        scores = {}

        for field_name, features in field_features.items():
            score = 0.0
            for feature_name, feature_value in features.items():
                weight = self.weights.get(feature_name, 0.0)
                score += weight * feature_value
            scores[field_name] = score

        # Sort and return top-k (ties broken stably by field name, not by
        # insertion order)
        sorted_results = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return sorted_results[:k]

    def save(self, output_path: str):
        """Save the ranker."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'model': self._model,
            }, f)
        print(f"Field ranker saved to: {output_path}")

    def load(self, input_path: str):
        """Load the ranker."""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self._model = data.get('model')
        print(f"Field ranker loaded from {input_path}")


class CombinedRanker:
    """Combined ranker.

    Integrates table ranking and field ranking.
    """

    def __init__(
        self,
        config: TERAGConfig,
        table_ranker: Optional[TableRanker] = None,
        field_ranker: Optional[FieldRanker] = None
    ):
        """Initialize the combined ranker.

        Args:
            config: TE-RAG configuration.
            table_ranker: table ranker.
            field_ranker: field ranker.
        """
        self.config = config
        self.table_ranker = table_ranker or TableRanker(config)
        self.field_ranker = field_ranker or FieldRanker(config)

    def rank(
        self,
        feature_extractor: FeatureExtractor,
        query: str,
        k_tables: int = 5,
        k_fields: int = 10,
        diagnostics: Optional[Dict] = None
    ) -> List[RankingResult]:
        """Run the full ranking.

        Args:
            feature_extractor: feature extractor.
            query: query text.
            k_tables: number of tables to return.
            k_fields: number of fields per table to return.

        Returns:
            [RankingResult, ...]
        """
        results = []

        # 1. Extract table-level features and rank
        table_features = feature_extractor.extract_table_features(query)
        top_tables = self.table_ranker.rank(table_features, k=k_tables)

        if diagnostics is not None:
            diagnostics.update({
                'candidate_scope': 'tables in extract_table_features; fields of returned tables only',
                'table_budget': k_tables, 'column_budget_per_table': k_fields,
                'tables': _feature_trace(table_features, self.table_ranker.weights),
                'returned_tables': [t for t, _ in top_tables], 'fields': {}})

        # 2. Extract field-level features and rank per table
        for table_node, table_score in top_tables:
            table_name = table_node.replace('T:', '')

            # Extract field-level features
            field_features = feature_extractor.extract_field_features(query, table_name)

            # Rank fields
            top_fields = self.field_ranker.rank(field_features, k=k_fields)

            if diagnostics is not None:
                diagnostics['fields'][table_name] = {
                    'candidates': _feature_trace(field_features, self.field_ranker.weights),
                    'returned_columns': [f for f, _ in top_fields]}

            # Build the result
            results.append(RankingResult(
                table=table_name,
                table_score=table_score,
                columns=[(f"C:{table_name}.{field}", score) for field, score in top_fields]
            ))

        return results

    def save(self, artifacts_dir: str):
        """Save the rankers."""
        self.table_ranker.save(os.path.join(artifacts_dir, 'table_ranker.pkl'))
        self.field_ranker.save(os.path.join(artifacts_dir, 'field_ranker.pkl'))

    def load(self, artifacts_dir: str):
        """Load the rankers."""
        self.table_ranker.load(os.path.join(artifacts_dir, 'table_ranker.pkl'))
        self.field_ranker.load(os.path.join(artifacts_dir, 'field_ranker.pkl'))


def _feature_trace(features, weights):
    """Capture already-computed inputs; do not recompute features or rankings."""
    return {key: {'features': dict(values),
                  'weighted_components': {name: value * weights.get(name, 0.0)
                                          for name, value in values.items()},
                  'score': sum(value * weights.get(name, 0.0)
                               for name, value in values.items())}
            for key, values in features.items()}
