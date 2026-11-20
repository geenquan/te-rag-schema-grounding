"""Weight learner.

Learns ranking weights with LogisticRegression.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from terag.engines.bird_runtime.config import TERAGConfig
from terag.engines.bird_runtime.feature_extractor import FeatureExtractor
from terag.engines.bird_runtime.ranker import TableRanker, FieldRanker


class WeightLearner:
    """Weight learner.

    Learns table-ranking and field-ranking weights with
    LogisticRegression.

    Usage:
        learner = WeightLearner(config, feature_extractor)
        learner.fit(train_data, dev_data)
        learner.save("artifacts/")

        # Get the learned weights
        table_weights = learner.get_table_weights()
        field_weights = learner.get_field_weights()
    """

    def __init__(self, config: TERAGConfig, feature_extractor: FeatureExtractor):
        """Initialize the weight learner.

        Args:
            config: TE-RAG configuration.
            feature_extractor: feature extractor.
        """
        self.config = config
        self.feature_extractor = feature_extractor

        # Models
        self.table_model = None
        self.field_model = None
        self.table_scaler = None
        self.field_scaler = None

        # Learned weights
        self.table_weights = {}
        self.field_weights = {}

    def fit(
        self,
        train_data: pd.DataFrame,
        dev_data: Optional[pd.DataFrame] = None
    ):
        """Train the weights.

        Args:
            train_data: training data.
            dev_data: validation data (optional, for early stopping).
        """
        print("Training table ranking weights...")
        self._fit_table_ranker(train_data)

        print("Training field ranking weights...")
        self._fit_field_ranker(train_data)

    def _fit_table_ranker(self, train_data: pd.DataFrame):
        """Train the table ranker."""
        X = []
        y = []

        feature_names = ['bm25_score', 'graph_score', 'pattern_score']

        for idx, row in train_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')

            if pd.isna(gt_table):
                continue

            gt_table_simple = gt_table.split('.')[-1]

            # Extract features
            table_features = self.feature_extractor.extract_table_features(query)

            for table_node, features in table_features.items():
                table_simple = table_node.replace('T:', '')
                label = 1 if table_simple == gt_table_simple else 0

                feature_vector = [features.get(fn, 0.0) for fn in feature_names]
                X.append(feature_vector)
                y.append(label)

        if not X:
            print("  Warning: not enough training data")
            return

        X = np.array(X)
        y = np.array(y)

        # Standardize
        self.table_scaler = StandardScaler()
        X_scaled = self.table_scaler.fit_transform(X)

        # Train
        self.table_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.seed
        )
        self.table_model.fit(X_scaled, y)

        # Extract the weights
        # The LogisticRegression coefficients reflect each feature's importance
        coef = self.table_model.coef_[0]

        # Convert the coefficients to positive weights
        weights = np.abs(coef)
        weights = weights / weights.sum()  # normalize

        self.table_weights = {fn: float(w) for fn, w in zip(feature_names, weights)}

        print(f"  Learned table ranking weights: {self.table_weights}")

    def _fit_field_ranker(self, train_data: pd.DataFrame):
        """Train the field ranker."""
        X = []
        y = []

        feature_names = ['direct_match', 'graph_propagation', 'role_prior', 'train_recommend']

        for idx, row in train_data.iterrows():
            query = row['question']
            gt_table = row.get('table', '')
            gt_fields = row.get('field', '')

            if pd.isna(gt_table):
                continue

            gt_table_simple = gt_table.split('.')[-1]

            gt_field_set = set()
            if pd.notna(gt_fields) and isinstance(gt_fields, str):
                gt_field_set = set(f.strip() for f in gt_fields.split('|') if f.strip())

            # Extract features
            field_features = self.feature_extractor.extract_field_features(query, gt_table_simple)

            for field_name, features in field_features.items():
                label = 1 if field_name in gt_field_set else 0

                feature_vector = [features.get(fn, 0.0) for fn in feature_names]
                X.append(feature_vector)
                y.append(label)

        if not X:
            print("  Warning: not enough training data")
            return

        X = np.array(X)
        y = np.array(y)

        # Standardize
        self.field_scaler = StandardScaler()
        X_scaled = self.field_scaler.fit_transform(X)

        # Train
        self.field_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.seed
        )
        self.field_model.fit(X_scaled, y)

        # Extract the weights
        coef = self.field_model.coef_[0]
        weights = np.abs(coef)
        weights = weights / weights.sum()

        self.field_weights = {fn: float(w) for fn, w in zip(feature_names, weights)}

        print(f"  Learned field ranking weights: {self.field_weights}")

    def get_table_weights(self) -> Dict[str, float]:
        """Get the table ranking weights."""
        return self.table_weights.copy()

    def get_field_weights(self) -> Dict[str, float]:
        """Get the field ranking weights."""
        return self.field_weights.copy()

    def get_table_ranker(self) -> TableRanker:
        """Get a table ranker configured with the learned weights."""
        ranker = TableRanker(self.config)
        if self.table_weights:
            ranker.set_weights(self.table_weights)
        return ranker

    def get_field_ranker(self) -> FieldRanker:
        """Get a field ranker configured with the learned weights."""
        ranker = FieldRanker(self.config)
        if self.field_weights:
            ranker.set_weights(self.field_weights)
        return ranker

    def save(self, artifacts_dir: str):
        """Save the weight learner."""
        os.makedirs(artifacts_dir, exist_ok=True)

        with open(os.path.join(artifacts_dir, 'weight_learner.pkl'), 'wb') as f:
            pickle.dump({
                'table_model': self.table_model,
                'field_model': self.field_model,
                'table_scaler': self.table_scaler,
                'field_scaler': self.field_scaler,
                'table_weights': self.table_weights,
                'field_weights': self.field_weights,
            }, f)

        # Also save the weights as JSON (easy inspection)
        import json
        with open(os.path.join(artifacts_dir, 'learned_weights.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'table_weights': self.table_weights,
                'field_weights': self.field_weights,
            }, f, indent=2, ensure_ascii=False)

        print(f"Weight learner saved to: {artifacts_dir}")

    def load(self, artifacts_dir: str):
        """Load the weight learner."""
        with open(os.path.join(artifacts_dir, 'weight_learner.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.table_model = data['table_model']
            self.field_model = data['field_model']
            self.table_scaler = data['table_scaler']
            self.field_scaler = data['field_scaler']
            self.table_weights = data['table_weights']
            self.field_weights = data['field_weights']

        print(f"Weight learner loaded from: {artifacts_dir}")
        print(f"  table ranking weights: {self.table_weights}")
        print(f"  field ranking weights: {self.field_weights}")


def main():
    """Weight learning demo."""
    import sys
    import json
    from pathlib import Path
    import networkx as nx

    # Add the project root
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.engines.bird_runtime.config import TERAGConfig
    from terag.engines.bird_runtime.graph_builder import BipartiteGraphBuilder
    from terag.engines.bird_runtime.pattern_miner import PatternMiner
    from terag.engines.bird_runtime.index_builder import IndexBuilder

    # Load the configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    # Load the training data
    train_data = []
    with open(config.get_split_path('train'), 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    print(f"Loaded training data: {len(train_df)} rows")

    # Build the components
    print("\nBuilding the graph...")
    graph_builder = BipartiteGraphBuilder(config)
    graph = graph_builder.build(train_df)

    print("\nMining patterns...")
    element_to_queries = graph_builder.get_element_to_queries(graph, train_df)
    pattern_miner = PatternMiner(config)
    patterns = pattern_miner.mine(element_to_queries)

    print("\nBuilding the index...")
    index_builder = IndexBuilder(config)
    index = index_builder.build()

    # Load the field and table schema
    field_df = pd.read_csv(config.data.field_csv)
    table_df = pd.read_csv(config.data.table_csv)

    # Create the feature extractor
    feature_extractor = FeatureExtractor(
        config, graph, index, patterns,
        train_df, field_df, table_df
    )

    # Train the weights
    print("\nTraining weights...")
    learner = WeightLearner(config, feature_extractor)
    learner.fit(train_df)

    # Save
    learner.save(config.output.artifacts_dir)

    # Print the learned weights
    print("\nLearned weights:")
    print(f"  table ranking: {learner.get_table_weights()}")
    print(f"  field ranking: {learner.get_field_weights()}")


if __name__ == "__main__":
    main()
