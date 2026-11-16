"""
Retriever factory.

Provides a unified interface for creating and managing retrievers.
"""

import os
from typing import Dict, List, Optional, Type, Callable
import pandas as pd

from retrievers.base_retriever import (
    BaseRetriever, RetrieverConfig, EvaluationMetrics, RetrievalResult
)


class RetrieverFactory:
    """
    Retriever factory.

    Supports:
    1. registering new retriever types
    2. creating retrievers from configuration
    3. creating and managing retrievers in batch
    """

    # Registered retriever types
    _registry: Dict[str, Type[BaseRetriever]] = {}

    # Default configurations
    _default_configs: Dict[str, RetrieverConfig] = {}

    @classmethod
    def register(cls, name: str, retriever_class: Type[BaseRetriever],
                 default_config: Optional[RetrieverConfig] = None):
        """
        Register a retriever type.

        Args:
            name: retriever name
            retriever_class: retriever class
            default_config: default configuration

        Example:
            >>> @RetrieverFactory.register("my_retriever", MyRetriever)
            >>> class MyRetriever(BaseRetriever):
            >>>     pass
        """
        cls._registry[name] = retriever_class
        if default_config:
            cls._default_configs[name] = default_config
        print(f"Registered retriever: {name}")

    @classmethod
    def unregister(cls, name: str):
        """
        Unregister a retriever type.

        Args:
            name: retriever name
        """
        if name in cls._registry:
            del cls._registry[name]
            if name in cls._default_configs:
                del cls._default_configs[name]
            print(f"Unregistered retriever: {name}")

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available retriever types.

        Returns:
            List of retriever names
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, field_csv: str, table_csv: str,
               config: Optional[RetrieverConfig] = None,
               **kwargs) -> BaseRetriever:
        """
        Create a retriever instance.

        Args:
            name: retriever name
            field_csv: path to the field CSV
            table_csv: path to the table CSV
            config: retriever configuration
            **kwargs: extra parameters

        Returns:
            A retriever instance

        Raises:
            ValueError: if the retriever type is not registered
        """
        if name not in cls._registry:
            available = cls.list_available()
            raise ValueError(
                f"Unknown retriever: {name}. "
                f"Available retrievers: {available}"
            )

        retriever_class = cls._registry[name]

        # Fall back to the default config when none is provided
        if config is None and name in cls._default_configs:
            config = cls._default_configs[name]

        return retriever_class(field_csv, table_csv, config=config, **kwargs)

    @classmethod
    def create_all(cls, field_csv: str, table_csv: str,
                   names: Optional[List[str]] = None,
                   configs: Optional[Dict[str, RetrieverConfig]] = None) -> Dict[str, BaseRetriever]:
        """
        Create multiple retrievers in batch.

        Args:
            field_csv: path to the field CSV
            table_csv: path to the table CSV
            names: names of the retrievers to create (None means all)
            configs: dict of retriever configurations

        Returns:
            Dict of retriever instances {name: retriever}
        """
        if names is None:
            names = cls.list_available()

        configs = configs or {}
        retrievers = {}

        for name in names:
            config = configs.get(name)
            try:
                retrievers[name] = cls.create(name, field_csv, table_csv, config)
            except Exception as e:
                print(f"Failed to create retriever {name}: {e}")

        return retrievers

    @classmethod
    def get_config_template(cls, name: str) -> Optional[RetrieverConfig]:
        """
        Get the config template of a retriever.

        Args:
            name: retriever name

        Returns:
            The config template
        """
        return cls._default_configs.get(name)


class RetrieverManager:
    """
    Retriever manager.

    Manages, trains, and evaluates multiple retrievers.
    """

    def __init__(self, field_csv: str, table_csv: str, qa_csv: str = None):
        """
        Initialize the manager.

        Args:
            field_csv: path to the field CSV
            table_csv: path to the table CSV
            qa_csv: path to the QA data CSV
        """
        self.field_csv = field_csv
        self.table_csv = table_csv
        self.qa_csv = qa_csv

        self.retrievers: Dict[str, BaseRetriever] = {}
        self.qa_df = pd.read_csv(qa_csv) if qa_csv else None

    def add_retriever(self, name: str, config: Optional[RetrieverConfig] = None):
        """
        Add a retriever.

        Args:
            name: retriever name
            config: retriever configuration
        """
        retriever = RetrieverFactory.create(name, self.field_csv, self.table_csv, config)
        self.retrievers[name] = retriever

    def remove_retriever(self, name: str):
        """Remove a retriever."""
        if name in self.retrievers:
            del self.retrievers[name]

    def get_retriever(self, name: str) -> Optional[BaseRetriever]:
        """Get a retriever by name."""
        return self.retrievers.get(name)

    def fit_all(self, train_data: pd.DataFrame = None):
        """
        Train all retrievers.

        Args:
            train_data: training data
        """
        for name, retriever in self.retrievers.items():
            print(f"Fitting {name}...")
            retriever.fit(train_data)

    def fit(self, name: str, train_data: pd.DataFrame = None):
        """Train a specific retriever."""
        if name in self.retrievers:
            self.retrievers[name].fit(train_data)

    def evaluate_all(self, test_data: pd.DataFrame, k: int = 5) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate all retrievers.

        Args:
            test_data: test data
            k: number of top-k results

        Returns:
            Dict of evaluation results {name: metrics}
        """
        results = {}
        for name, retriever in self.retrievers.items():
            print(f"Evaluating {name}...")
            if retriever.is_fitted:
                results[name] = retriever.evaluate(test_data, k)
            else:
                print(f"  Skipping {name} (not fitted)")
        return results

    def evaluate(self, name: str, test_data: pd.DataFrame, k: int = 5) -> Optional[EvaluationMetrics]:
        """Evaluate a specific retriever."""
        if name in self.retrievers and self.retrievers[name].is_fitted:
            return self.retrievers[name].evaluate(test_data, k)
        return None

    def compare(self, test_data: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """
        Compare the performance of all retrievers.

        Args:
            test_data: test data
            k: number of top-k results

        Returns:
            DataFrame with the comparison results
        """
        results = self.evaluate_all(test_data, k)

        rows = []
        for name, metrics in results.items():
            rows.append({
                'Method': name,
                'Table Accuracy': metrics.table_accuracy,
                'SQL Accuracy': metrics.sql_accuracy,
                'Avg Query Time (s)': metrics.avg_query_time,
                'Avg Memory (MB)': metrics.avg_memory_mb,
                'Total Queries': metrics.total_queries
            })

        return pd.DataFrame(rows)

    def run_experiment(self, train_ratio: float = 0.8,
                       k: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Run a complete experiment.

        Args:
            train_ratio: training-set ratio
            k: number of top-k results

        Returns:
            Experiment results
        """
        if self.qa_df is None:
            raise ValueError("QA data not provided")

        import numpy as np

        # Split the data
        indices = np.random.permutation(len(self.qa_df))
        n_train = int(len(indices) * train_ratio)

        train_data = self.qa_df.iloc[indices[:n_train]]
        test_data = self.qa_df.iloc[indices[n_train:]]

        # Train
        self.fit_all(train_data)

        # Evaluate
        comparison_df = self.compare(test_data, k)

        return {
            'comparison': comparison_df,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }


# Auto-register the built-in retrievers.
#
# Public release note: the registry ships exactly the method set reported in
# the paper (see docs/experiment_catalog.md). Legacy experimental retrievers
# from the research repository (a simulated LLM retriever, the TE-RAG v1
# implementation, a graph-only variant, and RAT-SQL/RESDSQL/Seq2SQL/IRNet
# adaptations) are intentionally not registered here; they were never part of
# the paper's reported matrix.
def _auto_register():
    """Auto-register the built-in retrievers."""
    try:
        from retrievers.bm25_retriever import BM25Retriever
        from retrievers.vector_retriever import VectorRetriever

        # Register BM25
        if 'BM25' not in RetrieverFactory._registry:
            RetrieverFactory.register(
                'BM25', BM25Retriever,
                RetrieverConfig(name='BM25', description='BM25 keyword-based retriever')
            )

        # Register Vector (TF-IDF baseline; legacy class name)
        if 'Vector' not in RetrieverFactory._registry:
            RetrieverFactory.register(
                'Vector', VectorRetriever,
                RetrieverConfig(name='Vector', description='TF-IDF vector retriever')
            )

        # Register the Hybrid retriever
        try:
            from retrievers.hybrid_retriever import HybridRetriever
            if 'Hybrid' not in RetrieverFactory._registry:
                RetrieverFactory.register(
                    'Hybrid', HybridRetriever,
                    RetrieverConfig(name='Hybrid', description='Hybrid retriever (BM25 + Vector)')
                )
        except ImportError as e:
            print(f"Warning: Failed to register Hybrid retriever: {e}")

        # Register the historical supervised baselines
        try:
            from retrievers.example_rag_retriever import ExampleRAGRetriever
            if 'ExampleRAG' not in RetrieverFactory._registry:
                RetrieverFactory.register(
                    'ExampleRAG', ExampleRAGRetriever,
                    RetrieverConfig(name='ExampleRAG',
                                    description='Historical Example Schema Retrieval (ExampleRRF)')
                )
        except ImportError as e:
            print(f"Warning: Failed to register ExampleRAG retriever: {e}")

        try:
            from retrievers.cooccurrence_retriever import CooccurrenceRetriever
            if 'Cooccurrence' not in RetrieverFactory._registry:
                RetrieverFactory.register(
                    'Cooccurrence', CooccurrenceRetriever,
                    RetrieverConfig(name='Cooccurrence',
                                    description='Term-Schema Cooccurrence (CooccurRRF)')
                )
        except ImportError as e:
            print(f"Warning: Failed to register Cooccurrence retriever: {e}")

        try:
            from retrievers.dense_retriever import DenseRetriever
            if 'Dense' not in RetrieverFactory._registry:
                RetrieverFactory.register(
                    'Dense', DenseRetriever,
                    RetrieverConfig(name='Dense',
                                    description='Dense retrieval with pretrained embedding')
                )
        except ImportError as e:
            print(f"Warning: Failed to register Dense retriever: {e}")

        try:
            from retrievers.linkalign_retriever import LinkAlignRetriever
            if 'DenseDeepSeek' not in RetrieverFactory._registry:
                RetrieverFactory.register(
                    'DenseDeepSeek', LinkAlignRetriever,
                    RetrieverConfig(name='DenseDeepSeek',
                                    description='Dense+DeepSeek Schema Selector (custom, NOT LinkAlign)')
                )
        except ImportError as e:
            print(f"Warning: Failed to register LinkAlign retriever: {e}")

        try:
            from retrievers.linkalign_official_retriever import OfficialLinkAlignRetriever
            from retrievers.kasla_retriever import KaSLAAdaptedRetriever
            for name, implementation in [
                ('LinkAlign-SingleDB-Adapted', OfficialLinkAlignRetriever),
                ('KaSLA-Adapted', KaSLAAdaptedRetriever),
            ]:
                if name not in RetrieverFactory._registry:
                    RetrieverFactory.register(name, implementation, RetrieverConfig(name=name))
        except ImportError as e:
            print(f'Warning: reviewer methods unavailable: {e}')

        try:
            from retrievers.matched_schema_retriever import MatchedSchemaRetriever
            if 'MatchedSchema' not in RetrieverFactory._registry:
                RetrieverFactory.register(
                    'MatchedSchema', MatchedSchemaRetriever,
                    RetrieverConfig(name='MatchedSchema',
                                    description='Comment-enriched schema index without history (MetadataMatched)')
                )
        except ImportError as e:
            print(f"Warning: Failed to register MatchedSchema retriever: {e}")

    except ImportError as e:
        print(f"Warning: Failed to auto-register retrievers: {e}")

    # Register TE-RAG V2 (the paper version)
    try:
        from terag.terag_retriever_v2 import TERAGRetrieverV2
        from terag.config import TERAGConfig

        # TE-RAG V2 needs special handling because it requires a config
        # file; wrap it in an adapter class
        class TERAGV2Adapter(BaseRetriever):
            """Adapter that makes TE-RAG V2 compatible with the legacy interface."""

            def __init__(self, field_csv: str, table_csv: str,
                         config: Optional[RetrieverConfig] = None,
                         terag_config_path: str = None):
                if config is None:
                    config = RetrieverConfig(
                        name="TE-RAG-V2",
                        description="Table-Enhanced RAG V2 (paper version)"
                    )
                super().__init__(field_csv, table_csv, config)

                # Load the TE-RAG config.
                #
                # Public release: configuration is ALWAYS explicit. Either a
                # TERAGConfig instance is injected via extra_params
                # ['terag_config'] (what terag_release.runner does, with
                # dataset paths and per-run artifacts dir already set), or a
                # YAML path is passed as terag_config_path. There is no
                # implicit discovery of a config.yaml next to the data and no
                # silent fallback to defaults — data paths would be empty and
                # the failure would surface far from its cause.
                import os

                injected = config.extra_params.get('terag_config') if config else None
                if injected is not None:
                    self.terag_config = injected
                elif terag_config_path is not None:
                    if not os.path.exists(terag_config_path):
                        raise FileNotFoundError(
                            f"TE-RAG-V2: terag_config_path does not exist: "
                            f"{terag_config_path}")
                    self.terag_config = TERAGConfig.from_yaml(terag_config_path)
                else:
                    raise ValueError(
                        "TE-RAG-V2: no TERAGConfig provided. Pass one via "
                        "extra_params['terag_config'] (see "
                        "terag_release.runner.create_retriever) or set "
                        "terag_config_path to a YAML file.")

                self.terag_retriever = None

            def fit(self, train_data: pd.DataFrame = None, diagnostics=None):
                """Train the retriever.

                Leakage fixes:
                - fit(train_data) uses only the given training data and
                  rebuilds from scratch; it no longer implicitly loads
                  artifacts/learned_weights.json (the old implementation
                  overwrote the current ranker with weights trained on a
                  historical split, which could contain samples from the
                  current test set).
                - fit(None) no longer silently restores from artifacts and
                  reports training as successful; the artifacts directory
                  must be named explicitly via
                  extra_params['from_artifacts_dir'], and the origin path
                  of each asset is then recorded.

                ``diagnostics`` (public release) forwards the opt-in
                DiagnosticExport to the inner retriever; it never changes
                retrieval outputs.
                """
                self.asset_sources = []
                if train_data is not None:
                    self.terag_retriever = TERAGRetrieverV2(self.terag_config)
                    if diagnostics is not None:
                        self.terag_retriever.fit(train_data,
                                                 diagnostics=diagnostics)
                    else:
                        self.terag_retriever.fit(train_data)
                    # Expose the annotation-channel outputs of the mined and
                    # transferred annotations (miner / transfer_result)
                    self.miner = getattr(self.terag_retriever, 'miner', None)
                    self.transfer_result = getattr(self.terag_retriever,
                                                   'transfer_result', None)
                    # Load learned weights from artifacts only on explicit
                    # opt-in (for reproducing historical behavior)
                    weights_opt = (self.config.extra_params or {}).get('load_artifact_weights')
                    if weights_opt:
                        import os
                        import json
                        artifacts_dir = self.terag_config.output.artifacts_dir
                        weights_path = os.path.join(artifacts_dir, 'learned_weights.json')
                        if os.path.exists(weights_path):
                            with open(weights_path, 'r') as f:
                                learned_weights = json.load(f)
                            if self.terag_retriever.ranker is not None:
                                if learned_weights.get('table_weights'):
                                    self.terag_retriever.ranker.table_ranker.set_weights(
                                        learned_weights['table_weights'])
                                if learned_weights.get('field_weights'):
                                    self.terag_retriever.ranker.field_ranker.set_weights(
                                        learned_weights['field_weights'])
                            self.asset_sources.append(weights_path)
                            print(f"TE-RAG V2: explicitly loaded learned weights: {weights_path}")
                        else:
                            print(f"TE-RAG V2: load_artifact_weights=True but {weights_path} not found; "
                                  f"using the fixed default weights from config")
                    self.asset_sources.extend(getattr(self.terag_retriever, 'asset_sources', []))
                else:
                    artifacts_dir = (self.config.extra_params or {}).get('from_artifacts_dir')
                    if not artifacts_dir:
                        raise ValueError(
                            "TE-RAG-V2: fit(None) no longer loads global artifacts silently. "
                            "Pass train_data, or set extra_params['from_artifacts_dir'] "
                            "to explicitly name the artifacts directory to load.")
                    self.terag_retriever = TERAGRetrieverV2.from_artifacts(
                        self.terag_config, artifacts_dir)
                    self.asset_sources.extend(getattr(self.terag_retriever, 'asset_sources', []))
                    print(f"TE-RAG V2: explicitly loaded from artifacts: {artifacts_dir} "
                          f"(sources: {self.asset_sources})")

                self._is_fitted = True

            def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
                """Retrieve."""
                if not self.terag_retriever:
                    return []

                results = self.terag_retriever.retrieve(query, k)

                # Convert to the shared result format
                return [
                    RetrievalResult(
                        table=r.table,
                        table_score=r.table_score,
                        columns=r.columns,
                        metadata=r.metadata
                    )
                    for r in results
                ]

        if 'TE-RAG-V2' not in RetrieverFactory._registry:
            RetrieverFactory.register(
                'TE-RAG-V2', TERAGV2Adapter,
                RetrieverConfig(name='TE-RAG-V2', description='Table-Enhanced RAG V2 (paper version)')
            )

    except ImportError as e:
        print(f"Warning: Failed to register TE-RAG V2: {e}")


# Run auto-registration
_auto_register()
