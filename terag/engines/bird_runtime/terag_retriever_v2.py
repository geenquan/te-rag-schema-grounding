"""TE-RAG V2 retriever.

The paper-version TE-RAG retriever: integrates all components and supports
loading prebuilt artifacts.
"""

import os
import json
import pickle
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from terag.engines.bird_runtime.config import TERAGConfig
from terag.engines.bird_runtime.graph_builder import BipartiteGraphBuilder
from terag.engines.bird_runtime.pattern_miner import PatternMiner
from terag.engines.bird_runtime.index_builder import IndexBuilder
from terag.engines.bird_runtime.feature_extractor import FeatureExtractor
from terag.engines.bird_runtime.ranker import TableRanker, FieldRanker, CombinedRanker, RankingResult
from terag.engines.bird_runtime.weight_learner import WeightLearner


@dataclass
class RetrievalResultV2:
    """Retrieval result V2."""
    table: str
    table_score: float
    columns: List[Tuple[str, float]]
    metadata: Dict = field(default_factory=dict)


class TERAGRetrieverV2:
    """TE-RAG V2 retriever.

    The paper-version TE-RAG integrating all components:
    - bipartite graph
    - pattern mining
    - BM25 index
    - feature extraction
    - learned-weight ranking

    Properties:
    - supports loading prebuilt artifacts
    - supports ablation switches
    - unified configuration management

    Usage:
        # Option 1: load from artifacts
        retriever = TERAGRetrieverV2.from_artifacts(config)

        # Option 2: build from scratch
        retriever = TERAGRetrieverV2(config)
        retriever.fit(train_data)

        # Retrieve
        results = retriever.retrieve(query, k=5)
    """

    def __init__(self, config: TERAGConfig):
        """Initialize the retriever.

        Args:
            config: TE-RAG configuration.
        """
        self.config = config

        # Components
        self.graph = None
        self.patterns = None
        self.index = None
        self.feature_extractor = None
        self.ranker = None

        # Data
        self.train_data = None
        self.field_df = pd.read_csv(config.data.field_csv)
        self.table_df = pd.read_csv(config.data.table_csv)

        # Record the actually loaded external asset sources (P1-A: fit loads
        # no historical assets by default)
        self.asset_sources: List[str] = []

        # State
        self._is_fitted = False

        # P2-D3: B/C annotation channel artifacts (available after fit;
        # default None = off)
        self.miner = None
        self.transfer_result = None
        self._annotation_terms = {}
        self._diagnostics = None

    @classmethod
    def from_artifacts(cls, config: TERAGConfig, artifacts_dir: str = None) -> 'TERAGRetrieverV2':
        """Create a retriever from prebuilt artifacts.

        Args:
            config: TE-RAG configuration.
            artifacts_dir: artifacts directory path (defaults to the config's).

        Returns:
            TERAGRetrieverV2 instance.
        """
        artifacts_dir = artifacts_dir or config.output.artifacts_dir

        retriever = cls(config)

        # Load the graph
        graph_path = os.path.join(artifacts_dir, 'graph.pkl')
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                retriever.graph = pickle.load(f)
            retriever.asset_sources.append(graph_path)
            print(f"Loaded graph: {graph_path}")

        # Load the patterns
        patterns_path = os.path.join(artifacts_dir, 'patterns.jsonl')
        if os.path.exists(patterns_path):
            miner = PatternMiner(config)
            retriever.patterns = miner.load(patterns_path)
            retriever.asset_sources.append(patterns_path)
            print(f"Loaded patterns: {patterns_path}")

        # Load the index
        index_dir = os.path.join(artifacts_dir, 'bm25_index')
        if os.path.exists(index_dir):
            builder = IndexBuilder(config)
            retriever.index = builder.load(index_dir)
            retriever.asset_sources.append(index_dir)
            print(f"Loaded index: {index_dir}")

        # Load the rankers
        table_ranker = TableRanker(config)
        field_ranker = FieldRanker(config)

        table_ranker_path = os.path.join(artifacts_dir, 'table_ranker.pkl')
        if os.path.exists(table_ranker_path):
            table_ranker.load(table_ranker_path)
            retriever.asset_sources.append(table_ranker_path)

        field_ranker_path = os.path.join(artifacts_dir, 'field_ranker.pkl')
        if os.path.exists(field_ranker_path):
            field_ranker.load(field_ranker_path)
            retriever.asset_sources.append(field_ranker_path)

        retriever.ranker = CombinedRanker(config, table_ranker, field_ranker)

        # Load the training data (when present)
        train_path = config.get_split_path('train')
        if os.path.exists(train_path):
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line))
            retriever.train_data = pd.DataFrame(train_data)
            retriever.asset_sources.append(train_path)

        # Create the feature extractor
        if retriever.graph and retriever.index:
            train_df = retriever.train_data if retriever.train_data is not None and not retriever.train_data.empty else pd.DataFrame()
            patterns_dict = retriever.patterns if retriever.patterns is not None else {}

            retriever.feature_extractor = FeatureExtractor(
                config,
                retriever.graph,
                retriever.index,
                patterns_dict,
                train_df,
                retriever.field_df,
                retriever.table_df
            )

        # P1-A: missing key assets must not be reported as a training success
        if retriever.graph is None or retriever.index is None:
            raise ValueError(
                f"from_artifacts: {artifacts_dir} lacks graph.pkl or bm25_index/, "
                f"cannot form a retrievable instance (refusing silent degradation)")

        retriever._is_fitted = True
        return retriever

    def fit(self, train_data: pd.DataFrame, load_artifact_weights: bool = False,
            diagnostics=None):
        """Train the retriever.

        Uses only the passed train_data and the fixed parameters of the
        current config to build graph/patterns/index/features from scratch.
        By default no historical artifacts are loaded (P1-A leak fix: the
        old implementation implicitly loaded table_ranker.pkl /
        field_ranker.pkl, whose weights could come from historical splits
        containing the current test set).

        P2-D3: the B/C annotation channels are wired into the mainline via
        config.annotations - with use_history=True, one fit does: training
        logs -> direct B annotations -> (optional) constrained C transfer ->
        IndexBuilder (extra_table_texts passed in memory) -> feature
        extractor (column-ranking annotation bonus). Switches support
        ablation; B/C are generated once and injected deduplicated; no
        evaluation SQL/gold is read, no global historical assets loaded.

        Args:
            train_data: training data (all supervision allowed in the
                current run).
            diagnostics: default None; an explicit DiagnosticExport only
                exports evidence newly computed here.
            load_artifact_weights: explicit opt-in to load ranker weights
                from artifacts (reproducing historical behavior only);
                asset sources are recorded when enabled.
        """
        from terag.engines.bird_runtime.config import AnnotationsConfig
        self._diagnostics = diagnostics
        ann_cfg = getattr(self.config, 'annotations',
                          AnnotationsConfig())  # default off (legacy behavior)
        self.train_data = train_data

        # 1. Build the graph
        print("Building the bipartite graph...")
        graph_builder = BipartiteGraphBuilder(self.config)
        self.graph = graph_builder.build(train_data)

        # 2. Mine patterns (the legacy keyword pipeline - coexists with the
        #    B/C annotations, no cross injection; its _generalize_patterns
        #    does not take part in the new transfer path)
        print("Mining query patterns...")
        element_to_queries = graph_builder.get_element_to_queries(self.graph, train_data)
        pattern_miner = PatternMiner(self.config)
        self.patterns = pattern_miner.mine(element_to_queries)

        # 2b. B/C annotations (P2-D3 mainline wiring: generated once,
        #     passed in memory)
        self.miner = None
        self.transfer_result = None
        extra_texts = None
        texts_b, texts_c = {}, {}
        self._annotation_terms = {}   # {table.column: [term, ...]} column-ranking bonus
        if ann_cfg.use_history:
            from terag.engines.bird_runtime.retrievers.annotation_miner import DirectAnnotationMiner
            print("Mining direct annotations (B)...")
            self.miner = DirectAnnotationMiner(self.field_df, self.table_df,
                                               dialect='mysql').mine(train_data)
            texts_b = self.miner.table_annotation_texts(
                self.config.index.field_weights.pattern)
            # B-class column bonus terms
            for a in self.miner.annotations:
                if a['type'] == 'TERM' and a.get('status') == 'confirmed' \
                        and a.get('target'):
                    self._annotation_terms.setdefault(a['target'], []).append(
                        str(a['content']))
            extra_texts = dict(texts_b)
            if ann_cfg.use_transfer:
                from terag.engines.bird_runtime.retrievers.annotation_transfer import (
                    SchemaInfo, transfer_annotations, transfer_index_texts,
                    transfer_column_bonus_map)
                print("Constrained transfer annotations (C)...")
                columns = {}
                for t, g in self.field_df.groupby('table'):
                    columns[t] = (g[['field_name', 'field_name_desc',
                                    'field_type']]
                                  .rename(columns={'field_name_desc': 'field_desc'})
                                  .to_dict('records'))
                table_desc = {r['table']: r.get('table_desc', '')
                              for _, r in self.table_df.iterrows()}
                self.transfer_result = transfer_annotations(
                    self.miner.annotations,
                    SchemaInfo(columns=columns, table_desc=table_desc))
                texts_c = transfer_index_texts(
                    self.transfer_result, self.config.index.field_weights.pattern)
                for t, items in texts_c.items():
                    merged = list(extra_texts.get(t, []))
                    existing = {txt for txt, _w in merged}
                    for txt, w in items:
                        if txt not in existing:   # B/C same-text dedup
                            merged.append((txt, w))
                            existing.add(txt)
                    extra_texts[t] = merged
                for key, terms in transfer_column_bonus_map(
                        self.transfer_result).items():
                    lst = self._annotation_terms.setdefault(key, [])
                    for term in terms:
                        if term not in lst:
                            lst.append(term)

        # 3. Build the index (B/C injected in memory via extra_table_texts)
        print("Building the index...")
        index_builder = IndexBuilder(self.config)
        self.index = index_builder.build(extra_table_texts=extra_texts)

        # 4. Create the feature extractor
        self.feature_extractor = FeatureExtractor(
            self.config,
            self.graph,
            self.index,
            self.patterns,
            train_data,
            self.field_df,
            self.table_df
        )
        # Column-ranking annotation bonus interface (P2-D3): the feature
        # extractor can query B/C term hits
        self.feature_extractor.annotation_terms = dict(self._annotation_terms)

        # 5. Create the ranker (fixed weights from config.feature by default)
        self.ranker = CombinedRanker(self.config)

        # Historical artifact weights load only on explicit opt-in
        # (P1-A leak fix)
        if load_artifact_weights:
            table_ranker_path = self.config.get_artifact_path('table_ranker.pkl')
            field_ranker_path = self.config.get_artifact_path('field_ranker.pkl')

            if os.path.exists(table_ranker_path):
                self.ranker.table_ranker.load(table_ranker_path)
                self.asset_sources.append(table_ranker_path)

            if os.path.exists(field_ranker_path):
                self.ranker.field_ranker.load(field_ranker_path)
                self.asset_sources.append(field_ranker_path)

        if self._diagnostics is not None:
            self._diagnostics.record_fit(self, train_data, texts_b, texts_c, extra_texts)
        self._is_fitted = True
        print(f"TE-RAG V2 training finished (ranker weight source: "
              f"{'artifacts(explicit load)' if load_artifact_weights else 'fixed config defaults'})")

    def retrieve(self, query: str, k: int = 5, *, diagnostic_qid=None) -> List[RetrievalResultV2]:
        """Retrieve.

        Args:
            query: query text.
            k: top-k results to return.

        Returns:
            [RetrievalResultV2, ...]
        """
        if not self._is_fitted:
            raise RuntimeError("Retriever has not been fitted. Call fit() first.")

        if diagnostic_qid is not None and self._diagnostics is None:
            raise ValueError('diagnostic_qid requires explicit diagnostics in fit')
        trace = {} if diagnostic_qid is not None else None
        rank_kwargs = {'k_tables': k, 'k_fields': 10}
        if trace is not None:
            rank_kwargs['diagnostics'] = trace
        ranking_results = self.ranker.rank(self.feature_extractor, query, **rank_kwargs)

        # Convert to RetrievalResultV2
        results = []
        for r in ranking_results:
            results.append(RetrievalResultV2(
                table=r.table,
                table_score=r.table_score,
                columns=r.columns,
                metadata={'method': 'TE-RAG-V2'}
            ))

        if trace is not None:
            self._diagnostics.record_query(diagnostic_qid, trace)
        return results

    def retrieve_with_ablation(
        self,
        query: str,
        k: int = 5,
        **ablation_overrides
    ) -> List[RetrievalResultV2]:
        """Retrieve with ablation switches.

        Args:
            query: query text.
            k: top-k results to return.
            **ablation_overrides: ablation switch overrides.

        Returns:
            [RetrievalResultV2, ...]
        """
        # Temporarily override the ablation switches
        original_values = {}
        for key, value in ablation_overrides.items():
            if hasattr(self.config.ablation, key):
                original_values[key] = getattr(self.config.ablation, key)
                setattr(self.config.ablation, key, value)

        try:
            return self.retrieve(query, k)
        finally:
            # Restore the original values
            for key, value in original_values.items():
                setattr(self.config.ablation, key, value)

    def build_with_ablation(
        self,
        train_data: pd.DataFrame,
        ablation_config: Dict[str, bool],
        output_dir: str
    ) -> 'TERAGRetrieverV2':
        """Rebuild artifacts under an ablation configuration.

        Args:
            train_data: training data.
            ablation_config: ablation configuration, e.g.
                {'use_graph_weight': False, 'use_template_mining': False}
            output_dir: output directory path.

        Returns:
            The rebuilt TERAGRetrieverV2 instance.
        """
        from terag.engines.bird_runtime.graph_builder import BipartiteGraphBuilder
        from terag.engines.bird_runtime.pattern_miner import PatternMiner
        from terag.engines.bird_runtime.index_builder import IndexBuilder

        print(f"Rebuilding artifacts for ablation config: {ablation_config}")
        print(f"Output directory: {output_dir}")

        # Temporarily modify the ablation configuration
        original_values = {}
        for key, value in ablation_config.items():
            if hasattr(self.config.ablation, key):
                original_values[key] = getattr(self.config.ablation, key)
                setattr(self.config.ablation, key, value)

        try:
            # 1. Build the graph
            print("\n[1/3] Building the bipartite graph...")
            graph_builder = BipartiteGraphBuilder(self.config)
            graph = graph_builder.build(train_data)

            graph_path = os.path.join(output_dir, 'graph.pkl')
            graph_builder.save(graph, graph_path)

            # 2. Mine the patterns
            print("\n[2/3] Mining query patterns...")
            element_to_queries = graph_builder.get_element_to_queries(graph, train_data)
            pattern_miner = PatternMiner(self.config)
            patterns = pattern_miner.mine(element_to_queries)

            patterns_path = os.path.join(output_dir, 'patterns.jsonl')
            pattern_miner.save(patterns, patterns_path)

            # 3. Build the index
            print("\n[3/3] Building the index...")
            index_builder = IndexBuilder(self.config)
            index = index_builder.build()

            index_dir = os.path.join(output_dir, 'bm25_index')
            index_builder.save(index, index_dir)

            # 4. Create the feature extractor
            feature_extractor = FeatureExtractor(
                self.config,
                graph,
                index,
                patterns,
                train_data,
                self.field_df,
                self.table_df
            )

            # 5. Create the ranker
            ranker = CombinedRanker(self.config)

            # Create the new retriever
            new_retriever = TERAGRetrieverV2(self.config)
            new_retriever.graph = graph
            new_retriever.patterns = patterns
            new_retriever.index = index
            new_retriever.feature_extractor = feature_extractor
            new_retriever.ranker = ranker
            new_retriever.train_data = train_data
            new_retriever._is_fitted = True

            print(f"\nAblation artifacts build finished: {output_dir}")

            return new_retriever

        finally:
            # Restore the original configuration
            for key, value in original_values.items():
                setattr(self.config.ablation, key, value)

    def get_table_weights(self) -> Dict[str, float]:
        """Get the table ranking weights."""
        return self.ranker.table_ranker.get_weights()

    def get_field_weights(self) -> Dict[str, float]:
        """Get the field ranking weights."""
        return self.ranker.field_ranker.get_weights()

    @property
    def is_fitted(self) -> bool:
        """Whether the retriever is fitted."""
        return self._is_fitted


def main():
    """TE-RAG V2 retrieval demo."""
    import sys
    from pathlib import Path

    # Add the project root
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Load the configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("TE-RAG V2 retriever demo")
    print("=" * 60)

    # Try loading from artifacts
    artifacts_dir = config.output.artifacts_dir
    if os.path.exists(os.path.join(artifacts_dir, 'graph.pkl')):
        print("\nLoading from artifacts...")
        retriever = TERAGRetrieverV2.from_artifacts(config, artifacts_dir)
    else:
        print("\nBuilding from scratch...")
        retriever = TERAGRetrieverV2(config)

        # Load the training data
        train_data = []
        with open(config.get_split_path('train'), 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))
        train_df = pd.DataFrame(train_data)

        retriever.fit(train_df)

    # Test retrieval
    print("\nTest retrieval:")
    test_queries = [
        "查询公司的售电量",
        "统计用户的电费金额",
        "分析供电所的回收率",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.table} (score: {result.table_score:.4f})")
            for col, score in result.columns[:3]:
                print(f"      - {col}: {score:.4f}")

    # Print the weights
    print("\nRanking weights:")
    print(f"  table ranking: {retriever.get_table_weights()}")
    print(f"  field ranking: {retriever.get_field_weights()}")


if __name__ == "__main__":
    main()
