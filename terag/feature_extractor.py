"""
Feature extractor.

Extracts table-level and column-level features.
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

from terag.config import TERAGConfig, FeatureConfig
from terag.index_builder import BM25Retriever
from terag.pattern_miner import Pattern, PatternMiner
from terag.code_mapper import CodeMapper
from terag.tokenization import tokenize_text
from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields


class FeatureExtractor:
    """
    Feature extractor.

    Extracts two feature families:
    1. Table-level: bm25_score, graph_score, pattern_score
    2. Column-level: direct_match, graph_propagation, role_prior, train_recommend

    Usage:
        extractor = FeatureExtractor(config, graph, index, patterns, train_data)
        table_features = extractor.extract_table_features(query)
        field_features = extractor.extract_field_features(query, table_name)
    """

    def __init__(
        self,
        config: TERAGConfig,
        graph: nx.Graph,
        index: Dict,
        patterns: Dict[str, List[Pattern]],
        train_data: pd.DataFrame,
        field_df: pd.DataFrame,
        table_df: pd.DataFrame
    ):
        """
        Initialize the feature extractor.

        Args:
            config: TE-RAG configuration
            graph: bipartite graph
            index: BM25 index
            patterns: pattern library
            train_data: training data
            field_df: field DataFrame
            table_df: table DataFrame
        """
        self.config = config
        self.graph = graph
        self.patterns = patterns
        self.train_data = train_data
        self.field_df = field_df
        self.table_df = table_df

        # BM25 retriever
        self.bm25_retriever = BM25Retriever(index)

        # Tokenization language (PowerGrid=chinese / BIRD=english)
        self.language = getattr(config.tokenizer, 'language', 'chinese')

        # Load feature computation parameters (ablation-aware)
        self._load_feature_params()

        # CodeMapper for query expansion
        self.code_mapper = None
        self._init_code_mapper()

        # Synonym dict (loaded from config)
        self.synonyms_dict = getattr(config.feature, 'synonyms_dict', {})

        # Precompute training-query tokens and gold labels (multi-table compatible)
        self._train_tokens_cache = {}
        self._train_gold_cache: Dict[int, Tuple[List[str], List[Tuple[str, str]]]] = {}
        if train_data is not None and not train_data.empty:
            for idx in range(len(train_data)):
                row = train_data.iloc[idx]
                self._train_tokens_cache[idx] = set(tokenize_text(str(row['question']), self.language))
                self._train_gold_cache[idx] = (
                    parse_ground_truth_tables(row),
                    parse_ground_truth_fields(row),
                )

        # Query token cache
        self._query_tokens_cache = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text in the active dataset's language."""
        return tokenize_text(text, self.language)

    def _init_code_mapper(self):
        """Initialize the CodeMapper used for query expansion.

        Leak safety: artifacts/code_mappings.json (a product of historical
        splits) is no longer loaded; code-value mappings are mined only from
        the training data passed to the current fit(train_data).
        """
        try:
            self.code_mapper = CodeMapper(self.config)
            if self.train_data is not None and not self.train_data.empty \
                    and 'sql' in self.train_data.columns:
                mined = self.code_mapper.mine_from_sql(self.train_data)
                n_mappings = sum(len(v) for v in mined.values()) if mined else 0
                print(f"CodeMapper: mined code-value mappings from the current training set ({n_mappings} entries)")
        except Exception as e:
            print(f"Warning: CodeMapper initialization failed: {e}")
            self.code_mapper = None

    def _load_feature_params(self):
        """Load the feature computation parameters from the config."""
        feat = self.config.feature

        # Table-level parameters
        self.graph_propagation_weight = getattr(feat, 'graph_propagation_weight', 3.0)
        self.template_match_weight = getattr(feat, 'template_match_weight', 0.6)

        # Column-level parameters
        self.direct_match_multiplier = getattr(feat, 'direct_match_multiplier', 1.5)
        self.field_name_match_bonus = getattr(feat, 'field_name_match_bonus', 0.4)
        self.field_graph_score_cap = getattr(feat, 'field_graph_score_cap', 5.0)
        self.train_recommend_multiplier = getattr(feat, 'train_recommend_multiplier', 1.5)
        self.train_recommend_cap = getattr(feat, 'train_recommend_cap', 3.0)

    def _get_query_tokens(self, query: str) -> Set[str]:
        """Return the query's token set (cached).

        Mirrors the expand step of Phase 1:
        1. Tokenization (jieba for Chinese / regex for English, per dataset config)
        2. CodeMapper expansion (code-value mappings)
        3. Synonym expansion
        """
        if query not in self._query_tokens_cache:
            # 1. Base tokenization
            tokens = set(self._tokenize(query))

            # 2. CodeMapper expansion (code-value mappings)
            if self.code_mapper is not None:
                try:
                    expanded_query = self.code_mapper.expand_query(query)
                    # Re-tokenize the expanded query
                    if expanded_query != query:
                        expanded_tokens = set(self._tokenize(expanded_query))
                        tokens = tokens | expanded_tokens
                except Exception:
                    pass

            # 3. Synonym expansion
            if self.synonyms_dict:
                expanded = set()
                for token in tokens:
                    if token in self.synonyms_dict:
                        expanded.update(self.synonyms_dict[token])
                tokens = tokens | expanded

            self._query_tokens_cache[query] = tokens
        return self._query_tokens_cache[query]

    def extract_table_features(self, query: str) -> Dict[str, Dict[str, float]]:
        """
        Extract table-level features.

        Args:
            query: query text

        Returns:
            {table_node: {'bm25_score': float, 'graph_score': float, 'pattern_score': float}}
        """
        features = {}
        query_tokens = self._get_query_tokens(query)

        # 1. BM25 retrieval
        bm25_results = self.bm25_retriever.search(query, k=len(self.table_df))
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}

        # 2. Graph propagation score
        graph_scores = self._compute_graph_scores(query, query_tokens)

        # 3. Pattern match score
        pattern_scores = self._compute_pattern_scores(query, query_tokens)

        # Merge features (sorted for cross-process determinism: set iteration
        # order is affected by hash randomization)
        all_tables = sorted(set(bm25_scores.keys()) | set(graph_scores.keys()) | set(pattern_scores.keys()))

        for table_node in all_tables:
            features[table_node] = {
                'bm25_score': bm25_scores.get(table_node, 0.0),
                'graph_score': graph_scores.get(table_node, 0.0),
                'pattern_score': pattern_scores.get(table_node, 0.0),
            }

        return features

    def _compute_graph_scores(self, query: str, query_tokens: Set[str]) -> Dict[str, float]:
        """Compute the graph propagation score.

        Ablation switch: returns an empty dict when use_graph_weight=False.
        """
        # Ablation: skip entirely when graph weighting is off
        if not self.config.ablation.use_graph_weight:
            return {}

        if self.train_data is None or self.train_data.empty:
            return {}

        scores = defaultdict(float)
        threshold = self.config.feature.graph_propagation_threshold

        for idx in range(len(self.train_data)):
            hist_tokens = self._train_tokens_cache.get(idx, set())
            if not hist_tokens:
                continue

            # Jaccard similarity
            intersection = len(query_tokens & hist_tokens)
            if intersection == 0:
                continue

            union = len(query_tokens | hist_tokens)
            sim = intersection / union if union > 0 else 0

            if sim > threshold:
                # Gold tables of this historical query (multi-table queries
                # propagate score to each table)
                gold_tables, _ = self._train_gold_cache.get(idx, ([], []))
                for table_id in gold_tables:
                    table_node = f"T:{table_id}"
                    # use the configured parameter
                    scores[table_node] += sim * self.graph_propagation_weight

        return dict(scores)

    def _compute_pattern_scores(self, query: str, query_tokens: Set[str]) -> Dict[str, float]:
        """Compute the pattern match score.

        Ablation switch: returns an empty dict when use_template_mining=False.
        """
        if not self.config.ablation.use_template_mining:
            return {}

        scores = {}

        for table_node, pattern_list in self.patterns.items():
            if not pattern_list:
                continue

            max_score = 0.0
            for pattern in pattern_list:
                pattern_tokens = set(self._tokenize(pattern.pattern_text))
                if pattern_tokens:
                    overlap = len(query_tokens & pattern_tokens) / len(pattern_tokens)
                    max_score = max(max_score, overlap)

            if max_score > 0:
                # use the configured parameter
                scores[table_node] = max_score * self.template_match_weight

        return scores

    def extract_field_features(
        self,
        query: str,
        table_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract column-level features.

        Args:
            query: query text
            table_name: table name

        Returns:
            {field_name: {'direct_match': float, 'graph_propagation': float,
                          'role_prior': float, 'train_recommend': float}}
        """
        features = {}
        query_tokens = self._get_query_tokens(query)
        query_terms = [t for t in self._tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        # Columns of this table
        columns = self.field_df[self.field_df['table'] == table_name]

        for _, col_row in columns.iterrows():
            field_name = col_row['field_name']
            field_desc = str(col_row.get('field_name_desc', ''))
            col_node = f"C:{table_name}.{field_name}"

            # 1. Direct match score
            s_direct = self._compute_direct_match_score(
                field_name, field_desc, query_terms, query_term_set
            )

            # 2. Graph propagation score
            s_graph = self._compute_field_graph_score(col_node, query, query_tokens)

            # 3. Pattern match score (role_prior)
            s_pattern = self._compute_field_pattern_score(col_node, query_tokens)

            # 4. Training-data column recommendation
            s_train = self._compute_train_field_score(table_name, field_name, query_tokens)

            # 5. Annotation bonus: query terms that hit this column's confirmed
            #    term annotations (history aliases/co-occurrence plus transfer);
            #    w_ann=0.5 per hit, matching the ablation baselines.
            #    annotation_terms is injected by TERAGRetrieverV2.fit; empty = off
            s_ann = 0.0
            ann_terms = getattr(self, 'annotation_terms', None) or {}
            ann_key = f"{table_name.lower()}.{field_name.lower()}"
            for term in ann_terms.get(ann_key, []):
                if term in query_term_set:
                    s_ann += 0.5

            features[field_name] = {
                'direct_match': s_direct,
                'graph_propagation': s_graph,
                'role_prior': s_pattern,
                'train_recommend': s_train,
                'annotation_hit': s_ann,
            }

        return features

    def _compute_direct_match_score(
        self,
        field_name: str,
        field_desc: str,
        query_terms: List[str],
        query_term_set: Set[str]
    ) -> float:
        """Compute the direct match score."""
        field_text = f"{field_name} {field_desc}"
        field_terms = set(t for t in self._tokenize(field_text) if len(t) > 1)

        overlap = len(query_term_set & field_terms)
        # use the configured parameter
        s_direct = overlap / max(len(query_terms), 1) * self.direct_match_multiplier

        # Field name/description directly contains a query term
        for term in query_terms:
            if term in field_name or term in field_desc:
                # use the configured parameter
                s_direct += self.field_name_match_bonus

        return s_direct

    def _compute_field_graph_score(
        self,
        col_node: str,
        query: str,
        query_tokens: Set[str]
    ) -> float:
        """Compute the column graph propagation score.

        Ablation switch: returns 0.0 immediately when use_graph_weight=False.
        """
        # Ablation: skip when graph weighting is off
        if not self.config.ablation.use_graph_weight:
            return 0.0

        if self.train_data is None or self.train_data.empty:
            return 0.0

        if not self.graph.has_node(col_node):
            return 0.0

        score = 0.0
        threshold = self.config.feature.graph_propagation_threshold

        for idx in range(len(self.train_data)):
            hist_tokens = self._train_tokens_cache.get(idx, set())
            if not hist_tokens:
                continue

            intersection = len(query_tokens & hist_tokens)
            if intersection == 0:
                continue

            union = len(query_tokens | hist_tokens)
            sim = intersection / union if union > 0 else 0

            if sim > threshold:
                query_node = f"Q_{idx}"
                if self.graph.has_edge(query_node, col_node):
                    weight = self.graph[query_node][col_node].get('weight', 1.0)
                    score += sim * weight

        # use the configured parameter
        return min(score, self.field_graph_score_cap)

    def _compute_field_pattern_score(self, col_node: str, query_tokens: Set[str]) -> float:
        """Compute the column pattern match score.

        Ablation switch: returns 0.0 when use_template_mining=False.
        """
        if not self.config.ablation.use_template_mining:
            return 0.0

        if col_node not in self.patterns:
            return 0.0

        max_sim = 0.0
        for pattern in self.patterns[col_node]:
            pattern_tokens = set(self._tokenize(pattern.pattern_text))
            if pattern_tokens:
                sim = len(query_tokens & pattern_tokens) / len(pattern_tokens)
                max_sim = max(max_sim, sim)

        return max_sim

    def _compute_train_field_score(
        self,
        table_name: str,
        field_name: str,
        query_tokens: Set[str]
    ) -> float:
        """Compute the training-data column recommendation score."""
        if self.train_data is None or self.train_data.empty:
            return 0.0

        score = 0.0
        threshold = self.config.feature.train_similarity_threshold

        for idx in range(len(self.train_data)):
            gold_tables, gold_fields = self._train_gold_cache.get(idx, ([], []))

            # Only training rows whose gold tables include this table
            if table_name not in gold_tables:
                continue

            # Query similarity
            hist_tokens = self._train_tokens_cache.get(idx, set())
            if not hist_tokens:
                continue

            union = len(query_tokens | hist_tokens)
            if union == 0:
                continue

            sim = len(query_tokens & hist_tokens) / union

            if sim > threshold:
                # Only when this column is among the row's gold columns
                if any(t == table_name and f == field_name for t, f in gold_fields):
                    # use the configured parameter
                    score += sim * self.train_recommend_multiplier

        # use the configured parameter
        return min(score, self.train_recommend_cap)
