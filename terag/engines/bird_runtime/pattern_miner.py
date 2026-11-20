"""Pattern miner.

Mines query patterns (entity-attribute, time, aggregation) with pattern
generalization support.
"""

import os
import json
import re
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

from terag.engines.bird_runtime.config import TERAGConfig
from terag.engines.bird_runtime.tokenization import tokenize_text

PATTERN_IMPLEMENTATION = 'deterministic_patterns_v1'


@dataclass
class Pattern:
    """A query pattern."""
    pattern_id: str
    pattern_type: str  # entity_attr, time, aggregation, etc.
    pattern_text: str
    count: int = 1
    source_element: str = ""
    transferred: bool = False

    def to_dict(self) -> dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'pattern_text': self.pattern_text,
            'count': self.count,
            'source_element': self.source_element,
            'transferred': self.transferred,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Pattern':
        return cls(
            pattern_id=data['pattern_id'],
            pattern_type=data['pattern_type'],
            pattern_text=data['pattern_text'],
            count=data['count'],
            source_element=data['source_element'],
            transferred=data['transferred'],
        )


class PatternMiner:
    """Pattern miner.

    Functionality:
    1. Entity-attribute pattern mining.
    2. Time pattern recognition.
    3. Aggregation pattern recognition.
    4. Pattern generalization (similarity-based transfer).

    Usage:
        miner = PatternMiner(config)
        patterns = miner.mine(element_to_queries)
        miner.save(patterns, "artifacts/patterns.jsonl")
    """

    def __init__(self, config: TERAGConfig):
        """Initialize the pattern miner.

        Args:
            config: TE-RAG configuration.
        """
        self.config = config

        # Mining mode: keywords (Chinese business lexicon) / generic
        # (English, learned from historical Query-SQL)
        self.mode = getattr(config.pattern_mining, 'mode', 'keywords')
        self.language = getattr(config.tokenizer, 'language', 'chinese')

        # Keyword configuration (all from config, nothing hardcoded here)
        self.entity_keywords = config.pattern_mining.entity_keywords
        self.attribute_keywords = config.pattern_mining.attribute_keywords
        self.similarity_threshold = config.pattern_mining.similarity_threshold
        self.min_pattern_count = config.pattern_mining.min_pattern_count

        # Aggregation keywords / time patterns (Chinese or English version
        # by mode)
        if self.mode == 'generic':
            self.aggregation_keywords = dict(config.pattern_mining.english_aggregation_keywords)
            self.time_patterns = [tuple(p) for p in config.pattern_mining.english_time_patterns
                                  if isinstance(p, (list, tuple)) and len(p) == 2]
        else:
            self.aggregation_keywords = dict(config.pattern_mining.aggregation_keywords)
            self.time_patterns = [tuple(p) for p in config.pattern_mining.time_patterns
                                  if isinstance(p, (list, tuple)) and len(p) == 2]

        # generic mode: lazy-load the schema vocabulary (used for
        # query<->schema signature token extraction)
        self._schema_vocab: Optional[Set[str]] = None

    def mine(self, element_to_queries: Dict[str, Set[str]]) -> Dict[str, List[Pattern]]:
        """Mine patterns.

        Args:
            element_to_queries: element-to-queries mapping {element_node: set of queries}.

        Returns:
            {element_node: [Pattern, ...]}
        """
        pattern_library = defaultdict(list)

        # Stage 1: direct mining
        for element_node, queries in sorted(element_to_queries.items()):
            patterns = self._mine_patterns_for_element(sorted(queries))
            for pattern_key, pattern_data in sorted(patterns.items()):
                if pattern_data['count'] >= self.min_pattern_count:
                    pattern = Pattern(
                        pattern_id=f"{element_node}_{len(pattern_library[element_node])}",
                        pattern_type=pattern_data['type'],
                        pattern_text=pattern_key,
                        count=pattern_data['count'],
                        source_element=element_node,
                        transferred=False
                    )
                    pattern_library[element_node].append(pattern)

        # Stage 2: pattern generalization (when enabled)
        if self.config.ablation.use_pattern_generalization:
            self._generalize_patterns(pattern_library)

        return dict(pattern_library)

    def _mine_patterns_for_element(self, queries: List[str]) -> Dict[str, dict]:
        """Mine patterns for one element.

        Returns:
            {pattern_text: {'type': str, 'count': int}}
        """
        patterns = defaultdict(lambda: {'type': '', 'count': 0})

        for query in queries:
            # 1. Entity-attribute pattern
            entity_attr_pattern = self._extract_entity_attr_pattern(query)
            if entity_attr_pattern:
                patterns[entity_attr_pattern]['type'] = 'entity_attr'
                patterns[entity_attr_pattern]['count'] += 1

            # 2. Time pattern
            time_pattern = self._extract_time_pattern(query)
            if time_pattern:
                patterns[time_pattern]['type'] = 'time'
                patterns[time_pattern]['count'] += 1

            # 3. Aggregation pattern
            agg_pattern = self._extract_aggregation_pattern(query)
            if agg_pattern:
                patterns[agg_pattern]['type'] = 'aggregation'
                patterns[agg_pattern]['count'] += 1

        return dict(patterns)

    def _load_schema_vocab(self) -> Set[str]:
        """Load the schema vocabulary (generic mode).

        Built by tokenizing table/field names and descriptions; used to
        identify schema-related signature tokens in historical queries -
        keeping the annotation learning grounded in the association between
        historical Query-SQL and the schema instead of a fixed business
        lexicon.
        """
        if self._schema_vocab is not None:
            return self._schema_vocab

        vocab: Set[str] = set()
        try:
            field_csv = self.config.data.field_csv
            table_csv = self.config.data.table_csv
            if field_csv and os.path.exists(field_csv):
                field_df = pd.read_csv(field_csv)
                for _, row in field_df.iterrows():
                    vocab.update(t for t in tokenize_text(str(row['field_name']), self.language) if len(t) > 1)
                    vocab.update(t for t in tokenize_text(str(row.get('field_name_desc', '')), self.language) if len(t) > 1)
            if table_csv and os.path.exists(table_csv):
                table_df = pd.read_csv(table_csv)
                for _, row in table_df.iterrows():
                    vocab.update(t for t in tokenize_text(str(row['table']), self.language) if len(t) > 1)
                    vocab.update(t for t in tokenize_text(str(row.get('table_desc', '')), self.language) if len(t) > 1)
        except Exception as e:
            print(f"  Warning: failed to load the schema vocabulary: {e}")

        self._schema_vocab = vocab
        return vocab

    def _extract_entity_attr_pattern(self, query: str) -> Optional[str]:
        """Extract the entity-attribute pattern."""
        if self.mode == 'generic':
            return self._extract_signature_pattern(query)

        # keywords mode (Chinese business lexicon, from config)
        words = list(tokenize_text(query, self.language))

        entities = [w for w in words if any(k in w for k in self.entity_keywords)]
        attributes = [w for w in words if any(k in w for k in self.attribute_keywords)]

        if entities or attributes:
            parts = []
            if entities:
                parts.append(f"ENTITY:{'|'.join(sorted(set(entities)))}")
            if attributes:
                parts.append(f"ATTR:{'|'.join(sorted(set(attributes)))}")
            return ' '.join(parts)

        return None

    def _extract_signature_pattern(self, query: str) -> Optional[str]:
        """Extract the schema signature pattern (generic mode).

        The content tokens of a historical query that intersect the schema
        vocabulary form the signature:
        SIG:tok1|tok2|...
        """
        vocab = self._load_schema_vocab()
        if not vocab:
            return None

        tokens = [t for t in tokenize_text(query, self.language) if len(t) > 1]
        signature = sorted({t for t in tokens if t in vocab})

        if not signature:
            return None
        return f"SIG:{'|'.join(signature[:8])}"

    def _extract_time_pattern(self, query: str) -> Optional[str]:
        """Extract the time pattern."""
        text = query.lower() if self.mode == 'generic' else query
        for pattern, pattern_type in self.time_patterns:
            if re.search(pattern, text):
                return f"TIME:{pattern_type}"
        return None

    def _extract_aggregation_pattern(self, query: str) -> Optional[str]:
        """Extract the aggregation pattern."""
        text = query.lower() if self.mode == 'generic' else query
        for keyword, agg_type in self.aggregation_keywords.items():
            if keyword in text:
                return f"AGG:{agg_type}"
        return None

    def _generalize_patterns(self, pattern_library: Dict[str, List[Pattern]]):
        """Pattern generalization.

        Elements with few patterns borrow patterns from similar elements.
        """
        element_list = sorted(pattern_library.keys())

        for i, e1 in enumerate(element_list):
            # Elements with few patterns
            if len(pattern_library[e1]) < 3:
                for j, e2 in enumerate(element_list):
                    if i != j and len(pattern_library[e2]) >= 3:
                        # Pattern similarity
                        similarity = self._compute_pattern_similarity(
                            pattern_library[e1],
                            pattern_library[e2]
                        )

                        if similarity > self.similarity_threshold:
                            # Transfer patterns
                            for pattern in pattern_library[e2][:2]:
                                transferred = Pattern(
                                    pattern_id=f"{e1}_{len(pattern_library[e1])}_transferred",
                                    pattern_type=pattern.pattern_type,
                                    pattern_text=pattern.pattern_text,
                                    count=int(pattern.count * similarity),
                                    source_element=e2,
                                    transferred=True
                                )
                                pattern_library[e1].append(transferred)

    def _compute_pattern_similarity(self, patterns1: List[Pattern], patterns2: List[Pattern]) -> float:
        """Pattern similarity between two elements."""
        set1 = set(p.pattern_text for p in patterns1)
        set2 = set(p.pattern_text for p in patterns2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def save(self, pattern_library: Dict[str, List[Pattern]], output_path: str):
        """Save the pattern library."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for element_node, patterns in pattern_library.items():
                record = {
                    'element': element_node,
                    'patterns': [p.to_dict() for p in patterns]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Pattern library saved to: {output_path}")

    def load(self, input_path: str) -> Dict[str, List[Pattern]]:
        """Load a pattern library."""
        pattern_library = {}

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                element = record['element']
                patterns = [Pattern.from_dict(p) for p in record['patterns']]
                pattern_library[element] = patterns

        return pattern_library

    def get_pattern_stats(self, pattern_library: Dict[str, List[Pattern]]) -> Dict:
        """Pattern statistics."""
        stats = {
            'total_elements': len(pattern_library),
            'total_patterns': sum(len(p) for p in pattern_library.values()),
            'transferred_patterns': sum(
                sum(1 for p in patterns if p.transferred)
                for patterns in pattern_library.values()
            ),
            'pattern_types': defaultdict(int),
        }

        for patterns in pattern_library.values():
            for pattern in patterns:
                stats['pattern_types'][pattern.pattern_type] += 1

        stats['pattern_types'] = dict(stats['pattern_types'])

        return stats


def main():
    """Pattern mining demo."""
    import sys
    from pathlib import Path
    import networkx as nx

    # Add the project root
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.engines.bird_runtime.config import TERAGConfig
    from terag.engines.bird_runtime.graph_builder import BipartiteGraphBuilder

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

    # Build the graph
    builder = BipartiteGraphBuilder(config)
    graph = builder.build(train_df)

    # Element-to-queries mapping
    element_to_queries = builder.get_element_to_queries(graph, train_df)
    print(f"Element count: {len(element_to_queries)}")

    # Mine patterns
    miner = PatternMiner(config)
    patterns = miner.mine(element_to_queries)

    # Print statistics
    stats = miner.get_pattern_stats(patterns)
    print(f"\nPattern statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save
    if config.output.save_patterns:
        miner.save(patterns, config.get_artifact_path('patterns.jsonl'))


if __name__ == "__main__":
    main()
