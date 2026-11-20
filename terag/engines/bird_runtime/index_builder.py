"""Index builder.

Builds the BM25 inverted index with multi-weight augmentation support.
"""

import os
import json
import pickle
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from terag.engines.bird_runtime.config import TERAGConfig
from terag.engines.bird_runtime.tokenization import tokenize_text

INDEX_IMPLEMENTATION = 'index_doc_uniqueness_v1'

from typing import Dict, List, Optional, Tuple


class IndexBuilder:
    """Index builder.

    Functionality:
    1. Build the BM25 inverted index.
    2. Multi-weight augmentation (table name=2.0, desc=1.5, field name=1.0).
    3. Optional pattern-library augmentation of the index.

    Usage:
        builder = IndexBuilder(config)
        index = builder.build()
        builder.save(index, "artifacts/bm25_index/")
    """

    def __init__(self, config: TERAGConfig):
        """Initialize the index builder.

        Args:
            config: TE-RAG configuration.
        """
        self.config = config

        # BM25 parameters
        self.k1 = config.index.bm25.k1
        self.b = config.index.bm25.b

        # Tokenization language (PowerGrid=chinese / BIRD=english)
        self.language = getattr(config.tokenizer, 'language', 'chinese')

        # Field weights
        self.field_weights = config.index.field_weights

        # Load table and field data
        self.field_df = pd.read_csv(config.data.field_csv)
        self.table_df = pd.read_csv(config.data.table_csv)

        # Pattern library (optional)
        self.pattern_library = None

    def load_patterns(self, pattern_path: str):
        """Load a pattern library."""
        from terag.engines.bird_runtime.pattern_miner import PatternMiner

        miner = PatternMiner(self.config)
        self.pattern_library = miner.load(pattern_path)
        print(f"Loaded pattern library: {len(self.pattern_library)} elements")

    def build(self, extra_table_texts: Optional[Dict[str, List[Tuple[str, float]]]] = None) -> Dict:
        """Build the index.

        Args:
            extra_table_texts: P2-D1 in-memory annotation channel -
                {table: [(text, weight), ...]} of direct annotations (B class)
                mined from the current training fold, injected into the
                corresponding table documents; no global files involved, no
                dependency on the historical pattern library in artifacts.
                Default None (pure metadata/pattern-library behavior
                unchanged).

        Returns:
            Index dict containing:
            - inverted_index: {term: [(doc_id, tf, doc_length), ...]}
            - document_norms: {doc_id: length}
            - documents: {doc_id: document_text}
            - avgdl: average document length
            - N: total document count
        """
        inverted_index = defaultdict(list)
        document_norms = {}
        documents = {}

        # Whether to use the enhanced index
        use_enhanced = self.config.ablation.use_enhanced_index

        # A document ID has one authoritative metadata record. Match the former
        # final document/norm (last row), preserving first-seen document order,
        # without appending the same document twice to term postings.
        table_rows = {}
        for _, row in self.table_df.iterrows():
            table_rows[row['table']] = row
        for row in table_rows.values():
            table_name = row['table']
            table_node = f"T:{table_name}"
            table_desc = row.get('table_desc', '')

            # Columns of this table
            columns = self.field_df[self.field_df['table'] == table_name]

            # Build the document
            document_parts = []

            if use_enhanced:
                # Enhanced index: multi weights
                document_parts.append((table_name, self.field_weights.table_name))

                if pd.notna(table_desc):
                    document_parts.append((table_desc, self.field_weights.table_desc))

                for _, col_row in columns.iterrows():
                    document_parts.append((col_row['field_name'], self.field_weights.field_name))

                    field_desc = col_row.get('field_name_desc', '')
                    include_desc = getattr(self.config.index, 'include_field_desc', True)
                    if pd.notna(field_desc) and include_desc:
                        document_parts.append((field_desc, self.field_weights.field_desc))

                # Add patterns (file-loaded pattern library - offline path,
                # not used in the official experiments)
                if self.pattern_library and table_node in self.pattern_library:
                    for pattern in self.pattern_library[table_node]:
                        document_parts.append((pattern.pattern_text, self.field_weights.pattern))

                # P2-D1: direct annotations of the current training fold
                # (passed in memory)
                # Ablation switch: with use_annotation_index=False the B/C
                # texts stay out of the index
                ann_cfg = getattr(self.config, 'annotations', None)
                ann_index_on = getattr(ann_cfg, 'use_annotation_index', True) if ann_cfg else True
                if ann_index_on:
                    for text, w in (extra_table_texts or {}).get(table_name, []):
                        document_parts.append((text, w))
            else:
                # Basic index: equal weights
                document_parts.append((table_name, 1.0))

                if pd.notna(table_desc):
                    document_parts.append((table_desc, 1.0))

                for _, col_row in columns.iterrows():
                    document_parts.append((col_row['field_name'], 1.0))

            # Index the document
            self._index_document(
                table_node, document_parts,
                inverted_index, document_norms, documents
            )

        # Compute statistics
        N = len(document_norms)
        avgdl = np.mean(list(document_norms.values())) if document_norms else 1.0

        return {
            'inverted_index': dict(inverted_index),
            'document_norms': document_norms,
            'documents': documents,
            'avgdl': avgdl,
            'N': N,
            'k1': self.k1,
            'b': self.b,
            'language': self.language,
        }

    def _index_document(
        self,
        doc_id: str,
        document_parts: List[Tuple[str, float]],
        inverted_index: Dict,
        document_norms: Dict,
        documents: Dict
    ):
        """Index a single document.

        Args:
            doc_id: document ID.
            document_parts: [(text, weight), ...]
            inverted_index: inverted index.
            document_norms: document lengths.
            documents: document texts.
        """
        term_frequencies = defaultdict(float)
        document_text_parts = []

        for text, weight in document_parts:
            words = tokenize_text(str(text), self.language)
            document_text_parts.append(str(text))

            for word in words:
                if len(word) > 1:
                    term_frequencies[word] += weight

        # Document length
        doc_length = math.sqrt(sum(tf ** 2 for tf in term_frequencies.values()))
        document_norms[doc_id] = doc_length if doc_length > 0 else 1.0

        # Store the document text
        documents[doc_id] = ' '.join(document_text_parts)

        # Add to the inverted index
        for term, tf in term_frequencies.items():
            inverted_index[term].append((doc_id, tf, doc_length))

    def save(self, index: Dict, output_dir: str):
        """Save the index."""
        os.makedirs(output_dir, exist_ok=True)

        # Save the inverted index
        with open(os.path.join(output_dir, 'inverted_index.pkl'), 'wb') as f:
            pickle.dump(index['inverted_index'], f)

        # Save the document lengths
        with open(os.path.join(output_dir, 'document_norms.pkl'), 'wb') as f:
            pickle.dump(index['document_norms'], f)

        # Save the document texts
        with open(os.path.join(output_dir, 'documents.jsonl'), 'w', encoding='utf-8') as f:
            for doc_id, doc_text in index['documents'].items():
                f.write(json.dumps({'doc_id': doc_id, 'text': doc_text}, ensure_ascii=False) + '\n')

        # Save metadata
        metadata = {
            'avgdl': index['avgdl'],
            'N': index['N'],
            'k1': index['k1'],
            'b': index['b'],
            'language': index.get('language', 'chinese'),
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Index saved to: {output_dir}")
        print(f"  documents: {index['N']}")
        print(f"  average document length: {index['avgdl']:.2f}")

    def load(self, input_dir: str) -> Dict:
        """Load an index."""
        with open(os.path.join(input_dir, 'inverted_index.pkl'), 'rb') as f:
            inverted_index = pickle.load(f)

        with open(os.path.join(input_dir, 'document_norms.pkl'), 'rb') as f:
            document_norms = pickle.load(f)

        documents = {}
        with open(os.path.join(input_dir, 'documents.jsonl'), 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                documents[record['doc_id']] = record['text']

        with open(os.path.join(input_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return {
            'inverted_index': inverted_index,
            'document_norms': document_norms,
            'documents': documents,
            'avgdl': metadata['avgdl'],
            'N': metadata['N'],
            'k1': metadata['k1'],
            'b': metadata['b'],
            'language': metadata.get('language', 'chinese'),
        }


class BM25Retriever:
    """BM25 retriever.

    Retrieves using a prebuilt index.
    """

    def __init__(self, index: Dict):
        """Initialize the retriever.

        Args:
            index: index dict.
        """
        self.inverted_index = index['inverted_index']
        self.document_norms = index['document_norms']
        self.avgdl = index['avgdl']
        self.N = index['N']
        self.k1 = index['k1']
        self.b = index['b']
        # Tokenization language persisted with the index
        # (PowerGrid=chinese / BIRD=english)
        self.language = index.get('language', 'chinese')

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve.

        Args:
            query: query text.
            k: top-k results to return.

        Returns:
            [(doc_id, score), ...]
        """
        query_terms = [t for t in tokenize_text(query, self.language) if len(t) > 1]
        scores = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            postings = self.inverted_index[term]
            df = len(postings)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf, doc_length in postings:
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                score = idf * numerator / denominator
                scores[doc_id] += score

        # Sort and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


def main():
    """Index construction demo."""
    import sys
    from pathlib import Path

    # Add the project root
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.engines.bird_runtime.config import TERAGConfig

    # Load the configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("Index construction")
    print("=" * 60)

    # Build the index
    builder = IndexBuilder(config)

    # Try to load a pattern library
    pattern_path = config.get_artifact_path('patterns.jsonl')
    if os.path.exists(pattern_path):
        builder.load_patterns(pattern_path)

    index = builder.build()

    # Save
    if config.output.save_index:
        builder.save(index, config.get_artifact_path('bm25_index'))

    # Test retrieval
    print("\nTest retrieval:")
    retriever = BM25Retriever(index)

    test_queries = [
        "查询公司的售电量",
        "统计用户的电费金额",
        "分析供电所的回收率",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, k=3)
        for doc_id, score in results:
            print(f"  {doc_id}: {score:.4f}")


if __name__ == "__main__":
    main()
