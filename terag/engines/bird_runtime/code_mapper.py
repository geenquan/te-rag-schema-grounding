"""Code-value mapper.

Mines phrase -> code mappings from historical SQL for query expansion.

Usage:
    mapper = CodeMapper(config)
    phrase_mapping = mapper.mine_from_sql(qa_data)
    expanded_query = mapper.expand_query(query)
"""

import os
import re
import json
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from terag.engines.bird_runtime.config import TERAGConfig


@dataclass
class PhraseMapping:
    """A phrase mapping."""
    phrase: str
    field: str
    code_value: str
    source_sql: str
    frequency: int = 1


class CodeMapper:
    """Code-value mapper.

    Mines (field=value) mappings from the WHERE conditions of historical
    SQL, used for query expansion (a phrase hit appends a code token).
    """

    def __init__(self, config: TERAGConfig):
        """Initialize the code-value mapper.

        Args:
            config: TE-RAG configuration.
        """
        self.config = config
        self.phrase_mappings: Dict[str, PhraseMapping] = {}
        self.field_df = pd.read_csv(config.data.field_csv)

    def mine_from_sql(self, qa_data: pd.DataFrame) -> Dict[str, List[PhraseMapping]]:
        """Mine code-value mappings from SQL.

        Args:
            qa_data: QA data with question, sql columns.

        Returns:
            {field: [PhraseMapping, ...]}
        """
        mappings = defaultdict(list)

        for _, row in qa_data.iterrows():
            sql = row.get('sql', '')
            if pd.isna(sql) or not sql:
                continue

            # Parse the WHERE conditions
            conditions = self._extract_where_conditions(str(sql))

            for condition in conditions:
                # Extract field=value pairs
                field_value_pairs = self._extract_field_value_pairs(condition)

                for field, value in field_value_pairs:
                    # Create the mapping
                    mapping = PhraseMapping(
                        phrase=self._infer_phrase(field, value, row['question']),
                        field=field,
                        code_value=value,
                        source_sql=str(sql),
                        frequency=1
                    )
                    mappings[field].append(mapping)

        # Merge mappings of the same phrase and value, accumulating frequency
        merged = defaultdict(list)
        for field, mapping_list in mappings.items():
            seen = {}
            for m in mapping_list:
                key = (m.phrase, m.code_value)
                if key in seen:
                    seen[key].frequency += 1
                else:
                    seen[key] = m
            merged[field] = list(seen.values())

        # Store into self.phrase_mappings
        self.phrase_mappings = dict(merged)

        return dict(merged)

    def _extract_where_conditions(self, sql: str) -> List[str]:
        """Extract the WHERE conditions."""
        conditions = []

        # Simple WHERE clause extraction
        where_match = re.search(
            r'\bWHERE\s+(.*?)(?:\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)',
            sql,
            re.IGNORECASE | re.DOTALL
        )

        if where_match:
            where_clause = where_match.group(1)
            # Split AND/OR conditions
            cond_parts = re.split(r'\bAND\b|\bOR\b', where_clause, flags=re.IGNORECASE)
            conditions.extend([c.strip() for c in cond_parts if c.strip()])

        return conditions

    def _extract_field_value_pairs(self, condition: str) -> List[Tuple[str, str]]:
        """Extract the field=value pairs of a condition."""
        pairs = []

        # Remove parentheses
        condition = condition.strip('()')

        # Match field = 'value' or field != 'value'
        # Several quote styles supported
        patterns = [
            r"(\w+)\s*=\s*'([^']*)'",   # field = 'value'
            r"(\w+)\s*=\s*\"([^\"]*)\"",  # field = "value"
            r"(\w+)\s*=\s*(\d+)",       # field = number
            r"(\w+)\s*LIKE\s+'%'",      # field LIKE '%'
        ]

        for pattern in patterns:
            match = re.search(pattern, condition, re.IGNORECASE)
            if match:
                field = match.group(1)
                value = match.group(2)
                pairs.append((field, value))

        return pairs

    def _infer_phrase(self, field: str, value: str, question: str) -> str:
        """Infer the phrase (simplified)."""
        # Simple handling: strip quotes, lowercase
        value = value.strip("'\"").lower()
        field = field.lower()

        # Try to extract a phrase from the question
        # If the value contains Chinese, use the value directly as the phrase
        if re.search(r'[一-龥]+', value):
            return value
        else:
            # Otherwise use the field name as the phrase
            return field

        return f"{field}_{value}"

    def build_phrase_mapping(self, qa_data: pd.DataFrame) -> Dict[str, List[PhraseMapping]]:
        """Build the phrase mapping dictionary.

        Args:
            qa_data: QA data.

        Returns:
            {field: [PhraseMapping, ...]}
        """
        all_mappings = defaultdict(list)

        # Group by field
        for field, mappings in self.mine_from_sql(qa_data).items():
            all_mappings[field].extend(mappings)

        # Sort by frequency
        for field in all_mappings:
            all_mappings[field].sort(key=lambda m: -m.frequency, reverse=True)

        return dict(all_mappings)

    def expand_query(self, query: str) -> str:
        """Expand the query (append code tokens).

        Args:
            query: original query.

        Returns:
            The expanded query.
        """
        expanded_tokens = []

        # Check every field mapping
        for field, mappings in self.phrase_mappings.items():
            for mapping in mappings:
                # If the query contains the phrase, append the code token
                if mapping.phrase in query:
                    code_token = f"CODE:{mapping.field}='{mapping.code_value}'"
                    expanded_tokens.append(code_token)

        if expanded_tokens:
            return f"{query} {' '.join(expanded_tokens)}"
        else:
            return query

    def save(self, output_path: str):
        """Save the mappings to a file.

        Args:
            output_path: output path.
        """
        data = {}
        for field, mappings in self.phrase_mappings.items():
            data[field] = [
                {
                    'phrase': m.phrase,
                    'field': m.field,
                    'code_value': m.code_value,
                    'source_sql': m.source_sql,
                    'frequency': m.frequency,
                }
                for m in mappings
            ]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Code-value mappings saved to: {output_path}")

    def load(self, input_path: str):
        """Load the mappings from a file.

        Args:
            input_path: input path.

        Returns:
            The number of loaded mappings.
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for field, mappings in data.items():
            for m in mappings:
                mapping = PhraseMapping(
                    phrase=m['phrase'],
                    field=m['field'],
                    code_value=m['code_value'],
                    source_sql=m['source_sql'],
                    frequency=m['frequency']
                )
                self.phrase_mappings[field].append(mapping)
                count += 1

        print(f"Loaded {count} code-value mappings from {input_path}")
        return count


class QueryProcessor:
    """Query processor.

    Integrates code-value mapping for query expansion.
    """

    def __init__(self, config: TERAGConfig, code_mapper: CodeMapper = None):
        self.config = config
        self.code_mapper = code_mapper or CodeMapper(config)
        self._token_cache = {}

    def process(self, query: str) -> str:
        """Process a query.

        Args:
            query: original query.

        Returns:
            The processed query (may contain code tokens).
        """
        if query in self._token_cache:
            return self._token_cache[query]

        # 1. Expand the query (append code tokens)
        expanded_query = self.code_mapper.expand_query(query)

        # 2. Tokenize
        import jieba
        tokens = list(jieba.cut(expanded_query))

        self._token_cache[query] = tokens

        return tokens

    def train_from_qa_data(self, qa_path: str, output_path: str = None):
        """Train the code-value mappings from QA data.

        Args:
            qa_path: QA data path.
            output_path: mapping output path.
        """
        # Load the QA data
        qa_data = pd.read_csv(qa_path)

        # Mine the mappings
        mappings = self.code_mapper.mine_from_sql(qa_data)

        # Save
        self.code_mapper.save(output_path)

        return len(mappings)


def main():
    """Code-value mapping demo."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.engines.bird_runtime.config import TERAGConfig

    # Load the configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("Code-value mapping demo")
    print("=" * 60)

    # Train
    mapper = CodeMapper(config)
    train_path = config.data.qa_csv

    mappings_path = config.get_artifact_path('code_mappings.json')

    if not os.path.exists(mappings_path):
        print(f"\nTraining code-value mappings from {train_path}...")
        n_mappings = mapper.mine_from_sql(pd.read_csv(train_path))
        mapper.save(mappings_path)
        print(f"Extracted mappings for {sum(len(v) for v in mappings.values())} fields")
    else:
        print(f"\nLoading existing mappings: {mappings_path}")
        mapper.load(mappings_path)

    # Test expansion
    print("\nTest query expansion:")
    test_queries = [
        "查询杭州供电公司的售电量",
        "统计浙江省所有用户的电费总额",
        "分析供电所A的回收率排名",
    ]

    processor = QueryProcessor(config, mapper)

    for query in test_queries:
        tokens = processor.process(query)
        print(f"\nQuery: {query}")
        print(f"Tokens: {tokens}")


if __name__ == "__main__":
    main()
