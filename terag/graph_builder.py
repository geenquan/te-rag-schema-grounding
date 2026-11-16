"""
Bipartite graph builder.

Builds the query-table-column bipartite graph; edge weights are determined
by SQL roles.
"""

import os
import pickle
import json
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

from terag.config import TERAGConfig
from terag.sql_role_parser import SQLRoleParser, ParseResult
from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields


class BipartiteGraphBuilder:
    """
    Bipartite graph builder.

    Builds a query-table-column bipartite graph:
    - Left-side nodes: queries (Q_0, Q_1, ...)
    - Right-side nodes: tables (T:table_name) and columns (C:table.field)

    Edge weights are determined by the column's SQL role:
    - WHERE: 2.0
    - JOIN: 1.8
    - GROUP_BY: 1.5
    - ORDER_BY: 1.3
    - SELECT: 1.2
    - FROM: 1.0

    Usage:
        builder = BipartiteGraphBuilder(config)
        graph = builder.build(train_data)
        builder.save(graph, "artifacts/graph.pkl")
    """

    def __init__(self, config: TERAGConfig):
        """
        Initialize the builder.

        Args:
            config: TE-RAG configuration
        """
        self.config = config
        self.parser = SQLRoleParser(
            default_role_weights=config.graph.role_weights,
            dialect=config.sql_dialect,
            mode=getattr(config.graph, 'parse_mode', 'ast'),
        )

        # Load table and field metadata
        self.field_df = pd.read_csv(config.data.field_csv)
        self.table_df = pd.read_csv(config.data.table_csv)

        # Simple table name -> physical schema column set, used to
        # disambiguate SQL parsing (handles multi-table / aliased SQL)
        self.schema_columns: Dict[str, Set[str]] = defaultdict(set)
        for _, row in self.field_df.iterrows():
            simple = str(row['table']).split('.')[-1]
            self.schema_columns[simple.lower()].add(str(row['field_name']))

        # Role occurrence statistics
        self.role_stats = defaultdict(lambda: defaultdict(int))

    def build(self, train_data: pd.DataFrame) -> nx.Graph:
        """
        Build the bipartite graph.

        Args:
            train_data: training data with question, table, field and sql columns

        Returns:
            A NetworkX Graph
        """
        graph = nx.Graph()

        # Add query nodes
        for idx in range(len(train_data)):
            graph.add_node(f"Q_{idx}", bipartite=0, node_type='query')

        # Add table nodes
        for _, row in self.table_df.iterrows():
            table_node = f"T:{row['table']}"
            graph.add_node(
                table_node,
                bipartite=1,
                node_type='table',
                description=row.get('table_desc', '')
            )

        # Add column nodes
        for _, row in self.field_df.iterrows():
            col_node = f"C:{row['table']}.{row['field_name']}"
            graph.add_node(
                col_node,
                bipartite=1,
                node_type='column',
                description=row.get('field_name_desc', '')
            )

        # Add edges
        use_role_parser = self.config.ablation.use_role_parser

        for pos in range(len(train_data)):
            row = train_data.iloc[pos]
            query_node = f"Q_{pos}"
            sql = row.get('sql', '')

            # Parse the ground truth uniformly (PowerGrid single-table legacy
            # format / BIRD multi-table structured gold columns)
            gold_tables = parse_ground_truth_tables(row)          # canonical table ids
            gold_fields = parse_ground_truth_fields(row)          # [(canonical table id, field name)]
            simple_to_canonical = {t.split('.')[-1].lower(): t for t in gold_tables}

            # Parse SQL roles (in the multi-table case expected_table is not
            # passed; disambiguation relies on aliases and the schema)
            if use_role_parser and pd.notna(sql):
                expected = gold_tables[0].split('.')[-1] if len(gold_tables) == 1 else None
                parse_result = self.parser.parse(
                    str(sql), expected, schema_columns=dict(self.schema_columns))
            else:
                parse_result = None

            # Add column edges
            for table_id, field in gold_fields:
                col_node = f"C:{table_id}.{field}"

                if graph.has_node(col_node):
                    # Compute the weight
                    if parse_result and parse_result.success:
                        table_simple = table_id.split('.')[-1].lower()
                        weight = parse_result.get_field_weight(
                            table_simple, field,
                            self.config.graph.role_weights
                        )
                        # Ablation: unify role weights (keep AST parsing and
                        # the edge set, only force weight -> 1.0)
                        if hasattr(self.config.ablation, 'use_role_weight') \
                                and not self.config.ablation.use_role_weight:
                            weight = 1.0
                        # Track role statistics
                        info = parse_result.find_field_info(table_simple, field)
                        if info is not None:
                            for role in info.roles:
                                self.role_stats[table_id][role] += 1
                    else:
                        # Fall back to string matching
                        weight = self._get_weight_by_string_match(str(sql) if pd.notna(sql) else '')
                        # Ablation: unify role weights (the fallback path also
                        # stays on unified weights)
                        if hasattr(self.config.ablation, 'use_role_weight') \
                                and not self.config.ablation.use_role_weight:
                            weight = 1.0

                    graph.add_edge(query_node, col_node, weight=weight)

            # Add table edges (multi-table queries get an edge for every gold table)
            for table_id in gold_tables:
                table_node = f"T:{table_id}"
                if graph.has_node(table_node):
                    table_weight = self.config.graph.role_weights.get('FROM', 1.0)
                    graph.add_edge(query_node, table_node, weight=table_weight)

        return graph

    def _get_weight_by_string_match(self, sql: str) -> float:
        """Derive the weight via string matching (fallback path)."""
        weight = 1.0
        sql_upper = sql.upper()

        if 'WHERE' in sql_upper:
            weight = max(weight, 2.0)
        if 'JOIN' in sql_upper:
            weight = max(weight, 1.8)
        if 'GROUP BY' in sql_upper:
            weight = max(weight, 1.5)
        if 'ORDER BY' in sql_upper:
            weight = max(weight, 1.3)
        if 'SELECT' in sql_upper:
            weight = max(weight, 1.2)

        return weight

    def save(self, graph: nx.Graph, output_path: str):
        """Save the graph."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Graph saved to: {output_path}")

    def load(self, input_path: str) -> nx.Graph:
        """Load a graph."""
        with open(input_path, 'rb') as f:
            return pickle.load(f)

    def save_role_stats(self, output_path: str):
        """Save role statistics."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.role_stats), f, ensure_ascii=False, indent=2)
        print(f"Role statistics saved to: {output_path}")

    def get_element_to_queries(self, graph: nx.Graph, train_data: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Build the element-to-queries mapping.

        Args:
            graph: the bipartite graph
            train_data: training data

        Returns:
            {element_node: set of questions}
        """
        element_to_queries = defaultdict(set)

        for pos in range(len(train_data)):
            row = train_data.iloc[pos]
            query_node = f"Q_{pos}"
            question = row['question']

            # Collect all neighbor nodes
            if graph.has_node(query_node):
                for neighbor in graph.neighbors(query_node):
                    element_to_queries[neighbor].add(question)

        return element_to_queries

    def get_graph_stats(self, graph: nx.Graph) -> Dict:
        """Return graph statistics."""
        query_nodes = [n for n in graph.nodes if n.startswith('Q_')]
        table_nodes = [n for n in graph.nodes if n.startswith('T:')]
        column_nodes = [n for n in graph.nodes if n.startswith('C:')]

        return {
            'total_nodes': graph.number_of_nodes(),
            'query_nodes': len(query_nodes),
            'table_nodes': len(table_nodes),
            'column_nodes': len(column_nodes),
            'total_edges': graph.number_of_edges(),
            'density': nx.density(graph),
        }


def main():
    """Demo of bipartite graph construction."""
    import sys
    from pathlib import Path

    # Make the project root importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.config import TERAGConfig

    # Load the configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    # Load the training data
    import json
    train_data = []
    with open(config.get_split_path('train'), 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    print(f"Loaded {len(train_df)} training examples")

    # Build the graph
    builder = BipartiteGraphBuilder(config)
    graph = builder.build(train_df)

    # Print statistics
    stats = builder.get_graph_stats(graph)
    print(f"\nGraph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save artifacts
    if config.output.save_graph:
        builder.save(graph, config.get_artifact_path('graph.pkl'))

    if config.output.save_role_stats:
        builder.save_role_stats(config.get_artifact_path('role_stats.json'))


if __name__ == "__main__":
    main()
