"""BIRD dataset adapter.

Converts the official BIRD data (dev.json / dev_tables.json /
dev_databases) into the experiment framework's standardized format
(table_df / field_df / qa_df), keeping the same interface as the PowerGrid
experiments.

Core conventions:
- Unique table id: "{db_id}.{table_name}" (e.g. california_schools.frpm);
  the db column stores db_id.
- field_name must use column_names_original (gold SQL uses the real
  physical column names).
- Gold tables / columns are parsed from the gold SQL (sqlglot, sqlite
  dialect + alias/subquery handling) and cross-checked against
  dev_tables.json; mismatches go to the parse-error list, never a silent
  accept.
- Whether evidence is appended to the query text is controlled by the
  use_evidence config flag.
"""

import json
import os
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from terag.engines.bird_runtime.datasets.base_adapter import BaseDatasetAdapter
from terag.engines.bird_runtime.config import BirdDatasetConfig
from terag.engines.bird_runtime.sql_role_parser import SQLRoleParser
from terag.engines.bird_runtime.tokenization import tokenize_text


class BirdDatasetAdapter(BaseDatasetAdapter):
    """BIRD dataset adapter."""

    def __init__(self, bird_config: BirdDatasetConfig, seed: int = 42,
                 source_split: Optional[str] = None):
        """Initialize the adapter.

        Args:
            bird_config: BIRD dataset configuration.
            seed: random seed (reproducible data splits).
            source_split: official split to use (defaults to the config's
                source_split).
        """
        self.config = bird_config
        self.seed = seed
        self.source_split = source_split or bird_config.source_split
        self.split_config = bird_config.get_source_split_config(self.source_split)

        # Raw data
        self.queries: List[Dict] = []
        self.db_schemas: Dict[str, Dict] = {}          # db_id -> dev_tables.json entry
        self.db_descriptions: Dict[str, Dict] = {}     # db_id -> {table: {col: desc dict}}

        # Parser (SQLite dialect)
        self.parser = SQLRoleParser(dialect=bird_config.sql_dialect)

        # Parse-error list
        self.parse_errors: List[Dict] = []

        # Simple table name (lower) -> canonical table id map, per db
        self._table_id_maps: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def ensure_databases_extracted(self):
        """Check the databases directory; auto-extract when only a zip exists."""
        databases_dir = self.split_config.databases_dir
        if not databases_dir or os.path.isdir(databases_dir):
            return

        zip_path = databases_dir.rstrip('/') + '.zip'
        if os.path.exists(zip_path):
            extract_to = os.path.dirname(databases_dir)
            print(f"{databases_dir} not found, auto-extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_to)
            print(f"Extraction finished: {databases_dir}")
        else:
            raise FileNotFoundError(
                f"BIRD databases directory does not exist and no zip found: {databases_dir}")

    def load_queries(self) -> List[Dict]:
        """Load the dev.json query data."""
        path = self.split_config.json
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"BIRD query file missing: {path} (check the "
                f"data.bird.{self.source_split}.json configuration)")

        with open(path, 'r', encoding='utf-8') as f:
            self.queries = json.load(f)
        print(f"Loaded BIRD queries: {len(self.queries)} (split={self.source_split})")
        return self.queries

    def load_schema(self) -> Dict:
        """Load the dev_tables.json schema data."""
        path = self.split_config.tables_json
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"BIRD schema file missing: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            tables = json.load(f)

        self.db_schemas = {t['db_id']: t for t in tables}
        print(f"Loaded BIRD schema: {len(self.db_schemas)} databases")

        # Build the simple table name -> canonical id map
        for db_id, schema in self.db_schemas.items():
            self._table_id_maps[db_id] = {
                t.lower(): self.canonical_table_id(db_id, t)
                for t in schema['table_names_original']
            }

        return self.db_schemas

    def load_database_descriptions(self) -> Dict:
        """Load the per-database database_description/<table>.csv field
        descriptions.

        Returns:
            {db_id: {table_name_original: {original_column_name:
            {desc fields}}}}
        """
        self.db_descriptions = {}
        databases_dir = self.split_config.databases_dir

        for db_id, schema in self.db_schemas.items():
            desc_dir = os.path.join(databases_dir, db_id, 'database_description')
            db_desc = {}

            if os.path.isdir(desc_dir):
                for table_name in schema['table_names_original']:
                    csv_path = os.path.join(desc_dir, f"{table_name}.csv")
                    if not os.path.exists(csv_path):
                        continue
                    try:
                        # BIRD description files carry a BOM (utf-8-sig); a few
                        # files are cp1252/latin-1 encoded
                        try:
                            desc_df = pd.read_csv(csv_path, encoding='utf-8-sig')
                        except UnicodeDecodeError:
                            desc_df = pd.read_csv(csv_path, encoding='cp1252')
                        table_desc = {}
                        for _, row in desc_df.iterrows():
                            col = row.get('original_column_name')
                            if pd.isna(col) or not str(col).strip():
                                continue
                            table_desc[str(col).strip()] = {
                                'column_name': '' if pd.isna(row.get('column_name')) else str(row.get('column_name')),
                                'column_description': '' if pd.isna(row.get('column_description')) else str(row.get('column_description')).strip(),
                                'data_format': '' if pd.isna(row.get('data_format')) else str(row.get('data_format')).strip(),
                                'value_description': '' if pd.isna(row.get('value_description')) else str(row.get('value_description')).strip(),
                            }
                        db_desc[table_name] = table_desc
                    except Exception as e:
                        print(f"  Warning: failed to read description file {csv_path}: {e}")

            self.db_descriptions[db_id] = db_desc

        n_desc = sum(len(t) for t in self.db_descriptions.values())
        print(f"Loaded database field descriptions: {n_desc} tables "
              f"(use_database_description={self.config.use_database_description})")
        return self.db_descriptions

    # ------------------------------------------------------------------
    # Schema conversion
    # ------------------------------------------------------------------

    def canonical_table_id(self, db_id: str, table_name: str) -> str:
        """Unique table id: {db_id}.{table} when prefix_table_with_db, else {table}."""
        if self.config.schema.prefix_table_with_db:
            return f"{db_id}.{table_name}"
        return table_name

    def _schema_columns_by_db(self, db_id: str) -> Dict[str, Set[str]]:
        """Per-database {simple table name: {physical column names}} (SQL
        parse disambiguation input)."""
        schema = self.db_schemas[db_id]
        columns = defaultdict(set)
        for table_idx, col_name in schema['column_names_original']:
            if table_idx < 0:
                continue  # skip the global '*'
            table_name = schema['table_names_original'][table_idx]
            columns[table_name].add(col_name)
        return dict(columns)

    def _build_field_desc(self, db_id: str, table_name: str, col_original: str,
                          col_nl_name: str) -> str:
        """Assemble the field description.

        Priority: database_description.column_description > BIRD
        column_names natural-language name. Optional appending of
        value_description / data_format is controlled by the configuration.
        """
        desc_entry = (self.db_descriptions.get(db_id, {})
                      .get(table_name, {})
                      .get(col_original, {}))

        parts = []
        if self.config.use_database_description:
            col_desc = desc_entry.get('column_description', '')
            if col_desc:
                parts.append(col_desc)
            data_format = desc_entry.get('data_format', '')
            if data_format:
                parts.append(f"[format: {data_format}]")
        if not parts and col_nl_name and col_nl_name != col_original:
            # No description file: fall back to the BIRD natural-language name
            parts.append(col_nl_name)
        if self.config.use_value_description:
            value_desc = desc_entry.get('value_description', '')
            if value_desc:
                parts.append(f"Value: {value_desc}")
        return ' '.join(parts)

    def build_table_dataframe(self) -> pd.DataFrame:
        """Build the standardized table schema DataFrame."""
        rows = []

        for db_id, schema in sorted(self.db_schemas.items()):
            table_names = schema['table_names_original']
            table_names_nl = schema['table_names']
            columns_by_table = self._schema_columns_by_db(db_id)

            for idx, table_name in enumerate(table_names):
                table_id = self.canonical_table_id(db_id, table_name)
                cols = sorted(columns_by_table.get(table_name, []))

                # Table description: prefer the BIRD natural-language name
                table_desc = table_names_nl[idx] if idx < len(table_names_nl) else table_name
                if not table_desc:
                    table_desc = table_name

                # Synthetic DDL (matches the PowerGrid schema column format)
                col_lines = []
                for col in cols:
                    col_lines.append(f"  {col}")
                ddl = (f"CREATE TABLE {table_id} (\n" + ",\n".join(col_lines) + "\n);")

                rows.append({
                    'table': table_id,
                    'db': db_id,
                    'table_desc': table_desc,
                    'field': '|'.join(cols),
                    'schema': ddl,
                })

        df = pd.DataFrame(rows)
        print(f"Built BIRD table schema: {len(df)} tables")
        return df

    def build_field_dataframe(self) -> pd.DataFrame:
        """Build the standardized field schema DataFrame."""
        rows = []
        schema_cfg = self.config.schema

        for db_id, schema in sorted(self.db_schemas.items()):
            table_names = schema['table_names_original']
            column_names_original = schema['column_names_original']
            column_names = schema['column_names']
            column_types = schema['column_types']
            primary_keys = schema.get('primary_keys', [])
            foreign_keys = schema.get('foreign_keys', [])

            # Primary-key column index set
            pk_cols = set()
            for pk in primary_keys:
                if isinstance(pk, list):
                    pk_cols.update(pk)
                else:
                    pk_cols.add(pk)

            # Foreign keys: column index -> referenced column index
            fk_map = {}
            for fk_col, fk_ref in foreign_keys:
                fk_map[fk_col] = fk_ref

            for col_idx, (table_idx, col_original) in enumerate(column_names_original):
                if table_idx < 0:
                    continue
                table_name = table_names[table_idx]
                table_id = self.canonical_table_id(db_id, table_name)
                col_nl = column_names[col_idx][1] if col_idx < len(column_names) else col_original
                col_type = column_types[col_idx] if col_idx < len(column_types) else 'text'

                field_desc = self._build_field_desc(db_id, table_name, col_original, col_nl)

                # Non-destructive PK/FK metadata
                is_pk = col_idx in pk_cols
                fk_target = ''
                if col_idx in fk_map:
                    ref_idx = fk_map[col_idx]
                    if 0 <= ref_idx < len(column_names_original):
                        ref_table_idx, ref_col = column_names_original[ref_idx]
                        if 0 <= ref_table_idx < len(table_names):
                            fk_target = f"{table_names[ref_table_idx]}.{ref_col}"

                if schema_cfg.include_pk_fk:
                    if is_pk:
                        field_desc = (field_desc + " [primary key]").strip()
                    if fk_target:
                        field_desc = (field_desc + f" [foreign key -> {fk_target}]").strip()

                rows.append({
                    'table': table_id,
                    'db': db_id,
                    'field_name': col_original,
                    'field_name_desc': field_desc,
                    'field_type': col_type,
                    'is_primary_key': is_pk,
                    'foreign_key_target': fk_target,
                })

        df = pd.DataFrame(rows)
        print(f"Built BIRD field schema: {len(df)} fields")
        return df

    # ------------------------------------------------------------------
    # Gold SQL parsing
    # ------------------------------------------------------------------

    def parse_gold_sql(self, item: Dict) -> Dict:
        """Parse one gold SQL, extracting gold tables / gold columns.

        Returns:
            {
                'parse_ok': bool,
                'gold_tables': [canonical_table_id, ...],
                'gold_fields': [[canonical_table_id, column_name], ...],
                'gold_joins': [[table, table], ...],      # simple names
                'gold_predicates': ["table.column", ...], # simple names
                'errors': [messages], 'warnings': [warnings]
            }
        """
        db_id = item['db_id']
        sql = item.get('SQL', '')
        result = {
            'parse_ok': False,
            'gold_tables': [],
            'gold_fields': [],
            'gold_joins': [],
            'gold_predicates': [],
            'errors': [],
            'warnings': [],
        }

        id_map = self._table_id_maps.get(db_id, {})
        schema_columns = self._schema_columns_by_db(db_id)
        # Schema column case-normalization map: (table_lower, col_lower) ->
        # original schema column name
        schema_col_lookup = {}
        for t, cols in schema_columns.items():
            for c in cols:
                schema_col_lookup[(t.lower(), str(c).lower())] = c

        parsed = self.parser.parse(
            sql, dialect=self.config.sql_dialect, schema_columns=schema_columns)

        if not parsed.success:
            result['errors'].append(f"SQL parse failed ({parsed.parse_method}): {parsed.error}")
            return result

        # Gold tables: simple name (lower) -> canonical id, verified in schema
        unknown_tables = []
        for simple_table in parsed.get_gold_tables():
            if simple_table in id_map:
                canonical = id_map[simple_table]
                if canonical not in result['gold_tables']:
                    result['gold_tables'].append(canonical)
            else:
                unknown_tables.append(simple_table)
        if unknown_tables:
            result['errors'].append(
                f"parsed tables not in schema: {unknown_tables}")

        # Gold columns: (simple table, SQL column name) ->
        # (canonical table id, original schema column name)
        for table_simple, col in sorted(parsed.get_all_fields()):
            canonical = id_map.get(table_simple)
            if canonical is None:
                continue
            schema_col = schema_col_lookup.get((table_simple, col.lower()), col)
            pair = [canonical, schema_col]
            if pair not in result['gold_fields']:
                result['gold_fields'].append(pair)

        result['gold_joins'] = [list(p) for p in parsed.join_pairs]
        result['gold_predicates'] = [f"{t}.{c}" for t, c in parsed.predicates]
        result['warnings'] = list(parsed.warnings)

        result['parse_ok'] = bool(result['gold_tables']) and not unknown_tables
        if not result['gold_tables']:
            result['errors'].append("no gold tables extracted")
        return result

    # ------------------------------------------------------------------
    # QA data
    # ------------------------------------------------------------------

    def build_query_text(self, item: Dict) -> str:
        """Build the retrieval query text.

        With use_evidence=True the evidence is appended (experiment output
        must record this flag).
        """
        question = str(item.get('question', '')).strip()
        evidence = str(item.get('evidence', '') or '').strip()
        if self.config.use_evidence and evidence:
            return f"{question} {evidence}"
        return question

    def build_qa_dataframe(self) -> pd.DataFrame:
        """Build the standardized QA DataFrame (with structured gold)."""
        if not self.db_schemas:
            self.load_schema()
        if not self.queries:
            self.load_queries()
        if self.config.use_database_description and not self.db_descriptions:
            self.load_database_descriptions()

        rows = []
        n_parse_ok = 0

        for item in self.queries:
            gold = self.parse_gold_sql(item)
            if gold['parse_ok']:
                n_parse_ok += 1

            # Parse errors / warnings are recorded (never silently accepted)
            if gold['errors'] or gold['warnings']:
                self.parse_errors.append({
                    'question_id': item.get('question_id'),
                    'db_id': item.get('db_id'),
                    'sql': item.get('SQL', ''),
                    'errors': gold['errors'],
                    'warnings': gold['warnings'],
                    'parsed_tables': gold['gold_tables'],
                    'parsed_columns': [f"{t}.{c}" for t, c in gold['gold_fields']],
                })

            # Simple table name display form (legacy-compatible field)
            simple_tables = [t.split('.')[-1] for t in gold['gold_tables']]
            field_display = '|'.join(
                f"{t.split('.')[-1]}.{c}" for t, c in gold['gold_fields'])

            rows.append({
                'question': self.build_query_text(item),
                'table': '|'.join(gold['gold_tables']),
                'field': field_display,
                'sql': item.get('SQL', ''),
                'db_id': item.get('db_id', ''),
                'evidence': str(item.get('evidence', '') or ''),
                'difficulty': item.get('difficulty', ''),
                'question_id': item.get('question_id'),
                'question_orig': str(item.get('question', '')),
                'gold_tables': gold['gold_tables'],
                'gold_fields': gold['gold_fields'],
                'gold_joins': gold['gold_joins'],
                'gold_predicates': gold['gold_predicates'],
                'parse_ok': gold['parse_ok'],
            })

        qa_df = pd.DataFrame(rows)
        total = len(qa_df)
        print(f"Built BIRD QA data: {total} rows, SQL parse success {n_parse_ok} "
              f"({n_parse_ok / total * 100:.1f}%)")
        n_multi = sum(1 for r in rows if len(r['gold_tables']) > 1)
        print(f"  multi-table queries: {n_multi}, single-table: {total - n_multi}")
        return qa_df

    # ------------------------------------------------------------------
    # Within-database data split
    # ------------------------------------------------------------------

    def split_within_database(self, qa_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Train/dev/test split within each database (historical queries and
        test questions share the same database schema).

        Databases with fewer than min_queries_per_db queries are skipped
        entirely with a log line (no silent drop).

        Returns:
            (train_df, dev_df, test_df, split_info)
        """
        split_cfg = self.config.split
        assert abs(split_cfg.train_ratio + split_cfg.dev_ratio + split_cfg.test_ratio - 1.0) < 1e-6, \
            "split ratios must sum to 1"

        rng = np.random.RandomState(self.seed)
        train_idx, dev_idx, test_idx = [], [], []
        skipped_dbs = {}

        for db_id in sorted(qa_df['db_id'].unique()):
            db_rows = qa_df[qa_df['db_id'] == db_id]
            n = len(db_rows)

            if n < split_cfg.min_queries_per_db:
                skipped_dbs[db_id] = n
                print(f"  Warning: database {db_id} has only {n} queries "
                      f"< min_queries_per_db={split_cfg.min_queries_per_db}, skipping the whole db")
                continue

            indices = db_rows.index.to_numpy()
            permuted = indices[rng.permutation(n)]

            n_train = int(n * split_cfg.train_ratio)
            n_dev = int(n * split_cfg.dev_ratio)

            train_idx.extend(permuted[:n_train])
            dev_idx.extend(permuted[n_train:n_train + n_dev])
            test_idx.extend(permuted[n_train + n_dev:])

        train_df = qa_df.loc[sorted(train_idx)].reset_index(drop=True)
        dev_df = qa_df.loc[sorted(dev_idx)].reset_index(drop=True)
        test_df = qa_df.loc[sorted(test_idx)].reset_index(drop=True)

        print(f"\nBIRD within-database split (seed={self.seed}, strategy={split_cfg.strategy}):")
        print(f"  train: {len(train_df)} ({len(train_df) / len(qa_df) * 100:.1f}%)")
        print(f"  dev:   {len(dev_df)} ({len(dev_df) / len(qa_df) * 100:.1f}%)")
        print(f"  test:  {len(test_df)} ({len(test_df) / len(qa_df) * 100:.1f}%)")

        split_info = {
            'strategy': split_cfg.strategy,
            'seed': self.seed,
            'train_ratio': split_cfg.train_ratio,
            'dev_ratio': split_cfg.dev_ratio,
            'test_ratio': split_cfg.test_ratio,
            'min_queries_per_db': split_cfg.min_queries_per_db,
            'train_size': len(train_df),
            'dev_size': len(dev_df),
            'test_size': len(test_df),
            'train_dbs': sorted(train_df['db_id'].unique().tolist()),
            'dev_dbs': sorted(dev_df['db_id'].unique().tolist()),
            'test_dbs': sorted(test_df['db_id'].unique().tolist()),
            'skipped_dbs': skipped_dbs,
        }
        return train_df, dev_df, test_df, split_info

    # ------------------------------------------------------------------
    # Full preparation flow
    # ------------------------------------------------------------------

    def prepare(self) -> Dict[str, pd.DataFrame]:
        """Run the full data preparation flow.

        Returns:
            {'table_df', 'field_df', 'qa_df', 'train_df', 'dev_df', 'test_df'}
        """
        print("=" * 60)
        print(f"BIRD data preparation (source_split={self.source_split}, seed={self.seed})")
        print("=" * 60)

        self.ensure_databases_extracted()
        self.load_schema()
        self.load_queries()
        self.load_database_descriptions()

        table_df = self.build_table_dataframe()
        field_df = self.build_field_dataframe()
        qa_df = self.build_qa_dataframe()

        train_df, dev_df, test_df, split_info = self.split_within_database(qa_df)

        return {
            'table_df': table_df,
            'field_df': field_df,
            'qa_df': qa_df,
            'train_df': train_df,
            'dev_df': dev_df,
            'test_df': test_df,
            'split_info': split_info,
        }
