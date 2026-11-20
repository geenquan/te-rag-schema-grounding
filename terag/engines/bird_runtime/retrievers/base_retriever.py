"""Retriever base class.

Defines the interface and the shared machinery every retriever must
implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import time
import tracemalloc
import jieba

from terag.engines.bird_runtime.tokenization import tokenize_english
from terag.engines.bird_runtime.datasets.gt_utils import (
    parse_ground_truth_tables,
    parse_ground_truth_fields,
    has_structured_gold,
    load_json_list,
)


@dataclass
class RetrieverConfig:
    """Retriever configuration."""
    name: str = "BaseRetriever"            # retriever name
    description: str = ""                  # description
    k1: float = 1.5                        # BM25 parameter k1
    b: float = 0.75                        # BM25 parameter b
    top_k: int = 5                         # top-k results to return
    use_chinese_tokenizer: bool = True     # whether to use the Chinese tokenizer

    # Extra parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    table: str                             # table name
    table_score: float                     # table score
    columns: List[tuple]                   # column list [(column_name, score), ...]
    metadata: Dict[str, Any] = field(default_factory=dict)  # extra metadata


@dataclass
class EvaluationMetrics:
    """Evaluation metrics."""
    table_accuracy: float = 0.0            # table selection accuracy
    sql_accuracy: float = 0.0              # SQL accuracy
    avg_query_time: float = 0.0            # average query time (seconds)
    avg_memory_mb: float = 0.0             # average memory usage (MB)
    total_queries: int = 0                 # total query count

    # ========== Added schema coverage metrics ==========
    table_recall: float = 0.0              # table recall
    column_recall: float = 0.0             # field recall
    column_precision: float = 0.0          # field precision
    column_f1: float = 0.0                 # field F1

    # ========== Added top-K table recall ==========
    top1_table_recall: float = 0.0         # top-1 table recall
    top3_table_recall: float = 0.0         # top-3 table recall
    top5_table_recall: float = 0.0         # top-5 table recall

    # ========== SQL parse rate ==========
    sql_parse_rate: float = 0.0            # SQL parseability rate


class BaseRetriever(ABC):
    """Abstract retriever base class.

    Every retriever implementation must subclass this and implement:
    - fit(): train/build the index
    - _retrieve(): the core retrieval logic

    Usage:
        class MyRetriever(BaseRetriever):
            def __init__(self, field_csv, table_csv, config=None):
                super().__init__(field_csv, table_csv, config)
                # custom initialization

            def fit(self, train_data=None):
                # build the index
                self._is_fitted = True

            def _retrieve(self, query, k=5):
                # retrieval logic
                return results
    """

    def __init__(self, field_csv: str, table_csv: str,
                 config: Optional[RetrieverConfig] = None):
        """Initialize the retriever.

        Args:
            field_csv: field CSV path.
            table_csv: table CSV path.
            config: retriever configuration (optional; subclasses may
                provide defaults).
        """
        # Configuration (subclasses may override the defaults)
        self.config = config or self._get_default_config()

        # Read the data
        self.field_df = pd.read_csv(field_csv)
        self.table_df = pd.read_csv(table_csv)

        # State
        self._is_fitted = False

        # BM25 parameters (from config)
        self.k1 = self.config.k1
        self.b = self.config.b

        # Data preprocessing
        self._preprocess_data()

    def _get_default_config(self) -> RetrieverConfig:
        """Get the default configuration (subclasses may override)."""
        return RetrieverConfig(name=self.__class__.__name__)

    def _preprocess_data(self):
        """Data preprocessing."""
        # Handle NaN values
        if self.table_df is not None:
            self.table_df['table_desc'] = self.table_df['table_desc'].fillna('').astype(str)

        if self.field_df is not None:
            self.field_df['field_name_desc'] = self.field_df['field_name_desc'].fillna('').astype(str)

    @abstractmethod
    def fit(self, train_data: pd.DataFrame = None):
        """Train/build the index.

        Args:
            train_data: training data (optional; some methods need it).
        """
        pass

    @abstractmethod
    def _retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Core retrieval logic (subclass implementation).

        Args:
            query: query text.
            k: top-k results to return.

        Returns:
            List of retrieval results.
        """
        pass

    def retrieve(self, query: str, k: int = None) -> List[RetrievalResult]:
        """Retrieval interface (with checks and wrapping).

        Args:
            query: query text.
            k: top-k results to return.

        Returns:
            List of retrieval results.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Retriever {self.config.name} has not been fitted. Call fit() first.")

        k = k or self.config.top_k
        return self._retrieve(query, k)

    def retrieve_with_metrics(self, query: str, k: int = None) -> tuple:
        """Retrieval with performance metrics.

        Args:
            query: query text.
            k: top-k results to return.

        Returns:
            (results, query_time, memory_usage_mb)
        """
        tracemalloc.start()
        start_time = time.time()

        results = self.retrieve(query, k)

        query_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return results, query_time, peak / 1024 / 1024

    def tokenize(self, text: str) -> List[str]:
        """Tokenize.

        Args:
            text: input text.

        Returns:
            Tokenization result (jieba when use_chinese_tokenizer=True,
            otherwise the English regex tokenizer).
        """
        if self.config.use_chinese_tokenizer:
            return list(jieba.cut(text))
        else:
            return tokenize_english(text)

    @staticmethod
    def _retrieved_column_keys(results: List['RetrievalResult']) -> List[str]:
        """Canonical column keys ("table.column", "C:" prefix stripped) from
        retrieval results.

        Used for the Column Recall / Precision / F1 computation in the
        multi-table setting (BIRD).
        """
        keys = []
        for r in results:
            for col, _ in r.columns:
                col_name = col[2:] if col.startswith('C:') else col
                if col_name not in keys:
                    keys.append(col_name)
        return keys

    def evaluate(self, test_data: pd.DataFrame, k: int = 5) -> EvaluationMetrics:
        """Evaluate the retriever.

        Both ground-truth formats are supported:
        - PowerGrid (single table, table/field columns): metric semantics
          exactly as the original implementation
        - BIRD (multi table, structured gold_tables/gold_fields columns):
          * Table Accuracy = Table Exact Match (all gold tables among the
            retrieved tables)
          * Top-K Table Recall = mean(|gold intersect topK| / |gold|)
          * Column metrics are set-level matches on "table.column" keys
            (macro average)
          * SQL Accuracy = schema-grounded accuracy (all gold tables hit and
            all gold columns covered)

        Args:
            test_data: test data with a question column plus table/field or
                gold_tables/gold_fields columns.
            k: top-k results.

        Returns:
            Evaluation metrics.
        """
        records = self.evaluate_detailed(test_data, k)
        return self._aggregate_records(records)

    def evaluate_detailed(self, test_data: pd.DataFrame, k: int = 5) -> List[Dict[str, Any]]:
        """Per-question evaluation returning one detail record per query.

        Since P1-A every record contains:
        - the official metrics (defined in eval/retrieval_metrics.py):
          hit_at_{1,3,5} (single table) or table_recall_at_k /
          all_gold_covered (multi table), column_precision/recall/f1,
          schema_coverage_success, gold_field_status;
        - compatibility aliases (old field names, semantics = the old
          implementation semantics confirmed by the P0 audit; internal use
          for the legacy pipeline only): table_accuracy_hit /
          table_recall_full (single table = Hit@k with this k),
          schema_sql_correct / sql_accuracy_hit (the old SCS variant that
          counts an empty gold as correct);
        - status fields: status (ok/error), error_type, error_message (a
          retrieval exception no longer aborts the whole evaluation nor
          masquerades as a normal empty prediction); exec_accuracy is
          always 'not_run'.

        Args:
            test_data: test data.
            k: top-k results.

        Returns:
            List of records, each with question/db_id/difficulty/gold/
            retrieved and the metrics.
        """
        import time as _time
        from terag.engines.bird_runtime.eval.retrieval_metrics import (
            evaluate_single_table_query, evaluate_multi_table_query,
            normalize_column_key,
        )

        records = []

        for _, row in test_data.iterrows():
            query = row['question']

            record = {
                'question': query,
                'db_id': row.get('db_id', ''),
                'difficulty': row.get('difficulty', ''),
                'question_id': row.get('question_id', None),
                'gold_tables': parse_ground_truth_tables(row),
                'status': 'ok',
                'error_type': None,
                'error_message': None,
            }

            # ---- Retrieval (per-question exception capture: an error record
            # ---- neither aborts the evaluation nor masquerades as a normal
            # ---- result)
            t0 = _time.time()
            try:
                results, query_time, memory_mb = self.retrieve_with_metrics(query, k)
            except Exception as e:  # noqa: BLE001 - any retrieval exception must be recorded
                results, query_time, memory_mb = [], _time.time() - t0, 0.0
                record.update({
                    'status': 'error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)[:500],
                })
                print(f"[evaluate_detailed] retrieval exception status=error: {type(e).__name__}: {str(e)[:200]}")

            retrieved_tables = [r.table for r in results]
            retrieved_column_keys = []
            retrieved_columns_detail = []
            for r in results:
                cols = [{'key': col, 'score': float(sc)} for col, sc in r.columns]
                retrieved_columns_detail.append(
                    {'table': r.table, 'table_score': float(r.table_score), 'columns': cols})
                for col, _ in r.columns:
                    key = normalize_column_key(col)
                    if key:
                        retrieved_column_keys.append(key)

            record.update({
                'retrieved_tables': retrieved_tables,
                'retrieved_column_keys': retrieved_column_keys,
                'retrieved_columns_detail': retrieved_columns_detail,
                'method_evidence': (results[0].metadata if results else None),
                'query_time': query_time,
                'memory_mb': memory_mb,
                'exec_accuracy': 'not_run',
            })

            if has_structured_gold(row):
                # ========== BIRD multi-table evaluation ==========
                gold_tables = record['gold_tables']
                gold_columns = [f"{t}.{f}" for t, f in parse_ground_truth_fields(row) if t]

                m = evaluate_multi_table_query(
                    gold_tables, gold_columns, retrieved_tables, retrieved_column_keys)

                # Compatibility aliases (old semantics: empty gold counts 0;
                # SCS with empty gold counts correct)
                gold_set = set(gold_tables)
                retrieved_set = set(retrieved_tables)
                table_exact = gold_set.issubset(retrieved_set) if gold_set else False
                gold_col_set = set(gold_columns)
                pred_col_set = set(retrieved_column_keys)
                inter = gold_col_set & pred_col_set
                c_recall_legacy = len(inter) / len(gold_col_set) if gold_col_set else 0.0
                schema_sql_correct_legacy = table_exact and (not gold_col_set or c_recall_legacy >= 1.0)
                recall_at = m['table_recall_at_k']

                record.update({
                    'multi_table': len(gold_tables) > 1,
                    'table_recall_at_k': recall_at,
                    'all_gold_covered': m['all_gold_covered'],
                    'column_precision': m['column_precision'],
                    'column_recall': m['column_recall'],
                    'column_f1': m['column_f1'],
                    'annotated_field_coverage_success': m['annotated_field_coverage_success'],
                    'schema_coverage_success': m['schema_coverage_success'],  # compatibility alias
                    'gold_field_status': m['gold_field_status'],
                    'gold_column_keys': m['gold_column_keys'],
                    # ---- compatibility aliases (old fields/semantics) ----
                    'table_exact_match': table_exact,
                    'table_accuracy_hit': table_exact,
                    'table_recall_full': (len(gold_set & retrieved_set) / len(gold_set))
                                        if gold_set else 0.0,
                    'table_recall_at_1': recall_at.get(1) or 0.0,
                    'table_recall_at_3': recall_at.get(3) or 0.0,
                    'table_recall_at_5': recall_at.get(5) or 0.0,
                    'schema_sql_correct': schema_sql_correct_legacy,
                    'sql_accuracy_hit': schema_sql_correct_legacy,
                })
            else:
                # ========== PowerGrid single-table evaluation ==========
                gt_table = row.get('table', '')
                gt_fields = row.get('field', '')

                m = evaluate_single_table_query(
                    gt_table if pd.notna(gt_table) else None,
                    gt_fields if isinstance(gt_fields, str) else None,
                    retrieved_tables, retrieved_column_keys)

                hit = m['hit']

                # Compatibility aliases (old semantics: column metrics only
                # over the hit table's returned fields; empty gold counts as
                # SCS-correct)
                gt_table_simple = gt_table.split('.')[-1] if pd.notna(gt_table) else ''
                table_hit_legacy = bool(gt_table_simple) and gt_table_simple in retrieved_tables
                c_recall = c_precision = c_f1 = 0.0
                sql_hit = False
                if table_hit_legacy:
                    retrieved_fields = []
                    for r in results:
                        if r.table == gt_table_simple:
                            for col, _ in r.columns:
                                field_name = col.split('.')[-1]
                                if field_name not in retrieved_fields:
                                    retrieved_fields.append(field_name)
                    gt_field_list = m['gold_column_keys'] and [k.split('.')[-1] for k in m['gold_column_keys']]
                    if gt_field_list:
                        sql_hit = all(f.lower() in {x.lower() for x in retrieved_fields}
                                      for f in gt_field_list)
                        gt_set = {f.lower() for f in gt_field_list}
                        pred_set = {x.lower() for x in retrieved_fields}
                        inter_l = gt_set & pred_set
                        c_recall = len(inter_l) / len(gt_set) if gt_set else 0.0
                        c_precision = len(inter_l) / len(pred_set) if pred_set else 0.0
                        c_f1 = (2 * c_recall * c_precision / (c_recall + c_precision)) \
                            if (c_recall + c_precision) > 0 else 0.0
                    else:
                        sql_hit = True
                else:
                    sql_hit = False

                record.update({
                    'multi_table': False,
                    'hit_at_1': bool(hit.get(1)),
                    'hit_at_3': bool(hit.get(3)),
                    'hit_at_5': bool(hit.get(5)),
                    'column_precision': m['column_precision'],
                    'column_recall': m['column_recall'],
                    'column_f1': m['column_f1'],
                    'annotated_field_coverage_success': m['annotated_field_coverage_success'],
                    'schema_coverage_success': m['schema_coverage_success'],  # compatibility alias
                    'gold_field_status': m['gold_field_status'],
                    'gold_column_keys': m['gold_column_keys'],
                    # ---- compatibility aliases (old fields/semantics) ----
                    'table_exact_match': table_hit_legacy,
                    'table_accuracy_hit': table_hit_legacy,
                    'table_recall_full': 1.0 if table_hit_legacy else 0.0,
                    'table_recall_at_1': 1.0 if hit.get(1) else 0.0,
                    'table_recall_at_3': 1.0 if hit.get(3) else 0.0,
                    'table_recall_at_5': 1.0 if hit.get(5) else 0.0,
                    'column_recall_legacy': c_recall,
                    'column_precision_legacy': c_precision,
                    'column_f1_legacy': c_f1,
                    'schema_sql_correct': sql_hit,
                    'sql_accuracy_hit': sql_hit,
                })

            records.append(record)

        return records

    @staticmethod
    def _aggregate_records(records: List[Dict[str, Any]]) -> EvaluationMetrics:
        """Aggregate per-question evaluation records into EvaluationMetrics.

        Compatibility note (P1-A): official reports use
        eval/retrieval_metrics.aggregate_records; this method serves only
        the legacy pipeline (--legacy, BIRD runner compatibility fields).
        PowerGrid records' column_* fields already carry the new semantics
        (full-union key level); the old semantics live in the column_*_legacy
        fields, which take priority here to keep the legacy entry points
        unchanged.
        """
        total = len(records)
        if total == 0:
            return EvaluationMetrics(total_queries=0)

        def mean(key: str, legacy_key: str = None) -> float:
            src = legacy_key if (legacy_key and legacy_key in records[0]) else key
            return float(np.mean([r.get(src, 0.0) for r in records])) if records else 0.0

        return EvaluationMetrics(
            table_accuracy=mean('table_exact_match'),
            sql_accuracy=mean('schema_sql_correct'),
            avg_query_time=mean('query_time'),
            avg_memory_mb=mean('memory_mb'),
            total_queries=total,
            table_recall=mean('table_recall_full'),
            column_recall=mean('column_recall', 'column_recall_legacy'),
            column_precision=mean('column_precision', 'column_precision_legacy'),
            column_f1=mean('column_f1', 'column_f1_legacy'),
            top1_table_recall=mean('table_recall_at_1'),
            top3_table_recall=mean('table_recall_at_3'),
            top5_table_recall=mean('table_recall_at_5'),
        )

    @property
    def name(self) -> str:
        """Get the retriever name."""
        return self.config.name

    @property
    def is_fitted(self) -> bool:
        """Whether the retriever is fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', fitted={self._is_fitted})"
