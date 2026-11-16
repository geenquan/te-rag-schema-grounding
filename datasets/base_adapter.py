"""
Base class for dataset adapters.

Defines the interface for converting an external dataset into the
framework's standardized format (table_df / field_df / qa_df).
Dataset-specific differences should stay concentrated in the adapter layer
rather than spread across individual retrievers.

The standardized output format follows PowerGrid:
- table_df:  table, db, table_desc, field, schema
- field_df:  table, db, field_name, field_name_desc, field_type
- qa_df:     question, table, field, sql (BIRD additionally carries
  structured columns such as db_id / gold_tables / gold_fields)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    @abstractmethod
    def load_queries(self) -> List[Dict]:
        """Load the raw query data."""
        pass

    @abstractmethod
    def load_schema(self) -> Dict:
        """Load the raw schema data."""
        pass

    @abstractmethod
    def build_table_dataframe(self) -> pd.DataFrame:
        """
        Build the standardized table-schema DataFrame.

        Returns:
            A DataFrame with the table / db / table_desc / field / schema columns
        """
        pass

    @abstractmethod
    def build_field_dataframe(self) -> pd.DataFrame:
        """
        Build the standardized field-schema DataFrame.

        Returns:
            A DataFrame with the table / db / field_name / field_name_desc /
            field_type columns
        """
        pass

    @abstractmethod
    def build_qa_dataframe(self) -> pd.DataFrame:
        """
        Build the standardized QA DataFrame.

        Returns:
            A DataFrame with the question / table / field / sql columns
            (multi-table datasets additionally carry the structured
            gold_tables / gold_fields columns)
        """
        pass

    def prepare(self) -> Dict[str, pd.DataFrame]:
        """
        Run the full data preparation pipeline.

        Returns:
            {'table_df': ..., 'field_df': ..., 'qa_df': ...}
        """
        self.load_schema()
        self.load_queries()
        return {
            'table_df': self.build_table_dataframe(),
            'field_df': self.build_field_dataframe(),
            'qa_df': self.build_qa_dataframe(),
        }
