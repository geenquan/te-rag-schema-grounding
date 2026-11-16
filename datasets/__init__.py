"""Dataset utilities package (public release).

``gt_utils`` provides the shared gold-annotation parsing used by retrievers,
evaluation, and graph construction. ``base_adapter`` defines the adapter
interface of the private research framework. The BIRD benchmark adapter is
not shipped: BIRD is not part of the paper's experiment matrix (see
docs/experiment_catalog.md).
"""

from datasets.base_adapter import BaseDatasetAdapter
from datasets import gt_utils
from datasets.gt_utils import (
    parse_ground_truth_tables,
    parse_ground_truth_fields,
    load_json_list,
    is_multi_table_row,
    has_structured_gold,
    gold_column_keys,
)

__all__ = [
    'BaseDatasetAdapter',
    'gt_utils',
    'parse_ground_truth_tables',
    'parse_ground_truth_fields',
    'load_json_list',
    'is_multi_table_row',
    'has_structured_gold',
    'gold_column_keys',
]
