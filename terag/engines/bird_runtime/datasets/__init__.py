"""Dataset adapter modules.

Convert external datasets (BIRD and others) into the experiment
framework's standardized format; dataset differences stay inside the
adapter layer instead of leaking into the retrievers.
"""

from terag.engines.bird_runtime.datasets.base_adapter import BaseDatasetAdapter
from terag.engines.bird_runtime.datasets.bird_adapter import BirdDatasetAdapter
from terag.engines.bird_runtime.datasets import gt_utils
from terag.engines.bird_runtime.datasets.gt_utils import (
    parse_ground_truth_tables,
    parse_ground_truth_fields,
    load_json_list,
    is_multi_table_row,
    has_structured_gold,
    gold_column_keys,
)

__all__ = [
    'BaseDatasetAdapter',
    'BirdDatasetAdapter',
    'gt_utils',
    'parse_ground_truth_tables',
    'parse_ground_truth_fields',
    'load_json_list',
    'is_multi_table_row',
    'has_structured_gold',
    'gold_column_keys',
]
