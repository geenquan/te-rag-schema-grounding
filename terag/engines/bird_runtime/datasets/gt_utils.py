"""
Ground-truth parsing utilities.

Uniformly handles the two QA annotation formats — PowerGrid (single table,
legacy table/field columns) and BIRD (multi-table, structured
gold_tables/gold_fields columns) — for reuse by retriever evaluation,
graph building, training statistics, and other modules, avoiding scattered
`table.split('.')[-1]`-style parsing.
"""

import json
import math
from typing import List, Tuple

import pandas as pd


def _get_cell(row, key: str):
    """Safely read a row cell (pandas.Series or dict); NaN/missing maps to None."""
    try:
        if key not in row:
            return None
        value = row[key]
    except TypeError:
        return None
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_json_list(value) -> List:
    """
    Parse a gold_tables / gold_fields cell into a list.

    Accepted forms: a native list (read from jsonl), a JSON string (read
    from CSV), NaN, and legacy '|' separated strings.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return []
    if s.startswith('['):
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    # legacy '|' separated string
    return [x.strip() for x in s.split('|') if x.strip()]


def parse_ground_truth_tables(row) -> List[str]:
    """
    Parse the gold table list (canonical table ids) of one QA row.

    Rules:
    1. Structured gold_tables column (BIRD): used as-is (canonical ids,
       e.g. "db.table").
    2. Legacy table column (PowerGrid): "db.table" → keep the last segment
       as the simple table name.

    Args:
        row: QA row (pandas.Series or dict)

    Returns:
        List of gold table ids (order of appearance preserved)
    """
    gold = load_json_list(_get_cell(row, 'gold_tables'))
    if gold:
        seen = []
        for t in gold:
            t = str(t)
            if t and t not in seen:
                seen.append(t)
        return seen

    raw = _get_cell(row, 'table')
    if raw is None:
        return []
    parts = [p.strip() for p in str(raw).split('|') if p.strip()]
    return [p.split('.')[-1] for p in parts]


def parse_ground_truth_fields(row) -> List[Tuple[str, str]]:
    """
    Parse the gold field list of one QA row.

    Rules:
    1. Structured gold_fields column (BIRD): a list of [table_id, column]
       pairs (or "table.column" strings).
    2. Legacy field column (PowerGrid): bare field names attached to the
       gold table (single-table case).

    Args:
        row: QA row (pandas.Series or dict)

    Returns:
        [(table_id, field_name), ...]
    """
    gold = load_json_list(_get_cell(row, 'gold_fields'))
    if gold:
        out = []
        for item in gold:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pair = (str(item[0]), str(item[1]))
            else:
                s = str(item)
                if '.' in s:
                    pair = tuple(s.rsplit('.', 1))
                else:
                    pair = ('', s)
            if pair[1]:
                out.append(pair)
        return out

    # legacy: bare field names + the single gold table
    tables = parse_ground_truth_tables(row)
    raw = _get_cell(row, 'field')
    if raw is None or not tables:
        return []
    fields = [f.strip() for f in str(raw).split('|') if f.strip()]
    return [(tables[0], f) for f in fields]


def is_multi_table_row(row) -> bool:
    """Return whether one QA row is a multi-table query."""
    return len(parse_ground_truth_tables(row)) > 1


def has_structured_gold(row) -> bool:
    """Return whether one QA row carries structured gold annotations (gold_tables / gold_fields)."""
    return bool(load_json_list(_get_cell(row, 'gold_tables'))) or \
        bool(load_json_list(_get_cell(row, 'gold_fields')))


def gold_column_keys(row) -> List[str]:
    """Return the canonical key list ("table.column") of the gold columns."""
    return [f"{t}.{f}" for t, f in parse_ground_truth_fields(row) if t]
