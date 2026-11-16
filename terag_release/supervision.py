"""Training supervision injection.

The formal protocols supervise TRAINING rows with their full SQL structure:
the full gold's complete physical column set (``gold_columns``: everything the
reference SQL needs — SELECT, WHERE, JOIN, GROUP BY, ... ) is attached to each
training row as structured ``gold_tables``/``gold_fields`` columns, which the
graph builder and the history-consuming retrievers read via
``datasets.gt_utils``. Test rows are never injected.

The alternative arm ``clean_projection`` attaches only the top-level SELECT
projection columns (kept from the historical development protocol for
completeness; the paper's formal runs use ``full_sql`` exclusively).
"""

import json
from typing import Dict

import pandas as pd


def inject_supervision(train_df: pd.DataFrame, gold: Dict[str, Dict],
                       arm: str = "full_sql") -> pd.DataFrame:
    """Attach structured gold supervision columns to training rows.

    Args:
        train_df: training rows; must contain a ``qid`` column.
        gold: full-gold records keyed by qid (see docs/data_format.md).
        arm: ``full_sql`` (all SQL-required columns) or ``clean_projection``.

    Returns:
        A copy of the frame with ``gold_tables`` (JSON list of deduplicated
        table names, first-seen order) and ``gold_fields`` (JSON list of
        ``[table, field]`` pairs, one per gold column) added.
    """
    if arm not in ("full_sql", "clean_projection"):
        raise ValueError(f"unknown supervision arm: {arm}")
    df = train_df.copy()
    tables_col, fields_col = [], []
    for _, row in df.iterrows():
        qid = str(row["qid"])
        if qid not in gold:
            raise KeyError(f"supervision requested but qid {qid} lacks gold")
        g = gold[qid]
        cols = (g["projection_columns"] if arm == "clean_projection"
                else g["gold_columns"])
        pairs, tabs = [], []
        for c in cols:
            t, f = c.split(".", 1)
            pairs.append([t, f])
            if t not in tabs:
                tabs.append(t)
        tables_col.append(json.dumps(tabs, ensure_ascii=False))
        fields_col.append(json.dumps(pairs, ensure_ascii=False))
    df["gold_tables"] = tables_col
    df["gold_fields"] = fields_col
    return df
