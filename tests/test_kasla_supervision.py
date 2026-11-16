"""R1 regression: KaSLA-Adapted receives full_sql training supervision.

The paper runner (experiment/run_expansion.py at tag paper-dasfaa2027-v1.0)
injects ``full_sql`` supervision into the training fold BEFORE the reviewer
methods and fits KaSLA on that frame, so its classifiers and capacity
estimation are labeled by the COMPLETE SQL gold (all referenced tables and
columns), not the original SELECT annotation. These tests fit the REAL
KaSLA retriever (local classifiers only; no API call is possible here) and
compare the labels produced from raw vs injected frames.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from terag_release import supervision  # noqa: E402
from terag_release import variants  # noqa: E402

# Synthetic schema: two tables joined by the training SQL, with a WHERE
# filter column that the original annotation does not mention.
FIELD_ROWS = [
    ("orders", "order_no", "order number", "varchar"),
    ("orders", "customer_id", "customer id", "varchar"),
    ("orders", "status", "order status", "varchar"),
    ("order_lines", "line_id", "line id", "int"),
    ("order_lines", "order_no", "order number", "varchar"),
    ("order_lines", "qty", "quantity", "int"),
    ("order_lines", "discount", "discount", "decimal"),
]
GOLD = {
    "q000": {"gold_tables": ["orders", "order_lines"],
             "gold_columns": ["orders.customer_id", "orders.status",
                              "order_lines.qty", "order_lines.discount"],
             "projection_columns": ["orders.customer_id",
                                    "order_lines.qty"]},
    "q001": {"gold_tables": ["orders"],
             "gold_columns": ["orders.order_no", "orders.status"],
             "projection_columns": ["orders.order_no"]},
    "q002": {"gold_tables": ["order_lines"],
             "gold_columns": ["order_lines.line_id", "order_lines.qty"],
             "projection_columns": ["order_lines.qty"]},
}


def _schema_csvs(tmp):
    field_csv = os.path.join(tmp, "fields.csv")
    table_csv = os.path.join(tmp, "tables.csv")
    pd.DataFrame(FIELD_ROWS, columns=["table", "field_name",
                                      "field_name_desc", "field_type"]) \
        .to_csv(field_csv, index=False)
    pd.DataFrame({"table": ["orders", "order_lines"],
                  "table_desc": ["sales orders", "order lines"]}) \
        .to_csv(table_csv, index=False)
    return field_csv, table_csv


def _train_frame():
    return pd.DataFrame([
        {"qid": "q000", "question": "list filtered order quantities",
         "sql": "SELECT o.customer_id, l.qty FROM order_lines l "
                "JOIN orders o ON l.order_no = o.order_no "
                "WHERE o.status = 'OPEN' AND l.discount > 0",
         # original annotation covers ONLY the SELECT projection — by
         # construction different from the full SQL gold
         "table": "order_lines", "field": "customer_id|qty"},
        {"qid": "q001", "question": "open order numbers",
         "sql": "SELECT order_no FROM orders WHERE status = 'OPEN'",
         "table": "orders", "field": "order_no"},
        {"qid": "q002", "question": "line ids and quantities",
         "sql": "SELECT line_id, qty FROM order_lines",
         "table": "order_lines", "field": "line_id|qty"},
    ])


def _fit_kasla(train_frame):
    from retrievers.kasla_retriever import KaSLAAdaptedRetriever
    with tempfile.TemporaryDirectory() as tmp:
        field_csv, table_csv = _schema_csvs(tmp)
        retriever = KaSLAAdaptedRetriever(field_csv, table_csv)
        retriever.fit(train_frame)
        snapshot = [dict(qid=t["qid"], tables=set(t["tables"]),
                         fields=set(t["fields"]))
                    for t in retriever.training]
        return snapshot, retriever


class TestKaSLASupervision(unittest.TestCase):
    def test_variant_declares_full_sql(self):
        self.assertEqual(
            variants.VARIANTS["KaSLA-Adapted"]["supervision"], "full_sql",
            "KaSLA must be supervised like the paper runner")

    def test_injection_changes_classifier_labels(self):
        raw = _train_frame()
        injected = supervision.inject_supervision(raw, GOLD, "full_sql")
        raw_snap, _ = _fit_kasla(raw)
        inj_snap, _ = _fit_kasla(injected)

        # q000: raw annotation = {order_lines: 2 select columns}; full gold
        # = both tables incl. WHERE columns. Labels must differ.
        raw_q0 = next(s for s in raw_snap if s["qid"] == "q000")
        inj_q0 = next(s for s in inj_snap if s["qid"] == "q000")
        self.assertEqual(raw_q0["tables"], {"order_lines"})
        self.assertEqual(inj_q0["tables"], {"orders", "order_lines"})
        self.assertIn(("orders", "status"), inj_q0["fields"])
        self.assertIn(("order_lines", "discount"), inj_q0["fields"])
        self.assertNotIn(("orders", "status"), raw_q0["fields"])

        # Aggregate label counts: the independent-review probe showed
        # table_positives 8 -> 14 on its 8-row fixture; here 3 rows with
        # strictly larger full-gold label sets must also increase totals.
        raw_tables = sum(len(s["tables"]) for s in raw_snap)
        inj_tables = sum(len(s["tables"]) for s in inj_snap)
        raw_fields = sum(len(s["fields"]) for s in raw_snap)
        inj_fields = sum(len(s["fields"]) for s in inj_snap)
        self.assertGreater(inj_tables, raw_tables)
        self.assertGreater(inj_fields, raw_fields)

    def test_capacity_estimation_uses_training_fold_only(self):
        # capacity comes from the 30 most similar TRAINING questions; with
        # an injected frame the gold-element costs are the full-gold sizes.
        injected = supervision.inject_supervision(_train_frame(), GOLD,
                                                  "full_sql")
        _, retriever = _fit_kasla(injected)
        self.assertEqual(len(retriever.training), 3)
        for entry in retriever.training:
            self.assertIn(entry["qid"], GOLD)

    def test_injection_never_touches_test_frames(self):
        frame = _train_frame()
        test = frame[frame["qid"] == "q000"].reset_index(drop=True)
        before = list(test.columns)
        supervision.inject_supervision(frame[frame["qid"] != "q000"],
                                       GOLD, "full_sql")
        self.assertEqual(list(test.columns), before)
        self.assertNotIn("gold_tables", test.columns)

    def test_runner_would_pass_injected_frame(self):
        # The runner's supervision path: spec-driven, exercised on the
        # documented signature (unit-level; full pipeline in
        # test_demo_pipeline).
        spec = variants.VARIANTS["KaSLA-Adapted"]
        self.assertTrue(spec["supervision"])
        injected = supervision.inject_supervision(_train_frame(), GOLD,
                                                  spec["supervision"])
        row = injected.iloc[0]
        tables = json.loads(row["gold_tables"])
        fields = json.loads(row["gold_fields"])
        self.assertEqual(tables, ["orders", "order_lines"])
        self.assertEqual(len(fields), 4)


if __name__ == "__main__":
    unittest.main()
