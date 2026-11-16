"""Metric-engine tests: normalization, denominators, budgets, prefix crop."""

import unittest

from eval.retrieval_metrics import (all_gold_covered, column_prf,
                                    normalize_column_key, normalize_table,
                                    schema_coverage_success, table_recall_at_k)
from eval.full_gold import (full_gold_metrics, index_predictions,
                            summarize_predictions, validate_population)
from analysis.saved_prefix import crop, eligible


class TestNormalization(unittest.TestCase):
    def test_table_and_column_normalization(self):
        self.assertEqual(normalize_table("DB.Tbl"), "tbl")
        self.assertEqual(normalize_table("T:abc"), "abc")
        self.assertEqual(normalize_column_key("C:schema.col"), "schema.col")
        self.assertEqual(normalize_column_key("a.b.c"), "b.c")

    def test_fgcs_uses_normalized_gold_columns(self):
        # The paper's case-sensitivity correction: gold column casing must
        # not flip FGCS once normalization is applied.
        self.assertTrue(schema_coverage_success(
            ["T"], ["t.Col"], ["t"], ["t.col"], k=5))


class TestDenominators(unittest.TestCase):
    def test_error_records_stay_in_denominator(self):
        record = {"qid": "q1", "status": "error",
                  "retrieved_tables": ["t1"],  # stale output must be ignored
                  "retrieved_column_keys": ["t1.c1"]}
        gold = {"q1": {"gold_tables": ["t1"], "gold_columns": ["t1.c1"]}}
        tr, cov, p, r, f1, fgcs = full_gold_metrics(record, gold)
        self.assertFalse(cov)
        self.assertFalse(fgcs)
        self.assertEqual(p, 0.0)

    def test_empty_prediction_scores_zero_not_none(self):
        record = {"qid": "q1", "status": "ok", "retrieved_tables": [],
                  "retrieved_column_keys": []}
        gold = {"q1": {"gold_tables": ["t1"], "gold_columns": ["t1.c1"]}}
        _, cov, p, r, f1, fgcs = full_gold_metrics(record, gold)
        self.assertFalse(cov)
        self.assertEqual((p, r, f1), (0.0, 0.0, 0.0))

    def test_duplicate_qid_rejected(self):
        with self.assertRaises(ValueError):
            index_predictions([{"qid": "q1"}, {"qid": "q1"}])

    def test_population_validation(self):
        gold = {"q1": {"gold_tables": ["t"], "gold_columns": ["t.c"]},
                "q2": {"gold_tables": ["t"], "gold_columns": ["t.c"]},
                "q_bad": {"gold_tables": [], "gold_columns": []}}
        validate_population(gold, valid=["q1"], feasible=["q1"])
        with self.assertRaises(ValueError):
            validate_population(gold, valid=["q1"], feasible=["q2"])
        with self.assertRaises(ValueError):
            validate_population(gold, valid=["q3"], feasible=[])
        with self.assertRaises(ValueError):
            validate_population(gold, valid=["q_bad"], feasible=["q_bad"])


class TestSummaries(unittest.TestCase):
    def _gold(self, n):
        return {f"q{i}": {"gold_tables": ["t1"],
                          "gold_columns": ["t1.c1", "t1.c2"]}
                for i in range(n)}

    def test_macro_means_and_counts(self):
        gold = self._gold(4)
        records = [
            {"qid": "q0", "status": "ok", "retrieved_tables": ["t1"],
             "retrieved_column_keys": ["t1.c1", "t1.c2"]},
            {"qid": "q1", "status": "ok", "retrieved_tables": ["t1"],
             "retrieved_column_keys": ["t1.c1"]},
            {"qid": "q2", "status": "error", "retrieved_tables": [],
             "retrieved_column_keys": []},
            {"qid": "q3", "status": "ok", "retrieved_tables": ["t9"],
             "retrieved_column_keys": ["t9.z"]},
        ]
        row = summarize_predictions(records, gold, valid=list(gold),
                                    feasible=["q0", "q1"])
        self.assertEqual(row["n"], 4)
        self.assertEqual(row["n_feasible"], 2)
        self.assertAlmostEqual(row["covered@5"], 0.5)
        self.assertAlmostEqual(row["colrec"], (1.0 + 0.5 + 0.0 + 0.0) / 4)
        self.assertEqual(row["fgcs_successes"], 1)
        self.assertEqual(row["n_error"], 1)
        self.assertEqual(row["n_empty_prediction"], 1)


class TestBudgets(unittest.TestCase):
    def test_eligible_recomputed_from_gold(self):
        small = {"gold_tables": ["t1"], "gold_columns": ["t1.c1"] * 1}
        big = {"gold_tables": ["t1"], "gold_columns": [f"t1.c{i}"
                                                       for i in range(11)]}
        many = {"gold_tables": [f"t{i}" for i in range(6)],
                "gold_columns": ["t0.c"]}
        self.assertTrue(eligible(small, 10))
        self.assertFalse(eligible(big, 10))
        self.assertFalse(eligible(many, 10))

    def test_crop_truncates_and_rebuilds_keys(self):
        record = {
            "status": "ok",
            "retrieved_tables": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "retrieved_column_keys": [f"t{i}.c{j}" for i in range(1, 7)
                                      for j in range(4)],
            "retrieved_columns_detail": [
                {"table": f"t{i}",
                 "table_score": 1.0 / (i + 1),
                 "columns": [{"key": f"t{i}.c{j}", "score": 1.0}
                             for j in range(4)]}
                for i in range(1, 7)],
        }
        cropped = crop(record, 3)
        self.assertEqual(cropped["retrieved_tables"],
                         ["t1", "t2", "t3", "t4", "t5"])
        for entry in cropped["retrieved_columns_detail"]:
            self.assertEqual(len(entry["columns"]), 3)
        # top-5 tables x 3 slots, in original order, deduplicated
        self.assertEqual(cropped["retrieved_column_keys"],
                         [f"t{i}.c{j}" for i in range(1, 6) for j in range(3)])

    def test_crop_keeps_error_records_empty(self):
        record = {"status": "error", "retrieved_tables": ["t1"],
                  "retrieved_column_keys": ["t1.c1"],
                  "retrieved_columns_detail": [
                      {"table": "t1", "columns": [{"key": "t1.c1"}]}]}
        cropped = crop(record, 10)
        self.assertEqual(cropped["retrieved_tables"], ["t1"])


if __name__ == "__main__":
    unittest.main()
