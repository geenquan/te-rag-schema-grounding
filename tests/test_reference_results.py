"""Reference-results integrity: the shipped BIRD reference predictions are
complete, self-consistent with their manifest, free of sanitized-away
content, and carry the paper's key values in the aggregates."""

import csv
import gzip
import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REF = REPO_ROOT / "results" / "reference" / "bird"

KEY_ROWS = [
    # (protocol, method, metric, expected percent)
    ("question", "ORIGINAL_FULL", "covered@5", 93.7908),
    ("question", "ORIGINAL_FULL", "colf1", 25.3027),
    ("question", "qB_rerank_gate_v1-field_filter_gate_off", "colf1", 25.9667),
    ("question", "qB_rerank_gate_v1-field_filter_gate_off", "fgcs", 81.2418),
    ("question", "qB_rerank_gate_v1", "colf1", 27.8742),
    ("question", "qB_rerank_gate_v1", "fgcs", 77.9085),
    ("heldout", "ORIGINAL_FULL", "covered@5", 76.5359),
    ("heldout", "full_href_1_8", "covered@5", 82.2876),
    ("heldout", "full_href_1_8", "colf1", 22.9843),
    ("heldout", "full_href_1_8", "fgcs", 57.9739),
    ("question", "optimized_full_Q", "colf1", 29.0187),
    ("question", "optimized_Q_noBIndex", "colf1", 29.5297),
]


class TestReferenceResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (REF / "MANIFEST.json").read_text(encoding="utf-8"))
        cls.aggregates = {}
        for protocol in ("question", "heldout"):
            with (REF / "aggregates" / f"{protocol}_summary.csv").open(
                    encoding="utf-8") as f:
                cls.aggregates[protocol] = {r["method"]: r
                                            for r in csv.DictReader(f)}

    def test_manifest_files_exist_and_hash(self):
        checked = 0
        for key, info in self.manifest["methods"].items():
            protocol, method = key.split(":", 1)
            path = REF / "predictions" / protocol / f"{method}.jsonl.gz"
            self.assertTrue(path.is_file(), f"missing {path}")
            self.assertEqual(info["n_questions"], 1530, key)
            self.assertIn(info["source_kind"],
                          ("unified_entry_rerun", "reused_hash_verified"))
            checked += 1
        self.assertEqual(checked, 104)  # 54 question + 50 heldout

    def test_predictions_sanitized(self):
        """No prompts, machine paths, request/response ids, or API ledger
        fields may appear in the normalized records."""
        banned_keys = {"prompt", "response", "req_id", "response_id",
                       "usage", "cost_cny", "error", "query_time",
                       "rewritten_question", "candidate_columns"}
        path_re = re.compile(r"/Users/|research_runs|miniconda")
        n = 0
        for key in sorted(self.manifest["methods"]):
            protocol, method = key.split(":", 1)
            with gzip.open(REF / "predictions" / protocol
                           / f"{method}.jsonl.gz", "rt",
                           encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    n += 1
                    self.assertEqual(
                        set(rec) & banned_keys, set(),
                        f"{key}: unsanitized keys {sorted(set(rec) & banned_keys)}")
                    for t in rec.get("retrieved_tables") or []:
                        self.assertFalse(path_re.search(str(t)))
                    self.assertLessEqual(len(rec.get("retrieved_tables")
                                             or []), 5)
                    per_table = {}
                    for ck in rec.get("retrieved_column_keys") or []:
                        tbl = ck.rsplit(".", 1)[0]
                        per_table[tbl] = per_table.get(tbl, 0) + 1
                    self.assertTrue(all(v <= 10 for v in per_table.values()),
                                    f"{key}: column budget")
        self.assertEqual(n, 104 * 1530)

    def test_paper_key_values_in_aggregates(self):
        for protocol, method, metric, expected in KEY_ROWS:
            row = self.aggregates[protocol].get(method)
            self.assertIsNotNone(row, f"{protocol}/{method} missing")
            self.assertAlmostEqual(float(row[metric]), expected, delta=0.02,
                                   msg=f"{protocol}/{method}/{metric}")

    def test_gate_and_bindex_boundaries_visible(self):
        """The scientific boundaries stay in the data: Gate costs recall and
        FGCS vs rerank-only; optimized Full−noBIndex is negative."""
        q = self.aggregates["question"]
        gate = float(q["qB_rerank_gate_v1"]["colrec"])
        rerank = float(q["qB_rerank_gate_v1-field_filter_gate_off"]["colrec"])
        self.assertLess(gate, rerank, "gate must cost column recall")
        gate_fgcs = float(q["qB_rerank_gate_v1"]["fgcs"])
        rerank_fgcs = float(
            q["qB_rerank_gate_v1-field_filter_gate_off"]["fgcs"])
        self.assertLess(gate_fgcs, rerank_fgcs, "gate must cost FGCS")
        self.assertLess(float(q["optimized_full_Q"]["colf1"]),
                        float(q["optimized_Q_noBIndex"]["colf1"]),
                        "optimized Full−noBIndex stays negative")

    def test_no_enterprise_predictions(self):
        """No power-grid / enterprise method may appear in the reference."""
        for protocol in ("question", "heldout"):
            for method in self.aggregates[protocol]:
                self.assertNotIn("grid", method.lower())
                self.assertNotIn("powergrid", method.lower())

    def test_per_question_metrics_shape(self):
        path = (REF / "metrics" / "question"
                / "qB_rerank_gate_v1_per_question.csv")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1530)
        for col in ("tr@1", "covered@5", "colf1", "fgcs"):
            self.assertIn(col, rows[0])


if __name__ == "__main__":
    unittest.main()
