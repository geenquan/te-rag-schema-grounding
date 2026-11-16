"""Statistics tests: Wilcoxon, cluster bootstrap, Bonferroni, zero diffs.

Exercises the paper's statistical machinery on fully synthetic prediction
groups (fixture records fabricated for the test — they validate the CODE,
not any method).
"""

import unittest

from eval.full_gold import paired_covered
from experiment.audit_saved_predictions import audit, FAMILIES


def make_gold(n, tables):
    """n questions spread over the given target-table clusters."""
    return {f"q{i:03d}": {"qid": f"q{i:03d}",
                           "gold_tables": [tables[i % len(tables)]],
                           "gold_columns": [f"{tables[i % len(tables)]}.c1"],
                           "gold_status": "verified_complete",
                           "original_table": tables[i % len(tables)]}
            for i in range(n)}


def make_predictions(gold, covered):
    """Records covering exactly the qids listed in ``covered``."""
    return [{"qid": q, "setting": "question_v1", "variant": "x",
             "status": "ok",
             "retrieved_tables": ([g["gold_tables"][0]] if q in covered
                                  else ["other"]),
             "retrieved_column_keys": [f'{g["gold_tables"][0]}.c1']}
            for q, g in gold.items()]


class TestPairedCovered(unittest.TestCase):
    def setUp(self):
        self.gold = make_gold(30, ["ta", "tb", "tc"])
        self.clusters = {q: g["original_table"] for q, g in self.gold.items()}

    def test_wilcoxon_and_cluster_ci(self):
        a = make_predictions(self.gold, covered=set(list(self.gold)[:20]))
        b = make_predictions(self.gold, covered=set(list(self.gold)[:8]))
        res = paired_covered(a, b, self.gold, list(self.gold), self.clusters)
        self.assertGreater(res["wilcoxon_p_raw"], 0)
        self.assertAlmostEqual(res["mean_diff"], (20 - 8) / 30, places=6)
        self.assertEqual(len(res["ci95_cluster_bootstrap"]), 2)
        self.assertLessEqual(res["ci95_cluster_bootstrap"][0],
                             res["ci95_cluster_bootstrap"][1])
        self.assertEqual(res["statistical_method"]["seed"], 42)
        self.assertEqual(res["statistical_method"]["resamples"], 10000)

    def test_zero_difference_has_no_fabricated_p(self):
        a = make_predictions(self.gold, covered=set())
        b = make_predictions(self.gold, covered=set())
        res = paired_covered(a, b, self.gold, list(self.gold), self.clusters)
        self.assertTrue(res["zero_difference"])
        self.assertNotIn("wilcoxon_p_raw", res)

    def test_bootstrap_is_seeded(self):
        a = make_predictions(self.gold, covered=set(list(self.gold)[:15]))
        b = make_predictions(self.gold, covered=set())
        r1 = paired_covered(a, b, self.gold, list(self.gold), self.clusters)
        r2 = paired_covered(a, b, self.gold, list(self.gold), self.clusters)
        self.assertEqual(r1["ci95_cluster_bootstrap"],
                         r2["ci95_cluster_bootstrap"])


class _TempDir:
    def __init__(self, path):
        import tempfile, shutil
        self._mk = tempfile.mkdtemp
        self.path = path

    def __enter__(self):
        import os
        os.makedirs(self.path, exist_ok=True)
        return self.path

    def __exit__(self, *a):
        import shutil
        shutil.rmtree(self.path, ignore_errors=True)


class TestFamilyAssembly(unittest.TestCase):
    """audit() family handling: complete families compute; incomplete ones
    are reported as not_computed with their missing members — never shrunk."""

    def _write_run(self, tmp, gold, groups):
        import json, os
        pred_path = os.path.join(tmp, "preds.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for variant, covered in groups.items():
                for rec in make_predictions(gold, covered):
                    rec["variant"] = variant
                    f.write(json.dumps(rec) + "\n")
        gold_path = os.path.join(tmp, "gold.jsonl")
        with open(gold_path, "w", encoding="utf-8") as f:
            for qid in sorted(gold):
                f.write(json.dumps(gold[qid]) + "\n")
        budget = {"n_verified": len(gold), "n_feasible": len(gold),
                  "infeasible_qids": [],
                  "per_question": [{"qid": q, "budget_feasible": True}
                                   for q in gold]}
        budget_path = os.path.join(tmp, "budget.json")
        with open(budget_path, "w", encoding="utf-8") as f:
            json.dump(budget, f)
        return gold_path, budget_path, pred_path

    def test_incomplete_family_recorded_not_computed(self):
        gold = make_gold(20, ["ta", "tb"])
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            gold_path, budget_path, pred_path = self._write_run(
                tmp, gold, {"TERAG-Full": set(), "ExampleRRF": set()})
            result = audit(gold_path, budget_path, [pred_path],
                           families="six")
            six = result["comparisons"]["six"]
            self.assertEqual(six["status"], "not_computed")
            self.assertIn("Dense", six["missing_members"])
            self.assertIn("TERAG-noBC", six["missing_members"])

    def test_complete_family_computes_with_bonferroni(self):
        gold = make_gold(40, ["ta", "tb", "tc", "td"])
        groups = {"TERAG-Full": set(list(gold)[:30])}
        for member in FAMILIES["six"]:
            groups[member] = set()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            gold_path, budget_path, pred_path = self._write_run(
                tmp, gold, groups)
            result = audit(gold_path, budget_path, [pred_path],
                           families="six")
            six = result["comparisons"]["six"]
            self.assertNotEqual(six.get("status"), "not_computed")
            for key, comp in six.items():
                if key.startswith("Full_vs_"):
                    self.assertEqual(comp["bonferroni_k"], 6)
                    self.assertGreaterEqual(
                        comp["wilcoxon_p_bonferroni"],
                        comp["wilcoxon_p_raw"])
                    self.assertLessEqual(comp["wilcoxon_p_bonferroni"], 1.0)


if __name__ == "__main__":
    unittest.main()
