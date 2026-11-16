"""End-to-end demo pipeline test (offline; a single fold per protocol to
keep the suite fast). Marks clearly: this validates the CODE PATH on
synthetic data, not any paper number."""

import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestDemoPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import subprocess
        import sys
        cls.tmp = tempfile.mkdtemp(prefix="terag_demo_test_")
        run_dir = Path(cls.tmp) / "run"
        cls.run_result = subprocess.run(
            [sys.executable, "-m", "terag_release", "run-experiments",
             "--dataset", "demo", "--experiment", "demo_offline",
             "--offline-only", "--fold", "0", "--diagnostics",
             "--output", str(run_dir)],
            cwd=REPO_ROOT, capture_output=True, text=True)
        cls.run_dir = run_dir

    def test_run_succeeds(self):
        self.assertEqual(self.run_result.returncode, 0, self.run_result.stderr[-2000:])

    def test_predictions_written_and_stamped(self):
        pred_path = self.run_dir / "predictions.jsonl"
        self.assertTrue(pred_path.exists())
        records = [json.loads(line) for line in
                   pred_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertGreater(len(records), 0)
        for rec in records[:20]:
            for key in ("qid", "setting", "fold", "variant", "method",
                        "status"):
                self.assertIn(key, rec)
        statuses = {r["status"] for r in records}
        self.assertTrue(statuses <= {"ok", "error"})

    def test_run_meta_provenance(self):
        meta = json.loads((self.run_dir / "run_meta" / "run_meta.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(meta["code"]["algorithm_identity"],
                         "transfer_v3_target_cap")
        self.assertEqual(meta["code"]["paper_tag"], "paper-dasfaa2027-v1.0")
        self.assertIn("sha256", meta["config"])
        self.assertIn("files_sha256", meta["code"])
        self.assertIn("environment", meta)
        self.assertIn("partial run", " ".join(meta["execution"]["notes"]))

    def test_diagnostics_export_and_transfer_flow(self):
        diag_root = self.run_dir / "diagnostics"
        self.assertTrue(diag_root.exists())
        annotation_files = list(diag_root.rglob("annotations.json"))
        self.assertGreater(len(annotation_files), 0)
        ann = json.loads(annotation_files[0].read_text(encoding="utf-8"))
        self.assertIn("direct", ann)
        self.assertIn("transfer", ann)
        transfer = ann["transfer"]
        self.assertIn("accepted", transfer)
        self.assertIn("rejected", transfer)
        for record in transfer["accepted"]:
            self.assertEqual(record.get("method_version"),
                             "transfer_v3_target_cap")
            self.assertEqual(record.get("status"), "accepted")

    def test_full_demo_and_evaluate_roundtrip(self):
        """Full 5-fold run + offline evaluation + a family verdict."""
        import subprocess
        import sys
        full_dir = Path(self.tmp) / "full"
        result = subprocess.run(
            [sys.executable, "-m", "terag_release", "run-demo",
             "--output", str(full_dir)],
            cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        summary = (full_dir / "summary_full_gold.csv").read_text()
        self.assertIn("TERAG-Full", summary)
        eval_dir = Path(self.tmp) / "eval"
        eval_result = subprocess.run(
            [sys.executable, "-m", "terag_release", "evaluate",
             "--gold", str(REPO_ROOT / "demo" / "data" / "full_gold_v2.jsonl"),
             "--budget", str(full_dir / "run_meta" /
                             "budget_feasibility.json"),
             "--predictions", str(full_dir / "predictions.jsonl"),
             "--output", str(eval_dir), "--families", "both"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(eval_result.returncode, 0, eval_result.stderr[-2000:])
        audit = json.loads((eval_dir / "audit.json").read_text())
        self.assertGreaterEqual(len(audit["summary"]), 18)
        # Offline recompute of saved predictions must match the run summary.
        by_key = {(r["setting"], r["method"]): r for r in audit["summary"]}
        full_row = by_key[("question_v1", "TERAG-Full")]
        run_rows = {}
        import csv
        for row in csv.DictReader(
                open(full_dir / "summary_full_gold.csv", encoding="utf-8")):
            run_rows[(row["setting"], row["variant"])] = row
        self.assertEqual(
            float(run_rows[("question_v1", "TERAG-Full")]["covered@5"]),
            full_row["covered@5"])


if __name__ == "__main__":
    unittest.main()
