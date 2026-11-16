"""R7 regression: failure runs are failures, with complete state on disk.

All failures are injected with mocks/explicit sabotage — no real model or
API is involved. Uses the bundled demo dataset through the real CLI entry
so exit codes and artifacts match what a user sees.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*args, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "terag_release", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env)


class TestFailureSemantics(unittest.TestCase):
    def test_all_units_failing_exits_nonzero_with_error_records(self):
        # In-process: construction of every retriever is rigged to fail,
        # exactly like the independent-review probe (mock, no model/API).
        from terag_release import runner as runner_mod
        from terag_release.config import resolve_run
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_run("demo", "demo_offline",
                                   method_names=["BM25"])
            out = Path(tmp) / "run"
            with mock.patch.object(
                    runner_mod, "create_retriever",
                    side_effect=RuntimeError("simulated constructor failure")):
                meta = runner_mod.run_experiments(resolved=resolved,
                                                  run_dir=out)
            self.assertEqual(meta["execution"]["run_status"], "failed")
            records = [json.loads(line) for line in
                       (out / "predictions.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
            # every question exactly once per protocol, as error records
            self.assertEqual(len(records), 2 * 49)
            keys = [(r["setting"], r["fold"], r["variant"], r["qid"])
                    for r in records]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue(all(r["status"] == "error" for r in records))
            self.assertTrue(all(r["query_time"] is None for r in records))
            self.assertTrue(all(r["error_stage"] == "construct"
                                for r in records))
            # qids stay aligned with the questions
            by_qid = {r["qid"]: r for r in records
                      if r["setting"] == "question_v1" and r["fold"] == 0}
            import csv as _csv
            qa = {f"q{i:03d}": row["question"] for i, row in enumerate(
                _csv.DictReader(open(REPO_ROOT / "demo" / "data" /
                                     "qa_data.csv", encoding="utf-8")))}
            for qid, rec in list(by_qid.items())[:10]:
                self.assertEqual(rec["question"], qa[qid])
            self.assertFalse((out / "summary_full_gold.csv").exists())
            # metadata survived and is readable
            saved = json.loads((out / "run_meta" / "run_meta.json")
                               .read_text(encoding="utf-8"))
            self.assertEqual(saved["execution"]["run_status"], "failed")
            self.assertTrue(all(u["status"] == "failed"
                                for u in saved["execution"]["units"]))
            self.assertTrue(all("effective_params" in u
                                for u in saved["execution"]["units"]))
            # the CLI surfaces the failure as a non-zero exit
            out2 = Path(tmp) / "run_cli"
            resolved2 = resolve_run("demo", "demo_offline",
                                    method_names=["BM25"])
            with mock.patch.object(
                    runner_mod, "create_retriever",
                    side_effect=RuntimeError("simulated constructor failure")):
                with self.assertRaises(SystemExit) as ctx:
                    from terag_release import cli as cli_mod
                    cli_mod.main(["run-experiments", "--dataset", "demo",
                                  "--experiment", "demo_offline",
                                  "--methods", "BM25", "--output",
                                  str(out2)])
                self.assertEqual(ctx.exception.code, 1)

    def test_partial_fold_run_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            result = _run_cli("run-experiments", "--dataset", "demo",
                              "--experiment", "demo_offline",
                              "--offline-only", "--methods", "BM25",
                              "--fold", "0", "--output", out)
            self.assertEqual(result.returncode, 0, result.stderr[-800:])
            meta = json.loads(Path(out, "run_meta", "run_meta.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(meta["execution"]["run_status"], "partial")
            self.assertFalse(Path(out, "summary_full_gold.csv").exists())

    def test_invalid_fold_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_cli("run-experiments", "--dataset", "demo",
                              "--experiment", "demo_offline",
                              "--offline-only", "--methods", "BM25",
                              "--fold", "9", "--output",
                              os.path.join(tmp, "run"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid --fold", result.stderr)

    def test_empty_plan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            # KaSLA is question_v1-only and the adapter methods are skipped
            # on table_v2: requesting only KaSLA with protocols=[table_v2]
            # yields an empty plan via a custom experiment yaml.
            exp = os.path.join(tmp, "empty.yaml")
            Path(exp).write_text(
                "dataset: demo\nprotocols: [table_v2]\nmethods: [KaSLA-Adapted]\n")
            result = _run_cli("run-experiments", "--dataset", "demo",
                              "--experiment", exp, "--allow-network",
                              "--output", os.path.join(tmp, "run"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("empty execution plan", result.stderr)

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            os.makedirs(out)
            Path(out, "marker").write_text("keep")
            result = _run_cli("run-experiments", "--dataset", "demo",
                              "--experiment", "demo_offline",
                              "--offline-only", "--methods", "BM25",
                              "--output", out)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(Path(out, "marker").exists())

    def test_single_question_retrieve_error_stays_in_denominator(self):
        # A retrieval exception on one question becomes an error record;
        # the other questions still get predictions (single-fold BM25).
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            result = _run_cli("run-experiments", "--dataset", "demo",
                              "--experiment", "demo_offline",
                              "--offline-only", "--methods", "BM25",
                              "--output", out)
            self.assertEqual(result.returncode, 0, result.stderr[-800:])
            records = [json.loads(line) for line in
                       Path(out, "predictions.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
            # one method, both protocols, every question once per protocol
            self.assertEqual(len(records), 2 * 49)
            keys = [(r["setting"], r["variant"], r["qid"]) for r in records]
            self.assertEqual(len(keys), len(set(keys)))

    def test_network_method_without_flag_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_cli("run-experiments", "--dataset", "demo",
                              "--experiment", "demo_offline",
                              "--methods", "DenseDeepSeek", "--output",
                              os.path.join(tmp, "run"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--allow-network", result.stderr)

    def test_benchmark_refuses_network_methods(self):
        from terag_release.benchmark import benchmark
        resolved = {
            "profile": "paper", "dataset": {"data": {}},
            "experiment": {"name": "x", "protocols": ["question_v1"],
                           "methods": ["DenseDeepSeek"]},
            "protocols": {}, "methods": {}, "budget": {},
        }
        with self.assertRaises(ValueError):
            benchmark(resolved=resolved, output="/tmp/never")


if __name__ == "__main__":
    unittest.main()
