"""Round-2 P1 regressions: per-question retrieval errors change run status.

The second-recheck probe: a mock retriever whose fit succeeds while every
evaluate record is status=error produced completed units, a completed run,
a summary, and exit code 0. All of that must now be a failure — with the
individual error records, real timings for executed queries, and
denominators preserved, and successes in partially-failing units kept.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FailingRetrieveRetriever:
    """Stub retriever: fit succeeds; N of the evaluate records are errors.

    Constructed via a patched runner.create_retriever, so the real
    supervision/alignment/status bookkeeping runs around it.
    """

    def __init__(self, fail_where):
        self.fail_where = fail_where  # callable(qid) -> bool

    def fit(self, train_data=None, **kwargs):
        self.train_qids = ([] if train_data is None
                           else list(train_data["qid"]))
        return self

    def evaluate_detailed(self, test_df, k=5):
        records = []
        for _, row in test_df.iterrows():
            rec = {
                "question": row["question"],
                "status": ("error" if self.fail_where(row["qid"]) else "ok"),
                "retrieved_tables": ([] if self.fail_where(row["qid"])
                                     else ["t_ok"]),
                "retrieved_column_keys": ([] if self.fail_where(row["qid"])
                                          else ["t_ok.c1"]),
                "retrieved_columns_detail": (
                    [] if self.fail_where(row["qid"]) else
                    [{"table": "t_ok", "table_score": 0.1,
                      "columns": [{"key": "t_ok.c1", "score": 0.1}]}]),
                "query_time": 0.001,
            }
            if rec["status"] == "error":
                rec["error_type"] = "SimulatedRetrievalError"
                rec["error_message"] = "injected retrieval failure"
            records.append(rec)
        return records


class TestRetrievalErrorStatus(unittest.TestCase):
    def _run(self, fail_where):
        from terag_release import runner as runner_mod
        from terag_release.config import resolve_run
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_run("demo", "demo_offline",
                                   method_names=["BM25"])
            out = Path(tmp) / "run"
            with mock.patch.object(
                    runner_mod, "create_retriever",
                    side_effect=lambda *a, **k:
                    _FailingRetrieveRetriever(fail_where)):
                meta = runner_mod.run_experiments(resolved=resolved,
                                                  run_dir=out)
            records = [json.loads(line) for line in
                       (out / "predictions.jsonl").read_text(
                           encoding="utf-8").splitlines() if line]
            return meta, records, out

    def test_all_retrievals_failing_fails_the_run(self):
        meta, records, out = self._run(lambda qid: True)
        self.assertEqual(meta["execution"]["run_status"], "failed")
        units = meta["execution"]["units"]
        self.assertTrue(all(u["status"] == "failed" for u in units))
        self.assertTrue(all(u["n_retrieval_errors"] == u["n_attempted"]
                            for u in units))
        self.assertEqual(
            meta["execution"]["aggregate"]["retrieval_errors_total"],
            2 * 49)
        self.assertTrue(all(r["status"] == "error" for r in records))
        # error records keep real, executed timing for the attempt
        self.assertTrue(all(r["query_time"] is not None for r in records))
        self.assertFalse((out / "summary_full_gold.csv").exists())

    def test_partial_retrieval_failures_degrade_the_run(self):
        meta, records, out = self._run(lambda qid: qid.endswith("007"))
        self.assertEqual(meta["execution"]["run_status"],
                         "completed_with_failures")
        units = meta["execution"]["units"]
        # folds without the failing question stay completed; folds with it
        # are degraded, never plain-completed
        self.assertTrue(all(u["status"] in ("completed",
                                            "completed_with_errors")
                            for u in units))
        self.assertTrue(any(u["status"] == "completed_with_errors"
                            for u in units))
        # successes preserved alongside per-question error records
        self.assertTrue(any(r["status"] == "ok" and
                            r["retrieved_tables"] == ["t_ok"]
                            for r in records))
        failed = [r for r in records if r["status"] == "error"]
        self.assertTrue(failed)
        self.assertTrue(all(r["retrieved_tables"] == [] for r in failed))
        # legal empty predictions are distinguished from errors
        unit = units[0]
        self.assertEqual(unit["n_attempted"],
                         unit["n_succeeded"] + unit["n_retrieval_errors"])
        self.assertFalse((out / "summary_full_gold.csv").exists())

    def test_legal_empty_predictions_are_successes(self):
        meta, records, _ = self._run(lambda qid: False)
        # every retrieval returns t_ok here; force-empty variant below
        self.assertEqual(meta["execution"]["run_status"], "completed")
        self.assertTrue(all(u["n_retrieval_errors"] == 0
                            for u in meta["execution"]["units"]))

    def test_legal_empty_output_is_not_an_error(self):
        from terag_release import runner as runner_mod
        from terag_release.config import resolve_run

        class _EmptyRetriever(_FailingRetrieveRetriever):
            def evaluate_detailed(self, test_df, k=5):
                records = super().evaluate_detailed(test_df, k)
                for rec in records:
                    rec["status"] = "ok"
                    rec["retrieved_tables"] = []
                    rec["retrieved_column_keys"] = []
                    rec["retrieved_columns_detail"] = []
                return records

        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_run("demo", "demo_offline",
                                   method_names=["BM25"])
            with mock.patch.object(
                    runner_mod, "create_retriever",
                    side_effect=lambda *a, **k: _EmptyRetriever(
                        lambda qid: False)):
                meta = runner_mod.run_experiments(resolved=resolved,
                                                  run_dir=Path(tmp) / "run")
        self.assertEqual(meta["execution"]["run_status"], "completed")
        self.assertTrue(all(u["n_legal_empty"] == u["n_attempted"]
                            for u in meta["execution"]["units"]))

    def test_cli_exit_code_propagates_retrieval_failures(self):
        import subprocess
        import sys
        import os
        # sabotage one question's retrieval through a tiny sitecustomize
        # is fragile; instead drive the CLI entry in-process with the
        # patched constructor
        from terag_release import cli as cli_mod
        from terag_release import runner as runner_mod
        from terag_release.config import resolve_run
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_run("demo", "demo_offline",
                                   method_names=["BM25"])
            with mock.patch.object(
                    runner_mod, "create_retriever",
                    side_effect=lambda *a, **k:
                    _FailingRetrieveRetriever(lambda qid: True)), \
                 mock.patch.object(cli_mod, "resolve_run",
                                   return_value=resolved):
                with self.assertRaises(SystemExit) as ctx:
                    cli_mod.main(["run-experiments", "--dataset", "demo",
                                  "--experiment", "demo_offline",
                                  "--methods", "BM25", "--output",
                                  str(Path(tmp) / "run_cli")])
                self.assertEqual(ctx.exception.code, 1)

    def test_benchmark_exit_status_propagates(self):
        from terag_release import benchmark as bench_mod
        from terag_release import runner as runner_mod
        from terag_release.config import resolve_run
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_run("demo", "question_only_offline",
                                   method_names=["BM25"])
            with mock.patch.object(
                    runner_mod, "create_retriever",
                    side_effect=lambda *a, **k:
                    _FailingRetrieveRetriever(lambda qid: True)):
                report = bench_mod.benchmark(resolved=resolved,
                                             output=str(Path(tmp) / "bench"))
            self.assertEqual(report["run_status"], "failed")
            self.assertIn("exit_nonzero_reason", report)


if __name__ == "__main__":
    unittest.main()
